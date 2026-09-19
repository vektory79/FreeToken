"""GGUF quant dispatch: geometry, kernel parity, and layer glue (glm5next Phase 4).

Dequant parity runs the borrowed ggml CUDA kernels against gguf-py's reference
``dequantize`` on random packed blocks. The CUDA paths round through half
(ggml-common.h:930 ``typedef half dfloat``), so bit parity against the fp32
reference is impossible; blocks are rejection-sampled so neither the reference nor
the kernel's fp16 chain overflows. MoE parity checks ``ggml_moe_a8_vec`` against a
torch reference on a small synthetic expert stack -- the kernel q8_1-quantizes X
internally, so the tolerance absorbs that x-side error; the weight side
dequantizes identically on both paths.
"""

from __future__ import annotations

import functools
import os
from types import SimpleNamespace

import gguf
import numpy as np
import pytest
import torch

from freetoken.models.gguf.dequant import (
    BLOCK_SHAPE,
    GGML_IQ3_XXS,
    GGML_IQ4_XS,
    GGML_Q3_K,
    GGML_Q4_K,
    GGML_Q6_K,
    GGML_Q8_0,
    dequantize,
    row_bytes,
)


@functools.cache
def _kernel_build_error() -> str | None:
    """None when the gguf CUDA extension builds; otherwise the error tail (last 400
    chars) so a real csrc regression shows up in the skip reason instead of
    masquerading as a missing host compiler. Runs once at collection time, cached."""
    try:
        from freetoken.kernel.gguf import _module

        _module()
        return None
    except Exception as e:
        return str(e)[-400:]


_BUILD_ERR = _kernel_build_error() if torch.cuda.is_available() else None

# fp16 scale-field byte offsets per format (gguf-py quants.py dequantize_blocks):
# Q4_K's d and dmin are adjacent half2 fields; the others carry a single half.
_FP16_SCALE_FIELDS = {
    GGML_Q8_0: ((0, 1),),
    GGML_Q6_K: ((208, 1),),
    GGML_Q3_K: ((108, 1),),
    GGML_Q4_K: ((0, 2),),
    GGML_IQ3_XXS: ((0, 1),),
    GGML_IQ4_XS: ((0, 1),),
}

# max |value| per element as a multiple of the block's fp16 d (from the gguf-py
# dequant formulas): int8 quants 127; k-quant sub-scales x nibble products;
# iq LUT/grid factors. Drives the per-element deviation bound above.
_DEQUANT_FACTOR = {
    GGML_Q8_0: 127.0,
    GGML_Q3_K: 32.0 * 4.0,
    GGML_Q4_K: 63.0 * 15.0 + 63.0,
    GGML_Q6_K: 127.0 * 31.0,
    GGML_IQ3_XXS: 8.0 * 118.0,
    GGML_IQ4_XS: 32.0 * 127.0,
}

cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or _BUILD_ERR is not None,
    reason=(
        f"gguf kernel JIT failed on this toolchain: ...{_BUILD_ERR}"
        if _BUILD_ERR
        else "needs CUDA + a buildable gguf kernel extension (clang++ host compiler)"
    ),
)

# every format the glm5next checkpoint uses: Q8_0/Q6_K dense, the rest expert banks.
_ALL_FORMATS = (GGML_Q8_0, GGML_Q6_K, GGML_Q3_K, GGML_Q4_K, GGML_IQ3_XXS, GGML_IQ4_XS)


def _qtype_enum(qtype: int) -> gguf.GGMLQuantizationType:
    return gguf.GGMLQuantizationType(qtype)


def _finite_fp16_chain_reference(rng, rows, rb, qtype):
    """Random packed blocks with CONSTRUCTED fp16 scale fields.

    The CUDA dequant chains round through half (ggml-common.h:930 ``typedef half
    dfloat``), so blind random scale bytes overflow fp16 for larger draws while the
    fp32 gguf-py reference stays finite (and gguf/quants.py emits cast warnings).
    The quant bits stay random; every half-typed scale is drawn from [0.25, 2],
    which bounds the full chain below 1e4 by construction (max multipliers: int8
    127, k-quant sub-scales 63/127, nibbles 15, iq grids ~118). A row spans
    rb/type_size blocks, so EVERY block's d (and dmin) field is bounded.
    """
    offsets = _FP16_SCALE_FIELDS[qtype]
    _, type_size = BLOCK_SHAPE[qtype]
    raw = rng.integers(0, 256, (rows, rb), dtype=np.uint8)
    per_row = raw.reshape(rows, -1, type_size)
    n_blocks = per_row.shape[1]
    for off, count in offsets:
        scales = (rng.random((rows, n_blocks, count)) * 1.75 + 0.25).astype(np.float16)
        per_row[:, :, off : off + 2 * count] = scales.view(np.uint8).reshape(
            rows, n_blocks, 2 * count
        )
    ref = gguf.dequantize(raw, _qtype_enum(qtype)).astype(np.float32)
    assert np.isfinite(ref).all()
    return raw, ref


@pytest.mark.parametrize("qtype", _ALL_FORMATS)
def test_block_shape_geometry_matches_gguf_py(qtype):
    block, type_size = BLOCK_SHAPE[qtype]
    ref_block, ref_size = gguf.GGML_QUANT_SIZES[_qtype_enum(qtype)]
    assert (block, type_size) == (int(ref_block), int(ref_size))
    assert row_bytes(512, qtype) == 512 // block * type_size


@cuda
@pytest.mark.parametrize("qtype", _ALL_FORMATS)
def test_dequant_matches_gguf_py_within_fp16_rounding(qtype):
    """Bit parity is impossible by design: the kernel rounds through half
    (ggml-common.h:930 ``typedef half dfloat``) AND the compiler contracts the
    __hmul/__hsub chains, so Q4_K's two-term form (d*sc*x - dmin*m) deviates from
    the fp32 reference by up to a few half-ulps of the TERM magnitudes - which at
    cancellation points exceeds any fixed rtol on the result (probe: max rel
    0.0048 at a near-cancellation element, max abs 0.28125 = ~1 half-ulp of the
    term at |y|~280). The assert therefore bounds |cuda - ref| per element by
    8 * 2^-11 * max_term_factor * |d| (+1e-3): tight enough that any decode bug
    (which shifts elements by O(factor * d)) fails loudly, loose enough for the
    documented fp16 rounding."""
    from freetoken.kernel.gguf import ggml_dequantize

    rng = np.random.default_rng(0)
    n = BLOCK_SHAPE[qtype][0] * 2  # two (super-)blocks per row
    rows, rb = 6, row_bytes(n, qtype)
    raw, ref = _finite_fp16_chain_reference(rng, rows, rb, qtype)
    out = ggml_dequantize(
        torch.from_numpy(raw.copy()).cuda(), qtype, rows, n, torch.float32
    )
    assert out.shape == (rows, n)

    _, type_size = BLOCK_SHAPE[qtype]
    d_off = _FP16_SCALE_FIELDS[qtype][0][0]
    d = (
        raw.reshape(rows, -1, type_size)[:, :, d_off : d_off + 2]
        .view(np.float16)
        .astype(np.float32)[:, :, 0]
    )  # (rows, blocks-per-row) the half d each chain starts from
    bound = 8 * 2**-11 * _DEQUANT_FACTOR[qtype] * d
    bound = np.repeat(bound, n // bound.shape[1], axis=1) + 1e-3
    dev = np.abs(out.cpu().numpy() - ref)
    assert (dev <= bound).all(), (
        f"max abs dev {dev.max():.6g} exceeds the fp16-chain bound "
        f"{bound.max():.6g}; {int((dev > bound).sum())} elements outside"
    )


@cuda
@pytest.mark.parametrize(
    "qtype", (GGML_Q8_0, GGML_Q6_K, GGML_Q3_K, GGML_Q4_K, GGML_IQ3_XXS, GGML_IQ4_XS)
)
def test_moe_vec_parity_vs_torch_reference(qtype):
    from freetoken.kernel.gguf import ggml_moe_a8_vec

    rng = np.random.default_rng(1)
    n = BLOCK_SHAPE[qtype][0] * 2
    out_rows, experts, tokens, top_k = 64, 3, 5, 2
    rb = row_bytes(n, qtype)
    packed, refs = [], []
    for _ in range(experts):
        raw, ref = _finite_fp16_chain_reference(rng, out_rows, rb, qtype)
        packed.append(torch.from_numpy(raw.copy()))
        refs.append(torch.from_numpy(ref).to(torch.float32))
    w = torch.stack(packed).cuda()  # [E, out_rows, rb]
    x = torch.randn(tokens, n, dtype=torch.float32, device="cuda")
    topk_ids = torch.stack(
        [torch.randperm(experts, device="cuda")[:top_k].to(torch.int32) for _ in range(tokens)]
    )
    out = ggml_moe_a8_vec(x, w, topk_ids, top_k, qtype, out_rows, tokens)
    assert out.shape == (tokens * top_k, out_rows)
    flat = out.reshape(tokens * top_k, out_rows).float().cpu()
    for t in range(tokens):
        for k in range(top_k):
            want = refs[int(topk_ids[t, k])] @ x[t].cpu()
            torch.testing.assert_close(
                flat[t * top_k + k], want, rtol=2e-2, atol=2e-2 * want.abs().max().item()
            )


def test_iq_formats_are_mmq_dispatchable():
    # the v2 grouped-MMQ work lifted the iq dense-batch gap: both formats must sit
    # in _MMQ (dense batches route to ggml_mul_mat_a8) and out of _IQ_ONLY.
    from freetoken.layers.gguf import _IQ_ONLY, _MMQ

    assert {GGML_IQ3_XXS, GGML_IQ4_XS} <= _MMQ
    assert not {GGML_IQ3_XXS, GGML_IQ4_XS} & _IQ_ONLY


@cuda
@pytest.mark.parametrize("qtype", (GGML_IQ3_XXS, GGML_IQ4_XS))
def test_dense_iq_above_mmvq_cutoff_uses_mmq(qtype):
    # dense batch 27 > _MMVQ_SAFE: dispatches to ggml_mul_mat_a8 (the former
    # NotImplementedError path) and must match the gguf-py dequant reference;
    # the tolerance absorbs the kernel's internal x q8_1 quantization only.
    from freetoken.layers.gguf import fused_mul_mat_gguf

    rng = np.random.default_rng(8)
    n = BLOCK_SHAPE[qtype][0]
    raw, ref = _finite_fp16_chain_reference(rng, 8, row_bytes(n, qtype), qtype)
    qw = torch.from_numpy(raw.copy()).cuda()
    x = torch.randn(27, n, dtype=torch.bfloat16, device="cuda")
    got = fused_mul_mat_gguf(x, qw, qtype).float()
    want = x.float() @ torch.from_numpy(ref).cuda().float().T
    torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2 * want.abs().max().item())


@cuda
@pytest.mark.parametrize("mtile", (4, 8, 16, 32))
@pytest.mark.parametrize("qtype", (GGML_IQ3_XXS, GGML_IQ4_XS))
def test_grouped_moe_parity_vs_reference(qtype, monkeypatch, mtile):
    # the grouped MMQ entry (moe_align trio -> ggml_moe_a8) for the two iq expert
    # formats at every v3a m-tile variant, checked pair-by-pair against the
    # gguf-py dequant reference. The weight side dequantizes identically on both
    # paths, so any tile/sign/scale bug shifts rows by O(factor * d) and fails
    # this bound loudly.
    monkeypatch.setenv("FREETOKEN_GGUF_MOE_MTILE", str(mtile))
    from freetoken.kernel.gguf import ggml_moe_a8, ggml_moe_get_block_size
    from freetoken.moe.fused import moe_align_block_size

    rng = np.random.default_rng(3)
    n = BLOCK_SHAPE[qtype][0] * 2
    out_rows, experts, tokens, top_k = 32, 5, 12, 4
    rb = row_bytes(n, qtype)
    packed, refs = [], []
    for _ in range(experts):
        raw, ref = _finite_fp16_chain_reference(rng, out_rows, rb, qtype)
        packed.append(torch.from_numpy(raw.copy()))
        refs.append(torch.from_numpy(ref).to(torch.float32))
    w = torch.stack(packed).cuda()  # [E, out_rows, rb]
    x = torch.randn(tokens, n, dtype=torch.float32, device="cuda")
    # deterministic routing: expert 0 carries 34 pairs, so every tile size serves
    # >nwarps live ds rows per bin (34 > 32 keeps ds rows 0..31 live at m=32);
    # random routing could leave every bin at <= nwarps rows and skip the ported
    # ds-fill coverage entirely
    topk_ids = torch.zeros(tokens, top_k, dtype=torch.int32, device="cuda")
    topk_ids.view(-1)[:34] = 0
    topk_ids.view(-1)[34:] = torch.arange(14, device="cuda") % 4 + 1
    sorted_ids, expert_ids, npp = moe_align_block_size(
        topk_ids, ggml_moe_get_block_size(qtype), experts
    )
    out = ggml_moe_a8(x, w, sorted_ids, expert_ids, npp, qtype, out_rows, top_k, tokens)
    assert out.shape == (tokens * top_k, out_rows)
    flat = out.float().cpu()
    for t in range(tokens):
        for k in range(top_k):
            want = refs[int(topk_ids[t, k])] @ x[t].cpu()
            torch.testing.assert_close(
                flat[t * top_k + k], want, rtol=2e-2, atol=2e-2 * want.abs().max().item()
            )


@cuda
def test_grouped_moe_expert_bound_288():
    # pins the moe.cuh expert-bound fix: the legacy `exp_idx > 255` guard silently
    # dropped experts 256..287 at E=288 and Y (torch.empty) kept garbage rows.
    from freetoken.kernel.gguf import ggml_moe_a8, ggml_moe_get_block_size
    from freetoken.moe.fused import moe_align_block_size

    qtype = GGML_IQ4_XS
    rng = np.random.default_rng(4)
    n = BLOCK_SHAPE[qtype][0]
    out_rows, experts, tokens, top_k = 16, 288, 4, 8
    rb = row_bytes(n, qtype)
    packed, refs = [], []
    for _ in range(experts):
        raw, ref = _finite_fp16_chain_reference(rng, out_rows, rb, qtype)
        packed.append(torch.from_numpy(raw.copy()))
        refs.append(torch.from_numpy(ref).to(torch.float32))
    w = torch.stack(packed).cuda()
    x = torch.randn(tokens, n, dtype=torch.float32, device="cuda")
    topk_ids = torch.stack(
        [torch.randperm(experts, device="cuda")[:top_k].to(torch.int32) for _ in range(tokens)]
    )
    # force half of token 0's routes onto experts the old guard dropped
    topk_ids[0] = torch.tensor([287, 256, 285, 260, 286, 263, 0, 1], device="cuda", dtype=torch.int32)
    assert int(topk_ids.max()) > 255
    sorted_ids, expert_ids, npp = moe_align_block_size(
        topk_ids, ggml_moe_get_block_size(qtype), experts
    )
    out = ggml_moe_a8(x, w, sorted_ids, expert_ids, npp, qtype, out_rows, top_k, tokens)
    flat = out.float().cpu()
    for t in range(tokens):
        for k in range(top_k):
            want = refs[int(topk_ids[t, k])] @ x[t].cpu()
            torch.testing.assert_close(
                flat[t * top_k + k], want, rtol=2e-2, atol=2e-2 * want.abs().max().item()
            )


@cuda
@pytest.mark.parametrize("mtile", (4, 8, 16, 32))
@pytest.mark.parametrize("qtype", (GGML_Q8_0, GGML_Q6_K, GGML_IQ3_XXS, GGML_IQ4_XS))
def test_grouped_moe_matches_moe_vec(qtype, monkeypatch, mtile):
    # all four MoE entries exist for both paths at every v3a m-tile: the grouped
    # MMQ path and the moe_vec GEMV must agree on identical inputs (same trio,
    # same x q8_1 quantization; only the tile reduction order differs). The iq
    # formats (types 18/23) are the ones the grouped prefill actually serves.
    # Deterministic routing keeps one bin at 12 live pairs, so ds rows > nwarps
    # (the v3a ds-fill port) feed real outputs instead of staying pad slots.
    monkeypatch.setenv("FREETOKEN_GGUF_MOE_MTILE", str(mtile))
    from freetoken.kernel.gguf import ggml_moe_a8, ggml_moe_a8_vec, ggml_moe_get_block_size
    from freetoken.moe.fused import moe_align_block_size

    torch.manual_seed(7)  # x draw must not flip a borderline tolerance run to run
    rng = np.random.default_rng(5)
    n = BLOCK_SHAPE[qtype][0] * 2
    out_rows, experts, tokens, top_k = 32, 5, 12, 4
    rb = row_bytes(n, qtype)
    packed = []
    for _ in range(experts):
        raw, _ = _finite_fp16_chain_reference(rng, out_rows, rb, qtype)
        packed.append(torch.from_numpy(raw.copy()))
    w = torch.stack(packed).cuda()
    x = torch.randn(tokens, n, dtype=torch.float32, device="cuda")
    topk_ids = torch.zeros(tokens, top_k, dtype=torch.int32, device="cuda")
    topk_ids.view(-1)[:12] = 0
    topk_ids.view(-1)[12:] = torch.arange(36, device="cuda") % 4 + 1
    vec = ggml_moe_a8_vec(x, w, topk_ids, top_k, qtype, out_rows, tokens)
    sorted_ids, expert_ids, npp = moe_align_block_size(
        topk_ids, ggml_moe_get_block_size(qtype), experts
    )
    grouped = ggml_moe_a8(x, w, sorted_ids, expert_ids, npp, qtype, out_rows, top_k, tokens)
    assert grouped.shape == vec.shape == (tokens * top_k, out_rows)
    if qtype in (GGML_Q8_0, GGML_Q6_K):
        torch.testing.assert_close(grouped.float(), vec.float(), rtol=1e-3, atol=1e-3)
    else:
        # iq dots reorder the k reduction and round the fp16 scale chain
        # differently from the MMVQ dot: seed-7 max rel err 4.2e-4 (IQ3_XXS),
        # 20-seed worst |a-b| ~1.2e-3, so this keeps 3x+ flake headroom while
        # a ds-fill bug (whole pairs off by O(1)) still trips loudly.
        torch.testing.assert_close(grouped.float(), vec.float(), rtol=2e-3, atol=4e-3)


@cuda
@pytest.mark.parametrize("bs", (4, 8, 16, 32))  # every env-selectable m-tile
def test_moe_align_trio_contract(bs):
    # the trio the grouped entry consumes: int32 buffers, sentinel fill = numel
    # (NOT -1), numel = pairs + (E+1)*(bs-1) whenever pairs >= E+1, and expert_ids
    # inside the padded region are real expert ids (the pad bin E is what the
    # kernel's expert bound must keep dropping). Swept over the m-tile set: the
    # v3a knob feeds block_size straight from get_block_size.
    from freetoken.moe.fused import moe_align_block_size

    tokens, top_k, experts = 40, 8, 288
    topk_ids = torch.randint(0, experts, (tokens, top_k), dtype=torch.int32, device="cuda")
    sorted_ids, expert_ids, npp = moe_align_block_size(topk_ids, bs, experts)

    pairs = topk_ids.numel()
    assert sorted_ids.dtype == expert_ids.dtype == npp.dtype == torch.int32
    assert sorted_ids.numel() == pairs + (experts + 1) * (bs - 1)
    assert expert_ids.numel() == (sorted_ids.numel() + bs - 1) // bs
    assert npp.shape == (1,)

    sentinel = sorted_ids == pairs  # the producer's sentinel fill is topk_ids.numel()
    assert sentinel.any(), "pad slots must carry the numel sentinel"
    assert not (sorted_ids[~sentinel] >= pairs).any(), "real entries are pair indices"
    total_padded = int(npp.item())
    assert total_padded % bs == 0 and total_padded <= pairs + experts * bs
    live = expert_ids[: total_padded // bs]
    assert (live >= 0).all() and (live < experts).all(), "pad bin ids must not reach the kernel"


@cuda
@pytest.mark.parametrize("bs", (4, 8, 16, 32))
@pytest.mark.parametrize("tokens", (40, 137))  # 320 and 1096 pairs: small and large producer paths
def test_moe_align_tail_contract(tokens, bs):
    # TP2 tail pin, driven through the TRITON producer directly (the fused wrapper
    # would pick the sgl_kernel op when it is installed): the grouped CUDA kernel
    # launches bins past num_tokens_post_pad (its grid covers the padded buffer),
    # so the producer must leave NO uninitialized slots. sorted_token_ids[npp:]
    # carries the numel sentinel and expert_ids[npp/bs:] carries the sentinel
    # expert id (num_experts), which moe_q's expert bound drops even before the
    # >= boundary check rejects the bin. Swept over the m-tile set.
    from freetoken.kernel.triton.moe_align import moe_align_block_size as triton_align

    top_k, experts = 8, 288
    topk_ids = torch.randint(0, experts, (tokens, top_k), dtype=torch.int32, device="cuda")
    sorted_ids, expert_ids, npp = triton_align(topk_ids, bs, experts)

    pairs = topk_ids.numel()
    total = int(npp.item())
    assert total % bs == 0
    tail = sorted_ids[total:]
    assert tail.numel() > 0
    assert (tail == pairs).all(), "sorted tail must carry the numel sentinel"
    tb = total // bs
    live = expert_ids[:tb]
    assert (live >= 0).all() and (live < experts).all(), "live blocks carry real expert ids"
    pad = expert_ids[tb:]
    assert pad.numel() > 0
    assert (pad == experts).all(), "expert_ids tail must carry the sentinel expert id"
    # belt (moe_q boundary): bins starting at/after total are rejected; the first
    # such bin is tb and the producer's sentinel values make it a safe read
    assert expert_ids.numel() > tb and tb * bs >= total


@cuda
@pytest.mark.parametrize("mtile", (4, 8, 16, 32))
def test_grouped_moe_ds_fill_analytic_uniform(mtile, monkeypatch):
    # v3a ds-fill port regression: exact analytic outputs at m-tiles > nwarps.
    # One 32-pair bin (tokens=32, top_k=1, all on expert 0) keeps every tile row
    # live with zero padding, so ds rows up to 31 feed real outputs. Q8_0 banks
    # carry d=0.25 (fp16-exact) with uniform int8 quants per row; activations are
    # integers whose 32-block amax is exactly 127, so the kernel's q8_1 has d=1
    # and every product is exact - a wrong ds row shifts its pair by O(scale)
    # and cannot hide in a tolerance (parity alone lets same-wrong-bytes cancel).
    monkeypatch.setenv("FREETOKEN_GGUF_MOE_MTILE", str(mtile))
    from freetoken.kernel.gguf import ggml_moe_a8, ggml_moe_get_block_size
    from freetoken.moe.fused import moe_align_block_size

    k, out_rows, experts, tokens, top_k = 32, 4, 3, 32, 1
    row_qs = torch.tensor([1, 2, -1, 3], dtype=torch.int8)
    w = torch.zeros(experts, out_rows, 34, dtype=torch.uint8)
    w[:, :, 0:2] = torch.tensor([0.25], dtype=torch.float16).view(torch.uint8)  # block_q8_0 d
    w[:, :, 2:] = row_qs.view(torch.uint8)[None, :, None]
    # integer activations, |x| <= 127 with amax exactly 127 per 32-block:
    # q8_1 d = amax/127 = 1.0 and round(x/d) = x, so quantization is lossless
    vals = (np.arange(1, k)[None, :] + 3 * np.arange(tokens)[:, None]) % 9 - 4
    x = torch.zeros(tokens, k, dtype=torch.float32)
    x[:, 0] = torch.where(torch.arange(tokens) % 2 == 0, 127.0, -127.0)
    x[:, 1:] = torch.from_numpy(vals.astype(np.float32))
    row_sums = x.double().sum(-1)
    want = (0.25 * row_qs.double()[None, :] * row_sums[:, None]).float()
    w, x = w.cuda(), x.cuda()
    topk_ids = torch.zeros(tokens, top_k, dtype=torch.int32, device="cuda")
    sorted_ids, expert_ids, npp = moe_align_block_size(
        topk_ids, ggml_moe_get_block_size(GGML_Q8_0), experts
    )
    out = ggml_moe_a8(x, w, sorted_ids, expert_ids, npp, GGML_Q8_0, out_rows, top_k, tokens)
    assert out.shape == (tokens * top_k, out_rows)
    torch.testing.assert_close(out.float().cpu(), want, rtol=0.0, atol=1e-3)


@cuda
@pytest.mark.parametrize("mtile", (16, 32))
def test_grouped_moe_ds_fill_analytic_tail_experts(mtile, monkeypatch):
    # tail-expert shape of the analytic ds fixture: n_e = 2/1 gives two bins
    # carrying 30/31 pad columns at tile 32 (2 and 1 live pairs) - the pad-slot
    # path must not leak into the live pairs, asserted against exact expected
    # outputs instead of tolerances. Per-expert quants keep the expert indexing
    # observable too.
    monkeypatch.setenv("FREETOKEN_GGUF_MOE_MTILE", str(mtile))
    from freetoken.kernel.gguf import ggml_moe_a8, ggml_moe_get_block_size
    from freetoken.moe.fused import moe_align_block_size

    k, out_rows, experts, tokens, top_k = 32, 4, 2, 3, 1
    expert_qs = torch.tensor([[1, 2, -1, 3], [-2, 1, 3, 2]], dtype=torch.int8)
    w = torch.zeros(experts, out_rows, 34, dtype=torch.uint8)
    w[:, :, 0:2] = torch.tensor([0.25], dtype=torch.float16).view(torch.uint8)  # block_q8_0 d
    w[:, :, 2:] = expert_qs.view(torch.uint8)[:, :, None]
    # same lossless-quantization construction as the analytic-uniform fixture
    vals = (np.arange(1, k)[None, :] + 5 * np.arange(tokens)[:, None]) % 9 - 4
    x = torch.zeros(tokens, k, dtype=torch.float32)
    x[:, 0] = torch.where(torch.arange(tokens) % 2 == 0, 127.0, -127.0)
    x[:, 1:] = torch.from_numpy(vals.astype(np.float32))
    row_sums = x.double().sum(-1)
    want = (0.25 * expert_qs.double()[torch.tensor([0, 0, 1])] * row_sums[:, None]).float()
    w, x = w.cuda(), x.cuda()
    topk_ids = torch.tensor([[0], [0], [1]], dtype=torch.int32, device="cuda")
    sorted_ids, expert_ids, npp = moe_align_block_size(
        topk_ids, ggml_moe_get_block_size(GGML_Q8_0), experts
    )
    assert int(npp.item()) == 2 * mtile  # two bins: 2 and 1 live pairs, the rest pad
    out = ggml_moe_a8(x, w, sorted_ids, expert_ids, npp, GGML_Q8_0, out_rows, top_k, tokens)
    assert out.shape == (tokens * top_k, out_rows)
    torch.testing.assert_close(out.float().cpu(), want, rtol=0.0, atol=1e-3)


@cuda
def test_moe_mtile_env_knob(monkeypatch):
    # FREETOKEN_GGUF_MOE_MTILE: default 4, each valid value, invalid values raise
    # out of the extension, and the python-visible get_block_size (the trio's
    # block_size source) matches the selected tile - the single-source-of-truth
    # regression against a kernel mmq_x / moe_align block_size divergence.
    from freetoken.kernel.gguf import ggml_moe_get_block_size

    monkeypatch.delenv("FREETOKEN_GGUF_MOE_MTILE", raising=False)
    assert all(
        ggml_moe_get_block_size(t) == 4 for t in (GGML_Q8_0, GGML_Q6_K, GGML_IQ3_XXS, GGML_IQ4_XS)
    )
    for tile in (4, 8, 16, 32):
        monkeypatch.setenv("FREETOKEN_GGUF_MOE_MTILE", str(tile))
        assert ggml_moe_get_block_size(GGML_Q8_0) == tile
        assert ggml_moe_get_block_size(GGML_Q6_K) == tile
        assert ggml_moe_get_block_size(GGML_IQ3_XXS) == tile
        assert ggml_moe_get_block_size(GGML_IQ4_XS) == tile
    # the knob is global across the served types but must not touch the
    # single-tile types (Q4_0/Q4_1/Q5_0/Q5_1/Q2_K/Q3_K/Q4_K/Q5_K keep their MOE_X)
    monkeypatch.setenv("FREETOKEN_GGUF_MOE_MTILE", "32")
    assert ggml_moe_get_block_size(2) == 4
    assert all(ggml_moe_get_block_size(t) == 4 for t in (3, 6, 7, 10, 11, 12, 13))
    # strcmp whitelist: no trim or numeric parsing, so stragglers fail too
    for bad in ("12", "0", "-4", "abc", " 8", "04", "+8"):
        monkeypatch.setenv("FREETOKEN_GGUF_MOE_MTILE", bad)
        with pytest.raises(RuntimeError, match="FREETOKEN_GGUF_MOE_MTILE"):
            ggml_moe_get_block_size(GGML_Q8_0)


@cuda
def test_dense_gguf_modules_forward_packed():
    from freetoken.layers.gguf import GGUFEmbedding, GGUFLinear

    rng = np.random.default_rng(2)
    h, vocab, out_dim = 256, 64, 256  # out_dim: Q6_K packs 256-wide input blocks
    lin = GGUFLinear(h, out_dim, GGML_Q8_0)
    emb = GGUFEmbedding(vocab, h, GGML_Q8_0)
    head = GGUFLinear(out_dim, vocab, GGML_Q6_K)  # the untied Q6_K lm_head analog
    for mod, rows, qt in (
        (lin, out_dim, GGML_Q8_0),
        (emb, vocab, GGML_Q8_0),
        (head, vocab, GGML_Q6_K),
    ):
        raw, _ = _finite_fp16_chain_reference(rng, rows, row_bytes(h, qt), qt)
        mod.qweight = torch.from_numpy(raw.copy()).cuda()

    ids = torch.randint(0, vocab, (3,), device="cuda")
    x = emb.forward(ids)
    assert x.dtype == torch.bfloat16 and x.shape == (3, h)
    y = lin.forward(x)
    logits = head.forward(y)

    # packed stays packed: the modules never materialize a dense copy of their own
    assert lin.qweight.dtype == torch.uint8 and emb.qweight.dtype == torch.uint8
    assert head.qweight.dtype == torch.uint8

    # numerics vs the python reference dequant (the kernel dequantizes on GPU)
    def ref(qw, qt, rows):
        return dequantize(qw.cpu().reshape(-1), qt, torch.float32).reshape(rows, h)

    # the MMVQ path q8_1-quantizes X and emits bf16: the tolerance scales with the
    # row magnitude (same convention as the moe parity test above)
    want_x = ref(emb.qweight, GGML_Q8_0, vocab)[ids.cpu()].to(torch.bfloat16)
    want_x = want_x.to(x.device)
    torch.testing.assert_close(x, want_x, rtol=2e-2, atol=2e-2 * want_x.abs().max().item())
    want_y = (want_x.float() @ ref(lin.qweight, GGML_Q8_0, out_dim).T.to(want_x.device)).to(
        torch.bfloat16
    )
    want_y = want_y.to(y.device)
    torch.testing.assert_close(y, want_y, rtol=2e-2, atol=2e-2 * want_y.abs().max().item())
    want_logits = (
        want_y.float() @ ref(head.qweight, GGML_Q6_K, vocab).T.to(want_y.device)
    ).to(torch.bfloat16)
    torch.testing.assert_close(
        logits, want_logits, rtol=2e-2, atol=2e-2 * want_logits.abs().max().item()
    )


@cuda
@pytest.mark.parametrize("batch", (4, 27))  # MMVQ GEMV path and MMQ path
def test_dense_gguf_noncontig_activations(batch):
    # KDA f_b_proj/g_b_proj feed torch.split views of the in_proj output (stride
    # (24896, 1)) into the GGUF GEMM. The CUDA kernels assume row-major
    # activations and silently returned garbage for non-contiguous x - the root
    # cause of the Phase 6 off-topic generation defect. The dispatch normalizes
    # x; this pins the guard on both kernel paths.
    from freetoken.layers.gguf import fused_mul_mat_gguf

    rng = np.random.default_rng(7)
    k, out_dim = 128, 96  # K = 4 Q8_0 blocks: the real f_b/g_b geometry
    raw, ref = _finite_fp16_chain_reference(rng, out_dim, row_bytes(k, GGML_Q8_0), GGML_Q8_0)
    qw = torch.from_numpy(raw.copy()).cuda()
    weight = torch.from_numpy(ref).cuda().float()

    wide = torch.randn((batch, 24896), device="cuda", dtype=torch.bfloat16)
    x = wide[:, :k]
    assert not x.is_contiguous()
    got = fused_mul_mat_gguf(x, qw, GGML_Q8_0).float()
    want = x.float() @ weight.t()
    torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2 * want.abs().max().item())

    got_c = fused_mul_mat_gguf(x.contiguous(), qw, GGML_Q8_0).float()
    torch.testing.assert_close(got, got_c, rtol=2e-2, atol=2e-2 * want.abs().max().item())


# ---- kda_in_proj N-split knob (FREETOKEN_GGUF_KDA_NSPLIT, wave 1 of
# .tasks/dense-q80-gemm): the fused KDA in_proj MMQ GEMM (production N=24896 =
# 108.35 MB > 96.0 MiB L2) may launch as two sub-N GEMMs whose weights fit L2.


def _spy_kda_a8(monkeypatch):
    """Record the out-feature count of every MMQ/vec launch. fused_mul_mat_gguf
    resolves the wrappers from freetoken.kernel.gguf lazily per call, so patching
    the module attrs intercepts every dispatch without touching csrc."""
    import freetoken.kernel.gguf as kg

    mmq, vec = [], []
    real_mmq, real_vec = kg.ggml_mul_mat_a8, kg.ggml_mul_mat_vec_a8

    def spy_mmq(w, x, qt, row):
        mmq.append(row)
        return real_mmq(w, x, qt, row)

    def spy_vec(w, x, qt, row):
        vec.append(row)
        return real_vec(w, x, qt, row)

    monkeypatch.setattr(kg, "ggml_mul_mat_a8", spy_mmq)
    monkeypatch.setattr(kg, "ggml_mul_mat_vec_a8", spy_vec)
    return mmq, vec


def _kda_nsplit_layer(rng, k, n):
    from freetoken.layers.gguf import GGUFLinear

    lin = GGUFLinear(k, n, GGML_Q8_0)
    raw, _ = _finite_fp16_chain_reference(rng, n, row_bytes(k, GGML_Q8_0), GGML_Q8_0)
    lin.qweight = torch.from_numpy(raw.copy()).cuda()
    return lin


@cuda
def test_kda_nsplit_env_knob(monkeypatch):
    # FREETOKEN_GGUF_KDA_NSPLIT mirrors the v3a m-tile knob's shape: default 1,
    # strict whitelist (no trim / numeric parsing, stragglers raise), read PER
    # CALL so flipping the env between calls flips the launch structure.
    from freetoken.layers.gguf import kda_in_proj_forward

    monkeypatch.delenv("FREETOKEN_GGUF_KDA_NSPLIT", raising=False)
    lin = _kda_nsplit_layer(np.random.default_rng(3), 512, 96)  # 96 = 3 x 32
    x = torch.randn((64, 512), device="cuda", dtype=torch.bfloat16) * 0.25

    mmq, vec = _spy_kda_a8(monkeypatch)
    kda_in_proj_forward(lin, x)
    assert mmq == [96] and vec == []
    # "" is not a straggler: the empty value must behave exactly like unset (default 1)
    monkeypatch.setenv("FREETOKEN_GGUF_KDA_NSPLIT", "")
    kda_in_proj_forward(lin, x)
    assert mmq == [96, 96] and vec == []
    monkeypatch.setenv("FREETOKEN_GGUF_KDA_NSPLIT", "2")
    kda_in_proj_forward(lin, x)
    assert mmq == [96, 96, 32, 64] and vec == []
    # per-call read: the same process flips back without any module state
    monkeypatch.setenv("FREETOKEN_GGUF_KDA_NSPLIT", "1")
    kda_in_proj_forward(lin, x)
    assert mmq == [96, 96, 32, 64, 96]
    # invalid values fail fast before any launch, for any shape
    for bad in ("12", "0", "-2", "abc", " 2", "02", "+2"):
        monkeypatch.setenv("FREETOKEN_GGUF_KDA_NSPLIT", bad)
        with pytest.raises(ValueError, match="FREETOKEN_GGUF_KDA_NSPLIT"):
            kda_in_proj_forward(lin, x)


@cuda
def test_kda_nsplit_bitwise_parity(monkeypatch):
    # production geometry: N=24896 = 2 x 12448 (12448 = 389 x 32), K=4096, MMQ
    # batch. Disjoint 32-row tiles + k-only per-element reduction + deterministic
    # x-quant: the split output must be BITWISE equal (torch.equal, no tolerance)
    # to the single launch, and knob=1 must equal the plain GGUFLinear forward.
    from freetoken.layers.gguf import kda_in_proj_forward

    lin = _kda_nsplit_layer(np.random.default_rng(11), 4096, 24896)
    x = torch.randn((256, 4096), device="cuda", dtype=torch.bfloat16) * 0.25

    mmq, _ = _spy_kda_a8(monkeypatch)
    monkeypatch.setenv("FREETOKEN_GGUF_KDA_NSPLIT", "1")
    single = kda_in_proj_forward(lin, x)
    assert mmq == [24896]
    assert torch.equal(single, lin.forward(x))

    mmq.clear()
    monkeypatch.setenv("FREETOKEN_GGUF_KDA_NSPLIT", "2")
    split = kda_in_proj_forward(lin, x)
    assert mmq == [12448, 12448]
    assert split.shape == single.shape and split.dtype == single.dtype
    assert torch.equal(split, single)


@cuda
def test_kda_nsplit_parity_m_tail_batches(monkeypatch):
    # M=7/13: batch > 6 (MMQ dispatch) but not a multiple of mmq_x = 4, so both
    # launches ride the m-tile tail path (clamped y-loads, guarded dst writes).
    # Production N=24896; the split output must stay BITWISE equal to the single
    # launch even with the batch tail present.
    from freetoken.layers.gguf import kda_in_proj_forward

    lin = _kda_nsplit_layer(np.random.default_rng(17), 4096, 24896)
    mmq, _ = _spy_kda_a8(monkeypatch)
    for batch in (7, 13):
        x = torch.randn((batch, 4096), device="cuda", dtype=torch.bfloat16) * 0.25
        monkeypatch.setenv("FREETOKEN_GGUF_KDA_NSPLIT", "1")
        mmq.clear()
        single = kda_in_proj_forward(lin, x)
        assert mmq == [24896]
        monkeypatch.setenv("FREETOKEN_GGUF_KDA_NSPLIT", "2")
        mmq.clear()
        split = kda_in_proj_forward(lin, x)
        assert mmq == [12448, 12448]
        assert split.shape == single.shape and split.dtype == single.dtype
        assert torch.equal(split, single)


@cuda
def test_kda_nsplit_parity_noncontig_x(monkeypatch):
    # a strided x slice (row stride 2 x K) rides the split path too: the per-launch
    # .contiguous() inside fused_mul_mat_gguf is deterministic, so both sub-launches
    # quantize identical bytes and the output stays bitwise equal to the single launch.
    from freetoken.layers.gguf import kda_in_proj_forward

    lin = _kda_nsplit_layer(np.random.default_rng(19), 512, 24896)  # production N
    base = torch.randn((26, 512), device="cuda", dtype=torch.bfloat16) * 0.25
    x = base[::2]  # 13 x 512 view with row stride 2 x 512
    assert not x.is_contiguous()

    mmq, _ = _spy_kda_a8(monkeypatch)
    monkeypatch.setenv("FREETOKEN_GGUF_KDA_NSPLIT", "1")
    single = kda_in_proj_forward(lin, x)
    assert mmq == [24896]
    mmq.clear()
    monkeypatch.setenv("FREETOKEN_GGUF_KDA_NSPLIT", "2")
    split = kda_in_proj_forward(lin, x)
    assert mmq == [12448, 12448]
    assert torch.equal(split, single)


@cuda
def test_kda_nsplit_boundary_math(monkeypatch):
    # the split boundary is pinned to the kernel's 32-row N tiles: chunks are
    # 32-aligned and >= 32 rows; N not a multiple of 32 (or N < 64, where the
    # second chunk would be < 32) falls back to the single launch, byte identical.
    from types import SimpleNamespace

    from freetoken.layers.gguf import kda_in_proj_forward

    rng = np.random.default_rng(5)
    x = torch.randn((64, 512), device="cuda", dtype=torch.bfloat16) * 0.25
    mmq, _ = _spy_kda_a8(monkeypatch)
    monkeypatch.setenv("FREETOKEN_GGUF_KDA_NSPLIT", "2")

    mark = len(mmq)
    for n in (48, 32):
        lin = _kda_nsplit_layer(rng, 512, n)
        got = kda_in_proj_forward(lin, x)
        assert mmq[mark:] == [n]  # no split: the ragged shape falls back
        assert torch.equal(got, lin.forward(x))
        mark = len(mmq)
    for n, want in ((64, [32, 32]), (96, [32, 64])):
        lin = _kda_nsplit_layer(rng, 512, n)
        single = lin.forward(x)
        assert mmq[mark:] == [n]
        mark = len(mmq)
        got = kda_in_proj_forward(lin, x)
        assert mmq[mark:] == want
        assert torch.equal(got, single)  # 32-aligned split stays bitwise equal
        mark = len(mmq)
    # non-GGUF layers (plain bf16 / quant-method modules) pass through untouched
    plain = SimpleNamespace(forward=lambda t: t + 1.0)
    assert torch.equal(kda_in_proj_forward(plain, x), x + 1.0)


@cuda
def test_kda_nsplit_decode_batch_stays_mmvq(monkeypatch):
    # batch <= 6 keeps the MMVQ vec kernels byte-identical even with the knob at
    # 2: the split's scope guard covers only the dense MMQ path (batch > 6).
    # (bs > 6 decode rides captured CUDA graphs - engine/graph.py
    # GraphRunner._capture_graphs bakes the MMQ structure at boot - so the A/B
    # wave re-captures and re-measures decode with the knob set.)
    from freetoken.layers.gguf import fused_mul_mat_gguf, kda_in_proj_forward

    lin = _kda_nsplit_layer(np.random.default_rng(13), 512, 24896)  # production N
    x = torch.randn((4, 512), device="cuda", dtype=torch.bfloat16) * 0.25

    mmq, vec = _spy_kda_a8(monkeypatch)
    monkeypatch.setenv("FREETOKEN_GGUF_KDA_NSPLIT", "2")
    got = kda_in_proj_forward(lin, x)
    assert mmq == [] and vec == [24896]
    assert torch.equal(got, fused_mul_mat_gguf(x, lin.qweight, GGML_Q8_0))


# ---- the JIT module load must not leak CC/CXX into the process env ----

# A nonexistent absolute path: resolution must fall through to the raw override.
_FAKE_HOST_CXX = "/opt/fake-toolchain/bin/clang++"


def _capture_load(monkeypatch):
    """Swap torch's cpp_extension.load for a recorder: no JIT build, no CUDA.

    Captures the CC/CXX the build subprocesses would have seen plus the cuda
    flags, so the env-scoping around the load call is testable on any machine.
    """
    seen = {}

    def fake_load(**kwargs):
        seen["env"] = {k: os.environ.get(k) for k in ("CC", "CXX")}
        seen["flags"] = list(kwargs["extra_cuda_cflags"])
        return object()

    monkeypatch.setattr("torch.utils.cpp_extension.load", fake_load)
    return seen


def test_module_load_scopes_cc_cxx_to_the_build(monkeypatch):
    """The clang host override applies to the build's subprocesses only.

    Regression: _module() used to write CC/CXX into os.environ and never
    restore them, so a later flashinfer JIT build in the same pytest process
    regenerated build.ninja with the clang host and failed to compile
    (alignas(64) below CUtensorMap's default under CUDA 13.3).
    """
    from freetoken.kernel import gguf as gguf_kernel

    seen = _capture_load(monkeypatch)
    monkeypatch.setenv("FREETOKEN_GGUF_HOST_CXX", _FAKE_HOST_CXX)
    # which() -> None keeps host-compiler resolution deterministic on any PATH
    monkeypatch.setattr(gguf_kernel, "shutil", SimpleNamespace(which=lambda _name: None))

    before = {k: os.environ.get(k) for k in ("CC", "CXX")}
    gguf_kernel._module.__wrapped__()  # bypass functools.cache

    # the build ran with the override as the host compiler, on both passes
    assert seen["env"]["CXX"] == _FAKE_HOST_CXX
    assert seen["env"]["CC"] == "clang"
    assert seen["flags"][-2:] == ["-ccbin", _FAKE_HOST_CXX]
    # ... and the process env is exactly as it was before the call
    assert {k: os.environ.get(k) for k in ("CC", "CXX")} == before


def test_module_load_without_a_host_compiler_sets_no_cc_cxx(monkeypatch):
    """No FREETOKEN_GGUF_HOST_CXX and no compiler on PATH: nothing is injected."""
    from freetoken.kernel import gguf as gguf_kernel

    seen = _capture_load(monkeypatch)
    monkeypatch.delenv("FREETOKEN_GGUF_HOST_CXX", raising=False)
    monkeypatch.setattr(gguf_kernel, "shutil", SimpleNamespace(which=lambda _name: None))

    before = {k: os.environ.get(k) for k in ("CC", "CXX")}
    gguf_kernel._module.__wrapped__()

    assert seen["env"] == before
    assert "-ccbin" not in seen["flags"]
    assert {k: os.environ.get(k) for k in ("CC", "CXX")} == before
