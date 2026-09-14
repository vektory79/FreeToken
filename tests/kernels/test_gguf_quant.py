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


@pytest.mark.parametrize("qtype", (GGML_IQ3_XXS, GGML_IQ4_XS))
def test_dense_iq_above_mmvq_cutoff_fails_loudly(qtype):
    # iq formats have no MMQ kernel: a dense batch above the MMVQ cutoff must raise,
    # not silently dequantize per call (routed experts route through moe_vec).
    from freetoken.layers.gguf import fused_mul_mat_gguf

    n = BLOCK_SHAPE[qtype][0]
    qw = torch.zeros((8, row_bytes(n, qtype)), dtype=torch.uint8)
    x = torch.zeros((8, n), dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match="no MMQ kernel"):
        fused_mul_mat_gguf(x, qw, qtype)


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
