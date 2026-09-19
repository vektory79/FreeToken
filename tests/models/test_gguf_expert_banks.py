"""Phase 5: glm5next gguf expert banks -> offload path.

Covers the hook resolution (weight.py -> glm5_next/gguf.load_gguf_expert_sources),
per-layer ggml-type preservation, bank geometry vs BLOCK_SHAPE, the provider
dispatch (expert_banks fmt "gguf"), sizing, the issue-#186 moe_vec chunk cap, and
a GPU smoke that pushes the fixture banks through an OffloadMoeCache into the
borrowed ggml kernel. The 147 GB checkpoint is never opened (iter fixture only).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import torch

import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from test_glm5_next_gguf import (  # reuse the iter fixture + writer helpers
    _DOWN_BANK_BYTES,
    _EFF,
    _H,
    _ITER_METADATA,
    _NE,
    _bank_vals,
    _iter_tensor_set,
    _pack_q8_0,
    _write_iter_gguf,
)


def _bank_cfg():
    from types import SimpleNamespace

    return SimpleNamespace(
        first_k_dense_replace=1,
        num_layers=3,
        num_moe_layers=2,
        num_experts=_NE,
        hidden_size=_H,
        moe_intermediate_size=_EFF,
        expert_quant="none",
        moe_weight_format="gguf",
    )


def test_gguf_expert_sources_hook(tmp_path):
    from freetoken.models.weight import load_gguf_moe_expert_sources

    path = _write_iter_gguf(tmp_path / "banks.gguf")
    banks, types = load_gguf_moe_expert_sources(str(path), _bank_cfg())

    # checkpoint layers 1,2 -> bank layers 0,1 (first_k_dense_replace = 1); blk.3
    # is the iter fixture's MTP slot and must not appear.
    assert sorted(banks) == ["down", "gate", "up"]
    for role, rb in (("gate", 272), ("up", 272), ("down", 210)):  # Q8_0/Q6_K row widths
        assert len(banks[role]) == 2
        for t in banks[role]:
            assert t.dtype == torch.uint8
            assert t.shape == (_NE, _EFF, rb)  # one expert plane per dim-0 slot, packed verbatim
            assert t.is_contiguous()
    assert types == ((8, 8, 14), (8, 8, 14))  # Q8_0/Q8_0/Q6_K per bank layer


def test_gguf_bank_slicing_matches_source(tmp_path):
    # a dim-0 slice of the bank tensor must equal the expert's packed rows verbatim
    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.models.weight import load_gguf_moe_expert_sources

    path = _write_iter_gguf(tmp_path / "banks.gguf")
    bank_list, _ = load_gguf_moe_expert_sources(str(path), _bank_cfg())
    gate0 = bank_list["gate"][0]
    src = next(
        t for t in iter_gguf_tensors(str(path)) if t.name == "blk.1.ffn_gate_exps.weight"
    )
    raw = src.packed().reshape(_NE, _EFF, -1)  # [E, rows, row_bytes], packed rows
    ref = raw
    # expert e's packed rows are the dim-0 slice verbatim (no dequant, no repack) -
    # exactly what the offload cache copies per slot
    assert torch.equal(gate0.reshape(-1), ref.reshape(-1))
    assert torch.equal(gate0[0], ref[0])


def test_provider_dispatch_gguf(tmp_path):
    from freetoken.moe.expert_banks import load_expert_banks

    path = _write_iter_gguf(tmp_path / "banks.gguf")
    banks = load_expert_banks(str(path), _bank_cfg(), device=torch.device("cpu"), dtype=torch.bfloat16)
    assert banks.quant_format == "gguf"
    assert sorted(banks.sources) == ["down", "gate", "up"]
    assert banks.gguf_types == ((8, 8, 14), (8, 8, 14))
    assert banks.streamed is False
    for role in ("gate", "up", "down"):
        assert len(banks.sources[role]) == 2
        assert banks.sources[role][0].shape[0] == _NE and banks.sources[role][0].shape[1] == _EFF


def test_bank_bytes_estimate_gguf():
    from freetoken.moe.expert_banks import bank_bytes_estimate
    from freetoken.models.gguf.dequant import BLOCK_SHAPE

    cfg = _bank_cfg()
    est = bank_bytes_estimate(cfg)
    gate_up = 2 * _EFF * (_H // 256) * 98  # IQ3_XXS rows
    down = _H * (_EFF // 256) * 210  # conservative Q6_K rows
    assert est == cfg.num_moe_layers * _NE * (gate_up + down)
    assert (BLOCK_SHAPE[18][1], BLOCK_SHAPE[23][1], BLOCK_SHAPE[14][1]) == (98, 136, 210)


def test_moe_vec_chunk_cap():
    from freetoken.layers.moe import _MOE_VEC_MAX_GRID, _assert_moe_vec_chunk

    _assert_moe_vec_chunk(4096, 8)  # default max-prefill-length is unaffected
    _assert_moe_vec_chunk(8191, 8)  # the cap itself
    _assert_moe_vec_chunk(65535, 1)  # the exact grid boundary passes
    with pytest.raises(ValueError, match="max-prefill-length"):
        _assert_moe_vec_chunk(8192, 8)
    # boundary-exact: 65536 = the first grid value the moe_vec kernel cannot index
    with pytest.raises(ValueError, match="65536 exceeds the ggml moe_vec grid limit 65535"):
        _assert_moe_vec_chunk(65536, 1)
    assert _MOE_VEC_MAX_GRID == 65535


def test_cpu_executor_claims_gguf():
    """Flipped for the W1 CPU-GEMV work: the CPU MoE executor now serves the gguf
    K-quant family. The per-projection types (IQ3_XXS/IQ4_XS/Q6_K) resolve from
    cache.gguf_types via _split_gguf_formats / _resolve_gguf_banks."""
    from freetoken.moe.cpu_executor import _GGUF_TYPE_FMTS, _WFMT_IDS

    assert "gguf" in _WFMT_IDS
    # every supported gguf type maps to a real C++ WFmt id (>= 0; "gguf" itself is
    # the executor-level alias, resolved per role before the C++ boundary)
    assert all(_WFMT_IDS[name] >= 0 for name in _GGUF_TYPE_FMTS.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_offload_cache_smoke_gguf_banks(tmp_path):
    # end-to-end fixture-level smoke: banks -> OffloadMoeCache -> borrowed kernel
    from freetoken.models.weight import load_gguf_moe_expert_sources
    from freetoken.moe.offload_cache import OffloadMoeCache

    path = _write_iter_gguf(tmp_path / "banks.gguf")
    banks, types = load_gguf_moe_expert_sources(str(path), _bank_cfg())
    cache = OffloadMoeCache(
        num_layers=2, num_experts=_NE, cache_size=2 * _NE,
        device=torch.device("cuda"), quant_format="gguf", gguf_types=types,
    )
    cache.set_bank_sources({role: [t.cuda() for t in per_layer] for role, per_layer in banks.items()})

    torch.manual_seed(0)
    # x scale ~1.0: the kernel quantizes x to q8_1 (fp32 d * int8 in HALF, so the
    # products must stay inside fp16 range); the byte-random fixture scales then
    # keep every dot product finite for the parity check.
    x = torch.randn(5, _H, dtype=torch.float32, device="cuda")
    topk_ids = torch.stack([torch.randperm(_NE, device="cuda")[:2].to(torch.int32) for _ in range(5)])
    # identity routing over the (tokens*top_k) slot dimension for the down call:
    # slot i of the stacked activations reads expert i mod E
    routed_ids = (torch.arange(10, device="cuda") % _NE).to(torch.int32).unsqueeze(1)
    from freetoken.kernel.gguf import ggml_moe_a8_vec

    gate_t, up_t, down_t = types[0]
    # the kernel consumes the stacked weight as [E, row, *] - exactly the bank
    # layout (expert-first planes); rows inside the plane stay contiguous.
    gate_w = cache.bank_sources["gate"][0].cuda()
    up_w = cache.bank_sources["up"][0].cuda()
    down_w = cache.bank_sources["down"][0].cuda()
    g = ggml_moe_a8_vec(x, gate_w, topk_ids, 2, gate_t, _EFF, 5)
    u = ggml_moe_a8_vec(x, up_w, topk_ids, 2, up_t, _EFF, 5)
    inter = torch.nn.functional.silu(g * u)
    # the fixture's Q8_0 scales are byte-random, so the intermediate magnitude is
    # ~1e6 - far outside fp16; the q8_1 quantizer stores its fp32 scale in HALF.
    # Scale to a realistic activation range before the down projection.
    inter = inter / inter.abs().amax(dim=1, keepdim=True).clamp(min=1.0) * 8.0
    # the down projection consumes ONE row per routed expert: identity routing
    # (slot i -> expert i of the routed set, activations already stacked).
    d = ggml_moe_a8_vec(inter, down_w, routed_ids, 1, down_t, _H, 10)
    assert d.shape == (10, _H) and torch.isfinite(d).all()

    # parity vs the dequant reference for the routed experts: the bank packs
    # [E, rows, row_bytes] with per-row Q8_0 blocks (block 256 / 34 bytes), so the
    # reference dequantizes row-wise from the same raw bytes.
    from freetoken.models.gguf.reader import iter_gguf_tensors

    src_t = next(t for t in iter_gguf_tensors(str(path)) if t.name == "blk.1.ffn_gate_exps.weight")
    # packed rows = E * EFF (one row per output element, 8 Q8_0 blocks of 34 bytes);
    # row r decodes to output values r*256..r*256+255 -> direct [E, EFF, H] grid
    packed = src_t.packed().reshape(_NE, _EFF, 8, 34)
    # _pack_q8_0 layout: byte 0x3C at b*34+1 is the fp16 scale head (LE), quants at
    # b*34+2..+33; the block d (1.0) then quants int8
    ds = np.ones((_NE, _EFF, 8), np.float32)  # the fixture's scale is exactly 1.0
    qs_np = np.ascontiguousarray(packed[..., 2:].numpy()).reshape(_NE, _EFF, 8 * 32).view(np.int8)
    qs = torch.from_numpy(qs_np).reshape(_NE, _EFF, 8, 32).float()
    ref_gate = (torch.from_numpy(ds).unsqueeze(-1) * qs).reshape(_NE, _EFF, _H).numpy()

    # ggml_moe_a8_vec quantizes x to q8_1 and the fixture's Q8_0 scales are
    # byte-random (unlike a real checkpoint), so compare DIRECTIONAL parity only:
    # the kernel output must correlate with the fp32 reference for each token's
    # first routed expert.
    x32 = x.float().cpu().numpy()
    for t in range(5):
        ref_g = ref_gate[int(topk_ids[t, 0])].astype(np.float32) @ x32[t]
        got_g = g.reshape(5, 2, _EFF)[t, 0].float().cpu().numpy()
        corr = np.corrcoef(ref_g, got_g)[0, 1]
        assert corr > 0.95, (t, corr)


def _craft_uniform_gguf_banks(types, E, rows_ne0=None):
    """Deterministic packed banks: Q8_0 rows decode to +0.25 everywhere (fp16 d=0.25,
    q=1); Q6_K rows to -16 (fp16 d=0.5, int8 scales=1, zero nibbles -> q=-32). Uniform
    weights make the whole MoE output analytic. ``rows_ne0`` maps role -> (output
    rows, input dim ne0); blocks pack over ne0, so the two are distinct for
    non-square geometries."""
    from freetoken.models.gguf.dequant import BLOCK_SHAPE

    rows_ne0 = rows_ne0 or {"gate": (_EFF, _H), "up": (_EFF, _H), "down": (_H, _EFF)}
    d_bytes = {
        8: torch.tensor([0.25], dtype=torch.float16).view(torch.uint8),  # block_q8_0 d
        14: torch.tensor([0.5], dtype=torch.float16).view(torch.uint8),  # block_q6_K d
    }
    banks = {}
    for role, typ in zip(("gate", "up", "down"), types):
        rows, ne0 = rows_ne0[role]
        blk, rb = BLOCK_SHAPE[typ]
        nblk = ne0 // blk
        t = torch.zeros(E, rows, nblk * rb, dtype=torch.uint8).view(E, rows, nblk, rb)
        if typ == 8:
            t[..., 0:2] = d_bytes[typ]  # block_q8_0: half d first, then 32 int8 quants
            t[..., 2:] = 1
        else:
            # block_q6_K keeps d LAST (ql, qh, scales, d; ggml-common.h:104-110) -
            # the fixture's 208:210 scale pin encodes the same convention
            t[..., 192:208] = 1  # int8 scales
            t[..., 208:210] = d_bytes[typ]
        banks[role] = t.view(E, rows, nblk * rb)
    return banks


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_expert_gemm_gguf_dispatch():
    # Drives OffloadMoELayer._expert_gemm's gguf branch end to end: the tokens/top_k
    # wiring, cache.gguf_types per-layer routing, and the gated epilogue. Layer 0 is
    # all-Q6_K, layer 1 all-Q8_0 -> the type dispatch must follow the layer, and the
    # uniform weights pin the output to an analytic reference.
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.moe.offload_cache import OffloadMoeCache

    layer_types = ((14, 14, 14), (8, 8, 8))  # Q6_K layer, then Q8_0 layer
    cache = OffloadMoeCache(
        num_layers=2, num_experts=_NE, cache_size=2 * _NE,
        device=torch.device("cuda"), quant_format="gguf", gguf_types=layer_types,
    )
    assert cache.gguf_types == layer_types

    layer = OffloadMoELayer.__new__(OffloadMoELayer)  # the branch only needs these attrs
    layer.quant_method = None
    layer.activation = "swiglu_clamp"
    layer.alpha = 1.0
    layer.limit = 10.0

    torch.manual_seed(0)
    # Per-token bias pushes every |g| past the swiglu_clamp limit: the clamped
    # epilogue absorbs the q8_1 x-quant noise exactly, so the analytic reference
    # stays tight (rtol 0.02) instead of chasing 2*delta_g/g noise on small signals.
    base = 0.05 * torch.randn(5, _H, dtype=torch.float32, device="cuda")
    bias = torch.linspace(1.0, 3.0, 5, device="cuda").unsqueeze(-1)
    x = (base + bias).contiguous()
    topk_ids = torch.stack(
        [torch.randperm(_NE, device="cuda")[:2].to(torch.int32) for _ in range(5)]
    )
    topk_weights = torch.rand(5, 2, dtype=torch.float32, device="cuda")
    topk_weights = (topk_weights / topk_weights.sum(-1, keepdim=True)).contiguous()

    outs = {}
    for lid, types in enumerate(layer_types):
        banks = _craft_uniform_gguf_banks(types, _NE)
        views = tuple(banks[role].cuda() for role in ("gate", "up", "down"))
        layer.layer_id = lid
        outs[lid] = layer._expert_gemm(
            cache, x, topk_weights, topk_ids, views=views, n=None, alphas=None,
            is_prefill=False,
        )
        assert outs[lid].shape == (5, _H)
        assert torch.isfinite(outs[lid]).all()

    # analytic reference: uniform gate/up rows make the per-slot gate/up scalars
    # w*sum(x); mirror the epilogue with the production kernel on the (5, [gate; up])
    # pair, then the down projection collapses to w * I * inter per slot.
    from freetoken.layers import swiglu_clamp_and_mul

    S = x.sum(-1)
    for lid, types in enumerate(layer_types):
        w = -16.0 if types[0] == 14 else 0.25
        pair = torch.stack((w * S, w * S), dim=-1)
        inter = swiglu_clamp_and_mul(pair, alpha=1.0, limit=10.0)  # (tokens, 1)
        ref = (topk_weights * (w * _EFF) * inter).sum(-1)
        ref = ref.unsqueeze(-1).expand(-1, _H)  # uniform down weights: every row identical
        assert torch.allclose(outs[lid], ref, rtol=0.02, atol=1e-2), lid
    assert not torch.allclose(outs[0], outs[1])  # per-layer types really dispatched


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("mtile", (4, 8, 16, 32))
def test_expert_gemm_gguf_grouped_prefill_matches_decode(monkeypatch, mtile):
    # v2 grouped MMQ prefill: is_prefill=True routes through the moe_align trio +
    # grouped ggml_moe_a8 (ONE trio shared by gate/up and down), stays numerically
    # consistent with the moe_vec decode path on identical inputs, and leaves the
    # decode path (moe_vec, no trio) completely untouched. Swept over the v3a
    # m-tile knob: gate/up/down must still share ONE align call at every tile
    # because the knob is global (all served types report the same block size).
    monkeypatch.setenv("FREETOKEN_GGUF_MOE_MTILE", str(mtile))
    import freetoken.moe.fused as fused_mod

    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.moe.offload_cache import OffloadMoeCache

    types = (8, 8, 8)  # Q8_0 layer: uniform banks keep the reference analytic
    cache = OffloadMoeCache(
        num_layers=1, num_experts=_NE, cache_size=_NE,
        device=torch.device("cuda"), quant_format="gguf", gguf_types=(types,),
    )
    layer = OffloadMoELayer.__new__(OffloadMoELayer)  # the branch only needs these attrs
    layer.quant_method = None
    layer.activation = "swiglu_clamp"
    layer.alpha = 1.0
    layer.limit = 10.0
    layer.layer_id = 0

    torch.manual_seed(0)
    # the per-token bias pushes every |g| past the swiglu_clamp limit, so the
    # clamped epilogue absorbs the q8_1 x-quant noise exactly on both paths
    base = 0.05 * torch.randn(5, _H, dtype=torch.float32, device="cuda")
    bias = torch.linspace(1.0, 3.0, 5, device="cuda").unsqueeze(-1)
    x = (base + bias).contiguous()
    topk_ids = torch.stack(
        [torch.randperm(_NE, device="cuda")[:2].to(torch.int32) for _ in range(5)]
    )
    topk_weights = torch.rand(5, 2, dtype=torch.float32, device="cuda")
    topk_weights = (topk_weights / topk_weights.sum(-1, keepdim=True)).contiguous()

    banks = _craft_uniform_gguf_banks(types, _NE)
    views = tuple(banks[role].cuda() for role in ("gate", "up", "down"))

    calls = {"align": 0}
    real_align = fused_mod.moe_align_block_size

    def counting_align(*args, **kwargs):
        calls["align"] += 1
        return real_align(*args, **kwargs)

    monkeypatch.setattr(fused_mod, "moe_align_block_size", counting_align)
    out_pre = layer._expert_gemm(
        cache, x, topk_weights, topk_ids, views=views, n=_NE, alphas=None, is_prefill=True
    )
    assert out_pre.shape == (5, _H)
    assert torch.isfinite(out_pre).all()
    # gate/up and down share one trio (equal MOE_X block sizes -> a single align)
    assert calls["align"] == 1

    # decode must not build a trio at all (the branch is is_prefill-only)
    def bomb(*args, **kwargs):
        raise AssertionError("decode must not build a moe_align trio")

    monkeypatch.setattr(fused_mod, "moe_align_block_size", bomb)
    out_dec = layer._expert_gemm(
        cache, x, topk_weights, topk_ids, views=views, n=None, alphas=None, is_prefill=False
    )
    torch.testing.assert_close(out_pre, out_dec, rtol=2e-2, atol=1e-2)

    # analytic reference: uniform gate/up rows make the per-slot gate/up scalars
    # w*sum(x); mirror the epilogue, then the uniform down bank collapses to a
    # weighted sum of w * _EFF * inter per slot (same construction as the
    # test_expert_gemm_gguf_dispatch reference above)
    from freetoken.layers import swiglu_clamp_and_mul

    S = x.sum(-1)
    w = 0.25
    pair = torch.stack((w * S, w * S), dim=-1)
    inter = swiglu_clamp_and_mul(pair, alpha=1.0, limit=10.0)
    ref = (topk_weights * (w * _EFF) * inter).sum(-1).unsqueeze(-1).expand(-1, _H)
    assert torch.allclose(out_pre, ref, rtol=0.02, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("mtile", (4, 8, 16, 32))
def test_grouped_prefill_noncontig_activations(monkeypatch, mtile):
    # lesson 028f2d9: the ggml kernels assume row-major activations; the grouped
    # prefill must .contiguous() a strided/split view (e.g. KDA projections) instead
    # of consuming it as garbage - the result must equal the contiguous input's.
    # Re-pinned across the v3a m-tile sweep (the tile must not change the fix).
    monkeypatch.setenv("FREETOKEN_GGUF_MOE_MTILE", str(mtile))
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.moe.offload_cache import OffloadMoeCache

    types = (8, 8, 8)  # Q8_0 layer; uniform banks, deterministic kernels
    cache = OffloadMoeCache(
        num_layers=1, num_experts=_NE, cache_size=_NE,
        device=torch.device("cuda"), quant_format="gguf", gguf_types=(types,),
    )
    layer = OffloadMoELayer.__new__(OffloadMoELayer)  # the branch only needs these attrs
    layer.quant_method = None
    layer.activation = "silu"
    layer.alpha = 1.0
    layer.limit = None
    layer.layer_id = 0

    torch.manual_seed(0)
    # bf16: the silu epilogue's flashinfer kernel dispatches fp16-family only, and
    # the ggml kernels take bf16 activations natively
    big = torch.randn(5, 2 * _H, dtype=torch.bfloat16, device="cuda")
    x_nc = big[:, :_H]  # split view: non-contiguous rows over a wider stride
    assert not x_nc.is_contiguous()
    topk_ids = torch.stack(
        [torch.randperm(_NE, device="cuda")[:2].to(torch.int32) for _ in range(5)]
    )
    topk_weights = torch.rand(5, 2, dtype=torch.float32, device="cuda")
    banks = _craft_uniform_gguf_banks(types, _NE)
    views = tuple(banks[role].cuda() for role in ("gate", "up", "down"))

    out_nc = layer._expert_gemm(
        cache, x_nc, topk_weights, topk_ids, views=views, n=_NE, alphas=None, is_prefill=True
    )
    out_c = layer._expert_gemm(
        cache, x_nc.contiguous(), topk_weights, topk_ids, views=views, n=_NE, alphas=None,
        is_prefill=True,
    )
    assert out_nc.shape == out_c.shape == (5, _H)
    assert torch.isfinite(out_nc).all()
    assert torch.equal(out_nc, out_c), "strided input must match the contiguous one exactly"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_grid_caps_raise_before_kernel_launch(monkeypatch):
    # the prefill grid caps fire in the entry guards, before any MoE kernel launch:
    # a [8192, 8] chunk (65536 pairs > 65535, issue #186) RUNS on the grouped MMQ
    # path (the grouped grid counts bins, not pairs) and only a bigger chunk trips
    # the grouped sorted_numel/block_size cap; the moe_vec pair cap still guards
    # its own branch - for prefill that branch is reachable only via ggml types
    # outside the grouped kernel's moe switch (declared IQ2_XS here). The grouped
    # cap arithmetic depends on the v3a m-tile, so the env is pinned explicitly
    # and both tile 4 and tile 8 boundaries are covered below.
    monkeypatch.setenv("FREETOKEN_GGUF_MOE_MTILE", "4")
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.moe.offload_cache import OffloadMoeCache

    types = (8, 8, 8)
    cache = OffloadMoeCache(
        num_layers=1, num_experts=_NE, cache_size=2 * _NE,
        device=torch.device("cuda"), quant_format="gguf", gguf_types=(types,),
    )
    layer = OffloadMoELayer.__new__(OffloadMoELayer)  # the branch only needs these attrs
    layer.quant_method = None
    layer.activation = "silu"
    layer.alpha = 1.0
    layer.limit = None
    layer.layer_id = 0

    banks = _craft_uniform_gguf_banks(types, _NE)
    views = tuple(banks[role].cuda() for role in ("gate", "up", "down"))
    ids = torch.randint(0, _NE, (8192, 8), dtype=torch.int32, device="cuda")
    w = torch.rand(8192, 8, dtype=torch.float32, device="cuda")
    # zeros, not empty: uninitialized bf16 garbage can sit at ~1e38, whose q8_1
    # scale d = amax/127 overflows fp16 and makes the dots inf; this test only
    # needs finite math on the runs that reach a kernel
    x = torch.zeros((8192, _H), dtype=torch.bfloat16, device="cuda")

    out = layer._expert_gemm(
        cache, x, w, ids, views=views, n=_NE, alphas=None, is_prefill=True
    )
    assert out.shape == (8192, _H)
    assert torch.isfinite(out).all()

    # the moe_vec 65535-pair cap fires before any kernel launch: the unsupported
    # type (17 = IQ2_XS, block size 0) routes prefill to the moe_vec fallback
    # whose entry guard raises - the Q8_0-layout bank bytes are never read
    no_group_cache = OffloadMoeCache(
        num_layers=1, num_experts=_NE, cache_size=2 * _NE,
        device=torch.device("cuda"), quant_format="gguf", gguf_types=((17, 17, 17),),
    )
    with pytest.raises(ValueError, match="moe_vec grid limit"):
        layer._expert_gemm(
            no_group_cache, x, w, ids, views=views, n=_NE, alphas=None, is_prefill=True
        )

    big_ids = torch.randint(0, _NE, (32768, 8), dtype=torch.int32, device="cuda")
    big_w = torch.rand(32768, 8, dtype=torch.float32, device="cuda")
    big_x = torch.zeros((32768, _H), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="grouped MMQ grid limit"):
        layer._expert_gemm(
            cache, big_x, big_w, big_ids, views=views, n=_NE, alphas=None, is_prefill=True
        )

    # v3a tile-8 re-pin: the grouped cap scales with block_size (262159 // 4 =
    # 65539 trips at tile 4; 262179 // 8 = 32772 must RUN) and the next boundary
    # sits near ~522k pairs (65536x8: 524323 // 8 = 65540). The moe_vec fallback
    # cap stays env-independent - moe_vec never reads the knob.
    monkeypatch.setenv("FREETOKEN_GGUF_MOE_MTILE", "8")
    out8 = layer._expert_gemm(
        cache, big_x, big_w, big_ids, views=views, n=_NE, alphas=None, is_prefill=True
    )
    assert out8.shape == (32768, _H)
    assert torch.isfinite(out8).all()

    huge_ids = torch.randint(0, _NE, (65536, 8), dtype=torch.int32, device="cuda")
    huge_w = torch.rand(65536, 8, dtype=torch.float32, device="cuda")
    huge_x = torch.zeros((65536, _H), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="grouped MMQ grid limit"):
        layer._expert_gemm(
            cache, huge_x, huge_w, huge_ids, views=views, n=_NE, alphas=None, is_prefill=True
        )
    with pytest.raises(ValueError, match="moe_vec grid limit"):
        layer._expert_gemm(
            no_group_cache, x, w, ids, views=views, n=_NE, alphas=None, is_prefill=True
        )


def test_engine_offload_cache_gets_gguf_types():
    # The wiring fixture tests cannot reach: _init_offload_moe_cache must hand
    # banks.gguf_types to the built OffloadMoeCache (the None field default keeps
    # every other format unaffected).
    import inspect

    from freetoken.engine.engine import Engine

    src = inspect.getsource(Engine._init_offload_moe_cache)
    assert "banks.gguf_types" in src  # per-group wiring: tuple(banks.gguf_types[l] ...)
    assert "self.moe_offload_caches = caches" in src  # the partition list is exported


def _variant_cfg(hidden, inter):
    from types import SimpleNamespace

    return SimpleNamespace(
        first_k_dense_replace=1,
        num_layers=3,
        num_moe_layers=2,
        num_experts=_NE,
        hidden_size=hidden,
        moe_intermediate_size=inter,
        expert_quant="none",
        moe_weight_format="gguf",
    )


def test_gguf_bank_rows_per_role(tmp_path, monkeypatch):
    # review B1: down packs H output rows while gate/up pack I - the loader must
    # take the row count from EACH tensor, not reuse gate's. The stock fixture is
    # square (_H == _EFF) and masks the bug, so patch the fixture to I = _H // 2.
    import test_glm5_next_gguf as fixture_mod

    monkeypatch.setattr(fixture_mod, "_EFF", _H // 2)
    path = fixture_mod._write_iter_gguf(
        tmp_path / "nonsquare.gguf",
        metadata={
            **_ITER_METADATA,
            "glm5next.expert_feed_forward_length": _H // 2,
            "glm5next.expert_shared_feed_forward_length": _H // 2,
        },
    )
    from freetoken.models.weight import load_gguf_moe_expert_sources

    banks, types = load_gguf_moe_expert_sources(str(path), _variant_cfg(_H, _H // 2))
    for layer in range(2):
        assert banks["gate"][layer].shape == (_NE, _H // 2, _H // 32 * 34)
        assert banks["up"][layer].shape == (_NE, _H // 2, _H // 32 * 34)
        assert banks["down"][layer].shape == (_NE, _H, 210)  # rows stay H (Q6_K width)
    assert types == ((8, 8, 14), (8, 8, 14))


def test_gguf_types_role_order_not_file_order(tmp_path):
    # review B2: the per-layer type tuple must be (gate, up, down) even when the
    # file lists the down tensor first; distinct per-role types make a swap visible
    # (the stock Q8_0/Q8_0/Q6_K fixture is symmetric and would not catch it).
    import gguf as gguf_mod

    import test_glm5_next_gguf as fixture_mod
    from freetoken.models.weight import load_gguf_moe_expert_sources

    q8 = gguf_mod.GGMLQuantizationType.Q8_0
    i3 = gguf_mod.GGMLQuantizationType.IQ3_XXS
    q6k = gguf_mod.GGMLQuantizationType.Q6_K
    full = fixture_mod._iter_tensor_set()  # full trunk (token_embd etc. for the spec resolver)
    downs = {n: e for n, e in full.items() if "ffn_down_exps" in n}
    gates = {n: e for n, e in full.items() if "ffn_gate_exps" in n}
    ups = {n: e for n, e in full.items() if "ffn_up_exps" in n}
    rest = {n: e for n, e in full.items() if "exps" not in n}
    for layer in (1, 2):
        downs[f"blk.{layer}.ffn_down_exps.weight"] = (fixture_mod._DOWN_BANK_BYTES, q6k)
        gates[f"blk.{layer}.ffn_gate_exps.weight"] = (
            fixture_mod._pack_q8_0(fixture_mod._bank_vals(0)), q8,
        )
        ups[f"blk.{layer}.ffn_up_exps.weight"] = (
            np.random.default_rng(layer).integers(0, 256, (_NE, _H, 98), dtype=np.uint8), i3,
        )
    ts = {**downs, **gates, **ups, **rest}  # down tensors FIRST in the file
    path = fixture_mod._write_iter_gguf(tmp_path / "downfirst.gguf", tensors=ts)
    banks, types = load_gguf_moe_expert_sources(str(path), _bank_cfg())
    assert types == ((8, 18, 14), (8, 18, 14))  # role order, not encounter order
    assert banks["gate"][0].shape == (_NE, _EFF, 272)
    assert banks["up"][0].shape == (_NE, _EFF, 98)
    assert banks["down"][0].shape == (_NE, _H, 210)


def test_gguf_rejects_kernel_unsupported_type(tmp_path):
    # review B6: Q5_K has no MMVQ case in the moe_vec switch - the loader must
    # reject it loudly instead of letting the kernel return silent zeros.
    import gguf as gguf_mod

    import test_glm5_next_gguf as fixture_mod

    q8 = gguf_mod.GGMLQuantizationType.Q8_0
    q5k = gguf_mod.GGMLQuantizationType.Q5_K
    ts = fixture_mod._iter_tensor_set()  # full trunk (token_embd etc. for the spec resolver)
    for layer in (1, 2):
        ts[f"blk.{layer}.ffn_down_exps.weight"] = (
            np.random.default_rng(layer).integers(0, 256, (_NE, _H, 176), dtype=np.uint8), q5k,
        )
    path = fixture_mod._write_iter_gguf(tmp_path / "q5k.gguf", tensors=ts)
    from freetoken.models.weight import load_gguf_moe_expert_sources

    with pytest.raises(ValueError, match="no MMVQ kernel"):
        load_gguf_moe_expert_sources(str(path), _bank_cfg())


def test_adjust_config_clamps_gguf_prefill_chunk():
    # review B5: the default 8192 chunk at top_k 8 is exactly the 65536 moe_vec
    # grid limit - config time clamps to 8191 on the gguf path; others untouched.
    from types import SimpleNamespace

    from freetoken.engine.engine import _adjust_config

    model_config = SimpleNamespace(
        single_stream_only=False,
        is_moe=True,
        expert_quant="none",
        moe_weight_format="gguf",
        hidden_act="swiglu_clamp",
        has_swa_attention=False,
        has_linear_attention=False,
        num_experts_per_tok=8,
    )

    class Cfg:
        moe_cache_auto = False
        moe_cache_size = 0
        moe_cache_rate = None
        moe_strategy = "offload"
        moe_cpu_layers = None
        max_running_req = 4
        cuda_graph_max_bs = 2
        cuda_graph_bs = [1, 2]
        max_seq_len = 1024
        page_size = 1
        attention_backend = "fi"
        num_page_override = None
        num_token_override = None
        max_extend_tokens = 8192

        @property
        def model_config(self):
            return model_config

    cfg = Cfg()
    _adjust_config(cfg)
    assert cfg.max_extend_tokens == 8191  # 65535 // 8

    cfg.max_extend_tokens = 8191
    _adjust_config(cfg)
    assert cfg.max_extend_tokens == 8191  # already inside the cap: no-op

    model_config.moe_weight_format = None  # non-gguf expert path: no clamp
    cfg.max_extend_tokens = 8192
    _adjust_config(cfg)
    assert cfg.max_extend_tokens == 8192


def test_set_bank_sources_rejects_heterogeneous_widths():
    # review B3: one unified slot pool per bank cannot serve layers with different
    # packed row widths (the moe_vec kernel reads slot rows at the layer's own
    # width) - reject loudly instead of tripping an assert (ggml issue #194).
    from freetoken.moe.offload_cache import OffloadMoeCache

    cache = OffloadMoeCache(
        num_layers=2, num_experts=_NE, cache_size=2 * _NE,
        device=torch.device("cpu"), quant_format="gguf",
        gguf_types=((8, 8, 14), (8, 8, 8)),
    )
    sources = {
        "gate": [torch.zeros(_NE, _EFF, 272, dtype=torch.uint8) for _ in range(2)],
        "up": [torch.zeros(_NE, _EFF, 272, dtype=torch.uint8) for _ in range(2)],
        "down": [
            torch.zeros(_NE, _H, 210, dtype=torch.uint8),
            torch.zeros(_NE, _H, 34, dtype=torch.uint8),  # layer 1 packs narrower
        ],
    }
    with pytest.raises(ValueError, match="issue #194"):
        cache.set_bank_sources(sources)


def test_moe_cache_budget_split_rule():
    # review B3: the slot budget splits proportionally to each group's total bank
    # bytes (here: equal byte weights -> equal shares), every group is floored at
    # num_experts, an unfundable explicit budget rejects naming the floors, and the
    # auto byte envelope shrinks the over-floor groups back inside the plan.
    from types import SimpleNamespace

    from freetoken.engine.cache_budget import expert_bytes_per_slot
    from freetoken.engine.engine import Engine

    E = 4
    # per-layer (gate rows, gate rb, down rows, down rb) chosen so every layer weighs
    # exactly 278,528 bytes: 2 sig-A layers, 1 sig-B, 1 sig-C
    sigs = [(128, 272, 256, 272), (128, 272, 256, 272), (256, 272, 512, 272), (128, 544, 256, 544)]
    sources = {
        "gate": [torch.empty(E, gr, grb, dtype=torch.uint8) for gr, grb, _, _ in sigs],
        "up": [torch.empty(E, gr, grb, dtype=torch.uint8) for gr, grb, _, _ in sigs],
        "down": [torch.empty(E, dr, drb, dtype=torch.uint8) for _, _, dr, drb in sigs],
    }
    banks = SimpleNamespace(sources=sources)
    groups = Engine._group_bank_layers(banks, len(sigs))
    assert groups == [[0, 1], [2], [3]]

    sizes = Engine._split_moe_cache_budget(banks, groups, 1000, E)
    assert sizes == [333, 333, 333]  # equal byte weights -> equal shares (1000//3)

    assert Engine._split_moe_cache_budget(banks, groups, 12, E) == [4, 4, 4]  # exactly the floors
    with pytest.raises(ValueError, match="cannot fund the per-signature slot floors"):
        Engine._split_moe_cache_budget(banks, groups, 11, E)
    # the advice names the TRUE binding group's minimum (ceil(E*W/w_i)), not groups*E
    with pytest.raises(ValueError, match=r"at least 12"):
        Engine._split_moe_cache_budget(banks, groups, 11, E)

    # auto envelope: the raw shares of the 400-slot budget are byte-proportional
    # (133 each); the envelope then shrinks the widest-footprint groups - B to its
    # floor, C partially - until the partitioned total fits 400 average-width slots
    b_avg = expert_bytes_per_slot(sources)
    cap_slots = 400
    sizes = Engine._split_moe_cache_budget(banks, groups, cap_slots, E, byte_cap_slots=cap_slots)
    per_group_slot_bytes = [
        expert_bytes_per_slot({n: [sources[n][l] for l in m] for n in sources})
        for m in groups
    ]
    assert all(s >= E for s in sizes)
    assert sum(s * b for s, b in zip(sizes, per_group_slot_bytes)) <= cap_slots * b_avg
    assert sizes == [133, 4, 129]


def test_engine_partitions_degenerate_single_signature():
    # a uniform-signature file must degenerate to ONE group covering every layer and
    # the untouched single-cache budget (today's exact behavior)
    from types import SimpleNamespace

    from freetoken.engine.engine import Engine

    sources = {
        "gate": [torch.empty(_NE, _EFF, 272, dtype=torch.uint8) for _ in range(2)],
        "up": [torch.empty(_NE, _EFF, 272, dtype=torch.uint8) for _ in range(2)],
        "down": [torch.empty(_NE, _H, 210, dtype=torch.uint8) for _ in range(2)],
    }
    banks = SimpleNamespace(sources=sources)
    assert Engine._group_bank_layers(banks, 2) == [[0, 1]]
    assert Engine._split_moe_cache_budget(banks, [[0, 1]], 50, _NE) == [50]


def test_engine_partitions_three_groups_end_to_end(tmp_path):
    # review B3 end to end: a three-signature bank set partitions into per-signature
    # caches; each layer's _expert_gemm reads ITS cache (local layer ids, per-group
    # gguf_types - a wrong routing would decode the other group's types and miss the
    # analytic reference); the single-layer minority group is whole-layer-resident.
    from types import SimpleNamespace

    from freetoken.engine.engine import Engine
    from freetoken.layers import swiglu_clamp_and_mul
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.moe.offload_cache import OffloadMoeCache

    E, H, I = 4, 512, 256  # non-square, like the real file (I != H); I >= 256 so the
    # Q6_K/IQ blocks pack over the down bank's ne0
    sigs = [(8, 8, 14), (8, 8, 14), (14, 14, 14), (8, 8, 8)]
    rows_ne0 = {"gate": (I, H), "up": (I, H), "down": (H, I)}

    def rb(rows, ne0, typ):
        from freetoken.models.gguf.dequant import BLOCK_SHAPE

        blk, width = BLOCK_SHAPE[typ]
        return rows, ne0 // blk * width

    per_layer = [_craft_uniform_gguf_banks(t, E, rows_ne0) for t in sigs]
    sources = {
        role: [per_layer[l][role] for l in range(4)] for role in ("gate", "up", "down")
    }
    banks = SimpleNamespace(
        sources=sources,
        gguf_types=tuple(sigs),
        quant_format="gguf",
        layer_residency=None,
        gate_up_alpha=None,
        down_alpha=None,
    )
    groups = Engine._group_bank_layers(banks, 4)
    assert groups == [[0, 1], [2], [3]]
    sizes = Engine._split_moe_cache_budget(banks, groups, 20, E)
    # byte-proportional with floors: every group can hold its whole layer
    assert all(s >= E for s in sizes) and sum(sizes) <= 20
    assert sizes[0] == max(sizes)  # the dominant signature keeps the largest share

    caches = []
    for members, size_g in zip(groups, sizes):
        cache = OffloadMoeCache(
            num_layers=len(members), num_experts=E, cache_size=size_g,
            device=torch.device("cuda"), quant_format="gguf",
            gguf_types=tuple(sigs[l] for l in members),
        )
        cache.set_bank_sources({
            role: [sources[role][l].cuda() for l in members] for role in ("gate", "up", "down")
        })
        caches.append(cache)
    assert caches[2].cache_size >= E  # the single-layer minority is whole-layer-resident

    torch.manual_seed(0)
    base = 0.05 * torch.randn(5, H, dtype=torch.float32, device="cuda")
    bias = torch.linspace(1.0, 3.0, 5, device="cuda").unsqueeze(-1)
    x = (base + bias).contiguous()  # |gate| >> swiglu limit: clamped, quant-noise-free refs
    topk_ids = torch.stack(
        [torch.randperm(E, device="cuda")[:2].to(torch.int32) for _ in range(5)]
    )
    topk_weights = torch.rand(5, 2, dtype=torch.float32, device="cuda")
    topk_weights = (topk_weights / topk_weights.sum(-1, keepdim=True)).contiguous()

    outs = {}
    for gi, members in enumerate(groups):
        for local, g in enumerate(members):
            layer = OffloadMoELayer.__new__(OffloadMoELayer)
            layer.quant_method = None
            layer.activation = "swiglu_clamp"
            layer.alpha = 1.0
            layer.limit = 10.0
            layer.offload_cache = caches[gi]
            layer.layer_id = local
            types = sigs[g]
            banks_l = _craft_uniform_gguf_banks(types, E, rows_ne0)
            views = tuple(banks_l[r].cuda() for r in ("gate", "up", "down"))
            out = layer._expert_gemm(
                caches[gi], x, topk_weights, topk_ids,
                views=views, n=None, alphas=None, is_prefill=False,
            )
            assert out.shape == (5, H) and torch.isfinite(out).all()
            outs[g] = out
            # the layer's own cache must carry ITS signature at the LOCAL id
            assert caches[gi].gguf_types[layer.layer_id] == types

    S = x.sum(-1)
    for g, types in enumerate(sigs):
        w_gate, w_up, w_down = (0.25 if t == 8 else -16.0 for t in types)
        pair = torch.stack((w_gate * S, w_up * S), dim=-1)
        inter = swiglu_clamp_and_mul(pair, alpha=1.0, limit=10.0)
        ref = (topk_weights * (w_down * I) * inter).sum(-1)
        ref = ref.unsqueeze(-1).expand(-1, H)
        assert torch.allclose(outs[g], ref, rtol=0.02, atol=1e-2), g
    assert not torch.allclose(outs[0], outs[2])
    assert not torch.allclose(outs[0], outs[3])


def test_partition_prefill_overlap_degrade_rule():
    # delta review F1: a group whose slot share < 2*num_experts degrades to
    # synchronous materialized prefill instead of flooring every group at 2E (the
    # wide minority signatures would double their slot cost). Real-file shape: the
    # 1326-slot auto plan gives the 39-layer dominant group ~1232 slots (overlap
    # stays on) and the 2-layer/1-layer minorities 56/38 (overlap off).
    from freetoken.engine.engine import Engine

    E = 288
    assert Engine._partition_prefill_overlap(True, 1232, E) is True
    assert Engine._partition_prefill_overlap(True, 56, E) is False
    assert Engine._partition_prefill_overlap(True, 38, E) is False
    assert Engine._partition_prefill_overlap(False, 4096, E) is False


def _hybrid_capability_stub(sig, num_layers, E):
    """Minimal decode_target=hybrid gguf cache for the capability screen (no banks:
    the screen reads quant_format + gguf_types only)."""
    from freetoken.moe.offload_cache import OffloadMoeCache

    return OffloadMoeCache(
        num_layers=num_layers, num_experts=E, cache_size=2 * E,
        device=torch.device("cpu"), quant_format="gguf",
        gguf_types=tuple([tuple(sig)] * num_layers), decode_target="hybrid",
    )


def test_engine_multi_partition_valueerror_lifted_for_capable_signatures():
    # Task 02: the real file's mixed signatures (18,18,14)/(23,23,14) alongside the
    # dominant (18,18,23) - the blanket multi-partition ValueError rejected this
    # exact shape before per-cache executors existed; now every partition's types
    # are executor-capable, so the lift applies (no rejection).
    from freetoken.engine.engine import Engine

    caches = [
        _hybrid_capability_stub(sig, n, 4)
        for sig, n in (((18, 18, 14), 2), ((23, 23, 14), 1), ((18, 18, 23), 1))
    ]
    assert Engine._partition_executor_rejections([[0, 1], [2], [3]], caches) == []


def test_engine_multi_partition_rejection_names_the_incapable_type():
    # a partition carrying a type without a CPU GEMV (Q3_K = 11) still fails
    # loudly, naming the incapable type id; the capable partition passes
    from freetoken.engine.engine import Engine

    bad = _hybrid_capability_stub((11, 11, 11), 1, 4)
    good = _hybrid_capability_stub((18, 18, 23), 1, 4)
    rejections = Engine._partition_executor_rejections([[0], [1]], [bad, good])
    assert len(rejections) == 1
    members, reason = rejections[0]
    assert members == [0]
    assert "11" in reason and "no CPU GEMV" in reason


def test_engine_multi_partition_lift_is_capability_gated():
    # source pin: the raise site consults the per-partition capability screen and
    # the blanket "cannot span partitions" rejection is gone
    import inspect

    from freetoken.engine.engine import Engine

    src = inspect.getsource(Engine._init_offload_moe_cache)
    assert "_partition_executor_rejections" in src
    assert "cannot span partitions" not in src


def test_engine_multi_partition_screen_names_incapable_partition_mid_and_last():
    # Task 07 B-5(b): the screen walks the group list in order, so the incapable
    # partition must be named with its members no matter WHERE it sits - not just
    # in the leading group the rejection test above covers.
    from freetoken.engine.engine import Engine

    good_a = _hybrid_capability_stub((18, 18, 23), 1, 4)
    bad = _hybrid_capability_stub((11, 11, 11), 1, 4)  # Q3_K: no CPU GEMV
    good_b = _hybrid_capability_stub((18, 18, 14), 1, 4)

    mid = Engine._partition_executor_rejections([[0], [1], [2]], [good_a, bad, good_b])
    assert [(m, "no CPU GEMV" in r and "11" in r) for m, r in mid] == [([1], True)]

    last = Engine._partition_executor_rejections([[0], [1], [2]], [good_a, good_b, bad])
    assert [(m, "no CPU GEMV" in r and "11" in r) for m, r in last] == [([2], True)]


def test_prefill_choreography_group_boundary_no_hop_contract():
    # Pins the B3 boundary decision (the comment at Engine._init_offload_moe_cache):
    # a group's last-layer +1 prefetch no-ops on the LOCAL id guard and hands
    # nothing to the next cache; the next group's first layer self-begins its own
    # choreography, or (degraded below 2E) runs synchronous materialize with no
    # copy stream / double buffer at all. The cross-group prefetch hop was
    # evaluated and skipped (2026-09-14): the tuned plan leaves every partition
    # below 2E so no choreography runs at all, and the only boundaries that can
    # exist here lead into degraded minorities with nowhere to land a prefetch.
    # A future hop must deliberately change what this test pins.
    import torch

    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.moe.offload_cache import OffloadMoeCache

    E, H, I = 4, 512, 256
    rows_ne0 = {"gate": (I, H), "up": (I, H), "down": (H, I)}

    def make_cache(sig, num_layers, size_g, overlap):
        cache = OffloadMoeCache(
            num_layers=num_layers, num_experts=E, cache_size=size_g,
            device=torch.device("cuda"), quant_format="gguf",
            gguf_types=tuple([sig] * num_layers), prefill_overlap=overlap,
            prefill_hit_d2d=False,
        )
        cache.set_bank_sources({
            role: [
                _craft_uniform_gguf_banks((sig, sig, sig), E, rows_ne0)[role].cuda()
                for _ in range(num_layers)
            ]
            for role in ("gate", "up", "down")
        })
        return cache

    def layer_for(cache, local):
        layer = OffloadMoELayer.__new__(OffloadMoELayer)
        layer.offload_cache = cache
        layer.layer_id = local
        return layer

    # the overlap-ON group runs the real double-buffer choreography
    cache_a = make_cache(8, 2, 2 * E, True)
    views = layer_for(cache_a, 0)._wait_prefill_overlap(cache_a)
    cache_a.release_prefill_layer(0)
    assert cache_a._prefill_buffer_layer == [0, 1]  # layer 1 was prefetched by layer 0's +1 call
    views_last = layer_for(cache_a, 1)._wait_prefill_overlap(cache_a)
    cache_a.release_prefill_layer(1)
    assert len(views) == len(views_last) == 3

    # THE BOUNDARY: the +1 prefetch targets LOCAL id == num_layers and no-ops
    # without touching any other cache's state
    cache_a.prefetch_prefill_layer(cache_a.num_layers)
    assert cache_a._prefill_buffer_layer == [0, 1]

    # hypothetical overlap->overlap successor: nothing was hopped - the next
    # group's layer 0 is fully self-contained (begin + prefetch + wait), and its
    # buffers hold ITS layer's bytes, not the previous group's
    cache_b = make_cache(14, 1, 2 * E, True)
    assert cache_b._prefill_buffer_layer == [None, None]
    views_b = layer_for(cache_b, 0)._wait_prefill_overlap(cache_b)
    cache_b.release_prefill_layer(0)
    torch.cuda.synchronize()
    for view, role in zip(views_b, ("gate", "up", "down")):
        assert torch.equal(view, cache_b.bank_sources[role][0])

    # the real-rig successor: degraded below 2E, so it owns no copy stream and no
    # double buffer - a boundary hop has nowhere to land
    cache_c = make_cache(14, 1, E, False)
    assert not cache_c.prefill_overlap
    assert getattr(cache_c, "prefill_copy_stream", None) is None
    assert not getattr(cache_c, "prefill_bank_buffers", None)
    assert cache_a.prefetch_prefill_layer(cache_a.num_layers) is None
    assert cache_c.prefetch_prefill_layer(0) is None  # sync group: choreography no-ops

    # degenerate single-group (uniform file): identical semantics on ONE cache
    cache_d = make_cache(8, 2, 2 * E, True)
    layer_for(cache_d, 0)._wait_prefill_overlap(cache_d)
    cache_d.release_prefill_layer(0)
    layer_for(cache_d, 1)._wait_prefill_overlap(cache_d)
    cache_d.release_prefill_layer(1)
    cache_d.prefetch_prefill_layer(cache_d.num_layers)
    assert cache_d._prefill_buffer_layer == [0, 1]


def test_engine_partitions_attach_routing():
    # delta review F4: behavioral pin of the attach loop - each stub layer must end
    # up on ITS signature's cache with the LOCAL layer id, the group's gguf_types at
    # that local id, and the gathered alpha slice for its global bank layer.
    from types import SimpleNamespace

    from freetoken.engine.engine import Engine
    from freetoken.moe.offload_cache import OffloadMoeCache

    E, H, I = 4, 512, 256
    sigs = [(8, 8, 14), (8, 8, 14), (14, 14, 14), (8, 8, 8)]
    rows_ne0 = {"gate": (I, H), "up": (I, H), "down": (H, I)}
    per_layer = [_craft_uniform_gguf_banks(t, E, rows_ne0) for t in sigs]
    sources = {role: [per_layer[l][role] for l in range(4)] for role in ("gate", "up", "down")}
    banks = SimpleNamespace(
        sources=sources,
        gguf_types=tuple(sigs),
        quant_format="gguf",
        layer_residency=None,
        gate_up_alpha=torch.arange(4 * E, dtype=torch.float32),
        down_alpha=torch.arange(4 * E, dtype=torch.float32) + 1000.0,
    )
    groups = Engine._group_bank_layers(banks, 4)
    assert groups == [[0, 1], [2], [3]]
    caches = []
    for members in groups:
        cache = OffloadMoeCache(
            num_layers=len(members), num_experts=E, cache_size=E,
            device=torch.device("cpu"), quant_format="gguf",
            gguf_types=tuple(sigs[l] for l in members),
        )
        gu_a, dn_a = Engine._slice_alphas(banks, members, E)
        cache.set_alphas(gu_a, dn_a)
        cache.set_bank_sources({
            role: [sources[role][l] for l in members] for role in ("gate", "up", "down")
        })
        caches.append(cache)

    routing = {}
    for gi, members in enumerate(groups):
        for local, g in enumerate(members):
            routing[g] = (caches[gi], local)
    layers = [SimpleNamespace(layer_id=l, offload_cache=None) for l in range(4)]
    Engine._route_offload_layers(layers, routing)
    for gi, members in enumerate(groups):
        for local, g in enumerate(members):
            layer = layers[g]
            assert layer.offload_cache is caches[gi]
            assert layer.layer_id == local
            assert caches[gi].gguf_types[local] == sigs[g]
            gu, dn = caches[gi].alphas_for_layer(local)
            assert torch.equal(gu, banks.gate_up_alpha[g * E:(g + 1) * E])
            assert torch.equal(dn, banks.down_alpha[g * E:(g + 1) * E])


def test_engine_partitions_real_model_routing(tmp_path):
    # The missing real-path guard for the layer_id IMA: build the REAL glm5next model
    # (OPList container -> Glm5NextDecoderLayer -> Glm5NextSparseBlock ->
    # make_moe_layer), run the REAL iter_offload_moe_layers walk + the attach, and
    # assert for EVERY OffloadMoELayer: the routed object IS the model's own
    # .experts module (id identity), layer.layer_id is LOCAL to its partition, and
    # layer.layer_id * num_experts stays inside the partition's slot/id arrays (the
    # exact lru_ensure id_base arithmetic that IMA'd on the real boot).
    from types import SimpleNamespace

    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.engine.engine import Engine
    from freetoken.models.gguf.config import build_gguf_shim
    from freetoken.models.glm5_next.gguf import parse_gguf_config
    from freetoken.models.glm5_next.model import Glm5NextForCausalLM
    from freetoken.models.glm5_next.moe import Glm5NextSparseBlock
    from freetoken.moe.offload_cache import OffloadMoeCache, iter_offload_moe_layers

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)

    # five trunk blocks parse to FOUR (the fixture tensors stop at blk.3 - the parse
    # clamps to the tensor-backed blocks): first_k_dense_replace=1 -> three MoE bank
    # layers. Two share a signature (the dominant 2-layer partition), one is a
    # single-layer minority partition; the third signature shape is covered by the
    # budget/e2e tests below.
    meta = {
        **_ITER_METADATA,
        "glm5next.block_count": 5,
        "glm5next.attention.head_count_kv": [0, 0, 1, 1, 1],
        "glm5next.swiglu_clamp_exp": [10.0] * 5,
        "glm5next.swiglu_clamp_shexp": [10.0] * 5,
    }
    path = _write_iter_gguf(tmp_path / "routing.gguf", metadata=meta)
    config = parse_gguf_config(build_gguf_shim(path))
    # the layer CLASS (resident MoELayer vs OffloadMoELayer) is chosen at MODEL BUILD
    # from model_config.moe_strategy - the engine syncs --moe-strategy into the model
    # config before create_model; mirror that here or the walk finds no offload layers
    object.__setattr__(config, "moe_strategy", "offload")
    if not getattr(config, "decode_target", None):
        object.__setattr__(config, "decode_target", "gpu")
    E = config.num_experts
    model = Glm5NextForCausalLM(config)

    sparse = [
        blk.mlp for blk in model.model.layers.op_list
        if isinstance(getattr(blk, "mlp", None), Glm5NextSparseBlock)
    ]
    assert len(sparse) == config.num_moe_layers == 3
    from freetoken.layers.moe import OffloadMoELayer as _OffloadMoELayer

    assert all(isinstance(blk.experts, _OffloadMoELayer) for blk in sparse)
    sigs = [(8, 8, 14), (8, 8, 14), (14, 14, 14)]
    per_layer = [_craft_uniform_gguf_banks(t, E) for t in sigs]
    sources = {role: [per_layer[l][role] for l in range(3)] for role in ("gate", "up", "down")}
    banks = SimpleNamespace(
        sources=sources,
        gguf_types=tuple(sigs),
        quant_format="gguf",
        layer_residency=None,
        gate_up_alpha=None,
        down_alpha=None,
    )
    groups = Engine._group_bank_layers(banks, config.num_moe_layers)
    assert groups == [[0, 1], [2]]
    sizes = Engine._split_moe_cache_budget(banks, groups, 20, E)
    assert sizes == [14, 5]  # byte-proportional, both groups above the E floor
    caches = []
    routing = {}
    for gi, (members, size_g) in enumerate(zip(groups, sizes)):
        cache = OffloadMoeCache(
            num_layers=len(members), num_experts=E, cache_size=size_g,
            device=torch.device("cpu"), quant_format="gguf",
            gguf_types=tuple(sigs[l] for l in members),
        )
        cache.set_bank_sources({
            role: [sources[role][l] for l in members] for role in ("gate", "up", "down")
        })
        caches.append(cache)
        for local, g in enumerate(members):
            routing[g] = (cache, local)

    # the REAL walk over the REAL module tree must find exactly the model's own
    # .experts modules (identity, not equality - this is what the boot IMA hinged on)
    walked = list(iter_offload_moe_layers(model))
    assert len(walked) == 3
    assert {id(x) for x in walked} == {id(blk.experts) for blk in sparse}
    Engine._route_offload_layers(walked, routing)

    for bank, layer in enumerate(walked):
        cache = layer.offload_cache
        assert layer.layer_id < cache.num_layers
        # the IMA invariant: lru_ensure's id_base = layer_id * num_experts must land
        # inside the partition's (num_layers, num_experts) slot/id arrays
        assert layer.layer_id * E < cache.slot_for_id.numel()
        assert cache.gguf_types[layer.layer_id] == sigs[bank]
    # dominant-group locals 0 and 1, minority single-layer-group local 0
    assert [walked[g].layer_id for g in (0, 1)] == [0, 1]
    assert walked[2].layer_id == 0


def test_engine_gguf_types_is_a_verbatim_passthrough():
    # banks.gguf_types reaches OffloadMoeCache as the constructor kwarg and nothing
    # rewrites it: heterogeneous per-projection tuples (the real file's shape) and
    # None (every non-gguf format, ExpertBanks.gguf_types default) both flow verbatim.
    import ast
    import inspect
    import textwrap

    from freetoken.engine.engine import Engine
    from freetoken.moe.offload_cache import OffloadMoeCache

    tree = ast.parse(textwrap.dedent(inspect.getsource(Engine._init_offload_moe_cache)))
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "OffloadMoeCache"
    ]
    assert len(calls) == 1
    kw = {k.arg: ast.unparse(k.value) for k in calls[0].keywords}
    # per-signature partitions hand each group its members' types verbatim
    assert "banks.gguf_types[l] for l in members" in kw["gguf_types"]

    # the cache keeps the tuple it was given (CPU fixture; the CUDA paths are covered
    # by the dispatch tests above)
    cache = OffloadMoeCache(
        num_layers=1, num_experts=_NE, cache_size=2 * _NE, device=torch.device("cpu"),
        quant_format="gguf", gguf_types=((8, 8, 14),),
    )
    assert cache.gguf_types == ((8, 8, 14),)


def _craft_hetero_bank(rows, typ, E):
    """One stacked bank [E, rows, row_bytes] whose every block decodes fp16-exact (kernel
    and gguf-py alike) to the format's crafted constant: IQ3_XXS 0.75 (qs code 71, zero
    scales, d 0.25), IQ4_XS 0.25 (scale byte 33 -> +1, nibble 8 -> kvalue 1, d 0.25),
    Q6_K -16 (ql/qh 0 -> q -32, int8 scales 1, d 0.5 - d LAST per ggml-common.h)."""
    from freetoken.models.gguf.dequant import BLOCK_SHAPE, GGML_IQ3_XXS, GGML_IQ4_XS, GGML_Q6_K

    blk, rb = BLOCK_SHAPE[typ]
    nblk = rows // blk
    t = torch.zeros(E, rows, nblk, rb, dtype=torch.uint8)
    if typ == GGML_IQ3_XXS:
        t[..., 0:2] = torch.tensor([0.25], dtype=torch.float16).view(torch.uint8)
        t[..., 2:66] = 71
    elif typ == GGML_IQ4_XS:
        t[..., 0:2] = torch.tensor([0.25], dtype=torch.float16).view(torch.uint8)
        t[..., 2:4] = 0xAA  # scales_h: every 2-bit high scale = 2
        t[..., 4:8] = 0x11  # scales_l: every nibble = 1 -> scale index 33 -> +1
        t[..., 8:136] = 0x88  # both nibbles = 8 -> kvalues_iq4nl[8] = 1
    elif typ == GGML_Q6_K:
        t[..., 192:208] = 1
        t[..., 208:210] = torch.tensor([0.5], dtype=torch.float16).view(torch.uint8)
    else:
        raise AssertionError(typ)
    return t.view(E, rows, nblk * rb)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_expert_gemm_gguf_per_projection_heterogeneity():
    # The REAL banks file mixes types WITHIN one layer: gate/up IQ3_XXS + down IQ4_XS on
    # 41 of 42 bank layers, gate/up IQ4_XS + down Q6_K on ckpt layer 11. Each projection
    # must dispatch on its OWN ggml type from cache.gguf_types[layer_id] - the uniform
    # per-layer dispatch test cannot catch a one-type-per-layer cache regression.
    from freetoken.layers import swiglu_clamp_and_mul
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.moe.offload_cache import OffloadMoeCache

    layer_types = ((18, 18, 23), (23, 23, 14))  # IQ3_XXS/IQ3_XXS/IQ4_XS, IQ4_XS/IQ4_XS/Q6_K
    consts = {18: 0.75, 23: 0.25, 14: -16.0}
    cache = OffloadMoeCache(
        num_layers=2, num_experts=_NE, cache_size=2 * _NE,
        device=torch.device("cuda"), quant_format="gguf", gguf_types=layer_types,
    )
    layer = OffloadMoELayer.__new__(OffloadMoELayer)  # the branch only needs these attrs
    layer.quant_method = None
    layer.activation = "swiglu_clamp"
    layer.alpha = 1.0
    layer.limit = 10.0

    torch.manual_seed(0)
    # g/u stay under the clamp limit so the epilogue is the plain swiglu of the exact
    # crafted constants; the only kernel-vs-reference deviation left is the q8_1
    # x-quant noise (the same contract as the moe_vec parity tests, rtol 2e-2).
    base = 0.05 * torch.randn(5, _H, dtype=torch.float32, device="cuda")
    bias = torch.linspace(0.015, 0.03, 5, device="cuda").unsqueeze(-1)
    x = (base + bias).contiguous()
    topk_ids = torch.stack(
        [torch.randperm(_NE, device="cuda")[:2].to(torch.int32) for _ in range(5)]
    )
    topk_weights = torch.rand(5, 2, dtype=torch.float32, device="cuda")
    topk_weights = (topk_weights / topk_weights.sum(-1, keepdim=True)).contiguous()

    outs = {}
    for lid, types in enumerate(layer_types):
        views = tuple(
            _craft_hetero_bank(rows, typ, _NE).cuda()
            for rows, typ in ((_EFF, types[0]), (_EFF, types[1]), (_H, types[2]))
        )
        layer.layer_id = lid
        outs[lid] = layer._expert_gemm(
            cache, x, topk_weights, topk_ids, views=views, n=None, alphas=None,
            is_prefill=False,
        )
        assert outs[lid].shape == (5, _H)
        assert torch.isfinite(outs[lid]).all()

    # analytic reference: every bank decodes to its projection's own constant, so the
    # per-slot gate/up are w*S and the down projection collapses to w_d * I * inter
    S = x.sum(-1)
    for lid, types in enumerate(layer_types):
        w_g, w_u, w_d = consts[types[0]], consts[types[1]], consts[types[2]]
        pair = torch.stack((w_g * S, w_u * S), dim=-1)
        inter = swiglu_clamp_and_mul(pair, alpha=1.0, limit=10.0)  # (tokens, 1)
        ref = (topk_weights * (w_d * _EFF) * inter).sum(-1)
        ref = ref.unsqueeze(-1).expand(-1, _H)
        assert torch.allclose(outs[lid], ref, rtol=0.02, atol=1e-2), lid
    assert not torch.allclose(outs[0], outs[1])  # the two layers really dispatched


def test_bank_bytes_estimate_gguf_real_geometry():
    # glm5.3-flash dims (the real file's metadata; the 147 GB gguf is never opened):
    # H=4096, I=2048, E=288, 45-3=42 MoE layers. The estimate pins gate/up at the
    # IQ3_XXS row width and down at the Q6_K width: EXACT for the 3 Q6_K down banks
    # (ckpt layers 11/12/44) and the IQ3_XXS gate/up pairs, overcounting the 39
    # IQ4_XS down banks - and UNDERcounting bank layer 8's IQ4_XS gate/up pair.
    from types import SimpleNamespace

    from freetoken.moe.expert_banks import bank_bytes_estimate

    H, I, E, L = 4096, 2048, 288, 42
    cfg = SimpleNamespace(
        num_moe_layers=L, num_experts=E, hidden_size=H, moe_intermediate_size=I,
        expert_quant="none", moe_weight_format="gguf",
    )
    est = bank_bytes_estimate(cfg)
    gate_up_iq3 = 2 * I * (H // 256) * 98
    gate_up_iq4xs = 2 * I * (H // 256) * 136
    down_q6k = H * (I // 256) * 210
    down_iq4xs = H * (I // 256) * 136
    assert est == L * E * (gate_up_iq3 + down_q6k)
    # the real composition (82 IQ3_XXS + 41 IQ4_XS + 3 Q6_K bank tensors): 41 IQ3_XXS
    # gate/up pairs, 1 IQ4_XS gate/up pair, 39 IQ4_XS downs, 3 Q6_K downs.
    true_bytes = 41 * E * gate_up_iq3 + E * gate_up_iq4xs + 39 * E * down_iq4xs + 3 * E * down_q6k
    assert est == true_bytes + 39 * E * (down_q6k - down_iq4xs) - E * (gate_up_iq4xs - gate_up_iq3)
    assert est > true_bytes  # conservative in aggregate (overcount >> undercount)


def _write_bank_table_gguf(path, layer_types, ne) -> str:
    """GGUF whose tensor table declares routed-expert stacks over a sparse, never-read
    data section: real per-layer types and real geometry (H, I, E) for kilobytes on
    disk instead of the ~134 GB the real table would weigh. GGUFReader builds lazy
    memmap views over the hole and the header-only sizing scan never touches a page;
    the file is not loadable - the sizing scan is its only consumer.
    """
    import math
    import struct

    from gguf.constants import (
        GGML_QUANT_SIZES,
        GGMLQuantizationType,
        GGUF_DEFAULT_ALIGNMENT,
        GGUF_MAGIC,
        GGUF_VERSION,
    )

    hidden, inter, n_exp = ne
    align = GGUF_DEFAULT_ALIGNMENT
    table = bytearray(struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION, 3 * len(layer_types), 0))
    cursor = 0  # next tensor's offset within the data section
    for layer, (t_gate, t_up, t_down) in sorted(layer_types.items()):
        for role, dims, ggml_type in (
            ("gate", (hidden, inter, n_exp), t_gate),
            ("up", (hidden, inter, n_exp), t_up),
            ("down", (inter, hidden, n_exp), t_down),
        ):
            block, type_size = GGML_QUANT_SIZES[GGMLQuantizationType(ggml_type)]
            cursor = -(-cursor // align) * align
            name = f"blk.{layer}.ffn_{role}_exps.weight"
            encoded = name.encode()
            table += struct.pack("<Q", len(encoded)) + encoded
            table += struct.pack("<I", len(dims))
            table += struct.pack(f"<{len(dims)}Q", *dims)
            table += struct.pack("<IQ", ggml_type, cursor)
            cursor += math.prod(dims) * type_size // block
    data_start = -(-len(table) // align) * align
    with open(path, "wb") as f:
        f.write(table)
        f.truncate(data_start + cursor)  # the data section stays a hole
    return str(path)


def test_bank_bytes_estimate_gguf_exact_header_scan(tmp_path):
    # a 3-signature synthetic table (the real file's three (gate, up, down) type
    # mixes) at non-square dims: the estimate must be the signature-weighted true
    # sum read off the header scan, not the conservative per-expert mix
    from types import SimpleNamespace

    from freetoken.moe.expert_banks import bank_bytes_estimate
    from freetoken.models.gguf.dequant import BLOCK_SHAPE

    H, I, E = 512, 256, 4
    sigs = {0: (18, 18, 23), 1: (18, 18, 14), 2: (23, 23, 14)}
    path = _write_bank_table_gguf(tmp_path / "signatures.gguf", sigs, (H, I, E))
    cfg = SimpleNamespace(
        first_k_dense_replace=0, num_layers=3, num_moe_layers=3, num_experts=E,
        hidden_size=H, moe_intermediate_size=I, expert_quant="none",
        moe_weight_format="gguf",
    )

    def stack(rows, row_len, ggml_type):
        block, type_size = BLOCK_SHAPE[ggml_type]
        return rows * (row_len // block) * type_size

    gate_iq3, gate_iq4 = stack(I, H, 18), stack(I, H, 23)
    down_iq4, down_q6k = stack(H, I, 23), stack(H, I, 14)
    est = bank_bytes_estimate(cfg, model_path=str(path))
    assert est == sum(
        E * (2 * gate_iq3 + down_iq4) if sig == (18, 18, 23)
        else E * (2 * gate_iq3 + down_q6k) if sig == (18, 18, 14)
        else E * (2 * gate_iq4 + down_q6k)
        for sig in sigs.values()
    )
    # without the path every layer is priced at the extreme (IQ3_XXS gate/up + Q6_K down)
    assert bank_bytes_estimate(cfg) == 3 * E * (2 * gate_iq3 + down_q6k)
    assert est != bank_bytes_estimate(cfg)


def test_bank_bytes_estimate_gguf_real_distribution_pin(tmp_path):
    # the real GLM-5.3-Flash UD-Q3_K_XL bank table, header-only: 39x (18,18,23) +
    # 2x (18,18,14) + 1x (23,23,14) across 42 MoE layers (ckpt 3..44; the Q6_K
    # downs sit on 11/12/44 and the single IQ4_XS gate/up pair on 11). The expected
    # value is derived here from BLOCK_SHAPE and only then pinned to the measured
    # ground truth, so the constant can never drift silently.
    from types import SimpleNamespace

    from freetoken.moe.expert_banks import bank_bytes_estimate
    from freetoken.models.gguf.dequant import BLOCK_SHAPE

    H, I, E = 4096, 2048, 288
    layer_types = {}
    for layer in range(3, 45):
        if layer == 11:
            layer_types[layer] = (23, 23, 14)
        elif layer in (12, 44):
            layer_types[layer] = (18, 18, 14)
        else:
            layer_types[layer] = (18, 18, 23)
    assert len(layer_types) == 42

    def stack(rows, row_len, ggml_type):
        block, type_size = BLOCK_SHAPE[ggml_type]
        return rows * (row_len // block) * type_size

    gate_iq3, gate_iq4 = stack(I, H, 18), stack(I, H, 23)
    down_iq4, down_q6k = stack(H, I, 23), stack(H, I, 14)
    cfg = SimpleNamespace(
        first_k_dense_replace=3, num_layers=45, num_moe_layers=42, num_experts=E,
        hidden_size=H, moe_intermediate_size=I, expert_quant="none",
        moe_weight_format="gguf",
    )
    path = _write_bank_table_gguf(tmp_path / "glm53-flash-banks.gguf", layer_types, (H, I, E))
    est = bank_bytes_estimate(cfg, model_path=str(path))
    exact = E * (
        39 * (2 * gate_iq3 + down_iq4) + 2 * (2 * gate_iq3 + down_q6k) + (2 * gate_iq4 + down_q6k)
    )
    assert est == exact
    assert exact == 134_404_374_528  # the measured real-file bank bytes
    conservative = bank_bytes_estimate(cfg)
    assert conservative == 42 * E * (2 * gate_iq3 + down_q6k)
    assert conservative == 160_922_861_568  # the old estimate: +24.7 GiB over exact
    assert conservative - est == 39 * E * (down_q6k - down_iq4) - 2 * E * (gate_iq4 - gate_iq3)
    assert conservative > est


def test_bank_bytes_estimate_gguf_writer_fixture_path(tmp_path):
    # the scan must agree with a gguf-py-written file (the writer owns offsets and
    # alignment), and blk.3's stray MTP gate bank must stay out: it is not a trunk
    # layer. A table-only twin of the same stacks must scan to the same number.
    from freetoken.moe.expert_banks import bank_bytes_estimate
    from freetoken.models.gguf.dequant import BLOCK_SHAPE

    path = _write_iter_gguf(tmp_path / "banks.gguf")
    cfg = _bank_cfg()
    block, type_size = BLOCK_SHAPE[8]
    gate_q8 = _EFF * (_H // block) * type_size  # per-expert gate/up stack
    block, type_size = BLOCK_SHAPE[14]
    down_q6k = _H * (_EFF // block) * type_size  # per-expert down stack
    est = bank_bytes_estimate(cfg, model_path=str(path))
    assert est == 2 * _NE * (2 * gate_q8 + down_q6k)
    twin = _write_bank_table_gguf(tmp_path / "twin.gguf", {1: (8, 8, 14), 2: (8, 8, 14)}, (_H, _EFF, _NE))
    assert bank_bytes_estimate(cfg, model_path=twin) == est
    assert est != bank_bytes_estimate(cfg)


def test_bank_bytes_estimate_gguf_path_fallbacks(tmp_path):
    # no path, a non-file path, and a trunk short one bank layer must all keep the
    # conservative per-expert table - the exact scan never undercounts
    from freetoken.moe.expert_banks import bank_bytes_estimate

    cfg = _bank_cfg()
    est = bank_bytes_estimate(cfg)
    assert bank_bytes_estimate(cfg, model_path=None) == est
    assert bank_bytes_estimate(cfg, model_path=str(tmp_path / "absent.gguf")) == est
    assert bank_bytes_estimate(cfg, model_path=str(tmp_path)) == est  # a dir, e.g. FTW
    path = _write_bank_table_gguf(tmp_path / "short.gguf", {0: (18, 18, 23)}, (256, 256, _NE))
    assert bank_bytes_estimate(cfg, model_path=path) == est


def test_gguf_expert_sources_missing_bank_layer_raises(tmp_path):
    # dropping one MoE layer's whole bank group must fail loudly naming the gap, not
    # hand the offload cache a None bank layer to copy from
    from freetoken.models.weight import load_gguf_moe_expert_sources

    tensors = _iter_tensor_set()
    for name in [n for n in tensors if n.startswith("blk.2.ffn_") and n.endswith("_exps.weight")]:
        del tensors[name]
    path = _write_iter_gguf(tmp_path / "partial.gguf", tensors=tensors)
    with pytest.raises(ValueError, match="missing for bank layers"):
        load_gguf_moe_expert_sources(str(path), _bank_cfg())


def test_gguf_expert_sources_first_k_dense_offset_and_dense_guard(tmp_path):
    # the real file's leading_dense_block_count=3: bank index 0 must be CHECKPOINT
    # layer 3 (bank_layer_of contract), and a bank below first_k_dense_replace is a
    # corrupt file that must fail loudly.
    import gguf
    from types import SimpleNamespace

    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.models.weight import load_gguf_moe_expert_sources

    tensors = _iter_tensor_set()
    for prefix in ("blk.1.", "blk.2.", "blk.3."):
        for name in [n for n in tensors if n.startswith(prefix) and n.endswith("_exps.weight")]:
            del tensors[name]
    q8 = gguf.GGMLQuantizationType.Q8_0
    q6k = gguf.GGMLQuantizationType.Q6_K
    tensors["blk.3.ffn_gate_exps.weight"] = (_pack_q8_0(_bank_vals(0)), q8)
    tensors["blk.3.ffn_up_exps.weight"] = (_pack_q8_0(_bank_vals(7)), q8)
    tensors["blk.3.ffn_down_exps.weight"] = (_DOWN_BANK_BYTES, q6k)
    tensors["blk.4.ffn_gate_exps.weight"] = (_pack_q8_0(_bank_vals(0)), q8)
    tensors["blk.4.ffn_up_exps.weight"] = (_pack_q8_0(_bank_vals(7)), q8)
    tensors["blk.4.ffn_down_exps.weight"] = (_DOWN_BANK_BYTES, q6k)
    path = _write_iter_gguf(tmp_path / "offset3.gguf", tensors=tensors)
    cfg = SimpleNamespace(
        first_k_dense_replace=3, num_layers=5, num_moe_layers=2, num_experts=_NE,
        hidden_size=_H, moe_intermediate_size=_EFF, expert_quant="none",
        moe_weight_format="gguf",
    )
    banks, types = load_gguf_moe_expert_sources(str(path), cfg)
    assert types == ((8, 8, 14), (8, 8, 14))
    # bank 0 IS ckpt layer 3: the packed rows are the source tensor verbatim
    src = next(t for t in iter_gguf_tensors(str(path)) if t.name == "blk.3.ffn_gate_exps.weight")
    raw = src.packed().reshape(_NE, _EFF, -1)
    assert torch.equal(banks["gate"][0].reshape(-1), raw.reshape(-1))
    assert torch.equal(banks["gate"][0][0], raw[0])
    assert torch.equal(banks["down"][1][0], next(
        t for t in iter_gguf_tensors(str(path)) if t.name == "blk.4.ffn_down_exps.weight"
    ).packed().reshape(_NE, _H, 210)[0])
    # and a bank under leading_dense_block_count (the unmodified fixture at first=3)
    # dies loudly instead of desyncing the bank count
    with pytest.raises(ValueError, match="on dense layer 1"):
        load_gguf_moe_expert_sources(str(_write_iter_gguf(tmp_path / "orig.gguf")), cfg)


# ---------------------------------------------------------------------------
# Early honest fail-fast for the --moe-cache-auto per-signature floors: the
# pre-load gate (Engine._gguf_auto_floor_gate) must reject an unfundable floor
# set with the same numbers the late _split_moe_cache_budget check computes.
# ---------------------------------------------------------------------------
from types import SimpleNamespace


# Three signature groups over 8 MoE bank layers, mirroring the real GLM-5.3 GGUF
# heterogeneity at toy geometry: 5x (IQ3_XXS, IQ3_XXS, IQ4_XS), 1x (IQ4_XS,
# IQ4_XS, Q6_K), 2x (IQ3_XXS, IQ3_XXS, Q6_K). H=512 (gate/up ne0), I=256 (down
# ne0), num_experts=4, so per-slot bytes are 169,984 / 246,784 / 207,872 and the
# floor set needs ceil(4 * 624,640 / 169,984) = 15 layer-0-width slots.
_TRIPLE_SIGS = [(18, 18, 23)] * 5 + [(23, 23, 14)] + [(18, 18, 14)] * 2
_TRIPLE_GROUP_SLOT_BYTES = [169_984, 246_784, 207_872]


def _write_triple_sig_gguf(path) -> str:
    import gguf

    # reuse the iter fixture's writer: it carries the glm5next.* metadata KV the
    # load path's config parse requires; only the tensor set is ours
    qt = {
        18: gguf.GGMLQuantizationType.IQ3_XXS,
        23: gguf.GGMLQuantizationType.IQ4_XS,
        14: gguf.GGMLQuantizationType.Q6_K,
        8: gguf.GGMLQuantizationType.Q8_0,
    }
    row_bytes = {18: 98, 23: 136, 14: 210}  # packed bytes per 256-element block
    entries = {
        # vocab embedding so the load path's gguf config parse (token_embd sizing) works
        "token_embd.weight": (np.zeros((4, 512 // 32 * 34), np.uint8), qt[8]),
    }
    for layer, (gt, ut, dt) in enumerate(_TRIPLE_SIGS):
        for role, type_id in (("gate", gt), ("up", ut), ("down", dt)):
            if role == "down":
                rows, ne0 = 512 * 4, 256  # rows = ne1 * ne2 (H output rows x E experts)
            else:
                rows, ne0 = 256 * 4, 512   # rows = I output rows x E experts
            raw = np.zeros((rows, ne0 // 256 * row_bytes[type_id]), np.uint8)
            entries[f"blk.{layer}.ffn_{role}_exps.weight"] = (raw, qt[type_id])
    return _write_iter_gguf(path, tensors=entries)


def _gate_engine_and_config(path, kv_reserve_tokens):
    # Engine.__new__ stub (same idiom as test_engine_resolve_auto_moe_cache_size_maps_kwargs)
    # + an auto-plan config over the triple-signature gguf: budget 89,000,000 B,
    # cache_per_page 98,304, per-slot (layer-0 width) 169,984 -> kv_reserve_tokens
    # picks the envelope: 14,080 -> 14 slots, 14,048 -> 15, 14,016 -> 16.
    import torch

    from freetoken.engine.engine import Engine
    from freetoken.kvcache.mha_pool import MHAKVCache
    from freetoken.models.config import KVCacheGroupSpec

    class StubModelConfig:
        num_experts = 4
        num_moe_layers = 8
        first_k_dense_replace = 0
        expert_quant = "none"
        moe_weight_format = "gguf"
        hidden_size = 512
        moe_intermediate_size = 256
        decode_target = "gpu"

        def kv_cache_group_specs(self):
            return [KVCacheGroupSpec(
                name="full", layer_ids=(0, 1, 2), num_kv_heads=8, head_dim=64, sliding_window=None,
            )]

        def linear_attention_group(self):
            return None

    class StubConfig:
        dtype = torch.float16
        page_size = 16
        max_running_req = 4
        hybrid_swa_cache_mode = "auto"
        memory_ratio = 0.9
        moe_cache_auto = True
        use_dummy_weight = False
        model_path = path
        moe_prefill_overlap = True
        moe_strategy = "offload"
        moe_cpu_layers = None
        expert_load = "auto"
        swa_full_tokens_ratio = 0.2
        swa_num_pages_override = None
        model_config = StubModelConfig()

        class tp_info:
            size = 1

    StubConfig.kv_reserve_tokens = kv_reserve_tokens  # class bodies cannot close over locals

    engine = Engine.__new__(Engine)  # bypass __init__/GPU
    engine._baseline_free = 100_000_000
    engine._weights_bytes = 1_000_000
    engine._pool_cls = MHAKVCache
    return engine, StubConfig()


def test_moe_auto_floor_gate_fails_fast_before_bank_load(tmp_path):
    # the 1M-reserve profile: the auto envelope (12 layer-0-width slots) cannot fund
    # the 3 signature groups' 4-slot floors (byte-weighted minimum 14), so the gate
    # must reject at planning time - BEFORE load_expert_banks reads the banks - and
    # name the honest numbers (groups, per-group floors, weighted need, envelope)
    # plus the remedy.
    from freetoken.engine.engine import Engine

    path = _write_triple_sig_gguf(tmp_path / "triple.gguf")
    engine, config = _gate_engine_and_config(path, kv_reserve_tokens=14_080)  # envelope 14
    with pytest.raises(ValueError, match=(
        r"3 signature groups \(5/1/2 layers.*each need 4 slots.*"
        r"at 169984/246784/207872 bytes/slot.*"
        r"byte-weighted minimum of 15 layer-0-width slots vs the planned 14.*"
        r"lower --kv-reserve-tokens or raise --memory-ratio"
    )):
        Engine._gguf_auto_floor_gate(config, None, engine._resolve_auto_moe_cache_size)


def test_moe_auto_floor_gate_boundary_envelope_passes(tmp_path):
    # envelope exactly == the weighted floor need (15 == 15): the early gate must
    # stay silent AND the late split must accept the plan (the floor set is exactly
    # byte-feasible, zero slack).
    from freetoken.engine.engine import Engine

    path = _write_triple_sig_gguf(tmp_path / "triple.gguf")
    engine, config = _gate_engine_and_config(path, kv_reserve_tokens=14_048)  # envelope 15
    Engine._gguf_auto_floor_gate(config, None, engine._resolve_auto_moe_cache_size)  # no raise

    ns = SimpleNamespace(sources=_bank_like_sources(path))
    groups = Engine._group_bank_layers(ns, 8)
    sizes = Engine._split_moe_cache_budget(ns, groups, 15, 4, byte_cap_slots=15)
    assert all(s >= 4 for s in sizes)
    assert sum(s * b for s, b in zip(sizes, _TRIPLE_GROUP_SLOT_BYTES)) <= 15 * _TRIPLE_GROUP_SLOT_BYTES[0]


def test_moe_auto_floor_gate_fitting_profile_unaffected(tmp_path):
    # the 786k-reserve profile analog: envelope 16 funds the 15-slot floor need with
    # one slot of headroom - the gate stays silent and the downstream split resolves
    # the very numbers the late check produces (dominant group shrunk just above its
    # floor, minority groups pinned at it), with no new exceptions.
    from freetoken.engine.cache_budget import expert_bytes_per_slot
    from freetoken.engine.engine import Engine
    from freetoken.moe.expert_banks import gguf_signature_groups

    path = _write_triple_sig_gguf(tmp_path / "triple.gguf")
    engine, config = _gate_engine_and_config(path, kv_reserve_tokens=14_016)  # envelope 16
    Engine._gguf_auto_floor_gate(config, None, engine._resolve_auto_moe_cache_size)  # no raise

    # exactness bridge: the header-only scan's grouping and per-group slot widths
    # match what _group_bank_layers / expert_bytes_per_slot see on the loaded-bank
    # shapes (uint8 (E, rows_per_expert, rb), the loader's layout)
    scan_groups, scan_slots = gguf_signature_groups(str(path), config.model_config)
    ns = SimpleNamespace(sources=_bank_like_sources(path))
    groups = Engine._group_bank_layers(ns, 8)
    assert scan_groups == groups == [[0, 1, 2, 3, 4], [5], [6, 7]]
    assert scan_slots == [
        expert_bytes_per_slot({n: [ns.sources[n][l] for l in m] for n in ns.sources}) for m in groups
    ]

    sizes = Engine._split_moe_cache_budget(ns, groups, 16, 4, byte_cap_slots=16)
    assert sizes == [5, 4, 4]


def test_gguf_signature_groups_match_loaded_banks(tmp_path):
    # exactness bridge on a file the full load path accepts (metadata intact): the
    # header-only scan's grouping and per-group slot widths equal what
    # _group_bank_layers and expert_bytes_per_slot see on the materialized banks
    from freetoken.engine.cache_budget import expert_bytes_per_slot
    from freetoken.engine.engine import Engine
    from freetoken.models.weight import load_gguf_moe_expert_sources
    from freetoken.moe.expert_banks import gguf_signature_groups

    path = _write_iter_gguf(tmp_path / "banks.gguf")
    cfg = _bank_cfg()
    scan_groups, scan_slots = gguf_signature_groups(str(path), cfg)
    banks, _ = load_gguf_moe_expert_sources(str(path), cfg)
    ns = SimpleNamespace(sources=banks)
    groups = Engine._group_bank_layers(ns, cfg.num_moe_layers)
    assert scan_groups == groups == [[0, 1]]
    assert scan_slots == [
        expert_bytes_per_slot({n: [banks[n][l] for l in m] for n in banks}) for m in groups
    ]


def _bank_like_sources(path, num_experts=4):
    # the uint8 (E, rows_per_expert, rb) shapes load_gguf_expert_sources materializes
    import torch

    from freetoken.models.gguf.reader import iter_gguf_tensors

    per = {"gate": [], "up": [], "down": []}
    for t in iter_gguf_tensors(str(path)):
        for suffix, role in (
            ("ffn_gate_exps.weight", "gate"), ("ffn_up_exps.weight", "up"),
            ("ffn_down_exps.weight", "down"),
        ):
            if t.name.endswith(suffix):
                per[role].append(
                    torch.zeros(num_experts, t.rows // num_experts, t.row_bytes, dtype=torch.uint8)
                )
                break
    return per


def test_init_offload_runs_floor_gate_before_bank_load(tmp_path, monkeypatch):
    # pins the gate's POSITION inside _init_offload_moe_cache: an unfundable floor
    # set must raise before load_expert_banks reads the banks, not after - a
    # reordering that moved the call site must fail here
    import freetoken.engine.engine as engine_mod

    path = _write_triple_sig_gguf(tmp_path / "triple.gguf")
    engine, config = _gate_engine_and_config(path, kv_reserve_tokens=14_080)  # envelope 14 < need 15
    engine.model = object()  # shared_offload_method is patched out below
    engine._host_tables_bytes = 0
    loaded = {"called": False}

    def _sentinel(*args, **kwargs):
        loaded["called"] = True
        raise AssertionError("load_expert_banks ran before the floor gate")

    monkeypatch.setattr(engine_mod, "shared_offload_method", lambda model: None)
    monkeypatch.setattr(engine_mod, "load_expert_banks", _sentinel)
    with pytest.raises(ValueError, match="byte-weighted minimum of 15"):
        engine._init_offload_moe_cache(config)
    assert loaded["called"] is False


def test_moe_auto_floor_gate_silent_for_non_gguf(tmp_path):
    # the gate only answers for the gguf provider on a bare .gguf: a non-gguf expert
    # format must pass silently even over a .gguf path (killer envelope 14 < need 15),
    # and a gguf format must pass silently over a non-.gguf path (FTW-like layout)
    from freetoken.engine.engine import Engine

    path = _write_triple_sig_gguf(tmp_path / "triple.gguf")
    engine, config = _gate_engine_and_config(path, kv_reserve_tokens=14_080)
    config.model_config.moe_weight_format = None  # non-gguf expert path (marlin/awq-style)
    Engine._gguf_auto_floor_gate(config, None, engine._resolve_auto_moe_cache_size)  # no raise

    ftw_engine, ftw_config = _gate_engine_and_config(str(tmp_path / "ckpt"), kv_reserve_tokens=14_080)
    Engine._gguf_auto_floor_gate(ftw_config, None, ftw_engine._resolve_auto_moe_cache_size)  # no raise
