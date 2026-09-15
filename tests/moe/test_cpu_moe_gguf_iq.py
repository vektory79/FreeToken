"""CPU MoE executor -- GGUF K-quant expert banks (IQ3_XXS / IQ4_XS / Q6_K).

The CPU GEMV (the gguf section of csrc/cpu_moe/cpu_moe_ext.cpp) dequantizes the
packed 256-wide blocks on the fly, reading the same pinned per-role banks the GPU
offload path streams. Two dot tiers: the scalar fp32-LUT kernels over bf16
activations (correctness reference + fallback) and, by default (AVX2+FMA CPUs,
FREETOKEN_GGUF_DOT_TIER=scalar|avx2 overrides), the verbatim ggml AVX2 integer
kernels (W4A8-K: activations q8_K-quantized once per row and reused across the
expert dots -- the same one-shot prepare the nvfp4/q4_0 W4A8 paths use). Parity
reference is the venv gguf-py ``dequantize`` (fp32; proven vs the CUDA kernels in
tests/kernels/test_gguf_quant.py), with the tier's activation prep mirrored via
_q8_k_roundtrip so reference and kernel consume the same integer activations.
The accepted dequant contract is the per-element dfloat bound
``8 * 2^-11 * factor * |d| + 1e-3`` (factor = _DEQUANT_FACTOR per format); the
parity tests propagate it through the GEMV and additionally carry a tight rel
check, which is the real decode-bug discriminator (the CPU chain is fp32, far
inside the bound).

Covers the per-projection requirement: the three projections of one layer may have
DIFFERENT types (the real glm5next file packs (18, 18, 23) / (18, 18, 14) /
(23, 23, 14) across its layers).
"""

from __future__ import annotations

from types import SimpleNamespace

import gguf
import numpy as np
import pytest
import torch

from freetoken.models.gguf.dequant import BLOCK_SHAPE, GGML_IQ3_XXS, GGML_IQ4_XS, GGML_Q6_K

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# max |value| per element as a multiple of the block's fp16 d (from the gguf-py
# dequant formulas); mirrors tests/kernels/test_gguf_quant.py::_DEQUANT_FACTOR for
# the three formats the CPU GEMV serves.
_DEQUANT_FACTOR = {
    GGML_IQ3_XXS: 8.0 * 118.0,
    GGML_IQ4_XS: 32.0 * 127.0,
    GGML_Q6_K: 127.0 * 31.0,
}

_TYPE_NAMES = {GGML_IQ3_XXS: "iq3_xxs", GGML_IQ4_XS: "iq4_xs", GGML_Q6_K: "q6_k"}

_ROLE_SEED = {"gate": 1, "up": 2, "down": 3}


# ------------------------------- pack helpers -------------------------------
# Byte-exact block layouts from the vendored ggml-common.h (block_q6_K :104,
# block_iq3_xxs :137, block_iq4_xs :181). fp16 scale fields are constructed
# (not random bytes), mirroring _FP16_SCALE_FIELDS in tests/kernels/test_gguf_quant.py.


def _pack_q6_k(ql: torch.Tensor, qh: torch.Tensor, scales: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """[.., 128] ql, [.., 64] qh, [.., 16] int8 scales, [..] fp16 d -> [.., 210].
    block_q6_K keeps d LAST (bytes 208:210)."""
    d_bytes = d.to(torch.float16).view(torch.uint8).reshape(*d.shape, 2)  # [.., nb, 2] LE
    return torch.cat(
        [ql.to(torch.uint8), qh.to(torch.uint8), scales.to(torch.int8).view(torch.uint8), d_bytes],
        dim=-1,
    )


def _pack_iq4_xs(qs: torch.Tensor, s6: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """[.., 128] code bytes, [.., 8] 6-bit scales (0..63), [..] fp16 d -> [.., 136].
    block_iq4_xs: half d FIRST, uint16 scales_h (2 bits per 32-block), uint8
    scales_l[4] (one nibble per 32-block), qs[128]."""
    s6i = s6.to(torch.int64)
    ib = torch.arange(8)
    low = ((s6i & 0xF) << (4 * (ib % 2))).to(torch.uint8)  # [.., 8]
    scales_l = low[..., 0::2] | low[..., 1::2]  # [.., 4]
    sh = torch.zeros(s6i.shape[:-1], dtype=torch.int64)  # one uint16 scales_h per block
    for i in range(8):
        sh = sh | ((s6i[..., i] >> 4) << (2 * i))
    sh_bytes = torch.stack([sh & 0xFF, (sh >> 8) & 0xFF], dim=-1).to(torch.uint8)
    d_bytes = d.to(torch.float16).view(torch.uint8).reshape(*d.shape, 2)  # [.., nb, 2] LE
    return torch.cat([d_bytes, sh_bytes, scales_l, qs.to(torch.uint8)], dim=-1)


def _pack_iq3_xxs(
    grid_idx: torch.Tensor, sign_idx: torch.Tensor, scale_nib: torch.Tensor, d: torch.Tensor
) -> torch.Tensor:
    """[.., 64] grid indices (8 per 32-block), [.., 32] sign indices (4 per 32-block,
    0..127), [.., 8] sub-scale nibbles, [..] fp16 d -> [.., 98]. block_iq3_xxs: half
    d FIRST, then 96 bytes = 64 grid bytes + 8 uint32 words (bits 28-31: sub-scale
    nibble, bits 0-27: the four 7-bit sign indices)."""
    lead = grid_idx.shape[:-1]
    signs = sign_idx.reshape(*lead, 8, 4).to(torch.int64)
    shift = torch.tensor([0, 7, 14, 21], dtype=torch.int64)
    # disjoint 7-bit sign fields: OR over il == sum; the sub-scale nibble joins once
    sign_bits = (signs << shift).sum(-1)  # [.., 8]
    aux = (scale_nib.to(torch.int64) << 28) | sign_bits  # [.., 8]
    aux_bytes = torch.stack([(aux >> (8 * k)) & 0xFF for k in range(4)], dim=-1).to(torch.uint8)
    d_bytes = d.to(torch.float16).view(torch.uint8).reshape(*d.shape, 2)  # [.., nb, 2] LE
    return torch.cat([d_bytes, grid_idx.to(torch.uint8), aux_bytes.reshape(*lead, 32)], dim=-1)


# uniform = analytic fixture (every dequantized weight a known constant, cross-
# checked against gguf-py below); random = bounded fp16 scales + full-range codes.
def _role_bank(uniform: bool, E: int, rows: int, ne0: int, qtype: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    nb = ne0 // 256
    if qtype == GGML_Q6_K:
        if uniform:
            # ql=qh=0xFF -> q = 63 (top code); sc=1; d=0.25 -> w = 0.25*1*31 = 7.75
            packed = _pack_q6_k(
                torch.full((E, rows, nb, 128), 0xFF, dtype=torch.uint8),
                torch.full((E, rows, nb, 64), 0xFF, dtype=torch.uint8),
                torch.ones(E, rows, nb, 16, dtype=torch.uint8),
                torch.full((E, rows, nb), 0.25, dtype=torch.float16),
            )
        else:
            packed = _pack_q6_k(
                torch.randint(0, 256, (E, rows, nb, 128), dtype=torch.uint8, generator=gen),
                torch.randint(0, 256, (E, rows, nb, 64), dtype=torch.uint8, generator=gen),
                torch.randint(-8, 9, (E, rows, nb, 16), dtype=torch.int8, generator=gen),
                (0.25 + 1.75 * torch.rand(E, rows, nb, generator=gen)).to(torch.float16),
            )
    elif qtype == GGML_IQ4_XS:
        if uniform:
            # nibbles 8 -> kvalues_iq4nl[8] = 1; s6=36 -> scale 36-32 = 4;
            # d=0.25 -> w = 0.25*4*1 = 1
            packed = _pack_iq4_xs(
                torch.full((E, rows, nb, 128), 0x88, dtype=torch.uint8),
                torch.full((E, rows, nb, 8), 36, dtype=torch.uint8),
                torch.full((E, rows, nb), 0.25, dtype=torch.float16),
            )
        else:
            packed = _pack_iq4_xs(
                torch.randint(0, 256, (E, rows, nb, 128), dtype=torch.uint8, generator=gen),
                torch.randint(24, 41, (E, rows, nb, 8), dtype=torch.uint8, generator=gen),
                (0.25 + 1.75 * torch.rand(E, rows, nb, generator=gen)).to(torch.float16),
            )
    elif qtype == GGML_IQ3_XXS:
        if uniform:
            # grid index 0 -> bytes {4, 4, 4, 4}; sign indices 0 -> all positive;
            # nib 2 -> mult (0.5 + 2) * 0.5 = 1.25; d=0.4 -> w = 0.4*1.25*4 = 2
            packed = _pack_iq3_xxs(
                torch.zeros(E, rows, nb, 64, dtype=torch.uint8),
                torch.zeros(E, rows, nb, 32, dtype=torch.uint8),
                torch.full((E, rows, nb, 8), 2, dtype=torch.uint8),
                torch.full((E, rows, nb), 0.4, dtype=torch.float16),
            )
        else:
            packed = _pack_iq3_xxs(
                torch.randint(0, 256, (E, rows, nb, 64), dtype=torch.uint8, generator=gen),
                torch.randint(0, 128, (E, rows, nb, 32), dtype=torch.uint8, generator=gen),
                torch.randint(0, 4, (E, rows, nb, 8), dtype=torch.uint8, generator=gen),
                (0.25 + 1.75 * torch.rand(E, rows, nb, generator=gen)).to(torch.float16),
            )
    else:  # pragma: no cover - guarded by parametrization
        raise ValueError(f"unsupported gguf type {qtype}")
    return packed.reshape(E, rows, nb * packed.shape[-1])


def _dequant_role(packed: torch.Tensor, qtype: int, ne0: int) -> torch.Tensor:
    """[E, rows, nb*rb] packed -> [E, rows, ne0] fp32 via the venv gguf-py reference."""
    raw = packed.contiguous().reshape(-1).numpy()
    flat = gguf.dequantize(raw, gguf.GGMLQuantizationType(qtype)).astype(np.float32)
    return torch.from_numpy(flat.copy()).reshape(packed.shape[0], packed.shape[1], ne0)


def _q8_k_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """Mirror of the W4A8-K tier's activation prep: quantize every 256-element
    block to q8_K and dequantize back (quantize_row_q8_K_ref semantics, ported
    from ggml-quants.c:2768): SIGNED amax, iscale = -127/max (negative on
    purpose; d = 1/iscale absorbs the sign), upper clamp at 127 only,
    round-half-to-even (torch.round == ggml's nearest_int mantissa trick).
    fp32 in / fp32 out; the integer kernels consume exactly these qs."""
    xf = x.float().reshape(-1, 256)
    amax = xf.abs().amax(dim=-1)
    signed_max = xf[torch.arange(xf.shape[0]), xf.abs().argmax(dim=-1)]
    iscale = torch.where(amax > 0, -127.0 / signed_max, torch.zeros_like(amax))
    qs = torch.minimum(torch.full_like(xf, 127.0), torch.round(iscale.unsqueeze(-1) * xf))
    deq = qs * (1.0 / iscale).unsqueeze(-1)
    deq[amax == 0] = 0.0  # the C++ zero-block branch: d = 0, qs = 0
    return deq.reshape(x.shape)


def _gguf_i8_active(ex) -> bool:
    """Whether the executor picked the W4A8-K (avx2) integer dot tier."""
    return "avx2-w4a8k" in getattr(ex, "isa", "")


def _host_has_avx2_fma() -> bool:
    """Host-side mirror of the C++ gate (__builtin_cpu_supports("avx2") &&
    ("fma")): the W4A8-K tier engages only when BOTH are present, whatever the
    FREETOKEN_GGUF_DOT_TIER override asks for (pick_gguf_dot_tier caps down)."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.split(":")[0].strip() == "flags":
                    flags = set(line.split(":")[1].split())
                    return "avx2" in flags and "fma" in flags
    except OSError:
        pass
    return False


_HOST_AVX2 = _host_has_avx2_fma()


def _make_gguf_cache(types, L: int, E: int, H: int, I: int, seed: int = 0, uniform: bool = False):
    """Pinned per-role gguf banks + the cache stub the executor reads.

    Per-role geometry (glm5next): gate/up are I rows packed over ne0 = H; down is H
    rows packed over ne0 = I -- the asymmetry square fixtures mask."""
    from freetoken.kernel.pinned import alloc_pinned_tensor

    gate_t, up_t, down_t = types
    banks = {}
    for role, rows, ne0, qtype in (
        ("gate", I, H, gate_t),
        ("up", I, H, up_t),
        ("down", H, I, down_t),
    ):
        per_layer = []
        for layer in range(L):
            packed = _role_bank(uniform, E, rows, ne0, qtype, seed * 1000 + 17 * layer + _ROLE_SEED[role])
            pinned = alloc_pinned_tensor(*packed.shape, dtype=torch.uint8)
            pinned.copy_(packed)
            per_layer.append(pinned)
        banks[role] = per_layer
    return SimpleNamespace(
        quant_format="gguf",
        gguf_types=[tuple(types)] * L,
        bank_sources=banks,
        num_layers=L,
        num_experts=E,
        decode_target="cpu",
        cpu_executor=None,
    )


def _reference_decode(banks, types, layer, hidden, w, ids, top_k, q8k_acts=False):
    """gguf-py-dequant fp32 reference mirroring the executor pipeline:
    silu(gate@x) * up -> bf16 intermediate -> down@ -> router weight -> bf16.
    Returns (y [bs, H] f32, extras for the dfloat bound).

    q8k_acts mirrors the W4A8-K (avx2) tier's activation prep: x and the bf16
    intermediate are q8_K-quantized+dequantized (_q8_k_roundtrip) before each
    dot, so reference and kernel consume the SAME integer activations and the
    residual is weight-dequant + accumulation error only (harness-grade parity)."""
    gate_t, up_t, down_t = types
    gate_w = _dequant_role(banks["gate"][layer], gate_t, hidden.shape[-1])  # [E, I, H]
    up_w = _dequant_role(banks["up"][layer], up_t, hidden.shape[-1])  # [E, I, H]
    down_w = _dequant_role(banks["down"][layer], down_t, banks["gate"][layer].shape[1])  # [E, H, I]
    x = hidden.float()
    if q8k_acts:
        x = _q8_k_roundtrip(x)
    bs = x.shape[0]
    y = torch.empty(bs, x.shape[1], dtype=torch.float32)
    inter_all = torch.zeros(bs, top_k, down_w.shape[-1])
    silu_mag = torch.zeros(bs, top_k, down_w.shape[-1])  # |silu'(g)|*|u| + |silu(g)|
    for t in range(bs):
        acc = torch.zeros(x.shape[1], dtype=torch.float32)
        for k in range(top_k):
            e = int(ids[t, k])
            if e < 0:
                continue
            gate = gate_w[e] @ x[t]
            up = up_w[e] @ x[t]
            silu_g = torch.nn.functional.silu(gate)
            inter = (silu_g * up).to(torch.bfloat16).float()
            if q8k_acts:
                inter = _q8_k_roundtrip(inter)  # the down leg's per-route q8_K row
            acc = acc + float(w[t, k]) * (down_w[e] @ inter)
            inter_all[t, k] = inter
            sig = torch.sigmoid(gate)
            dsilu = sig * (1.0 + gate * (1.0 - sig))
            silu_mag[t, k] = dsilu.abs() * up.abs() + silu_g.abs()
        y[t] = acc.to(torch.bfloat16).float()
    return y, {"inter": inter_all, "silu_mag": silu_mag, "down_w": down_w}


def _dfloat_bound(ref, extras, w, ids, hidden, factor_gu, factor_dn, dmax=2.0):
    """Per-element output bound: the dequant contract 8*2^-11*factor*|d| + 1e-3
    (per weight element) propagated through gate/up (input x), the silu(g)*u
    product, and the down GEMV, plus bf16-rounding slack for the intermediate
    and the output."""
    eps_gu = 8 * 2**-11 * factor_gu * dmax + 1e-3
    eps_dn = 8 * 2**-11 * factor_dn * dmax + 1e-3
    x_l1 = hidden.float().abs().sum(-1)  # [bs]
    inter = extras["inter"]  # [bs, top_k, I]
    silu_mag = extras["silu_mag"]  # [bs, top_k, I]
    down_w = extras["down_w"].abs()  # [E, H, I]
    bound = torch.zeros_like(ref)
    for t in range(ref.shape[0]):
        for k in range(w.shape[1]):
            e = int(ids[t, k])
            if e < 0:
                continue
            wt = float(w[t, k])
            # gate/up dequant error -> inter error: |dsilu(g)*du| + |silu(g)*dg|,
            # plus one bf16 ulp where the two g rows round apart
            inter_err = eps_gu * x_l1[t] * silu_mag[t, k] + 2.0**-8 * inter[t, k].abs()
            row = down_w[e]  # [H, I]
            bound[t] += wt * (eps_dn * inter[t, k].abs().sum() + inter_err @ row.t())
    return bound + 2.0**-8 * ref.abs() + 1e-3


def _run_executor(cache, top_k, bs, H, dev, seed, layer=0, x_scale=0.5, ids=None):
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    ex = CpuMoeExecutor(
        cache,
        top_k=top_k,
        activation="silu",
        apply_router_weight_on_input=False,
        num_threads=2,
        max_tokens=bs,
        device=dev,
    )
    gen = torch.Generator().manual_seed(seed)
    hidden = (torch.randn(bs, H, generator=gen) * x_scale).to(torch.bfloat16)
    if ids is None:
        ids = torch.stack(
            [torch.randperm(cache.num_experts, generator=gen)[:top_k] for _ in range(bs)]
        ).to(torch.int32)
    w = torch.rand(bs, top_k, generator=gen)
    cpu_out = ex.decode(layer, hidden.to(dev), w.to(dev), ids.to(dev)).float()
    torch.cuda.synchronize()
    return ex, cpu_out.cpu(), hidden, w, ids


@pytest.mark.parametrize(
    "qtype", (GGML_IQ3_XXS, GGML_IQ4_XS, GGML_Q6_K), ids=["iq3_xxs", "iq4_xs", "q6_k"]
)
def test_cpu_moe_gguf_parity_vs_gguf_py(qtype):
    """CPU GEMV vs the gguf-py reference for one uniform-type layer, inside the
    propagated dfloat bound (plus a tight rel check as the bug discriminator)."""
    bs, top_k, L, E, H, I = 3, 3, 2, 8, 512, 256
    dev = torch.device("cuda")
    types = (qtype, qtype, qtype)
    cache = _make_gguf_cache(types, L, E, H, I, seed=100 + qtype)
    ex, cpu_out, hidden, w, ids = _run_executor(cache, top_k, bs, H, dev, seed=400 + qtype, layer=1)
    assert ex.quant_format == _TYPE_NAMES[qtype]

    # The W4A8-K (avx2) tier quantizes the activations, so the reference mirrors
    # that prep; the scalar tier keeps the raw bf16 activations. Either way the
    # reference and the kernel see the same inputs.
    ref, extras = _reference_decode(
        cache.bank_sources, types, 1, hidden, w, ids, top_k, q8k_acts=_gguf_i8_active(ex)
    )
    rel = (cpu_out - ref).abs().max() / (ref.abs().max() + 1e-6)
    assert rel < 2e-2, f"{_TYPE_NAMES[qtype]} rel err {rel.item()}"

    bound = _dfloat_bound(ref, extras, w, ids, hidden, _DEQUANT_FACTOR[qtype], _DEQUANT_FACTOR[qtype])
    dev_abs = (cpu_out - ref).abs()
    assert (dev_abs <= bound).all(), (
        f"{_TYPE_NAMES[qtype]}: max abs dev {dev_abs.max():.6g} exceeds the propagated "
        f"dfloat bound {bound.max():.6g}; {int((dev_abs > bound).sum())} elements outside"
    )


@pytest.mark.parametrize(
    "qtype", (GGML_IQ3_XXS, GGML_IQ4_XS, GGML_Q6_K), ids=["iq3_xxs", "iq4_xs", "q6_k"]
)
def test_cpu_moe_gguf_uniform_analytic(qtype):
    """Analytic uniform banks: every dequantized weight is a known positive constant,
    so the decode output is exact up to fp32 accumulation + bf16 rounding. Any
    block- or nibble-ordering bug shifts the output loudly. Also cross-checks the
    crafted bytes against the gguf-py reference (uniformity + value)."""
    bs, top_k, L, E, H, I = 1, 1, 1, 2, 512, 256
    dev = torch.device("cuda")
    types = (qtype, qtype, qtype)
    cache = _make_gguf_cache(types, L, E, H, I, seed=3, uniform=True)

    gate_w = _dequant_role(cache.bank_sources["gate"][0], qtype, H)
    value = gate_w[0, 0, 0].item()
    assert (gate_w == value).all(), f"fixture not uniform for {_TYPE_NAMES[qtype]}"

    ex, cpu_out, hidden, w, ids = _run_executor(cache, top_k, bs, H, dev, seed=9, x_scale=0.01)
    x = hidden.float()[0]
    if _gguf_i8_active(ex):
        # W4A8-K tier: the input row is q8_K-quantized once per token; mirror it in
        # the analytic gate. (The intermediate's own roundtrip is EXACT here: every
        # element is the same bf16 scalar, so each block's signed amax IS that
        # element and quantizes back bit-exactly.)
        x = _q8_k_roundtrip(x)
    gate = value * x.sum()  # uniform weights: every gate row dots to the same scalar
    inter = (torch.nn.functional.silu(gate) * gate).to(torch.bfloat16).float()
    # down_h = value * sum_i(inter_i); every intermediate element is the SAME bf16
    # scalar (uniform gate/up), so the sum over I is I * inter
    want = (float(w[0, 0]) * value * inter * I).to(torch.bfloat16).float()

    rel = (cpu_out[0] - want).abs().max() / (want.abs().max() + 1e-3)
    assert rel < 5e-3, f"{_TYPE_NAMES[qtype]} analytic rel err {rel.item()} (w={value})"


@pytest.mark.skipif(
    not _HOST_AVX2,
    reason="the avx2 W4A8-K tier engages only on AVX2+FMA hosts (the tier override "
    "caps down to scalar elsewhere, so the A/B would compare scalar vs scalar)",
)
@pytest.mark.parametrize(
    "qtype", (GGML_IQ3_XXS, GGML_IQ4_XS, GGML_Q6_K), ids=["iq3_xxs", "iq4_xs", "q6_k"]
)
def test_cpu_moe_gguf_i8_tier_ab(qtype, monkeypatch):
    """Tier A/B on the SAME fixture (identical banks + inputs, only the tier env
    differs): FREETOKEN_GGUF_DOT_TIER=avx2 (the verbatim ggml W4A8-K kernels) vs
    scalar (the bf16 fp32-LUT reference tier).

    Acceptance mirrors the ggml microbench harness: against the q8_K-MIRRORED fp32
    reference (same integer activations on both sides) the avx2 tier must sit at
    harness-grade parity inside the propagated dfloat bound; the scalar tier must
    keep its pre-existing parity vs the raw-bf16 reference; and the two tiers may
    differ only by the activation-quantization error the W4A8-K tier introduces
    (triangle bound over the two references). The rel bar is 3e-3, not the
    harness's 1e-3: the harness fed IDENTICAL fp32 activations to both kernels,
    while this pipeline re-quantizes the bf16 intermediate for the down leg, where
    the C-vs-torch expf/silu ulp drift flips a bf16 rounding and q6_K's single
    down block re-derives its amax (measured 1.3e-3 worst case, q6_K). The ISA
    discriminator value is unchanged: any sign/LUT/scale bug in the ported
    kernels sits at O(1) relative, two orders above the bar.

    Skipped on hosts without AVX2+FMA: there the tier override caps down to
    scalar (pick_isa's cap-down semantics) and the A/B would compare scalar
    against scalar."""
    bs, top_k, L, E, H, I = 3, 3, 2, 8, 512, 256
    dev = torch.device("cuda")
    types = (qtype, qtype, qtype)

    monkeypatch.setenv("FREETOKEN_GGUF_DOT_TIER", "avx2")
    cache = _make_gguf_cache(types, L, E, H, I, seed=200 + qtype)
    ex_avx, out_avx, hidden, w, ids = _run_executor(
        cache, top_k, bs, H, dev, seed=500 + qtype, layer=1
    )
    assert "avx2-w4a8k" in ex_avx.isa, ex_avx.isa

    # Cap-down invariant: the selected tier never exceeds CPU support. Auto must
    # pick the W4A8-K tag exactly when the host has AVX2+FMA (asserted via the
    # isa tag), so the forced-avx2 run above engages through the cap on this
    # host -- and on a non-AVX2 host the override would cap to scalar instead
    # of SIGILLing on the ggml kernels.
    monkeypatch.delenv("FREETOKEN_GGUF_DOT_TIER", raising=False)
    ex_auto, _, _, _, _ = _run_executor(
        _make_gguf_cache(types, L, E, H, I, seed=200 + qtype),
        top_k, bs, H, dev, seed=500 + qtype, layer=1,
    )
    assert ("avx2-w4a8k" in ex_auto.isa) == _HOST_AVX2, ex_auto.isa

    monkeypatch.setenv("FREETOKEN_GGUF_DOT_TIER", "scalar")
    ex_sc, out_sc, _, _, _ = _run_executor(
        _make_gguf_cache(types, L, E, H, I, seed=200 + qtype),
        top_k, bs, H, dev, seed=500 + qtype, layer=1,
    )
    assert "avx2-w4a8k" not in ex_sc.isa, ex_sc.isa

    ref_q8, extras_q8 = _reference_decode(
        cache.bank_sources, types, 1, hidden, w, ids, top_k, q8k_acts=True
    )
    ref_bf16, extras_bf16 = _reference_decode(cache.bank_sources, types, 1, hidden, w, ids, top_k)

    rel_q8 = (out_avx - ref_q8).abs().max() / (ref_q8.abs().max() + 1e-6)
    assert rel_q8 < 3e-3, (
        f"{_TYPE_NAMES[qtype]} avx2 tier rel err vs the q8_K-mirrored reference "
        f"{rel_q8.item()} (harness discriminator)"
    )
    bound_q8 = _dfloat_bound(
        ref_q8, extras_q8, w, ids, hidden, _DEQUANT_FACTOR[qtype], _DEQUANT_FACTOR[qtype]
    )
    dev_q8 = (out_avx - ref_q8).abs()
    assert (dev_q8 <= bound_q8).all(), (
        f"{_TYPE_NAMES[qtype]} avx2 tier: max abs dev {dev_q8.max():.6g} exceeds the "
        f"propagated dfloat bound {bound_q8.max():.6g}"
    )

    rel_sc = (out_sc - ref_bf16).abs().max() / (ref_bf16.abs().max() + 1e-6)
    assert rel_sc < 2e-2, f"{_TYPE_NAMES[qtype]} scalar tier rel err {rel_sc.item()}"

    # Tier-to-tier difference == the q8_K activation quantization the W4A8-K tier
    # introduces (plus each tier's own tolerated error): triangle over the two
    # references. A tier bug shows up far outside this envelope.
    act_gap = (ref_q8 - ref_bf16).abs()
    bound_sc = _dfloat_bound(
        ref_bf16, extras_bf16, w, ids, hidden, _DEQUANT_FACTOR[qtype], _DEQUANT_FACTOR[qtype]
    )
    d_tiers = (out_avx - out_sc).abs()
    envelope = bound_q8 + bound_sc + act_gap + 1e-3
    assert (d_tiers <= envelope).all(), (
        f"{_TYPE_NAMES[qtype]} tier A/B: max |avx2 - scalar| {d_tiers.max():.6g} exceeds "
        f"the quantization envelope {envelope.max():.6g}"
    )


# The three real glm5next per-projection signatures (gate, up, down): the three
# projections may differ within one layer; the fixtures support all these types.
_GLM5NEXT_SIGNATURES = (
    (GGML_IQ3_XXS, GGML_IQ3_XXS, GGML_IQ4_XS),
    (GGML_IQ3_XXS, GGML_IQ3_XXS, GGML_Q6_K),
    (GGML_IQ4_XS, GGML_IQ4_XS, GGML_Q6_K),
)


@pytest.mark.parametrize(
    "types",
    _GLM5NEXT_SIGNATURES,
    ids=["-".join(_TYPE_NAMES[t] for t in sig) for sig in _GLM5NEXT_SIGNATURES],
)
def test_cpu_moe_gguf_per_projection_formats(types):
    """The real glm5next per-projection signatures: the three projections of one
    layer take DIFFERENT formats and the geometry is asymmetric (gate/up rows = I
    over K = H; down rows = H over K = I); ids may carry -1 (hybrid GPU-routes),
    which the C++ skips."""
    bs, top_k, L, E, H, I = 2, 3, 2, 8, 512, 256
    dev = torch.device("cuda")
    cache = _make_gguf_cache(types, L, E, H, I, seed=7)

    from freetoken.moe.cpu_executor import CpuMoeExecutor

    ex = CpuMoeExecutor(
        cache,
        top_k=top_k,
        activation="silu",
        apply_router_weight_on_input=False,
        num_threads=2,
        max_tokens=bs,
        device=dev,
    )
    assert ex.quant_format == _TYPE_NAMES[types[0]]
    assert ex.fmt_up == _TYPE_NAMES[types[1]] and ex.fmt_down == _TYPE_NAMES[types[2]]

    gen = torch.Generator().manual_seed(11)
    hidden = (torch.randn(bs, H, generator=gen) * 0.5).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(E, generator=gen)[:top_k] for _ in range(bs)]).to(torch.int32)
    ids[0, 0] = -1  # hybrid GPU-routed lane: the CPU side must skip it
    w = torch.rand(bs, top_k, generator=gen)
    w[0, 0] = 0.0

    cpu_out = ex.decode(0, hidden.to(dev), w.to(dev), ids.to(dev)).float()
    torch.cuda.synchronize()
    cpu_out = cpu_out.cpu()

    ref, extras = _reference_decode(
        cache.bank_sources, types, 0, hidden, w, ids, top_k, q8k_acts=_gguf_i8_active(ex)
    )
    rel = (cpu_out - ref).abs().max() / (ref.abs().max() + 1e-6)
    assert rel < 2e-2, f"per-projection rel err {rel.item()}"

    bound = _dfloat_bound(
        ref, extras, w, ids, hidden, _DEQUANT_FACTOR[types[0]], _DEQUANT_FACTOR[types[2]]
    )
    dev_abs = (cpu_out - ref).abs()
    assert (dev_abs <= bound).all(), (
        f"per-projection: max abs dev {dev_abs.max():.6g} exceeds the propagated "
        f"dfloat bound {bound.max():.6g}; {int((dev_abs > bound).sum())} elements outside"
    )


@pytest.mark.parametrize(
    "qtype", (GGML_IQ3_XXS, GGML_IQ4_XS, GGML_Q6_K), ids=["iq3_xxs", "iq4_xs", "q6_k"]
)
def test_cpu_moe_gguf_all_skipped_ids_yield_exact_zeros(qtype):
    """All-(-1) topk ids (every lane hybrid GPU-routed): the C++ pass 2 writes y
    unconditionally with acc = 0 for skipped lanes, so every row must be EXACTLY
    zero. Pins the hybrid merge invariant against NaN/garbage leaking into the
    merge output."""
    bs, top_k, L, E, H, I = 2, 3, 1, 4, 512, 256
    dev = torch.device("cuda")
    cache = _make_gguf_cache((qtype, qtype, qtype), L, E, H, I, seed=5)
    ids = torch.full((bs, top_k), -1, dtype=torch.int32)
    _, cpu_out, _, _, _ = _run_executor(cache, top_k, bs, H, dev, seed=13, ids=ids)
    assert (cpu_out == 0).all(), (
        f"{_TYPE_NAMES[qtype]}: all-skipped rows must be exactly zero, got max |y| "
        f"{cpu_out.abs().max().item()}"
    )


def test_cpu_moe_gguf_out_of_range_expert_is_skipped():
    """An out-of-range expert id (>= num_experts, as a corrupted router could emit)
    alongside valid ones must be skipped like a -1 lane (the C++ guard bounds e by
    the configured num_experts; skipped lanes ignore their weight) without
    corrupting the other rows or lanes."""
    bs, top_k, L, E, H, I = 2, 3, 1, 8, 512, 256
    dev = torch.device("cuda")
    types = (GGML_IQ3_XXS, GGML_IQ3_XXS, GGML_IQ4_XS)
    cache = _make_gguf_cache(types, L, E, H, I, seed=21)

    gen = torch.Generator().manual_seed(23)
    ids = torch.stack([torch.randperm(E, generator=gen)[:top_k] for _ in range(bs)]).to(
        torch.int32
    )
    ids[0, 0] = E  # out of range: must be skipped, never dereferenced
    ex, cpu_out, hidden, w, _ = _run_executor(cache, top_k, bs, H, dev, seed=23, ids=ids)

    ref_ids = ids.clone()
    ref_ids[0, 0] = -1  # the reference skips on e < 0; skipped lanes contribute 0
    ref, extras = _reference_decode(
        cache.bank_sources, types, 0, hidden, w, ref_ids, top_k, q8k_acts=_gguf_i8_active(ex)
    )
    rel = (cpu_out - ref).abs().max() / (ref.abs().max() + 1e-6)
    assert rel < 2e-2, f"out-of-range expert id rel err {rel.item()}"

    bound = _dfloat_bound(
        ref, extras, w, ref_ids, hidden,
        _DEQUANT_FACTOR[GGML_IQ3_XXS], _DEQUANT_FACTOR[GGML_IQ4_XS],
    )
    dev_abs = (cpu_out - ref).abs()
    assert (dev_abs <= bound).all(), (
        f"out-of-range expert id: max abs dev {dev_abs.max():.6g} exceeds the propagated "
        f"dfloat bound {bound.max():.6g}; {int((dev_abs > bound).sum())} elements outside"
    )


def test_cpu_moe_gguf_unsupported_type_fails_loudly():
    """A gguf type without a CPU GEMV (e.g. Q3_K = 11) must raise, not mis-decode."""
    from freetoken.moe.cpu_executor import _split_gguf_formats

    cache = SimpleNamespace(gguf_types=[(11, 11, 11)])
    with pytest.raises(NotImplementedError, match="no CPU GEMV"):
        _split_gguf_formats("gguf", cache)


def test_cpu_moe_gguf_gate_row_bytes_not_block_multiple_fails_loudly():
    """A gate bank whose row bytes are not an exact multiple of the gate format's
    block bytes must assert BEFORE H is floor-derived from row_bytes // rb: a
    silent floor would under-derive H and mis-address every row."""
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    bs, L, E, H, I = 2, 1, 8, 1024, 256
    dev = torch.device("cuda")
    qtype = GGML_IQ3_XXS
    cache = _make_gguf_cache((qtype, qtype, qtype), L, E, H, I, seed=31)
    rb = BLOCK_SHAPE[qtype][1]  # the same table the resolver reads
    bad = 3 * rb + 2  # 3 full blocks plus 2 stray bytes
    cache.bank_sources["gate"] = [b[..., :bad] for b in cache.bank_sources["gate"]]
    with pytest.raises(AssertionError, match=rf"{bad}, {rb}, '{_TYPE_NAMES[qtype]}'"):
        CpuMoeExecutor(
            cache,
            top_k=2,
            activation="silu",
            apply_router_weight_on_input=False,
            num_threads=2,
            max_tokens=bs,
            device=dev,
        )


def test_cpu_moe_gguf_short_expert_bank_fails_loudly():
    """A bank with fewer expert rows than the cache's num_experts must raise:
    the C++ skip guard bounds e by the CONFIGURED count, so a short bank would
    read out of bounds. Only the down bank is short -- gate/up pass the same
    per-role check first."""
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    bs, L, E, H, I = 2, 1, 8, 512, 256
    dev = torch.device("cuda")
    qtype = GGML_Q6_K
    cache = _make_gguf_cache((qtype, qtype, qtype), L, E, H, I, seed=37)
    short = E - 2
    cache.bank_sources["down"] = [b[:short] for b in cache.bank_sources["down"]]
    with pytest.raises(
        NotImplementedError,
        match=rf"down bank has {short} expert rows but the cache declares num_experts={E}",
    ):
        CpuMoeExecutor(
            cache,
            top_k=2,
            activation="silu",
            apply_router_weight_on_input=False,
            num_threads=2,
            max_tokens=bs,
            device=dev,
        )


def test_cpu_moe_gguf_diverging_layer_types_fail_loudly():
    """One layer whose (gate, up, down) triple diverges from layer 0 must fail
    loudly: the executor reads types[0] for the whole signature partition."""
    from freetoken.moe.cpu_executor import _split_gguf_formats

    base = (GGML_IQ3_XXS, GGML_IQ3_XXS, GGML_IQ4_XS)
    diverging = (GGML_IQ4_XS, GGML_IQ4_XS, GGML_Q6_K)  # (23, 23, 14) at layer 2
    cache = SimpleNamespace(gguf_types=[base, base, diverging])
    with pytest.raises(
        NotImplementedError, match="must be uniform within a partition"
    ) as excinfo:
        _split_gguf_formats("gguf", cache)
    assert f"[(2, {tuple(diverging)})]" in str(excinfo.value)


if __name__ == "__main__":
    for qt in (GGML_IQ3_XXS, GGML_IQ4_XS, GGML_Q6_K):
        test_cpu_moe_gguf_parity_vs_gguf_py(qt)
        print(f"gguf cpu gemv parity {_TYPE_NAMES[qt]} OK")
        test_cpu_moe_gguf_uniform_analytic(qt)
        print(f"gguf cpu gemv analytic {_TYPE_NAMES[qt]} OK")
        test_cpu_moe_gguf_all_skipped_ids_yield_exact_zeros(qt)
        print(f"gguf cpu gemv all-skipped zeros {_TYPE_NAMES[qt]} OK")
    for sig in _GLM5NEXT_SIGNATURES:
        test_cpu_moe_gguf_per_projection_formats(sig)
        print(f"gguf cpu gemv per-projection {'-'.join(_TYPE_NAMES[t] for t in sig)} OK")
    test_cpu_moe_gguf_out_of_range_expert_is_skipped()
    print("gguf cpu gemv out-of-range skip OK")
    test_cpu_moe_gguf_unsupported_type_fails_loudly()
    print("gguf cpu gemv unsupported-type OK")
