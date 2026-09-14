"""Hybrid decode's bandwidth-matched fetch split.

Covers the two halves of --moe-hybrid-max-fetch auto: the profile reader that turns
`ft bench bw` kernel bandwidths into a fetch fraction, and the ensure kernel's
per-step integer split (GPU kernel vs CPU reference mirror, and the balance rule).
"""

import json
import os

import pytest
import torch

from freetoken.moe.bench_profile import default_profile_path, load_backend_recommendation, load_hybrid_fetch_fraction
from freetoken.moe.offload_cache import OffloadMoeCache

Q = 1 << 16


def _balanced_fetch(num_missing: int, frac_q16: int) -> int:
    """Reference split: F ~ frac * misses, rounded to whichever integer neighbor
    minimizes the slower overlapped side (fetch ~ F*(1-frac), CPU ~ (M-F)*frac)."""
    lo = (num_missing * frac_q16) >> 16
    cost = lambda f: max(f * (Q - frac_q16), (num_missing - f) * frac_q16)  # noqa: E731
    return min(num_missing, lo if cost(lo) <= cost(lo + 1) else lo + 1)


def test_balanced_fetch_tracks_fraction():
    # The split follows fetched : cpu = pcie : (cpu - pcie) up to integer rounding, and
    # never over/under-shoots by more than one expert.
    for frac in (0.1, 0.415, 0.454, 0.7, 1.0):
        q = round(frac * Q)
        for m in range(0, 65):
            f = _balanced_fetch(m, q)
            assert 0 <= f <= m
            assert abs(f - frac * m) <= 1.0
    # ceil would over-fetch here (the regression this rule fixed): 41.5% of 3 misses is
    # 1.24 -> fetching 2 makes the PCIe side ~1.6x slower than balance; keep it at 1.
    assert _balanced_fetch(3, round(0.415 * Q)) == 1
    assert _balanced_fetch(4, round(0.415 * Q)) == 2


def test_load_hybrid_fetch_fraction(tmp_path):
    prof = {
        "gpu": {"name": "FAKE GPU"},
        "dtype_kernels": {
            "bf16": {"cpu_moe_gbs": 100.0, "pcie_gather_gbs": 40.0},
            # overlapped (contended) pair wins over the standalone numbers when present
            "nvfp4_x": {"cpu_moe_gbs": 100.0, "pcie_gather_gbs": 40.0,
                        "cpu_moe_overlap_gbs": 90.0, "pcie_gather_overlap_gbs": 30.0},
        },
        "workloads": {
            "m": {"kernels": {"ds_fp4": {"cpu_moe_gbs": 80.0, "pcie_gather_gbs": 50.0}}}
        },
    }
    path = tmp_path / "benchbw.json"
    path.write_text(json.dumps(prof))
    # standalone fallback: full-contention assumption -> pcie / cpu
    assert load_hybrid_fetch_fraction("bf16", path=str(path)) == pytest.approx(0.4)
    # overlapped pair preferred: pcie_ov / (pcie_ov + cpu_ov)
    assert load_hybrid_fetch_fraction("nvfp4_x", path=str(path)) == pytest.approx(0.25)
    # per-model fallback when there is no per-dtype entry for the format
    assert load_hybrid_fetch_fraction("ds_fp4", path=str(path)) == pytest.approx(0.625)
    assert load_hybrid_fetch_fraction("nvfp4", path=str(path)) is None
    # a profile from different hardware is ignored
    assert load_hybrid_fetch_fraction("bf16", gpu_name="OTHER", path=str(path)) is None


def test_profile_lookup_prefers_the_gpu_uuid_file(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("FREETOKEN_BENCHBW_PATH", raising=False)
    uuid = "GPU-2f3a9b1c-0000-1111-2222-333344445555"

    def write(path, name, verdict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"gpu": {"name": name}, "dtypes": {"bf16": verdict}}, f)

    # legacy single file only: used when the name matches, ignored otherwise
    write(default_profile_path(), "FAKE GPU", "hybrid")
    assert load_backend_recommendation("bf16", gpu_name="FAKE GPU", gpu_uuid=uuid) == "hybrid"
    assert load_backend_recommendation("bf16", gpu_name="OTHER", gpu_uuid=uuid) is None
    # this card's own file wins over the legacy one
    write(default_profile_path(uuid), "FAKE GPU", "offload")
    assert load_backend_recommendation("bf16", gpu_name="FAKE GPU", gpu_uuid=uuid) == "offload"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_hybrid_fraction_gpu_matches_cpu_reference():
    torch.manual_seed(0)
    num_experts, cache_size, top_k, frac = 32, 40, 8, 0.415

    def make():
        return OffloadMoeCache(
            num_layers=2, num_experts=num_experts, cache_size=cache_size,
            device=torch.device("cuda"), quant_format="bf16", decode_target="hybrid",
            hybrid_max_fetch=num_experts, hybrid_fetch_fraction=frac,
        )

    gpu, ref = make(), make()
    frac_q16 = round(frac * Q)
    for step in range(64):
        ids = torch.randperm(num_experts)[:top_k].to(torch.int32)
        g, c = ids.clone().cuda(), ids.clone()  # a CPU ids tensor drives the reference path
        gpu.ensure_experts_hybrid(0, g)
        ref.ensure_experts_hybrid(0, c)
        missing = int(gpu.num_missing_full.item())
        fetched = int(gpu.num_indices.item())
        assert missing == int(ref.num_missing_full.item())
        assert fetched == int(ref.num_indices.item()) == _balanced_fetch(missing, frac_q16)
        # slot rewrites (hit/fetched -> slot, overflow -> -1) and LRU state stay identical
        assert torch.equal(g.cpu(), c)
        assert torch.equal(gpu.slot_for_id.cpu(), ref.slot_for_id.cpu())
        assert torch.equal(gpu.id_of_slot.cpu(), ref.id_of_slot.cpu())
        assert (g >= 0).sum().item() == len(set(ids.tolist())) - (missing - fetched)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_hybrid_fixed_cap_unchanged():
    # fraction 0 (no profile / explicit --moe-hybrid-max-fetch) keeps the fixed cap.
    cache = OffloadMoeCache(
        num_layers=1, num_experts=32, cache_size=40, device=torch.device("cuda"),
        quant_format="bf16", decode_target="hybrid", hybrid_max_fetch=1,
    )
    ids = torch.arange(8, dtype=torch.int32).cuda()
    cache.ensure_experts_hybrid(0, ids)
    assert int(cache.num_missing_full.item()) == 8
    assert int(cache.num_indices.item()) == 1


def test_benchbw_gguf_profile_sanity(monkeypatch, tmp_path):
    # The "gguf" bench row: the workload constructs (glm5.3-flash geometry), the bank
    # specs yield the three stacked projections at the exact offload_cache widths, the
    # profile reader resolves the format through the identity mapping, and the bench's
    # CPU-MoE section skips (no CPU weight path) so the verdict is always offload.
    import json

    from freetoken.moe import benchbw
    from freetoken.moe.offload_cache import _BANK_BYTES_PER_EXPERT

    wl = benchbw.DTYPE_WORKLOADS["gguf"]
    assert (wl.hidden, wl.inter, wl.experts, wl.top_k) == (4096, 2048, 288, 8)  # glm5.3-flash
    assert wl.formats == ("gguf",)
    assert wl.activation == "swiglu_clamp" and wl.swiglu_alpha == 1.0 and wl.swiglu_limit == 10.0

    specs = benchbw._offload_bank_specs("gguf", wl.hidden, wl.inter)
    assert tuple(specs) == ("gate", "up", "down")
    assert {k: v for k, (v, _) in specs.items()} == {
        "gate": wl.inter * (wl.hidden // 256) * 98,
        "up": wl.inter * (wl.hidden // 256) * 98,
        "down": wl.hidden * (wl.inter // 256) * 210,
    }
    assert all(dt is torch.uint8 for _, dt in specs.values())
    # the bench geometry is the sizing table the engine consults pre-load
    assert benchbw._expert_bytes("gguf", wl.hidden, wl.inter) == _BANK_BYTES_PER_EXPERT["gguf"](wl.hidden, wl.inter)

    # identity profile row: the reader maps "gguf" through unchanged
    prof = {
        "gpu": {"name": "FAKE GPU"},
        "dtypes": {"gguf": "offload"},
        "dtype_kernels": {"gguf": {"cpu_moe_gbs": 100.0, "pcie_gather_gbs": 40.0}},
    }
    path = tmp_path / "p.json"
    path.write_text(json.dumps(prof))
    assert load_backend_recommendation("gguf", path=str(path)) == "offload"
    assert load_hybrid_fetch_fraction("gguf", path=str(path)) == pytest.approx(0.4)

    # the bench itself: the CPU-MoE section skips (gguf not in _CPU_MOE_FORMATS), so the
    # verdict is offload even when the PCIe gather side would look slow
    assert "gguf" not in benchbw._CPU_MOE_FORMATS
    eb = benchbw._expert_bytes("gguf", wl.hidden, wl.inter)
    monkeypatch.setattr(
        benchbw, "measure_pcie_gather_bw",
        lambda *a, **k: {"bw_gbs": 40.0, "expert_bytes": eb, "synth_experts": 160, "fused": True},
    )
    entry = benchbw._bench_format(
        "gguf", wl, torch.device("cuda"), threshold=1.0, cpu_threads=0, cpu_iters=1, pcie_iters=1
    )
    assert entry["recommended"] == "offload"
    assert entry["cpu_moe_gbs"] is None and entry["cpu_moe_isa"] is None
    assert entry["ratio"] is None and entry["pcie_gather_gbs"] == 40.0
    assert "CPU MoE has no gguf weight path" in entry["note"]
