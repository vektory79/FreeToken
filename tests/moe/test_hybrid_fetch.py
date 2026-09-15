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
    # profile reader resolves the format through the identity mapping, and the CPU-MoE
    # leg RUNS (the gguf K-quant GEMV rides the executor's "gguf" alias), so the
    # verdict follows the measured CPU-vs-PCIe pair instead of being pinned to offload.
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

    # the bench itself: the CPU leg runs for the gguf alias (and the concrete family
    # types + q4_0), so the verdict comes from recommend() over the measured pair and
    # the contended pair sets the fetch split. Legs mocked -> unit-level.
    assert {"gguf", "iq3_xxs", "iq4_xs", "q6_k", "q4_0"} <= benchbw._CPU_MOE_FORMATS
    eb = benchbw._expert_bytes("gguf", wl.hidden, wl.inter)
    monkeypatch.setattr(
        benchbw, "measure_pcie_gather_bw",
        lambda *a, **k: {"bw_gbs": 40.0, "expert_bytes": eb, "synth_experts": 160, "fused": True},
    )
    monkeypatch.setattr(
        benchbw, "measure_cpu_moe_bw",
        lambda *a, **k: {"bw_gbs": 100.0, "isa": "scalar", "isa_sweep": {"scalar": 100.0},
                         "expert_bytes": eb, "synth_experts": 160},
    )
    monkeypatch.setattr(
        benchbw, "measure_overlap_bw",
        lambda *a, **k: {"cpu_gbs": 90.0, "pcie_gbs": 30.0},
    )
    entry = benchbw._bench_format(
        "gguf", wl, torch.device("cuda"), threshold=1.0, cpu_threads=0, cpu_iters=1, pcie_iters=1
    )
    assert entry["recommended"] == "hybrid"
    assert entry["cpu_moe_gbs"] == 100.0 and entry["cpu_moe_isa"] == "scalar"
    assert entry["ratio"] == 2.5 and entry["pcie_gather_gbs"] == 40.0
    assert entry["cpu_moe_overlap_gbs"] == 90.0 and entry["pcie_gather_overlap_gbs"] == 30.0

    # the fetch split the engine consults (load_hybrid_fetch_fraction): the contended
    # pair, strictly inside (0, 1)
    kernels_path = tmp_path / "kernels.json"
    kernels_path.write_text(json.dumps({"gpu": {"name": "FAKE GPU"}, "dtype_kernels": {"gguf": entry}}))
    frac = load_hybrid_fetch_fraction("gguf", path=str(kernels_path))
    assert 0.0 < frac < 1.0
    assert frac == pytest.approx(30.0 / (30.0 + 90.0))


def test_benchbw_non_cpu_capable_profile_stays_offload(monkeypatch):
    # fp8_block has no CPU MoE weight path: the CPU leg is skipped and the verdict is
    # offload no matter how slow the PCIe gather looks (threshold 1.0 here) -- the
    # gguf flip must not loosen the not-CPU-capable branch.
    from freetoken.moe import benchbw

    assert "fp8_block" not in benchbw._CPU_MOE_FORMATS
    wl = benchbw.DTYPE_WORKLOADS["fp8_block"]
    eb = benchbw._expert_bytes("fp8_block", wl.hidden, wl.inter)
    monkeypatch.setattr(
        benchbw, "measure_pcie_gather_bw",
        lambda *a, **k: {"bw_gbs": 40.0, "expert_bytes": eb, "synth_experts": 160, "fused": True},
    )
    entry = benchbw._bench_format(
        "fp8_block", wl, torch.device("cuda"), threshold=1.0, cpu_threads=0, cpu_iters=1, pcie_iters=1
    )
    assert entry["recommended"] == "offload"
    assert entry["cpu_moe_gbs"] is None and entry["cpu_moe_isa"] is None
    assert entry["ratio"] is None and entry["pcie_gather_gbs"] == 40.0
    assert "CPU MoE has no fp8_block weight path" in entry["note"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_benchbw_gguf_banks_build_the_executor(monkeypatch):
    # The bench's synthetic gguf banks + cache stub construct a real CpuMoeExecutor
    # end to end: the stub's gguf_types triple resolves the "gguf" alias and the
    # per-role bank shapes satisfy _resolve_gguf_banks at the exact offload widths.
    from freetoken.models.gguf.dequant import BLOCK_SHAPE, GGML_IQ3_XXS, GGML_Q6_K
    from freetoken.moe import benchbw

    wl = benchbw.DTYPE_WORKLOADS["gguf"]
    H, I = wl.hidden, wl.inter
    assert BLOCK_SHAPE[GGML_IQ3_XXS] == (256, 98) and BLOCK_SHAPE[GGML_Q6_K] == (256, 210)
    # shrink the synthetic expert count so the pinned banks stay small in a unit test
    monkeypatch.setattr(benchbw, "_SYNTH_BANK_BUDGET", 64 << 20)
    E = benchbw._synth_experts(wl.experts, benchbw._expert_bytes("gguf", H, I))
    assert 0 < E < wl.experts
    banks = benchbw._cpu_moe_bank_sources("gguf", H, I, E)
    assert tuple(banks) == ("gate", "up", "down")
    assert banks["gate"].shape == (E, I, (H // 256) * 98)  # IQ3_XXS rows over K = H
    assert banks["up"].shape == (E, I, (H // 256) * 98)
    assert banks["down"].shape == (E, H, (I // 256) * 210)  # Q6_K rows over K = I
    # the CPU leg reads exactly the bytes the gather leg's _offload_bank_specs sums
    assert sum(int(t.numel()) for t in banks.values()) == E * benchbw._expert_bytes("gguf", H, I)

    ex = benchbw._build_cpu_moe_executor("gguf", wl, banks, num_threads=2, E=E)
    assert (ex.quant_format, ex.fmt_up, ex.fmt_down) == ("iq3_xxs", "iq3_xxs", "q6_k")
    assert (ex.H, ex.I, ex.num_experts) == (H, I, E)


def test_hybrid_multi_partition_executors_mixed_signatures():
    """Per-cache executors over the real glm5next signature set -- (18,18,14) and
    (23,23,14) alongside the dominant (18,18,23): the blanket multi-partition
    ValueError rejected this exact shape before per-cache executors existed. Each
    partition's executor must resolve ITS OWN per-layer per-role types (the CPU
    mirror of the GPU path's cache.gguf_types[layer] dispatch), and the thread
    budget must split into DISJOINT per-pool core sets (every pool pins its
    workers, so shared cores would oversubscribe N-wide)."""
    from types import SimpleNamespace

    from freetoken.engine.engine import Engine
    from freetoken.moe.cpu_executor import _GGUF_TYPE_FMTS, resolve_pool_affinities
    from freetoken.moe.offload_cache import OffloadMoeCache

    E, H, I = 4, 512, 256
    sigs = [(18, 18, 14), (23, 23, 14), (18, 18, 23)]  # the real file's groups
    layer_counts = (2, 1, 1)  # dominant + two single-layer minorities
    caches = []
    for sig, num_layers in zip(sigs, layer_counts):
        cache = OffloadMoeCache(
            num_layers=num_layers, num_experts=E, cache_size=2 * E,
            device=torch.device("cpu"), quant_format="gguf",
            gguf_types=tuple([tuple(sig)] * num_layers), decode_target="hybrid",
        )
        # construction stub: the executor only reads these dicts' pointers/shapes
        # (per-role widths follow the ROLE's own type: gate/up rows pack over H,
        # down rows over I -- never one shared row width)
        from freetoken.models.gguf.dequant import BLOCK_SHAPE

        cache.bank_sources = {
            "gate": [
                torch.zeros(E, I, (H // 256) * BLOCK_SHAPE[sig[0]][1], dtype=torch.uint8)
                for _ in range(num_layers)
            ],
            "up": [
                torch.zeros(E, I, (H // 256) * BLOCK_SHAPE[sig[1]][1], dtype=torch.uint8)
                for _ in range(num_layers)
            ],
            "down": [
                torch.zeros(E, H, (I // 256) * BLOCK_SHAPE[sig[2]][1], dtype=torch.uint8)
                for _ in range(num_layers)
            ],
        }
        caches.append(cache)

    config = SimpleNamespace(moe_cpu_threads=3, max_running_req=4, cuda_graph_max_bs=2)
    layers = [
        SimpleNamespace(
            top_k=2, activation="swiglu_clamp", apply_router_weight_on_input=False,
            quant_method=None, alpha=1.0, limit=10.0,
        )
    ]
    fake = SimpleNamespace(device=torch.device("cpu"), cpu_moe_executors=[])
    Engine._init_cpu_moe_executors(fake, config, caches, layers)

    assert len(fake.cpu_moe_executors) == 3
    assert len({id(e) for e in fake.cpu_moe_executors}) == 3  # one instance per cache
    for cache, sig, ex in zip(caches, sigs, fake.cpu_moe_executors):
        assert cache.cpu_executor is ex
        # THIS partition's role triple, never another partition's
        assert {ex.quant_format, ex.fmt_up, ex.fmt_down} == {_GGUF_TYPE_FMTS[t] for t in sig}
    assert (fake.cpu_moe_executors[0].quant_format, fake.cpu_moe_executors[0].fmt_down) == ("iq3_xxs", "q6_k")
    assert (fake.cpu_moe_executors[1].quant_format, fake.cpu_moe_executors[1].fmt_down) == ("iq4_xs", "q6_k")
    assert (fake.cpu_moe_executors[2].quant_format, fake.cpu_moe_executors[2].fmt_down) == ("iq3_xxs", "iq4_xs")

    # explicit --moe-cpu-threads 3 over 3 pools: one worker each, disjoint cores
    pools = [ex.core_ids for ex in fake.cpu_moe_executors]
    flat = [c for pool in pools for c in pool]
    assert all(ex.num_threads == 1 for ex in fake.cpu_moe_executors)
    assert len(flat) == len(set(flat)), f"executor pools share cores: {pools}"

    # the splitter itself: auto covers every physical core exactly once, explicit
    # splits as evenly as possible (remainder to the earlier pools) and never
    # starves a pool
    from freetoken.moe.cpu_executor import physical_core_cpus

    reps = physical_core_cpus()
    auto = resolve_pool_affinities(3, 0)
    auto_flat = [c for p in auto for c in p]
    assert sorted(auto_flat) == sorted(reps)  # disjoint + complete cover
    assert [len(p) for p in resolve_pool_affinities(3, 4)] == [2, 1, 1]
    assert [len(p) for p in resolve_pool_affinities(3, 2)] == [1, 1, 1]  # floor at one


def test_resolve_pool_affinities_clamps_explicit_overspend():
    # Task 07 B-2: explicit --moe-cpu-threads above the usable core count used to
    # wrap the WHOLE core order, so a later pool received cores an earlier pool
    # already pins (the documented disjoint invariant broken). Clamped now: the
    # pools stay a disjoint physical-core cover and the overspend is dropped.
    import os

    from freetoken.moe.cpu_executor import physical_core_cpus, resolve_pool_affinities

    reps = physical_core_cpus()
    allowed = sorted(os.sched_getaffinity(0))
    usable = len(reps) + len([c for c in allowed if c not in set(reps)])
    pools = resolve_pool_affinities(3, usable + 10)
    flat = [c for p in pools for c in p]
    assert len(flat) == len(set(flat)), f"pools share cores: {pools}"
    # clamped to the usable core set: a disjoint cover of every allowed CPU
    assert sorted(flat) == sorted(allowed)
    assert all(len(p) >= 1 for p in pools)
    # sane budgets keep the exact split / floor-at-one semantics
    assert [len(p) for p in resolve_pool_affinities(3, 4)] == [2, 1, 1]
    assert [len(p) for p in resolve_pool_affinities(3, 2)] == [1, 1, 1]


def test_resolve_pool_affinities_weights_by_layer_count(monkeypatch):
    """Task 06 B: the thread budget is distributed across pools PROPORTIONAL TO
    THE PARTITION LAYER COUNT. Task 05 measured the even split starving the
    dominant partition (6/16 threads on the real file's 39/2/1 layer distribution
    -> ~2.96 GB/s effective CPU leg vs 11.3 benched). The real file's weights at
    16 threads: 39/42*16 = 14.86 -> 15 workers (largest-remainder + floor-at-one),
    the two single-layer partitions keep one each. Disjointness and floor-at-one
    invariants are unchanged."""
    from freetoken.moe.cpu_executor import resolve_pool_affinities

    weights = [39, 2, 1]  # the real glm5next file: dominant + two minority partitions
    # explicit --moe-cpu-threads: weighted largest-remainder, floor-at-one preserved;
    # below the usable core set the split may still overspend the flag by pools-1
    # workers (as the even split could) -- this rig has 20 cores + SMT siblings, so
    # both splits fit; the tight-box trim is pinned by
    # test_resolve_pool_affinities_weighted_tight_box_never_wraps
    assert [len(p) for p in resolve_pool_affinities(3, 16, weights=weights)] == [15, 1, 1]
    assert [len(p) for p in resolve_pool_affinities(3, 20, weights=weights)] == [19, 1, 1]

    # auto: the same weighting over the physical cores, trimmed back to a DISJOINT
    # cover (floor-at-one alone would hand out 19+1+1 = 21 cores on a 20-core box)
    reps = list(range(100, 120))  # deterministic 20-physical-core rig
    monkeypatch.setattr("freetoken.moe.cpu_executor.physical_core_cpus", lambda: list(reps))
    pools = resolve_pool_affinities(3, 0, weights=weights)
    flat = [c for p in pools for c in p]
    assert len(flat) == len(set(flat)), f"pools share cores: {pools}"
    assert sorted(flat) == reps  # disjoint + complete cover of the physical cores
    assert [len(p) for p in pools] == [18, 1, 1]

    # explicit split keeps disjoint physical-first ordered slices
    pools = resolve_pool_affinities(3, 16, weights=weights)
    flat = [c for p in pools for c in p]
    assert len(flat) == len(set(flat)), f"pools share cores: {pools}"
    assert pools[0] == reps[:15]

    # floor-at-one invariant under weights: a starved pool is worse than overspending
    assert all(len(p) >= 1 for p in resolve_pool_affinities(3, 3, weights=weights))

    # even split preserved when weights are absent (backward compatibility)
    assert [len(p) for p in resolve_pool_affinities(3, 4)] == [2, 1, 1]


def test_resolve_pool_affinities_weighted_tight_box_never_wraps():
    """Corner: skewed weights + requested >= usable cores. Floor-at-one alone
    overspends the clamped budget (sum(counts) up to total + pools-1), which
    used to push the last pools' slices past the usable set and WRAP them onto
    cores an earlier pool already pins (the eb7de4c never-wraps invariant).
    The overflow is trimmed back out of the largest (dominant) pool:
    floor-at-one, disjointness and the dominant share all hold."""
    import os

    from freetoken.moe.cpu_executor import physical_core_cpus, resolve_pool_affinities

    reps = physical_core_cpus()
    allowed = sorted(os.sched_getaffinity(0))
    usable = len(reps) + len([c for c in allowed if c not in set(reps)])
    if usable < 8:
        pytest.skip("needs at least two cores per pool for the 4-pool corner")
    weights = [100, 1, 1, 1]  # every minority share floors below one at this budget
    for requested in (usable, usable + 7):  # exact fit and the clamped over-budget flag
        pools = resolve_pool_affinities(4, requested, weights=weights)
        flat = [c for p in pools for c in p]
        assert len(flat) == len(set(flat)), f"pools wrap/share cores ({requested}): {pools}"
        assert set(flat) <= set(allowed), f"cores outside the usable set ({requested})"
        assert all(len(p) >= 1 for p in pools), f"floor-at-one broken ({requested}): {pools}"
        sizes = [len(p) for p in pools]
        assert sum(sizes) == usable, f"split not re-clamped to the usable set: {sizes}"
        assert sizes[0] == max(sizes) and sizes[0] >= sum(sizes[1:]), (
            f"overflow not trimmed from the dominant pool ({requested}): {sizes}"
        )


def test_resolve_threads_and_affinity_physical_core_free_subset_is_coherent():
    # Task 07 A-info: auto sizing inside an allow_cores subset with NO physical-core
    # representative used to return nthreads=0 with non-empty core ids.
    import os

    from freetoken.moe.cpu_executor import physical_core_cpus, resolve_threads_and_affinity

    logical_only = [c for c in sorted(os.sched_getaffinity(0)) if c not in set(physical_core_cpus())]
    if not logical_only:
        pytest.skip("no SMT siblings: every allowed CPU is a physical-core rep")
    n, cores = resolve_threads_and_affinity(0, logical_only)
    assert n == len(cores) >= 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_hybrid_cpu_miss_rows_match_gpu_path(monkeypatch):
    """S2, multi-partition hybrid dispatch: a layer whose cache splits CPU-miss +
    GPU-hit -- the CPU partial computed for the overflow-missed experts must match
    what the GPU path computes for the SAME rows (same banks, same routing, the
    accepted CPU-vs-GPU tolerance model), and the merged hybrid output must equal
    the pure-GPU answer for the full routing."""
    import pathlib
    import sys

    # This test pins the hybrid MERGE machinery, not the dot tier: the W4A8-K tier's
    # activation quantization is a real (bounded) CPU-vs-GPU numeric gap, covered by
    # the tier A/B in test_cpu_moe_gguf_iq.py. Keep the comparison scalar.
    monkeypatch.setenv("FREETOKEN_GGUF_DOT_TIER", "scalar")

    sys.path.insert(0, str(pathlib.Path(__file__).parent))  # noqa: F401 - body imports below
    import test_cpu_moe_gguf_iq  # noqa: F401 - sibling fixture module

    _hybrid_cpu_miss_rows_match_gpu_path_body()


def _hybrid_cpu_miss_rows_match_gpu_path_body():
    # the wrapper test already put this directory on sys.path and imported the
    # sibling fixture module
    from test_cpu_moe_gguf_iq import _make_gguf_cache

    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    types = ((18, 18, 23),)  # the real file's dominant signature
    E, H, I, bs, top_k = 8, 512, 256, 3, 4
    dev = torch.device("cuda")
    stub = _make_gguf_cache(types[0], 1, E, H, I, seed=101)
    banks = {role: list(per_layer) for role, per_layer in stub.bank_sources.items()}

    def make(decode_target, fetch):
        cache = OffloadMoeCache(
            num_layers=1, num_experts=E, cache_size=2 * E, device=dev,
            quant_format="gguf", gguf_types=types, decode_target=decode_target,
            hybrid_max_fetch=fetch,
        )
        cache.set_bank_sources(banks)
        return cache

    cache = make("hybrid", fetch=1)
    ref_cache = make("gpu", fetch=1)

    ex = CpuMoeExecutor(
        cache, top_k=top_k, activation="silu", apply_router_weight_on_input=False,
        num_threads=2, max_tokens=bs, device=dev,
    )
    cache.set_cpu_executor(ex)

    def layer_for(c):
        layer = OffloadMoELayer.__new__(OffloadMoELayer)  # the branch only needs these attrs
        layer.quant_method = None
        layer.activation = "silu"
        layer.alpha = 1.0
        layer.limit = None
        layer.offload_cache = c
        layer.layer_id = 0
        return layer

    gen = torch.Generator().manual_seed(5)
    hidden = (torch.randn(bs, H, generator=gen) * 0.5).to(torch.bfloat16).to(dev)
    raw = torch.stack(
        [torch.randperm(E, generator=gen)[:top_k] for _ in range(bs)]
    ).to(torch.int32).to(dev)
    raw[:, 0] = 0  # lane 0 pre-resident: the GPU hit
    w = torch.rand(bs, top_k, generator=gen).to(dev)

    hits = raw[:, :1].clone()
    cache.ensure_experts(0, hits)
    cache.copy_missing()

    ids = raw.clone()
    out = layer_for(cache)._decode_hybrid(cache, hidden, w, ids)
    torch.cuda.synchronize()
    cpu_lanes = ids < 0  # rewritten in place: slot (hit/fetched) or -1 (CPU-owned)
    assert cpu_lanes.any() and (~cpu_lanes).any(), "fixture must mix CPU misses and GPU hits"

    # CPU partial for exactly the overflow-missed rows (raw ids, -1 on GPU lanes)
    w_miss = torch.where(cpu_lanes, w, w.new_zeros(())).contiguous()
    cpu_ids = torch.where(cpu_lanes, raw, raw.new_full((), -1)).contiguous()
    cpu_part = ex.decode(0, hidden, w_miss, cpu_ids).float()
    torch.cuda.synchronize()

    # GPU path for the SAME rows: same banks, the missed experts fetched, zero
    # weight on the hit lanes (the hybrid GPU side's own zeroing convention)
    ref_ids = raw.clone()
    ref_cache.ensure_experts(0, ref_ids)
    ref_cache.copy_missing()
    gpu_miss = layer_for(ref_cache)._expert_gemm(
        ref_cache, hidden, w_miss, ref_ids,
        views=ref_cache.bank_views(), n=None, alphas=None, is_prefill=False,
    ).float()
    gpu_full = layer_for(ref_cache)._expert_gemm(
        ref_cache, hidden, w, ref_ids,
        views=ref_cache.bank_views(), n=None, alphas=None, is_prefill=False,
    ).float()
    torch.cuda.synchronize()

    rel = (cpu_part - gpu_miss).abs().max() / (gpu_miss.abs().max() + 1e-6)
    assert rel < 2e-2, f"CPU-miss rows diverge from the GPU path: rel {rel.item()}"

    # merged hybrid output == the pure-GPU answer for the FULL routing
    rel = (out.float() - gpu_full).abs().max() / (gpu_full.abs().max() + 1e-6)
    assert rel < 2e-2, f"hybrid merge diverges from the GPU path: rel {rel.item()}"


def _stub_gguf_cache(sig, num_layers, E, H, I, device):
    """OffloadMoeCache + shape-correct zero banks the executor construction can
    resolve (per-role widths follow the ROLE's own gguf type)."""
    from freetoken.models.gguf.dequant import BLOCK_SHAPE
    from freetoken.moe.offload_cache import OffloadMoeCache

    cache = OffloadMoeCache(
        num_layers=num_layers, num_experts=E, cache_size=2 * E, device=device,
        quant_format="gguf", gguf_types=tuple([tuple(sig)] * num_layers),
        decode_target="hybrid",
    )
    cache.bank_sources = {
        "gate": [torch.zeros(E, I, (H // 256) * BLOCK_SHAPE[sig[0]][1], dtype=torch.uint8) for _ in range(num_layers)],
        "up": [torch.zeros(E, I, (H // 256) * BLOCK_SHAPE[sig[1]][1], dtype=torch.uint8) for _ in range(num_layers)],
        "down": [torch.zeros(E, H, (I // 256) * BLOCK_SHAPE[sig[2]][1], dtype=torch.uint8) for _ in range(num_layers)],
    }
    return cache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_multi_pool_auto_executors_carve_a_coordinator_core_per_pool():
    # Task 07 B-5(c): the AUTO (--moe-cpu-threads 0) multi-pool boot on CUDA was
    # unexercised - each pool must get a disjoint physical-core slice and carve its
    # OWN coordinator core out of that slice (allow_cores), so workers +
    # coordinators still cover every physical core exactly once, and the health
    # error must say WHICH pool fired (the engine-passed label).
    from types import SimpleNamespace

    from freetoken.engine.engine import Engine
    from freetoken.moe.cpu_executor import physical_core_cpus, resolve_pool_affinities

    E, H, I = 4, 512, 256
    sigs = [(18, 18, 23), (18, 18, 14), (23, 23, 14)]
    caches = [_stub_gguf_cache(sig, 1, E, H, I, torch.device("cuda")) for sig in sigs]
    config = SimpleNamespace(moe_cpu_threads=0, max_running_req=4, cuda_graph_max_bs=2)
    layers = [
        SimpleNamespace(
            top_k=2, activation="swiglu_clamp", apply_router_weight_on_input=False,
            quant_method=None, alpha=1.0, limit=10.0,
        )
    ]
    fake = SimpleNamespace(device=torch.device("cuda"), cpu_moe_executors=[])
    Engine._init_cpu_moe_executors(fake, config, caches, layers)

    executors = fake.cpu_moe_executors
    assert len(executors) == 3
    if not executors[0]._flag_sync:
        pytest.skip("flag-sync stream memops unavailable on this driver")
    pools = resolve_pool_affinities(len(caches), 0)
    if min(len(p) for p in pools) <= 2:
        pytest.skip("too few physical cores for a per-pool coordinator carve-out")

    for pool_idx, (ex, cache, pool) in enumerate(zip(executors, caches, pools)):
        assert cache.cpu_executor is ex
        assert ex.num_threads == len(pool) - 1  # the coordinator core is carved out
        assert ex.core_ids == pool[:-1]
        assert ex._coord_core == pool[-1]
        assert ex.label.startswith(f"pool {pool_idx + 1}/{len(caches)}")
    assigned = [c for ex in executors for c in ex.core_ids] + [ex._coord_core for ex in executors]
    assert len(assigned) == len(set(assigned)), "workers/coordinators must be disjoint"
    assert sorted(assigned) == sorted(physical_core_cpus())

    # B-1: the health error carries the failing pool's label
    executors[1]._err[0] = 1
    with pytest.raises(RuntimeError, match=r"pool 2/3"):
        executors[1].raise_if_unhealthy()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_hybrid_overlap_disabled_multi_partition_serial_path(monkeypatch):
    # Task 07 B-5(d): FREETOKEN_HYBRID_OVERLAP=0 (the A/B escape hatch) was only
    # ever exercised single-partition; across per-partition executors each
    # partition's _decode_hybrid must sync ITS OWN executor's pending handle before
    # the PCIe fetch (the serial path) and still match the pure-GPU answer.
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from test_cpu_moe_gguf_iq import _make_gguf_cache

    from freetoken.layers import moe as moe_module
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.offload_cache import OffloadMoeCache

    monkeypatch.setattr(moe_module, "_HYBRID_OVERLAP", False)
    # Merge-machinery test (see test_hybrid_cpu_miss_rows_match_gpu_path): pin the
    # scalar dot tier so the CPU-vs-GPU rel bounds stay about the routing.
    monkeypatch.setenv("FREETOKEN_GGUF_DOT_TIER", "scalar")

    E, H, I, bs, top_k = 8, 512, 256, 3, 4
    dev = torch.device("cuda")
    gen = torch.Generator().manual_seed(9)
    hidden = (torch.randn(bs, H, generator=gen) * 0.5).to(torch.bfloat16).to(dev)
    raw = torch.stack(
        [torch.randperm(E, generator=gen)[:top_k] for _ in range(bs)]
    ).to(torch.int32).to(dev)
    raw[:, 0] = 0  # lane 0 pre-resident: the GPU hit
    w = torch.rand(bs, top_k, generator=gen).to(dev)

    def layer_for(c):
        layer = OffloadMoELayer.__new__(OffloadMoELayer)  # the branch only needs these attrs
        layer.quant_method = None
        layer.activation = "silu"
        layer.alpha = 1.0
        layer.limit = None
        layer.offload_cache = c
        layer.layer_id = 0
        return layer

    for sig in ((18, 18, 23), (23, 23, 14)):  # dominant + minority partition
        # uniform analytic banks: the random fixture's weight magnitudes push the
        # unbounded silu intermediate past the GPU path's activation-quant range
        # (inf * the zero-weighted hit lanes -> nan) - bounded weights keep the
        # CPU-vs-GPU comparison about the routing, not the fixture
        stub = _make_gguf_cache(sig, 1, E, H, I, seed=13, uniform=True)
        banks = {role: list(per_layer) for role, per_layer in stub.bank_sources.items()}

        def make(decode_target, fetch):
            cache = OffloadMoeCache(
                num_layers=1, num_experts=E, cache_size=2 * E, device=dev,
                quant_format="gguf", gguf_types=(tuple(sig),), decode_target=decode_target,
                hybrid_max_fetch=fetch,
            )
            cache.set_bank_sources(banks)
            return cache

        cache = make("hybrid", fetch=1)
        ref_cache = make("gpu", fetch=1)
        ex = CpuMoeExecutor(
            cache, top_k=top_k, activation="silu", apply_router_weight_on_input=False,
            num_threads=2, max_tokens=bs, device=dev,
        )
        cache.set_cpu_executor(ex)

        ids = raw.clone()
        out = layer_for(cache)._decode_hybrid(cache, hidden, w, ids)
        torch.cuda.synchronize()
        cpu_lanes = ids < 0  # rewritten in place: slot (hit/fetched) or -1 (CPU-owned)
        assert cpu_lanes.any() and (~cpu_lanes).any(), "fixture must mix CPU misses and GPU hits"

        # CPU partial for exactly the overflow-missed rows (raw ids, -1 on GPU lanes)
        w_miss = torch.where(cpu_lanes, w, w.new_zeros(())).contiguous()
        cpu_ids = torch.where(cpu_lanes, raw, raw.new_full((), -1)).contiguous()
        cpu_part = ex.decode(0, hidden, w_miss, cpu_ids).float()
        torch.cuda.synchronize()

        ref_ids = raw.clone()
        ref_cache.ensure_experts(0, ref_ids)
        ref_cache.copy_missing()
        gpu_miss = layer_for(ref_cache)._expert_gemm(
            ref_cache, hidden, w_miss, ref_ids,
            views=ref_cache.bank_views(), n=None, alphas=None, is_prefill=False,
        ).float()
        gpu_full = layer_for(ref_cache)._expert_gemm(
            ref_cache, hidden, w, ref_ids,
            views=ref_cache.bank_views(), n=None, alphas=None, is_prefill=False,
        ).float()
        torch.cuda.synchronize()

        # 3e-2, not the parity tests' 2e-2: the CPU-vs-GPU gap is the GPU down-GEMV's
        # q8_1 activation quant, whose relative size on small outputs is
        # fixture-dependent (the dequant contract itself is pinned by the parity tests)
        rel = (cpu_part - gpu_miss).abs().max() / (gpu_miss.abs().max() + 1e-6)
        assert rel < 3e-2, f"[{sig}] CPU-miss rows diverge from the GPU path: rel {rel.item()}"
        rel = (out.float() - gpu_full).abs().max() / (gpu_full.abs().max() + 1e-6)
        assert rel < 2e-2, f"[{sig}] serial merge diverges from the GPU path: rel {rel.item()}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_hybrid_avx2_tier_weighted_pools_production_combo(monkeypatch):
    """The production combo end to end: the avx2 W4A8-K dot tier (what this box
    boots with) + WEIGHTED per-partition pools resolved by
    Engine._init_cpu_moe_executors + one hybrid merge step on the dominant
    partition.

    Tier-aware variant of the scalar-pin rationale (see
    test_hybrid_cpu_miss_rows_match_gpu_path): the merge machinery is already
    GPU-anchored at the scalar tier, so here the avx2 merge is anchored to the
    SAME merge at the scalar tier within the tier A/B activation-quant envelope
    (test_cpu_moe_gguf_iq's bound_q8 + bound_bf16 + act_gap contract) instead of
    to the GPU path directly -- the W4A8-K quantization must not be counted
    against the GPU's own activation quant a second time."""
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).parent))  # noqa: F401 - body imports below
    import test_cpu_moe_gguf_iq as iq  # noqa: F401 - sibling fixture module

    if not iq._HOST_AVX2:
        pytest.skip("the avx2 W4A8-K tier engages only on AVX2+FMA hosts")

    from types import SimpleNamespace

    from freetoken.engine.engine import Engine
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.moe.offload_cache import OffloadMoeCache

    monkeypatch.setenv("FREETOKEN_GGUF_DOT_TIER", "avx2")

    sigs = ((18, 18, 23), (23, 23, 14), (18, 18, 14))  # dominant + two minorities
    layer_counts = (2, 1, 1)  # the weighted split source (real file's shape, scaled)
    E, H, I, bs, top_k = 8, 512, 256, 3, 4
    dev = torch.device("cuda")

    # uniform analytic banks: the merge comparison stays about the tier + pools,
    # not the fixture (see the serial-path test's rationale). One bank set per
    # partition slot, built ONCE and shared by both tier runs (identical data,
    # no per-run re-pack); each partition's layers get their own banks.
    def build_banks(sig, num_layers, seed):
        stub = iq._make_gguf_cache(sig, num_layers, E, H, I, seed=seed, uniform=True)
        return {role: list(per_layer) for role, per_layer in stub.bank_sources.items()}

    bank_sets = [
        build_banks(sig, num_layers, 101 + 13 * idx)
        for idx, (sig, num_layers) in enumerate(zip(sigs, layer_counts))
    ]

    def make_caches():
        caches = []
        for (sig, num_layers), banks in zip(zip(sigs, layer_counts), bank_sets):
            cache = OffloadMoeCache(
                num_layers=num_layers, num_experts=E, cache_size=2 * E, device=dev,
                quant_format="gguf", gguf_types=tuple([tuple(sig)] * num_layers),
                decode_target="hybrid", hybrid_max_fetch=1,
            )
            cache.set_bank_sources(banks)
            caches.append(cache)
        return caches

    config = SimpleNamespace(moe_cpu_threads=8, max_running_req=4, cuda_graph_max_bs=2)
    layers = [
        SimpleNamespace(
            top_k=top_k, activation="silu", apply_router_weight_on_input=False,
            quant_method=None, alpha=1.0, limit=10.0,
        )
    ]
    caches = make_caches()
    fake = SimpleNamespace(device=dev, cpu_moe_executors=[])
    Engine._init_cpu_moe_executors(fake, config, caches, layers)
    executors = fake.cpu_moe_executors
    assert len(executors) == 3

    # weighted pools: layer counts 2/1/1 at --moe-cpu-threads 8 give the dominant
    # partition 4 workers (not the even 3/3/2) as disjoint slices, every executor
    # on the avx2 W4A8-K tier
    pools = [ex.core_ids for ex in executors]
    flat = [c for pool in pools for c in pool]
    assert [ex.num_threads for ex in executors] == [4, 2, 2]
    assert len(flat) == len(set(flat)), f"weighted pools share cores: {pools}"
    for ex in executors:
        assert "avx2-w4a8k" in ex.isa, ex.isa

    def layer_for(c):
        layer = OffloadMoELayer.__new__(OffloadMoELayer)  # the branch only needs these attrs
        layer.quant_method = None
        layer.activation = "silu"
        layer.alpha = 1.0
        layer.limit = None
        layer.offload_cache = c
        layer.layer_id = 0
        return layer

    # shared routing: lane 0 pre-resident (the GPU hit), the rest CPU-owned
    gen = torch.Generator().manual_seed(5)
    hidden = (torch.randn(bs, H, generator=gen) * 0.5).to(torch.bfloat16).to(dev)
    raw = torch.stack(
        [torch.randperm(E, generator=gen)[:top_k] for _ in range(bs)]
    ).to(torch.int32).to(dev)
    raw[:, 0] = 0
    w = torch.rand(bs, top_k, generator=gen).to(dev)

    def run_merge(cache):
        hits = raw[:, :1].clone()
        cache.ensure_experts(0, hits)
        cache.copy_missing()
        ids = raw.clone()
        out = layer_for(cache)._decode_hybrid(cache, hidden, w, ids)
        torch.cuda.synchronize()
        cpu_lanes = ids < 0  # rewritten in place: slot (hit/fetched) or -1 (CPU-owned)
        assert cpu_lanes.any() and (~cpu_lanes).any(), "fixture must mix CPU misses and GPU hits"
        return ids, out.float()

    ids_avx, out_avx = run_merge(caches[0])
    assert torch.isfinite(out_avx).all()

    # the identical merge at the scalar tier (fresh caches, same banks + routing):
    # the GPU hit part is the same computation, so the merged outputs differ by
    # exactly the W4A8-K activation quantization on the CPU lanes
    monkeypatch.setenv("FREETOKEN_GGUF_DOT_TIER", "scalar")
    caches_sc = make_caches()
    fake_sc = SimpleNamespace(device=dev, cpu_moe_executors=[])
    Engine._init_cpu_moe_executors(fake_sc, config, caches_sc, layers)
    assert "avx2-w4a8k" not in fake_sc.cpu_moe_executors[0].isa
    ids_sc, out_sc = run_merge(caches_sc[0])
    assert torch.equal(ids_avx, ids_sc), "routing rewrite must be tier-independent"

    # tier A/B envelope over the CPU lanes' contribution: both tiers sit within
    # their dfloat bound of their own mirrored reference, and the references
    # differ by exactly the q8_K activation quantization (the A/B triangle).
    # The reference helpers are CPU-side math (fixture banks are host tensors).
    w_miss = torch.where(ids_avx < 0, w, w.new_zeros(())).contiguous()
    cpu_ids = torch.where(ids_avx < 0, raw, raw.new_full((), -1)).contiguous()
    hidden_c = hidden.cpu()
    w_miss_c = w_miss.cpu()
    cpu_ids_c = cpu_ids.cpu()
    ref_q8, extras_q8 = iq._reference_decode(
        bank_sets[0], sigs[0], 0, hidden_c, w_miss_c, cpu_ids_c, top_k, q8k_acts=True
    )
    ref_bf16, extras_bf16 = iq._reference_decode(
        bank_sets[0], sigs[0], 0, hidden_c, w_miss_c, cpu_ids_c, top_k
    )
    bound_q8 = iq._dfloat_bound(
        ref_q8, extras_q8, w_miss_c, cpu_ids_c, hidden_c,
        iq._DEQUANT_FACTOR[iq.GGML_IQ3_XXS], iq._DEQUANT_FACTOR[iq.GGML_IQ4_XS],
    )
    bound_bf16 = iq._dfloat_bound(
        ref_bf16, extras_bf16, w_miss_c, cpu_ids_c, hidden_c,
        iq._DEQUANT_FACTOR[iq.GGML_IQ3_XXS], iq._DEQUANT_FACTOR[iq.GGML_IQ4_XS],
    )
    act_gap = (ref_q8 - ref_bf16).abs()
    out_avx_c, out_sc_c = out_avx.cpu(), out_sc.cpu()
    envelope = bound_q8 + bound_bf16 + act_gap + 1e-3 + 2.0**-8 * out_avx_c.abs()
    d = (out_avx_c - out_sc_c).abs()
    assert (d <= envelope).all(), (
        f"avx2-tier merge diverges from the scalar merge beyond the W4A8-K envelope: "
        f"max {d.max():.6g} vs {envelope.max():.6g}"
    )
