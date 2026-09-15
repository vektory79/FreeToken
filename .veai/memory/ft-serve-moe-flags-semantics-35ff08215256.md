---
name: "ft-serve-moe-flags-semantics"
description: "ft serve MoE flags: cpu-threads per-partition split (aad5d3a); eb7de4c clamp/help fixes; fetch fractions"
type: project
lastUpdated: 2026-09-15T03:05
lastRecall: 2026-09-15T17:10
---

# ft serve MoE flag semantics (code audit, 2026-09-12, file:line verified)

## --moe-cpu-threads
- DEAD for --moe-strategy offload: CpuMoeExecutor built only when decode_target is cpu/hybrid (engine.py:686-688). Default 0 = auto per physical core with pinning (config.py:51, cpu_executor.py:119-140). Offload decode = ensure_experts LRU (offload_cache.py:843-853) -> copy_missing -> GPU GEMM (layers/moe.py:196-230); no CPU GEMV.
- Hybrid (_decode_hybrid, layers/moe.py:263-314): CPU decode_submit kicked off BEFORE the PCIe fetch + GPU GEMM; layer completes at max(cpu, pcie+gemm); overlap toggle env FREETOKEN_HYBRID_OVERLAP=0 serializes the two (A/B only; layers/moe.py:22-24).

## Offload transfer paths
- Decode-miss: fast_index_copy_multi_jit ZERO-COPY kernel reads pinned host banks over PCIe directly (kernel/fast_index_copy.py:153-187); docstring constant "~31 GB/s measured" is PRE-iommu=pt (now ~46+; re-bench pending). NOT cudaMemcpyAsync.
- Prefill: full-layer pinned buffer.copy_ non_blocking (offload_cache.py:678-684) + hit-D2D cudaMemcpyBatchAsync CUDA>=13 (offload_cache.py:880-900) -> prefill wins from iommu=pt (25->46 GB/s) directly.

## --disable-moe-prefill-overlap
- Applies to whole offload family (shared _prefill_routed, layers/moe.py:346-380). Dropping overlap lowers slot floor 2E->E (offload_cache.py:163-167, cache_budget.py:72-74) -> LOAD-BEARING for 1M nvfp4 KV plan (frees ~4.15 GiB). Does not affect decode.

## benchbw profile (hybrid auto fetch fraction) - v4, refreshed post-iommu=pt
- This box's profile: 2026-09-12 22:26, version 4, post-iommu=pt. Ceilings: cpu_stream_read 76.36, pcie H2D 45.15, D2H 57.03 GB/s; benched with threads_used=20 (20 physical cores).
- v4 structure: per-dtype entries under `dtype_kernels{fmt:{expert_bytes, synth_experts, cpu_moe_gbs, cpu_moe_isa, pcie_gather_gbs, cpu_moe_overlap_gbs, pcie_gather_overlap_gbs, ratio, recommended}}`. Top-level `workloads: {}` is legacy-empty and BENIGN - the fraction source is dtype_kernels.
- nvfp4 entry: cpu_moe_gbs 71.18 (isa avx2+vnni nvfp4-w4a8), pcie_gather_gbs 43.21, overlap pair cpu 56.25 / pcie 18.8 -> hybrid fetch fraction = pcie_ov/(pcie_ov+cpu_ov) = 25.0% (engine log: "fetching 25.0% of each decode step's expert misses over PCIe"). Old pre-iommu profile gave ~19.7%.
- load_hybrid_fetch_fraction (bench_profile.py:156-191): prefers dtype_kernels[fmt] overlapped pair, falls back to standalone pcie/cpu, then per-model workloads; None -> cache.hybrid_max_fetch=1 + warning "no usable ft bench bw profile". Engine applies it only when --moe-hybrid-max-fetch=-1 (default) and decode_target=hybrid (engine.py:692-716); explicit >=0 = fixed cap.
- Key = GPU UUID only; NO TTL/fingerprint -> silently stale forever. Refresh: `ft bench bw` (atomic overwrite) or FREETOKEN_BENCHBW_PATH override.
- Thread-neutrality resolved: profile's CPU GEMV benched at 20 threads vs serving at 16 was suspected mistuned; the 16-vs-20 serving A/B (2026-09-12) is FLAT (decode 15.13 vs 15.29 tok/s, noise) -> the 20t profile is fine at 16t; measured numbers in glm53-post-iommu-baseline.

## Assorted verified
- --expert-load ignored for FTW checkpoints (expert_banks.py:314-322 early return).
- No assert offload+nvfp4: moe_strategy and kv_quant orthogonal; DSA pool family check passes (engine.py:1446-1465).
- CPU executor boot log carries the serving geometry: "CPU MoE executor ready: threads=16 (pinned to cores 0..23) isa=avx2+vnni(nvfp4-w4a8) fmt=nvfp4 H=4096 I=2048 experts=288 layers=42 top_k=8" -> GLM hybrid decode working set = 42x8 = 336 pairs (NOT 43x8=344; layer 45 is the MTP layer, not a serving MoE layer).
- --moe-prefill-hit-d2d (store_true, default False): help states "Effective with --moe-cache-size > 2 * num_experts" = 576 slots for 288 experts -> unusable with large-KV auto plans (only 347-480 slots).
- Doc rot: config.py:52 "Ignored by other backends" is stale (hybrid uses moe_cpu_threads); --kv-cache-dtype nvfp4 help (args.py:433-447) predates #408 MLA/DSA support.

## 2026-09-15 (hybrid campaign Task 02, commit aad5d3a): per-partition cpu-threads split
Engine._init_cpu_moe_executors builds ONE CpuMoeExecutor per OffloadMoeCache partition; resolve_pool_affinities(num_pools, requested) returns DISJOINT core sets (explicit N splits evenly, remainder to earlier pools, each share floored at 1 worker; 0=auto splits physical cores evenly, coordinator core carved out per pool inside the executor). Single-cache path byte-identical (whole-machine pool). The multi-partition cpu/hybrid ValueError is lifted iff every partition passes partition_executor_rejection (capability: _WFMT_IDS; "gguf" alias requires all layer type ids in _GGUF_TYPE_FMTS). Known nit: explicit --moe-cpu-threads > total cores wraps and can overlap pool cores (extreme misconfig, perf-only).

## 2026-09-15 correction (commit eb7de4c, Task 07 hardening)
The wrapped-overlap nit above is FIXED: resolve_pool_affinities now clamps the explicit total to the usable core set (one warning, floor-at-one, never wraps across pools); resolve_threads_and_affinity(0, core-free subset) returns max(1, len(core_ids)). The config.py/args.py help-text rot is also fixed: --moe-cpu-threads help now states the cpu/hybrid/offload+--moe-cpu-layers usage and the per-partition disjoint split. Watchdog errors carry pool attribution ("[pool i/n: ...]"). Gate advice: offload + --moe-cpu-layers failures advise "adjust or drop --moe-cpu-layers" (not "use offload instead").
