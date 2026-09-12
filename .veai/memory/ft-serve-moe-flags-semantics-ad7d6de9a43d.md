---
name: "ft-serve-moe-flags-semantics"
description: "ft serve MoE flags: moe-cpu-threads dead for offload; benchbw stale pre-iommu; fast_index_copy path"
type: project
lastUpdated: 2026-09-12T22:17
---

# ft serve MoE flag semantics (code audit, 2026-09-12, file:line verified)

## --moe-cpu-threads
- DEAD for --moe-strategy offload: CpuMoeExecutor built only when decode_target is cpu/hybrid (engine.py:686-688). Default 0 = auto per physical core with pinning (config.py:51, cpu_executor.py:119-140). Offload decode = ensure_experts LRU (offload_cache.py:843-853) -> copy_missing -> GPU GEMM (layers/moe.py:196-230); no CPU GEMV.
- Hybrid (_decode_hybrid, layers/moe.py:233-281): capped fetch + CPU GEMV (decode_submit) overlapped with PCIe fetch + GPU GEMM; overlap toggle env FREETOKEN_HYBRID_OVERLAP=0 (layers/moe.py:22-24, 258-265).

## Offload transfer paths
- Decode-miss: fast_index_copy_multi_jit ZERO-COPY kernel reads pinned host banks over PCIe directly (kernel/fast_index_copy.py:153-187); docstring constant "~31 GB/s measured" is PRE-iommu=pt (now ~46+; re-bench pending). NOT cudaMemcpyAsync.
- Prefill: full-layer pinned buffer.copy_ non_blocking (offload_cache.py:678-684) + hit-D2D cudaMemcpyBatchAsync CUDA>=13 (offload_cache.py:880-900) -> prefill wins from iommu=pt (25->46 GB/s) directly.

## --disable-moe-prefill-overlap
- Applies to whole offload family (shared _prefill_routed, layers/moe.py:346-380). Dropping overlap lowers slot floor 2E->E (offload_cache.py:163-167, cache_budget.py:72-74) -> LOAD-BEARING for 1M nvfp4 KV plan (frees ~4.15 GiB). Does not affect decode.

## benchbw profile (hybrid auto fetch fraction)
- Key = GPU UUID only (bench_profile.py:45-49, validation :120-135); NO TTL/hw/thread fingerprint -> silently stale forever.
- Contents: ceilings (CPU STREAM, linear pinned H2D/D2H, benchbw.py:276-300/231-274) + production kernels: pcie_gather = copy_missing (559-590), cpu_moe GEMV bs=1, overlap pair (:594-660).
- Fraction = pcie_ov/(pcie_ov+cpu_ov) (bench_profile.py:156-191), used only when --moe-hybrid-max-fetch=-1 and decode_target=hybrid (engine.py:692-716). Refresh: `ft bench bw` (atomic overwrite, benchbw.py:725-727) or delete ~/.cache/freetoken/benchbw/<gpu-uuid>.json; FREETOKEN_BENCHBW_PATH env override (:113).
- TRAP: all pre-2026-09-12 profiles measured with IOMMU Translated (PCIe ~25/24 GB/s); after iommu=pt (46/57) they understate PCIe -> hybrid auto fetch fraction too low. User's profile dated Sep 2.

## Assorted verified
- --expert-load ignored for FTW checkpoints (expert_banks.py:314-322 early return).
- No assert offload+nvfp4: moe_strategy and kv_quant orthogonal; DSA pool family check passes (engine.py:1446-1465).
- Doc rot: config.py:52 "Ignored by other backends" is stale (hybrid uses moe_cpu_threads); --kv-cache-dtype nvfp4 help (args.py:433-447) predates #408 MLA/DSA support.

