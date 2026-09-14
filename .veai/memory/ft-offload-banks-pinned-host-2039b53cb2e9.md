---
name: "ft-offload-banks-pinned-host"
description: "OffloadMoeCache gather needs pinned host banks; tracker note-count trap; IMA resolved on hardware"
type: project
lastUpdated: 2026-09-14T02:03
lastRecall: 2026-09-14T18:54
---

# OffloadMoECache gather requires pinned host banks; raw mmap VAs are illegal

Discovered in Phase 5 of the glm5next GGUF work (2026-09-13, both reviews flagged it independently; the green GPU smoke masked it by passing .cuda() sources).

- The offload zero-copy gather path (set_bank_sources -> _build_fused_copy_plan -> device_ptr(source) -> fast_index_copy_multi_jit / _prefetch_split) requires host banks that are PINNED and device-mapped (contract documented at kernel/pinned.py:1-5).
- Raw gguf mmap-backed views are pageable: on Linux UVA device_ptr() returns data_ptr() unchecked - the file-backed VA - and the plan then records a non-device-visible pointer. Result: illegal memory access or garbage at the FIRST ensure/copy, i.e. after the entire multi-minute load, not at load time.
- Correct pattern (the q4_0/gemma4 precedent, gemma4/gguf.py:440-443): materialize banks into allocated pinned HostBanks via alloc_layer_banks + PinPipeline (pin-after-fill). Do NOT cudaHostRegister the mmap VA: GGUF tensor data is 32B-aligned, not page-aligned, and the file is 147 GB.
- Second trap in the same area: set_bank_sources asserts a uniform bank shape across ALL layers. Real checkpoints have cross-layer width heterogeneity (the GLM file: 3 signatures (18,18,23)x39 / (18,18,14)x2 / (23,23,14)x1) -> the unified slot cache needs partition-by-width-signature (or per-layer storage). An off-by-role bug is also easy here: gguf_types must be assembled in explicit role order ("gate","up","down"), never slots.values() (file encounter order).

Why: the failure mode fires after a long successful load, on the first decode miss - the worst possible debugging profile; and CUDA smoke tests that hand .cuda() tensors to the cache bypass the pinned path entirely.
How to apply: any new expert-bank source (new quant family, new file format) materializing banks for the offload cache - pin via the q4_0 pattern, drive the REAL OffloadMoELayer._expert_gemm path in tests with heterogeneous per-layer shapes, and never validate with .cuda() shortcuts.

Status 2026-09-14: the pinned pattern LANDED in the gguf loader (per-layer HostBanks + LayerCompletionTracker + PinPipeline, one-time ~125 GiB copy; no cudaHostRegister) - yet the partitioned fused gather STILL hit a CUDA IMA on the real boot (runs 3-4, reproduced byte-for-byte on the current tree): first-bad-launch = fast_index_copy_multi_jit in OffloadMoeCache.copy_missing (offload_cache.py:1057; csrc check utils.cuh:113), crashing at bank 0 where global==local - the layer_id-leak hypothesis is FALSIFIED; lru_ensure completed before the copy. Eager boot (--cuda-graph-max-bs 0) reaches /ready 200 - the IMA is in the copy path during the first eager warmup forward, NOT capture-specific. Suspects: partitioned sub-source aliases, _build_fused_copy_plan descriptor geometry, dst slot bounds. Live state and the sanitizer pass: ft-serve-gguf-glm5next-unsupported.

RESOLVED 2026-09-14 (hardware-verified, boot-smoke run 5): the partitioned-gather IMA was NOT the aliases themselves - LayerCompletionTracker.expected_per_layer=3 vs ONE combined tracker.note(bank) per layer in the gguf fill loop (gemma4 semantics = NOTE-COUNT-matching: gemma4 notes 2 per layer, one per write). The completion hook never fired, PinPipeline pinned nothing, all 126 HostBanks stayed pageable through the whole load; IMA at the first decode copy (worst debugging profile). Fix: expected_per_layer=1 for glm5next + a permanent construction-time guard in OffloadMoeCache._build_fused_copy_plan rejecting (a) pageable CPU banks, (b) per-layer row-width != plan feat, (c) slot-pool geometry != cache_size x feat - this class now fails at construction for any future unpinned source. Hardware: boot 2m40s, 3 graphs captured, copy_missing clean, prefill 4213 tok + decode 14-16 tok/s on the real 147 GB file. TRAP for any future bank source: expected_per_layer must equal the per-layer NOTE count, else pinning silently skips entirely (invisible to CPU tests - pageable-vs-pinned only exists on a CUDA device).
