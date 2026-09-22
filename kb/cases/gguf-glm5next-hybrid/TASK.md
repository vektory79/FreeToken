# Task: --moe-strategy hybrid for glm5next GGUF (CPU GEMV for IQ/K banks)

Status: CAMPAIGN COMPLETE (2026-09-15): Tasks 01-08 CLOSED. 9 commits (5a04423, bd02232, 427a431, 2169baa, aad5d3a, f6b94ad, ef83ee8, eb7de4c, 63b9bff + 1635ecd). Final re-acceptance on the real file: hybrid 16.67 tok/s @64k vs offload 12.97 = 1.28x, all three gates PASS, battery 24/24. HYBRID IS NOW THE RECOMMENDED STRATEGY for glm5next GGUF on this rig. The mid-campaign NO-GO was reversed by the NVFP4 A/B discovery (per-layer 'handshake floor' was mis-attributed fetch volume) + the ggml integer-kernel port validated by microbench (>=40 GB/s gate passed at 53-66).
Composed 2026-09-14 at the close of the offload-only campaign
(.tasks/gguf-glm5next-path-a/PLAN.md - read it FIRST; this task assumes every lesson
there as ground truth and only lists the hybrid-specific DELTA).

## Goal

`ft serve --model-path .../GLM-5.3-Flash-UD-Q3_K_XL.gguf --moe-strategy hybrid
--moe-cpu-threads <N>` boots and serves with the hybrid decode path (LRU ensure +
capped PCIe fetch OVERLAPPED with CPU GEMV on misses) for the gguf bank formats.
The offload-only path is DONE and committed (028f2d9 + 78115df); hybrid is the
natural next milestone for lower VRAM pressure and higher decode at deep contexts.

## Current state (verified this session - do NOT re-derive)

- ENGINE GATE BLOCKS IT: engine.py _adjust_config rejects moe_weight_format == "gguf"
  with strategy cpu/hybrid/fused (message: "gguf moe_weight_format supports offload
  only; drop the flag"). This gate was ADDED by the offload-first campaign (Phase 5,
  F1) because the CPU executor had no gguf formats - it is a feature gate, not a bug.
- CPU GEMV FORMATS (moe/cpu_executor.py _WFMT_IDS :72): bf16 / nvfp4 (W4A8 via
  cpu_moe_ext.cpp:1120) / mxfp4_triton / ds_fp4 / q4_0 (q4_0 resolver :407-432).
  MISSING: IQ3_XXS(18) / IQ4_XS(23) / Q6_K(14) / Q3_K(11) / Q4_K(12) - the exact
  formats in the real GLM file.
- THE REAL FILE's per-layer bank types: gate/up IQ3_XXS (IQ4_XS on ckpt 11),
  down IQ4_XS (Q6_K on ckpt 11,12,44). H=4096, I=2048, E=288, top_k=8.
- OFFLOAD SIDE (done, committed): pinned HostBanks (note-count trap fixed),
  per-signature partitions (3 groups: (18,18,23)x39, (18,18,14)x2, (23,23,14)x1),
  _expert_gemm gguf branch with per-projection dispatch, #186 clamp (8191 at top_k 8),
  benchbw "gguf" profile row (DTYPE_WORKLOADS + _offload_bank_specs; the CPU-MoE
  section auto-skips via _CPU_MOE_FORMATS -> verdict always "offload").
- BOUNDARY CHOREOGRAPHY: cross-group prefetch hop is a PROVEN no-op (all partitions
  overlap-degraded in the tuned plan: slots < 2E) - see Wave C skip verdict in
  verification/phase6/. Hybrid may change this (different budget solve), but the
  minorities' 2E floor is the dominant constraint (see ft-serve-prefill-overlap-
  512k-infeasible memory: 2E floors are infeasible at big KV on this card).

## Work items (ordered by dependency)

### W1. CPU-side dequant/GEMV for the gguf bank formats
- cpu_moe_ext.cpp (the C++ CPU executor): add scalar/LUT GEMV for IQ3_XXS, IQ4_XS,
  Q6_K (minimum for the real file; Q3_K/Q4_K optional - they appear only in the
  MTP block, which is skipped). Reference dequant logic: the vendored
  kernel/csrc/gguf/dequantize.cuh (the CUDA paths are the source of truth for the
  block layouts; the CPU path must produce the SAME values within the dfloat
  contract - see the Phase 4 parity tests in tests/kernels/test_gguf_quant.py
  for the accepted tolerance model: per-element bound 8*2^-11*factor*|d| + 1e-3).
- Alternative (SLOWER, but a valid first step): python-side dequant via gguf-py
  reference (quants.py IQ3_XXS/IQ4_XS/Q6_K dequantize_blocks) feeding bf16 CPU
  GEMV - proves the plumbing end-to-end before optimizing the C++ path. The
  offline recompute in the Phase 6 bisect already proved gguf-py dequant matches
  the CUDA kernel to cos 0.99994-0.99997.
- Wire the new formats into cpu_executor.py _WFMT_IDS + a resolver (mirror the
  q4_0 resolver :407-432 pattern: accept the bank's GgufTensor, produce the CPU
  GEMV output for the requested expert rows).
- PER-PROJECTION types: the cache carries gguf_types per (gate,up,down) - the CPU
  executor must handle the three projections having DIFFERENT types in the SAME
  layer (the real file: gate IQ3_XXS + down IQ4_XS in one layer).

### W2. benchbw CPU-MoE section for gguf formats
- _CPU_MOE_FORMATS must include the gguf types so the benchbw CPU-MoE section
  stops auto-skipping and computes a fetch fraction (the auto fetch fraction
  feeds load_hybrid_fetch_fraction -> --moe-hybrid-max-fetch auto resolves).
- The existing "gguf" profile row (benchbw.py :146-149 + _offload_bank_specs
  :308-315) already has the geometry - extend it with the CPU timing leg.
- Validate: the fetch fraction is sane (not 0%, not 100%) and the hybrid verdict
  is "hybrid" not "offload".

### W3. Engine gate relaxation
- Change the _adjust_config rejection: allow hybrid when the CPU executor can
  handle ALL the file's per-layer bank types (query the resolver); keep the
  rejection for cpu strategy (no CPU-resident gguf banks - offload cache still
  owns them) and for fused (gguf banks are packed, not resident bf16).
- Keep the #186 clamp (8191 at top_k 8) - it is strategy-independent.

### W4. Partition-aware hybrid
- The per-signature partitions (3 caches) each need CPU executor instances with
  their OWN per-layer types (the offload cache already routes per-projection;
  the CPU path must do the same). The LRU/fetch/residency state is per-cache -
  the CPU executor should be per-cache too (mirror the offload partition wiring).
- The _route_offload_layers attach already sets layer.offload_cache + local
  layer_id - the hybrid CPU path should use the same routing.

### W5. Tests
- CPU dequant parity vs gguf-py reference (the offline recompute chain from the
  Phase 6 bisect is the template; extend to a unit test).
- _expert_gemm hybrid dispatch: a layer whose cache has CPU-miss + GPU-hit mix -
  verify the CPU output for missed experts matches the GPU path for the same rows.
- benchbw: the "gguf" profile produces a hybrid verdict with a sane fetch fraction.
- Engine gate: hybrid accepted for glm5next gguf (the gate test needs flipping).
- Full regression: all existing suites (25/102/21/146/58-59) must stay green.

### W6. Hardware acceptance
- Boot with `--moe-strategy hybrid --moe-cpu-threads <N>` on the real file.
- Measure: decode tok/s at 64k fill (compare vs offload-only 14.3-15.6 tok/s;
  the NVFP4 hybrid baseline is 14.1-15.5 tok/s - the GGUF hybrid target is
  parity-or-better vs the GGUF offload number, not vs NVFP4).
- Check: the fetch fraction log line, CPU utilization during decode misses,
  VRAM delta vs offload-only (hybrid frees the GPU slot budget for KV or slots).
- Battery: re-run the 24-prompt quality battery (must stay 20+/24 on-topic).

## Lessons from this session (apply ALL of them)

1. LayerCompletionTracker note-count trap: expected_per_layer MUST equal the
   per-layer NOTE count (glm5next 1/layer, gemma4 2/write) - mismatch silently
   skips ALL pinning -> IMA at the first decode copy. If hybrid adds a new bank
   source path, re-audit the tracker contract.
2. Non-contiguous activations: ggml GEMM kernels assume row-major; f_a/g_a from
   the in_proj split are views with stride (24896,1). The fix (normalization in
   fused_mul_mat_gguf) landed in 028f2d9 - any new GEMM call site must inherit it.
3. Pin-count contract: the fused gather's device aliases require PINNED host
   banks (pinned.py:1-5 contract). Raw mmap VAs are illegal (pageable on Linux
   UVA). The HostBank materialization in glm5_next/gguf.py handles this - do not
   regress it.
4. per-role geometry: down rows = H (4096), gate/up rows = I (2048) - never reuse
   one row count for all roles.
5. Role-ordered gguf_types: always explicit ("gate","up","down"), never
   slots.values() (file encounter order).
6. The 2E prefill-overlap floor: minorities floored at 1E cannot reach 2E on
   32 GiB - the degrade rule (Engine._partition_prefill_overlap) is the accepted
   solution; hybrid must not regress it.
7. The dfloat precision contract: bit-parity vs fp32 gguf-py is impossible for
   the half-chain kernels; use the per-element bound 8*2^-11*factor*|d| + 1e-3
   (_DEQUANT_FACTOR per format).
8. compute-sanitizer + CUDA_LAUNCH_BLOCKING are the IMA debugging tools (the
   bisect in verification/phase6/bisect/ROOT-CAUSE.md is the template).
9. The boot smoke recipe (watchdog FAILPAT, /ready gate, model field from
   /v1/models, SIGTERM exit 143) is in ft-serve-test-and-e2e-gotchas - reuse it.
10. The tuned serve command: base + --num-tokens 70016 --memory-ratio 0.85
    --max-prefill-length 4096 (explicit --moe-cache-size 864 fail-fasts on the
    per-signature floors; auto slots + capped KV + ratio 0.85 is the working lever).

## Constraints

- PRIVATE-USE work (upstream #34 gate waived by the user - see
  ft-gguf-glm5next-private-scope). No pushes/PRs by agents.
- One change per commit, conventional commits, Assisted-by: Veai.
- The vendored .cu files are verbatim sgl-kernel ports - the CPU-side dequant
  lives in cpu_moe_ext.cpp (NOT the .cu files); if a .cu change is unavoidable,
  follow the vendor discipline (check sgl-kernel/llama.cpp upstream first).
- IQ3_XXS python dequant: models/gguf/dequant.py has NO iq3xxs reference (the
  BLOCK_SHAPE is geometry-only). The venv gguf-py quants.py HAS the reference
  (proven in the Phase 6 bisect). Do not port it into dequant.py unless needed
  for a test - the hot path is the C++ CPU GEMV, not python.

## Acceptance criteria

- `--moe-strategy hybrid --moe-cpu-threads <N>` boots the real file and serves.
- Decode @64k fill >= the offload-only number (14.3 tok/s) - the hybrid value
  proposition is better decode at deep contexts (less PCIe traffic per miss).
- The benchbw verdict for "gguf" is "hybrid" with a sane fetch fraction.
- All existing suites green + the new tests.
- The 24-prompt battery stays 20+/24 on-topic.