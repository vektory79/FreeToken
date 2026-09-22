# Task 02: Partition-aware hybrid wiring (per-cache CPU executors) - W4

Type: implementation (engine + cache + layer wiring)
Suggested agent: Code (strong)
Spec coverage: R2, S2. Depends on Task 01 (executor must accept the gguf
formats first).

## Goal

Hybrid works with per-signature partitions: each OffloadMoeCache partition
(3 groups for the real file) gets its own CPU executor instance with its own
per-layer per-role types; the engine multi-partition ValueError for cpu/hybrid
is lifted exactly when every partition's types are executor-capable.

## Why This Task Exists

engine.py raises ValueError for multi-signature partitions with cpu/hybrid
(cited as engine.py:881-885 and :946-949 in the research map) - the real file
has 3 signature groups, so hybrid cannot boot on it without this wiring.

## Required Inputs (read first)

1. .tasks/gguf-glm5next-hybrid/research/hybrid-machinery-mapping.md - sections
   1 (hybrid machinery), 3 (partition wiring), 5 (tests).
2. .tasks/gguf-glm5next-hybrid/research/w1-cpu-gemv-mapping.md - sections 4-5
   (hybrid call chain, bank exposure) and the blocker list.
3. .tasks/gguf-glm5next-hybrid/TASK.md - Lessons 1, 4, 5, 6.
4. Task 01's landed code (per-projection executor API).

## Files/Areas (verified line refs)

- python/freetoken/engine.py: partition loop :829-879 (one OffloadMoeCache per
  signature group: local gguf_types, set_bank_sources, _slice_alphas, local
  cpu_layer_ids remap); _init_cpu_moe_executor :941-976 (currently ONE executor
  bound to caches[0]; num_threads=config.moe_cpu_threads, 0=auto + pinned per
  physical core, coordinator reserves a core); multi-partition ValueError
  :946-949 (also referenced at :881-885); self.cpu_moe_executor single-pointer
  readers; _resolve_hybrid_fetch called :873-874; _route_offload_layers
  :550-557 (sets layer.offload_cache + local layer_id); _partition_prefill_overlap
  :540-548 (>=2E degrade - MUST NOT regress).
- python/freetoken/moe/offload_cache.py: ensure_experts :888,
  ensure_experts_hybrid :900-920 (rewrites ids to slot/-1), copy_missing :1056,
  fields hybrid_max_fetch :148 / hybrid_fetch_fraction / gguf_types :159.
- python/freetoken/layers/moe.py: _decode_hybrid :299-339 (clone ids ->
  ensure_experts_hybrid -> executor.decode_submit:317 BEFORE copy_missing +
  GPU _expert_gemm:323-331 -> decode_sync -> gpu+cpu merge); _HYBRID_OVERLAP
  env toggle :26; gguf branch of _expert_gemm :429-459 (per-projection
  ggml_moe_a8_vec with cache.gguf_types[layer]:433).

## Design points (from research)

- Thread budget: split the configured CPU thread budget across the per-cache
  executor pools (coordinator reserves a core per pool today); 0=auto must
  still work sensibly with 3 pools.
- self.cpu_moe_executor is a single pointer read in several places - route
  decode via layer.offload_cache (already set per layer by
  _route_offload_layers) or introduce a per-layer executor resolution that
  keeps pre-capture construction timing intact.
- Keep the _HYBRID_OVERLAP env toggle functional.
- Keep the ValueError for the genuinely-incapable case (some partition has a
  type the executor cannot run) - only lift it when ALL partitions are capable.
- Triage delta (2026-09-15): per-partition executor INIT - today the engine binds ONE executor to caches[0] (engine.py:901-903); instantiate per cache. Per-layer gguf_types resolution (mirror layers/moe.py:433 per-layer indexing; Task 01 reads gguf_types[0] under the uniformity assert added in the fix wave). Add a mixed-cache executor test (mixed signatures (18,18,14)/(23,23,14) alongside (18,18,23)) - blocked today by the ValueError, unlocked by this task.

## Constraints / Non-Goals

- Do not change offload-only behavior (it is production-usable; suites green).
- Do not regress the per-partition prefill-overlap degrade (>=2E rule).
- No benchbw changes here (Task 03), no engine gate changes here (Task 04).
- One change per commit; conventional commits; Assisted-by: Veai. No push.

## What to Do

1. Read the inputs; verify the cited lines still match (line drift possible
   after Task 01).
2. Implement per-cache executor instances with per-layer per-role formats;
   thread budget split; decode routing via layer.offload_cache.
3. Lift the multi-partition ValueError for cpu/hybrid conditioned on executor
   capability of every partition's types; keep the incapable-case rejection
   with a clear message.
4. Tests:
   - Extend tests/moe/test_hybrid_fetch.py (6 tests today) with a
     multi-partition hybrid case (S2): a layer whose cache has CPU-miss +
     GPU-hit mix - CPU output for missed experts matches the GPU path for the
     same rows.
   - tests/models/test_gguf_expert_banks.py partition tests (:529/:546/:645/
     :660/:744/:799) stay green (note: the file now has 30 test defs).
   - Engine tests around the lifted ValueError updated to the new contract.
5. Run: tests/moe + tests/models/test_gguf_expert_banks.py + tests/engine
   (record exact outputs).
6. Commit (conventional, Assisted-by: Veai).

## Expected Output (report_result)

- Summary of the executor-instance/ownership design chosen.
- Exact changed files; test commands + exact outputs; blockers.

## Acceptance Criteria

- [ ] Multi-partition hybrid no longer raises when all partitions are
      CPU-capable; incapable case still fail-fasts with a clear message
- [ ] Per-cache executors carry correct per-layer per-role types
- [ ] S2 hybrid dispatch test passes (CPU-miss output == GPU path output for
      the same rows)
- [ ] Existing hybrid/offload/partition suites green
- [ ] I've created a git commit for this task
