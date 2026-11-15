---
name: "ft-decode-stats-instruments-unwired"
description: "T1 decode driver traps: replay bypasses host code, kineto aborts, Engine singular moe_offload_cache, FREETOKEN_DR_ROUTES"
type: project
lastUpdated: 2026-11-15T14:22
lastRecall: 2026-11-15T14:20
---

# Decode-stats instrumentation: unwired in code, T1 harness driver now exists (2026-09-26)

## Code facts (unchanged)
- `moe_collect_stats` — EngineConfig field (engine/config.py:50) WITHOUT a CLI flag; wired at engine.py:1033 `cache.collect_stats = config.moe_collect_stats` (must be True BEFORE CUDA-graph capture so the device-side stat += ops are captured).
- `collect_decode_freq` (offload_cache.py:262) is never set True in-repo; `decode_routing_stats()` (oracle_hit/working set/entropy) and `decode_miss_stats_per_layer()` have no callers in the repo.
- Stat tensors are allocated UNCONDITIONALLY in OffloadMoeCache.__init__ — the flags only gate recording; flipping them post-init but pre-capture is safe.
- The decode_freq histogram is ONLY accurate WITHOUT CUDA graphs: the host-side scatter_add in ensure_experts does not re-run on graph replay. Eager decode = `--cuda-graph-max-bs 0`.
- The ServerArgs banner prints moe_collect_stats=False even when a later hook sets it — never read the banner as "stats off".

## T1 harness driver EXISTS (2026-09-26)
- `.tasks/decode-research/driver/sitecustomize.py` — inert unless env FREETOKEN_DECODE_RESEARCH=<outdir> is set (runner passes it + PYTHONPATH=<driver dir> + FREETOKEN_DR_TAG to the ft serve tree); mirrors to kb/harness/decode-research/driver/ (edits go to .tasks original, then byte-copy).
- `.tasks/decode-research/analyze_trace.py` — decomposes chrome traces: GPU busy/idle, per-class kernels, top kernels/cpu ops.
- Plain boots are unaffected (env-gated sitecustomize).

## TRAP: trace window keyed on layer-calls can silently never fire
First instrumented run produced an EMPTY outdir: that boot decoded only 467 tokens x 42 = 19,614 layer-calls, never reaching the default START. No error is raised. Fixed: START=42*150, END=42*160 plus a "[decode-research] driver armed" banner. Decode length is boot-to-boot nondeterministic under temp0 (467..763 tokens) — key windows LOW and verify the outdir is non-empty before trusting a run.

**Why:** every decode-stats wave (R3 routing oracle, per-layer misses, the 44 ms GPU_chain+sync decomposition) depends on this driver; the routing oracle additionally requires the eager (graphs-off) run.

**How to apply:** hybrid trace + miss stats = instrumented graphs-on run; routing oracle = separate short eager run with `--cuda-graph-max-bs 0`; always confirm the armed banner and non-empty outdir.

## TRAP 2: CUDA-graph replay bypasses ALL model host code per decode step (2026-09-26)
Decode steps are `GraphRunner.replay()` calls: model Python (incl. `OffloadMoELayer._decode_routed`) runs only during capture/warmup — NEVER per replayed step. A layer-call-keyed window can never fire in graphs-on decode. **Fix:** hook `GraphRunner.replay` (one call = one decode step) for graphs-on runs; `_decode_routed` serves ONLY the eager run. (Also why collect_stats works under graphs — its += ops are baked in — while the freq histogram does not.)

## TRAP 3: torch.profiler start inside the scheduler NATIVELY ABORTS the process (2026-09-26)
Kineto CPU+CUDA window at decode step 150 killed the scheduler outright (C++ stack, "backend worker exited", no Python exception; consistent with ncu ERR_NVGPUCTRPERM on this box). **Fix (driver v3):** profiler gated behind `FREETOKEN_DR_PROF=1` (default OFF); CUDA-event pairs per step; stats dump at window end UNCONDITIONAL — never gate the dump on profiler success.

## Driver v3 summary
- Hooks: `Engine._init_offload_moe_cache`, `GraphRunner.replay` (steps 150..160 via FREETOKEN_DR_SSTART/SEND), `OffloadMoELayer._decode_routed` (eager fallback, layer-calls 6300..6720).
- Outputs: `stats_<tag>.json` always (aggregate + per_layer + routing + step_spans_ms); `trace_<tag>.json` only with PROF=1. Markers: "driver armed" (x4 processes), "events-only window OPEN", "window CLOSED", "DUMPED stats".
- mean(step_spans_ms) vs wall ms/step = GPU-stream-engaged vs host/idle split; routing only valid in the eager arm.

## TRAP 4: Engine keeps the cache SINGULAR; the driver read a plural attr that never exists (2026-11-15, offline-coverage-curve, cost 4 boots)
`Engine` has `moe_offload_cache` (singular, engine.py:1062); the plural `moe_offload_caches` list lives on GraphRunner (engine/graph.py:159-164). The driver's `_dump` AND `_patched_init` read `getattr(eng, "moe_offload_caches", [])` -> empty: stats files carried `"caches": []`, and collect_stats/collect_decode_freq were NEVER flipped (one boot produced a stats file with a cache entry but ZERO counters). The 2026-09-26 campaign never noticed because its stats files were not retained. **Fix:** `_engine_caches(eng)` fallback (plural -> singular, same shape as kvcache/cache_status.py `_moe_offload_caches`) used everywhere, mirror updated in kb.

## TRAP 5: module-level armor makes arm failures SILENT; helper processes clobber stats (2026-11-15)
- The arm block's try/except swallows exceptions: an import-time NameError (`atexit.register(_dump)` written ABOVE `def _dump`) left the driver INERT for a whole boot with zero log output. **Fixes:** `atexit.register(lambda: _dump())` (name resolves at call time), and the except prints `[decode-research] driver INERT: %r`. ALWAYS after driver edits: run `PYTHONPATH=<driver dir> FREETOKEN_DECODE_RESEARCH=/tmp/x python -c pass` and require the "driver armed" line; also grep each serve log for it.
- Tokenizer/helper processes arm the driver too; their atexit `_dump()` rewrote the engine process's stats file with empty caches (last writer wins). **Fix:** `_dump` early-returns when `_ENGINES` is empty.
- Routes mode (FREETOKEN_DR_ROUTES=1, 2026-11-15): each eager `_decode_routed` call copies (layer_id, topk_ids) into a device buffer BEFORE ensure_experts_hybrid rewrites ids to slot ids IN PLACE; dumps every 2688 calls + window close + exit; in routes mode the cadence rewrites stats too (final stats = full run); n_replay_steps==0 in the dump asserts eager.
- Per-layer slot bytes: cache `_copy_feat_bytes` can be None (fused-copy plan not built); authoritative bytes live on the Engine: `engine._moe_layer_slot_bytes` (layer_slot_bytes(banks.sources)) — the routes meta falls back to it.
