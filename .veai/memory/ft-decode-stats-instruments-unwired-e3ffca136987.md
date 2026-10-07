---
name: "ft-decode-stats-instruments-unwired"
description: "Decode T1 driver v3: hook GraphRunner.replay (replay bypasses layer host code); kineto aborts scheduler; events-only"
type: project
lastUpdated: 2026-09-26T22:58
lastRecall: 2026-10-07T11:55
---

# Decode-stats instrumentation: unwired in code, T1 harness driver now exists (2026-09-26)

## Code facts (unchanged)
- `moe_collect_stats` — EngineConfig field (engine/config.py:50) WITHOUT a CLI flag; wired at engine.py:1033 `cache.collect_stats = config.moe_collect_stats` (must be True BEFORE CUDA-graph capture so the device-side stat += ops are captured).
- `collect_decode_freq` (offload_cache.py:262) is never set True in-repo; `decode_routing_stats()` (oracle_hit/working set/entropy for a frequency-aware cache) and `decode_miss_stats_per_layer()` have no callers in the repo.
- Stat tensors are allocated UNCONDITIONALLY in OffloadMoeCache.__init__ (lru_stats, stat_missing/active/calls/fetched + per-layer variants, decode_freq) — the flags only gate recording; flipping them post-init but pre-capture is safe.
- The decode_freq histogram is ONLY accurate WITHOUT CUDA graphs: the host-side scatter_add in ensure_experts does not re-run on graph replay. Eager decode = `--cuda-graph-max-bs 0` (engine/graph.py `_determine_cuda_graph_bs` returns [] when max_bs < 1).
- The old campaign driver /tmp/ft_serve_driver.py is lost; the ServerArgs banner prints moe_collect_stats=False even when a later hook sets it — never read the banner as "stats off".

## T1 harness driver EXISTS (2026-09-26, decode-research session)
- `.tasks/decode-research/driver/sitecustomize.py` — inert unless env FREETOKEN_DECODE_RESEARCH=<outdir> is set (runner passes it + PYTHONPATH=<driver dir> + FREETOKEN_DR_TAG to the ft serve tree): patches `Engine._init_offload_moe_cache` to flip collect_stats+collect_decode_freq on every partition cache, wraps `OffloadMoELayer._decode_routed` with a layer-call counter (42 layer-calls = 1 decode step), runs a torch.profiler CPU+CUDA window over layer-calls [START,END], exports trace_<tag>.json (chrome) + stats_<tag>.json (aggregate + per_layer + routing per cache).
- `.tasks/decode-research/analyze_trace.py` — decomposes the trace: GPU busy/idle, per-class kernels (fetch_copy / moe / dense_quant_gemm / attention / gdn / memcpy), top kernels, top cpu ops.
- `run_arm.py` passes the env pairs via --env; plain boots are unaffected.

## TRAP: trace window keyed on layer-calls can silently never fire
First instrumented run produced an EMPTY outdir: that boot decoded only 467 tokens x 42 = 19,614 layer-calls, never reaching the default START=42*480=20,160. No error is raised. Fixed: START=42*150, END=42*160 plus a "[decode-research] driver armed" banner at patch time. Decode length is boot-to-boot nondeterministic under temp0 (467..763 tokens observed) — key instrumentation windows LOW and verify the outdir is non-empty before trusting a run.

**Why:** every decode-stats wave (R3 routing oracle, per-layer misses, the 44 ms GPU_chain+sync decomposition) depends on this driver; the routing oracle additionally requires the eager (graphs-off) run.

**How to apply:** hybrid trace + miss stats = instrumented graphs-on run; routing oracle = separate short eager run with `--cuda-graph-max-bs 0`; always confirm the armed banner and non-empty outdir.

## TRAP 2: CUDA-graph replay bypasses ALL model host code per decode step (2026-09-26, second empty run)
Decode steps are `GraphRunner.replay()` calls: the model's Python (incl. `OffloadMoELayer._decode_routed`) runs only during capture/warmup — NEVER per replayed step. Second instrumented run: 546 tokens x 42 = 22.9k nominal layer-calls, threshold 6300 — still no window, because those steps were replays. A layer-call-keyed window can never fire in graphs-on decode regardless of thresholds. **Fix:** hook `GraphRunner.replay` (engine/graph.py:216; one call = one decode step) for graphs-on runs; the `_decode_routed` hook serves ONLY the eager run (`--cuda-graph-max-bs 0`), where replay is not used. (This is also why `collect_stats` works under graphs — its `+=` ops are baked in — while the freq histogram does not.)

## TRAP 3: torch.profiler start inside the scheduler NATIVELY ABORTS the process (2026-09-26, third run)
Opening a kineto CPU+CUDA window at decode step 150 killed the scheduler outright: C++ stack, "backend worker freetoken-TP0-scheduler exited", no Python exception to catch (consistent with ncu ERR_NVGPUCTRPERM on this box). The arm harness then idles the full 420 s readiness timeout on the dying server (looks like a hung background task). **Fix (driver v3):** profiler gated behind `FREETOKEN_DR_PROF=1` (default OFF); CUDA-event pairs bracket each replayed step and `step_spans_ms` are computed after one `torch.cuda.synchronize()` at window close; the stats dump at window end is UNCONDITIONAL — never gate the dump on profiler success.

## Driver v3 (supersedes the v1 description above)
- Hooks: `Engine._init_offload_moe_cache` (flip collect_stats + collect_decode_freq pre-capture), `GraphRunner.replay` (step counter, default steps 150..160 via FREETOKEN_DR_SSTART/SEND, CUDA-event pairs per step), `OffloadMoELayer._decode_routed` (eager fallback, layer-calls 6300..6720).
- Outputs in FREETOKEN_DECODE_RESEARCH outdir: `stats_<tag>.json` always (aggregate + per_layer + routing per cache + step_spans_ms); `trace_<tag>.json` only with PROF=1. Markers in the serve log: "driver armed" (x4 processes), "trace window OPEN"/"events-only window OPEN", "window CLOSED", "DUMPED stats".
- Interpretation without kineto: mean(step_spans_ms) vs the wall ms/step = GPU-stream-engaged vs host/idle split of the 44 ms block; per_cache aggregate gives fetched/cpu per layer; routing only valid in the eager arm. Verify non-empty outdir + the DUMPED marker before trusting any run.
