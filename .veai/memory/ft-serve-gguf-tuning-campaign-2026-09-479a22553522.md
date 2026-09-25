---
name: "ft-serve-gguf-tuning-campaign-2026-09"
description: "GGUF ft serve tuning winner (mr1+8191+0.85, radix L-drop root cause), harness gotchas, task briefs"
type: project
lastUpdated: 2026-09-20T18:20
lastRecall: 2026-09-24T14:17
---

# ft serve GGUF GLM-5.3-Flash: tuning campaign 2026-09-16/17 (RTX 5090)

Hardware A/B campaign on user's command (fp8 KV, 500k reserve, hybrid, 147.5 GB GGUF glm5next). Artifacts: .tasks/ft-gguf-serve-tuning/ (REPORT.md, phase2/PHASE2.md, CAMPAIGN.json, measure.py [patched], logs/).

## Winner config (measured, replaces user's 0.89/4096/mr4)
`--moe-cache-auto --kv-reserve-tokens 500000 --kv-cache-dtype fp8 --memory-ratio 0.85 --max-prefill-length 8191 --moe-strategy hybrid --moe-cpu-threads 16 --max-running-requests 1`
- Prefill (median full chunks): 4096=262, 6144=281, 8128=293 tok/s (+12% vs user's BASE-mr4, +21% vs mr1@4096).
- Decode back-to-back @65k: BASE 13.50 vs winner 14.72 tok/s = +9% (earlier "-10%" was daytime noise). Winner decode wall per repeated 65k request 7.6-8.6s (radix HIT); BASE ~251s content-TTFT per repeat (full re-prefill, cached=0 every time).
- Slots 410/288/288 (winner) vs 295/288/288 (BASE); group-1 working set 312. All groups < 576 -> prefill overlap OFF everywhere (synchronous materialized prefill; overlap needs 576/partition).
- 8191@0.89 = OOM on first prefill (triton do_bench 256 MiB + transients vs 2.89 GiB free); 0.85 gives 4.11 GiB free. Scheduler chunks 65k at 8128 (not 8191).
- Deep fill on winner: ~447k tokens depth, NO OOM, peak 32145/32607 MiB (~460 spare) - razor thin near pool limit.

## Radix-reuse root cause (the big finding)
- fp8 KV is NOT broken. Reuse HITs at <=9.7k prompts in all configs. At 65k: switch = CHUNK SIZE, not ratio/mr/timing: 4096-chunks -> MISS (0/49) at 0.85/0.89/0.90, mr1/mr4; 8128-chunks -> HIT 65536.
- ROOT CAUSE: `mamba_last_track_seqlen` (L) is NOT forwarded across chunk transitions - scheduler/prefill.py:236-256 (try_add_one/_add_one_req) forwards ping_pong/next_track_idx/restore_src but not L. 65585-tok prompt at 4096 -> final chunk 49 tok -> c=0 -> L=None at final-batch commit (scheduler.py:398) AND finish-donate (scheduler/cache.py:346-362) -> no snapshot -> MISS. 8128: final chunk 561 -> c=8 -> donation lands.
- Fix-1: persist L across transitions (~5-10 lines + test) -> reuse at ANY chunk size. Fix-2 (optional): per-chunk donation -> partial-prefix resume; Fix-3: refresh snapshot timestamps on insert (LRU cascade currently wipes deep reuse points back to 65536; hybrid_radix_cache.py:168-237).
- Match semantics: HybridRadixCache.match_prefix climbs to deepest node with LIVE GDN snapshot (hybrid_radix_cache.py:76-88); no live snapshots on path -> cached_len=0 even with intact KV pages.

## memory-ratio verdict
- 0.85 required for 8191 chunks; also protects the early floor-gate: needs 1059 slot-0-eq, BASE@0.89 envelope 1056 when hoptodesk (remote desktop) eats ~492 MiB -> BOOT FAIL-FAST (reproduced 4x on 2026-09-17 morning; ok the day before). Production (llama-swap wrapper ~1.27 GiB) sits even closer to the gate.
- Lower than 0.85: -0.01 ratio ~ +0.307 GiB free-after-init, ~-80-90 group-1 slots; 0.83 unverified but likely > working set; do NOT chase: greedy fill never frees VRAM outward at fixed reserve except via free-after-init.
- GDN pool: 25 slots x 140.76 MiB = 3.44 GiB at mr4; mr1 -> 9 slots = 1.24 GiB (the +226 group-1 slots).

## Load speed
- --expert-load parallel is a NO-OP for GGUF: _gguf_banks raises NotImplementedError -> silent serial (expert_banks.py:179-183). Serial pass ~134 GB memcpy + 1-thread PinPipeline; boot 94-132s.
- FREETOKEN_BANK_CUDA_ALLOC=1 (born-pinned): SLOWER, 132.5 vs 104.2s (+27%) - memset dominates. Rejected.
- Real lever = code (S): GGUF->FTW serve path incomplete: ftw.py:619 kind_kernel_for("gguf") KeyError; gguf_types not persisted in convert finalize meta / restored by load_ftw_banks (ftw.py:645-659) -> cpu_executor.py:106-112 raises. Fix ~0.5-1 day -> NVFP4-style fast path (multi-worker O_DIRECT, born-pinned): expected boot 115s -> ~50-60s (NVFP4 ladder 245->72s). No tests exist for load_ftw_banks - create from scratch. Also floor-gate silent for FTW (gguf_signature_groups only scans bare .gguf).
- gguf CUDA kernel JIT cached on disk (clang++ host needed); triton autotune warmup = first prefill chunk (~110 tok/s vs ~293 steady).

## Decode levers: exhausted at flag level
f=30.7% auto from fresh benchbw v4 profile (GPU-uuid keyed, no TTL); pools [15,1,1] @16T == auto; FREETOKEN_GGUF_DOT_TIER default avx2 already; FREETOKEN_HYBRID_OVERLAP keep 1. Remaining gap = per-layer sync + CPU-leg DRAM BW (not flag-tunable). Big prefill code lever: ggml_moe_a8_vec is GEMV-grade on GEMM shape (copy share only ~20% of chunk) - tiled MMQ prefill kernel would beat chunk scaling.

## Harness gotchas (measurement)
- First SSE frame arrives BEFORE prefill -> decode steady must skip times[0]; throughput parser must anchor "input throughput (token/s):" (token usage caught otherwise); measure.py stdout has leading/trailing log lines -> naive json.loads fails (run1_base.json cleaned manually); ladder cum bookkeeping used last-chunk new (undercounted); pkill must use '[f]t serve' bracket pattern or kills own wrapper; #mamba-slot/token-usage gauges EXCLUDE evictable tree snapshots (cannot distinguish empty tree from match-fail).

## Pending user decisions
S (FTW fast path) and/or fix-1 (radix L persistence) implementation; optional 0.83 ratio probe for desktop VRAM.

## 2026-09-17 follow-ups
- User ordered task briefs for fresh-session execution (composed same day, anchors re-verified by direct file reads): `.tasks/fix1-radix-track-seqlen/TASK.md` (fix-1; verified: prefill.py try_add_one continuation forwards ping_pong/next_track_idx/restore_src/swa_evicted_seqlen but NOT L) and `.tasks/ftw-gguf-fastpath/TASK.md` (S; verified: ftw.py:~620 kind_kernel_for(quant_format) KeyError line, ExpertBanks return ~657 without gguf_types, convert.py finalize meta ~306 lacks gguf_types, cache.py:346-362 frozen-donate conditions verbatim).
- MMQ prefill kernel case opened by user request ("how much will tiled MMQ give") - see memory ft-gguf-prefill-mmq-roofline (2026-09-18 consolidation absorbed the old ft-gguf-moe-prefill-mmq-gap notes); kernel-study fork was still running at session end.

## fix-1 implemented (2026-09-18, vektory79, UNCOMMITTED)
- The root-cause statement above is now HISTORICAL: python/freetoken/scheduler/prefill.py forwards mamba_last_track_seqlen across chunk transitions (+3 lines: _add_one_req param/assignment, try_add_one continuation forwarding). Tests: regression tests in tests/scheduler/test_hybrid_cache_manager.py (pre-fix FAIL verified: None==64) + triaged strengtheners; scheduler + kvcache/radix 244 passed; full gate 2135 passed / 6 baseline-Environment.
- Hardware PASS (RTX 5090, measure.py): repeat 65k @4096 -> HIT #cached-token 65472 (pre-fix 0); 8191 @ 0.85+mr1 -> HIT exactly 65536; prefill medians 300.6 (4096) / 336.4 (8128) tok/s, decode 13.9-14.7 tok/s. Boot deviations: 0.89 fail-fast (desktop VRAM tax) -> 0.90 for Run A, winner config 0.85+mr1 for Run B; 8191@0.90 OOMs in triton do_bench (256 MiB) - known headroom, not fix-related.
- Artifacts: .tasks/fix1-radix-track-seqlen/ (TASK.md status, research/code-anchors.md, research/tests-and-harness.md, verification/hardware-report.md). Commit pending user request; stage ONLY prefill.py + the test file (working tree also has modified .veai/memory files).

## fix-1 COMMITTED (2026-09-18)
- 5b72aba "fix(scheduler): carry mamba_last_track_seqlen across prefill chunk transitions" on vektory79 (parent ebf071b), single commit, 3 files (prefill.py +3; test_hybrid_cache_manager.py +160; test_abort_inflight_prefill.py +51/-5), 214+/5-. No push. The "commit pending" note above is superseded.

UPDATE (2026-09, post fix-1): "Pending user decisions S" above is RESOLVED and implemented - full outcome, tests and hardware numbers live in the dedicated memory ft-ftw-gguf-fastpath-outcome (boot 52.1 s; committed on vektory79 as 8568807/e9ad38d/ef14efb/2e9fcf1, no push). Also: the 293 tok/s prefill anchor above is the PRE-dense-q80/m-tile number; current bare baseline is 792-808.
