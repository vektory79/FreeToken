# Plan: --moe-strategy hybrid for glm5next GGUF (CPU GEMV for IQ/K banks)

Seed spec: TASK.md in this folder (read alongside; it holds the 10 session lessons).
Status source of truth: the Task Checklist below.

## Goal

`ft serve --model-path .../GLM-5.3-Flash-UD-Q3_K_XL.gguf --moe-strategy hybrid
--moe-cpu-threads <N>` boots and serves with the hybrid decode path (LRU ensure +
capped PCIe fetch OVERLAPPED with CPU GEMV on misses) for the gguf bank formats.
Decode @64k fill >= 14.3 tok/s (offload-only number). benchbw "gguf" verdict
becomes "hybrid" with a sane fetch fraction. All suites green, battery 20+/24.

## Scope

- In scope: CPU GEMV for IQ3_XXS(18)/IQ4_XS(23)/Q6_K(14) in cpu_moe_ext.cpp;
  per-projection bank formats in the CPU executor; multi-partition hybrid wiring;
  benchbw CPU-MoE gguf leg; engine gate relaxation; tests; hardware acceptance.
- Out of scope: Q3_K/Q4_K GEMV (MTP block only, skipped); edits to vendored
  kernel/csrc/gguf/*.cu (verbatim sgl-kernel ports); python dequant in the hot
  path (gguf-py is test-only reference); NVFP4 hybrid behavior changes; any
  push/PR activity (private-use, upstream #34 gate waived).

## Specification

Requirements:
- R1. cpu_moe_ext.cpp computes GEMV for IQ3_XXS, IQ4_XS, Q6_K matching the CUDA
  block layouts (Q6_K: 210 B, d LAST 208:210, ql128/qh64/int8 scales16;
  IQ3_XXS: 98 B, d FIRST, 8x uint32 aux32 in qs bytes 64:96, iq3xxs_grid +
  ksigns_iq2xs LUTs; IQ4_XS: 136 B, d FIRST, scales_h + scales_l[4],
  kvalues_iq4nl; dfloat = half) within the dfloat contract: per-element bound
  8*2^-11*factor*|d| + 1e-3 (tests/kernels/test_gguf_quant.py:156).
- R2. The CPU executor handles per-projection types in one layer (gate IQ3_XXS +
  up IQ3_XXS + down IQ4_XS simultaneously) and multi-signature partitions
  (3 groups in the real file), lifting the engine multi-partition ValueError
  for cpu/hybrid when every partition is executor-capable.
- R3. benchbw "gguf" profile measures the CPU leg and recommends "hybrid" with a
  fetch fraction in (0,1), flowing to --moe-hybrid-max-fetch auto.
- R4. Engine gate accepts gguf+hybrid when ALL the file's per-layer bank types
  are CPU-executable (gguf header scan + executor capability set); still rejects
  gguf+cpu and gguf+fused; #186 clamp (8191 at top_k 8) untouched.
- R5. NVFP4 hybrid behavior unchanged; all existing suites stay green.
- R6. Hardware: hybrid boots the real 147 GB file and serves; decode @64k
  >= 14.3 tok/s; 24-prompt battery 20+/24 on-topic.

Non-Goals:
- NG1: Q3_K(11)/Q4_K(12) GEMV (appear only in the skipped MTP block).
- NG2: edits to vendored .cu/.cuh gguf files (host LUT copies live in cpu_moe code).
- NG3: python-side dequant in the hot path (gguf-py reference is for tests only;
  models/gguf/dequant.py has NO iq3xxs reference and must not gain one for this).
- NG4: changes to the 2E prefill-overlap degrade semantics (must not regress).
- NG5: push/PR/issue activity by agents.

Acceptance Scenarios:
- S1. CPU GEMV output for crafted IQ3_XXS/IQ4_XS/Q6_K banks matches the gguf-py
  reference within the dfloat bound (unit parity test).
- S2. A layer with CPU-miss + GPU-hit expert mix: CPU output for missed experts
  matches the GPU path for the same rows.
- S3. benchbw "gguf" profile -> verdict "hybrid", fetch fraction in (0,1).
- S4. Engine gate: hybrid accepted for a glm5next gguf config whose types are
  all CPU-capable; cpu/fused still rejected; incapable type still rejected.
- S5. `uv run pytest tests/ -m "not slow"` fully green.
- S6. Hardware acceptance run recorded under verification/ with decode numbers
  and battery digest.

## How to Use This Plan

Execute task files in checklist order (dependency-driven). Each task file is
written for direct handoff to a Code agent: Required Inputs names the exact
artifacts to read; the research files carry file:line evidence maps. Per task:
run the quality loop (Test -> Review x2 -> Triage -> Fix TP), then commit
(conventional commits, one change per commit, Assisted-by: Veai), then update
the checklist here and the status line in TASK.md.

## Task Checklist

- [x] Task 01: CPU GEMV for gguf formats (W1) - task-01-cpu-gemv-gguf-formats.md - CLOSED 2026-09-15: 5a04423 (implementation) + bd02232 (11-TP fix wave, verified GO) + 427a431 (negative-path tests, 17 passed). Triage: 11 TP fixed / 1 FP / 1 partial; post-fix review 1 TP fixed / 1 FP (assert-style). STOP GATE passed.
- [x] Task 03: benchbw CPU-MoE gguf leg (W2) - task-03-benchbw-cpu-moe-gguf.md - CLOSED 2026-09-15: commit 2169baa; review GO (byte layout, r math, verdict flow all verified; fraction flows regardless of verdict string; WeakWarning dead-bench-paths = deferred cosmetic). STOP GATE passed. DISCOVERY for Task 04: second gate site at engine.py:~1817 (auto->hybrid upgrade excludes gguf by name).
- [x] Task 02: Partition-aware hybrid wiring (W4) - task-02-partition-aware-hybrid.md - CLOSED 2026-09-15: commit aad5d3a (per-cache executors via _init_cpu_moe_executors, resolve_pool_affinities disjoint cores, capability-gated lift via partition_executor_rejection; layers/moe.py + offload_cache.py untouched). Reviews x2 GO; triage: B-1/B-2/B-3/B-5 -> Task 07 hardening; B-4 -> Task 04. STOP GATE passed. Bonus: pre-existing CC/CXX leak root-caused and FIXED separately (1635ecd).
- [x] Task 04: Engine gate relaxation (W3) - task-04-engine-gate-relaxation.md - CLOSED 2026-09-15: commits f6b94ad + ef83ee8. Review GO (scan = faithful factor of _gguf_bank_bytes; capability fully delegated to Task 02 surface; deviation SAFE - strictly earlier-or-equal loud failures, no new hang/corruption path). Triage: Issue 1 (offload+explicit-ids engine test) + Issue 2 (stale advice text) -> Task 07; Issue 3 (FTW+hybrid fails safe, Info) -> recorded, no action. Engine 150 passed. STOP GATE passed.
- [x] Task 07: Post-Task-02 hardening ride-along - task-07-post-task02-hardening.md - CLOSED 2026-09-15: commit eb7de4c. Focused review GO (all 6 verdicts OK; findings 2 Info + 1 WeakWarning = cosmetic on unreachable/extreme paths, no-action). Tests 327 passed. STOP GATE passed.
- [x] Task 05: Hardware acceptance + full regression (W6+W5) - task-05-hardware-acceptance.md - CLOSED 2026-09-15 (no code changes, HEAD eb7de4c): FUNCTIONAL PASS - boots 136 s, serves, fraction 78% active (not cap-1), VRAM delta noise, battery 24/24 (gate 20+), all 16 pytest failures map 1:1 to the known baseline classes. PERF GATE FAIL - hybrid decode 2.99 tok/s vs the 14.3 gate (same-session offload control 14.25); root-caused in verification/bisect-notes.md: per-layer GPU<->CPU handshake floor ~1.9-3.0 ms x 42 layers = 80-127 ms/step (> offload's whole 70 ms step) + dominant pool starved at 6/16 threads (2.96 GB/s effective vs 11.3 benched); modeled best in-scope fix ~4-5 tok/s. A/B: OVERLAP=0 -> 2.57; cap-6 -> 5.08. Task 06 RE-SCOPED (handshake batching + SIMD + weighted affinities) and scheduled by its own trigger (gate miss). Artifacts: verification/ (6 files). Review skipped: no code diff; measurements cross-validated (benchbw refresh reproduces Task 03; offload control matches the campaign 14.3).
- [x] Task 06 (RE-SCOPED): ggml integer-kernel port + weighted pools - task-06-simd-tier-contingent.md - CLOSED 2026-09-15: commits 11f1a80 + 979e3fc + fix 63b9bff (cap-down env override, weighted-split wrap guard + corner test, tier skipif + cap-down assertion, AVX2 full-path production-combo test, MM256 attribution). Reviews: A GO (instruction-by-instruction fidelity, LUTs machine-checked byte-identical, quantizer verbatim), B GO w/ 2 warnings (both fixed). benchbw FULL rerun restored the profile: nvfp4 f=24.8%, gguf f=30.7%, CPU leg 66.8 standalone / 50.0 overlap. Tests 71+145+154 passed (1 pre-existing excluded). STOP GATE passed.
- [x] Task 08: Hardware re-acceptance post-Task-06 (task-05-hardware-acceptance.md as template) - CLOSED 2026-09-15 (no code changes): ALL THREE GATES PASS. Decode @64k: hybrid 16.67 tok/s (16.12 short; burst min 15.04) vs offload control 12.97 (14.32; burst max 14.36) = 1.28x offload, 1.17x over the 14.3 campaign gate, 5.6x over pre-port 2.99; 60.0 ms/step (1.43/layer - better than NVFP4 hybrid's 1.73). Asserts: f=30.7% auto (not cap-1), pools [15,1,1], isa avx2-w4a8k, plan 1234 slots; offload PCIe-bound 2.14 GiB/step @28.4 GB/s vs hybrid 573 MB/step + CPU 1.61 GB/step. Battery 24/24. NVFP4 sanity: user 1M command booted, f=24.8%, short decode 15.18 (profile restoration confirmed). Artifact: verification/re-acceptance.md. CAMPAIGN CLOSED - hybrid RECOMMENDED for this file.

Dependencies: 01 first (foundation). Then 02+03 may run in parallel (disjoint
files: engine/moe/layers vs benchbw). Then 04 (engine.py shared with 02, so it
must follow 02). 05 last (needs everything).

## Shared Context

- Branch vektory79; prior offload campaign complete through 78115df (6 commits:
  4162f7c -> 4443d7e -> 2637714 -> f801a08 -> 028f2d9 -> 78115df); tree clean at
  plan time. Offload-only GGUF serve is production-usable (decode 13.8-15.6
  tok/s @57-64k, battery 20/24).
- Real file: /media/ai/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/
  GLM-5.3-Flash-UD-Q3_K_XL.gguf (147.5 GB, arch glm5next, 1412 tensors).
  H=4096, I=2048, E=288, top_k=8. Per-layer bank types: gate/up IQ3_XXS
  (IQ4_XS on ckpt 11), down IQ4_XS (Q6_K on ckpt 11,12,44). Signature groups:
  (18,18,23)x39, (18,18,14)x2, (23,23,14)x1.
- Binding lessons (TASK.md "Lessons" 1-10): LayerCompletionTracker note-count
  trap; non-contiguous activations (f_a/g_a stride (24896,1)) - any new GEMM
  call site must inherit the 028f2d9 normalization; PINNED host banks only
  (pageable mmap VAs illegal on Linux UVA); per-role geometry (down rows=H,
  gate/up rows=I - never reuse one row count); role-ordered gguf_types
  ("gate","up","down", never slots.values()); 2E prefill-overlap floor;
  dfloat tolerance contract; compute-sanitizer + CUDA_LAUNCH_BLOCKING for IMA;
  boot smoke recipe (ft-serve-test-and-e2e-gotchas); tuned serve command
  (base + --num-tokens 70016 --memory-ratio 0.85 --max-prefill-length 4096;
  explicit --moe-cache-size FAIL-FASTS - auto slots only).
- Commit discipline: conventional commits, one change per commit,
  Assisted-by: Veai. No push/PR. CONTRIBUTING.md is binding.
- Test count drift note: tests/models/test_gguf_expert_banks.py now has 30 test
  defs (older docs said 25) - verify with --collect-only when relevant.

## Research Artifacts

- research/w1-cpu-gemv-mapping.md - CPU executor + C++ extension map, gguf block
  layouts, hybrid call chain, test templates, W1 blocker list (a)-(d).
- research/hybrid-machinery-mapping.md - hybrid decode path, engine gate,
  partition wiring, benchbw sections, test map with flip points (W2/W3/W4).
- research/verification/ (to be created by Task 05) - hardware run logs.

## Plan Deltas (2026-09-15, post-Task-01 triage)

- B-2 verdict: scalar-vs-fetch ratio r ~ 1.5-6 (pessimistic edge plausible: ~0.7-1.5 ms/expert on 16 threads vs 0.24-0.44 ms fetch). Hybrid beats offload at ANY miss rate when the fetch fraction is bandwidth-matched (f = r/(1+r), gain (1+r)/r ~ 14-40%); the real regression path is the cap-1 fallback (engine.py:925-929, full-burst worst case ~2.5x worse than offload ~ 3.4 tok/s). SIMD tier NOT mandatory pre-Task 05.
- Task 03 becomes the DECISION GATE: measured scalar CPU-leg bandwidth -> r; r > ~4 schedules the contingent Task 06 (SIMD tier).
- Task 05 additions: assert auto-fraction active (not cap-1), per-layer overflow stats (collect_stats), FREETOKEN_HYBRID_OVERLAP=0 A/B, one burst probe with per-step "not slower than offload" check; the 14.3 tok/s gate stays.
- Task 02 additions: per-partition executor init (caches[0] -> per-cache), per-layer gguf_types resolution, mixed-cache executor test.
- Task 04 additions: _cpu_moe_executor_viable (engine.py:1435-1450) becomes gguf-type-aware when the gate lifts.
- Task 01 fix wave (11 TP): gate row_bytes exactness assert, uniformity check in _split_gguf_formats, fmt-keyed gguf dispatch + loud role errors, drop "gguf" alias from supported-formats message, docstrings, row_bytes->block_bytes rename, load_u16le for aux32, tests (all-(-1) -> y=0, out-of-range id, per-projection parametrized over the 3 real signatures).
- Task 03 gate results (2026-09-15, commit 2169baa): r = 3.56-3.71 < ~4 -> Task 06 (SIMD) NOT scheduled; Task 01's gguf GEMV already runs under the AVX2 ISA tier (cpu 11.71 GB/s vs STREAM 75.84). VERDICT NUANCE: benchbw's real-rig verdict for gguf is "offload" (ratio 0.27 < 2.0 threshold) while the bandwidth-matched fetch fraction 0.781 still flows to the engine (B-2 math: matched-fraction hybrid beats offload by (1+r)/r ~ 21-22% at any miss rate). Task 05's hardware A/B (hybrid vs offload, 14.3 tok/s gate, "not slower than offload" burst probe) is the arbiter; if the gate misses, revisit Task 06 with the measured margin (~8-11% below the r>~4 line).
- Task 06 rework estimate (2026-09-15, research/task06-rework-estimate.md): the per-layer GPU<->CPU round trip is architecturally FORCED (block k+1 routing consumes block k MERGED output - cross-layer pipeline infeasible); the lever is per-round-trip COST (memop h=3.02 ms vs hostfunc h=1.90; irreducible floor ~0.1-0.3 ms/layer). Projections: B alone 3.9-4.3 tok/s; B+A(h=1.0) 5.7-6.1; B+A+C 7.5-7.9; BEST case 14.6 (zero margin); realistic landing 8-11. VERDICT: NO-GO on the 14.3 gate as a committed target - even best case only TIES offload (14.25) while freeing no VRAM and losing the burst regime; hybrid's entire prize is ~8-11 ms/step because PCIe 40.1 GB/s beats any CPU leg. RECOMMENDATION: keep offload for glm5next GGUF; optionally run M-A0 microbench (S, 1-2 d, gate floor<=0.5 ms/layer) + B weighted split (S, 2-4 d, low risk, ->3.9-4.3 tok/s) as de-risking; A+C research-only.
- NVFP4 A/B CORRECTION (2026-09-15, verification/nvfp4-ab-measurements.md): on the user's exact 1M NVFP4 command hybrid measures 13.77-14.46 tok/s vs offload 9.91-10.12 = 1.39-1.43x WIN; both modes boot today (341 slots, identical plans - VRAM competition refuted). The GGUF "handshake floor" premise of the estimate above is CORRECTED: fixed sync is only ~0.6-1.0 ms/layer (NVFP4 all-in 1.73 ms/layer); the rest was per-layer PCIe fetch volume, endogenous to CPU-leg speed (slow scalar leg -> f=78% -> ~6 experts/layer fetched). GGUF path gaps vs NVFP4: scalar-only IQ/QK tier (no W4A8 VNNI), no activation-prequant reuse, per-role tables vs packed gate_up, starved pool split, benchbw baked f=78%. REVISED Task 06 verdict: the ggml integer-kernel port (research/ggml-cpu-kernel-study.md; ~350-450 LOC; microbench gate >=40 GB/s sustained vs 11.7) + weighted pools + packed tables project GGUF hybrid ~11-14 tok/s - offload parity hinges on the microbench; CPU-only mode projected 10-14 tok/s with ~134 GB VRAM freed.
