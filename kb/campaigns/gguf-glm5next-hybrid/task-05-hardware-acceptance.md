# Task 05: Hardware acceptance + full regression (W6 + W5 closure)

Type: verification (hardware + full test battery)
Suggested agent: Code (strong; needs run_command for serve + pytest)
Spec coverage: R5, R6, S5, S6. Depends on Tasks 01-04 all committed.

## Goal

Prove the hybrid path end-to-end on the real file: full regression green;
hybrid boot + serve; decode @64k fill >= 14.3 tok/s (offload-only reference);
fetch fraction sane; VRAM delta recorded; 24-prompt battery 20+/24 on-topic.

## Required Inputs (read first)

1. .tasks/gguf-glm5next-hybrid/TASK.md - W5/W6 sections + all 10 Lessons
   (esp. 8: compute-sanitizer + CUDA_LAUNCH_BLOCKING for IMA; 9: boot smoke
   recipe; 10: tuned serve command).
2. .tasks/gguf-glm5next-hybrid/PLAN.md - Shared Context (tuned command, real
   file path, signature groups).
3. The prior campaign's acceptance record:
   .tasks/gguf-glm5next-path-a/PLAN.md (verification/ sections) - the template
   for run reports and the 24-prompt battery.
4. Task 03 decision-gate measurements (commit 2169baa, real rig): CPU-MoE leg
   11.71 GB/s (gguf kernels ride the AVX2 ISA tier), PCIe gather 43.45 GB/s,
   bandwidth-matched fetch fraction 0.781, r = 3.56-3.71. EXPECT the benchbw
   headline verdict to read "offload" (ratio 0.27 < 2.0 threshold) - that is
   the conservative rule, NOT a blocker: the fraction still flows and the
   matched-fraction hybrid should beat offload by ~21% per the B-2 math. The
   14.3 tok/s gate + the burst probe decide; if missed, revisit Task 06 (SIMD).

## What to Do

1. Full regression first: `uv run pytest tests/ -m "not slow"` from the venv.
   Record the exact tail (counts). Expected ~2-4 min; GPU tests self-skip; do
   NOT report success from a killed/skipped run.
2. Hardware run on the REAL file:
   /media/ai/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL.gguf
   Command: known-good base (`ft serve --model-path <gguf> --moe-cache-auto
   --port 18081` - positional path rejected) + tuned flags (`--num-tokens
   70016 --memory-ratio 0.85 --max-prefill-length 4096`) + `--moe-strategy
   hybrid --moe-cpu-threads <N>`. Do NOT pass explicit --moe-cache-size (it
   FAIL-FASTS below the per-signature floors).
   Boot smoke: watchdog with FAILPAT, /ready gate, model field from
   /v1/models, SIGTERM -> exit 143 (the recipe lives in the
   ft-serve-test-and-e2e-gotchas memory - reproduce it).
3. Measure and record:
   - decode tok/s at 64k fill (target >= 14.3; NVFP4 hybrid baseline for
     context: 14.1-15.5);
   - the fetch fraction log line (sane, matches benchbw auto);
   - ASSERT the auto fetch fraction is active, NOT the cap-1 fallback (engine.py:925-929 - the fallback is the real regression path per B-2 triage: full-burst worst case ~2.5x worse than offload);
   - per-layer overflow stats (collect_stats exists);
   - FREETOKEN_HYBRID_OVERLAP=0 A/B + one burst probe with per-step "not slower than offload" check;
   - CPU utilization during decode misses;
   - VRAM delta vs offload-only (hybrid should free slot/KV budget).
4. Battery: the 24-prompt quality battery (prior campaign's template in
   .tasks/gguf-glm5next-path-a/PLAN.md) - must stay 20+/24 on-topic. Compare
   divergence pattern vs the offload-only run (chars 49-339 quant drift was
   the accepted baseline).
5. If anything fails: triage (Implementation bug vs Test bug vs Environment)
   BEFORE changing code; for IMAs use compute-sanitizer + CUDA_LAUNCH_BLOCKING
   (Lesson 8) and write the bisect notes to verification/.
6. Write artifacts to .tasks/gguf-glm5next-hybrid/verification/:
   boot log tail, decode numbers, battery digest, any bisect notes.
7. Commit only if code fixes were needed (conventional, Assisted-by: Veai).

## Constraints / Non-Goals

- No new features here; measurement + fixes strictly within the landed scope.
- The GPU must be free: ensure no other serve/bench process holds VRAM.
- Long-running serve steps need the watchdog pattern - never an open-ended wait.

## Expected Output (report_result)

- Exact pytest tail; boot verdict + timing; decode tok/s vs target; fetch
  fraction; VRAM delta; battery digest (x/24 + divergence note); paths to
  verification/ artifacts; any fixes committed.

## Acceptance Criteria

- [ ] Full regression green (exact counts recorded, not assumed)
- [ ] Hybrid boots and serves the real file
- [ ] Decode @64k >= 14.3 tok/s
- [ ] Battery 20+/24 on-topic
- [ ] verification/ artifacts written
- [ ] I've created a git commit for this task (only if fixes were needed)
