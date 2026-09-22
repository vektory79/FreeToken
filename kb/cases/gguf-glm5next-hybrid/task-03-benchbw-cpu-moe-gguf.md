# Task 03: benchbw CPU-MoE leg for gguf formats - W2

Type: implementation (benchmark + tests)
Suggested agent: Code (balanced)
Spec coverage: R3, S3. Depends on Task 01 (formats must be executor-claimable);
independent of Task 02 (disjoint files - may run in parallel with it).

## Goal

benchbw's "gguf" profile measures the CPU-MoE leg (stops auto-skipping) and the
recommendation becomes "hybrid" with a sane fetch fraction (not 0%, not 100%),
flowing via bench_profile.load_hybrid_fetch_fraction -> per-partition
_resolve_hybrid_fetch -> --moe-hybrid-max-fetch auto.

## Why This Task Exists

The benchbw CPU-MoE section auto-skips gguf (_CPU_MOE_FORMATS lacks the gguf
types), so the verdict is always "offload" and no fetch fraction is produced -
hybrid would have no auto sizing signal.

## Required Inputs (read first)

1. .tasks/gguf-glm5next-hybrid/research/hybrid-machinery-mapping.md - section 4
   (benchbw map) and section 5 (tests).
2. .tasks/gguf-glm5next-hybrid/TASK.md - W2 section + Lessons 4, 5.

## Files/Areas (verified line refs)

- python/freetoken/moe/benchbw.py:
  - _CPU_MOE_FORMATS :66 = {bf16, nvfp4, mxfp4_triton, ds_fp4} (note: q4_0 is
    ALSO absent despite executor support - a bench-only gap; include it here
    too and note it in the report).
  - gguf profile row DTYPE_WORKLOADS :146-150 (anchor :148); _offload_bank_specs
    gguf :316-327 (geometry already there); CPU-MoE skip :648-650 ->
    recommended="offload" :672-675; both legs -> ratio + recommend :610
    (threshold 2.0); overlap pair measure_overlap_bw :550 sets the fetch split.
  - gguf layout needed in _cpu_moe_bank_sources (:334+).
- python/freetoken/moe/bench_profile.py: load_hybrid_fetch_fraction :159
  (overlapped pair preferred).
- tests/moe/test_hybrid_fetch.py: test_benchbw_gguf_profile_sanity :131
  hard-pins the "offload" verdict - flip to expect "hybrid" + sane fraction.

## Constraints / Non-Goals

- Engine gate and partition wiring are NOT in this task (Tasks 04 / 02).
- Do not change offload-verdict logic for non-gguf profiles.
- Per-role geometry for bank sources: down rows = H (4096), gate/up rows = I
  (2048); role-ordered gguf_types ("gate","up","down").
- One change per commit; conventional commits; Assisted-by: Veai. No push.

## What to Do

1. Read the inputs; verify cited lines.
2. Add the gguf types (and q4_0) to _CPU_MOE_FORMATS; add the gguf layout to
   _cpu_moe_bank_sources; make the CPU timing leg run for the gguf profile.
3. Validate: fetch fraction in (0,1); verdict "hybrid" for the gguf profile;
   offload verdicts elsewhere unchanged.
4. Update tests/moe/test_hybrid_fetch.py:131 (flip the pinned verdict) and add
   a case asserting the fraction is strictly between 0 and 1; keep/add a case
   where a non-CPU-capable profile still recommends offload (if such a config
   exists in the current tests).
5. Run tests/moe (record exact output).
6. Commit (conventional, Assisted-by: Veai).
7. DECISION GATE (B-2 triage, 2026-09-15): from the measured scalar CPU-leg bandwidth compute r = t_cpu/t_fetch per signature (per-expert bytes: (18,18,23) = 10.88 MB, (18,18,14) = 13.3 MB, (23,23,14) = 15.8 MB; PCIe 25-45 GB/s) and report r in report_result. Bandwidth-matched fraction f = r/(1+r) beats offload by (1+r)/r at any miss rate. r > ~4 schedules the contingent Task 06 (SIMD tier); otherwise scalar stays.

## Expected Output (report_result)

- Exact changed files; the new benchbw gguf verdict + fraction from a local
  run or test fixture; test commands + exact outputs; blockers.

## Acceptance Criteria

- [ ] benchbw "gguf" profile -> verdict "hybrid", fetch fraction in (0,1)
- [ ] test_benchbw_gguf_profile_sanity updated + green; tests/moe green
- [ ] I've created a git commit for this task
