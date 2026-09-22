# Task 06 (CONTINGENT): AVX2/AVX-512 SIMD tier for the gguf dot kernels

Type: implementation (C++ SIMD tier)
Suggested agent: Code (strong)
Spec coverage: supports R6 (decode >= 14.3 tok/s) if the scalar tier proves too slow.

## Scheduling trigger (from Task 01 triage B-2, 2026-09-15)

Schedule this task ONLY IF at least one holds:
- Task 03's measured scalar ratio r = t_cpu/t_fetch > ~4 (per the decision gate
  in task-03-benchbw-cpu-moe-gguf.md, step 7); or
- Task 05 hardware acceptance misses the 14.3 tok/s decode gate at 64k fill.

TRIGGER MET 2026-09-15: Task 05's decode gate MISSED (hybrid 2.99 tok/s vs
14.3; see verification/bisect-notes.md). Task is SCHEDULED - but re-scoped, see
below.

## Re-scope (Task 05 bisect, 2026-09-15)

The SIMD tier alone is INSUFFICIENT. The measured dominant costs:
1. Per-layer GPU<->CPU handshake floor ~1.9-3.0 ms x 42 layers = 80-127
   ms/step, alone exceeding offload's whole ~70 ms step - needs handshake
   batching/amortization (fewer sync points per step).
2. 3-pool thread split starves the dominant pool (6/16 threads -> CPU leg
   ~2.96 GB/s effective vs 11.3 benched) - needs a weighted pool split
   proportional to per-partition miss volume + per-pool fetch fractions.
3. SIMD tier (LUT gather) still helps the CPU leg on top of 1+2.
The 14.3 gate is reachable only with (1)+(2); (3) amplifies. Weighted split
+ per-pool fractions ALONE model out at ~4-5 tok/s - do not ship them alone
and call it done.

## Estimate verdict (2026-09-15, research/task06-rework-estimate.md)

NO-GO on the 14.3 gate as a committed target: the per-layer round trip is
architecturally forced (routing dependency - block k+1 consumes block k's
MERGED output), realistic landing 8-11 tok/s, best case 14.6 = a tie with
offload (14.25) at zero margin, no VRAM freed, burst regime lost. Recommended
instead: M-A0 microbench (S, 1-2 d; gate: floor <= 0.5 ms/layer) + B weighted
split (S, 2-4 d, low risk, -> 3.9-4.3 tok/s) as optional de-risking;
A + C research-only.

## Goal

AVX2 (and optionally AVX-512) LUT-dequant dot tiers for iq3_xxs/iq4_xs/q6_k,
replacing the scalar tier on the hot path while keeping scalar as fallback.

## Files/Areas

- python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp: the dispatch slot is
  ALREADY designated - select_ggufdot() is the sole selector site
  (cpu_moe_ext.cpp:1247-1251); scalar kernels live at :1299-1374 (~2.5-4
  cyc/elem). Add AVX2 dot functions + selector entries per format; keep the
  ISA-selector structure (select_q4dot:1216 / select_nvi8dot:756 are the
  precedent).
- LUTs are already present as verbatim host copies (iq3xxs_grid, ksigns_iq2xs,
  kmask_iq2xs, kvalues_iq4nl).

## Constraints / Non-Goals

- Same dfloat tolerance contract (8*2^-11*factor*|d| + 1e-3).
- No vendored file edits; no python hot-path changes.
- One change per commit; conventional commits; Assisted-by: Veai. No push.

## What to Do

1. Implement AVX2 dot per format (LUT gather + FMA; follow the nvfp4/q4_0
   SIMD precedents in the same file for input handling).
2. Re-run the parity battery (tests/moe/test_cpu_moe_gguf_iq.py,
   tests/kernels/test_gguf_quant.py) - all bounds must hold.
3. Re-measure the CPU leg via benchbw (Task 03 harness) -> new r; update
   PLAN.md Plan Deltas with the measured numbers.
4. Commit (conventional, Assisted-by: Veai).

## Acceptance Criteria

- [ ] AVX2 tier lands behind select_ggufdot; scalar remains the fallback
- [ ] Parity battery green within the dfloat bound
- [ ] New r measured and recorded in PLAN.md; decode impact stated
- [ ] I've created a git commit for this task
