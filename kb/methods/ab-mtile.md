# A/B sweep - dense q8_0 token-tile (Candidate A, replan wave 2): 4/8/16/32/64 on hardware

Executed 2026-09-19 22:24 - 2026-09-20 02:03 (+03:00), branch vektory79, HEAD 4505572 +
the uncommitted Candidate A changeset (docstring :505 de-indent applied at sweep start).
Single variable per boot: FREETOKEN_GGUF_DENSE_MTILE. All 6 boots: FREETOKEN_GGUF_MOE_MTILE=32,
NSPLIT at its committed default 2 (knob never set - verified in every server env pin),
port 18801, winner flags, 65,585-token fill + streamed decode re-send (radix HIT 65536 on
every boot) + 6 greedy prompts, nsys steady-span capture (fill chunks 3..7), serialized +
reaped, trap-on-EXIT + FAILPAT watchdog + hard timeout 2400 s (never hit), SIGTERM->10s->SIGKILL.

Pre-step (Review-B F3/F4 + Review-A F2 closure): see the appended section in
wave2-candidate-a.md - gate-candidate-a.log (16 failed / 2179 passed / 206 skipped,
nodeid list recorded, all Environment, zero gguf-path failures), ptxas regs/SMEM table
archived verbatim from /tmp/ptxas_dense.log (pre-TTL), preflight census clean.

## Sweep table (prefill median c2..c7, T43 discipline)

| tile | prefill median tok/s | delta vs tile4 | chunk time s | dense term s/chunk | dense launches/chunk | kda launches/chunk (gridX=389) | kda per-half ms | grid.y (all classes) |
|------|---------------------|----------------|--------------|--------------------|----------------------|-------------------------------|-----------------|----------------------|
| 4 (unset)  | 563.90 | baseline | 14.414 | 5.515 | 348.8 | 67.88 | 37.82 | 2032 |
| 8          | 658.45 | +16.77%  | 12.344 | 3.371 | 346.4 | 67.31 | 23.55 | 1016 |
| 16         | 726.37 | +28.81%  | 11.190 | 2.223 | 346.1 | 67.70 | 15.45 | 508 |
| 32         | 769.95 | +36.54%  | 10.556 | 1.640 | 348.8 | 67.78 | 11.16 | 254 |
| 64         | 792.08 | +40.46%  | 10.262 | 1.386 | 349.9 | 68.12 | 9.39  | 127 |

Chunk split at tile 64 (sqlite, per chunk, values quoted from the regenerated
ab-mtile-results.json): dense q8_0 1.386 (kda halves 0.656 + other 0.730) + grouped MoE
4.357 + fetch copies 3.074 + rest 1.439 = busy 10.256 s (idle 0.006). (An earlier draft
quoted busy 10.247 / idle 0.014 / rest 1.430 - stale pre-regeneration numbers; the JSON
values are authoritative.)
Grouped MoE and fetch are tile-invariant as designed (MoE 4.36-4.40, fetch 3.07-3.14,
rest 1.43-1.45 across all five tiles); the ENTIRE gain is the dense term
5.515 -> 1.386 s = -4.129 s/chunk, and the bridge CLOSES: e2e chunk time
14.414 -> 10.262 s = -4.152 s vs dense -4.129 s = a 0.023 s residual carried by the
tile-invariant MoE/fetch/rest terms (busy gain -4.152 s matches the e2e gain within
0.001 s; review-a A3).

Dense rate scale-up (kda half shape, FLOP-constant, per-half rates, results JSON
tflop_per_s_per_half): 21.9 TF/s (tile 4) -> 35.2 (8) -> 53.6 (16) -> 74.3 (32) -> 88.2
(64) - 4.03x for the 16x re-read cut (grid.y 2032 -> 127), clearly SUB-LINEAR - the v3a
sibling-family prediction (13.7 -> 39.6 TF/s from tile 4 -> 32) was directionally right
and the approaching-saturation reading is strengthened. Floor model predicted
+1.9/+2.9/+3.4/+3.6% - REFUTED BY 8-10x: the class-cluster model was correct, other dense
classes (gx=128 bucket incl. o_proj/ffn_down, gx=64 shexp, gx=512 mla_q_b, gx=384
ffn_gate_up) scale at the same rate and dominate the payoff. CORRECTION (review-a F1):
an earlier draft quoted "21.7 -> 107.3 (16) -> 176.5 (64) TF/s" - a 2x fused-FLOP
miscount at tiles 16/64 (fused 1.6577 TFLOP divided by the per-half time); 176.5 TF/s
exceeds the RTX 5090 FP32-FMA peak (~104.8 TF/s), physically impossible for the
non-tensor-core MMQ path, which is how the error was caught.

## T44 validity - PASS

Tile-4 boot reproduced the 561.50 tok/s instrumented serving class at 563.90 (+0.43%,
band -2..+3%, same-mode CUPTI framing). Context: plain-boot equivalents 546.5/553.8;
instrumented-vs-plain delta is the known symmetric ~1.2% CUPTI overhead; all cross-tile
deltas above are SAME-mode (all boots nsys-instrumented with identical window mechanics).

## T46 liveness - PASS on every boot (constancy + exact grid.y carry the proof)

- Total dense MMQ launches: 346-350/chunk across all tiles - CONSTANT: the tile changes
  grid dims, not launch count. Anchor calibration (review-b F7): the planned ~356 was an
  estimate and is miscalibrated - the actual dense census is 346-350/chunk, outside the
  analyzer's +/-4 window, so the analyzer honestly prints launch_liveness CHECK on the
  absolute anchor; the T46 proof rests on the per-tile CONSTANCY of the count plus the
  EXACT single grid.y per boot, both of which hold.
- kda halves: 67.3-68.1/chunk (anchor 68, gridX=389 each) on every boot - NSPLIT=2 path
  live and untouched.
- per-launch grid.y = ceil(8128/tile): sqlite shows EXACTLY one grid.y per boot,
  2032/1016/508/254/127, zero mismatch classes - the tile factor shrink is proven ON the
  launches (Review-A F3 wiring gap closed with direct evidence).
- Decode: 13.79/14.36/13.21/14.62/13.53 tok/s (mtile4b 14.95) - observed 13.21-14.95
  across the six boots, wider than the cited 13.5-14.7 class band (review-a F6: exceeded
  on both sides; the honest statement is no tile ordering within a ~6% run-variance
  spread), radix HIT
  65536 every boot: the replayed bs=1 graph is MMVQ (neither knob enters it), as anchored.

## Winner + NET gain

WINNER: tile 64, prefill median 792.08 tok/s = +40.46% e2e vs the tile-4 baseline
(563.90), same-mode. NET vs the morning-class context: vs 546.5 plain = +44.9%; the
instrumented-class gain (+40.46%) is the honest same-mode number. Monotone increase over
4/8/16/32/64 with diminishing steps (+16.8 / +12.0 / +7.7 / +3.9 pp) - the dense term is
approaching its sub-kernel residual (1.39 s/chunk at tile 64 vs ~1.4 s attention+rest
floor); further tile growth would need mmq_y widening (N-side), out of this wave's scope.

## Bitwise quality gate - BLOCKER-CLASS FINDING (resolved as measurement artifact, evidence below)

6-prompt greedy diff, baseline tile-4 vs each tile, sha256 over the raw choices JSON,
same-mode all boots:

- tile-4 vs tile-8/16/32/64: 3-4 of 6 prompts bitwise-EQUAL; divergence is CONFINED to
  prompts 2/3/4. Per-prompt precisely (review-b F5 - "six-way disagreement" was the
  wrong summary): p2 shows 4 distinct hashes across the 6 boots (tile4 == tile16), p3
  and p4 show 5 each (tile4b == tile64); prompts 0/1/5 are identical across ALL 6 boots.
- DISCRIMINATOR boot (mtile4b: tile 4 via EXPLICIT env "4", separate boot): vs the unset
  tile-4 baseline, the SAME prompts 2/3/4 diverge the same way; vs tile-64, mtile4b is
  6/6 IDENTICAL.
- Cross-boot matrix is pairwise-inconsistent (tile8 differs from tile4 on the same prompt
  that tile16 matches; tile4b matches tile64 6/6 but not tile4 6/6) - impossible under a
  deterministic tile-dependent kernel difference. The decisive observation is the
  campaign's FIRST SAME-CONFIG pair: tile4b vs tile4 (identical tile-4 config, different
  boots) diverges on prompts 2/3/4 while tile4b == tile64 6/6 - the earlier same-mode
  pairs (v3a 24/24, N-split 6/6) were all cross-config, so same-config reproducibility
  had never been tested. Plain boot-to-boot nondeterminism (scheduler/atomics in a
  non-dense path, e.g. the MoE grouped pipeline) is the only consistent explanation. The
  unit-level bitwise tile-invariance of the dense kernel is ALREADY proven (torch.equal
  battery, wave 2); this e2e gate CANNOT confirm or refute quality at 6-prompt
  granularity and the observed pattern carries no tile signal.

**Operational lesson (review-b F2): a single-boot 6-prompt e2e greedy bitwise diff is
NOT usable as a quality gate on this tree - boot-to-boot nondeterminism at fixed tile
makes any single-boot diff uninterpretable. Unit-level torch.equal remains the quality
proof. Optional quiet-box arbitration (2x tile-64 + 2x tile-4, ~4 boots / ~40 min, zero
code change) is a USER OPTION to rehabilitate the e2e gate or date the nondeterminism -
not a blocker for the commit/default decision.**

VERDICT: NOT a tile-caused quality regression (evidence above); also NOT a clean 6/6
confirmation. The honest statement: e2e bitwise at this harness granularity is
boot-to-boot flaky on ~3 of 6 prompts regardless of tile; a rigorous quality gate would
need longer generations and repeated boots per leg. Recorded as an OPEN harness/quality
question, NOT a Candidate A blocker (the kernel-level invariance proof stands).

## Decode

bs=1 streamed decode (radix-cached re-send, 64 tokens): 13.79 / 14.36 / 13.21 / 14.62 /
13.53 tok/s for tiles 4/8/16/32/64, mtile4b 14.95 - observed 13.21-14.95 across the six
boots, wider than the documented 13.5-14.7 class band (no tile ordering within the ~6%
run-variance spread; bs=1 replays the MMVQ graph, neither knob enters it).

## Deviations

1. MID-SWEEP PLATFORM KILL (403 security-policy error): the original run was force-killed
   after completing tile 4 + tile 8. Recovery pass (2026-09-20 ~00:40): census found a
   foreign llama-swap-managed `ft serve` on port 18084 (a ~30 GB reading at that moment,
   started 23:31 - AFTER my tiles 4/8 completed at 22:40) plus its 7 sem.mp- files.
   NOT mine - NOT killed, NOT signalled. tiles 4/8 artifacts were verified sound (sqlite
   kernel counts, env pins, grid.y, run.json integrity) and REUSED. The boots that
   overlapped the foreign serve are tiles 16/32/64 + the mtile4b discriminator ONLY -
   capture timestamps are the evidence (mtile4 22:26:59 and mtile8 22:37:56 completed
   BEFORE the 23:31 foreign boot; the pre-step - gate 22:19, ptxas re-read + preflight
   census 22:20 - also predates it; the late boots ran 01:19-02:03). No 403 recurrence
   anywhere in the resumed session.
2. Foreign-serve interaction, CORRECTED (review-a A5 + review-b B2): smaller than
   originally documented, and the ranking is unaffected. (a) The earlier "~30 GB VRAM
   co-residency - both fit (mine ~26 GB)" claim is DELETED: arithmetically impossible on
   the 31.40 GiB card (26 + 30 > 31.4). The sweep's own telemetry shows device-wide free
   memory 4.03-4.15 GiB at graph capture in EVERY boot, indistinguishable pre/post the
   foreign boot, and mtile4b's gpu_after_boot ~28 GB (28069 MiB, leaving 4.5 GB free) is
   the sweep's own boot ALONE - the foreign serve held NO significant VRAM during any
   measurement window (llama-swap idle swap-out explains the 00:40 ~30 GB reading). Zero
   OOM/watchdog hits, clean teardowns, c2..c7 spreads <1%. (b) De-risk argument, no
   clean re-run needed: mtile4b - the SAME tile-4 config booted at 02:00 deep inside the
   alleged interference window - reproduces mtile4 within 0.35% (+0.32% prefill / -0.35%
   dense term), so any interference effect is <0.5%, an order of magnitude below the
   2.88% tile32-vs-64 step; contention would also have to appear in the tile-invariant
   MoE/fetch/rest terms, which are flat across ALL boots including 4/8. (c) Persistence
   contradiction resolved: the earlier "persisted through the sweep" phrasing contradicts
   the postflight census (no ft serve survivors). Whether the foreign serve exited on
   its own (llama-swap idle unload) or was stopped externally is UNVERIFIED - census raw
   outputs were not archived; no evidence the sweep signalled it.
3. Second nsys export flag fix mid-wave: `--force` -> `--force-overwrite=true` (mtile4
   sqlite was exported manually with the corrected flag; all later boots used the fixed
   stage script).
4. mtile4b discriminator boot added beyond the brief (5 boots) to arbitrate the bitwise
   finding - documented above.
5. Analyzer bugs fixed post-first-run: the top-level dense_term was computed /1e3
   (ms->"s") - fixed to /1e9; the per-class dense_term.classes[*].s_per_chunk fields
   carried the same /1e3 leftover (MICROSECONDS labeled seconds, review-a F4 / review-b
   F4) - fixed to /1e9; the kda tflop_per_s fields held GFLOP/s under a TFLOP/s label
   (/1e6 divisor) - fixed to /1e9. Results regenerated from the same sqlite files;
   headline numbers unchanged (recount-exact: 5.5148 -> 1.3855 s/chunk dense term).
6. One-off 58 s JIT rebuild at the mtile16 boot (review-b B2/F3): the mtile16 graph
   capture ran 58.30 s/batch with an in-line nvcc rebuild of gguf_kernel.cu (every other
   boot: "ninja: no work to do" at 2.85-4.08 s/batch). Cause UNIDENTIFIED - not
   tile-driven (tile 8 booted a new tile value with no rebuild) and not foreign-driven
   (mtile4b reproduces tile-4 perf within 0.35% and identical launch geometry);
   CPU-side, finished before the nsys window (capture start 01:31:46), chunk medians
   unaffected, no perf impact.

## Artifacts

- Harness: .tasks/dense-q80-gemm/{ab_mtile_stage.sh, ab_mtile_run.py, ab_mtile_analyze.py}
  (adapted from the nsplit wave; nsys mechanics per D09 + nsys-silent-no-collection-trap:
  report presence + sqlite kernel counts, never banners).
- Per boot: .tasks/dense-q80-gemm/mtile{4,8,16,32,64,4b}.{nsys-rep,sqlite,_run.json};
  logs .tasks/ft-gguf-serve-tuning/logs/mtile{4,8,16,32,64,4b}.log (+.pid).
- Results: .tasks/dense-q80-gemm/ab-mtile-results.json (5-tile sweep + comparison block).
- Gate: .tasks/dense-q80-gemm/gate-candidate-a.log; pre-step archive in wave2-candidate-a.md.

## Postflight census (2026-09-20 02:0x)

No ft serve / nsys / pytest / ab_mtile survivors; GPU 1463-1569 MiB (desktop baseline);
sem.mp- 5 (IDE baseline; the sweep's transient counts self-resolved); only foreign
compute apps = hoptodesk (302 MiB, 190 MiB). Postflight == preflight modulo the foreign
llama-swap tree (documented in D1/D2; not touched).
