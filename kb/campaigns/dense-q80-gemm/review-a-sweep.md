# Review-A - dense-q80-gemm Candidate A m-tile sweep (DENSE_MTILE 4/8/16/32/64)

Scope: measurement validity + arithmetic behind the +40.46% verdict. Read-only audit;
independent sqlite recount performed for mtile4, mtile4b, mtile64 (precedent:
review-a-abnsplit.md), plus timeline reconstruction from `_run.json` capture timestamps
and file mtimes. Branch vektory79 @ 4505572 + uncommitted changeset.

## A1 medians, exclusions, T44, env pins - CONFIRMED

- Median discipline verified: analyzer `chunk_times` takes `vals[1:-2]` of the
  throughput lines = c2..c7, excluding c1 warmup (168.97-210.66), the bogus
  last-full-chunk line (2097.41 / 2214.67 / 2277.74 / 2303.54 / 2339.47 - the T43
  artifact, present in every boot), and the tail line. Recomputed medians match the
  sweep table exactly: 563.90 / 658.45 / 726.37 / 769.95 / 792.08 tok/s.
- T44: 563.90 / 561.50 = +0.43%, inside the -2..+3% same-mode band - PASS. Recount of
  mtile4b (same tile, explicit env) reproduces 565.70 (+0.32% vs mtile4) - the
  instrumented class is stable across boots.
- Env pins (`/proc/<pid>/environ`, all six `_run.json`): FREETOKEN_GGUF_MOE_MTILE=32 on
  every boot; DENSE_MTILE absent (mtile4) / 8 / 16 / 32 / 64 / 4; NSPLIT key absent on
  every boot (committed default 2, knob never set); identical boot args
  (`--memory-ratio 0.85 --max-prefill-length 8191 --max-running-requests 1`); NSYS_SESSION
  is the only other varying pin and is not a server knob. Same-mode: all five sweep boots
  nsys-instrumented with identical window mechanics (start after 2nd, stop after 6th
  throughput line), all captures started+stopped rc=0 with sqlite kernel data present.
  mtile4-unset behaves as tile 4 (verified behaviorally via mtile4b, see A8).

## A2 independent sqlite recount - CONFIRMED, with one corrected md series

Recount (same window math as the analyzer: kernel+memcpy span / (8128/median)):

| boot | mul_mat_q8_0/chunk | dense s/chunk | kda gridX=389 /chunk | kda med ms | grid.y histogram |
|------|--------------------|---------------|----------------------|------------|------------------|
| mtile4  | 348.80 | 5.5148 | 67.88 | 37.823 | {2032: 1408} - single value |
| mtile4b | 348.72 | 5.4952 | 68.16 | 37.823 | {2032: 1412} - single value |
| mtile64 | 349.94 | 1.3855 | 68.12 | 9.394  | {127: 1346} - single value |

- Matches the deliverable to the 4th decimal (5.5148 / 1.3855). Dense launches 346-350
  constant across tiles; kda 67.9-68.2 at gridX=389; grid.y EXACTLY one value per boot
  (full histogram, not just the median - stronger than the analyzer's med-only check).
- FLOP constant verified: kda half = 2 x 8128 x 12448 x 4096 = 0.8288 TFLOP
  (gridX 389 = 12448/32, grid.y = ceil(8128/tile) - shapes confirmed by the recount).
- DENSE-RATE SERIES IN ab-mtile.md IS WRONG (finding F1): the md claims
  "21.7 -> 107.3 (16) -> 176.5 (64) TF/s per-half". Tiles 16/64 are 2x inflated: they
  divide the FUSED FLOPs (1.6577 TFLOP) by the per-half time. Corrected per-half rates
  (results JSON `tflop_per_s_per_half` is already correct): 21.9 -> 35.2 -> 53.6 -> 74.3
  -> 88.2 TF/s. Sanity: 176.5 TF/s would exceed the RTX 5090 FP32-FMA peak (~104.8 TF/s)
  - physically impossible for the non-tensor-core MMQ path; 88.2 TF/s = ~84% of peak,
  plausible. Corrected rate gain 88.2/21.9 = 4.03x for the 16x re-read cut (2032 -> 127)
  - still clearly sub-linear, so the T52-style diminishing-returns conclusion SURVIVES
  (and the "approaching saturation" reading is strengthened, not weakened).

## A3 dense-term vs e2e bridge - CONFIRMED (closes at 0.023 s)

- e2e chunk gain 14.414 -> 10.262 = -4.152 s; dense-term gain 5.5148 -> 1.3855 =
  -4.1293 s; busy gain 14.408 -> 10.256 = -4.152 s.
- Per-tile invariance recomputed from the results JSON: grouped MoE 4.3635 / 4.4018 /
  4.3908 / 4.3518 / 4.3573 (+-0.6%), fetch 3.0900 / 3.1339 / 3.1350 / 3.1113 / 3.0742
  (+-1.0%), rest 1.4396 / 1.4312 / 1.4352 / 1.4468 / 1.4387 (+-0.5%). Invariance holds.
- Non-dense terms moved -0.0229 s tile4 -> tile64; -4.1293 + (-0.0229) = -4.1522 =
  the e2e gain. The bridge closes within 0.001 s of the busy sum and 0.023 s of the
  e2e chunk median. The ENTIRE e2e gain is the dense term - verified.
- FINDING F2: ab-mtile.md's sentence "-3.588 s shows up as e2e chunk-time reduction
  (the +0.54 s difference is attention/GDN/DSA scaling + drift)" contradicts its own
  sweep table (14.414 - 10.262 = 4.152, not 3.588) and is stale pre-regeneration text.
  The corrected bridge is BETTER than the md claims.

## A4 the +40.46% headline - CONFIRMED (no correction needed)

- Arithmetic: 792.08 / 563.90 - 1 = +40.46% - exact.
- Apples-to-apples: both endpoint boots instrumented, same fill (65,585 tokens, same
  9-line throughput structure), same boot flags, same MoE pin, NSPLIT default, only
  DENSE_MTILE differs; tile4-unset proven equivalent to explicit 4 (A8). The one
  asymmetry (mtile4 pre-foreign-serve vs mtile64 inside the alleged window) is bounded
  in the conservative direction by mtile4b at <0.5% (A5c), so the true same-mode delta,
  if anything, is a hair LARGER. +40.46% stands as the honest same-mode headline; the
  md correctly keeps the +44.9% plain-boot cross-mode figure as context only.

## A5 foreign-serve interaction - NEED-MORE-DATA on evidence; ranking de-risked, no re-run needed

- (a) Presence evidence: NOT archived. grep of the whole task folder + mtile boot logs
  finds 18084 / llama-swap / 23:31 only inside ab-mtile.md prose - the 00:40 recovery
  census output (pgrep / nvidia-smi / sem.mp-) was not preserved as an artifact
  (finding F3).
- (b) VRAM fit: all four late boots (16/32/64/4b) completed fill + decode + prompts +
  capture with clean teardowns (leftover [], no OOM in any mtile log). But the md's
  "30 GB VRAM co-residency - both fit (mine ~26 GB of 32 GB)" is arithmetically
  impossible on the 31.40 GiB card (26 + 30 > 31.4), and my boots' own telemetry shows
  no such co-tenant: avail_mem at graph capture is 4.03-4.15 GiB on tiles 16/32/64,
  indistinguishable from pre-serve tiles 4/8 (4.05-4.10). A 30 GB holder would have
  made the boot impossible. Most likely the foreign serve was a proxy with an unloaded
  model, or it exited before 01:19. Either way VRAM contention is NOT evidenced.
- (c) Conservative-direction bias: the direction of the argument is right (hostile
  load slows, never speeds), but its premise (persistence through the sweep) is
  unevidenced. Ranking risk is empirically bounded: mtile4b - the SAME tile-4 config,
  booted at 02:00 deep inside the alleged interference window - reproduces mtile4 at
  +0.32% prefill / -0.35% dense term. So any interference effect is <0.5%, an order of
  magnitude below the 32-vs-64 step (769.95 -> 792.08 = +2.88%) and the 16-vs-32 step
  (+6.0%). A differential slowdown large enough to flip tile 32 vs 64 is excluded.
  Also, contention would have to appear in the tile-invariant terms (MoE/fetch/rest),
  which are flat across ALL boots including 4/8 - it does not.
- (d) Tile-4 baseline free of it: YES on the evidence available - mtile4 capture
  22:26:59-22:28:01, run.json 22:29:34, sqlite 22:33:59, mtile8 done 22:40:31; the
  foreign serve allegedly started 23:31. Consistent, no contradicting artifact.
- Verdict: the ranking and the winner do NOT depend on the unresolved presence
  question; NO clean re-run is required. Optional: re-run tile 64 on a verified-idle
  GPU only if the absolute +40.46% is to be quoted publication-grade.

## A6 analyzer unit bug - CONFIRMED for all verdict-bearing numbers; residues noted

- Current `ab_mtile_analyze.py` computes `dense_total_s` with /1e9 (ns -> s) and the
  kda/moe/fetch/align paths with ms/1e3 (correct); results JSON headline dense terms
  (5.5148 / 3.3714 / 2.2230 / 1.6402 / 1.3855) reproduce EXACTLY from the same sqlite
  files in my independent recount - regeneration is proven, not stale.
- Residue 1 (F4, NIT): the per-class fields `dense_term.classes[*].s_per_chunk` still
  carry the /1e3 bug (values are MICROSECONDS labeled seconds, e.g. mla_o_proj
  559379.0388 = 0.5594 s; kda_in_proj_half 2614058.4297 = 2.6141 s). No verdict number
  reads these fields, but "all other fields unchanged" in the deviation note hides a
  still-broken unit. One-line fix: /1e3 -> /1e9 in the dense_cls dict comprehension.
- Residue 2 (NIT): ab-mtile.md tile-64 split says busy 10.247 / idle 0.014; the
  regenerated JSON says 10.256 / 0.006 - stale prose numbers.

## A7 decode + radix - CONFIRMED (one band NIT)

- cached_token = 65536 on all six boots (radix HIT per boot, verified in every
  `_run.json` decode block).
- bs=1 graph replay: `graph_capture_lines` show "Start capturing CUDA graphs with
  sizes: [1]" on every boot - only the bs=1 graph exists; bs<=6 routes to the MMVQ vec
  path, so neither knob enters the captured graph, as anchored.
- Decode 13.79 / 14.36 / 13.21 / 14.62 / 13.53 (mtile4b 14.95): flat with no tile
  ordering. NIT (F5): the cited 13.5-14.7 class band is exceeded on both sides by the
  six-boot data (13.21-14.95); "flat within the class" is a mild overstatement - the
  honest statement is no tile signal across a +-6% run-variance spread.

## A8 deviation hygiene - CONFIRMED

- mtile4b discriminator: justified and verified from the sha256 matrix (6 greedy
  prompts, raw choices JSON). Prompt 2/3/4 hashes across the six boots:
  p2 = e8d515(4) / 9ca671(8) / e8d515(16) / 80c32a(32) / fe5fba(64) / fe5fba(4b);
  p3 = 71df40 / 6f83e3 / 2c14b5 / 630eb3 / 6a0584 / 6a0584;
  p4 = 75875a / dcc22a / d6a3cb / bffd00 / 717dc9 / 717dc9.
  Pairwise-inconsistent clustering (tile16 matches tile4 on p2 but not p3/p4;
  mtile4b matches mtile64 6/6 while diverging from mtile4) - impossible under a
  deterministic tile-dependent kernel difference; boot-to-boot nondeterminism in a
  non-dense path is the only consistent reading. Unit-level torch.equal tile-invariance
  (review-a-candidate-a.md A1) stands; the e2e gate carries no tile signal. The
  BLOCKER-class heading in ab-mtile.md is correctly resolved; no quality regression
  claim either way is supported at this granularity.
- nsys --force-overwrite fix: stage script now exports with `--force-overwrite=true`;
  boot 1's capture is VALID - mtile4.sqlite exists (exported manually at 22:33:59 after
  the 22:28:01 rep) and yields a clean recount (1408 launches, 58.184 s window).
- No 403 recurrence: grep over all six mtile*.log finds no 403 (only unrelated older
  logs contain the string in a port number). The mid-sweep 403 kill + recovery-reuse of
  tiles 4/8 is internally consistent with the artifact mtimes.
- Benign observation (FP): mtile16's graph capture line shows a mid-sweep nvcc/JIT
  rebuild of gguf_kernel (58.3 s capture at 01:29). Same source and flags -> same SASS;
  it completes before the fill, so the measurement window is unaffected.

## Findings by severity

- F1 MAJOR (md arithmetic): ab-mtile.md dense-rate series "21.7 -> 107.3 -> 176.5 TF/s"
  is 2x inflated for tiles 16/64 (fused FLOPs divided by per-half time). Corrected:
  21.9 -> 35.2 -> 53.6 -> 74.3 -> 88.2 TF/s per-half; rate gain 4.03x (not 8.1x);
  176.5 TF/s exceeds the 5090 FP32-FMA peak - impossible. Sub-linearity conclusion
  survives.
- F2 MAJOR (md arithmetic): stale bridge sentence "-3.588 s e2e / +0.54 s drift"
  contradicts the table; corrected bridge closes at 0.023 s (e2e -4.152 vs dense
  -4.129), with MoE/fetch/rest verified invariant - the verdict rests on firmer
  ground than the md states.
- F3 MAJOR (evidence hygiene): foreign-serve presence has no archived primary census
  artifact, and the "30 GB VRAM co-residency - both fit" claim is arithmetically
  impossible (26 + 30 > 31.4 GiB) and contradicted by my boots' own avail_mem
  (4.03-4.15 GiB, same as pre-serve boots). No impact on the ranking (A5c bound), but
  D1/D2 need a factual rewrite.
- F4 NIT: results-JSON dense_term.classes[*].s_per_chunk still µs-labeled-as-s
  (analyzer /1e3 leftover); headline dense numbers are unaffected (recount-exact).
- F5 NIT: T46 wording - dense launches 346.1-349.9 vs anchor ~356 "PASS (+/-4)" does
  not hold numerically (the analyzer itself says CHECK); constancy across tiles + exact
  grid.y are the load-bearing liveness evidence and hold. The ~356 anchor was an
  estimate; consider re-anchoring at ~348.
- F6 NIT: decode class band cited 13.5-14.7 vs observed 13.21-14.95 across six boots.
- F7 NIT: tile-64 busy/idle prose 10.247/0.014 vs regenerated 10.256/0.006.
- FP: the 6-prompt bitwise divergence is confirmed as boot nondeterminism, not a tile
  effect (sha matrix above); no quality claim is supportable at this harness
  granularity, matching the md's own downgrade.

## Overall verdict

SUPPORTS the claimed ranking and the winner. Monotone prefill gains
+16.77 / +28.81 / +36.54 / +40.46% over tiles 4/8/16/32/64 are carried exactly by the
dense term (recount-exact: 5.5148 -> 3.3714 -> 2.2230 -> 1.6402 -> 1.3855 s/chunk) with
a closing bridge (0.023 s residual) and tile-invariant MoE/fetch/rest. T43/T44/T46
discipline verified; single-variable design verified from process env pins. The
headline +40.46% same-mode number is arithmetically and methodologically correct and
needs NO correction. Corrected numbers to propagate into ab-mtile.md: dense rate
21.9 -> 88.2 TF/s per-half (4.03x, not 8.1x), bridge residual 0.023 s (not 0.54 s).
No BLOCKERs; 3 MAJORs are md-text/evidence-hygiene defects, none of which touch the
verdict arithmetic. No clean re-run is needed for the ranking; a clean-GPU tile-64
re-run is optional polish for the absolute headline.
