# Review-A: A/B wave NSPLIT 1 vs 2 (measurement validity + verdict arithmetic)

Date 2026-09-19. Scope: audit of ab-nsplit.md + ab-nsplit-results.json against the
raw evidence. Read-only; artifacts verified: ab_nsplit_analyze.py (full read),
ab-nsplit-results.json, nsplit{1,2}_run.json (raw throughput lines, env_pairs,
ServerArgs), step0-profile.md, step0-split.json, triage-step0.md, wave1-notes.md,
orchestration-notes.md (T43/T44/T46/D09). Independent recount: an IDE-kernel
read-only pass over nsplit{1,2}.sqlite reproduced every analyzer aggregate
(counts 138/272 at gridX 778/389; medians 91.7445/37.8289 ms; sums 12.7052/10.5506 s;
W 59.969/57.967 s; dense 6.0952/5.5291 s/chunk; busy 14.8471/14.4697; moe
4.2973/4.3856; fetch 3.0580/3.1107; pair-median 76.5838 ms). The JSON is
reproducible from the raw sqlite; no hand-tuning detected.

## A1. Medians + T43/T44 - CONFIRMED

- Medians re-derived from the raw throughput lines: boot-1 c2..c7
  551.22/547.28/549.13/547.20/547.19/545.99 -> median 547.24 (8128 -> 14.853 s);
  boot-2 564.58/562.13/562.80/560.31/560.87/559.38 -> median 561.50 (14.476 s).
  Delta +2.61%; chunk gain 0.377 s. Both correct. Steady bands do not overlap
  (boot-1 max 551.22 < boot-2 min 559.38) - the gap is robust, not median luck.
- Methodology (ab_nsplit_analyze.py `vals[1:-2]`): excludes line 1 (c1 warmup
  168.7/168.9), line 8 (last full chunk = the T43-bogus 2111.06/2100.82 tok/s
  tail-drain line) and line 9 (561-token tail). Exactly the T43 rule.
- T44 re-verified: 547.24/546.5 - 1 = +0.14%; 547.24/556.43 - 1 = -1.65%. PASS,
  inside the -2..+3% band; boot-2 is +2.75%/+0.91% vs the same refs.
- Single-variable check from nsplit{1,2}_run.json: ServerArgs strings byte-equal
  (same model, port 18801, max-running-requests=1, moe-cpu-threads=16,
  max-prefill 8191, hybrid); env_pairs differ only by NSPLIT (absent vs "2")
  and the NSYS_SESSION name; MTILE=32 pinned both. Immaterial deltas: KV alloc
  500736 vs 500800 tokens (num_pages 7824 vs 7825, free-VRAM rounding),
  ready_s 116 vs 140, graphs re-captured bs=1 in both.
- NIT (N-A1): ab-nsplit.md prose says nsplit2 mul census "1356"; the sqlite
  recount gives 1396 total mul_mat_q8_0 launches (348.6/chunk vs 356 census,
  ~2% partial-window shortfall as documented for step-0). Prose typo only.

## A2. T46 liveness + per-call arithmetic - CONFIRMED

- Counts: 34.18 fused/chunk (expected 34) and 67.92 halves/chunk (expected 68);
  67.92/2 = 33.96 ~= 34 per call. Both ~2% low = the capture window starts
  inside chunk 3 (same documented effect as step-0's 314.1/322); liveness PASS
  for both, gridX 778 -> 389 exactly as designed. Zero old-shape launches in
  boot-2 (gridX 778: none) - the T46 kernel-name contract holds.
- Per-call: 2 x 37.829 = 75.658 ms kernels; + 2 x 0.2631 = 0.526 ms copies ->
  76.184 ms constructed. The quoted 76.584 ms is the measured MEDIAN OF PER-PAIR
  KERNEL SUMS (copies excluded); the 0.93 ms gap vs 75.658 is median-of-sums vs
  sum-of-medians dispersion, not an inconsistency. So the exact figure:
  kernels-only 76.584 (measured), + copies 76.711 wall.
- Rates: FLOP = 2 x 8128 x 24896 x 4096 = 1.65769 TFLOP/call. Per-half
  21.91 TF/s; per-call kernel-only 21.65 TF/s; incl. copies 21.76 TF/s.
  Step-0 cluster band (W <= 71.3 MB) = 21.8-22.4 TF/s. The half rate is inside
  the band; the per-call rate sits 0.2-0.7% below its low edge. "Consistent
  with cluster rate" = confirmed within ~1%. NIT (N-A2): state which basis the
  quoted 21.65 uses (kernel-only pair-median).

## A3. Recovery arithmetic - CONFIRMED with a MAJOR framing correction

- Verified: dense term 6.0947 -> 5.5481 s = 0.5466 s; copies 19.19 ms/chunk
  (2 x 0.2631 ms x 34 = 17.9 ms + tiny launches; above the 15.3-16.8 ms
  prediction, acknowledged); fixed excess 34 x (91.577 - 75.350 ms) = 0.5517 s
  (75.350 ms = 1.65769 TFLOP / 22 TF/s).
- The report's "92.7%" is NOT misattached: it divides the KERNEL-ONLY kda
  reduction (3.1466 - 2.6347 = 0.5119 s) by the bound: 0.5119/0.5517 = 92.8%
  (92.7% with the rounded 0.552 bound). That percentage is arithmetically
  correct. The orchestrator's suspicion (0.5466/0.5517 = 99.1%) is an FP
  against the 92.7% figure itself - but it points at a real problem nearby:
  the verdict section also says "the split recovered essentially the FULL L2
  excess", which leans on the 0.5466 (99.1%) number.
- The 0.5466 window delta is NOT the split's causal recovery. Decomposition
  (all from the class tables, sqlite-verified):
  0.5466 = 0.5119 (kda kernel delta, causal)
         + 0.0764 (dense-FFN window artifact: boot-1 caught 10.9 ffn
           launches/chunk vs 8.74 in boot-2; per-launch meds identical
           37.28-37.60 ms -> pure window-edge count difference, favorable)
         - 0.0226 (small real drift across mla/shexp/o_proj classes)
         - 0.0192 (the split's own copies).
- CORRECTED NUMBERS: kernel-only recovery = 92.8% of the 0.5517 s bound;
  net of the split's own copies = 0.4927 s = 89.3%. The suggested
  (0.5466-0.0192)/0.5517 = 95.6% is wrong - 0.5466 already nets the copies,
  so that formula double-counts them. Severity MAJOR (verdict-section
  percentage framing), non-blocking: even the corrected 0.493 s is ~2x the
  NET window upper bound (0.26 s), so SHIP is unaffected.
- NIT (N-A3): quote 92.8% (bound 0.5517) or state the 0.552 rounding.

## A4. Per-call prediction vs measured - CONFIRMED, residual accounted

- Predicted from per-launch deltas: 34 x (91.744 - 76.584) = 0.5154 s.
- Measured kda-only delta: 0.5119 s (-0.7%). Residuals identified and sized:
  launch-count 34.18 vs 33.96 (+0.65% inflates boot-1), sum-vs-median skew
  (boot-2 mean half 38.789 vs med 37.829 = +2.5%; boot-1 mean 92.067 vs med
  91.744 = +0.35% - the halves are more right-skewed, so sum-based boot-2 is
  relatively heavier), and the pair-median dispersion (+0.93 ms/call). These
  close 0.5154 -> 0.5119 within noise. Window normalization is consistent
  (both terms are sum/chunks_eq).
- Bridge to the dense-term delta: 0.5466 = 0.5154 - 0.0035 (count/skew)
  + 0.0764 (ffn window artifact) - 0.0226 (drift) - 0.0192 (copies). Closes
  exactly. The report does not show this decomposition; it should (see A3).
- Other dense kernels: per-launch meds are flat between boots (mla_o 48.72 ->
  48.82, shexp 6.197 -> 6.198, kda_o 25.03 -> 24.88) - no hidden dense-kernel
  regression from the split.

## A5. e2e vs dense-term reconciliation - CONFIRMED

- Drift attribution verified exactly: MoE 4.2973 -> 4.3856 (+0.0884), fetch
  3.0580 -> 3.1107 (+0.0529), rest 1.3964 -> 1.4251 (+0.0287) = +0.1700 s,
  and 0.5466 - 0.1700 = 0.3766 ~= the e2e chunk gain 0.377 s. Busy-sum delta
  (14.847 -> 14.470) also equals 0.377 - window and e2e agree to 1 ms.
- In-class: MoE -1.7%/+0.4% and fetch -1.7%/+0.0% vs the step-0 anchors 4.37/3.11
  - strictly in-class. Rest 1.396/1.425 vs the 1.35 anchor is +3.4%/+5.6% in
  BOTH boots (N-A5: "all within anchor classes" is loose for rest; it is a
  residual bucket and the between-boot delta 0.029 s is what matters).
- Headline: +2.61% e2e is the CORRECT conservative headline - the drift was
  UNFAVORABLE, so e2e understates the effect (drift-free equivalent ~0.49-0.55 s
  = +3.3-3.7%). Quoting the dense-term recovery as the mechanism number is
  legitimate; quote it as 0.493 s net causal (89.3%), not 0.547 s / 99.1%.

## A6. Floor-model framing - measurement CONFIRMED, "REFUTED" framing needs correction (NIT)

- Rates verified: per-half 21.91 TF/s = 2 x MACs-per-call / 75.658 ms
  (FLOP_HALF/37.829 ms identical); per-call 21.65-21.76 incl. copies.
- Apples-to-apples: NO. The 19.6-19.9 TF/s M-sweep cap was measured on the
  FUSED shape (N=24896, W=108.35 MB > L2). The halves are a DIFFERENT shape
  (N=12448, W=54.2 MB <= L2) which step-0's own class table places in the
  20.9-22.4 TF/s W<=71.3 MB cluster. The measured 21.91 lands inside that
  band - i.e. the outcome was PREDICTED by the step-0 class model; nothing
  was falsified. The conservative 0.24-0.28 s window was built on a cap that
  never applied to the half shape.
- Corrected framing: "the L2 recovery exceeded the conservative window because
  the conservative model borrowed a cap from a different (fused) shape; the
  halves run at the cluster rate exactly as the step-0 class table predicts."
  Not "the cap was wrong" and not "REFUTED" as a kernel-model statement.

## A7. Remaining in_proj share + next lever - CONFIRMED

- Measured post-split in_proj: 2.6347 s kernels (avg 77.58 ms/call, sum-based)
  + 0.0192 copies = 2.654 s effective. Cluster-rate floor: 34 x 75.658 ms +
  34 x 0.526 ms = 2.590 s (2.580 s at exact 22 TF/s). Measured is 2.5-2.9%
  above floor (dispersion/skew) - "runs at the cluster rate" holds within 3%.
  NIT (N-A7): the report's "2.63 s at cluster rate" is the measured kernels
  figure; the constructed cluster-rate figure is ~2.59 s.
- Whole dense term 5.5291 s vs its 22 TF/s floor 5.48 s = +0.9%: the dense term
  is now essentially at its per-MAC ALU floor; the L2-excess lever is spent
  (0.073 s left of the 0.552). "Next dense lever = Candidate A (m-tile
  widening)" is consistent with the plan; its payoff remains UNMEASURED
  (tile-invariance untested) and is now capped small unless the v3a-style
  upside materializes - correct framing already in step-0/TASK.md.

## Findings

| id | sev | verdict | note |
|----|-----|---------|------|
| F-A3 | MAJOR | TP | "essentially the FULL L2 excess recovered" (99.1%) conflates the window dense-term delta with causal recovery. Corrected: kernel-only 92.8% (0.5119/0.5517), net of copies 89.3% (0.4927/0.5517); the 0.5466 contains +0.076 s ffn window-count artifact - 0.023 s drift. Decision unaffected (0.493 s still ~2x the NET window). |
| N-A6 | NIT | TP | "floor model REFUTED / cap did not bind" - the 19.6-19.9 cap belongs to the fused shape; halves are a different L2-fitting shape whose 21.8-22.4 cluster band predicted 21.91. Reframe as "conservative window exceeded; class-cluster model confirmed". |
| N-A1 | NIT | TP | md prose "1356" mul launches for nsplit2; sqlite recount 1396 (348.6/chunk). |
| N-A2 | NIT | TP | 21.65 TF/s is kernel-only pair-median; state basis (incl. copies = 21.76). |
| N-A5 | NIT | TP | "all within anchor classes" loose for rest (+3.4-5.6% vs 1.35 in both boots). |
| N-A3 | NIT | TP | 92.7% vs 92.8% rounding (0.552 vs 0.5517 bound). |
| N-A7 | NIT | TP | "2.63 s at cluster rate" = measured kernels; constructed cluster floor ~2.59 s. |
| - | record | - | Uninstrumented plain boot ran 553.8 tok/s vs 547.24 instrumented: CUPTI overhead ~1.2% depresses both A/B boots symmetrically; deltas valid, absolute e2e % may differ in production. |
| - | FP | - | Orchestrator suspicion "0.5466/0.5517 = 99.1%, not 92.7%" as a misattachment claim: the 92.7% correctly attaches to the kernel-only 0.5119. The real issue is the adjacent "full excess recovered" phrasing (F-A3). The suggested 95.6% correction is itself wrong (double-counts copies). |

## Overall verdict

The A/B measurement is VALID and SUPPORTS the verdict (NSPLIT=2 PAYS NET /
SHIP): single variable verified at the process-env level; T44 PASS with
non-overlapping steady bands; T43 median discipline correctly implemented;
T46 liveness exact (34 -> 68/chunk, gridX 778 -> 389, zero old-shape
launches); analyzer numbers reproduced independently from the raw sqlite;
bitwise 6/6 same-mode; decode flat and structurally explained. The judged
effect is real and exceeds the NET window even after the honest correction
(0.493 s net causal = 89.3% of the 0.5517 s bound vs the 0.22-0.26 s NET
target), and e2e +2.61% is conservative (unfavorable 0.170 s inter-boot
drift in MoE/fetch/rest, verified to the millisecond). Required fix before
the artifact is quoted onward: correct the recovery framing to 92.8%
kernel-only / 89.3% net-of-copies and drop "essentially the FULL excess";
optional NIT cleanups N-A1/2/5/7 and the A6 reframe. One scratch file was
created by the verification pass (recount_scratch.ipynb, read-only probes
only) - safe to delete.
