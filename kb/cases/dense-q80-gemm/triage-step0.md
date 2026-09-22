# Triage: Step-0 wave findings (Review-A + Review-B), 2026-09-19

Summary: the Step-0 REGIME classification stands (dense q8_0 ALU/issue-bound at
tile 4, NOT BW-bound; kda_in_proj L2-overflow exception real, but its -17.5%
DRAM-stall attribution overstated). The PROJECTION/DECISION layer does not
stand as recorded: the machine-readable deliverable is corrupt (F1) and the
headline bounds are paper numbers (F2/F4, T35 pattern). All decision-facing
numbers reframed; fixes delegated to the fix wave.

## Findings

| id | sev | src | verdict | action |
|----|-----|-----|---------|--------|
| F1 | BLOCKER | A | TP | final_split.py mixes per-chunk (3.1136 s) with per-launch (75.3 ms; missing x34 for kda_in_proj 34x91.6ms); residual/recovery inverted (JSON: tile8 "best" 608.8 tok/s; tile8 dense 4.518 s < own 5.36 s ALU floor). Fix script; regenerate step0-split.json; cross-check vs profile tables. |
| F2 | MAJOR | A | TP | md projection table unreproducible by any script (T35). Make reproducible end-to-end (one deterministic command); regenerate md from script output. |
| F3 | MAJOR | A | TP (reframing) | De-prioritization of Candidate A rests on tile-invariance of the 22 TF/s rate - untested by the C3 M-sweep, contradicted by the v3a sibling family (13.7 -> 39.6 TF/s from tile 4 -> 32). Candidate A payoff = UNMEASURED; conservative floor ~+2-3% e2e, upside plausibly much larger; decide via cheap A/B (T52). |
| F4 | MAJOR | A | TP (reframing) | N-split ~0.5 s target ~2x optimistic (assumed recovery to the 22 TF/s cross-shape cluster; the same shape's own M-sweep caps at 19.6-19.9 TF/s) -> conservative 0.24-0.28 s. -17.5% in_proj attribution = mix of effects, overstated. |
| F5 | NIT | A | TP (doc) | 0.1328 is B/FLOP not "B/MAC" (0.2656 B/MAC). |
| F6 | NIT | A | TP (doc) | cluster band 0.550-0.592; ssm 0.592 outside the quoted 0.55-0.57 band. |
| F7 | NIT | A | record only | raw microbench outputs not saved; caveat in profile.md (no re-run). |
| F8 | NIT | A | TP (doc) | L2 size was asserted, not captured; record device props: L2 = 100,663,296 B = 96.0 MiB (Review-A device query, 2026-09-19). |
| F9 | NIT | A | TP (doc) | wording fixes. |
| N3 | NIT | B | TP (doc) | hardcoded 117.8 TF understates op_classes sum ~120.7; floor 5.36 -> 5.49 s (closes the slack vs the in_proj excess). |
| N5 | NIT | B | TP (doc) | wq_b K/N transposed + omitted from the dequant list. |
| N10 | NIT | B | TP (ops) | superseded step0_analyze.py would clobber the final step0-split.json if re-run: rename/guard. |
| N6/N8 | NIT | B | TP (doc) | json summary quirks: steady_mean includes the T43 artifact; boot_env empty. Fix generator or annotate. |
| rest | NIT | B | record | see review-b-step0.md; no re-runs needed. |

## Closed by the orchestrator before this file (B4, confirmed by Review-B)

- v3a-ab.md: CORRECTION section at file end + inline pointer at the "~1.8 s
  (e2e fit)" sentence (direct nsys 4.37 s; non-MoE share 70.6%, not ~80%;
  v3b no-fire + ALU/LDS-floor conclusions unaffected).
- dense-q80-gemm/TASK.md: anchor bullet corrected with a pointer to
  review-b-step0.md B4.
- Memory ft-gguf-v3-mtile-campaign: correction appended.

## Status

No FP findings. No BLOCKER/MAJOR against the regime classification itself.
Open TPs -> fix wave (in flight: script/JSON repair + doc reframing).
Step-0 STOP GATE closes when the fix wave reports; the campaign-lever decision
is presented to the user in parallel (it does not depend on the artifact
repairs, only on the reframed bounds above).

## STOP GATE (Step 0) - CLOSED 2026-09-19

Fix wave complete (all TPs addressed):
- F1: excess = 34 x (91.577 - 75.350 ms) = 0.552 s (per-chunk/per-launch mix
  removed); recovery = excess x (1 - 4/tile); floor/totals derived from
  op_classes (N3: floor 5.48 s, closes against the 0.55 s excess).
- F2: `python3 .tasks/dense-q80-gemm/final_split.py` regenerates
  step0-split.json + step0-projection-table.md; profile fragment
  byte-identical (TABLE_MATCH); command documented in step0-profile.md.
- F3/F4 reframing in step0-profile.md: Candidate A payoff UNMEASURED
  (floor model: tile8 +1.9%, tile16 +2.9%, tile32 +3.4%, tile64 +3.6%;
  full recovery ~+3.9%; tile-invariance of the 22 TF/s rate untested);
  N-split conservative 0.24-0.28 s, upper 0.55 s.
- N5/N6/N8/N10 + provenance: fixed or annotated per triage; N10 via
  step0_analyze_superseded.py with a write-guard (tested refusal rc=1);
  step0_analyze.py deleted (15 broken-ref reports verified false positives).
- JSON-vs-profile consistency: zero mismatches.

Gate items: validation gates pass (ncu optional, documented env blocker);
findings triaged, no open TP; Test WAIVED - measurement-only wave, changed
no behavior; TRAPS consulted (D09/D10, T43/T44/T46/T52); commit N/A
(measurement-only, results live as artifacts); artifacts recorded (paths
above); process census clean (pre==post); brief checkboxes updated
(TASK.md Replanning section).

Next: N-split implementation wave (in flight) -> Review x2 -> triage ->
fix TP -> STOP GATE -> A/B wave (two boots, MTILE=32 anchor).
Lever decision by the user (2026-09-19): sequential N-split, then
Candidate A A/B.
