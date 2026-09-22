# Arbitration battery: boot nondeterminism vs tile-4/64 output agreement (24 prompts)

Date: 2026-09-19/20. Hardware: single RTX 5090, GLM-5.3-Flash-UD-Q3_K_XL, port 18801.
Status: VALIDATED. Full per-prompt data: [../baselines/dense-q80-gemm/quality-arbitration/](../baselines/dense-q80-gemm/quality-arbitration/).

## Verdict (read this first)

- The 6-pair flip matrix is exactly a **two-mode boot signature, not a tile
  effect**: Mode A = {t4a} alone, Mode B = {t4b, t64a, t64b} with ALL intra-mode
  pairs bitwise identical 24/24, including both cross-config pairs.
- **22/24 prompts** diverge between modes (p04 and p20 are bitwise stable in all
  four boots); quantified boot nondeterminism: 11/24 per boot pair (45.8%).
- First divergence is ALWAYS inside `reasoning_content` (20x mid-answer greedy
  cascade, 2x suffix-flip p07/p21) - never whole-answer-from-first-char.
- Tile-invariance holds e2e: cross-config flips do not exceed same-config
  (boot-noise) flips; the tile config adds NOTHING beyond boot-to-boot
  nondeterminism. The tile does not select the mode (Mode A and B both occur
  under tile 4: t4a=A, t4b=B; both boots under tile 64 landed in B).
- Consequence for quality A/Bs on this rig: a single-boot comparison has a
  45.8% boot-noise flip floor; qualify boots by mode before attributing any
  output divergence to a config change.

## Method summary

4 plain boots (no nsys, single instrumentation mode), strictly serialized and
reaped, interleaved t4a/t64a/t4b/t64b. Per boot: census -> quiet-box gate ->
boot (FREETOKEN_GGUF_DENSE_MTILE=<tile>, FREETOKEN_GGUF_MOE_MTILE=32, NSPLIT
unset = committed default 2, winner serving flags, port 18801) -> /ready ->
65,585-token fill -> 24-prompt greedy battery (p00..p23, reasoning+content
verbatim + sha256) -> SIGTERM/SIGKILL -> census.

Note: the earlier 6-prompt sweep battery used a DIFFERENT prompt set
(req_mtile_*.json), so its per-prompt flips cannot be mapped by index here;
the comparison is the flip rate and flip-position instability.

## Compact results

| quantity | value |
|---|---|
| prefill anchor t4 (median c2..c7) | 578.4 tok/s (anchor 563.9, +2.6%) |
| prefill anchor t64 (median c2..c7) | 808.1 tok/s (anchor 792.08, +2.0%) |
| same-config flips: t4a~t4b / t64a~t64b | 22/24 vs 0/24 |
| cross-config flips: t4a~t64a, t4a~t64b, t64a~t4b, t4b~t64b | 22, 22, 0, 0 (of 24) |
| output modes | A = {t4a}, B = {t4b, t64a, t64b} |
| per-prompt flip frequency between modes | 22/24 (p04, p20 stable) |
| divergence class (per divergent pair) | 20x mid-answer (LCP 68-430 chars, tail delta ~1-9%), 2x suffix-flip (p07, p21); first divergence always in reasoning |
| boot census | ready 117.1-144.3 s, gpu_after_boot ~28.0 GiB, teardown rc=-15 (graceful SIGTERM by design), leftovers none |

Attempt-1 note: an earlier t4a boot completed its 24/24 battery but the
watchdog false-positived on the NORMAL teardown line "backend worker
freetoken-detokenizer-0 exited" and SIGKILLed the group during graceful
shutdown; its outputs already showed the same phenomenon and were excluded -
the wave was relaunched with the watchdog disarmed during teardown. census.log
keeps attempt-1 lines.

## Full data

Per-prompt flip matrix, pairwise sha matrices, LCP/LCS divergence tables and
raw battery outputs live in
[../baselines/dense-q80-gemm/quality-arbitration/](../baselines/dense-q80-gemm/quality-arbitration/)
(machine-readable twin:
[arbitration-results.json](../baselines/dense-q80-gemm/arbitration-results.json)).
Runner/analyze scripts: [../harness/ab-runner/arbitration_run.py](../harness/ab-runner/arbitration_run.py),
[arbitration_analyze.py](../harness/ab-runner/arbitration_analyze.py).