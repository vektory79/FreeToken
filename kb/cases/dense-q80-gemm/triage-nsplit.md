# Triage: Replan wave 1 (N-split kda_in_proj) review findings, 2026-09-19

Reviews: review-a-nsplit.md (SHIP) + review-b-nsplit.md (SHIP). No BLOCKER, no
MAJOR. Open TPs are cheap polish; the A/B wave may be prepared in parallel.

## Review-A findings

| id | sev | verdict | action |
|----|-----|---------|--------|
| NA-F1 | NIT | TP | One-line comment reword at gguf.py:146-148: need_check is a perf guard keyed on N%32; exactness rests on disjoint 32-col tiles + k-only reduction + deterministic per-launch x-quant. Pre-commit polish. |
| NA-F2 | NIT | TP (partial) | Add cheap coverage: bitwise parity at M=7 and M=13 (batch>6, non-multiple-of-4), non-contiguous x through the split path, "" env value -> default 1 pin. SKIPPED with justification: q6_K-through-split case (dense linears are Q8_0-only in production, glm5_next/gguf.py:693 - unreachable), op-level GGUF integration test (covered on hardware by the A/B wave's T46 liveness assertion 34 -> 68 launches/chunk). |
| NA-F3 | NIT | record | ROCm mmq_y=128 note - informational only. |

## Review-B findings

| id | sev | verdict | action |
|----|-----|---------|--------|
| NB-F1 | NIT | TP | Durability before STOP GATE: copy /tmp/gate-nsplit.log into the campaign folder with an invocation header (command/date/HEAD/env); record census snapshot; record the copy-overhead figure (16.8-18.2 ms/chunk = 0.10-0.12% of chunk) and the GROSS-vs-NET target framing. |
| NB-F2 | NIT | TP (record + memory) | Collection-nodeid drift: test_quant_config.py parametrizes at collection time over globbed model dirs (HF cache + /mnt/nvme/models) -> observed 2473 vs 2410 nodeids with identical failure sets. Orchestrator appends to memory ft-serve-test-and-e2e-gotchas; run-2 + --collect-only authoritative is JUSTIFIED. |
| NB-F3 | NIT | TP (doc) | Target in durable artifacts is GROSS 0.24-0.28 s; NET ~0.22-0.26 s after copy overhead - record alongside NB-F1. |
| NB-F4 | NIT | deferred | No test pins the GGUF-path wiring end-to-end - accepted: the A/B wave's T46 nsys liveness assertion (kda launches 34 -> 68/chunk, knob=2 boot) is the on-hardware wiring pin. Recorded here so the A/B brief carries it. |
| FP1 | FP | no action | Env fail-fast also fires on the non-GGUF path - by design (knob validation precedes scope gates). |

## Cross-cutting confirmations worth keeping

- Full-gate 16 failures verified per-test from /tmp/gate-nsplit.log: exact
  baseline composition (11x import class, pinned_tensor UVA, qsa_fp8 [1-1-64],
  qwen4_ple :421, 2x glm_dsa OOR); zero non-baseline failures. Baseline drift:
  3x test_prefill_hit_d2d did NOT fire this run.
- B6: decode graphs capture through the modified call site -> knob=2 changes
  baked graph contents -> the re-capture + decode re-measure mandate is real.
- B7: copies = 68/chunk x 404.7 MB = 27.5 GB = 15.3-16.8 ms ~= "~15 ms" claim.

## Status

Open TPs -> fix wave (comment reword, 3 tests, artifact durability). STOP
GATE of wave 1 closes when the fix wave reports; commit remains user-gated
and is expected only after the A/B wave validates. A/B wave brief must carry:
net-vs-gross framing, T46 liveness with the 34 -> 68 assertion, decode
re-measure, cross-boot bitwise output diff on a few prompts.

## STOP GATE (wave 1: N-split) - CLOSED 2026-09-19

Fix wave complete (NA-F1, NA-F2 partial, NB-F1, NB-F3):
- comment reworded at layers/gguf.py:146-149 (need_check = perf guard, not
  an exactness condition);
- tests added: M=7/13 bitwise parity, non-contiguous x parity, "" env pin;
  tests/kernels/test_gguf_quant.py 76/76 passed, tests/models/
  test_glm5_next_kda_op.py 4/4 passed (serialized, exit 0); full gate not
  re-run (comment+tests only; A/B boot exercises the serving path);
- gate-nsplit-run2.log (796 lines, invocation header, HEAD cb8db23) +
  wave1-notes.md written (self-sufficient A/B anchor: knob contract, test
  inventory, copy-overhead arithmetic 15.3-16.8 ms/chunk = 0.10-0.12%,
  GROSS 0.24-0.28 s vs NET ~0.22-0.26 s, judge-on-NET rule);
- NA-F2 skipped items (q6_K-through-split, op-level GGUF integration)
  remain skipped with recorded justification (dense Q8_0-only in
  production; T46 A/B liveness covers the wiring on hardware).

Gate items: findings triaged (no open TP); Test executed (76+4) with the
wave-1 full gate previously run (16 Environment-classified baseline);
TRAPS consulted (T45/T46); commit USER-GATED - deferred until the A/B
validates; artifacts recorded; census clean; TASK.md updated.

A/B wave in flight. NOTE: HEAD moved 1706aae -> cb8db23 mid-campaign
(origin to be identified in the A/B preflight; internal A/B on one tree is
unaffected).

## A/B wave review triage (2026-09-19)

Reviews: review-a-abnsplit.md + review-b-abnsplit.md. Measurement VALID;
verdict SHIP / PAYS-NET stands. Triage:

- RA-A3 MAJOR: recovery FRAMING (the arithmetic itself was correct). The
  report's 92.7% = kernel-only 0.5119/0.5517 = 92.8%, correctly attached;
  the real error is the adjacent "recovered essentially the FULL excess"
  (0.5466/0.5517 = 99.1% is non-causal: +0.076 s dense-FFN window-count
  artifact, -0.023 s drift). CORRECTED: kernel-only 92.8% (0.512 s), net of
  copies 89.3% (0.493 s). The 95.6% suggested at review time was WRONG
  (double-counts copies) - FP. -> TP: quote 92.8%/89.3%, drop "full excess".
- RB-M1 MAJOR: D1 discriminator reword - the "Collecting data" banner is
  wrapper-config-dependent (absent from instrumented nsplit logs too);
  anchor the no-collection check on report presence + sqlite kernel counts.
  -> TP.
- NITs -> TP (doc/ops, fix wave): A6 reframe (halves at 21.91 TF/s sit
  INSIDE the predicted 21.8-22.4 cluster band - "conservative window
  exceeded", not "cap wrong"); delete recount_scratch.ipynb; census line
  1356 vs ~1396; plain decode 15.46 framed as CUPTI overhead (~1.2%
  symmetric); VRAM typo 1469 -> 1394; cross-mode shas into results.json;
  watchdog/rc=1 polish notes.
- Record: CUPTI overhead ~1.2% symmetric (plain 553.8 vs instrumented
  547.24 prefill).

CORRECTED HEADLINE: N-split PAYS NET +2.61% e2e prefill; kernel-only
recovery 92.8% of the L2 excess; net of copies 89.3% (0.493 s) >> the NET
0.22-0.26 s window. L2 lever spent; Candidate A is the next dense lever.

## STOP GATE (A/B wave) - CLOSED with the doc-fix wave, 2026-09-19

Gate items: measurement validated by Review-A (independent sqlite recount)
and Review-B; findings triaged (2 MAJOR doc rewords + NITs -> doc-fix wave,
no code changes required); Test N/A (measurement-only); commit USER-GATED
(decision requested); artifacts recorded (ab-nsplit.md, ab-nsplit-results.json,
nsys rep/sqlite x2, run jsons, logs); census clean; deviations documented
with corrected evidence anchors.

User decision requested (2026-09-19): commit N-split (default flip
included or not), then proceed to Candidate A per the sequential plan.

## Sweep-wave (Candidate A) triage + STOP GATE - CLOSED 2026-09-20

Reviews: review-a-sweep.md + review-b-sweep.md - sweep VALID, winner tile 64
(+40.46% same-mode) stands. Triage:
- RA-1/2/3 + RB-F1 MAJORs (doc-level): corrected rate series (21.9 -> 88.2
  TF/s = 4.03x; the 176.5 figure was a 2x fused-FLOP error exceeding the
  FP32 peak), bridge sentence aligned (e2e -4.152 vs dense -4.129 = 0.023
  residual), foreign-serve narrative corrected (overlap = tiles 16/32/64/4b
  only; "30 GB co-residency" impossible - device-free unchanged 4.03-4.15
  GiB; mtile4b de-risk: reproduces tile-4 within 0.35% -> interference
  <0.5% << the 2.88% tile32-vs-64 gap) -> TP, APPLIED by the doc-fix wave
  (incl. two analyzer unit bugs: /1e3->/1e9 per-class and GFLOP/s under a
  TFLOP/s label; JSON regenerated, 21.91/35.20/53.64/74.25/88.23 series).
- RB-F2 MAJOR (lesson): single-boot 6-prompt e2e bitwise is boot-to-boot
  NONDETERMINISTIC - not usable as a gate; first same-config-pair
  observation (tile4-unset vs tile4b diverges on prompts 2/3/4; tile4b==
  tile64 6/6). -> recorded in ab-mtile.md + campaign memory; quiet-box
  arbitration (2x64 + 2x4) = user option, NOT a blocker.
- RB-F3: one-off 58 s JIT rebuild at mtile16 - recorded, cause
  unidentified, not perf-relevant.
- NITs F4-F8 -> APPLIED (json units, six-way wording, wave2 glm_dsa class
  note, anchor calibration 346-350, census archival note).
- FP (both reviews): bitwise divergence = boot nondeterminism, not
  tile-caused.

Gate items: T43/T44/T46 PASS (corrected anchors); findings triaged, TPs
fixed; Test N/A (measurement wave; implementation gate logged in
gate-candidate-a.log); TRAPS consulted; commit USER-GATED (Candidate A
changeset uncommitted on 4505572); artifacts recorded; census clean;
TASK.md updated with the measured outcome.

Open user decisions: commit Candidate A; default flip DENSE_MTILE=64;
pyproject ipykernel line (keep as committed / revert); optional arbitration
battery (2x64 + 2x4, quiet box).

## FINAL STOP GATE (campaign) - CLOSED 2026-09-20

User decisions executed: commit Candidate A + default flip 64 (arbitration
battery completed first per user sequencing; pyproject + MoE default left
unanswered -> untouched).

- Commit c464273a: "perf(kernels): widen dense q8_0 mmq tile via
  FREETOKEN_GGUF_DENSE_MTILE" (5 files +214/-19; staged exactly 5 paths;
  tests 80/0 pre-commit; message byte-exact).
- Commit 0d81cba: "perf(kernels): default FREETOKEN_GGUF_DENSE_MTILE=64"
  (4 files +22/-12; one intra-wave fix - the flip initially broke the
  explicit-"4" case, restructured to unset->64 + separate "4" branch;
  tests 80/0 re-run green pre-commit; kernel/gguf.py docstring consistency
  fix included). 2 JIT rebuilds total (~45 s each). NO push.
- Production serving defaults: DENSE_MTILE=64 + NSPLIT=2 (committed) +
  MOE_MTILE=32 (user env) -> ~792-808 tok/s @8128 prefill class
  (instrumented; plain ~+1.2%); decode 13.2-15.0.
- Campaign ledger: Step 0 (ALU-bound verdict) -> wave 1 N-split (+2.61%,
  5254ffa + 9f09596) -> wave 2 Candidate A (+40.46%, c464273a + 0d81cba);
  all reviews SHIP; all STOP GATEs closed; artifacts complete under
  .tasks/dense-q80-gemm/.
- Open items (user, unanswered): pyproject ipykernel line; MoE MTILE default
  flip 4->32 (v3a leftover); session-end skill self-update in flight (never
  commits itself).
- Residual ops notes: single-boot e2e bitwise gate stays dead (bimodal boot
  nondeterminism, arbitration battery verdict); e2e prefill class now
  ~792-808 (any future T44 gate keys on this anchor).

## Arbitration wave - CLOSED 2026-09-20

Wave: arbitration battery (the quiet-box option from the sweep reviews;
user-approved as a pre-commit step). Verification: orchestrator single-stage
direct read of the decisive sha matrix (waiver rationale: the decisive fact
is a sha equality check on recorded artifacts, not arithmetic; both sweep
reviews framed the question and recommended exactly this experiment).
Verdict verified from arbitration-battery.md + arbitration-results.json:
- cross-config Mode-B pairs 24/24 IDENTICAL (t4b~t64b, t64a~t4b, t64a~t64b)
  -> tile-invariance holds e2e at 24-prompt granularity; tile 4 == tile 64
  outputs;
- e2e nondeterminism is bimodal (Mode A = {t4a}, 22/24 flips vs every
  Mode-B boot; same-config t64a~t64b 24/24); the tile does not select the
  mode;
- flip anatomy: 20x mid-answer greedy-cascade in reasoning_content, 2x
  suffix-flip; p04/p20 stable;
- prefill sanity: t4 578.42/578.83, t64 808.15/808.12 - tile effect
  reproduced, no drift.
Quality question RESOLVED - the default-flip decision is de-risked.
Harness lesson recorded: FAILPAT false-positive on the normal teardown line
SIGKILLed attempt-1 (watchdog now disarmed at teardown; 4 sem.mp- leaked,
8->12, documented). Gate: Test N/A (sha matrix); census file-logged per
boot (closes the archival gap); commit user-gated.
