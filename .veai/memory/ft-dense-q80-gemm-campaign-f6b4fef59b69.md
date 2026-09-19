---
name: "ft-dense-q80-gemm-campaign"
description: "dense-q80-gemm campaign: q8_0 ALU-bound; N-split +2.61% e2e (92.8% recovery); commit+flip pending; Candidate A next"
type: project
lastUpdated: 2026-09-19T21:03
lastRecall: 2026-09-19T21:05
---

# dense-q80-gemm campaign: dense q8_0 ALU-bound; N-split measured +2.61% e2e

Campaign .tasks/dense-q80-gemm (follow-up of v3a; branch vektory79 @ 1706aae; private-use, commit user-gated). USER DECISION 2026-09-19: sequential - N-split first, then Candidate A A/B; Candidate B (int8 mma, D06 gate) stays gated.

## Step 0 VERDICT (2026-09-19, MTILE=32 anchor)
Dense q8_0 (mul_mat_q8_0, 6.04 s/chunk, 322 launches) is ALU/issue-bound at ~22 TF/s per-MAC, NOT BW-bound: naive traffic model 9.78 s vs measured 6.04 s = 0.62x (moe_vec precedent closed at 0.92x). Proofs: measured/model uniform 0.550-0.592 across all classes (ssm 0.592 outlier); M-sweep on the in_proj shape linear in time at constant 18.1-19.9 TF/s -> implied BW 2413-2660 GB/s > 1792 peak = impossible; q6_K at identical shape/MACs has 23% fewer bytes but is 6.3% SLOWER. bytes/MAC = 0.2656 (0.1328 B/FLOP). L2 = 96.0 MiB (100,663,296 B, device-verified). kda_in_proj exception: W=108.35 MB > L2 -> 18.08 TF/s; all W<=71.3 MB classes run 20.9-22.4 TF/s.
Chunk split: 14.87 s = dense 6.04 + grouped MoE 4.37 + fetch copies 3.11 + rest 1.35; boot 546.5 tok/s (T44 -1.77% PASS vs 556.43). Census gotchas: indexer wq_b NOT on the MMQ path (Q8_0 on disk but loaded bf16, weight.py:147-151); kv_b dequantized for MLA bmm absorption (attention.py:133-151); one q_a (48,2032) per MLA layer.

## ANALYSIS-SCRIPT TRAP (cost a BLOCKER + a fix wave)
The first step0 projection mixed per-chunk and per-launch numbers (kda 3.1136 s/chunk vs 75.3 ms/launch, missing x34) and inverted residual/recovery -> step0-split.json claimed tile8 "best" at 608.8 tok/s with dense 4.518 s BELOW the kernel's own 5.48 s ALU floor. Lesson: every projection artifact needs an absurdity check (a projection below its own floor = broken analysis); derive floors/totals from measured op sums, never hardcoded constants (117.8 vs real 120.7 TF); regenerate from raw medians with per-launch x launch-count. Fixed final_split.py: one command regenerates step0-split.json + step0-projection-table.md, byte-identical to the profile table.

## Corrected floor model + Candidate A status
in_proj L2 excess = 34 x (91.577-75.350 ms) = 0.552 s/chunk; widening floor-model projections tile 8/16/32/64 -> +1.9/+2.9/+3.4/+3.6% e2e (full recovery ~+3.9%) - but the PAYOFF IS UNMEASURED: tile-invariance of the 22 TF/s rate is untested and the v3a sibling family grew 13.7 -> 39.6 TF/s from tile 4 -> 32, so upside plausibly much larger (review-a F3). Decide via cheap A/B (T52), never on paper (T35).

## N-split (Replan wave 1): IMPLEMENTED, Review-A + Review-B SHIP, fix wave applied
+241/-2 python-only (plus fix wave -> ~+293/-2): layers/gguf.py _kda_nsplit_env() + kda_in_proj_forward() (~:112-170); kda.py:111-116 routes through it. Knob FREETOKEN_GGUF_KDA_NSPLIT, per-call env read (T45), default 1, "2" = split 2x12448 (= 2x(389x32), ~54.2 MB < L2); gates = GGUFLinear AND n_tok>6 (n_tok IS the GEMM M) AND type in _MMQ AND N%32==0 AND N>=64 - matches fused_mul_mat_gguf's MMQ dispatch exactly; boundary (N//2)//32*32 keeps need_check=false (a perf guard keyed on N%32, NOT an exactness condition - comment reword APPLIED). Assembly: preallocated [M,N] out + two narrow copy_ (extension API has NO out= param) -> ~17 ms/chunk = 0.10-0.12% of the chunk. Bitwise parity via torch.equal pinned (production N=24896 + asymmetric [32,64] + ragged fallback + decode batch=4 stays MMVQ; fix wave added M=7/13 non-multiple-of-4 parity + non-contiguous x + "" env pin; q6_K split + op-level integration SKIPPED with justification: production dense Q8_0-only, glm5_next/gguf.py:693, T46 A/B liveness covers the wiring). Full gate: 16 failed / 2173 passed / 206 skipped - all 16 Environment (stash-restored tree re-ran identical); collection-count anomaly 2473 vs 2410 nodeids (see ft-pytest-worktree-baseline-gotchas).
Decode caveat: dense MMQ is captured in decode CUDA graphs for bs>6 -> the knob changes graph contents; "do not flip after boot".

## A/B MEASURED + VALIDATED (2026-09-19)
- NSPLIT 1 vs 2, two boots (MTILE=32, port 18801, serialized+reaped): prefill 547.24 -> 561.50 tok/s (+2.61%), chunk 14.853 -> 14.476 s; T44 PASS (+0.14% vs the morning 546.5 anchor). T46 wiring pin PASS: kda launches 34.18 -> 67.92/chunk (fused gridX 778 -> halves 389), per-call 91.744 -> 76.58 ms = 21.65 TF/s - the cluster band, PREDICTED (the "cap refuted" wording was wrong: 19.6-19.9 belongs to the fused 108 MB shape; W=54.2 MB halves sit at 21.8-22.4). Dense term 6.095 -> 5.548 s incl. 19.2 ms/chunk assembly copies. Decode 13.35 vs 13.61 flat (bs=1 MMVQ graph at max-running-requests=1; knob cannot enter the graph). Bitwise 6/6 same-mode cross-boot; cross-MODE (plain vs instrumented boot) diverges 2/6 - bound bitwise claims to same-mode.
- CORRECTED recovery (review-a-abnsplit.md, independent sqlite recount): kernel-only 92.8% (0.5119 s of the 0.5517 s excess); net of copies 89.3% (0.493 s) >> the NET 0.22-0.26 s window. "Essentially the full excess" (99.1%) was NON-CAUSAL (+0.076 s dense-FFN window artifact, -0.023 s drift). Drift bridge closes exactly: +0.17 s = MoE 0.0884 + fetch 0.0529 + rest 0.0287. L2 lever SPENT; Candidate A is the next dense lever.
- Reviews: review-a-nsplit.md + review-b-nsplit.md (SHIP), review-a-abnsplit.md + review-b-abnsplit.md (measurement valid, verdict stands). Doc-fix wave applied corrected numbers + the D1 reword into ab-nsplit.md / ab-nsplit-results.json.
- Ops traps: nsys silent no-collection (rc=0 != collection; report presence is the only verified signal; the "Collecting data" banner is wrapper-config-dependent - see memory nsys-silent-no-collection-trap); CUPTI overhead ~1.2% symmetric (plain 553.8 vs instrumented 547.24 prefill; plain decode 15.46).

## Commit state + tree notes (2026-09-19)
- USER DECISION: commit N-split, then a default-flip micro-commit (NSPLIT=2), then Candidate A.
- Commit 1 = the 4-path changeset (layers/gguf.py, models/glm5_next/kda.py, tests/kernels/test_gguf_quant.py, tests/models/test_glm5_next_kda_op.py) with the measured body; a first attempt correctly ABORTED on a stray worktree modification (pyproject.toml "+ipykernel>=7.3.0" core dep, origin unknown, NOT part of the campaign) - relaunched with explicit 4-path authorization, pyproject.toml left untouched.
- HEAD moved 1706aae -> cb8db23 mid-campaign: cb8db23 = .veai-only "Memory" commit, parent 2d64a58 "Skill" (parallel-session bookkeeping; python tree == 1706aae + this changeset).
