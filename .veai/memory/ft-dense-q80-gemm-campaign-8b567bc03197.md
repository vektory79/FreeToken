---
name: "ft-dense-q80-gemm-campaign"
description: "dense-q80-gemm campaign: q8_0 ALU-bound; N-split +2.61%; dense tile64 +40.46%; all committed (HEAD 8e2e4c7)"
type: project
lastUpdated: 2026-09-20T12:37
lastRecall: 2026-09-20T14:22
---

# dense-q80-gemm campaign: dense q8_0 ALU-bound; N-split +2.61% committed; Candidate A tile64 +40.46%

Campaign .tasks/dense-q80-gemm (follow-up of v3a; branch vektory79; private-use, commit user-gated). USER DECISION 2026-09-19: sequential N-split -> Candidate A; Candidate B (int8 mma, D06 gate) stays gated behind A.

## Durable lessons
- Step-0 regime: dense q8_0 (mul_mat_q8_0) is ALU/issue-bound ~22 TF/s per-MAC, NOT BW-bound (naive traffic model 9.78 s vs measured 6.04 s = 0.62x; moe_vec precedent closed 0.92x; proofs: M-sweep linear at constant rate -> implied BW > peak = impossible; q6_K fewer-bytes-but-slower). L2 = 96.0 MiB device-verified. kda_in_proj (W=108.35 MB > L2) ran 18.08 TF/s - the only DRAM-affected class. Census gotchas: indexer wq_b loaded bf16 (not on the MMQ path), kv_b dequantized for MLA bmm absorption.
- Analysis-script trap: per-chunk vs per-launch mixing + inverted recovery produced a projection BELOW its own ALU floor. Absurdity check (a projection under its own floor = broken analysis) + derive floors from measured op sums, never constants.
- Floor-model humility: BOTH widening projections were wrong. The conservative floor model (+1.9..+3.6%) was refuted ~10x; the v3a class-cluster model (rate scales with tile) was CONFIRMED.

## Wave 1: N-split (kda_in_proj) - COMMITTED
- 5254ffa "perf(gguf): split kda in_proj gemm under L2 via FREETOKEN_GGUF_KDA_NSPLIT" (+293/-2, 4 files); 9f09596 default-flip NSPLIT=2. Mechanism: per-call env knob (unset->2, "1"=single, strict fail-fast), split 2x12448 at the 32-col boundary, preallocated out + two narrow copy_ (~17 ms/chunk), gates = GGUFLinear + M>6 + _MMQ + N%32==0 + N>=64 (matches fused_mul_mat_gguf dispatch exactly; n_tok IS the GEMM M). Measured: 547.24 -> 561.50 tok/s (+2.61%), kernel-only recovery 92.8% of the 0.55 s L2 excess (net 89.3%), decode flat, bitwise same-mode cross-boot. NOT pushed.
- pyproject.toml carries a stray committed "+ipykernel>=7.3.0" (landed in parallel-session commit 4505572 "Memory"; keep/revert = user decision, pending).

## Wave 2: Candidate A (dense m-tile) - sweep measured, COMMIT PENDING
- Changeset UNCOMMITTED: mmq.cuh mul_mat_q8_0 templated on token-tile {4,8,16,32,64} (static_asserts mmq_x>=nwarps && mmq_x%nwarps==0; ROCm zero-length hazard kept single-tile), gguf_kernel.cu ft_gguf_dense_mtile() single-source knob FREETOKEN_GGUF_DENSE_MTILE (per-call getenv, unset->4, 8/16/32/64, strict fail-fast, case-8-only scope; "do not flip after boot" - decode graphs capture dense MMQ bs>6), probe ggml_dense_get_mtile, layers/moe.py docstring documents both knobs, tests +108 (bitwise torch.equal tile-invariance incl. M-tail/need_check/NSPLIT composition; knob unit; MMVQ cutoff unchanged). PTXAS zero spills all tiles (SMEM 5.3-14 KB). Full gate 16 failed / 2179 passed / 206 skipped (all Environment; same FILE set as wave 1: 11x import class + 2x glm_dsa OOR + pinned UVA + qsa_fp8 + ple; wave2 notes' "OOR x3" was a misread).
- SWEEP (5 boots + mtile4b discriminator; T43/T44/T46; reviews SHIP): tile 4 = 563.90 tok/s (T44 +0.43% PASS) / 8 = 658.45 (+16.77%) / 16 = 726.37 (+28.81%) / 32 = 769.95 (+36.54%) / 64 = 792.08 (+40.46%); dense term 5.52 -> 1.39 s/chunk; dense rate 21.9 -> 88.2 TF/s (4.03x for the 16x re-read cut; the md's "176.5 TF/s" was a 2x fused-FLOP error, corrected by review-a-sweep); MoE 4.36-4.40 / fetch 3.07-3.14 / rest tile-invariant; bridge residual 0.023 s. Decode flat (bs=1 MMVQ graph - neither knob enters). T46 PASS per boot (dense 346-350/chunk const, kda 68 gridX=389, grid.y exactly 2032/1016/508/254/127).
- Bitwise lesson (first same-config-pair test ever): single-boot 6-prompt e2e bitwise is boot-to-boot NONDETERMINISTIC (tile4-unset vs tile4b-explicit diverges on prompts 2/3/4; tile4b==tile64 6/6; source unidentified - CPU MoE threads/atomics plausible, unproven). NEVER use single-boot e2e bitwise as a gate; torch.equal unit proofs are the quality evidence; optional quiet-box arbitration (2x tile64 + 2x tile4) only if the user wants the e2e gate rehabilitated. (v3a 24/24 and N-split 6/6 were cross-config pairs - same-config was never tested before.)
- Foreign-serve correction (review-a-sweep A5 + review-b-sweep B2): the llama-swap ft serve (18084) overlapped ONLY tiles 16/32/64/4b; "30 GB co-residency both fit" was impossible (26+30 > 31.4 GiB; device-free 4.03-4.15 GiB unchanged; mtile4b gpu_after_boot ~28 GB = the sweep's own boot alone); mtile4b reproduces tile-4 within 0.35% -> interference <0.5% << the 2.88% tile32-vs-64 gap -> ranking de-risked, no clean re-run needed. One-off 58 s JIT rebuild at mtile16 (cause unidentified, not tile/foreign-driven, not perf-relevant - mtile4b reproduces tile-4).
- Ops leftovers: the ~356 launch anchor is miscalibrated (actual 346-350; constancy + grid.y carry the proof); analyzer per-class s_per_chunk fields still microsecond-labeled (doc-fix pending); census/stage raw outputs unarchived (prose only).

## Open user decisions (2026-09-20)
Commit Candidate A; default flip FREETOKEN_GGUF_DENSE_MTILE=64 (winner) or keep 4; pyproject ipykernel line keep/revert; optional quiet-box bitwise arbitration battery.

## Arbitration battery (2026-09-20, user-approved pre-commit) - VERDICT
- 4 plain boots interleaved (t4a/t64a/t4b/t64b), 24 prompts each, census file-logged per boot. VERIFIED by direct artifact read (orchestrator single-stage verification in lieu of Review x2 - the decisive fact is a sha equality check on artifacts, not arithmetic; both sweep reviews framed the question and recommended exactly this experiment): cross-config pairs in the same boot-mode are 24/24 IDENTICAL (t4b~t64b, t64a~t4b, t64a~t64b) -> tile-invariance holds e2e at 24-prompt granularity; tile 4 == tile 64 outputs.
- e2e nondeterminism is REAL and BIMODAL: Mode A = {t4a} diverges from every Mode-B boot on the same 22/24 prompts (t4a~t4b 2/24 identical); the tile does not select the mode (both modes occurred under tile 4). Flip anatomy: 20x mid-answer greedy-cascade inside reasoning_content (first divergence ALWAYS in reasoning - never first-char, never pure truncation), 2x suffix-flip (p07/p21); p04/p20 stable in all boots. e2e single-boot bitwise stays DEAD as a gate (lesson re-confirmed at higher granularity).
- Prefill sanity: t4 578.42/578.83 (+2.6% vs the 563.90 anchor), t64 808.15/808.12 (+2.03% vs 792.08) - tile effect reproduced, no environment drift.
- Harness lesson: the FAILPAT watchdog false-positived on the NORMAL teardown line "backend worker freetoken-detokenizer-0 exited" and SIGKILLed during graceful shutdown (attempt-1 aborted; 4 sem.mp- leaked 8->12) - disarm the watchdog at teardown before graceful shutdown completes; fixed in arbitration_run.py.
- Artifacts: arbitration-battery.md, arbitration-results.json, quality-arbitration/.

## COMMITTED (2026-09-20, user request)
c464273a "perf(kernels): widen dense q8_0 mmq tile via FREETOKEN_GGUF_DENSE_MTILE" (5 files +214/-19) + 0d81cba "perf(kernels): default FREETOKEN_GGUF_DENSE_MTILE=64" (4 files +22/-12; incl. kernel/gguf.py docstring consistency fix). One intra-wave fix before commit: the flip initially broke the explicit-"4" case -> restructured to unset->64 + separate explicit-"4" branch; tests 80/0 green pre-commit both stages. NOT pushed. Production defaults now: DENSE_MTILE=64 + NSPLIT=2 (committed) + MOE_MTILE=32 (user env) -> ~792-808 tok/s @8128 prefill class. All campaign STOP GATEs closed (triage-step0.md / triage-nsplit.md). Open (user, unanswered): pyproject ipykernel line (left as committed by the parallel session); MoE MTILE default flip 4->32 (v3a leftover).

## CLOSED (2026-09-20)
All three closing items executed: skill docs committed e277d4b; MoE MTILE default flipped 4->32 via 8e2e4c7 (unset/empty->32, separate explicit "4" branch, strict whitelist; moe.py docstring + knob tests updated; cross-knob check in test_dense_mtile_env_knob synced to 32; tests 129/0 across test_gguf_quant.py + test_gguf_expert_banks.py, the cap test already pinned env explicitly; one JIT rebuild); pyproject left as committed. HEAD 8e2e4c7, nothing pushed. Campaign fully closed.
