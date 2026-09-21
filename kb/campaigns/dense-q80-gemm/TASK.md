# TASK: dense q8_0 GEMM - the next prefill lever after the v3a m-tile

Status: NOT STARTED. Composed 2026-09-19 from the v3a outcome
(`.tasks/mmq-v3-stationary-moe/TASK.md` sweep result + `v3a-ab.md`). Read
CONTRIBUTING.md first - it is binding. Line anchors verified 2026-09-19
against HEAD 1706aae; re-verify before editing (they drift). Private-use
scope for GGUF glm5next: no push, no upstream PR.

## Goal

Dense q8_0 GEMM (`mul_mat_q8_0`, ~312 launches/chunk: attn/gdn/dsa
projections, shared expert, leading dense) is the top single item after v3a:
6.04 s/chunk in the step0 profile, which is ~41% of the 14.61 s chunk at the
v3a winner config. Goal: cut the dense term toward its measured roofline.
If the traffic model holds (UNVERIFIED - Step 0 pins the regime first), the
ceiling is chunk ~10 s -> ~820-870 tok/s @8128; treat that number as a
projection to be re-anchored, never as a plan (T35).

## Verified background (do not re-derive; re-anchor before editing)

- ANCHOR CONFIG for this campaign: `FREETOKEN_GGUF_MOE_MTILE=32`
  (556.43 tok/s @8128 class; commit 1706aae). The knob DEFAULT remains 4 -
  flipping the production default to 32 is an OPEN user decision, separate
  micro-commit. All A/B boots here run with MTILE=32 held fixed; the only
  variable is the dense-side change.
- step0-profile.md (tile-4-era, 28.77 s chunk): mul_mat_q8_0 6.04 s =
  21.0%, ~312 launches/chunk (attn/gdn/dsa projections, shared expert,
  leading dense); rest of the split: MoE ~1.8 s at tile 32 (e2e-fit
  estimate, REFUTED 2026-09-19 by direct nsys: 4.37 s/chunk; non-MoE share
  70.6%, not ~80% - see review-b-step0.md B4), fetch copies 3.15 s,
  attention/GDN/DSA ~1.45 s.
- mmq.cuh:433-438: CUDA dense tile is `MMQ_X_Q8_0=4, MMQ_Y_Q8_0=32,
  NWARPS=4`; the ROCm branch runs `64 x 128, 8 warps`. The CUDA branch
  serves only 4 tokens per block-tile -> dense weights are re-read
  ceil(M/4) = 2032x per projection at M=8128. Same disease v3a cured for
  the grouped MoE path. NOT yet verified that dense is BW-bound (Step 0).
- Dense ds-fill is already full-coverage (mmq.cuh:77-88 - the loop the v3a
  moe port was adapted from); no moe_align trio involved (plain GEMM grid,
  no padding, no sentinel bins).
- Dispatch: dense batch >6 routes to `ggml_mul_mat_a8` (MMQ kernels);
  batch <=6 uses the MMVQ vec kernels (decode at bs<=6). CUDA-graph decode
  captures batches >6 too -> dense MMQ kernels ARE in decode graphs for
  larger capture batches. UNLIKE v3a (decode untouched), a dense-tile
  change alters decode-graph contents for bs>6: re-measure decode and
  verify graph re-capture; do NOT assume decode stays flat.
- v3a lessons that transfer: tile-invariance of outputs (per-element
  reduction order - battery was bitwise identical across tiles); the
  template + env-knob mechanism with a single source of truth; traps
  T40/T51 (audit silent bound assumptions in the vendored body before
  reusing it at new tile shapes) and T52 (traffic cuts pay sub-linearly
  once the kernel leaves the BW-bound regime - measure the regime first).

## Step 0 - pin the dense regime (first, before any design)

1. nsys split of the 14.61 s chunk under MTILE=32 (D09/D10 recipe): per
   dense kernel - exact names, launches/chunk, time; confirm the 6.04 s
   figure carries over; decompose the ~312 launches by op class with
   M/N/K per class (shapes from the model definition, not guessed).
2. Traffic model: dense weight bytes/layer from the real GGUF header
   (gguf-py scan); expected traffic = Sum_op ceil(M/MMQ_X) x W_op; compare
   vs measured time at 1638 GB/s - the same BW-bound check step0 ran for
   moe_vec (92% roofline).
3. If BW-bound: the widening projection is legitimate. If already
   ALU/LDS-bound: candidate A pays less - quantify before designing (T52).
4. Optional: ncu single-op microbench only if model-vs-measurement gap
   exceeds ~20%.

## Replanning (2026-09-19, post Step 0 - user decision recorded)

Step 0 COMPLETE. Dense q8_0 is ALU/issue-bound at tile 4 (~22 TF/s per-MAC
class, 0.1328 B/FLOP), NOT BW-bound: the naive traffic model (9.78 s) does
not close vs measured 6.04 s (0.62x; moe_vec precedent 0.92x). kda_in_proj
(34 launches/chunk, 3.11 s, W=108.35 MB > L2 96.0 MiB) runs 18.08 TF/s; all
W<=71.3 MB classes run 20.9-22.4 TF/s. Chunk @MTILE=32: 14.87 s = dense 6.04
+ grouped MoE 4.37 + fetch copies 3.11 + rest 1.35; boot 546.5 tok/s (T44
PASS, -1.77% vs 556.43). ncu blocked (ERR_NVGPUCTRPERM). Census 322/chunk
(indexer wq_b NOT on the MMQ path - loaded bf16; kv_b dequantized for MLA
bmm absorption). Artifacts: step0-profile.md, step0-split.json (projection
table CORRUPT per review-a F1 - fix wave in flight), denseq80.nsys-rep,
denseq80.sqlite, microbench scripts, review-a-step0.md, review-b-step0.md,
triage-step0.md.

The traffic-bound projection in the Goal above (chunk ~10 s -> ~820-870
tok/s) is REFUTED (T52). Candidate A's payoff is UNMEASURED: the
"+1.3-3.0% max" bound rested on tile-invariance of the 22 TF/s rate, which
the M-sweep does not test and the v3a sibling family contradicts
(13.7 -> 39.6 TF/s from tile 4 -> 32). Conservative floor ~+2-3% e2e
(in_proj L2-excess reasoning only); upside plausibly much larger. Decide
via cheap A/B, never on paper (review-a F3).

USER DECISION (2026-09-19): sequential - (1) N-split wave first, (2) then
Candidate A cheap A/B. Candidate B stays gated behind Candidate A's A/B
outcome.

### Replan wave 1: N-split of kda_in_proj (config-level)

- Spec (Review-A C7-verified): single call site kda.py:122
  (GGUFLinear -> fused_mul_mat_gguf -> ggml_mul_mat_a8; dim-0 packed
  slices, no repack). N=24896 = 2 x 12448 (= 389 x 32; split at a 32-col
  tile edge); two sub-N launches of ~54.2 MB weights each < 96.0 MiB L2.
- Bitwise-exact by construction (disjoint 32-col tiles, k-only reduction,
  deterministic x-quant) - pinned by a parity test.
- Knob FREETOKEN_GGUF_KDA_NSPLIT, per-call read (T45), default 1 = current
  single launch; 2 = split; invalid values fail-fast; docstring "do not
  flip after boot". Default flip = user-gated micro-commit after the A/B.
- Conservative dense-term target 0.24-0.28 s/chunk (the ~0.5 s figure was
  ~2x optimistic: the same shape's own M-sweep caps at 19.6-19.9 TF/s;
  review-a F4).
- A/B wave after review: two boots (env unset vs =2), MTILE=32 anchor,
  port 18801, T43/T44/T46 discipline; kda launches 34 -> 68/chunk; decode
  graphs re-captured + decode re-measured (dense MMQ in bs>6 graphs).
- Status (2026-09-19): implemented (+241/-2, 4 python files, uncommitted
  on cb8db23) + Review-A/B SHIP + fix wave done (comment reword, M=7/13 +
  non-contig-x + empty-env tests, gate log + wave1-notes.md). Wave-1 STOP
  GATE CLOSED (triage-nsplit.md); commit user-gated, deferred until the
  A/B validates. NOTE: HEAD moved 1706aae -> cb8db23 mid-campaign (origin
  to be identified in the A/B preflight; internal A/B unaffected).
- Measured outcome (A/B wave, 2026-09-19): +2.61% e2e prefill; kernel-only
  recovery 92.8% of the L2 excess (0.512 s); net of copies 89.3% (0.493 s);
  L2 lever spent; decode flat; bitwise 6/6 same-mode.
- Measured outcome (Candidate A sweep, 2026-09-19/20): WINNER tile 64 -
  +40.46% e2e prefill (563.90 -> 792.08 tok/s @8128, chunk 14.414 -> 10.262
  s); dense term 5.515 -> 1.386 s; dense rate 21.9 -> 88.2 TF/s (4.03x for
  the 16x re-read cut; the draft 176.5 TF/s was a 2x fused-FLOP error,
  corrected by review-a-sweep); MoE/fetch/rest tile-invariant (bridge
  residual 0.023 s). T44 PASS (+0.43%); T46 PASS all boots (dense
  346-350/chunk const, grid.y 2032/1016/508/254/127); decode flat;
  unit-level bitwise tile-invariance stands (single-boot e2e bitwise is
  boot-to-boot nondeterministic - NOT a gate; see ab-mtile.md). Reviews
  SHIP; doc fixes applied. Commit + default flip user-gated.

## Candidate A - widen the dense m-tile (config-level, mirrors v3a)

Template `mul_mat_q8_0` on mmq_x variants 4/8/16/32 (+64 if SMEM and
registers allow - ROCm runs 64x128 today, which is the in-tree reference
shape). Mechanism: compile-time template + a SEPARATE knob
`FREETOKEN_GGUF_DENSE_MTILE` (do NOT overload the MoE knob - keep its
meaning stable; document both in moe.py). Run only other dense formats
(q6_K/iq) through the knob if Step 0 shows they matter in the 6.04 s;
otherwise leave single-tile.
Pre-audit (T40/T51): token_offs sizing, ds coverage at nwarps 4 vs 8,
need_check boundary math, per-thread array dims - the dense body shares
lineage with moe_q but is NOT identical; verify before templating.
No trio to re-pin; grid = (row-tiles, col-tiles). Campaign shapes M=8128
divide evenly by 8/16/32 (no tail-tile padding waste at this chunk; other
M values have normal tails).
A/B sweep 4/8/16/32(/64) on hardware: winner flags, MTILE=32 fixed, port
18801, T43 medians (exclude c1 warmup + last full chunk), T44 validity
gate vs the 561.50 class (post-NSPLIT-flip baseline; CUPTI overhead ~1.2%
symmetric - judge same-mode deltas), T46 nsys liveness with the CORRECTED
anchors (review-b-candidate-a F2): total dense launches ~356/chunk
(68 kda sub-N at gridX=389 per half + ~288 other) CONSTANT across tiles;
kda stays 68; per-launch grid.y = ceil(M/tile) shrinks by the tile factor
(M=8128 -> 2032/1016/508/254/127); decode replays the bs=1 MMVQ graph
(neither knob enters it) - expected flat. Battery on the winner tile.

## Candidate B - tensor-core / dequant path (bigger effort, only if A saturates)

- v1 lesson: dequant-to-bf16 was killed for EXPERTS by 14.5 GB/layer. Dense
  weights are far smaller per layer (~236 MB/layer IF the BW-bound model
  holds - verify from the real header in Step 0): bf16 materialization
  ~2x that, x42 layers -> likely 10-20 GB, which does NOT fit alongside KV
  at 0.85 - expect NO-GO on full materialization, but check the real number.
- int8 tensor-core MMQ (llama.cpp post-#8495 mma pipeline as the reference;
  the vendored tree is PRE-#8495; local clone /media/ai/src/llama.cpp -
  check which revision is available and whether it has the mma mul_mat_q).
  Microbench gate BEFORE any port (D06 recipe): standalone kernel, same
  ISA/flags class as the reference build, sustained throughput + parity on
  random rows; only port if the gate clears the in-tree MMQ baseline.

## Tests

Extend `tests/kernels/test_gguf_quant.py` (mirror, do not create):
dense parity per format at the new tiles
(`test_dense_gguf_modules_forward_packed` ~:518),
`test_dense_gguf_noncontig_activations` (~:567),
mmvq-cutoff tests (`test_dense_iq_above_mmvq_cutoff_uses_mmq` ~:208)
must stay green unchanged; add a knob unit + single-source-of-truth
regression mirroring `test_moe_mtile_env_knob` (~:488); decode-path
coverage: one test asserting batch<=6 still routes to the vec kernels
with the knob set. Full gate before commit
(`tests/ -m "not slow" --ignore=tests/models/test_glm5_next_kda_snapshot.py`).

## Commits

Conventional Commits, lowercase, imperative. Suggested:
`perf(kernels): widen dense q8_0 mmq tile for prefill` (candidate A).
Commit only when the user asks. Never push.

## Open decisions for the user

- Flip the MoE knob default 4 -> 32 for production (battery says quality
  is bitwise identical; separate micro-commit after this campaign lands).
- Candidate B only if A saturates below the roofline (same gate logic as
  the v3b decision - measured, not paper).

## Out of scope (separate tasks)

- Fetch copies 3.15 s (PCIe-class copy kernel `fast_index_copy_multi`).
- Prefill overlap enablement (576 slots/partition redesign).
- MoE re-tuning (done: 1706aae); KV/pool changes.
