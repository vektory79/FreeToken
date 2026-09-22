# Review-A - Step 0 regime classification (dense-q80-gemm)

Scope: measurement validity of the Step-0 wave (regime classification,
per-op decomposition, BW-bound check, projections). Read-only paper review of
recorded artifacts + source verification at HEAD 1706aae lineage; no
measurement re-run (one exception: a device-property query, see F8).
Reviewer-A: correctness / behavior / architecture.

## 1. Per-claim verdicts

### C1 - Naive traffic model 9.78 s vs measured 6.04 s = 0.62x -> NOT BW-bound
**CONFIRMED** (arithmetic re-derived end to end).
- Per-op bytes re-derived from denseq80_header.json (Q8_0 = 34 B / 32 elems
  = 1.0625 B/elem): in_proj 108,347,392 B (header components sum exactly:
  3x attn_q/k/v [4096,8192] = 106,954,752 + ssm_beta [4096,64] 278,528 +
  2x ssm_f_a/g_a [4096,128] = 1,114,112 -> 108,347,392, blk.20 section);
  kda_o 35,651,584; mla_o 71,303,168; shexp 8,912,896 x3/op; q_b 26,738,688;
  q_a 6,684,672; kv_a 2,228,224; dense_ffn 53,475,328; f_b/g_b 1,114,112.
  All match the step0-split.json weight_MB column.
- Sum_op W x launches/chunk = 7.886 GB; x ceil(8128/4) = 2032 -> 16.02 TB;
  /1638 GB/s = 9.783 s (JSON naive_model_s_per_chunk = 9.783 exact).
  measured 6.037 / 9.783 = 0.617 ("0.62" ok).
- Activation and OUTPUT terms are indeed negligible vs the re-read term:
  x one-pass ~9.7 GB (6 ms) + GEMM output writes 36.6 GB/chunk (22 ms) at
  1638 GB/s, vs 16.02 TB -> the "~20 GB, negligible" dismissal is right in
  substance (its 20 GB figure is a rough placeholder; harmless).
- moe_vec precedent: 29.53 TB / 18.13 s = 1629 GB/s = 0.91-0.92 of the
  1792 peak model -> closed; dense at 0.62 cannot close. Reading verified.
- Caveat recorded: 0.62x alone is necessary-not-sufficient (an L2-aware
  one-pass floor is far lower); the refutation is carried by C3+C4. The
  report presents all three; fine.

### C2 - bytes/MAC 0.1328 shape-constant; 0.55-0.57 uniform = ~22 TF/s ceiling
**CONFIRMED with one unit-label error (F5) and a uniformity nuance (F6).**
- Re-derivation: per MAC the kernel loads 1.0625/4 = 0.2656 B of weights
  (each weight byte feeds 4 tokens at MMQ_X=4; q8_0 block 34 B/32 elems
  verified) plus negligible activation bytes (each x element is read by
  N/32 col-tiles -> 1.0625/(32xM) B/MAC ~ 4e-6 at M=8128). So the
  shape-constant traffic is 0.2656 B/MAC = 0.1328 B/FLOP. The report's
  "0.1328 B/MAC" is bytes per FLOP mislabeled as B/MAC; its downstream
  arithmetic (1638/0.1328 = 12.34 TFLOP/s DRAM ceiling; 12.34/0.556
  = 22.2 TF/s) is internally consistent under the B/FLOP reading.
- Uniformity: measured/model ratios in step0-split.json are 0.550-0.565 for
  all classes except kda_ssm_f_b/g_b at 0.592 (TF/s 20.84). So "0.55-0.57
  across ALL classes" is slightly overstated; the cluster statement
  "W <= 71.3 MB runs 20.9-22.4 TF/s" excludes ssm (20.84).
- The ~22 TF/s figure is real for tile 4. What C2/C3 do NOT establish:
  tile-invariance of that rate (see F3).

### C3 - M-sweep: time linear in M, rate 18.1-19.9 TF/s from gridY 16->2032
**CONFIRMED** (methodology and numbers verified; artifact gap F7).
- microbench_dense.py verified: exact in_proj geometry (N=24896, K=4096,
  row_bytes 4352, grid (778, ceil(M/4)) matches the mmq.cuh launcher
  block_num_x = ceil(nrows_x/mmq_y), block_num_y = ceil(ncols_y/mmq_x));
  M 8128..64 -> gridY 2032..16; 3 warmup + 2-per-point warmup + 20 timed
  iters (means, not medians - F9); CUDA events; idle GPU, serialized
  (hygiene log consistent).
- Internal consistency of the quoted numbers: us/row-tile 41.6-45.6 ->
  per-MAC rate 17.9-19.6 TF/s (md quotes 18.1-19.9; rounding mismatch,
  no raw output saved to arbitrate); implied naive BW 2413-2660 GB/s
  reproduced (2032 x 108.35 MB / 91.2 ms = 2414 GB/s) > 1792 peak.
- Peak BW 1792 GB/s = GDDR7 28 Gbps x 512-bit spec; sustained 1638 was
  hardware-validated in the moe_vec campaign (29.53 TB / 18.13 s).
- Logic check: the sweep proves the naive re-read traffic is not DRAM
  traffic; BW-bound refutation proper is carried by C4 (any traffic-bound
  model, L2 included, cannot make 23% fewer bytes SLOWER at equal MACs).
  Correctly composed in the report.

### C4 - Format A/B: q6_K 23% fewer bytes, 6.3% slower (1.063 vs 0.772)
**CONFIRMED** (apples-to-apples verified).
- Fixtures: same N=24896 / K=4096 / M=8128, same MACs; q6_K row_bytes
  3360 = K/256 x 210 (block_q6_K layout [2B d][16B scales][128B ql][64B qh]
  correctly assembled); q8_0 4352. Ratio 0.8203/1.0625 = 0.772 exact.
- Same kernel family AND same tile: MMQ_X_Q6_K = 4 on CUDA (mmq.cuh:814),
  identical to MMQ_X_Q8_0 = 4 (mmq.cuh:438) -> clean same-tile comparison.
- Decode work is data-independent for both formats (integer ops, fixed
  scales) -> synthetic fixtures valid. x re-drawn unseeded per bench
  (harmless - timing is data-independent; F9).
- Conclusion holds: traffic is not the binder at tile 4. Gap: measured
  ratio exists only as md prose (F7).

### C5 - kda_in_proj W=108.35 MB > ~96 MB L2 -> 18.08 TF/s (-17.5%)
**CONFIRMED on W and rates; the ~96 MB L2 figure is CORRECT but was
asserted, not captured (F8); the -17.5% causal split is overstated (F4).**
- W verified exact from the header (108,347,392 B = 108.35 MB).
- All W <= 71.3 MB classes: 20.84-22.41 TF/s in step0-split.json (md band
  "20.9-22.4" clips ssm at 20.84 - F6).
- L2: NOT evidenced in any artifact (hardcoded `L2_MB = 96.0  # RTX 5090`
  in final_split.py; no get_device_properties capture anywhere; step0_run.json
  has no device record). This review queried the device:
  torch.cuda.get_device_properties(0) -> NVIDIA GeForce RTX 5090,
  L2_cache_size = 100,663,296 B = exactly 96.0 MiB, 170 SMs, 31.4 GiB.
  So the figure is correct; provenance must be recorded (F8).
- Overstatement: the report attributes the whole -17.5% (18.08 vs the
  21.9-22.4 cross-shape cluster) to L2 overflow, but its own M-sweep shows
  the SAME shape running 19.6-19.9 TF/s at small M (minimal re-read DRAM
  pressure). The M-dependent (DRAM/L2) component is only ~6-9%
  (18.1 -> ~19.7 same-shape ceiling); the remaining ~10% is shape-inherent
  (or small-M L2 misses - indistinguishable without ncu). Direct consequence
  for C6/C7 magnitudes below.

### C6 - Widening projection 5.85/5.75/5.66/5.61 s -> 554/557/561/563 tok/s
**REFUTED as an auditable artifact; model verdict NEED-MORE-DATA (F1-F3).**
- step0-split.json and step0-profile.md give CONTRADICTORY tables:
  JSON 4.518/5.277/5.657/5.847 s -> 608.8/576.0/560.9/553.7 tok/s
  (wider tile = worse); md 5.85/5.75/5.66/5.61 -> 554/557/561/563.
- The JSON table is provably buggy (F1): final_split.py computes
  `excess_s4 = in_proj["seconds_per_chunk"] - 2*M*24896*4096/22.0e12`,
  mixing a per-chunk total (3.1136 s) with a per-launch 22 TF/s time
  (0.0753 s) - the x34 launch multiplier is missing -> excess_s4 = 3.038
  instead of 0.552 s (5.5x); and `rec = excess_s4*(4.0/tile)` is the
  RESIDUAL excess subtracted from tot_meas -> tile trend inverted. JSON
  tile8 dense 4.518 s is below the report's own 5.36 s ALU floor at
  22 TF/s - impossible under its own regime. Verified by exact numeric
  match of the script formulas to the JSON values (1.519/0.76/0.38/0.19).
- The md table does not reproduce from the stated model either (F2):
  stated model = 6.037 - 0.552 x (1 - 4/T) -> 5.76/5.62/5.55/5.52 s ->
  557/559/565/566 tok/s (+1.9 to +3.6% e2e). No script in the folder
  generates the md numbers. The quoted +1.3-3.0% is not anchored.
- Model-of-why check: "widening helps an ALU-bound kernel only via fewer
  weight-load events; non-in_proj classes unchanged" is consistent with
  C2/C3 INTERNALLY, but the zero-gain assumption for non-in_proj classes is
  contradicted in DIRECTION by the campaign's own v3a evidence (F3): the
  same vec_dot kernel family at m-tile 4 -> 32 raised its per-MAC rate
  ~2.9x (13.7 -> 39.6 TF/s at the report's own 173 TF convention; 31.5 TF/s
  on a strict pair-MAC count - either far above dense's 22 with HEAVIER IQ
  decodes). C3 varies M at FIXED tile 4 and says nothing about the tile
  axis. So "22 TF/s is a per-MAC ceiling" is only a tile-4 measurement
  presented as an invariant; the projection bakes it in as a hard ceiling.
  T52 says traffic cuts pay SUB-LINEARLY, not zero - the projection reads
  T52 as zero.
- T35 repetition (as asked): yes - the de-prioritization verdict
  ("Candidate A not justified as a major lever") is a paper-model decision
  with no measured anchor, exactly the T35 pattern (paper NO-GO vs measured
  GO), and it is the input the replanning consumes.

### C7 - N-split of kda_in_proj (2 x ~54 MB < L2, numerics-exact, ~0.5 s)
**(a) CONFIRMED config-level; (b) CONFIRMED exact; (c) NEED-MORE-DATA,
magnitude 1.7-2x optimistic (F4).**
- (a) Launch site: GGUFLinear.forward (layers/gguf.py:89-95) ->
  fused_mul_mat_gguf (layers/gguf.py:52-73) -> single ggml_mul_mat_a8 call
  on the fused [24896, 4352] qweight (wrapper gguf_kernel.cu:196-210:
  Y = empty[batch, row]; quantize_row_q8_1 then one kernel). The fusion is
  a load-time Q8_0 Linear swap of attn.in_proj (glm5_next/gguf.py:676-681);
  dim-0 row slices of the packed tensor are contiguous with no repack
  (documented pattern, glm5_next/gguf.py:591-593). Split 24896 -> 2 x 12448
  rows = 2 x 389 x 32 exact col-tiles, ~54.18 MB each < 96 MB, + a 405 MB
  cat (~0.25 ms) - pure python/config change, no kernel change. Confirmed.
- (b) Numerics-exact: output columns are disjoint; mul_mat_q reduces only
  over k (vec_dot per (i,j), no cross-column reduction); both halves have
  exact 32-col tile boundaries (12448 = 389x32) and exact 4-row m-tiles;
  x is re-quantized deterministically per launch -> bitwise identical
  outputs. Verified structurally against mmq.cuh / vecdotq.cuh
  (vec_dot_q8_0_q8_1_mul_mat reduces k only). Same guarantee class as the
  v3a tile-invariance battery.
- (c) Anchor: 0.5 s assumes recovery to the >=20.9-22.4 TF/s cluster
  (34 x (91.58 - 75.3 ms) = 0.55 s). But the report's own M-sweep caps the
  same shape at 19.6-19.9 TF/s even at minimal DRAM pressure -> conservative
  recovery = 34 x (91.58 - ~84.6 ms) = 0.24-0.28 s. Which target applies
  once the 54 MB half fits L2 is exactly what ncu (blocked) would decide;
  the artifacts cannot. Also internally inconsistent: the Verdict section
  says "~0.38 s" (which equals the BUGGY JSON tile32 row), the closing
  note says "~0.5 s" - F4.

## 2. Findings (protocol format)

- **F1 BLOCKER (for the replanning input; the artifact itself is corrupt)**:
  step0-split.json `widening_projection_issue_bound` is generated by
  final_split.py with two arithmetic bugs (per-chunk/per-launch mix, missing
  x34; residual-vs-recovery inversion). The machine-readable deliverable
  contradicts the md and its own regime model (tile8 dense 4.518 s < the
  report's own 5.36 s ALU floor; tile64 worse than tile32). Grounding:
  final_split.py proj block; exact numeric match with the JSON rows.
- **F2 MAJOR**: the md widening table (5.85/5.75/5.66/5.61 -> 554/557/561/563)
  is not reproducible from the stated model (re-derived 5.76/5.62/5.55/5.52
  -> 557/559/565/566) nor generated by any kept script; +1.3-3.0% e2e is
  unanchored (T35 pattern feeding the Candidate-A decision).
- **F3 MAJOR**: the "not justified as a major lever" verdict rests on the
  unverified assumption that the 22 TF/s per-MAC rate is m-tile-invariant.
  The repo's own v3a evidence shows strong tile dependence of the same
  kernel family (13.7 -> 39.6 TF/s from tile 4 to 32, sub-linear vs an 8x
  traffic cut but far from zero - T52 says sub-linear, not zero). C3 covers
  M at fixed tile only. Honest Step-0 output: regime pinned at tile 4;
  Candidate-A upside UNKNOWN pending the cheap A/B sweep - possibly ~+2-3%,
  possibly ~2x on the dense term.
- **F4 MAJOR**: in_proj deficit decomposition overstated: full -17.5%
  attributed to L2 overflow, but the M-sweep caps the same shape at
  19.6-19.9 TF/s at small M -> shape-inherent ~10% + DRAM-dependent ~6-9%.
  Conservative N-split recovery 0.24-0.28 s vs the quoted ~0.5 s
  (~2x optimistic); the Verdict's "~0.38 s" equals the buggy F1 row and
  contradicts the "~0.5 s" note.
- **F5 NIT**: unit error - "0.1328 B/MAC" is 0.1328 B/FLOP (= 0.2656 B/MAC);
  internal arithmetic consistent under B/FLOP; conclusion unaffected.
- **F6 NIT**: uniformity band overstated - ssm f_b/g_b at 0.592 / 20.84 TF/s
  sits outside "0.55-0.57" / "20.9-22.4"; md in_proj model 4.558 vs JSON
  4.5699; ratio 0.68 vs 0.681.
- **F7 NIT**: microbench raw outputs (M-sweep table, format A/B ratio) are
  not saved as artifacts - quoted numbers exist only in step0-profile.md
  prose; internally consistent, scripts kept and deterministic, but not
  auditable without a re-run. Save stdout next run.
- **F8 NIT**: L2_MB = 96.0 hardcoded with a comment, no device capture.
  Verified CORRECT by this review (RTX 5090 reports 100,663,296 B =
  96.0 MiB); record a get_device_properties capture in the artifacts.
- **F9 NIT**: md says the sweep used "medians"; the scripts use 20/30-iter
  means with warmups excluded (fine in practice; fix the wording).

Verified-sound items (for the record): T44 gate re-derived exactly from
step0_run.json throughput lines (14.81/14.81/14.87/14.87/14.90/14.93 ->
median 14.87 s, 546.5 tok/s, -1.78% vs 556.43, chunk-1 warmup 170.5 and the
2077.1 T43 artifact excluded); census 322 = 34x7 + 12x7 closes against the
seq_probe block motif and the header tensor inventory (indexer wq_b and kv_b
correctly excluded from MMQ); grid geometry and tile constants verified in
mmq.cuh:434-439 + launcher; ncu ERR_NVGPUCTRPERM confirmed (ncu_dense.log);
hygiene log consistent (census, serialization, idle-GPU microbenches).

## 3. Overall verdict

The REGIME CLASSIFICATION is sound and can be relied on: dense q8_0 at tile 4
is ALU/issue-bound (not BW-bound); the three software discriminators are
valid and their arithmetic closes; the naive traffic model re-derives to
9.783 s exactly and the traffic-bound dividend is correctly declared dead;
kda_in_proj is a genuine underperforming exception. The PROJECTION AND
DECISION LAYER on top of it is not sound enough for replanning as-is: the
machine-readable deliverable's projection table is corrupt (F1), the md table
is unreproducible (F2), and both the Candidate-A de-prioritization (F2/F3)
and the N-split magnitude (F4) are paper numbers without a measured anchor -
F3 in particular reverses the risk direction (Step-0 under-states Candidate A
using tile-invariance the artifacts do not establish and v3a contradicts).

Recommended handling: keep "regime = ALU/issue-bound at tile 4" as the Step-0
product; strike the widening table and the "+1.3-3.0% max" / "~0.5 s" figures
from the decision input; present the user with "Candidate A upside uncertain -
conservative floor ~+2-3% (in_proj excess only), plausible upside much larger
if the v3a rate-vs-tile scaling transfers; decide with the cheap measured A/B
(T52: measure, don't model)"; fix final_split.py and regenerate the JSON;
record device properties and microbench outputs as artifacts.