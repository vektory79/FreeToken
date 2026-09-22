# Review-B - Step 0 measurement wave (dense-q80-gemm)

Scope: edge cases / data consistency / integration / operability of the Step-0
artifacts. Paper audit + source verification only: no GPU touched, no
measurement re-run, no files modified except this report. All file:line anchors
re-verified 2026-09-19 against the working tree (branch vektory79, engine state
1706aae).

Verdict summary: B1-B8 all CONFIRMED. 0 BLOCKER, 0 MAJOR, 11 NIT, 0 FP.
Data consistency holds; integration readiness for the flagged next steps: GO.

---

## B1. Census / decomposition - CONFIRMED

The 322/chunk census and the class decomposition are correct against the real
model paths, the GGUF header scan, and the measured launch sequence.

- Kernel attribution is by launch geometry, not names (only one kernel name,
  `mul_mat_q8_0` exists in the dense term). classify_launches.py maps
  gridX = ceil(N/32), gridY = 2032 (M=8128/4); the (128,2032) bucket is split
  by duration (shexp_down <10 ms, kda_o <33, dense_ffn_down <44, mla_o else).
  Bucket boundaries are clean: class medians 6.19 / 25.02 / 37.45 / 48.67 ms
  leave >30% gaps; no unexplained launches (window total 1215 = 314.1/chunk
  with the mid-chunk-3 capture start explained in step0-profile.md).
- kda_in_proj is ONE fused GEMM: `_in_proj_split = [p, p, p, h, d, d]` with
  p=64*128=8192, h=64, d=128 (kda.py:63-67), one `LinearColParallelMerged`
  (kda.py:64-67), one forward call (kda.py:122), single `quant_method.apply`
  per call (linear.py:57-58), loader concatenates the six checkpoint parts
  into one weight (weight.py:126-131). Header: attn_q/k/v [4096,8192] Q8_0 +
  ssm_f_a/g_a [4096,128] + b (64) -> N=24896.
- kda f_b/g_b: `LinearReplicated(128, 8192)` (kda.py:68-70); header
  ssm_f_b/g_b [128,8192] Q8_0. kda_o: `LinearReplicated(8192, 4096)`
  (kda.py:79); header attn_output [8192,4096].
- MLA: q_a 4096->1536 (attention.py:110-113), q_b 1536->16384
  (attention.py:115-118), kv_a 4096->512 with the NoPE note "no
  +qk_rope_head_dim rows" (attention.py:124-127), o 16384->4096
  (attention.py:136-139). Header blk.11.attn_q_a [4096,1536],
  attn_kv_a_mqa [4096,512].
- shexp 4096->2048 / 2048->4096 (header ffn_gate_shexp / ffn_down_shexp),
  43 blocks; dense-FFN 4096->12288 / 12288->4096 (header blk.0.ffn_gate
  [4096,12288]), 3 blocks.
- Brief deviation 1 VERIFIED: indexer wq_b is ABSENT from the MMQ path. The
  seq probe MLA motif is exactly 7 launches [48@4.6 ms, 512@18.7, 16@1.5,
  128@48, 64, 64, 128@6] - exactly one (48,2032) launch per MLA layer (= q_a),
  and no extra ~5 ms (128,2032) launch, although wq_b is Q8_0 ON DISK
  (header blk.11.indexer.attn_q_b [1536,4096] Q8_0, 6.68 MB): the loader
  dequantizes all indexer projections at load - "Kept bf16", weight.py:147-151.
  See N5 for the prose/table blemishes this uncovered.
- Brief deviation 2 VERIFIED: kv_b is dequantized for MLA bmm absorption.
  attention.py:129-135 asserts `kv_b_proj.quant_method.kind is QuantKind.NONE`
  and _kv_b (attention.py:140-151) repacks the weight into bmm operands w_uk /
  w_uv used at attention.py:166 and :185; header stores attn_k_b as 3D Q8_0
  [256,512,64] (dequantized at load).
- Numbers close: 3.1136 (in_proj) + 0.8507 (o_gdn) + 0.5841 (o_mla) + 0.798
  (shexp 0.532+0.266) + 0.336 (dense-FFN) + 0.225 (q_b) + 0.1295 (small:
  f_b/g_b 0.0556 + q_a 0.0553 + kv_a 0.0186) = 6.037 s = the 6.04 s dense term.
  Launch census: 34x4 (KDA) + 12x4 (MLA) + 43x3 (shexp) + 3x3 (dense-FFN)
  = 322. MHC/hc_* tensors ([16384,24], K=24 not block-aligned) are not
  mul_mat_q8_0 launches - the 512-bucket holds exactly 12/chunk (q_b only).

## B2. kda_in_proj shape - CONFIRMED

- N = 24896 = 3x8192 (q|k|v) + 64 (b = num_heads) + 128 (f_a) + 128 (g_a):
  kda.py:63-67 with linear_num_heads=64, linear_head_dim=128 (proj_size=8192);
  header confirms the per-part tensors (blk.12.attn_q/k/v [4096,8192],
  ssm_f_a/g_a [4096,128], ssm_dt.bias [8192]).
- gridX = ceil(24896/32) = 778 exactly (24896 = 32 x 778); measured
  (778,2032) x 91.58 ms.
- 34 launches/chunk = the KDA layer count: 46 blocks total, 12 MLA blocks
  (header attn_q_a present in blk.3,7,11,15,19,23,27,31,35,39,43,45) -> 46-12
  = 34 KDA blocks, each issuing exactly one in_proj GEMM (kda.py:122).

## B3. Chunk accounting - CONFIRMED (2 NITs)

- Sum: 6.037 dense + 4.370 grouped-MoE + 3.106 fetch + 1.35 rest (residual:
  14.87 - 6.037 - 4.370 - 3.106 = 1.357) = 14.87 s median. Rest is a residual,
  labeled "~" in the profile - acceptable.
- Median methodology is T43-compliant: step0_analyze.py filters
  `#new-token == 8128` lines (drops the 561-token tail automatically),
  converts to seconds, drops first and last full chunk (`vals[1:-1]`), median
  of the remaining 6 (c2..c7: 14.81, 14.81, 14.87, 14.87, 14.90, 14.93 ->
  14.87 s). Chunk-1 warmup (170.5 tok/s) and the bogus last-full line (2077
  tok/s) are both outside the window.
- Window math: window_s 57.519 / T 14.87 = chunks_eq 3.868 (json meta);
  1215 mul_mat_q8_0 launches / 3.868 = 314.1/chunk window-normalized. The
  322/chunk figure is the model-code census, not the window-normalized count;
  step0-profile.md explains the difference (capture starts inside chunk 3, so
  its head blocks are outside the window). Closure: median-based 6.037 s vs
  window-normalized 5.982 s = 0.9% - consistent.
- N1: the profile TEXT says "chunks_eq = 57.52/14.87 = 3.871" and
  "1215/3.871 = 313.9" - arithmetic slip; the json (authoritative) has 3.868
  and 314.1. Cosmetic.
- N2: boot throughput is quoted as 546.5 tok/s in the gate line but the json
  boot_tok_s is 546.6 (8128/14.87); median of the c2..c7 throughput lines is
  546.58. 0.02% spread, no impact on the gate.

## B4. Grouped-MoE correction - CONFIRMED; correction note REQUIRED

- v3a-ab.md:57-58 builds the e2e-subtraction model on "rest = 12.85 s" treated
  as tile-invariant, and :79 records "At tile 32 the MoE term is ~1.8 s (e2e
  fit) of a 14.61 s chunk". Direct nsys measurement at the same MTILE=32 anchor
  gives 4.37 s/chunk (moe_iq3_xxs 2.749 + moe_iq4_xs 1.531 + moe_q6_K 0.090).
  The subtraction estimate was wrong by ~2.4x: rest is NOT tile-invariant.
- The correction is over-determined by the artifacts themselves: v3a-ab's own
  tile-16 nsys measured grouped MoE 5.01 s/chunk; even its measured e2e ratio
  (0.152) applied to tile-4's 12.62 s gives ~2.9 s, not 1.8 s - the recorded
  estimate was internally inconsistent before Step 0.
- A correction note IS needed on v3a-ab.md:79 ("~1.8 s (e2e fit)" -> measured
  4.37 s, rest not tile-invariant), and the same stale number propagates into
  dense-q80-gemm/TASK.md:19 ("MoE now ~1.8 s at tile 32") and the
  ft-gguf-v3-mtile-campaign memory ("chunk now ~80% rest"; with 4.37 s the
  non-MoE share is 70.6%). Orchestrator to propagate.
- No verdict in v3a's sweep changes: the tile ranking, saturation point (32),
  and quality battery are e2e measurements independent of the decomposition.

## B5. T44 validity gate - CONFIRMED PASS, apples-to-apples

- Median recomputed from step0_run.json throughput lines: c2..c7 = 548.72,
  548.91, 546.68, 546.48, 545.60, 544.48 tok/s -> 546.58 tok/s (chunk times
  14.81..14.93, median 14.87 s). Delta vs the committed 556.43 class =
  -1.77% (or -1.78% via 8128/14.87), inside the brief's ~-2% band and well
  inside T44's generic ~3% run-variance band (v3a's own BEFORE gate passed at
  -1.45%).
- Same flags: boot command identical to v3a-ab.md:13-19 (ServerArgs confirms
  kv_quant fp8, memory_ratio 0.85, max_extend_tokens 8191, mr1, port 18801);
  same env FREETOKEN_GGUF_MOE_MTILE=32. Same fill: measure.do_prefill on the
  same 65,585-token filler.txt (8x8128 + 561), 9 chunks, all cached=0.
- Anchor proven: the 14.87 s chunk IS the tile-32 class - a tile-4 boot runs
  24.43 s / 332.72 tok/s (v3a-ab table), nowhere near the measurement; the
  json's empty boot_env record does not weaken this (see N8).
- Same engine: step0 ran at engine state == 1706aae, which IS the committed
  v3a changeset v3a-ab measured (verified via merge-base per the profile);
  HEAD cb8db23 on top is Skill/Memory-only.
- Only runtime difference: night (v3a 03:17-04:08) vs afternoon (14:42-15:10)
  - exactly what the run-variance band absorbs. Decode not re-measured is
  justified: no dense-side code change, knob is prefill-only.

## B6. Hygiene compliance - CONFIRMED (2 NITs)

- Preflight == postflight: dq80_stage.sh census (pgrep ft serve/nsys,
  nvidia-smi, `ls /dev/shm | grep -c '^sem\.mp-'`) at start and after cleanup;
  profile logs 1372 MiB / 5 sem.mp- both sides; step0_run.json teardown
  leftover=[] and gpu_mib 1371. No stray report1.* anywhere in the repo
  (verified by search).
- Runner: trap-on-EXIT cleanup with kill_tree (SIGTERM -> 10 s -> SIGKILL) +
  nsys session shutdown + stray-report rm (dq80_stage.sh:15-27); outer
  `timeout 2400`; FAILPAT watchdog inside step0_run.py =
  measure.WATCH (AssertionError|OutOfMemoryError|Backend worker is gone|CUDA
  error|Traceback) + `backend worker .* exited`, hard deadline 1800 s
  (step0_run.py:29-32). nsys start/stop rc=0/0 after the 2nd/6th
  "input throughput" line; capture window n_thr=6.
- Serialization: one boot, microbenches run AFTER the capture on the idle GPU
  under `timeout`; measure.teardown reaped the server (leftover=[]).
- Pre-boot abort: first stage attempt died in start_server on the non-
  executable wrapper (+x missing) before anything launched; trap cleanup ran,
  census clean - documented in the profile hygiene log. Consistent with the
  artifacts (no orphan reports, no pid files besides dq80_nsys.log.pid).
- N7: the runner escalates SIGKILL after 10 s vs the protocol digest's 30 s -
  more aggressive, documented honestly in the profile; harmless.
- N8: step0_run.json boot_env is {} (common_tail records the DRIVER's env,
  not the server's); the knob is proven active by throughput (B5), but the
  machine-readable record of the boot env is missing.

## B7. Artifact completeness / machine-readability - CONFIRMED (4 NITs)

- Present and named as reported: denseq80.nsys-rep, denseq80.sqlite (export
  rc=0 in stop_err), denseq80_header.json (1412 tensors), step0-split.json,
  step0-profile.md, all scripts, seq_probe.out, ncu_dense.log, and the server
  log .tasks/ft-gguf-serve-tuning/logs/dq80_nsys.log (+ .pid). The nsys stop
  wrote report1.nsys-rep into the task dir and it was renamed to
  denseq80.nsys-rep as documented.
- ncu_dense.log documents ERR_NVGPUCTRPERM as a user-permission environment
  blocker (3 lines: connect / ERR / disconnect), and the profile substitutes
  two software discriminators (M-sweep, format A/B) plus the enabling
  instructions - correctly framed as environment, not measurement error.
- step0-split.json vs step0-profile.md: consistent on every decision-relevant
  number (T, per-class K/N/layers/medians/s-per-chunk, W MB, ratios, regime,
  projections). Blemishes:
  - N3: totals.dense_tflop_per_chunk = 117.8 is hardcoded (final_split.py:166)
    and understates the sum of its own op_classes (~120.7 TF; the four
    smallest classes f_b/g_b/q_a/kv_a = 2.8 TF are omitted). The profile's
    "floor ~5.36 s at 22 TF/s" should read ~5.49 s - which actually CLOSES the
    measured-vs-floor slack to the independently computed in_proj excess
    (0.55 s), strengthening the report. No verdict changes.
  - N4: the json's other_terms (grouped/fetch/rest), the dense window total
    5.982, and the gate string are hardcoded constants in final_split.py, not
    derived from denseq80.sqlite in the same pipeline; the raw sqlite is
    retained so they are re-derivable.
  - N9: "tflop_per_s_at_fixed_173TF: 39.6" - the 173 TF/chunk fixed-FLOP
    denominator is not anchored in any visible artifact; a recount of routed
    MoE FLOPs at this chunk (2 x 8128 x 8 pairs x 3 proj x 2 x 4096 x 2048 x
    42 layers) gives ~137.5 TF -> ~31.5 TF/s. Auxiliary metric only.
  - N10: step0_analyze.py (superseded by final_split.py) still writes the SAME
    step0-split.json with the stale pre-audit guesses (N=24832, an
    indexer_wq_b MMQ class) - re-running scripts in the wrong order would
    overwrite the final json with wrong data. Recommend a header note or
    rename.

## B8. Integration preconditions for the flagged next steps - CONFIRMED (flag only)

(a) kda_in_proj call sites / graph constraints:
- ONE call site. The projection is constructed once per KDA layer as a single
  LinearColParallelMerged (kda.py:64-67) with the six segments fused at load
  (weight.py:126-131) and invoked exactly once per forward (kda.py:122) via
  _LinearTPImpl.forward -> quant_method.apply (linear.py:57-58) -> one
  mul_mat_q8_0 launch (gguf.py:99-101, M>6 routes to ggml_mul_mat_a8). Not
  scattered; no other consumer of the fused weight.
- Prefill is eager: prefill overlap is disabled for every signature partition
  (step0_run.json: "got 391/288 slots < 2*288: prefill overlap disabled ->
  synchronous materialized prefill"), so no CUDA-graph capture on the path the
  N-split targets. Decode: dense MMQ kernels are captured in decode graphs for
  bs>6 (TASK.md background; bs<=6 uses the MMVQ vec kernels, gguf.py:92-93),
  so a 2-launch split changes decode-graph contents - handled by the already
  mandated boot-time graph re-capture + decode re-measure; nothing makes 2
  launches a problem (prefill adds 34 tiny launches vs a 0.55 s target; decode
  doubles only GEMV-grade in-graph launches).
- Implementation note: the natural split point 2x12448 rows does not align with
  the q|k|v|b|f_a|g_a segment boundaries; outputs must be concatenated back
  before kda.py:123-125's torch.split. Row-independent GEMM -> numerics-exact.

(b) Out-of-scope check:
- The N-split reorganizes the dense q8_0 GEMM term itself - the campaign's
  subject. It touches neither fetch copies (fast_index_copy_multi), prefill
  overlap (576-slot redesign), MoE re-tuning (frozen at 1706aae), nor KV/pool
  sizing. No conflict with the TASK.md "Out of scope" list.

---

## Findings (protocol format)

- N1 [NIT] step0-profile.md (op-class motif paragraph): "chunks_eq = 57.52/14.87
  = 3.871" is an arithmetic slip; step0-split.json meta has the correct 3.868
  (and 314.1 launches/chunk, not 313.9). No downstream use. TP=cosmetic fix.
- N2 [NIT] step0-profile.md / step0-split.json: boot tok/s quoted 546.5 (median
  of throughput lines = 546.58) vs boot_tok_s 546.6 (8128/14.87). 0.02%;
  gate unaffected.
- N3 [NIT] final_split.py:166 + step0-profile.md: totals.dense_tflop_per_chunk
  117.8 omits f_b/g_b/q_a/kv_a (op_classes sum ~120.7 TF); floor line
  "~5.36 s" should be "~5.49 s" (which closes the slack to the 0.55 s in_proj
  excess). Fix the constant before it is quoted downstream.
- N4 [NIT] final_split.py: other_terms / 5.982 / gate string hardcoded instead
  of derived from denseq80.sqlite; raw sqlite retained, provenance documented
  here.
- N5 [NIT] step0-profile.md: prose lists only "wk/weights_proj/compressor gate"
  as dequantized indexer tensors - wq_b is ALSO Q8_0 on disk (header
  blk.11.indexer.attn_q_b [1536,4096] Q8_0 6.68 MB) and loaded bf16
  (weight.py:147-151); the table row "indexer wq_b (4096->1536)" also
  transposes K/N (in 1536 -> out 4096). Zero MMQ launches either way.
- N6 [NIT] step0_run.json prefill_thr_summary.steady_mean 740.6 includes the
  T43-bogus 2077 line and the 561 tail (measure.py thr_summary drops only
  chunk 1). Foot-gun field; no verdict uses it (medians recomputed per T43).
- N7 [NIT] dq80_stage.sh:18: SIGTERM -> 10 s -> SIGKILL vs the digest's 30 s.
  More aggressive, documented; no consequence.
- N8 [NIT] step0_run.json boot_env={} does not record
  FREETOKEN_GGUF_MOE_MTILE=32 (common_tail reads the driver env). Anchor
  proven via the 14.87 s chunk vs the 24.43 s tile-4 class.
- N9 [NIT] step0-split.json "tflop_per_s_at_fixed_173TF": 173 TF/chunk
  denominator unanchored in artifacts; recount ~137.5 TF routed-MoE FLOPs
  -> ~31.5 TF/s. Auxiliary metric only.
- N10 [NIT] step0_analyze.py (superseded) writes the same step0-split.json
  with stale pre-audit values (N=24832, indexer_wq_b class); a wrong-order
  re-run would clobber the final json. Rename or annotate.
- N11 [NIT] capture_shutdown_rc=1 in step0_run.json (post-stop `nsys shutdown`
  of an already-stopped session) - benign, but unexplained in the artifacts.

Tally: BLOCKER 0, MAJOR 0, NIT 11, FP 0. No open TP beyond the NIT list; the
B4 correction note is a documentation action for the orchestrator, not a data
defect in Step 0 itself.

## Overall verdict

Step-0 data is INTERNALLY CONSISTENT and SOURCE-VERIFIED: the census (322/chunk),
the class decomposition (6.04 s dense), the shape/grid claims (24896/778/34),
the T43-compliant median (14.87 s, 546.5-546.6 tok/s, -1.8% PASS), and the
hygiene chain all check out against model code, the GGUF header, the launch
sequence, and the runner scripts. The v3a-ab e2e-subtraction MoE estimate
(~1.8 s) is refuted by the direct measurement (4.37 s) and needs a correction
note before propagation. Integration readiness: GO - the flagged cheap lever
(kda_in_proj N-split) has a single call site, no graph/batching blocker, and no
out-of-scope collision; Candidate A's de-prioritization rests on the regime
evidence (M-sweep + format A/B), which is Review-A's scope.
