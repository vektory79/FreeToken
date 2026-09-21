# Step 0 - dense q8_0 GEMM: pin the regime (dense-q80-gemm campaign)

Measured 2026-09-19 14:42-15:10, vektory79 (engine state == 1706aae; HEAD
cb8db23 is Skill/Memory-only on top, verified `merge-base --is-ancestor`),
single RTX 5090, GGUF GLM-5.3-Flash-UD-Q3_K_XL, MTILE=32 anchor config, port
18801. One serial boot, no engine code changed, no commits, no push.

## Verdict (the deliverable)

**Dense q8_0 (`mul_mat_q8_0`, MMQ_X_Q8_0=4) is NOT bandwidth-bound. It is
ALU/issue-bound at a ~22 TFLOP/s per-MAC ceiling** (per-MAC vec_dot decode +
dp4a + LDS work, invariant across M and across shapes). One exception:
`kda_in_proj` (N=24896, W=108.35 MB) exceeds the L2 (100,663,296 B = 96.0
MiB; device-query capture recorded below) and runs 18.1 TF/s vs the
21.8-22.4 cluster (~ -17.5%).

Consequences (T52): the traffic model's 1/tile dividend is dead. Widening the
dense m-tile only cuts weight/activation requests, which are NOT the binder;
the only software-discriminable recoverable share is the in_proj DRAM-stall
excess alone (~0.55 s of the 6.04 s term at tile 4). **Candidate A payoff is
UNMEASURED.** Conservative floor ~+2-3% e2e from the in_proj L2-excess
reasoning alone (+1.6-1.9% at the N-split's M-sweep-cap recovery target
0.24-0.28 s; ~+3.9% if the full 0.55 s excess is recovered - the tile-64 row
below gives +3.6%); the upside is plausibly much larger, because
tile-invariance of the 22 TF/s rate is UNTESTED by this wave (the M-sweep
varies M at fixed tile 4 only) and the sibling v3a grouped-MoE family of the
same vec_dot kernel grew 13.7 -> 39.6 TF/s from tile 4 -> 32. Decide via the
cheap measured A/B (T52: measure, don't model). The naive-traffic projection
(tile 32 -> 0.75 s dense, 848 tok/s) is REFUTED.

## Method (D09/D10; zero code changes)

1. Preflight census: no `ft serve`/`nsys`, VRAM 1372 MiB desktop baseline
   (hoptodesk app present, untouched), 5 `sem.mp-` baseline, HEAD
   cb8db23 = 1706aae + Skill/Memory commits only.
2. Boot under `nsys launch --session=denseq80cap --trace=cuda
   --cuda-graph-trace=node` (wrapper `.tasks/dense-q80-gemm/ft_nsys_dq80.sh`),
   runner `dq80_stage.sh` (trap-on-EXIT cleanup + SIGTERM->10 s->SIGKILL +
   outer `timeout 2400`; FAILPAT watchdog inside `step0_run.py` =
   measure.WATCH + `backend worker .* exited`, hard deadline 1800 s, nsys
   start after the 2nd "input throughput" line, stop after the 6th).
   Postflight census == preflight (no processes, 1371 MiB, 5 sem.mp-).
3. Fill = the 65,585-token campaign filler (8x8128 + 561 tail). Analysis:
   `nsys export --type sqlite` -> per-kernel sums; `mul_mat_q8_0` attributed
   by launch geometry (grid = (ceil(N/32), ceil(M/4)); all chunk launches have
   M=8128 -> gridY=2032; the (128,2032) group split by per-launch duration
   buckets) cross-checked against the per-block launch motif in the launch
   sequence (`seq_probe.py`) and the census from the model code + real GGUF
   header (`header_scan.py` -> denseq80_header.json).

Exact boot line (measure.py BASE_ARGS + extras, env on the process):

```
env FREETOKEN_GGUF_MOE_MTILE=32 .venv/bin/ft serve --port 18801 --host 127.0.0.1 \
  --model-path /media/ai/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL.gguf \
  --moe-cache-auto --kv-reserve-tokens 500000 --kv-cache-dtype fp8 \
  --memory-ratio 0.85 --max-prefill-length 8191 --moe-strategy hybrid \
  --moe-cpu-threads 16 --max-running-requests 1
```

## Validity gate (T44) - PASS

Steady full chunks (s): 14.81, 14.81, 14.87, 14.87, 14.90, 14.93 -> median
**T = 14.87 s = 546.6 tok/s** (8128/14.87; median of the per-chunk
throughput lines 546.58) vs the 556.43 class = **-1.8%**, inside the ~-2%
band (v3a precedent -1.45%). Chunk-1 warmup 47.67 s (170.5 tok/s) and the
bogus last-full-chunk line (3.91 s, 2077.1 tok/s, T43 tail artifact) excluded.
Decode not re-measured: no dense-side code change (MoE knob is prefill-only,
v3a already measured decode flat 14.07-14.74).

## Split of the 14.87 s chunk (MTILE=32, dense tile 4)

| term | kernels | launches/chunk | s/chunk | % |
|---|---|---:|---:|---:|
| dense q8_0 GEMM | mul_mat_q8_0 | 322 (census) | **6.04** | 40.6% |
| grouped MoE (MTILE=32) | moe_iq3_xxs/moe_iq4_xs/moe_q6_K (+moe_align) | 126+42 | 4.37 | 29.4% |
| fetch copies | fast_index_copy_multi | 42 | 3.11 | 20.9% |
| rest (attn/GDN/DSA/MHC/glue, idle ~0) | - | - | ~1.35 | ~9.1% |

Note: v3a-ab.md's e2e-subtraction estimate put the MoE term at tile 32 at
~1.8 s; direct nsys measurement says 4.37 s (the subtraction assumed a
tile-invariant rest; corrected here). Window totals: 5.988 s dense (x1215,
re-derived from denseq80.sqlite by final_split.py; the pre-fix record said
5.982), 3.106 s fetch (x164), grouped 2.749+1.531+0.090 s; chunks_eq =
57.519/14.87 = 3.868. GPU idle ~nil. Grouped-MoE rate at the routed-FLOP
recount (137.5 TF/chunk; the earlier 173 TF denominator was unanchored -
N9): 31.5 TF/s.

The 6.04 s figure carries over from the tile-4-era profile (6.04 s there,
6.04 s median-based here / 5.99 s window-normalized) - CONFIRMED.

## Op-class table (M=8128 everywhere; K/N from the real GGUF header scan)

gguf-py scan of GLM-5.3-Flash-UD-Q3_K_XL.gguf (147.5 GB, 1412 tensors,
`denseq80_header.json`): every dense-side 2D weight is Q8_0 (34 B/32 elems =
1.0625 B/elem); routed experts = batched 3D ffn_{gate,up,down}_exps
(IQ3_XXS/IQ4_XS, 43 blocks); router ffn_gate_inp [288x4096] is F32 (not MMQ);
kv_b (attn_k_b/attn_v_b 3D) and the small indexer tensors wk/wq_b/
weights_proj/compressor gate are dequantized/bf16 by design (MLA bmm
absorption; the attention.py assert pins kv_b to QuantKind.NONE; wq_b is
Q8_0 on disk - blk.11.indexer.attn_q_b [1536,4096], 6.68 MB - dequantized
at load, weight.py:147-151); lm_head (output.weight
Q6_K) runs at M=1 through the MMVQ vec kernel, not MMQ.

| class | K | N | W MB | blocks | launches/chunk | med ms | s/chunk | TF/s | model s (naive, 1638 GB/s) | meas/model |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| kda_in_proj (fused q|k|v|b|f_a|g_a, N=3x8192+64+128+128=24896) | 4096 | 24896 | 108.35 | 34 | 34 | 91.58 | 3.114 | 18.1 | 4.570 | 0.68 |
| kda_o_proj | 8192 | 4096 | 35.65 | 34 | 34 | 25.02 | 0.851 | 21.8 | 1.504 | 0.57 |
| mla_o_proj | 16384 | 4096 | 71.30 | 12 | 12 | 48.67 | 0.584 | 22.41 | 1.061 | 0.55 |
| shexp gate/up (4096->2048) | 4096 | 2048 | 8.91 | 43 | 86 | 6.19 | 0.532 | 22.04 | 0.951 | 0.56 |
| shexp down (2048->4096) | 2048 | 4096 | 8.91 | 43 | 43 | 6.19 | 0.266 | 22.05 | 0.476 | 0.56 |
| mla_q_b | 1536 | 16384 | 26.74 | 12 | 12 | 18.75 | 0.225 | 21.82 | 0.398 | 0.57 |
| dense_ffn gate/up (4096->12288) | 4096 | 12288 | 53.48 | 3 | 6 | 37.27 | 0.224 | 21.96 | 0.398 | 0.56 |
| dense_ffn down (12288->4096) | 12288 | 4096 | 53.48 | 3 | 3 | 37.45 | 0.112 | 21.85 | 0.199 | 0.56 |
| kda_ssm_f_b/g_b (128->8192) | 128 | 8192 | 1.11 | 34 | 68 | 0.82 | 0.056 | 20.84 | 0.094 | 0.59 |
| mla_q_a | 4096 | 1536 | 6.68 | 12 | 12 | 4.60 | 0.055 | 22.21 | 0.100 | 0.56 |
| mla_kv_a_mqa | 4096 | 512 | 2.23 | 12 | 12 | 1.55 | 0.019 | 22.03 | 0.033 | 0.56 |
| indexer wq_b (in 1536 -> out 4096) | 1536 | 4096 | 6.68 | 12 | **0** | - | - | - | - | not on MMQ path |
| **total** | | | | | **322** | | **6.04** | 20.0 avg | **9.78** | **0.62** |

Block motif (launch sequence probe): KDA block = [in_proj, f_b, g_b, o,
shexp gate, shexp up, shexp down] (blocks 0-2 take dense-FFN gate/up/down
instead of shexp); MLA block = [q_a, q_b, kv_a, o, shexp x3]. Per-layer MLA
launches are 4 attn + 3 shexp; kv_b and indexer wq_b are NOT mul_mat_q8_0
launches (the sequence shows exactly one (48,2032) launch per MLA layer =
q_a). Counts match the census 34x7 + 12x7 = 322/chunk; the window-normalized
count (1215/3.868 = 314.1) is lower because the capture window starts inside
chunk 3 (its head blocks 0..k are outside) - the per-launch medians are
unaffected and the median-based total (6.04 s) closes to the window total
(5.99 s) within 1%.

Other dense formats (q6_K/iq) inside the dense term: NONE. The entire dense
prefill term is a single format, q8_0. q6_K appears only in the grouped MoE
term (3 routed down-projections, moe_q6_K 0.090 s/chunk) and in lm_head
(MMVQ, decode). Candidate A only needs the q8_0 dense tile templated.

## Traffic model vs measurement (the BW-bound check)

Naive expected traffic = Sum_op ceil(M/MMQ_X) x W_op (+ activation one-pass
~20 GB/chunk, negligible): **9.78 s at 1638 GB/s** vs **measured 6.04 s = 0.62x**
(the moe_vec precedent measured 0.92x -> BW-bound; 0.62x does NOT close).

Why it does not close, and what the regime actually is - three independent
measurements (microbenches in this folder, run on the idle GPU after the
capture; synthetic qweights, exact in_proj shape N=24896/K=4096/row_bytes
4352, grid (778,2032), JIT cache warm):

1. **M-sweep at tile 4** (`microbench_dense.py`): time is exactly linear in
   M with NO intercept and NO saturation: 44.9 / 45.6 / 44.4 / 43.3 / 44.3 /
   42.2 / 41.6 / 42.8 us per row-tile at gridY = 2032 / 1016 / 508 / 254 /
   127 / 64 / 32 / 16. Per-MAC rate constant 18.1-19.9 TFLOP/s across a
   127x gridY range. A naive-DRAM-bound kernel is impossible here: the
   implied naive BW is 2413-2660 GB/s > the 1792 GB/s peak.
2. **Format A/B** (`microbench_format_ab.py`): identical shape and MACs,
   q6_K packs 23% fewer weight bytes (0.8203 vs 1.0625 B/elem) but has a
   heavier per-MAC decode. Traffic-bound predicts time ratio 0.772;
   **measured 1.063 (q6_K SLOWER)**. Traffic is not the binder; per-MAC
   decode work is.

Caveat (F7/F9): the raw microbench outputs (M-sweep table, format A/B
ratio) were NOT saved as artifacts - the numbers above exist only in this
prose; the scripts are kept and deterministic, but re-verification requires
a re-run (save stdout next time). Timing = 20/30-iter means with warmup
excluded, not medians.
3. **Per-class rate cluster**: every class with W <= 71.3 MB runs at
   20.8-22.4 TFLOP/s (ssm f_b/g_b the low outlier at 20.84; all others
   21.8-22.4; bytes/FLOP is shape-constant for q8_0 dense, 0.1328 B/FLOP =
   0.2656 B/MAC, so the measured/model ratio band 0.550-0.592 - all classes
   0.550-0.566 except ssm 0.592 - IS the ~22 TF/s ceiling expressed in
   traffic units; the pure-DRAM ceiling at 1638 GB/s would be 12.3 TF/s).
   Only `kda_in_proj` (108.35 MB > L2 100.66 MB = 96.0 MiB) drops to 18.1
   TF/s (~ -17.5%): the excess = weight re-reads from DRAM that L2 cannot
   serve. L2 provenance (F8): torch.cuda.get_device_properties(0) device
   query 2026-09-19 (Review-A): NVIDIA GeForce RTX 5090, L2_cache_size =
   100,663,296 B = 96.0 MiB, 170 SMs, 31.4 GiB.
   Attribution caveat (F4): the ~ -17.5% is a MIX, not L2 overflow alone -
   the same shape at small M (minimal re-read pressure) still runs only
   19.6-19.9 TF/s, so the DRAM/L2-dependent component is ~6-9% and the rest
   (~10%) is shape-inherent (or small-M L2 misses; ncu would tell, blocked).

ncu counter confirmation (dram__bytes.sum / warp-stall mix) was attempted and
is **blocked: ERR_NVGPUCTRPERM** (GPU perf counters disabled for this user).
The two software discriminators above (M-sweep + format A/B) pin the regime
without it; enabling counters (`NVreg_RestrictProfilingToAdminUsers=0` or
sudo ncu) would add the dram__bytes/L2-hit numbers as belt-and-braces.

## Widening projection (floor model; payoff UNMEASURED)

Reproducibility (F2): one deterministic command regenerates both
step0-split.json and the table below verbatim (step0-projection-table.md is
the generated fragment, the source of this table):

```
python3 .tasks/dense-q80-gemm/final_split.py
```

Total dense work = 120.7 TFLOP/chunk (2 x MACs, tile-invariant; sum of the
op-class table). At the 22 TF/s cluster rate the floor is ~5.48 s; measured
6.04 s; the 0.55 s slack = the in_proj excess (closes). The only recoverable
share in this model is the in_proj DRAM-stall excess (in_proj at 22 TF/s =
75.3 ms vs 91.6 ms measured -> 0.55 s/chunk at tile 4, shrinking as
(1 - 4/tile) as its weight re-read count drops); every other class is held
at its measured tile-4 time. **This is a floor model, not a prediction**: the
payoff is UNMEASURED, and tile-invariance of the 22 TF/s rate is untested
(the M-sweep varies M at fixed tile 4 only; the v3a sibling family grew
13.7 -> 39.6 TF/s from tile 4 -> 32). Conservative floor ~+2-3% e2e; upside
plausibly much larger. Decide via the cheap measured A/B (T52).

| dense tile | dense term (s) | chunk (s) | tok/s | vs this run (546.6 tok/s) |
|---|---:|---:|---:|---:|
| 4 (this run) | 6.04 | 14.87 | 546.6 | - |
| 8 | 5.76 | 14.59 | 556.9 | +1.9% |
| 16 | 5.62 | 14.46 | 562.2 | +2.9% |
| 32 | 5.55 | 14.39 | 564.9 | +3.4% |
| 64 | 5.52 | 14.35 | 566.3 | +3.6% |

(For contrast, the REFUTED traffic-bound projection at tile 32 was 0.75 s
dense / 9.58 s chunk / 848 tok/s - do not plan against it.)

Also note the register/SMEM cost of bigger dense x-tiles (sum[mmq_y/32]
[mmq_x/nwarps] grows like v3a) may erode even the floor - measure, don't
model (T52). Cheaper alternative that targets part of the SAME in_proj
excess without any kernel change: split kda_in_proj into 2 sequential sub-N
launches (24896 -> 2x 12448 rows, ~54 MB each < L2), config-level and
numerics-exact (row-independent outputs). Conservative recovery
0.24-0.28 s/chunk (+1.6-1.9% e2e) if the shape recovers only to its own
M-sweep cap of 19.6-19.9 TF/s; up to 0.55 s (~+3.9% e2e) if the halves
reach the 22 TF/s cluster. Which target applies is exactly what ncu
(blocked this wave) would decide. Out of Step-0 scope, flagged for the user.

Candidate A verdict: payoff UNMEASURED - decide via the T52 A/B, not this
model (conservative floor ~+2-3% e2e; upside plausibly much larger if the
v3a rate-vs-tile scaling transfers). The dense term's other levers are
Candidate B class (per-MAC cost: tensor-core MMQ / cheaper vec_dot) or the
out-of-scope fetch copies (3.11 s).

## Hygiene log

- Preflight (14:42): census empty, VRAM 1372 MiB, 5 sem.mp-. Postflight
  (14:47 + 15:10): identical state; `measure.teardown` reported leftover=[]
  and 1371 MiB; no FAILPAT hits; watchdog n_thr=6; nsys start/stop rc=0.
- First stage attempt aborted harmlessly before boot (wrapper lacked +x;
  PermissionError in start_server; nothing launched, census clean, trap
  cleanup verified).
- nsys artifacts confined to this task folder: denseq80.nsys-rep (renamed
  from the session-stop default report1.nsys-rep), denseq80.sqlite export;
  the aborted-iteration stray-cleanup (`rm -f report1.nsys-rep` at repo root)
  armed in the runner but not needed.
- Microbenches ran AFTER the capture, GPU idle (VRAM 1371 MiB), serialized,
  under `timeout`; no leftover processes (checked).

## Caveats (recorded; no re-runs needed)

- N4: other_terms (grouped/fetch/rest) and the gate string are constants
  from the nsys window sums; final_split.py re-derives everything
  dense-side, including the window total (5.988 s), from denseq80.sqlite
  and carries a _provenance key on the constant block. The raw sqlite is
  retained.
- N6: step0_run.json prefill_thr_summary.steady_mean (740.6) includes the
  T43-bogus last-full-chunk line (2077 tok/s) and the 561-token tail -
  foot-gun field; every verdict here uses the T43 medians (chunk-1 warmup
  + last full chunk excluded), never steady_mean.
- N8: step0_run.json boot_env={} records the DRIVER's env, not the
  server's; the MTILE=32 knob is proven by throughput (14.87 s chunk vs
  the 24.43 s tile-4 class), not by the boot-env record.
- N10: the superseded pre-audit generator is renamed to
  step0_analyze_superseded.py and write-guarded; only
  `python3 .tasks/dense-q80-gemm/final_split.py` writes step0-split.json.
- N11: capture_shutdown_rc=1 in step0_run.json is the post-stop
  `nsys shutdown` of an already-stopped session - benign.

## Artifacts

- `.tasks/dense-q80-gemm/step0-profile.md` (this file)
- `.tasks/dense-q80-gemm/step0-split.json` (machine-readable:
  kernel/class -> {launches_per_chunk, seconds_per_chunk}, regime + projections)
- `.tasks/dense-q80-gemm/denseq80.nsys-rep` + `denseq80.sqlite` (the trace;
  kept)
- `.tasks/dense-q80-gemm/denseq80_header.json` (gguf-py scan)
- `.tasks/dense-q80-gemm/step0-projection-table.md` (generated widening-table
  fragment; `python3 .tasks/dense-q80-gemm/final_split.py` regenerates it +
  step0-split.json - F2)
- `.tasks/dense-q80-gemm/step0_run.json`, `seq_probe.py`/`.out`,
  `classify_launches.py`, `microbench_dense.py`, `microbench_format_ab.py`,
  `final_split.py` (the single step0-split.json generator; opens the sqlite
  read-only), `ft_nsys_dq80.sh`, `dq80_stage.sh`, `step0_run.py`,
  `step0_analyze_superseded.py` (superseded pre-audit generator, renamed +
  write-guarded - N10), `header_scan.py`, `ncu_dense.log` (the ERR log)
- Server log: `.tasks/ft-gguf-serve-tuning/logs/dq80_nsys.log` (+ .pid)
