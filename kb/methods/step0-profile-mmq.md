# Step 0 - pin the GEMM/copies/rest split of an 8128-token prefill chunk

Measured 2026-09-17 (evening), vektory79 @ 1a444e4, single RTX 5090, GGUF
GLM-5.3-Flash-UD-Q3_K_XL, winner flags (`--moe-cache-auto --kv-reserve-tokens
500000 --kv-cache-dtype fp8 --memory-ratio 0.85 --max-prefill-length 8191
--moe-strategy hybrid --moe-cpu-threads 16 --max-running-requests 1`), port
18801. GPU was idle before and after; one serial run; no production or kernel
code changed.

## Method

Chosen: **nsys interactive session, zero code changes** (least invasive).
`nsys` 2026.1.3 is installed; torch.profiler would have required env-gated
edits to the MoE path plus a revert cycle, so the nsys route was taken.

Flow (artifacts in `.tasks/mmq-prefill-kernel/`):

- `ft_nsys.sh` (TEMPORARY, deleted after measurement): `exec nsys launch
  --session=step0cap --trace=cuda .venv/bin/ft "$@"` - boots the server with
  profiling configured but collection deferred. Note: in modern nsys
  `--sample` belongs on `nsys start`, not `launch` (two smoke runs failed on
  this before the flow validated).
- `step0_run.py`: driver that imports
  `.tasks/ft-gguf-serve-tuning/measure.py` UNCHANGED (only `measure.FT` is
  pointed at the wrapper), boots winner config via `measure.start_server`,
  waits `/ready` via `measure.wait_ready`, sends the 65,585-token filler
  prompt via `measure.do_prefill` (same as campaign `prefill` mode), tears
  down via `measure.teardown`. A watchdog thread tails the server log:
  FAILPAT (`AssertionError|OutOfMemoryError|Backend worker is gone|CUDA
  error|Traceback`) -> hard kill; after the 2nd `input throughput` line ->
  `nsys start --session=step0cap --sample=none`; after the 6th line ->
  `nsys stop --session=step0cap` (report written immediately). Hard deadline
  1800 s; whole run under `timeout 2400`.
- Capture window therefore spans: end of chunk 2's report -> start of chunk
  7, i.e. chunk 3 (minus ~1-2 s), chunks 4-6 full, head of chunk 7 -
  approximately 4 chunk-equivalents of steady state, away from the chunk-1
  autotune warmup and the tail anomaly (see below).
- `step0_analyze.py`: `nsys export --type sqlite`, then SQL sums over
  `CUPTI_ACTIVITY_KIND_KERNEL` (per kernel name) and
  `CUPTI_ACTIVITY_KIND_MEMCPY` (per copyKind, bytes), plus the server
  throughput lines from `step0_run.json`.

Attribution rules: GEMM = all kernels whose name matches `moe|mul_mat|ggml`
(the gguf JIT kernels; top-kernel table in the results section shows exactly
what matched); copies = H2D `cudaMemcpy` records (copyKind=1, the per-chunk
bank materialization); rest = steady chunk time - GEMM - copies (includes
attention/GDN/DSA/dense, CPU-side scheduler, GPU idle and the CPU-side part
of the gather). Normalization: `chunks_eq = W / T` where W = GPU-activity
span of the capture window, T = median steady full-chunk time from the
server's own `input throughput` lines (the same authoritative metric as the
campaign). Cross-checks: moe-kernel launch count / chunks_eq (expected ~126 =
42 layers x 3 projections) and chunk times inside vs outside the capture
window (profiler overhead).

## Exact commands

- preflight census: `pgrep -af '[f]t serve'` (empty), `nvidia-smi`
  (1438 MiB desktop baseline / 32607 total, 0% util), `ls /dev/shm | grep -c
  "^sem\.mp-"` (6, pre-existing baseline, no growth after run), `git status`
  (only `.veai/memory/*` modified by the memory bank - untouched), HEAD
  1a444e4.
- `cd .tasks/mmq-prefill-kernel && timeout 2400 .venv/bin/python step0_run.py`
- analysis: `nsys export --type sqlite -o step0.sqlite report1.nsys-rep`;
  `python3 step0_analyze.py` (prints top kernels, memcpy table, split).

## Raw measurements

Server `input throughput` full-chunk times (8128 new tokens each; this run,
2026-09-17 17:07-17:10): chunk 1 = 76.93 s (105.66 tok/s, triton autotune warmup,
excluded), chunks 2-7 = 28.73, 28.72, 28.77, 28.90, 28.80, 28.87 s (282.2-282.9
tok/s), chunk 8 = 5.23 s (1552.9 tok/s - tail anomaly, see below), 561-token tail =
1.15 s (485.9 tok/s). Steady median T = 28.77 s (n=7 incl. outlier; excluding the
outlier: mean 28.80 s, spread 28.72-28.90). Chunks inside the capture window (3-6)
vs outside (2, 7) are indistinguishable -> profiler overhead is nil.

Capture window: started inside chunk 3, stopped inside chunk 7; W = 113.89 s of GPU
activity, chunks_eq = W / T = 3.958. All numbers below are per 8128-token chunk =
window totals / chunks_eq.

Top kernels in the window (nsys sqlite, CUPTI_ACTIVITY_KIND_KERNEL):

| kernel | total s | launches | per chunk | per-chunk interpretation |
|---|---|---|---|---|
| moe_vec_q | 71.750 | 501 | 18.13 s | MoE GEMM: 126 launches/chunk = 42 routed layers x 3 projections (gate, up, down) |
| mul_mat_q8_0 | 23.914 | 1236 | 6.04 s | dense-side quantized GEMM (q8_0 MMQ kernel): ~312 launches/chunk (attn/gdn/dsa projections, shared expert, leading dense) |
| fast_index_copy_multi | 12.457 | 168 | 3.15 s | expert bank materialization: ~42 launches/chunk = one per routed layer; 74 ms x 42 = 3.13 GB/layer at ~42 GB/s = PCIe-class read of host banks |
| _glm_dsa_sparse_kernel | 2.156 | 44 | 0.54 s | DSA sparse attention |
| elementwise/vectorized/cat/quantize_q8_1 | ~1.6 | - | ~0.41 s | activation glue + q8_1 activation quant |
| _mhc_stage1/2/3, _indexer_logits | ~0.92 | - | ~0.23 s | DSA indexer/MHC |
| GDN/KDA kernels (chunk_gla, chunk_kda, chunk_gated_delta_rule, causal_conv1d, l2norm, recompute_w_u, ...) | ~0.8 | 133-block class | ~0.20 s | gated delta net / linear attention |
| everything else | ~0.2 | - | ~0.05 s | misc |

Memcpy records (CUPTI_ACTIVITY_KIND_MEMCPY) are negligible: H2D = 66 copies of
~13 KiB, ~0 s; D2D = 1429 copies, 26.3 GB total (6.65 GB/chunk, avg 18 MB, max 533
MB), 0.032 s; D2H = 6 tiny. The expert materialization is NOT a cudaMemcpy - it is
the fast_index_copy_multi KERNEL reading host banks (mapped/pinned) over PCIe.

Pinned split per 8128-token chunk (T = 28.77 s):

| bucket | s | % of chunk |
|---|---|---|
| MoE GEMM (moe_vec_q, 42 layers x 3 proj) | 18.13 | 63.0 |
| expert copy/fetch (fast_index_copy_multi, 42 layers) | 3.15 | 10.9 |
| rest (attention/GDN/DSA/dense/scheduler/everything else) | 7.49 | 26.0 |
|   - of which dense q8_0 GEMM (mul_mat_q8_0) | 6.04 | 21.0 |
|   - remaining (GDN/KDA/DSA/router/quantize/elementwise + idle ~0.015 s) | ~1.45 | ~5.0 |

GPU busy = 28.757 s/chunk vs T = 28.773 s -> idle 0.015 s/chunk (0.05%): the GPU is
never waiting on the CPU; there is no hidden host-side stall in wall time.

Cross-checks: moe_vec_q = 126.6 launches/chunk (expect 126); fast_index_copy_multi =
42.4/chunk (expect 42); moe_vec_q effective bandwidth = 29.7 TB/chunk (model traffic:
65,024 pairs x 10.88 MB x 42 layers) / 18.13 s = 1638 GB/s = ~92% of the ~1.79 TB/s
VRAM roofline -> the GEMM really is ~100% VRAM-BW-bound; fast_index_copy_multi =
3.13 GB / 74.2 ms = 42.2 GB/s vs the benched 43.6 GB/s PCIe.

Tail anomaly (reproduced in every run incl. campaign logs p_8191_r085 and
win_p_8191_r085: 1598-1602 tok/s there): the LAST full 8128 chunk reports ~5.2-5.4 s
and the 561-token tail ~1.15 s. That is hardware-impossible against the measured
steady split (GEMM alone is 18 s), so it is a tail accounting/pipelining artifact of
the final chunk reports, not real speedup. It is excluded from the split; median of
full chunks (the campaign metric) is robust to it. Flagged for follow-up only.

Implied ceilings (from THESE numbers, not the TASK.md model):

- v0 (sort pairs by expert, same moe_vec_q kernel; VRAM reads drop to 131.5
  GB/chunk, re-reads served from L2 at 29.7 TB/chunk): ceiling is set by L2
  bandwidth: L2 at 4 TB/s -> MoE 7.4 s -> chunk 18.1 s -> ~450 tok/s; L2 at
  8 TB/s -> MoE 3.7 s -> chunk 14.4 s -> ~566 tok/s. Naive GEMM/2 basis ->
  chunk 19.7 s -> 412 tok/s. So v0 lands ~1.5-2.0x (~412-566 tok/s).
- v2 (grouped MMQ, weights read once): MoE = 131.5 GB / 1638 GB/s (measured
  effective BW) = 0.08 s (+ MMQ inefficiency; even at 60% BW ~0.13 s) ->
  chunk ~10.7 s -> ~758 tok/s (2.68x). After v2 the ranking of what is left:
  dense q8_0 GEMM 6.04 s > expert copies 3.15 s > attention/GDN/DSA ~1.45 s;
  hard floor ~758 tok/s unless the dense GEMM or the copy term is addressed.
- Note the model's "copies ~3.0 s" term exists but is a copy KERNEL over PCIe,
  not a memcpy storm; and the "rest" is ~100% GPU kernels, not CPU overhead.

Full machine-readable results: step0_split.json; capture/driver state:
step0_run.json; trace: report1.nsys-rep (+ step0.sqlite export).

## Reconciliation

vs the TASK.md model (chunk 8128 = 27.7 s: GEMM 16.6-20.8 s + copies ~3.0 s + rest
4-10 s):

- GEMM: measured 18.13 s -> INSIDE the 16.6-20.8 band, and the effective 1638 GB/s
  (~92% of VRAM roofline) confirms the 226x-re-read traffic model.
- copies: measured 3.15 s -> matches the ~3.0 s estimate almost exactly, but the
  mechanism is a PCIe-reading copy kernel (fast_index_copy_multi, 42/layer, ~42 GB/s),
  NOT H2D memcpys (those are ~0). Magnitude right, mechanism corrected.
- rest: measured 7.49 s -> mid-band of 4-10 s; decomposed: 6.04 s is the dense
  q8_0 GEMM (mul_mat_q8_0) - a bigger, more actionable target than expected - and
  only ~1.45 s is attention/GDN/DSA/router/glue. GPU idle ~0.
- Verdict: the 16.6-20.8 / ~3.0 / 4-10 model HOLDS (sum 18.1+3.1+7.5 = 28.77 s,
  vs 27.7 model with overlapping bands). No correction needed to the v0/v2 plan;
  the new information is the rest decomposition (dense q8_0 GEMM dominates rest)
  and that the GPU is never CPU-starved.

vs the NVFP4 reference: the 510 tok/s figure is ~507 tok/s steady prefill measured
on 4096-token chunks (.tasks/gguf-glm5next-hybrid/verification/nvfp4-
ab-measurements.md, hybrid == offload, prefill identical in both strategies) =
8.08 s per 4096-token chunk. Consistency: that chunk must cover NVFP4's own MoE
GEMM + its rest; its per-token rest budget (<= ~1.97 ms/token) vs GGUF's measured
rest (7.49 s / 8128 = 0.92 ms/token at 8128-token amortization) leaves ample room -
no contradiction; the reference only bounds rest from above and holds only if
rest_4096 + NVFP4 MoE <= 8.08 s. After v2 (~758 tok/s implied) GGUF converges with
and exceeds the NVFP4 507 on the same geometry, as predicted. Note the NVFP4
reference was taken at 4096 chunks (fixed per-chunk costs amortize over half the
tokens there, making its per-token number look worse than an 8128-chunk run).

Also: this run's steady chunks ran 28.72-28.90 s (282-283 tok/s) vs the campaign's
291-293 tok/s - ~3% run-to-run/daytime variance; split percentages are unaffected.

## Hygiene log

- Preflight census (before launch): `pgrep -af '[f]t serve'` -> empty; `nvidia-smi`
  -> 1440 MiB desktop baseline of 32607 MiB, 0% util; `ls /dev/shm | grep -c
  "^sem\.mp-"` -> 6 (pre-existing box baseline); git HEAD = 1a444e4 (vektory79),
  working tree had only `.veai/memory/*` changes from the memory bank - untouched,
  nothing stashed or committed.
- Boot: winner flags exactly as specified (ServerArgs dump confirms memory_ratio
  0.85, max_extend_tokens 8191, max_running_req 1, moe_prefill_overlap True but
  disabled per partition by the engine: 385/288/288 slots < 2x288 -> synchronous
  materialized prefill, same as the campaign's 410-slot boot); `--moe-cache-auto`
  resolved 1156 slots; free after init 4.05 GiB; f = 30.7% fetch fraction. No
  fail-fast -> the 0.90 fallback was NOT needed (no deviation).
- Runner: step0_run.py wraps measure.py (reused unchanged) - FAILPAT watchdog every
  2 s (AssertionError|OutOfMemoryError|Backend worker is gone|CUDA error|Traceback),
  hard deadline 1800 s, external `timeout 2400`; single serial run; teardown via
  measure.teardown (pkill -TERM 'ft serve', 10 s grace, then SIGKILL escalation) -
  reported leftover=[] and gpu back to 1439 MiB; no FAILPAT hits, no manual
  SIGKILL needed.
- Postflight census: no ft serve; VRAM 1439 MiB; semaphores 6 (no growth).
- nsys gotchas hit during smoke (two failed iterations before the flow validated):
  `nsys launch` rejects `-o/--output` and `--sample` (the latter moved to
  `nsys start`); the report is written by `nsys stop` into the CWD of the start
  command as report1.nsys-rep. The aborted smokes left report1.nsys-rep +
  report1.sqlite at the repo ROOT (untracked) - both removed after the real run;
  no other repo-root pollution. No code changes were made to the engine at any
  point (nsys is external); `git diff`/status show no profiling scaffolding
  (temporary ft_nsys.sh wrapper was deleted after the run; driver/analysis scripts
  live only in this git-ignored task folder).
- Server log: .tasks/ft-gguf-serve-tuning/logs/step0_nsys.log (measure.py reused;
  do_prefill rebuilt req_prefill.json from filler.txt with identical content).
