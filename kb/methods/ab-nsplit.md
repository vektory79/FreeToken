# A/B wave: NSPLIT 1 vs 2 on hardware (kda_in_proj N-split, dense-q80-gemm replan wave 1)

Date: 2026-09-19 18:57-19:14 local. Branch vektory79, HEAD cb8db23 + the UNCOMMITTED
N-split changeset (4 python files). Single variable: env `FREETOKEN_GGUF_KDA_NSPLIT`
(unset = 1 vs "2"). Everything else fixed at the MTILE=32 anchor, port 18801, winner
flags from v3a-ab.md. One boot per value, strictly serialized + reaped, trap-on-EXIT
runner, FAILPAT watchdog, hard timeout 2400 s, SIGTERM->10 s->SIGKILL. NO commits,
NO push. Machine-readable twin: `ab-nsplit-results.json`.

## Preflight

- `git log --oneline -5`: cb8db23 "Memory" <- 2d64a58 "Skill" <- 1706aae (v3a m-tile)
  <- 7ed25e0 "Memory" <- 055b978. `git show --stat cb8db23`: 26 files, ALL under
  `.veai/memory/`; parent 2d64a58 is also .veai-only. **cb8db23 touches NO
  serving/kernel/python path** -> no external-validity caveat; the python diff vs
  1706aae is exactly the N-split changeset (git status: layers/gguf.py,
  models/glm5_next/kda.py, tests/kernels/test_gguf_quant.py,
  tests/models/test_glm5_next_kda_op.py; + .veai churn).
- Census: no `ft serve`/`nsys`, VRAM 1369-1399 MiB (desktop baseline), 5 `sem.mp-`,
  no CC/CXX. Postflight census == preflight (see Hygiene).

## Validity gate (T44) - PASS

| boot | prefill median c2..c7 (tok/s) | chunk (s) | vs morning anchor 546.5 | vs class 556.43 |
|---|---:|---:|---:|---:|
| NSPLIT=1 | **547.24** | 14.853 | +0.14% | -1.65% |
| NSPLIT=2 | **561.50** | 14.476 | +2.75% | +0.91% |

c1 warmup (168.7/168.9 tok/s) and the bogus last-full-chunk line (2111/2101 tok/s,
T43) excluded; steady lines: 551.2/547.3/549.1/547.2/547.2/546.0 (NSPLIT=1) and
564.6/562.1/562.8/560.3/560.9/559.4 (NSPLIT=2). Radix re-send HIT exactly 65536 in
both. `server_env_pins` from /proc/<pid>/environ: NSPLIT absent in boot-1,
`FREETOKEN_GGUF_KDA_NSPLIT=2` in boot-2, MTILE=32 both.

## T46 liveness (the NB-F4 wiring pin) - PASS both boots

nsys window = fill chunks 3..7 (start after the 2nd, stop after the 6th
input-throughput line), `--cuda-graph-trace=node`, sqlite export.

| boot | kda_in_proj launches/chunk | expected | per-launch med | TF/s |
|---|---:|---:|---:|---:|
| NSPLIT=1 (fused, gridX=778) | 34.18 | 34 | **91.744 ms** | 18.07 |
| NSPLIT=2 (halves, gridX=389) | 67.92 | 68 | **37.829 ms** per half | 21.91 |

mul_mat_q8_0 census: 1277 launches / 4.038 chunk-eq (NSPLIT=1) and ~1396 / 4.004
(NSPLIT=2; 348.6/chunk per the class tables, ~2% partial-window shortfall vs the
nominal 322 + 34 extra split launches per chunk, the documented step-0 effect).
Per-pair (per-call) kernel
time NSPLIT=2: **76.584 ms** = 21.65 TF/s. The knob is LIVE in the serving path.

## 4-way split of the chunk (nsys window sums, per chunk)

| term | NSPLIT=1 | NSPLIT=2 | step-0 anchor |
|---|---:|---:|---:|
| dense mul_mat_q8_0 | 6.095 s | 5.529 s | 6.04 |
| + kda span copies (assembly) | - | 0.019 s | - |
| **dense term effective** | **6.095 s** | **5.548 s** | - |
| grouped MoE (MTILE=32) | 4.297 s | 4.385 s | 4.37 |
| fetch copies (fast_index_copy_multi) | 3.058 s | 3.111 s | 3.11 |
| rest (attn/GDN/DSA/glue; idle 0.007/0.006) | 1.396 s | 1.425 s | 1.35 |
| busy | 14.846 s | 14.469 s | 14.87 |

Copy streams inside the kda span (NSPLIT=2): 2 narrow `elementwise_kernel` copies
per pair at median **0.2631 ms** (= the predicted 202.35 MB strided copy_; read+write
404.7 MB -> ~0.26 ms exactly as the wave-1 arithmetic) + 2 tiny launches; total
19.19 ms/chunk vs the 15.3-16.8 ms prediction. NSPLIT=1 shows only a 0.24 ms/chunk
elementwise baseline near the fused launch.

## e2e and the judge-on-NET verdict

- **Dense-term recovery: kernel-only 0.5119 s/chunk** (kda kernels 3.1466 ->
  2.6347) = **92.8% of the 0.5517 s step-0 L2-excess bound** (34 x (91.577 -
  75.350 ms); 92.7% with the rounded 0.552 bound); net of the split's own copies
  (19.2 ms) = **0.493 s = 89.3%**. The window dense-term delta (6.0947 -> 5.5481 =
  0.5466 s, 99.1% of the bound) is NOT causal recovery: it decomposes as the
  0.5119 s kernel delta + 0.076 s dense-FFN window-count artifact (boot-1 caught
  10.9 ffn launches/chunk vs 8.74 in boot-2, per-launch meds identical 37.28-37.60
  ms) - 0.023 s drift - 0.019 s copies.
- Judge-on-NET window 0.22-0.26 s (GROSS 0.24-0.28): measured **ABOVE the upper
  bound, in the good direction** (0.493 s net causal ~ 2x the NET window). The
  halves run at **21.91 TF/s**, INSIDE the predicted 21.8-22.4 cluster band
  (W = 54.2 MB < L2, step-0 class table) - "conservative window exceeded": the
  conservative 0.24-0.28 s window borrowed its 19.6-19.9 TF/s cap from the FUSED
  shape (N=24896, W=108 MB > L2), which never applied to the half shape; the
  class-cluster model was confirmed, not falsified.
- e2e: prefill median +2.61% (547.24 -> 561.50 tok/s), chunk -0.377 s
  (14.853 -> 14.476). The chunk-time gain (0.377 s) is smaller than the net causal
  recovery (0.493 s) because the MoE/fetch/rest terms drifted +0.17 s between boots
  (all within their step-0 anchor classes; inter-boot session noise, not a split
  cost - the split's own copy overhead is the measured 19.2 ms).

**Verdict: the N-split PAYS NET.** +2.6% e2e prefill for a python-only change with
bitwise-exact outputs; 92.8% kernel-only (89.3% net of copies) of the L2 excess
recovered, conservative window exceeded, and the halves run exactly where the
step-0 class-cluster model predicted (the 19.6-19.9 TF/s cap belonged to the fused
108 MB shape, not the halves). The remaining in_proj share (2.63 s/chunk) now
runs at the cluster rate - the L2-excess lever is spent; the next dense lever is
Candidate A (m-tile widening), as planned.

## Decode

| boot | steady tok/s | radix | graphs |
|---|---:|---|---|
| NSPLIT=1 | 13.35 | HIT 65536 | re-captured at boot ("Start capturing CUDA graphs with sizes: [1]", bs=1 only) |
| NSPLIT=2 | 13.61 | HIT 65536 | re-captured at boot (same line, bs=1) |

Flat within noise. At max-running-requests=1 only the bs=1 graph exists (MMVQ vec
path) - the dense-MMQ split cannot enter any captured graph in this config, so
decode flatness is structural; measured anyway per the brief. 13.35/13.61 sit at
the low edge of the 13.5-14.7 class; the same-session plain (non-instrumented)
boot decoded at 15.46 - above the class ceiling - which frames the gap as CUPTI
instrumentation overhead, symmetric across both A/B boots (instrumentation
depresses prefill ~1.2%: plain 553.8 vs instrumented 547.24; decode by ~1.5-2
tok/s the same way), not daytime noise; same-mode deltas and the verdict are
unaffected, absolute production tok/s sits higher.

## Cross-boot bitwise diff - PASS (the parity contract)

6 greedy prompts (2 digit, mixed lengths/topics; temperature 0, max_tokens 96,
identical order in both boots), sha256 over the raw choices JSON:
**6/6 BITWISE IDENTICAL between NSPLIT=1 and NSPLIT=2** (both CUPTI-instrumented
boots). The split is bitwise-exact end-to-end in serving, as designed.

Out-of-scope observation: a third (plain, non-instrumented) boot's outputs matched
4/6 - the two ~96-token creative generations diverged across the plain-vs-instrumented
mode. Same-mode cross-boot agreement is 6/6; this bounds cross-MODE bitwise
comparisons of long generations, not the N-split (both A/B boots agree 6/6). The
2/6 diverged prompt shas (plain vs instrumented boot-1, i=2 and i=4) are recorded
in `ab-nsplit-results.json` (comparison.bitwise_diff_cross_mode).

## Deviations / caveats

1. **First boot-1 attempt ran without nsys injection** - the driver forgot the
   `measure.FT` wrapper swap, so `nsys start/stop` silently saw no session data and
   no report was produced (plain boot; numbers still valid as a reference:
   prefill 553.8 tok/s median, decode 15.46, HIT 65536). Detected via report
   presence: silent rc=0 start/stop + "no report file found after shutdown"; a
   non-trivial `.nsys-rep` grabbed right after stop + the expected kernel counts
   in the exported sqlite are the verified collection signal on the re-runs. The
   "Collecting data" banner is NOT a valid no-collection discriminator: it is
   wrapper-config-dependent and was absent from the instrumented nsplit1/2 logs
   too (present exactly once in both step-0 logs - historical context only).
   Fixed (`measure.FT = ft_nsys_nsplit.sh`), boot-1 re-run under nsys. Evidence
   preserved: `nsplit1_plain_boot_run.json`, `logs/nsplit1_plain.log`.
2. nsys mechanics pinned by two CPU/GPU mini-probes: `stop` generates the report
   SYNCHRONOUSLY into the stop-call's cwd (empty start/stop output = no data);
   the driver now grabs the report right after stop. Note for future waves:
   silent rc=0 from `nsys start`/`stop` does NOT imply collection.
3. e2e chunk delta (0.377 s) < net causal recovery (0.493 s) - explained above
   (MoE/fetch/rest inter-boot drift, all inside anchor classes).
4. Decode 13.35/13.61 at the low edge of the class - noise band, not a regression.

## Artifacts

- This file; machine-readable results: [ab-nsplit-results.json](../baselines/dense-q80-gemm/ab-nsplit-results.json)
- Per-stage run jsons: [nsplit1_run.json](../baselines/dense-q80-gemm/nsplit1_run.json),
  [nsplit2_run.json](../baselines/dense-q80-gemm/nsplit2_run.json),
  [nsplit1_plain_boot_run.json](../baselines/dense-q80-gemm/nsplit1_plain_boot_run.json);
  the .nsys-rep/.sqlite traces were not mirrored (distilled away)
- req_ns_{0..5}.json (bitwise-diff prompts): not mirrored
- Boot logs: not mirrored (distilled away; note from the wave: `nsplit1_plain.log`
  had no `.pid` sidecar because none was written)
- Tooling (reusable): [ab_nsplit_stage.sh](../harness/ab-runner/ab_nsplit_stage.sh),
  [ab_nsplit_run.py](../harness/ab-runner/ab_nsplit_run.py),
  [ab_nsplit_analyze.py](../harness/ab-runner/ab_nsplit_analyze.py),
  [ft_nsys_nsplit.sh](../harness/ab-runner/ft_nsys_nsplit.sh)

## Hygiene log

- Preflight == postflight census: no `ft serve`/`nsys` processes, VRAM
  1369-1399 -> 1394 MiB, `sem.mp-` = 5 throughout (zero leaks); every stage serial
  and reaped; FAILPAT never fired; no backend deaths; no OOM.
- Probe leftovers cleaned (REPO/report1.nsys-rep removed); no stray reports outside
  .tasks/dense-q80-gemm/.
