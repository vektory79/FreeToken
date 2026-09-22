# v2 hardware A/B: grouped MMQ prefill vs moe_vec (GGUF glm5next)

Date: 2026-09-17/18. Branch vektory79 @ 1a444e4, dirty tree = the 11-file v2
changeset (uncommitted). Single RTX 5090, same-session A/B, two env-flip boots
per ladder point - NO stash, NO commits, NO production edits in this wave.

## Headline result

The v2 grouped MMQ prefill is ACTIVE at runtime (kernel-level liveness
confirmed, see below) and delivers +14.4% / +15.6% / +16.4% prefill at
4096 / 6144 / 8128-token chunks. The 2.3-2.7x expectation (~758 tok/s @8128)
did NOT materialize: grouped MoE costs ~9.5-10.5 s of the 8128 chunk (traced
~100 ms per projection call), not the modeled 1.5-3 s - it cut the MoE term
roughly in half (18.13 s -> ~9.5-10.5 s), and the rest of the chunk (fetch
copies 3.15 s, dense q8_0 GEMM 6.04 s, attention/GDN/DSA ~1.45 s, Step-0
split) now dominates. Decode unchanged within noise (byte-identical code).
Verdict: REAL but small gain; honest verdict block below.

## Method

Env-flip A/B on the same source tree (kill switch `FREETOKEN_GGUF_GROUPED_PREFILL`,
read per call in layers/moe.py:489; default ON):

- BEFORE stage: 3 boots with `FREETOKEN_GGUF_GROUPED_PREFILL=0` (pre-v2
  moe_vec prefill behavior, same compiled extension).
- AFTER stage: 3 boots with `FREETOKEN_GGUF_GROUPED_PREFILL=1` (grouped path).
- Ladder point per boot: chunk size is a boot flag, so 4096 / 6144 / 8191 are
  three separate boots per stage (mode `prefill` for 4096/6144, mode `full`
  for 8191 = prefill + decode @ cached prefix).
- Serial boots, reaped; census before/after every stage; llama-swap inactive.

ft serve cmdline (both stages, BASE_ARGS + extra; argparse last-wins -> effective
memory-ratio 0.85, max-running-requests 1, max-prefill-length per ladder point):

```
.venv/bin/ft serve --port 18801 --host 127.0.0.1 \
  --model-path /media/ai/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL.gguf \
  --moe-cache-auto --kv-reserve-tokens 500000 --kv-cache-dtype fp8 \
  --memory-ratio 0.89 --max-prefill-length 4096 --moe-strategy hybrid \
  --moe-cpu-threads 16 --max-running-requests 1 \
  --max-prefill-length <4096|6144|8191> --memory-ratio 0.85
```

Stage runner: [v2ab_stage.sh](../harness/mmq/v2ab_stage.sh) `<name> <log> <mode>
"<extra>" "<ENV=V>"` = census -> `timeout 1800 python3 measure.py <mode>
--name <name> --log <log> --extra "<extra>" --env "<ENV=V>"` -> census.
Analyzer: `v2ab_analyze.py` -> `v2-ab-results.json`.
Metric: median of full chunks from "input throughput (token/s):" lines,
EXCLUDING chunk 1 (triton autotune warmup) and the last full chunk (tail-report
artifact, memory ft-last-chunk-throughput-artifact). Chunk geometry for the
65,585-token filler: 16x4096+49, 10x6144+4145, 8x8128+561.

## Summary table

| metric | BEFORE (env=0) | AFTER (env=1) | delta | campaign ref |
|---|---|---|---|---|
| prefill @4096 tok/s | 263.82 | 301.88 | +14.4% | ~262 |
| prefill @6144 tok/s | 281.38 | 325.28 | +15.6% | ~281 |
| prefill @8128 tok/s | 289.92 | 337.61 | +16.4% | 292.30 |
| decode steady @65k cached (63 tok) | 13.94 | 14.60 | +4.7% (noise) | 13.58-14.72 |
| boot ready_s 4096 / 6144 / 8128 | 120.3 / 146.3 / 136.7 | 130.4 / 140.4 / 128.3 | noise | 99.1 |
| quality battery | SKIPPED | SKIPPED | - | - |

Interpretation:
- The gain GROWS with chunk size (+14.4 -> +16.4%), consistent with the MoE
  term scaling with tokens while the fixed rest stays put.
- Decode is byte-identical code; +4.7% is inside the campaign's daytime
  variance band (13.58-14.72). Decode-unchanged expectation HOLDS.
- BEFORE numbers reproduce the campaign winner config almost exactly
  (263.8/281.4/289.9 vs 262/281/292.3) - valid baseline.

## Raw chunk lines (server "input throughput (token/s):", in order)

BEFORE 4096 (ab_v2_before_4096):
```
c1 84.84 (warmup, excl)
c2..c15: 263.14 262.91 252.81 264.20 264.67 264.35 264.50 264.35 264.18
         262.50 263.72 263.92 263.29 263.48            -> median 263.82
c16 881.59 (last-full artifact, excl)
tail 60.01 (49-tok, excl)
```

BEFORE 6144 (ab_v2_before_6144):
```
c1 98.31 (warmup, excl)
c2..c9: 281.85 281.61 281.69 281.91 281.14 279.70 280.90 280.30 -> median 281.38
c10 401.79 (last-full artifact, excl)
tail 2084.12 (4145-tok artifact, excl)
```

BEFORE 8191 (ab_v2_before_8191):
```
c1 108.97 (warmup, excl)
c2..c7: 290.61 289.79 290.23 288.87 290.05 289.30           -> median 289.92
c8 1572.21 (last-full artifact, excl)
tail 535.43 (561-tok, excl)   decode-req prefill 10.77 (excl)
```

AFTER 4096 (ab_v2_after_4096):
```
c1 93.03 (warmup, excl)
c2..c15: 303.20 302.54 302.90 302.24 302.69 302.75 301.98 301.62 301.79
         301.01 301.15 301.27 300.46 300.74                -> median 301.88
c16 944.37 (last-full artifact, excl)
tail 59.32 (49-tok, excl)
```

AFTER 6144 (ab_v2_after_6144):
```
c1 111.06 (warmup, excl)
c2..c9: 327.59 325.70 326.21 325.41 325.15 323.98 324.53 323.60 -> median 325.28
c10 462.63 (last-full artifact, excl)
tail 2389.80 (4145-tok artifact, excl)
```

AFTER 8191 (ab_v2_after_8191):
```
c1 123.84 (warmup, excl)
c2..c7: 338.85 338.57 337.81 337.40 336.29 336.16           -> median 337.61
c8 1760.31 (last-full artifact, excl)
tail 554.46 (561-tok, excl)   decode-req prefill 10.60 (excl)
```

Decode (full-mode 8191 boots, radix HIT both stages: #new-token: 49,
#cached-token: 65536): BEFORE steady 13.94 (p50 14.42, wall 8.92 s);
AFTER steady 14.60. Probe boot decode: 14.24-14.39 (same class).

## Liveness probe (kernel-level, report3.nsys-rep)

Instrument: one extra boot, env=1, winner flags, under `nsys launch
--session=v2probecap --trace=cuda --cuda-graph-trace=node ft serve ...`
(capture-from-launch), 65,585-token prefill + one 65k radix-hit decode, then
`nsys stop`. Driver `v2ab_probe.py` (PYTHONPATH sitecustomize surfaces the
bare-logger marker), analysis `v2ab_probe_analyze2.py` +
`v2ab_probe_liveness.json` (sqlite: v2probe2.sqlite, 386,783 kernel records).

Assertions - ALL PASS:
1. Grouped MMQ kernels present in prefill chunks: `moe_iq3_xxs`,
   `moe_iq4_xs`, `moe_q6_K` - counts 1134 / 378 / 378:
   grouped = exactly 42 routed layers x 3 projections x 9 chunk-equivalents;
   `moe_align_block_size_kernel` (sgl backend; the triton `_moe_align_small`
   is NOT used on this rig) = exactly 42 x 9.
2. `moe_vec_q` instances INSIDE the prefill chunk span (first kernel ~3.0 s ->
   last grouped end 218.826 s): ZERO. The 126 -> 42-per-layer moe_vec calls of
   the old path are gone.
3. `moe_vec_q` present in decode: 7938 = exactly 42 x 63 steps x 3 projections,
   inside the CUDA-graph replays (node-mode trace records them).
4. Marker line (sitecustomize probe, bare-logger workaround):
   `INFO:freetoken.layers.moe:gguf moe grouped mmq prefill active
   (FREETOKEN_GGUF_GROUPED_PREFILL)` - fired exactly once, on the first chunk.

Explained non-signal: 378 tiny `moe_vec_q` instances (each <= 0.037 ms) appear
in a 0.3 s burst AFTER the last prefill chunk (t = 219.169-219.475 s). These
are the prefill request's own 3 sampled tokens: 3 x 42 x 3 one-token GEMV
calls, eager (not graphed) - not prefill MoE work. The decode request's own
49-token prefill is grouped (126 grouped + 42 align in the decode phase).

Traced grouped per-call times @8128: gate (iq3_xxs) ~97-110 ms, up (iq3_xxs)
~98-110 ms, down (iq4_xs) ~105-111 ms per layer per chunk -> grouped MoE
~9.5-10.5 s/chunk vs moe_vec 18.13 s (Step-0). The Step-0 ceiling assumed the
expert weights are read ONCE per layer per chunk (~131.5 GB -> 1.5-3 s); the
measured grouped kernel is ~3-5x off that. Working hypothesis (NOT verified in
this wave, do not tune from it): the vendored vLLM moe.cuh structure assigns
one CTA per (4-token m-block, expert) pair, so each expert's weights are
re-read once per m-block (~57x per expert at 226 avg tokens/expert) and rely
on L2 exactly like the refuted v0 model - the traffic term is cut ~4x
(pairs/4), not to read-once. A read-once schedule would need persistent
weight-resident tiles. Follow-up analysis belongs in a new task.

## Verdict block

- Gain: REAL, +14.4/+15.6/+16.4% (4096/6144/8128), far below the 2.3-2.7x
  ceiling. The MoE term roughly halved; the modeled read-once roofline was NOT
  reached by the vendored grouped structure.
- Liveness: CONFIRMED at kernel level (counts exact to the layer/projection/
  chunk arithmetic) + marker line. The A/B measured the grouped path.
- Decode: unchanged within noise (code byte-identical; bomb-test pinned).
- Quality battery: SKIPPED - no reusable tooling exists in the campaign folder
  (v0 precedent); outputs were not compared for wording parity.
- Keep/drop recommendation: the change is correct, tested (126 tests green)
  and helps, but the roofline gap means the follow-up (read-once schedule or
  dense q8_0 GEMM + fetch-copy work, per Step-0 ranking) matters more than
  this kernel's current form. User decides; numbers above are the basis.

## JIT / capture notes

- NO JIT build lines in any of the 7 boots: the v2-source extension was
  already in the disk JIT cache (the fix wave's pytest runs compiled it).
  The task's "first boot JIT-rebuilds" expectation did not materialize; boot
  deltas between stages are load-path noise, not JIT.
- No CC/CXX leak: CC/CXX unset in the environment (preflight-checked);
  measure.py passes the env through unchanged.
- Default CUDA-graph capture succeeded on every boot ("Capturing graphs:
  bs = 1"); decode radix HIT (#cached-token: 65536) on all full-mode boots.

## Hygiene log

- Preflight: no ft serve alive, GPU 1497 MiB (desktop baseline), 6 sem.mp-
  semaphores, llama-swap inactive (systemctl), HEAD 1a444e4, git status =
  exactly the 11 v2 files + .veai/memory churn (parallel sessions, untouched).
- Every stage: census before/after (no ft serve, 1496-1506 MiB, 6 semaphores);
  measure.py rc=0 for all 6 stages; teardown leftover=[] every run.
- Runs strictly serialized and reaped (one measure.py at a time).
- Final census: no ft serve, 1505 MiB, 6 semaphores; tree still = the 11 v2
  files only; HEAD 1a444e4; no stash, no commits.
- Watchdog (AssertionError|OutOfMemoryError|Backend worker is gone|CUDA
  error|Traceback) active in all boots; never fired.

## Deviations / caveats

1. Liveness instrumentation took 3 attempts (probe boots only; the A/B was
   never blocked): (a) `nsys launch` 2026.1.3 rejects `-o`; (b) wrapper argv
   convention (session name must be hardcoded, `$@` goes to ft); (c) mid-run
   `nsys start` after /ready left the report with API/graph records but ZERO
   eager kernel activities (attempt 1 report2.nsys-rep), and skipping `nsys
   start` made `nsys stop` fail with "Collection stop is not allowed in this
   state" (attempt 2). Working recipe: `nsys launch` + `nsys start` right
   after /ready + `--cuda-graph-trace=node` (attempt 3, report3.nsys-rep).
   Step-0's interactive flow did not reproduce - keep the attempt-3 recipe.
2. boot_env in measure.py JSON captures the HARNESS env (always {}), not the
   server child env; the env-flip evidence = stage-runner echoes
   (`env=FREETOKEN_GGUF_GROUPED_PREFILL=0/1` recorded in the chain output),
   the behavioral difference, and the probe marker.
3. Quality battery SKIPPED (no reusable 24-prompt/parity tooling; v0
   precedent). Not fabricated.
4. The probe boot's prefill throughput (~326-330 tok/s under nsys, chunk 3
   dipped to 287.7 in one node-mode run) is excluded from all A/B statistics;
   its decode (14.17-14.39) matches the A/B class.
5. Campaign refs for 4096/6144 (~262/~281) are the phase-2 winner-family
   ladder from the tuning campaign; REPORT.md's "prefill steady" columns
   include the tail artifact and are NOT comparable.

## Artifacts

- Stage outputs: `v2ab_{before,after}_{4096,6144,8191}.out` (this folder)
- Results: `v2-ab-results.json` (per-chunk lines, medians, deltas, boot times)
- Probe: `v2ab_probe.py`, `ft_nsys_v2.sh`, `v2ab_probe.json`,
  `v2ab_probe_analyze2.py` (raw analyze dump distilled away; see
  kb/baselines/mmq-prefill-kernel/README.md), `v2probe2.sqlite`,
  `v2ab_probe_liveness.json`, `report3.nsys-rep`
- A/B boot logs: not mirrored (distilled away)
- Broken-attempt evidence kept: `report2.nsys-rep` (kernels missing),
  `report1.nsys-rep` (Step-0)
