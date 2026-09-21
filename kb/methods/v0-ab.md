# v0 hardware A/B: expert-sorted prefill pair order (GGUF moe_vec)

Date: 2026-09-17. Branch vektory79 @ 1a444e4. Single RTX 5090, same-session A/B.
v0 diff = 6 files (layers/moe.py, kernel/gguf.py, kernel/csrc/gguf/moe_vec.cuh,
kernel/csrc/gguf/gguf_kernel.cu, tests/kernels/test_gguf_quant.py,
tests/models/test_gguf_expert_banks.py; 319 insertions / 28 deletions), uncommitted.

## Headline result

The expert-sorted prefill pair order is ACTIVE at runtime (probe-confirmed marker
line) and delivers NO measurable prefill gain: 289.54 -> 291.26 tok/s (+0.6%,
within the ~3% run variance). The expected 412-566 tok/s (Step-0-implied) /
580-680 (pre-Step-0 estimate) did NOT materialize. Decode unchanged. The
boot-log liveness line required by the protocol can never appear (defective
instrument, see Liveness below) - it was recovered with a harness-side probe.

## Method

Serial same-session A/B, one boot per stage, winner flags, port 18801,
llama-swap stopped (systemctl: inactive). Harness:
`.tasks/ft-gguf-serve-tuning/measure.py` mode `full` (boot + one 65,585-token
prefill + one decode @ cached prefix + teardown), which self-censuses, refuses
concurrent ft serve, watchdogs AssertionError|OutOfMemoryError|Backend worker
is gone (plus CUDA error/Traceback), and tears down SIGTERM -> SIGKILL.

ft serve cmdline (both stages, BASE_ARGS + extra; argparse last-wins ->
effective memory-ratio 0.85 / max-prefill-length 8191 / max-running-requests 1;
verified by free-after-init 4.07 GiB and the 65024+561 chunk split):

```
.venv/bin/ft serve --port 18801 --host 127.0.0.1 \
  --model-path /media/ai/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL.gguf \
  --moe-cache-auto --kv-reserve-tokens 500000 --kv-cache-dtype fp8 \
  --memory-ratio 0.89 --max-prefill-length 4096 --moe-strategy hybrid \
  --moe-cpu-threads 16 --max-running-requests 1 \
  --max-prefill-length 8191 --memory-ratio 0.85
```

Stage runner: `.tasks/mmq-prefill-kernel/v0ab_stage.sh <name> <log>` = census
before -> `timeout 1800 python3 measure.py full --name <name> --log <log>
--extra "--max-running-requests 1 --max-prefill-length 8191 --memory-ratio
0.85"` -> census after.

1. BEFORE: `git stash push -m "v0-ab-baseline" -- <the 6 files>`; verified
   `git diff` clean for them (only .veai/memory/* remained, untouched). Ran
   stage v0_before (log logs/ab_v0_before).
2. `git stash pop` - clean, no conflicts; `git diff --stat` matched the original
   319/28 exactly.
3. Sanity: `timeout 1500 uv run --extra dev pytest tests/kernels/test_gguf_quant.py
   -q` -> 29 passed in 36.4 s (this also built the v0-source JIT extension).
4. AFTER: stage v0_after (log logs/ab_v0_after).
5. Liveness probe: one extra boot + 4,371-token prefill with a PYTHONPATH-
   injected `sitecustomize` that adds a root logging handler (harness-side
   only, no production edits): `.tasks/mmq-prefill-kernel/v0ab_probe.py`.

Prefill metric: median of the six full 8128-token chunks excluding chunk 1
(triton autotune warmup) and excluding the last full chunk (tail-report
artifact ~1568-1598 tok/s; see memory ft-last-chunk-throughput-artifact). The
561-token tail and the decode request's 49-token prefill line are excluded.
Decode metric: SSE steady tok/s over the streamed 63-token completion on the
cached 65,585-token prefix (#new-token: 49, #cached-token: 65536 - radix HIT;
first SSE frame skipped by the patched harness).

## Raw numbers (server "input throughput (token/s):" lines, in order)

BEFORE (HEAD, JIT recompiled during boot):
```
c1  108.20   (warmup, excluded)
c2  289.64
c3  290.29
c4  289.70
c5  288.44
c6  289.45
c7  288.94
c8 1585.78   (last-full-chunk artifact, excluded)
c9  534.13   (561-token tail, excluded)
    10.57    (decode request's 49-token prefill, excluded)
```

AFTER (v0 restored, JIT cache hit):
```
c1  110.87   (warmup, excluded)
c2  292.22
c3  292.29
c4  291.44
c5  291.08
c6  290.77
c7  290.70
c8 1568.70   (last-full-chunk artifact, excluded)
c9  600.68   (561-token tail, excluded)
```

Campaign reference (logs/win_p_8191_r085.log, same method):
c2..c7 = 293.25, 293.18, 292.62, 291.35, 291.98, 291.55 -> median 292.30.

## Summary table

| metric | BEFORE (HEAD) | AFTER (v0) | delta | campaign ref |
|---|---|---|---|---|
| prefill median (full 8128 chunks, excl. c1+c8) | 289.54 tok/s | 291.26 tok/s | +0.6% | 292.30 |
| decode steady @65k cached (63 tok) | 14.92 tok/s | 15.23 tok/s | +2.1% | 13.58 |
| decode p50 | 15.23 tok/s | 15.32 tok/s | +0.6% | 13.43 |
| decode wall (63 tok, radix HIT) | 8.69 s | 8.53 s | -1.8% | - |
| boot ready_s | 146.6 (JIT recompile) | 124.3 (JIT hit) | -22.3 s (JIT cache state, not code) | 99.1 |
| free-after-init | 4.07 GiB | 4.07 GiB | 0 | 4.11 GiB |
| peak VRAM (prefill) | 32078 MiB | 32078 MiB | 0 | - |
| prompt tokens / chunk split | 65585 = 8x8128 + 561 | 65585 = 8x8128 + 561 | same geometry | same |
| quality battery | SKIPPED | SKIPPED | - | - |

Interpretation:
- Prefill gain +0.6% is inside the ~3% run-variance band (this session's BEFORE
  289.5 vs Step-0 run 282-283 vs campaign 291-293 are the same config). NO GAIN.
- The sort is prefill-only (decode passes order=None); decode steady +2.1% is
  noise-class. Decode-unchanged expectation HOLDS.
- This-session decode (14.9-15.2) runs ~+10% above the campaign's 13.58 -
  consistent with the daytime-variance class seen earlier in the campaign;
  BEFORE/AFTER share it, so the A/B comparison stands.

## Liveness: the boot-log marker is a defective instrument; sort CONFIRMED ACTIVE

Protocol said the AFTER boot log must contain the one-time line
"gguf moe prefill: expert-sorted pair order active" and that absence means the
sorted path never ran. The line was absent from BOTH A/B boot logs - and it can
NEVER appear in any boot log:

- `layers/moe.py:29` uses a bare stdlib `logger = logging.getLogger(__name__)`;
  the emission at :453 is `logger.info(...)`.
- freetoken logging (`utils/logger.py init_logger`) attaches handlers only to
  loggers created through it; nothing configures the root logger.
- A handler-less logger propagates to root, which has none, so Python's
  `logging.lastResort` handles the record - and it drops INFO (WARNING+ only).

The server log's visible records all come from init_logger'd loggers
([...|core|rank=0], [...|FrontendAPI], ...); `freetoken.layers.moe` is not one.

Probe recovery (no production edits): `.tasks/mmq-prefill-kernel/probe_sitecustomize/sitecustomize.py`
calls `logging.basicConfig(level=logging.INFO)` and is injected via
`PYTHONPATH`; `v0ab_probe.py` boots the same winner config, sends one 4,371-
token prefill, greps the log:

```
INFO:freetoken.layers.moe:gguf moe prefill: expert-sorted pair order active
```

The marker fired exactly once, on the first prefill chunk. So the v0 sorted
path RAN during real chunked prefill - and the prefill numbers above are a
valid measurement of it. Conclusion: the sort is active but does not help.

## Why this matters for v2

The sort's premise was that clustering same-expert pairs lets the 96 MB L2
absorb the per-pair re-reads of the expert's 10.88 MB weight set, cutting the
MoE GEMM's ~29.7 TB/chunk VRAM traffic. Measured: the MoE GEMM runs at the
same effective rate with sorting on. The L2-reuse model does not hold on this
hardware/workload (block-scheduling wavefront vs L2 behavior, or a
read-path that bypasses L2 reuse) - the exact microarchitectural reason is
out of scope for this measurement wave. v2 (grouped MMQ: weights read ONCE per
expert per layer) changes the traffic term itself and remains the promising
path; v0 as implemented is a throughput no-op and should not be kept on its
performance merits.

## Hygiene log

- Preflight census: no ft serve, GPU 1430 MiB (hoptodesk desktop baseline),
  6 /dev/shm sem.mp- semaphores, llama-swap inactive (systemctl).
- Each A/B stage and the probe: census before and after in the runner output;
  measure.py rc=0; teardown leftover=[]; GPU back to 1430 MiB; semaphore count
  unchanged at 6. Runs strictly serialized, reaped before the next.
- Watchdog active in all boots (AssertionError|OutOfMemoryError|Backend worker
  is gone|CUDA error|Traceback); none fired.
- Final census: no ft serve, 1430 MiB, 6 semaphores - clean.
- Working tree end state: ONLY the 6 v0 files modified (+ pre-existing
  .veai/memory/* churn, untouched); no stash left (stash list empty); nothing
  committed.

## Deviations / caveats

1. Quality battery SKIPPED: no reusable 24-prompt/parity tooling exists in
   `.tasks/ft-gguf-serve-tuning` (grep over py/md/json found nothing; REPORT.md
   has no quality section). Not fabricated.
2. Liveness check as specified (line in boot log) is unpassable by design -
   instrument defect documented above; recovered via the PYTHONPATH probe.
   The moe.py marker should be re-wired through `init_logger` (or print) in a
   later wave if boot-log liveness is wanted.
3. BEFORE boot included a one-time JIT recompile of the HEAD-hash gguf
   extension (clang++ lines in the log; cache miss) - boot 146.6 s vs 124.3 s.
   Boot-time delta reflects JIT cache state, not the v0 change; steady-state
   prefill/decode are unaffected.
4. Decode metric is the campaign's full-mode single decode @ cached prefix
   (13.58 reference), not the phase2 repeated-decode variant; radix HIT
   verified both stages (#new-token: 49, #cached-token: 65536).
5. The probe boot is a third boot with a 4,371-token request; its throughput
   line (137.09 tok/s) is excluded from all A/B statistics.
6. memory-ratio 0.90 fallback was NOT needed (boots passed the floor gate at
   0.85 both stages, free-after-init 4.07 GiB).

## Artifacts

- Stage runner: `.tasks/mmq-prefill-kernel/v0ab_stage.sh`
- Analyzer: `.tasks/mmq-prefill-kernel/v0ab_analyze.py`
- Stage raw outputs: `v0ab_v0_before.out`, `v0ab_v0_after.out` (task folder)
- Server logs: `.tasks/ft-gguf-serve-tuning/logs/ab_v0_before`, `ab_v0_after`
- Probe: `v0ab_probe.py`, `probe_sitecustomize/sitecustomize.py`, `v0ab_probe.out`,
  `logs/ab_v0_probe`
- Machine-readable results: `v0-ab-results.json` (this folder)
