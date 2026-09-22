# Task 05 A/B table - hybrid vs offload on the REAL file (2026-09-15)

All rows: same tuned plan (`--moe-cache-auto --num-tokens 70016 --memory-ratio 0.85
--max-prefill-length 4096`), same 64512-token fill prompt + radix-prefix reuse, same
client method (streamed deltas, TTFT excluded). Single-variable deltas per row.

| # | config | decode @64k tok/s | ms/step | burst mean/min | vs gate 14.3 |
|---|---|---|---|---|---|
| 1 | offload control (same-session anchor) | **14.25** | 70.2 | 14.36 / 14.21 | anchor, PASS |
| 2 | hybrid auto f=0.78, overlap ON, flag-sync (THE tuned command) | **2.99** | 334.4 | 3.07 / 2.95 | FAIL 4.8x |
| 3 | hybrid, `--moe-hybrid-max-fetch 6` (CPU leg ~0) | **5.08** | 196.9 | 4.20 / 4.02 | FAIL 2.8x |
| 4 | hybrid auto, `FREETOKEN_HYBRID_OVERLAP=0` (serial) | **2.57** | 389.1 | 2.45 / 2.39 | FAIL 5.6x |
| 5 | hybrid auto, `FREETOKEN_CPU_MOE_FLAG_SYNC=0` (hostfunc) | **3.48** | 287.4 | 3.34 / 3.27 | FAIL 4.1x |

## Burst probe verdict ("not slower than offload" per step)

FAIL - every burst step on every hybrid config is 3-6x slower than the offload control's
worst step (hybrid best per-step 4.2 vs offload worst 14.21). The burst regime
(cold routing, ~95% miss rate) is exactly where the B-2 math promised the largest
hybrid win; instead the per-layer handshake + starved CPU leg dominate (bisect-notes.md).

## Overlap A/B verdict

`FREETOKEN_HYBRID_OVERLAP=0` costs ~15% (2.99 -> 2.57; 334 -> 389 ms): the CPU/fetch
overlap works within a layer, but the per-layer WAIT node prevents the pipeline from
spanning layers, so overlap can only hide the CPU leg up to the gather time.

## Handshake A/B verdict

cudaLaunchHostFunc sync (row 5) is ~16% faster than the flag memop handshake
(row 2: 287 vs 334 ms) but the floor remains: rows 3+5 decompose to a per-layer
GPU<->CPU handshake cost of ~1.9-3.0 ms x 42 layers, which alone exceeds the whole
70 ms offload step. No env-level lever recovers the gate.

## Reading

The benchbw headline verdict "offload" for gguf (ratio 0.27 < 2.0) is empirically
CORRECT on this rig: scalar-gguf hybrid as landed cannot beat offload at any miss rate.
The B-2 "+21%" premise (CPU leg at full 20-thread bandwidth + zero per-layer sync cost)
fails on both counts in the real multi-partition config. See bisect-notes.md for the
decomposition and the Task 06 re-scope recommendation.
