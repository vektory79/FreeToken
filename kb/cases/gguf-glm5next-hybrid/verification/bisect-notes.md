# Task 05 bisect notes - why hybrid misses the 14.3 tok/s gate (RTX 5090, real file)

Triage class: IMPLEMENTATION (perf integration gap), not correctness, not environment.
No IMA/crash anywhere; outputs sane (24/24 battery); the gate fails on speed alone.
No code was changed; every number below is measured on the real 147.5 GB file.

## Step-time decomposition (767-token 64k decode, 42 MoE layers)

| config | tok/s | ms/step | delta vs offload |
|---|---|---|---|
| offload control (same-session) | 14.25 | 70.2 | - |
| hybrid, cap-6 fetch (CPU leg ~0, flag-sync) | 5.08 | 196.9 | +126.7 ms |
| hybrid, auto f=0.78 (cpu 0.82/layer, flag-sync) | 2.99 | 334.4 | +264.2 ms |
| hybrid, auto f=0.78, OVERLAP=0 | 2.57 | 389.1 | +318.9 ms |
| hybrid, auto f=0.78, hostfunc sync | 3.48 | 287.4 | +217.2 ms |

(1) Hybrid per-layer handshake floor: cap-6 moves the CPU leg to ~0 (fraction field
defaults 0.0 under an explicit cap, so min(6, miss) ~ 5.4 experts/layer are fetched and
the CPU computes only the <=0.6 remainder). The step is still 197 ms vs offload's 70 ms
for the SAME gather+GEMM work -> **+3.0 ms per MoE layer** of GPU<->CPU handshake cost
under the flag memop handshake (D2H routing copies, submit host node/memop, per-layer
WAIT for done[], H2D partial merge). The hostfunc handshake variant lowers the floor to
~1.9 ms/layer (287 ms step with the CPU leg present, ~80 ms handshake + ~137 ms CPU leg
+ ~70 ms gather/GEMM) but is still ~2.7x the whole offload step budget.

(2) Starved CPU leg: auto-f minus cap-6 (both flag-sync) = +137.5 ms/step for
0.8233 experts/layer x 39 layers = 407 MB of scalar IQ/QK GEMV -> **~2.96 GB/s
effective**. The profile benched 11.3 GB/s (overlapped, 20 threads); the even 3-pool
split gives the dominant partition 6 of 16 threads (log: "pool 1/3 ... threads=6"),
and the LUT scalar path scales sub-linearly. CPU sampler during decode: 7.5 of 20
cores busy - the machine is idle-starved, not saturated.

(3) Overlap is real but small: ov0 serializes and loses only ~15% (2.99 -> 2.57); the
per-layer WAIT node serializes each layer's CPU completion with the next layer's fetch,
so the overlap window never spans layers.

## Why the B-2 model missed this

B-2: step_hybrid = miss_bytes / (pcie_ov + cpu_ov) = 55.7 ms (17.9 tok/s, +21% vs
offload) ASSUMED (a) the CPU leg at the full benched 11.3 GB/s (20 threads) and
(b) zero per-layer handshake cost. Both fail on the real multi-partition config:
(a) the even affinity split (Task 02) starves the dominant pool 4x,
(b) the handshake floor alone (+80..127 ms) exceeds the whole offload step.

## In-scope fix ceiling (why no fix was committed)

- Weighted affinity split (layer-proportional 14/1/1) + per-pool matched fraction:
  CPU leg -> ~8.5-9 GB/s -> step ~ 42 x (1.46 gather + ~1.3 cpu + 1.9..3.0 handshake)
  = ~200-270 ms -> ~4-5 tok/s. Still 3x below the gate.
- The gate needs the handshake floor gone (batch the whole step's CPU work into ONE
  submit/wait, or a device-side doorbell) AND a faster CPU leg (Task 06 SIMD).
  That is a new work item, not a patch inside the landed scope.

## Recommendation to the plan

Re-scope Task 06 (SIMD tier) into a two-part wave: (a) per-step batched CPU submit /
cheaper per-layer handshake (decisive: -80..127 ms), (b) SIMD tier + workload-weighted
pool affinities + per-pool fetch fraction (residual: -100 ms). Modeled best case with
both: ~42 x (1.46 + 0.6) = ~87 ms -> ~11.5 tok/s with scalar; with SIMD cpu_ov ~25 GB/s
-> step ~ 42 x (68.5 MB / 65 GB/s + 0.6 ms) = ~70 ms -> ~14.5 tok/s, i.e. the gate is
reachable only with BOTH parts. benchbw headline verdict for gguf stays "offload"
(ratio 0.27 < 2.0) and is now empirically CORRECT on this rig, not merely conservative.

Artifacts: /tmp/ft_t05_*.log (full server logs), /tmp/ft_t05_*_result.json (client),
/tmp/hybrid_decode_stats.json (overflow stats), /tmp/ft_battery_prompts.json (battery),
/tmp/ft_serve_driver.py + /tmp/ft_task05_runner.sh + /tmp/ft_hybrid_client.py (harness).
