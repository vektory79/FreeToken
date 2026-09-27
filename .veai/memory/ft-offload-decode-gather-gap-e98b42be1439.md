---
name: "ft-offload-decode-gather-gap"
description: "Offload gather-gap lever REFUTED by T1: prod 33.5-36 GB/s vs 43.6 bench; closing only ties hybrid; offload deprioritized"
type: project
lastUpdated: 2026-09-27T02:47
lastRecall: 2026-09-27T17:48
---

# Offload decode gather gap: premise REFUTED by T1 measurement (2026-09-26)

Initial estimate (decode-research session, 2026-09-26): production offload gather
~28-30 GB/s effective vs benched 43.6 (benchbw dtype_kernels.gguf pcie_gather_gbs;
auto f=30.7% = 22.12/(22.12+49.97) matches the boot log exactly; offload
re-acceptance control 12.97 tok/s @64k moved ~2.14-2.31 GiB/step). Projection:
closing the gap -> ~54 ms/step ~= 18.4 tok/s, ABOVE hybrid 16.29 - made offload
look worth optimizing.

## T1-measured outcome (2026-09-26 evening): the lever is DEAD
- Offload on the user config measures 13.72 tok/s (72.9 ms/step); production
  gather is ~33.5-36 GB/s effective, not ~29. Real gap to bench is only x1.2-x1.3.
- Closing it fully only TIES hybrid (~16.7 band): offload has no per-layer CPU
  sync to remove, so best case is parity. Offload improvements DEPRIORITIZED.
- Full decomposition lives in ft-decode-lever-research; harness
  .tasks/decode-research/ (driver hooks: ft-decode-stats-instruments-unwired).

## Durable lesson (still valid)
The "benched vs production effective rate" gap pattern is a trap (hybrid CPU leg
26.9 vs ~50 benched misled the same way; both dissolved under instrumentation) -
decompose with a T1 trace BEFORE writing an optimization brief; a bench-gap
projection alone is not a business case.
Bench-geometry caveat stands: the profile benched 161 synthetic experts at
13.3 MB vs the real 10.88 MB Q3_K_XL mix - standalone bench rates are not
directly comparable to production gather.
