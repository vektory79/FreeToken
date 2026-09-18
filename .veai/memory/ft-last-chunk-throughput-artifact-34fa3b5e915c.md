---
name: "ft-last-chunk-throughput-artifact"
description: "Final full prefill chunk's input-throughput line is bogus (~1552-1602 tok/s); use median of full chunks minus last"
type: project
lastUpdated: 2026-09-17T18:22
lastRecall: 2026-09-17T23:08
---

# Last-chunk throughput artifact in ft serve prefill logs

Discovered 2026-09-17 (Step-0 nsys wave, .tasks/mmq-prefill-kernel/). The FINAL full 8128-token chunk always reports ~5.2-5.4 s (1552-1602 tok/s) and the trailing 561-token tail ~1.15 s in the server "input throughput (token/s):" lines - hardware-impossible given the measured per-stage split (MoE GEMM alone is 18 s/chunk). It is a tail report/pipelining artifact, and it reproduces in the 2026-09 tuning-campaign logs too (previously unnoticed).

**Why:** the last chunk's throughput line is emitted while the tail is still draining; the final full chunk's per-chunk throughput is bogus.

**How to apply:** prefill throughput stats = median of full chunks EXCLUDING the last one; an implausibly fast final chunk in any serve log is this artifact, not a real speedup.
