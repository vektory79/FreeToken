---
name: "ft-offload-decode-gather-gap"
description: "Offload gather ~29 GB/s effective vs benched 43.6 (benchbw pcie_gather_gbs); fixing gives ~18.4 tok/s > hybrid 16.29"
type: project
lastUpdated: 2026-09-26T22:42
lastRecall: 2026-09-26T23:46
---

# Offload decode: gather ~29 GB/s effective vs benched 43.6 — a real x1.5 headroom (2026-09-26)

Found in the decode-research session when the user asked to also check `--moe-strategy offload` for improvements.

## Numbers with configs
- benchbw profile on disk (`~/.cache/freetoken/benchbw/GPU-ec07f396-8269-f150-1f86-0c511e4972fb.json`), dtype_kernels.gguf: cpu_moe_gbs=66.82, pcie_gather_gbs=**43.6** (standalone), cpu_moe_overlap_gbs=49.97, pcie_gather_overlap_gbs=22.12. Auto f=30.7% = 22.12/(22.12+49.97) — matches the boot log line exactly. The profile's "dtypes" verdict for gguf is "offload" (the user overrides to hybrid explicitly).
- Production offload (re-acceptance, same real file): 12.97 tok/s @64k = 77.1 ms/step with ~2.14-2.31 GiB/step of gather = **~28-30 GB/s effective** — about 2/3 of the benched standalone rate.
- If the gather reached the benched 43.6 GB/s: step ~54 ms ~= **18.4 tok/s — above the hybrid's measured 16.29** on the user config.

**Why:** offload was written off after hybrid won 1.28x, but its production rate sits well below its own bench — the same "benched vs production rate gap" pattern as the hybrid CPU leg (26.9 vs ~50 benched), except offload has NO per-layer CPU sync: the per-layer gather is inherently serial (routing(k) known only after attn(k); the copy feeds GEMM(k)), so OVERLAP is not the lever — the copy RATE is (kernel efficiency at ~5 experts x 10.88 MB per launch, or larger batched copies, e.g. cudaMemcpyBatchAsync-style).

**How to apply:** before any offload optimization brief, decompose with the T1 trace on an instrumented offload run (copy-kernel ms/step + gaps vs the 43.6 GB/s bench). Bench-geometry caveat: the profile benched 161 synthetic experts at 13.3 MB expert_bytes vs the real 10.88 MB Q3_K_XL mix. Harness: .tasks/decode-research/ (see ft-decode-stats-instruments-unwired).
