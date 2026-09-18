---
name: "gguf-hybrid-decode-handshake-floor"
description: "GGUF hybrid per-layer cost = fetch volume + 0.6-1.0 ms sync; hardware-validated 1.43 ms/layer at 16.67 tok/s"
type: project
lastUpdated: 2026-09-15T18:18
lastRecall: 2026-09-18T23:01
---

# GGUF hybrid per-layer cost: fetch volume (endogenous to CPU-leg speed) + ~0.6-1.0 ms sync

2026-09-15, RTX 5090. CORRECTED by the NVFP4 A/B (.tasks/gguf-glm5next-hybrid/verification/nvfp4-ab-measurements.md). The earlier claim "per-layer handshake floor 1.9-3.0 ms x 42 layers" (Task 05 bisect) was MIS-ATTRIBUTION.

What actually holds:
- Fixed GPU<->CPU sync: ~0.6-1.0 ms/layer (NVFP4 hybrid all-in 1.73 ms/layer = 72.6 ms/step at 15 tok/s class, fetch only 1.06 experts/layer).
- Cross-layer pipeline IS infeasible (routing dependency: block k+1 needs block k's merged output), but per-round-trip cost is small.
- The rest of the GGUF "floor" was per-layer PCIe fetch volume, ENDOGENOUS to CPU-leg speed: the scalar fp32-LUT leg (11.3 GB/s benched; 2.96 GB/s effective on the starved dominant pool) pushed f to 78% -> ~6 experts/layer fetched (~1.7-2.5 ms PCIe/layer), booked by the bisect as "handshake".
- NVFP4 hybrid wins 1.39-1.43x over offload today (13.77-14.46 vs 9.91-10.12 @64k) because its integer W4A8 leg (55.6 GB/s overlapped, 19.9/20 cores) takes 75% of misses off PCIe at f=24.9%.

**Why:** per-layer cost = max(CPU leg, f x misses x PCIe) + fixed sync; a fast CPU leg both computes misses cheaply AND shrinks the fetch volume. There is no fixed 1.9-3 ms architectural floor.

**How to apply:** model GGUF hybrid/CPU-only from CPU-leg bandwidth + fetch-volume coupling, never from a fixed handshake constant. In bisects, isolate sync at f->0 (cap-0) instead of booking fetch volume as sync. The lever is the ggml integer-kernel port (research/ggml-cpu-kernel-study.md; microbench gate >=40 GB/s sustained vs our 11.7).

## 2026-09-15 (final): hardware-validated end-to-end
The corrected model survived hardware re-acceptance: GGUF hybrid with the ported ggml kernels measured 16.67 tok/s @64k = 60.0 ms/step = 1.43 ms/layer all-in (f=30.7%, CPU leg 26.9 GB/s effective on the weighted [15,1,1] split, 19.9/20 cores) - BELOW even the 0.6-1.0 ms fixed-sync band plus modeled legs, i.e. the legs overlap better than the conservative model assumed. The fixed-sync-only footprint stays small; the actionable lesson stands: per-layer cost tracks CPU-leg bandwidth and fetch volume, not a fixed floor.
