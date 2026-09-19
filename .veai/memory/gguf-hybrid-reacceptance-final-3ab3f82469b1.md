---
name: "gguf-hybrid-reacceptance-final"
description: "GGUF glm5next hybrid FINAL verdict: re-acceptance gates pass, 16.67 tok/s vs offload 12.97, hybrid RECOMMENDED @63b9bff"
type: project
lastUpdated: 2026-09-16T20:16
lastRecall: 2026-09-19T03:10
---

# GGUF glm5next hybrid: FINAL re-acceptance - hybrid RECOMMENDED

2026-09-15, hardware re-acceptance of the hybrid campaign on the real file (HEAD 63b9bff, no code changes; artifact: .tasks/gguf-glm5next-hybrid/verification/re-acceptance.md).

## Verdict
All three gates PASS; hybrid is now the RECOMMENDED strategy for GLM-5.3-Flash-UD-Q3_K_XL.gguf on this rig (RTX 5090):
- Decode @64k: hybrid 16.67 tok/s (short 16.12; burst min 15.04) vs offload control 12.97 (14.32 short; burst max 14.36) = 1.28x offload, 1.17x over the 14.3 campaign gate, 5.6x over the pre-port 2.99.
- ms/step 60.0 = 1.43 ms/layer all-in - BETTER per-layer than the NVFP4 hybrid's 1.73.
- Battery 24/24 on-topic (gate 20+); divergence 0-66 chars vs accepted baseline.

## Verified mechanism
Offload is PCIe-bound: all 5.05 misses/layer cross PCIe = 2.14 GiB/step at 28.4 GB/s effective. Hybrid: f=30.7% auto (not cap-1), fetched 1.33/layer (~573 MB/step), CPU leg 3.72/layer (~1.61 GB/step at 26.9 GB/s effective, 19.9/20 cores), pools [15,1,1], isa avx2-w4a8k, plan 1234 slots (463/288/288). NVFP4 sanity: user's 1M command booted, f=24.8%, short decode 15.18 (profile restoration confirmed).

## Commit chain
Full chain (11 hybrid commits + 1635ecd, incl. 11f1a80 ggml AVX2 W4A8-K tier and 979e3fc weighted pool split [15,1,1]) lives in ft-serve-gguf-glm5next-campaign - single copy kept there. Re-acceptance verified HEAD 63b9bff with no code changes; plan artifacts in .tasks/gguf-glm5next-hybrid/.

## Optional leftovers (not blocking)
- Vendored kernel/csrc/gguf/*.cuh lack MIT attribution headers (gap noted by the kernel study).
- benchbw profile has no kernel-tier fingerprint (pre-port profile silently consumed; doc note only) - expanded in benchbw-profile-clobber-trap.
