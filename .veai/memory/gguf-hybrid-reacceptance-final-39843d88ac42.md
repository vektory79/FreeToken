---
name: "gguf-hybrid-reacceptance-final"
description: "GGUF glm5next hybrid FINAL: re-acceptance 16.67 tok/s vs offload 12.97 - hybrid RECOMMENDED; 9 commits through 63b9bff"
type: project
lastUpdated: 2026-09-15T18:18
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

## Commit chain (9 + 1)
5a04423 (CPU GEMV per-projection) -> bd02232 (fix wave) -> 427a431 (negative tests) -> 2169baa (benchbw leg) -> aad5d3a (per-cache executors) -> f6b94ad + ef83ee8 (engine gate) -> eb7de4c (hardening) -> 63b9bff (cap-down + wrap guard + profile restore) -> 1635ecd (CC/CXX JIT leak). Campaign history in the ft-serve-gguf-glm5next-unsupported memory; plan artifacts in .tasks/gguf-glm5next-hybrid/.

## Optional leftovers (not blocking)
- Vendored kernel/csrc/gguf/*.cuh lack MIT attribution headers (gap noted by the kernel study).
- benchbw profile has no kernel-tier fingerprint (pre-port profile silently consumed; doc note only).
