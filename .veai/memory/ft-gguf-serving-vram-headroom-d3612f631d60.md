---
name: "ft-gguf-serving-vram-headroom"
description: "GGUF glm5next VRAM headroom: ratio 0.80 + reserve 400k = 1.96 GiB free at -0.1% prefill; ratio floor 0.80 fail-fast"
type: project
lastUpdated: 2026-09-20T14:37
lastRecall: 2026-09-20T17:40
---

# GGUF glm5next serving VRAM headroom: measured config matrix (RTX 5090, 32607 MiB)

Measured 2026-09-20 (7 boots, 2 s phase-tagged VRAM sampler; artifacts .tasks/dense-q80-gemm/vram-headroom.md + vram-headroom-results.json). Serving defaults: MOE_MTILE=32 + DENSE_MTILE=64 + NSPLIT=2 (committed), winner flags, fp8 KV, hybrid, moe-cpu-threads 16, max-running-requests 1.

## Verified recipe for >=1.5-2 GiB free at peak
- WINNER: --memory-ratio 0.80 --kv-reserve-tokens 400000 --max-prefill-length 8191 -> 2008 MiB (1.96 GiB) free at peak, prefill 804.85 tok/s @8128 (-0.1% vs control 805.3), decode 14.68 (control 13.85, inside the 13.2-15.0 boot band - no real cost). Radix HIT 65536 at the 401k pool (65k contexts cache fine).
- RATIO FLOOR = 0.80: both 0.80/500k and 0.75/400k FAIL-FAST pre-serving at cache_budget.py:77 ("budget cannot fund the per-signature slot floors: 3 signature groups (39/1/2 layers; groups [[0..40],[8],[9,41]]) each need 288 slots"). The ratio floor holds only with the reserve cut to 400k.
- kv-reserve-tokens alone is NOT a headroom lever: 0.85/350k gave the SAME 508 MiB peak-free as 0.85/500k - the planner refilled the freed KV into +88 MoE slots (1148 -> 1236); peak tracks the ratio envelope only. (Refutes the paper advice "cut reserve to free VRAM".)
- Chunk lever: --max-prefill-length 4095 at 0.85/500k -> 2526 MiB (2.47 GiB) free for -25.9% prefill (597.06 vs 805.3; observed chunk size 4032 tokens). Use only when VRAM matters more than throughput.
- Measured VRAM-per-0.05-ratio = 1500 MiB (peak 32103 -> 30603 MiB) - the earlier paper estimate (1.3-1.4 GB) was ~10% low but directionally right.
- Decode peak-free ~= fill peak-free (2004 vs 2008 MiB at the winner config).
- Known benign side effect: slot-floor fail-fast kills leak sem.mp-* (observed 12 -> 18, /dev/shm only, no VRAM impact).
