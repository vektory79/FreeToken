---
name: "ft-gguf-serving-vram-headroom"
description: "GGUF glm5next VRAM headroom: 0.80/400k winner needs >=29.28 GiB free at boot; flips with desktop VRAM swing; options"
type: project
lastUpdated: 2026-09-20T20:41
lastRecall: 2026-09-26T18:28
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

## Flip condition found (2026-09-20, repro .tasks/ft-serve-crash-mr080-400k/)
The winner recipe sits AT the slot floor: it needs >= ~29.28 GiB free VRAM at boot (the 09-20 run had only +39 MiB margin: planned 1062 slots vs floor 1059). A desktop VRAM swing flips it: +357 MiB desktop usage -> planned 1035 < 1059 -> fail-fast ValueError at cache_budget.py:77 BEFORE any load (~9 s boot). Not a merge regression (gate untouched since 315b278; fp8 vs nvfp4 KV irrelevant - no dtype branch in the crash path; pre-merge the late check_partition_floors net would reject the same plan after the multi-minute load). Options computed from the closed-form plan model (fitted once on the archived 09-20 waves, reproduces every recorded plan exactly): 0.81 -> 1062-1064 (razor-thin); reserve 350000 @0.80 -> 1063-1065 (~zero peak-VRAM cost, 351k pool still radix-hits 65k contexts); robust 0.82 (or 0.81 + 350k) -> 1091-1093, tolerates ~400-500 MiB desktop swing at ~300 MiB peak-free cost per 0.01 ratio (measured 1500 MiB per 0.05).
