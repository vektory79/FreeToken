---
name: "gguf-hybrid-decode-handshake-floor"
description: "gguf hybrid decode: per-layer handshake floor 1.9-3 ms x42 layers dominates; pool split starves CPU; needs batching"
type: project
lastUpdated: 2026-09-15T05:27
---

# gguf hybrid decode: per-layer handshake floor + pool-split starvation

Measured 2026-09-15 (hybrid campaign Task 05 hardware acceptance, RTX 5090, GLM-5.3-Flash-UD-Q3_K_XL.gguf, 42 moe layers, 3 partitions).

Fact: `--moe-strategy hybrid` as landed (commit eb7de4c) is functionally correct (boots 136 s, serves, battery 24/24, auto fetch fraction 78% active, VRAM delta noise) but decodes at 2.99 tok/s @64k vs 14.25 offload control - a 4.8x gap.

Root cause (bisect in .tasks/gguf-glm5next-hybrid/verification/bisect-notes.md):
1. Per-layer GPU<->CPU handshake floor ~1.9-3.0 ms x 42 layers = 80-127 ms/step - ALONE exceeds offload's whole ~70 ms step. Handshake count scales with layers, not misses.
2. The 3-pool thread split (resolve_pool_affinities) gives the dominant partition only 6/16 threads -> CPU leg ~2.96 GB/s effective vs 11.3 GB/s benched. The bandwidth-matched fraction model (B-2: matched f beats offload by (1+r)/r) assumed a full-bandwidth CPU leg and zero handshake cost - BOTH fail in practice.
3. Modeled best fix within the landed scope (weighted split + per-pool fractions) reaches only ~4-5 tok/s - insufficient.

**Why:** the B-2 plan-level math was correct for idealized legs but omitted the per-layer synchronization architecture; on a 42-layer model the fixed per-layer handshake dominates everything.

**How to apply:** any future hybrid/perf work on this model must attack the handshake first (batch/amortize GPU<->CPU syncs across layers), then weight the pool split by per-partition miss volume with per-pool fetch fractions, and only then add SIMD. Never model hybrid gains from CPU-leg bandwidth alone. benchbw's cached profile now contains the gguf entry (refreshed 2026-09-15).
