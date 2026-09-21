---
name: "ft-gguf-moe-mtile-kernel-traps"
description: "moe.cuh m-tile traps: x=weights y=activations naming, ds-fill garbage rows, block_size single source of truth"
type: project
lastUpdated: 2026-09-19T02:54
lastRecall: 2026-09-21T15:36
---

# gguf moe.cuh m-tile modification traps (moe_q kernel structure)

Lessons from the v3a wave (verified by Review-A against the vendored source; lineage vLLM PR #36226 + llama.cpp 4696d5674). File: python/freetoken/kernel/csrc/gguf/moe.cuh.

- NAMING TRAP: in moe.cuh x = WEIGHTS (vx indexed by exp_idx*exp_stride, moe.cuh:62), y = ACTIVATIONS (q8_1 tokens). The x-tile scales with mmq_y = MOE_Y (32, m-invariant); only the y-tile scales with mmq_x = MOE_X (tile_y_qs[mmq_x*32], tile_y_ds[mmq_x*4] = 144*m B). Briefs saying "tile_x scales with the m-tile" are half wrong - cost a review cycle to catch.
- DS-FILL GATE (silent corruption class): moe_q's y-ds fill had NO i-loop - it filled only ds rows [0, nwarps) and assumed mmq_x/nwarps == 1 via token_offs[0]. Any m-tile > nwarps leaves ds rows >= nwarps as UNINITIALIZED GARBAGE that passes shape checks and shows only as parity noise. Correct port: full-coverage loop reading sorted_token_ids[col_dst_0+ids] DIRECTLY (token_offs is per-thread and cannot index cross-warp columns); do NOT copy the dense mul_mat_q min-clamp (pad slots rely on skip-guards: sentinel ids >= pairs skip the write; garbage tile rows feed only sums discarded by the col_dst >= ncols_dst write guard).
- SINGLE SOURCE OF TRUTH: the moe_align trio block_size (ggml_moe_get_block_size, consumed by python layers/moe.py trio/cap) and the kernel's mmq_x (ggml_moe_a8 dispatch) MUST derive from one selection; divergence silently consumes bins cross-expert / drops tail pairs - no error, wrong numbers.
- token_offs[mmq_x/nwarps] indexing: on ROCm nwarps=8, so tile 4 gives a zero-length array. This is why the CUDA-only knob branch (FREETOKEN_GGUF_MOE_MTILE) keeps ROCm at its MOE_X_<T> (all 12 ROCm defines were 8 as of 2026-09-18; nothing pins them equal - a future per-type ROCm tile needs static_asserts or per-type returns).
- Structure facts: all tiles are static __shared__ (48 KB limit never approached, no cudaFuncSetAttribute in tree); NO __launch_bounds__ on CUDA (ROCm-only at moe.cuh:173-175); q6_K is the SMEM-occupancy binding format (10/10/9/7 blocks/SM at m=4/8/16/32 on sm_120, 128 threads); tail padding waste ~ (bs-1)/2 rows per expert (0.7/1.6/3.5/7.1% of pair slots at 4/8/16/32) - compute waste only, never correctness.
- JIT/cache: torch cpp_extension load keys on source hash -> header edits rebuild once (~3 min, clang++ host); freetoken-kernel-cache wheel has ZERO gguf coverage - no stale-prebuilt hazard for gguf kernel changes.
