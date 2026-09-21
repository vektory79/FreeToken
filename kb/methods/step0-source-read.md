# Step 0 - moe_q m-block re-read verification (source read, 2026-09-18)

READ-ONLY source analysis by research wave R1. Nothing was run. Anchors are
file:line in the v2 changeset (commit 7f8c570, branch vektory79). Status:
VERIFIED with a reconciliation caveat (kernel left the BW-bound regime).

Key files: `python/freetoken/kernel/csrc/gguf/moe.cuh` (moe_q @22-161; IQ twins
@1444-1716), `vecdotq.cuh` (iq tile glue), `moe_vec.cuh` (baseline),
`gguf_kernel.cu` (entry), `layers/moe.py:491-576`, `moe/fused.py:47-113`.
Constants: WARP_SIZE_GGUF=32, QK_K=256, QK8_1=32, QI8_1=8 (ggml-common.h:5,7,61-63);
block_iq3_xxs=98 B (ggml-common.h:139-142); block_iq4_xs=135 B (standard ggml
layout: 2 d + 4 scales_l + 1 scales_h + 128 qs).

## 1. Loop structure - hypothesis CONFIRMED

Block mapping: `row_dst_0 = blockIdx.x * mmq_y` (moe.cuh:43, 32 output rows),
`col_dst_0 = blockIdx.y * mmq_x` (moe.cuh:46, 4 pairs);
`exp_idx = expert_ids[blockIdx.y]` (moe.cuh:53);
`x = vx + exp_idx * exp_stride` (moe.cuh:62). One block = 32 rows x 4 pairs of
ONE expert.

k-slice loop: `for (ib0 = 0; ib0 < blocks_per_row_x; ib0 += blocks_per_warp)`
(moe.cuh:77); blocks_per_warp = 32/qi = 1 for both iq types (QI=QK_K/(4*QR)=32,
QR=2: vecdotq.cuh:2053-2054, 2177-2179). Per k-step
`load_tiles(x + row_x_0*blocks_per_row_x + ib0, ...)` (moe.cuh:78-88) loads
32 rows x 1 packed block = 32x98 B = 3,136 B (IQ3_XXS); loop runs K/256 times
(16 gate/up, 8 down) -> per-block weight bytes = 32 x K x 98/256 = 50,176 B
(gate) = the full 32-row k-span. tile_y_qs/ds staged from pre-quantized q8_1
(moe.cuh:96-118; quantized once per CALL over tokens, not per m-block -
gguf_kernel.cu:381-383).

Per-m-block weight bytes = ceil(R/32) row-blocks x 32 x (K/256) x B = the full
projection matrix S (gate IQ3_XXS: 2048x4096x98/256 = 3,211,264 B; down
IQ4_XS: 4096x2048x135/256 = 4,423,680 B). Per-expert traffic =
ceil(n_e/MOE_X) x S. VERIFIED.

## 2. Traffic model

- moe_vec (moe_vec.cuh:24-61, GGML_CUDA_MMV_Y=1 ggml-common.h:1030):
  grid=(nrows,1,pairs), one warp per (row, pair); each pair's row-warps
  collectively stream the full matrix -> pairs x S.
- Chunk numbers: pairs = 8128x8 = 65,024. moe_vec weights =
  65,024x(3.211+3.211+4.39 MB avg down) x42 = 29.53 TB/chunk -> 18.03 s at
  1638 GB/s vs measured 18.13 s (0.6% error; y fits L2: 8128x4.6 KB = 37 MB
  < 96 MB).
- grouped = Sum_e ceil(n_e/4) x S; at mean 226: 65,024/4 + 0.375x288 pad =
  16,364 -> 52.5+52.5+71.9 GB x42 = 7.43 TB/chunk -> 4.54 s at 1638 GB/s.
  Cut = 3.97x; per-expert re-read 16,364/288 = 56.8 = ~57. VERIFIED.
- Read-once floor: 42x288x10.56 MB = 128-134 GB -> ~0.08 s at VRAM BW; the
  1.5-3 s TASK.md figure is bounded in practice by the 3.15 s measured
  host-fetch copies (offload), not GEMM traffic.

## 3. Reconciliation (18.13 -> ~10 s vs 4x predicted)

Grouped measured ~10 s = 2.2x above its 4.5 s traffic floor -> the kernel left
the BW-bound regime; moe_vec hid per-MAC work under 18 s of streaming.
Source-grounded terms:

- (a) Non-scaling/fixed: indices ~24 B/block (moe.cuh:48-60) = 25 MB/proj,
  negligible; output writes 533 MB/proj IDENTICAL in both kernels (moe.cuh:158
  vs moe_vec.cuh:59-61); quantization per call (excluded); moe_align excluded
  per task. y-staging is DRAM-neutral (19 GB L2 traffic, <=37 MB DRAM) but
  adds LDS+sync cost.
- (b) SMEM/occupancy: ~4.9 KB smem/block (tile_x_ql 32x33 ints + tile_x_d
  33 floats, vecdotq.cuh:2182-2190; tile_y_qs 512 B + tile_y_ds 64 B,
  moe.cuh:72-73), 128 threads, NO __launch_bounds__ on CUDA (moe.cuh:1469-1472
  is ROCm-only) -> smem not limiting; registers are the only source-invisible
  unknown. Unlikely primary cause.
- (c) Padding: in-segment pad columns run the FULL schedule (weights+vec_dot)
  and are dropped only at the write guard (moe.cuh:148-150): ~0.75x288 = 216
  extra m-blocks = 1.3% of 16,364; pair-slot waste 432/65,024 = 0.66%.
  Grid-y overshoot blocks exit early (moe.cuh:57-60). ~1-2%, negligible.
- (d) L2: grid (64, ~16.5k) dispatches x-fastest; expert-sorted blockIdx.y ->
  co-resident CTAs are ~1,400-2,700 CTAs = consecutive m-blocks of the SAME
  expert; consecutive m-blocks read identical weight bytes -> L2 (96 MB holds
  ~25 gate matrices) serves them -> DRAM likely BELOW 7.43 TB. L2 pushes time
  down, cannot explain the shortfall.
- Best explanation: per-MAC decode work (vec_dot_iq3_xxs: 8 dp4a + ~20-30 ALU
  + LUT lookups + LDS per call, vecdotq.cuh:2250-2280) is IDENTICAL in both
  kernels (same MACs: 65,024xRxK), and k-step serialization (load_tiles ->
  syncthreads -> vec_dot -> syncthreads, no double-buffering, moe.cuh:77-142,
  64 syncs/block) plus unaligned 98-B loads assembled from byte reads
  (get_int_from_uint8, vecdotq.cuh:2200-2210, 4x LSU amplification) add
  per-block latency. Pure-issue arithmetic (~1.6e12 warp-instr/chunk ~ 1 s)
  under-explains 10 s, so throughput/latency stalls (LDS, LUT L1, barriers),
  not issue count, dominate the gap.

ncu cross-check (stated, not run): `dram__bytes.sum` (model check ~7.4
TB/chunk), `sm__warps_active.avg.pct_of_peak_sustained_active`,
`smsp__warp_issue_stalled_{short_scoreboard,long_scoreboard,barrier,mio_throttle,not_selected}_per_warp_active`,
`l1tex__data_pipe_lsu_wavefronts_mem_shared.sum`, `smsp__inst_executed.sum`.
mio_throttle/short_scoreboard/not_selected dominant -> ALU/LDS-bound (m-tile
sweep gains little); barrier/long_scoreboard dominant -> staging/latency-bound
(bigger tiles amortize); dram >> 7.4 TB -> model wrong.

## 4. Verdict

- "~57x per-expert re-read at MOE_X=4": VERIFIED - factor = ceil(n_e/MOE_X)
  per projection (57 at n_e=226).
- "~4x traffic cut vs moe_vec": VERIFIED as traffic model (3.97x analytic;
  moe_vec back-solve error 0.6%), PARTIALLY-VERIFIED as time explanation
  (1.8x measured; remainder = non-DRAM floor).
- Re-read factor F(MOE_X) = ceil(n_e/MOE_X): 4 -> 56.5 (57 w/pad), 8 -> 28.3
  (~28.5), 16 -> 14.1, 32 -> 7.1; chunk cut approx = MOE_x asymptotically
  (7.9x/15.8x/30.3x incl. padding). v3a caveat: the ds-load `token_offs[0]`
  fix (moe.cuh:113-118) is only valid while mmq_x/nwarps == 1; sum registers
  grow as (mmq_y/32)x(mmq_x/nwarps).

## Executive summary

1. moe_q's k-loop (moe.cuh:77) re-runs load_tiles per 4-pair m-block:
   per-expert weight re-read = ceil(n_e/4) = ~57 at mean n_e=226 - verified in
   source.
2. DRAM traffic model: 29.53 TB (moe_vec, back-solves to measured 18.13 s
   within 0.6%) -> 7.43 TB (grouped) = 3.97x cut -> 4.5 s BW floor.
3. Measured ~10 s = 2.2x above that floor: the kernel is no longer BW-bound;
   per-MAC decode ALU/LDS/LUT work (invariant across kernels) plus k-step
   serialization and byte-assembly loads bind.
4. Padding (~1-2%), indices, writes, and L2 are negligible or push the OTHER
   way; occupancy is structurally fine (4.9 KB smem, no CUDA launch bounds).
5. Verdict: re-read hypothesis VERIFIED; the single decisive ncu set is
   dram__bytes.sum + warp-stall breakdown + LDS wavefronts, which separates
   ALU/LDS-bound (m-tile sweep futile) from staging/latency-bound (m-tile
   sweep pays).
