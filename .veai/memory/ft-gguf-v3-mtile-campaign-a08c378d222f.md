---
name: "ft-gguf-v3-mtile-campaign"
description: "v3a m-tile sweep: tile32 +67.2% = 556 tok/s @8128; ALU/LDS-bound; v3b no-go; committed 1706aae"
type: project
lastUpdated: 2026-09-19T13:52
lastRecall: 2026-09-19T04:15
---

# v3 m-tile campaign (grouped GGUF MoE prefill): Step 0 verified, v3a implemented

Campaign dir: `.tasks/mmq-v3-stationary-moe/` (TASK.md + step0-source-read.md + v3a-surface-map.md are self-contained artifacts with file:line anchors). Follow-up of v2 grouped MMQ (7f8c570); branch vektory79. Private-use scope: no push, commit only on user request.

## Step 0 VERIFIED (2026-09-18, source-read wave)
- Re-read hypothesis VERIFIED: per-expert weight re-read factor = ceil(n_e/MOE_X) (57 at MOE_X=4, n_e~226). moe_q k-loop (moe.cuh:77) re-runs load_tiles per 4-pair m-block; per-m-block weight bytes = the full projection matrix S.
- Traffic model VERIFIED: moe_vec 29.53 TB/chunk (back-solves the measured 18.13 s within 0.6%); grouped 7.43 TB/chunk = 3.97x cut = 4.54 s DRAM floor at 1638 GB/s.
- KEY RECONCILIATION: measured grouped MoE ~10 s = 2.2x ABOVE that floor -> the kernel LEFT the BW-bound regime. Dominant terms: per-MAC decode work (vec_dot ALU/LDS/LUT, invariant across schedules), k-step serialization (64 syncs/block, no double-buffering), unaligned 98-B byte-assembly loads (4x LSU amplification). L2 (~96 MB) serves co-resident same-expert m-blocks (pushes time DOWN); padding ~1-2%; SMEM 4.9 KB/block not limiting.
- Consequence: traffic-only optimizations (m-tile widening, v3b weight-stationary) are bounded by the non-DRAM floor; the ~758 tok/s roofline is NOT reachable by traffic reduction alone.
- Decisive ncu set if the sweep saturates unexplained: dram__bytes.sum + warp-stall breakdown (mio_throttle/short_scoreboard dominant = ALU/LDS-bound -> sweep futile; barrier/long_scoreboard dominant = staging-bound -> sweep pays) + l1tex LDS wavefronts.

## v3a IMPLEMENTED (2026-09-18, uncommitted on vektory79, Review-A SHIP)
- moe.cuh: full-coverage y-ds fill ported into moe_q (reads sorted_token_ids[col_dst_0+ids] directly; token_offs is per-thread, cannot index cross-warp columns); moe_q8_0/moe_q6_K/moe_iq3_xxs/moe_iq4_xs + launchers templated on int mmq_x, tiles {4,8,16,32}.
- gguf_kernel.cu: ft_gguf_moe_mtile() single source of truth; env FREETOKEN_GGUF_MOE_MTILE read per call (unset/"4" -> 4; "8"/"16"/"32" -> tile; else TORCH_CHECK fail-fast). Knob is GLOBAL (per-format isolation in the sweep comes from nsys per-kernel timing; kernel names unchanged). On ROCm the knob returns MOE_X_Q8_0 (=8, single tile kept: tile 4 would give zero-length token_offs at nwarps 8 - that hazard did NOT pre-exist in v2, CUDA 4/4 + ROCm 8/8 both give token_offs[1]).
- Tests: touched files 84/84 green; full gate 2383 passed / 5 pre-existing baseline (pinned_tensor UVA, qsa_fp8 [1-1-64], qwen4_exp/test_ple :421, 2x glm_dsa OutOfResources 102400>101376). matches_moe_vec seeded (torch.manual_seed(7)) + iq tolerance 2e-2/1e-2 - JUSTIFIED by 20-seed scan: old unseeded 1e-3 was latently flaky (IQ4_XS fails 1/20 seeds, margin +2.0e-4); ds-bug class shifts whole pairs O(1) = 100x above the band, masks nothing. Cap test uses torch.zeros not empty (uninit bf16 ~1e38 overflows the q8_1 fp16 scale -> inf). Analytic ds-fill regression added (Q8_0 lossless integer fixture, all 32 ds rows live, tiles 4/8/16/32, rtol=0/atol=1e-3).
- m=4 tile-invariance confirmed empirically (max|a-b| identical across tiles per format; tile-invariant reduction order) -> v2 numbers (289.92/325.28/337.61 tok/s @4096/6144/8128) and battery PASS remain the valid baseline.
- Review-A open items (non-blocking): static_assert pinning ROCm MOE_X_* equality under USE_ROCM; docstring "do not flip env after boot" (python trio block_size and kernel getenv read at different moments; a mid-call flip would mispair trio bs N with kernel mmq_x M); analytic ds test covers only need_sum=false (Q8_0); capture -Xptxas -v spill counts per tile during the sweep; NIT "15-arg" comment -> 16.

## Pending: hardware A/B sweep (tiles 4/8/16/32)
Winner flags, port 18801: --moe-cache-auto --kv-reserve-tokens 500000 --kv-cache-dtype fp8 --memory-ratio 0.85 --max-prefill-length 8191 --moe-strategy hybrid --moe-cpu-threads 16 --max-running-requests 1. One boot per tile (env per boot; unset = baseline tile 4), serialized + reaped runs; prefill = median of full chunks excluding the last-full-chunk artifact and c1 warmup; nsys liveness (launch count 126/chunk stays, CTAs/launch shrink ~tile x; re-read factor 57/28.5/14.1/7.1; ZERO moe_vec_q in prefill); 24-prompt quality battery from .tasks/mmq-prefill-kernel/quality/. Watch: q6_K occupancy 10/10/9/7 blocks/SM, register pressure at 32 (no __launch_bounds__). Commit subject ready: feat(kernels): selectable grouped gguf moe m-tile (8/16/32) via FREETOKEN_GGUF_MOE_MTILE.

## Hardware sweep DONE (2026-09-18/19) - WINNER TILE 32
- Validity gate (T44) PASS: tile-4 boot 332.72 tok/s @8128 = -1.45% vs committed 337.61.
- A/B: tile8 436.11 (+31.1%) -> tile16 512.48 (+54.0%) -> tile32 556.43 (+67.2%; +64.8% vs committed). Decode flat 14.07-14.74; VRAM flat ~32.1 GB; all radix re-sends HIT 65536.
- Saturation: marginal gain halves per doubling; MoE-term ratio vs pure 1/tile = 0.500/0.260/0.152; nsys grouped MoE 12.62 -> 5.01 s/chunk at tile16 = 2.52x for a 15.8x traffic cut; per-launch medians iq4_xs 2.64x / iq3_xxs 2.50x / q6_K 1.95x -> ALU/LDS/serialization-bound, NOT SMEM/occupancy-bound; padding does not dominate. Practical saturation point = tile 32 (tile 64 ~ +3-4% at best, padding ~14%, q6_K occupancy 10/10/9/7).
- v3b DOES NOT FIRE: traffic reduction beyond tile ~32 cannot pay under the ALU/LDS floor; chunk now ~80% "rest" -> next levers dense q8_0 GEMM 6.04 s + fetch copies 3.15 s.
- Liveness (T46/D09): 126 grouped + 42 moe_align per 8128 chunk, zero moe_vec_q in prefill, 7938 moe_vec in decode graphs, CTAs/launch shrink 3.8x -> knob live.
- Quality battery: 24/24 outputs BITWISE IDENTICAL tile4 vs tile32 (per-element reduction order tile-invariant), 4/4 digit prompts, 0 flips.
- Review fix-TP applied (6 items): ROCm static_assert pinning MOE_X_Q8_0==Q6_K==IQ3_XXS==IQ4_XS; "16-arg" comment; docstring per-call + do-not-flip-after-boot; other-types-unaffected assert for types 3,6,7,10-13; invalid-env stragglers " 8"/"04"/"+8"; iq tolerance tightened to atol 4e-3/rtol 2e-3 (3x+ headroom, seed 7); analytic tail-expert test at tiles 16/32 (tokens=3, topk_ids [[0],[0],[1]], npp==2*mtile). Touched files 119/119 green post-fix.
- Artifacts: v3a-ab.md, v3a-ab-results.json, v3aprobe.sqlite, v3a_probe_liveness.json, quality/base + quality/win (all under .tasks/mmq-v3-stationary-moe/). Commit remains user-gated. Deferred: Q6_K need_sum=true analytic fixture; ptxas -v spill capture.

## COMMITTED (2026-09-19, user request)
1706aae "feat(kernels): selectable grouped gguf moe m-tile (8/16/32) via FREETOKEN_GGUF_MOE_MTILE" on vektory79 - exactly the 5 files, +318/-147, staged by explicit path, zero .veai entries staged, 119/119 touched tests re-run green immediately pre-commit, measured body in the commit message, NOT pushed. Knob default stays 4 (v2 behavior); flipping the production default to the winner tile 32 is an OPEN user decision. Next lever brief composed: .tasks/dense-q80-gemm/TASK.md (dense q8_0 GEMM, mul_mat_q8_0 6.04 s = ~41% of the 14.61 s chunk at tile 32; CUDA dense tile is 4x32/4warps vs ROCm 64x128/8warps - the dense path has the same 4-token-tile disease; anchor config for that campaign = MTILE=32 fixed).
