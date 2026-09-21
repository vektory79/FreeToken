# R2: upstream source for the iq MMQ tile glue (v2 step 1) (2026-09-17)

Read-only research result (HEAD 1a444e4; local llama.cpp clone verified). Input for TASK.md v2 steps 1-2.

## 1. Vendored vintage/provenance (verified headers)
- vecdotq.cuh:1-4, mmq.cuh:1-3, moe.cuh:1-3, ggml-common.h:1-3 = vLLM commit 4492e3a55428e161ca8db381edc28263e5da4c8d (csrc/quantization/gguf/*), lineage comment llama.cpp b2899; gguf_kernel.cu:1-2 = vLLM 755ed7b05be4743237d3339c4ff8c22bcaae04f4; dispatch.h:1-3 = sgl-kernel gguf (AT_DISPATCH + SGLANG_SHFL). The b2899 comment is stale for iq parts (b2899 predates iq3_xxs). Engine = llama.cpp PRE-#8495 ("dedup") plugin interface.

## 2. vec_dot signatures to pair with (vendored vecdotq.cuh, quoted)
`static __device__ __forceinline__ float vec_dot_iq3_xxs_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1, const int& iqs)` (:1848) and same shape for iq4_xs (:2017). block_q8_1 is llama.cpp-style `{int8_t qs[32]; half2 ds={d,sum}}` (ggml-common.h; quantize_q8_1 in gguf_kernel.cu:38-60 sets ds.x=d, ds.y=sum); iq dots use only ds.x -> run tiles with need_sum=false (engine then stores float scale in tile_y_ds; matches upstream D4 layout). Engine plugin typedefs: ggml-common.h:936-948 (allocate_tiles_cuda_t, load_tiles_cuda_t(9 args), vec_dot_q_mul_mat_cuda_t).

## 3. Chosen upstream source + why
No upstream vintage has iq glue in the vendored pre-dedup interface (pickaxe over complete local clone /media/ai/src/llama.cpp history: `allocate_tiles_iq` never existed; iq glue entered mmq.cuh at 69c487f4e 2024-07-20 #8495, moved at 6eddde06a 2026-07-13 #24127; vLLM main and sgl-kernel main csrc/quantization/gguf are 404 - removed). Chosen: llama.cpp **4696d5674 (2025-05-14, MoE-crash-fix era, post-#8495 pre-#24127)**, ggml/src/ggml-cuda/mmq.cuh (3217 lines) + vecdotq.cuh; local clone reachable at that exact hash; reference copies saved: /tmp/mmq_4696d5674.cuh, /tmp/mmq_cu_4696d5674.cu (if /tmp was cleaned, regenerate via `git -C /media/ai/src/llama.cpp show 4696d5674:ggml/src/ggml-cuda/mmq.cuh`). Current master (mmq-load-tiles.cuh, config tables) rejected: larger engine mismatch.

## 4. Inventory to vendor (from 4696d5674)
- load_tiles_iq3_xxs<mmq_y,nwarps,need_check> mmq.cuh:2073-2134 (62 lines; take DP4A `#else` branch) - decodes iq3xxs_grid+ksigns64 via __vsub4 into x_qs (stride i*(2*WARP_SIZE+1)), folds scale ((ls*d+d/2)/2) into x_df.
- load_tiles_iq4_xs 2248-2306 (59 lines) - kvalues_iq4nl table decode + d*(ls-32) fold.
- vec_dot dot glue is GENERIC upstream: vec_dot_q8_0_q8_1_dp4a mmq.cuh:624-659 + vec_dot_q8_0_q8_1_impl; per-type vec_dot_*_mul_mat for the vendored contract is a ~15-line adaptation (analogous to vendored vec_dot_q8_0_q8_1_mul_mat).
- txs macro MMQ_DP4A_TXS_Q8_0 + mmq_get_dp4a_tile_x_sizes IQ cases (mmq.cuh:200-240); traits IQ3_XXS/IQ4_XS :2480-2521; MMQ_ITER_K=256 :13.
- allocate_tiles shim ~12 lines/type (carve one __shared__ int array; *x_ql=tile, *x_dm=(half2*)(tile+txs.qs)); loader adapter maps vx/k/i_offset/blocks_per_row -> x/x_tile/kbx0=0/i_max/stride (~20 lines/type). Total ~150-180 lines/type incl. launchers, matching the task estimate.

## 5. Missing pieces (all located)
VDR_IQ3_XXS_Q8_1_MMQ=2 (upstream vecdotq.cuh:1153) and VDR_IQ4_XS_Q8_1_MMQ=4 (:1338) - absent in vendored vecdotq.cuh; add. MMQ_X/Y/NWARPS_IQ3_XXS/IQ4_XS + MOE_* twins - absent; mirror the vendored 4/32/4 CUDA convention (decision point). LUTs/structs ALREADY vendored: iq3xxs_grid ggml-common.h:563, ksigns64 :897, kvalues_iq4nl :927, block_iq3_xxs :140, block_iq4_xs :192, QI/QR3_XXS :135-136, QI/QR4_XS :185-186. kvalues_iq3xxs NOT needed. Switch sites: ggml_mul_mat_a8 (gguf_kernel.cu ~178-465, cases 2..14) and ggml_moe_a8 (~335-547) need cases 18/23; the MMVQ switch already has them.

## 6. Attribution header
Upstream files carry no in-file license block (mmq.cuh starts `#pragma once`); llama.cpp is repo-level MIT, vLLM Apache-2.0; repo vendored files have only URL comments (known gap). For new vendored code: `// Vendored from llama.cpp ggml/src/ggml-cuda/mmq.cuh @ 4696d5674 (MIT, (c) The ggml authors)` + URL + `// adapted to the vendored vLLM-style tile plugin interface`. Retrofit note: existing files should additionally name vLLM 4492e3a5/755ed7b0 (Apache-2.0).

## Correction to the roofline memory claim
"Iq tile glue exists in upstream mmq.cu since 2024, vendorable verbatim" is FALSE for the vendored pre-dedup engine: it exists only in the post-#8495 interface (69c487f4e 2024-07-20); verbatim vendoring requires the ~30-line/type interface shim described above.
