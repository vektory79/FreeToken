---
name: "ft-gguf-prefill-mmq-roofline"
description: "GGUF prefill MMQ roofline analysis: step-0 nsys split, v0 no-op, v2 grouped +16.4%, ~758 ceiling unreachable (ALU-bound)"
type: project
lastUpdated: 2026-09-19T18:18
lastRecall: 2026-09-20T14:22
---

# GGUF prefill MMQ roofline + campaign history (RTX 5090, 2026-09-17)

Why GGUF glm5next prefill was slow and what the v2/v3 kernel waves did about it. This file is the ANALYSIS record; campaign state lives in ft-gguf-v2-grouped-mmq-measured (v2 outcome/commit 7f8c570), ft-gguf-v3-mtile-campaign (m-tile, commit 1706aae), ft-dense-q80-gemm-campaign (next lever). Artifacts: .tasks/ft-gguf-serve-tuning/ and .tasks/mmq-prefill-kernel/ (TASK.md, step0-profile.md, v0-ab.md, research-r1-moe-align.md, research-r2-upstream-tile-glue.md).

## Geometry (real GGUF header, gguf-py)
H=4096, I=2048, E=288, top_k=8, 46 blocks, leading_dense=3 -> 42 routed MoE layers (+1 shared dense expert). Types: 39x (IQ3_XXS gate, IQ3_XXS up, IQ4_XS down), 2x (IQ3_XXS, IQ3_XXS, Q6_K), 1x (IQ4_XS, IQ4_XS, Q6_K). 10.88 MB/expert (g1), 134.4 GB expert total.

## Why prefill was slow
moe_vec.cuh: grid = (row_blocks, 1, tokens*top_k); each block = ONE output row of ONE (token,expert) pair and re-reads that row's packed blocks -> a pair streams the FULL expert weights. Traffic = 65,024 pairs x 10.88 MB = 707 GB/layer = 29.7 TB/chunk = 16.6 s at 1.79 TB/s. MoE GEMM ~100% VRAM-BW-bound, 226x redundant weight reads. Activation q8_1-quantized once per token (good).

## MEASURED split (Step 0, nsys 2026.1.3; step0-profile.md, step0_split.json, report1.nsys-rep)
Steady 8128-chunk T = 28.77 s median (282-283 tok/s this run, ~3% run variance vs campaign 291-293). moe_vec_q 18.13 s = 63.0% (126 launches/chunk = 42 layers x 3 projections; 1638 GB/s = ~92% of roofline -> BW-bound confirmed). fast_index_copy_multi 3.15 s = 10.9% (42/chunk; PCIe-class COPY KERNEL, not H2D memcpy - those measure ~0). Rest 7.49 s = 26.0%, dominated by dense q8_0 GEMM mul_mat_q8_0 6.04 s (21.0%); attention/GDN/DSA/router/quantize ~1.45 s; GPU idle 0.015 s/chunk.

## v0 expert-sort: MEASURED 2026-09-17: REFUTED - throughput no-op
Same-session A/B (pathspec-stash baseline, winner flags, probe-confirmed sorted path active): prefill 289.54 -> 291.26 tok/s (+0.6%, noise), decode unchanged. WHY the L2-reuse model fails: thousands of CTAs are resident at once, so consecutive-in-z pairs of the same expert execute CONCURRENTLY, not sequentially - ordering cannot cut concurrent traffic, only total-traffic reduction can. Lesson: do not chase cache-locality reorderings for this kernel. User dropped the 6-file v0 set (reverted to HEAD 1a444e4).

## Ceilings (corrected - see Current chain)
v2 grouped MMQ was implemented and measured +16.4% @8128 (289.92 -> 337.61 tok/s), but the projected ~758 tok/s ceiling (MoE 1.5-3 s) is NOT reachable by traffic cuts: measured grouped MoE stayed ~2.2x above its 4.54 s DRAM floor (kernel left the BW-bound regime). Untouched-by-v0 logic holds: only TOTAL-traffic reduction helps. Levers after v2/v3: dense q8_0 GEMM 6.04 s (TOP), fetch copies 3.15 s, attention ~1.45 s; NVFP4 reference ~507 tok/s steady prefill on 4096-token chunks. Prefill overlap NOT a knob (needs 576 slots/partition; g1=410 at 0.85). v1 "dequant segment to bf16 + cuBLAS" KILLED: 14.5 GB bf16/layer -> +610 GB/chunk, as slow as today - dequant must be fused into GEMM tiles.

## v2 implementation: durable facts (outcome, A/B, battery, commit 7f8c570 = ft-gguf-v2-grouped-mmq-measured)
- Upstream archaeology: vendored tree = vLLM 4492e3a5 (engine = llama.cpp PRE-#8495 plugin interface; gguf_kernel.cu = 755ed7b0; dispatch.h = sgl-kernel); iq glue never existed pre-#8495 (entered llama.cpp mmq.cuh 69c487f4e 2024-07-20, moved 6eddde06a 2026-07-13); vLLM/sgl-kernel main csrc gguf dirs are REMOVED (404). llama.cpp 4696d5674 available in the local clone /media/ai/src/llama.cpp.
- VENDOR BLUEPRINT = vLLM PR #36226 (dense-only, never touched moe.cuh - the moe iq twins are honestly labeled FreeToken-side), NOT the R2 llama.cpp-shim design (superseded): RAW block bytes in tile_x_ql + decode inside vec_dot, because the vendored 32-int tile row cannot hold the pre-decoded form. VDR=4 + need_sum=true are REQUIRED by that layout (need_sum=false overwrites the half2 {d,sum}; VDR=2 maps k and k+2 to the same ib32 = double-count); R2's VDR 2/4 belong to the abandoned post-#8495 pre-decoded layout. Review-A verified the quant math line-equivalent to the PR and arithmetically identical to the in-repo MMVQ dots; the global kvalues_iq4nl LUT is reused.
- Two load-bearing fixes beyond the vendor (details here; v2 file summarizes): (1) moe.cuh expert guard `exp_idx >= num_experts` (W.size(0) threaded through moe_q + 10 wrappers; signature order (top_k, num_experts) - an initial swap was caught by kernel printf) replacing the vendored `>255` cap that silently dropped experts 256-287 at E=288 and closed a latent OOB W-read; moe_vec has NO such cap (reads topk_ids per pair) - no production hazard there. (2) moe_q ds-load `token_offs[threadIdx.y]` -> `[0]`: real OOB read of a size-1 array, an upstream vLLM latent bug still present in vLLM v0.9.2; correct because MOE_X/nwarps==1 on CUDA (4/4) and ROCm (8/8).
- Python glue: prefill-only `_gguf_grouped_prefill` via the EXISTING producer freetoken.moe.fused.moe_align_block_size (moe/fused.py:47; sgl_kernel + triton backends; sorted_token_ids sentinel = topk_ids.numels(), NOT -1; ONE trio per layer serves gate/up (top_k=8) AND down (top_k=1, tokens=M*top_k) because moe.cuh divides the flat pair index by its top_k arg; block size MOE_X=4 for types 18/23 via ggml_moe_get_block_size). layers/gguf.py: _MMQ now includes IQ3_XXS/IQ4_XS, _IQ_ONLY emptied (dense batch >6 routes to ggml_mul_mat_a8; K%256==0 structurally guaranteed by block-quant formats, no alignment guard needed). Decode untouched (byte-identical, bomb-test pinned).
- Attribution headers: "Vendored from vllm-project/vllm PR #36226 (Apache-2.0, (c) The vLLM authors)" + llama.cpp mmq.cuh @ 4696d5674 (MIT, (c) The ggml authors) lineage + adaptation note.

## Current chain (2026-09-19)
v2 grouped MMQ COMMITTED 7f8c570 on vektory79: +16.4% @8128, decode unchanged, battery 24/24, kill switch FREETOKEN_GGUF_GROUPED_PREFILL removed (grouped unconditional for supported types). v0 no-op reverted. v3 m-tile sweep COMMITTED 1706aae: tile 32 winner = 556.43 tok/s @8128 (+64.8% vs v2), grouped MoE direct-measured 4.37 s/chunk at tile 32 (the "~1.8 s e2e fit" and "~80% rest" estimates REFUTED; non-MoE share 70.6%). Next levers: dense q8_0 GEMM (ft-dense-q80-gemm-campaign) + fetch copies 3.15 s. Regime discriminators without ncu (permission-blocked on this box, see cuda-debug-tool-strategy): mio_throttle/short_scoreboard = ALU-bound, barrier/long_scoreboard = staging-bound.
