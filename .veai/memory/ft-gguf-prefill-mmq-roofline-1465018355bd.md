---
name: "ft-gguf-prefill-mmq-roofline"
description: "GGUF prefill MMQ roofline: v0 no-op, v2 grouped +16.4%, 758-tok/s ceiling corrected (kernel left BW-bound regime)"
type: project
lastUpdated: 2026-09-19T02:54
lastRecall: 2026-09-19T03:10
---

# GGUF prefill MMQ roofline + campaign state (RTX 5090, 2026-09-17)

Why GGUF glm5next prefill is slow and what the v2 grouped MMQ prefill does about it. Artifacts context: .tasks/ft-gguf-serve-tuning/ and .tasks/mmq-prefill-kernel/ (TASK.md, step0-profile.md, v0-ab.md, research-r1-moe-align.md, research-r2-upstream-tile-glue.md).

## Geometry (real GGUF header, gguf-py)
H=4096, I=2048, E=288, top_k=8, 46 blocks, leading_dense=3 -> 42 routed MoE layers (+1 shared dense expert). Types: 39x (IQ3_XXS gate, IQ3_XXS up, IQ4_XS down), 2x (IQ3_XXS, IQ3_XXS, Q6_K), 1x (IQ4_XS, IQ4_XS, Q6_K). 10.88 MB/expert (g1), 134.4 GB expert total.

## Why prefill was slow
moe_vec.cuh: grid = (row_blocks, 1, tokens*top_k); each block = ONE output row of ONE (token,expert) pair and re-reads that row's packed blocks -> a pair streams the FULL expert weights. Traffic = 65,024 pairs x 10.88 MB = 707 GB/layer = 29.7 TB/chunk = 16.6 s at 1.79 TB/s. MoE GEMM ~100% VRAM-BW-bound, 226x redundant weight reads. Activation q8_1-quantized once per token (good).

## MEASURED split (Step 0, nsys 2026.1.3; step0-profile.md, step0_split.json, report1.nsys-rep)
Steady 8128-chunk T = 28.77 s median (282-283 tok/s this run, ~3% run variance vs campaign 291-293). moe_vec_q 18.13 s = 63.0% (126 launches/chunk = 42 layers x 3 projections; 1638 GB/s = ~92% of roofline -> BW-bound confirmed). fast_index_copy_multi 3.15 s = 10.9% (42/chunk; PCIe-class COPY KERNEL, not H2D memcpy - those measure ~0). Rest 7.49 s = 26.0%, dominated by dense q8_0 GEMM mul_mat_q8_0 6.04 s (21.0%); attention/GDN/DSA/router/quantize ~1.45 s; GPU idle 0.015 s/chunk.

## v0 expert-sort: MEASURED 2026-09-17: REFUTED - throughput no-op
Same-session A/B (pathspec-stash baseline, winner flags, probe-confirmed sorted path active): prefill 289.54 -> 291.26 tok/s (+0.6%, noise), decode unchanged. WHY the L2-reuse model fails: thousands of CTAs are resident at once, so consecutive-in-z pairs of the same expert execute CONCURRENTLY, not sequentially - ordering cannot cut concurrent traffic, only total-traffic reduction can. Lesson: do not chase cache-locality reorderings for this kernel. User dropped the 6-file v0 set (reverted to HEAD 1a444e4).

## Ceilings
v2 grouped MMQ: MoE 1.5-3 s -> chunk ~10.7 s -> ~758 tok/s (2.68x). Untouched by the v0 refutation: v2 reduces TOTAL traffic (weights read once per expert per layer). After v2 bottlenecks: dense q8_0 GEMM 6.04 s (TOP), copies 3.15 s, attention ~1.45 s; NVFP4 reference ~507 tok/s steady prefill on 4096-token chunks. Prefill overlap NOT a knob (needs 576 slots/partition; g1=410 at 0.85). v1 "dequant segment to bf16 + cuBLAS" KILLED: 14.5 GB bf16/layer -> +610 GB/chunk, as slow as today - dequant must be fused into GEMM tiles.

## v2 grouped MMQ: IMPLEMENTED 2026-09-17 (uncommitted on vektory79 @ 1a444e4; fix wave + hardware A/B pending)
8 files +1177/-57; 119 tests green across 5 waves incl. IQ grouped parity vs gguf-py (no drift; worst ~1e-2 rel = documented x-q8_1 noise), E=288 with experts >255 pinned, q8_0/q6_k grouped-vs-moe_vec identical-input consistency, trio contract, dense iq lift, dispatch pin, layer-level prefill==decode==analytic. Tail experts (n_e 1-2): measure in A/B, do not special-case.

- VENDOR BLUEPRINT = vLLM PR #36226, NOT the R2 llama.cpp-shim design (superseded): RAW block bytes in tile_x_ql + decode inside vec_dot, because the vendored 32-int tile row cannot hold the pre-decoded form. VDR=4 + need_sum=true are REQUIRED by that layout: need_sum=false overwrites the half2 {d,sum} with a float and corrupts the .x read; VDR=2 maps k and k+2 to the same ib32 (double-count). R2's VDR 2/4 belong to the abandoned post-#8495 pre-decoded layout. Review-A verified the quant math line-equivalent to the PR and arithmetically identical to the hardware-validated in-repo MMVQ dots; the global kvalues_iq4nl LUT is reused (improvement over the PR's stack table).
- Upstream archaeology: vendored tree = vLLM 4492e3a5 (engine = llama.cpp PRE-#8495 plugin interface; gguf_kernel.cu = 755ed7b0; dispatch.h = sgl-kernel); iq glue never existed pre-#8495 (entered llama.cpp mmq.cuh 69c487f4e 2024-07-20, moved 6eddde06a 2026-07-13); vLLM/sgl-kernel main csrc gguf dirs are REMOVED (404). llama.cpp 4696d5674 available in the local clone /media/ai/src/llama.cpp.
- Attribution headers: "Vendored from vllm-project/vllm PR #36226 (Apache-2.0, (c) The vLLM authors)" + llama.cpp mmq.cuh @ 4696d5674 (MIT, (c) The ggml authors) lineage + adaptation note. PR #36226 is dense-only (never touched moe.cuh) - the moe iq twins are honestly labeled FreeToken-side.
- Two load-bearing fixes beyond the vendor: (1) moe.cuh expert guard `exp_idx >= num_experts` (num_experts = W.size(0) threaded through moe_q + 10 wrappers/launchers; signature order (top_k, num_experts) - an initial swap was caught by kernel printf) replacing the vendored `>255` cap that silently dropped experts 256-287 at E=288 (Y = torch.empty -> garbage rows) and closed a latent OOB W-read; moe_vec has NO such cap (reads topk_ids per pair directly) - no production hazard there. (2) moe_q ds-load `token_offs[threadIdx.y]` -> `token_offs[0]`: real OOB read of a size-1 array, an upstream vLLM latent bug still present in vLLM v0.9.2; correct because MOE_X/nwarps==1 on CUDA (4/4) and ROCm (8/8).
- Python: layers/moe.py is_prefill-only `_gguf_grouped_prefill` via the EXISTING producer `freetoken.moe.fused.moe_align_block_size` (moe/fused.py:47; sgl_kernel + triton backends; sorted_token_ids sentinel = topk_ids.numel(), NOT -1; one trio per layer serves gate/up (top_k=8) AND down (top_k=1, tokens=M*top_k) because moe.cuh divides the flat pair index by its top_k arg; block size MOE_X=4 for types 18/23 via ggml_moe_get_block_size). layers/gguf.py: _MMQ now includes IQ3_XXS/IQ4_XS, _IQ_ONLY emptied (dense batch >6 routes to ggml_mul_mat_a8; K%256==0 structurally guaranteed by block-quant formats, no alignment guard needed). Decode untouched (byte-identical, bomb-test pinned).
- Fix wave in flight at session end (reviews: Review-A SHIP, Review-B FIX-FIRST, no blockers): env kill switch `FREETOKEN_GGUF_GROUPED_PREFILL` (default on, read per call) falling back to the moe_vec prefill branch (also the unsupported-type fallback when ggml_moe_get_block_size <= 0, e.g. IQ2_XS/IQ4_NL); triton `_moe_align_small` tail sentinel fill (sentinel only [0, npp) = latent silent-corruption on non-sgl rigs) + moe_q boundary `>` -> `>=` belt; base-file attribution retrofit; NWARPS_IQ* dedup between mmq.cuh/moe.cuh; 4 tests (prefill+hybrid fetch views, non-contiguous x through grouped path, iq grouped-vs-moe_vec, 65536-pair cap unit).
- Hardware A/B PENDING. Liveness contract: nsys probe must show moe_vec_q ABSENT from prefill chunks and grouped kernels (moe_iq3_xxs_q8_1 / moe_iq4_xs_q8_1 + moe_align launches) PRESENT; moe_vec_q still present in decode. Boot-log markers are unreliable (bare stdlib logger trap - see ft-bare-logger-liveness-trap). Use two boots with the env kill switch flipped, NOT a same-boot flip: radix-cache fill makes same-boot prefill before/after incomparable.

## STATE (2026-09-18)
v2 grouped MMQ prefill implemented + hardware A/B DONE: +16.4% @8128 (289.92 -> 337.61 tok/s), decode unchanged; liveness kernel-confirmed; the read-once ceiling was NOT met (grouped MoE ~1.8x on the term, m-block re-read hypothesis) - full details in memory ft-gguf-v2-grouped-mmq-measured. v0 expert-sort: no-op, reverted. 11-file v2 changeset uncommitted on vektory79 @ 1a444e4, commit pending user.

## CEILING CORRECTION (2026-09-18, v3 Step-0)
The ~758 tok/s projection is NOT reachable by traffic cuts alone: grouped DRAM floor is 4.54 s but measured ~10 s = 2.2x above it - the kernel LEFT the BW-bound regime (per-MAC ALU/LDS, 64-sync/block serialization, 98-B byte-assembly loads; L2 pushes the other way). m-tile / weight-stationary schedules are bounded by this non-DRAM floor. ncu discriminator: mio_throttle/short_scoreboard = ALU-bound, barrier/long_scoreboard = staging-bound. Details: memory ft-gguf-v3-mtile-campaign + .tasks/mmq-v3-stationary-moe/step0-source-read.md.
