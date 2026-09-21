# R1: existing moe_align_block_size producer for the grouped GGUF entry (2026-09-17)

Read-only research result (verified file:line at HEAD 1a444e4). Input for TASK.md v2 step 3.

## ANSWER: YES - the codebase already ships the trio producer. Direct reuse; no vLLM port needed.

## The producer
`freetoken.moe.fused.moe_align_block_size(topk_ids, block_size, num_experts)` - python/freetoken/moe/fused.py:47. Returns `(sorted_token_ids, expert_ids, num_tokens_post_pad)`. Dual backend:
- sgl_kernel CUDA op (fused.py:96-111), pre-allocated int32 buffers, called with `num_experts + 1` and a final boolean flag (True; likely pad_sorted_ids - sgl-kernel source not in tree, name unverified).
- Triton fallback `freetoken.kernel.triton.moe_align.moe_align_block_size` (kernel/triton/moe_align.py:216; provenance "moe_align<-vllm/redesign"; docstring states buffer sizes mirror fused.py exactly). Legacy staged variant also exists: kernel/moe_impl.py:65 `moe_align_block_size_triton` (exported kernel/__init__.py:40).

## Trio contract (both backends)
- topk_ids: int32, contiguous, [M, top_k]; -1 ids tolerated (spare histogram bin).
- sorted_token_ids: int32 1-D, numel = topk_ids.numel() + (num_experts+1)*(block_size-1) (or numel*block_size if numel < num_experts+1). Sentinel fill = numel, NOT -1 (fused.py docstring example; moe_align.py docstring).
- expert_ids: int32 1-D, numel = ceil(buffer/bs); entries beyond npp/bs not guaranteed written by the triton path (safe: kernel guards + sentinel rows).
- num_tokens_post_pad: int32, shape (1,), device tensor.
- Expert-count cap: sgl op needs round_up(E,32) < 1024 (moe/offload_cache.py:1122 comment) - E=288 fine.

## ggml_moe_a8 contract it must feed
- Python wrapper kernel/gguf.py:113: `ggml_moe_a8(x, weight, sorted_token_ids, expert_ids, num_tokens_post_padded, quant_type, row, top_k, tokens)` -> csrc gguf_kernel.cu:335. x [tokens,col] fp32/fp16/bf16 (dispatch.h:21-25); weight [E,row,*] packed, exp_stride=W.stride(0); output Y [tokens*top_k, row] (torch.empty).
- block_size is NOT free: kernel m-tile = MOE_X_<type>, 4 on CUDA / 8 on ROCm for every case (moe.cuh:152-156 etc.), exposed via `ggml_moe_get_block_size(quant_type)` (gguf_kernel.cu:812, bound kernel/gguf.py:151). Producer must pass that value as block_size. For types 18/23 (IQ3_XXS/IQ4_XS) the getter returns 0 and no MOE_X_IQ* constants exist - v2 must add them (pick 4; note gate/up=IQ3_XXS vs down=IQ4_XS differ per layer, equal constants let one trio serve all three projections).
- Grid is sized host-side off `sorted_token_ids.numel()` (moe.cuh launch, block_num_y = buffer/mmq_x); device `num_tokens_post_padded[0]` is only an in-kernel early-exit (moe.cuh:53) - so the exact fused.py buffer convention must be kept. Token mapping: kernel reads flat pair index and divides by its top_k arg (moe.cuh:100), so ONE trio serves gate/up (top_k=8) and the down call (top_k=1, tokens=M*top_k) - same pattern as the existing moe_vec down call (layers/moe.py:478) and the ds_fp4 precedent (fused_ds_fp4.py:217 "one moe_align sort shared by both GEMMs"; fused_nvfp4.py:309).

## topk_ids at the gguf prefill call site
layers/moe.py: prefill_forward:241 -> fused_topk -> kernel/triton/moe_router.py:122 returns int32 [M, topk] -> _prefill_routed:349 (ids pass through unmapped, position==expert id) -> _expert_gemm gguf branch (layers/moe.py:442-490, currently only ggml_moe_a8_vec). int32 dtype also asserted in fused.py:301. Producer input = exactly this tensor.

## Tests
No caller drives grouped `ggml_moe_a8` anywhere (find_usages: only TASK/docs refs + __all__) - it is dead but fully bound. Nearest fixtures: tests/moe/test_dsfp4_grouped_prefill.py:82 (drives moe_align_block_size against a vLLM-style grouped prefill GEMM, second GEMM with top_k=1) and tests/kernels/test_gguf_quant.py:172-188 / tests/models/test_gguf_expert_banks.py:162-179 (drive ggml_moe_a8_vec).

## MUST-FIX hazard found (beyond the question)
moe.cuh:52: `if (exp_idx > 255 || exp_idx < 0) return;` - with E=288, experts 256-287 are silently dropped, and Y is torch.empty so their rows are garbage. v2 must replace 255 with the real expert bound (sentinel id E must stay dropped). Also verify the sgl path is exercised on this rig (fused.py silently picks triton if sgl_kernel is absent - either backend satisfies the contract).
