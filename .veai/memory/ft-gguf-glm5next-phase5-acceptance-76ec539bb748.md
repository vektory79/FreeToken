---
name: "ft-gguf-glm5next-phase5-acceptance"
description: "GGUF glm5next: off-topic defect RESOLVED - non-contiguous activation bug in fused_mul_mat_gguf; battery 5/5 on-topic"
type: project
lastUpdated: 2026-09-14T15:10
lastRecall: 2026-09-14T15:15
---

# GGUF glm5next Phase 5: offload acceptance PASSED on hardware (2026-09-14)

Phase 5 (expert banks -> offload) of plan .tasks/gguf-glm5next-path-a/PLAN.md is code-complete and hardware-verified; all changes UNCOMMITTED on top of 2637714 (Phases 1-4).

## What landed (crash-driven chain, each fix evidence-backed)
- Per-signature OffloadMoeCache partitions (engine.py: _group_bank_layers by (shape,dtype) per bank in role order; _split_moe_cache_budget byte-proportional with num_experts floors + auto BYTE-ENVELOPE shrink-to-fit; local layer-id remap proven by id()-identity test test_engine_partitions_real_model_routing; graph.py resets every partition; legacy single-cache readers point at the dominant partition).
- Per-partition prefill_overlap DEGRADE (Engine._partition_prefill_overlap): overlap only when the group's slots >= 2*num_experts; minorities ([8] and [9,41], 288 slots) run synchronous materialized prefill. Do NOT floor at 2E (wide minority signatures would cost ~16 GiB). The 2E invariant lives at offload_cache.py:176.
- Role-ordered gguf_types (explicit (gate,up,down), never slots.values()) + per-role geometry (down rows = H, not gate's I).
- #186 clamp: engine auto-clamps max_extend_tokens 8192 -> 8191 at top_k=8 (65535//top_k), loud log; runtime assert as backstop. NOTE: the default max_extend_tokens is 8192, NOT 4096.
- Pinned banks: HostBanks + LayerCompletionTracker + PinPipeline (q4_0 pattern; NO cudaHostRegister of the mmap). TRAP: expected_per_layer must equal the per-layer NOTE count (glm5next: 1 combined note/layer; gemma4: 2 per write) - mismatch silently skips ALL pinning -> CUDA IMA at the first decode copy (invisible to CPU tests). Details: ft-offload-banks-pinned-host.
- Guards: _build_fused_copy_plan rejects pageable CPU banks / per-layer width != plan feat / slot-pool geometry mismatch AT CONSTRUCTION; load-time validation of bank ggml types against the kernel MMVQ set; aggregated readouts (scheduler/cache_status); binding-group budget advice.
- graph.py: module-level OffloadMoeCache import + plural-aware single-or-list handling; boundary sync between groups (cross-group prefetch hop not wired - 4 syncs per chunk on the real file).

## Hardware acceptance (run 5, real 147 GB file)
`ft serve --model-path /media/ai/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL.gguf --moe-strategy offload --moe-cache-auto` (positional path REJECTED by argparse; --cuda-graph-max-bs 0 = eager).
- Boot 2m40s -> /ready 200; type histogram {(18,18,14):2, (18,18,23):39, (23,23,14):1}; minorities overlap OFF; 3 CUDA graphs captured; copy_missing clean.
- Prefill 4213 tok / 65s; decode 14.3 tok/s cached (TTFT 4.13s) / 15.6 tok/s fresh. NO IMA.
- Operational: auto-KV ~9k tokens (prompts must fit; 8k prefill + 768 decode = 8959 tokens fits 9024 barely - Phase 6 tunes the budget); minorities prefill synchronously; KDA autotuner needs ~256 MiB scratch (the eager-tail OOM at 268 MiB free was an artifact of --cuda-graph-max-bs 0).

## Not done (deferred, conscious)
- Exact per-layer sizing via header scan (current conservative mix overcounts +24.7 GiB aggregate; no active consumer misled - --moe-cache-auto sizes from actual sources).
- --moe-cache-auto vs explicit --moe-cache-size conflict error (pre-existing).
- Cross-group prefetch hop; cache_status under-reports minority partitions (dominant-only).
- Phase 6 full A/B (boot, prefill @64k fill, decode, quality battery vs NVFP4 - compare DECODED TEXT not ids; code-prompt tokenization diverges, see PLAN Phase 6 caveat).

- PHASE 6 A/B (2026-09-14, artifacts .tasks/gguf-glm5next-path-a/verification/phase6/): tuned GGUF cmd = base + `--num-tokens 70016 --memory-ratio 0.85 --max-prefill-length 4096` (auto gives ~9k KV; --num-tokens must be a page-64 multiple; explicit --moe-cache-size 864 fail-fasts - per-signature floors demand >= 8510 slots, so explicit sizing cannot shrink below auto; the working lever is auto slots + capped KV + ratio 0.85). Boot 115s vs NVFP4 56s. Prefill 64k: GGUF 263.9-268.1 vs NVFP4 510.3-515.2 tok/s (0.52x). Decode cached @64k: GGUF 14.67 (TTFT 3.29s) vs NVFP4 14.08 (TTFT 5.12s) - parity-or-better. 768-token run: both hit natural EOS early (GGUF 363@14.28, NVFP4 232@14.16). Plans: GGUF 1227 slots + 70,016 KV (0.78 GiB), overlap OFF on ALL partitions (464/288/288 < 576); NVFP4 322 slots + 526,784 KV. RAM 137/188 GiB (pinned banks).
- BLOCKING QUALITY FINDING: 24-prompt greedy battery - 0/24 identical, divergence at char 0 in 19/24, ON-TOPIC GGUF 3/24 vs NVFP4 24/24. GGUF answers a DIFFERENT task (GUI-agent JSON, image math, bounding boxes) - the user prompt effectively never reaches the model. Deterministic across boots. NOT the offload machinery (decode/prefill are clean) - a conversation-rendering defect in the GGUF tokenizer/chat-template path. PRIME SUSPECT: the real file's embedded vocab/template was never validated against the RedHatAI reference (the Phase 3 round-trip used a synthetic fixture built FROM the reference vocab); candidates: embedded-vocab misalignment, template rendering on the real file. BLOCKS private chat serving; performance alone is acceptable.

- RESOLVED 2026-09-14 (battery 5/5 on-topic vs 0/5 pre-fix): the blocking chat-quality defect was NOT the rendering path - it was a NON-CONTIGUOUS ACTIVATION bug in fused_mul_mat_gguf (layers/gguf.py): f_a/g_a from the in_proj split (glm5_next/kda.py:94-96) are views with stride (24896,1), and ggml_mul_mat_a8/vec_a8 assume row-major -> garbage gates -> KDA state decay corrupted in all 34 KDA layers from L0 -> context loss -> prior-driven text. Proof chain: per-layer bisect (L0 KDA output cos 0.9357 while input 0.99998; GGUF prompt-sensitivity frozen 0.77-0.97 vs NVFP4 decaying to 0.1-0.5; identical top-10 logits across 3 prompts) -> engine capture (g1 cos 0.083, g2 cos 0.053, upstream 0.99998) -> kernel unit test (contiguous 0.99998 vs non-contiguous 0.033). Fix: fused_mul_mat_gguf normalizes non-contiguous x (+4 lines) + regression test test_dense_gguf_noncontig_activations[4|27] (FAILING-before/PASSING-after). MoE EXONERATED three ways (router per-row in 42/42; offline recompute via gguf-py IQ3_XXS/IQ4_XS dequant cos 0.99994/0.99997 - also closed the IQ3_XXS tooling gap; NVFP4 replay cos 0.996/0.973/1.0). BONUS: residual-norm explosion 8.9->40k is architecture-normal mHC growth (both models). Fix UNCOMMITTED (layers/gguf.py + tests/kernels/test_gguf_quant.py).
