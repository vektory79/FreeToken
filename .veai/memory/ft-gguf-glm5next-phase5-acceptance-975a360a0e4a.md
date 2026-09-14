---
name: "ft-gguf-glm5next-phase5-acceptance"
description: "GGUF glm5next: Phases 1-6 committed through 028f2d9; digit-split fixed (battery 6/6); sizing+prefetch waves in flight"
type: project
lastUpdated: 2026-09-14T19:10
lastRecall: 2026-09-14T22:47
---

# GGUF glm5next: Phase 5 acceptance + Phase 6 closure + post-campaign fixes (2026-09-14)

Private-use work (upstream #34 gate waived, see ft-gguf-glm5next-private-scope). Plan: .tasks/gguf-glm5next-path-a/PLAN.md.

## Phase 5 (committed f801a08): expert banks -> offload, hardware-accepted
- Per-signature OffloadMoeCache partitions (role-ordered gguf_types, local layer-id remap proven by id()-identity test, byte-proportional budget with E floors + auto byte-envelope, graph.py resets every partition).
- Per-partition prefill_overlap DEGRADE (>= 2E only; minorities synchronous; do NOT floor at 2E). The 2E invariant: offload_cache.py:176.
- #186 clamp: max_extend_tokens 8192 -> 8191 at top_k=8 (the DEFAULT is 8192, not 4096).
- Pinned banks via LayerCompletionTracker + PinPipeline. TRAP: expected_per_layer must equal the per-layer NOTE count (glm5next 1/layer, gemma4 2/write) - mismatch silently skips ALL pinning -> CUDA IMA at the first decode copy. _build_fused_copy_plan now rejects pageable banks/width mismatches/geometry errors at construction.
- Acceptance (run 5, real 147 GB file): boot 2m40s, 3 graphs, prefill 4213 tok, decode 14.3/15.6 tok/s, no IMA.

## Phase 6 (closed): A/B + the non-contiguous fix
- A/B: prefill 264-268 vs 510-515 tok/s (0.52x); decode 14.67 vs 14.08 @64k (TTFT 3.3 vs 5.1s); boot 115 vs 56s. Tuned command: base + `--num-tokens 70016 --memory-ratio 0.85 --max-prefill-length 4096` (explicit --moe-cache-size 864 FAIL-FASTS: per-signature floors need >= 8510 slots - explicit sizing cannot shrink below auto).
- Blocking quality defect ROOT-CAUSED AND FIXED (commit 028f2d9): fused_mul_mat_gguf (layers/gguf.py) fed NON-CONTIGUOUS activations (f_a/g_a views, stride (24896,1)) to row-major-assuming ggml GEMMs -> garbage gates -> KDA state decay from L0 -> prior-driven text. Fix: +4 lines normalize non-contiguous x + regression test (FAILING-before/PASSING-after). Battery: 3/24 -> 20/24 on-topic; perf parity (the guard costs nothing measurable).
- Post-fix battery (20/24): divergence chars 49-339 = quantization-level drift. Residual-norm explosion 8.9->40k is architecture-normal mHC (both models). MoE exonerated three ways (incl. gguf-py reference IQ dequant - closed the IQ3_XXS tooling gap).

## Post-campaign fixes (2026-09-14)
- DIGIT-SPLIT FIXED (uncommitted): GGUFGPTConverter's GPT-2 regex glued digits ("in 100 days" -> "[blank]"); the reference pre_tokenizer = Sequence(Split Regex with \p{N}{1,3} chunking, Isolated) + ByteLevel(add_prefix_space=false) - ported natively in models/gguf/tokenizer.py (_GLM4_SPLIT_REGEX + _glm4_pre_tokenizer), glm5next-gated, gemma4 untouched. Live battery 6/6 with correct arithmetic (100 mod 7 -> Friday; 2^10=1024; 2+2=4). Tests: env-set 59 / env-unset 58+1 / gemma4 gate / engine 146 / kernels 23.
- EXACT SIZING (uncommitted): _gguf_bank_bytes header-only scan wired into bank_bytes_estimate (model_path available at both production call sites); real-distribution pin 134,404,374,528 B vs conservative 160,922,861,568 (+24.7 GiB); no-path fallback kept; benchbw untouched. Tests: banks 29, engine 146, models 79.
- IN FLIGHT: cross-group prefetch hop evaluation (engine.py/offload_cache.py - may honestly conclude negligible gain; minority sync is inherent by the 2E degrade).
- Deferred: cache_status minority under-report; kernel-level assert for non-contiguous activations; clang-requirement docs (only in kernel/gguf.py:27-39 docstring).

## Campaign status
Commits: 4162f7c (shim) -> 4443d7e (iterator) -> 2637714 (dispatch+tokenizer) -> f801a08 (offload banks) -> 028f2d9 (contiguity fix). Production-usable for private chat: full NVFP4-parity quality (modulo quant drift), prefill 264 tok/s (0.52x), decode 13.8-15.6 tok/s @57-64k. All quality defects closed.
