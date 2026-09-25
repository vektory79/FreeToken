---
name: "ft-serve-gguf-glm5next-campaign"
description: "GGUF glm5next Path A campaign: offload chain to 78115df, hybrid campaign, final verdict hybrid RECOMMENDED (16.67 tok/s)"
type: project
lastUpdated: 2026-09-16T15:11
lastRecall: 2026-09-24T14:15
---

# GGUF glm5next (Path A): campaign record - see PLAN.md for the full record

Private-use work (upstream #34 gate WAIVED by the user, see ft-gguf-glm5next-private-scope). The authoritative, always-current record is .tasks/gguf-glm5next-path-a/PLAN.md (+ research/ + verification/); this memory holds only the durable anchors.

## Target and how to run
- On MAIN: boot dies at models/gguf/config.py:65 "GGUF architecture 'glm5next' is not supported" (GGUF_ARCH_TO_REGISTRY whitelist, only gemma4). Real file: /media/ai/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL.gguf (147.5 GB, arch glm5next, 1412 tensors; mmproj-BF16.gguf one level up).
- Known-good serve: `ft serve --model-path <gguf> --moe-strategy offload --moe-cache-auto --port 18081` (positional path rejected). Tuned Phase 6 command: base + `--num-tokens 70016 --memory-ratio 0.85 --max-prefill-length 4096`; explicit --moe-cache-size FAIL-FASTS (per-signature floors need >= 8510 slots - explicit sizing cannot shrink below auto).

## Offload campaign (2026-09-14, vektory79): COMPLETE, tree clean
Chain: 4162f7c (config shim + registry + AOT claim) -> 4443d7e (iter_gguf_weights + iter_gguf_expert_sources + dequant_q8_0) -> 2637714 (dispatch wiring + tokenizer + engine gate) -> f801a08 (Phase 5 offload banks) -> 028f2d9 (contiguity fix) -> 78115df (digit-split pre-tokenizer, exact bank sizing, cache-status tests, clang docs).
- Quality: full NVFP4 parity modulo quant-level wording drift; 24-prompt battery 3/24 -> 20/24 on-topic after the contiguity fix (NVFP4 24/24), 6/6 arithmetic battery after the digit-split (100 mod 7 -> Friday, 2^10=1024). Residual-norm explosion 8.9->40k is architecture-normal mHC (both models).
- Perf vs NVFP4 on the real 147 GB file: prefill 264-268 vs 510-515 tok/s (0.52x), decode 13.8-15.6 tok/s, boot 115 vs 56s. Production-usable for private prose chat.
- Only remaining deferred: kernel-level assert for non-contiguous activations (vendor discipline; python-side construction guard in _build_fused_copy_plan suffices).

## Durable root causes
- Chat-quality defect (028f2d9): fused_mul_mat_gguf (layers/gguf.py) fed NON-CONTIGUOUS activations (f_a/g_a views, stride (24896,1)) from the in_proj split to row-major-assuming ggml GEMM kernels -> garbage gates -> KDA state decay from L0 -> prior-driven text. NOT tokenizer/vocab/template. Fix: +4 lines normalize non-contiguous x + failing-before regression test; perf parity. MoE exonerated three ways (incl. gguf-py reference IQ dequant). Chain: verification/phase6/bisect/ROOT-CAUSE.md.
- Digit-split (78115df): GGUFGPTConverter's GPT-2 regex glued digits in USER INPUT; reference pre_tokenizer ported as _GLM4_SPLIT_REGEX + _glm4_pre_tokenizer (models/gguf/tokenizer.py), glm5next-gated, gemma4 untouched.
- Phase 5 offload banks: per-signature OffloadMoeCache partitions (role-ordered gguf_types, layer-id remap proven by id()-identity test, byte-proportional budget with E floors + auto byte-envelope, graph.py resets per partition); pinned banks via LayerCompletionTracker + PinPipeline - the note-count trap lives in ft-offload-banks-pinned-host.
- Per-partition prefill_overlap DEGRADE (>= 2E only; minorities synchronous; do NOT floor at 2E): invariant at offload_cache.py:176, implemented by Engine._partition_prefill_overlap. Phase 6 tuned plan runs overlap OFF on all partitions (464/288/288 < 576); cross-group prefetch hop = HONEST SKIP (contract-pinning test added).
- #186 clamp: max_extend_tokens 8192 -> 8191 at top_k=8 (the DEFAULT is 8192, not 4096).
- Exact sizing (78115df): _gguf_bank_bytes header-only scan wired into bank_bytes_estimate; real-distribution pin 134,404,374,528 B vs conservative 160,922,861,568 (+24.7 GiB); no-path fallback kept; benchbw untouched.
- Kernel status: IQ4_XS IS in kernel/csrc/gguf (dequant + MMVQ + moe_vec); iq lacks only MMQ; the real gap was python dispatch (2637714). CPU GEMV untouched (offload-first decision; cpu_executor must NOT claim gguf formats).
- cache_status aggregation was ALREADY in f801a08; that wave fixed 2 silently-broken fixtures + 3 partition tests (kvcache 19 passed, was 16 with 2 failing).
- Known conscious divergences: indexer_rope_interleave shim False vs HF True (inert); code-prompt tokenization diverges from NVFP4 - compare decoded text, not ids.
- Tests: tests/models/test_glm5_next_gguf.py (59 env-set), tests/models/test_gguf_expert_banks.py (29), tests/kernels/test_gguf_quant.py, tests/moe/test_cpu_moe_gguf_iq.py, engine 146. Toolchain: gguf kernel JIT needs clang++ host - see ft-gguf-kernel-jit-toolchain.
- Upstream GGUF issues for context: umbrella #34; gemma4 raw #358/#186/#194. NONE target glm5next.

## Hybrid campaign (2026-09-15, follow-on) - FINAL: hybrid RECOMMENDED
Commits: 5a04423 (CPU GEMV iq3_xxs/iq4_xs/q6_k, per-projection tables) -> bd02232 (fix wave) -> 427a431 (negative tests) -> 2169baa (benchbw CPU leg) -> aad5d3a (per-cache executors, disjoint pools, capability-gated multi-partition) -> f6b94ad + ef83ee8 (engine gate via header-scan capability check) -> eb7de4c (hardening) -> 11f1a80 (verbatim ggml AVX2 W4A8-K tier in select_ggufdot_i8; TIER env override) + 979e3fc (weighted pool split [15,1,1] @16T) -> 63b9bff (cap-down + wrap guard + profile restore) -> 1635ecd (CC/CXX JIT leak fix, kernel/gguf.py).
- Stages: initial hybrid decode 2.99 tok/s vs 14.25 offload (per-layer "handshake floor" 1.9-3 ms x 42 layers - later MIS-ATTRIBUTED, see gguf-hybrid-decode-handshake-floor); NVFP4 A/B reversed the Task 06 NO-GO (hybrid 1.39-1.43x over offload even with the slow scalar leg); Task 06 re-profile baked f=78% -> 29.5%, CPU leg 11.65 -> 65.6 GB/s.
- FINAL (see gguf-hybrid-reacceptance-final): re-acceptance PASSED, hybrid 16.67 tok/s @64k vs offload 12.97 = 1.28x, 1.43 ms/layer all-in, battery 24/24. Plan artifacts: .tasks/gguf-glm5next-hybrid/.
