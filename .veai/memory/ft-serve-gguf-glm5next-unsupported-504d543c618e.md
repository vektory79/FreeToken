---
name: "ft-serve-gguf-glm5next-unsupported"
description: "GGUF glm5next hybrid follow-on campaign: 8 commits, functional PASS, perf MISS (handshake floor), Task 06 re-scoped"
type: project
lastUpdated: 2026-09-15T05:27
lastRecall: 2026-09-15T03:11
---

# GGUF glm5next (Path A): campaign complete - see PLAN.md for the full record

Private-use work (upstream #34 gate WAIVED by the user, see ft-gguf-glm5next-private-scope). The authoritative, always-current record is .tasks/gguf-glm5next-path-a/PLAN.md (+ research/ + verification/); this memory holds only the durable anchors.

## Target and how to run
- On MAIN: boot dies at models/gguf/config.py:65 "GGUF architecture 'glm5next' is not supported" (GGUF_ARCH_TO_REGISTRY whitelist, only gemma4). Real file: /media/ai/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL.gguf (147.5 GB, arch glm5next, 1412 tensors; mmproj-BF16.gguf one level up).
- Known-good serve: `ft serve --model-path <gguf> --moe-strategy offload --moe-cache-auto --port 18081` (positional path rejected). Tuned Phase 6 command: base + `--num-tokens 70016 --memory-ratio 0.85 --max-prefill-length 4096`; explicit --moe-cache-size FAIL-FASTS (per-signature floors need >= 8510 slots - explicit sizing cannot shrink below auto).

## Final status (2026-09-14, vektory79): CAMPAIGN COMPLETE, tree clean
Chain: 4162f7c (config shim + registry + AOT claim) -> 4443d7e (iter_gguf_weights + iter_gguf_expert_sources + dequant_q8_0) -> 2637714 (dispatch wiring + tokenizer + engine gate) -> f801a08 (Phase 5 offload banks) -> 028f2d9 (contiguity fix) -> 78115df (digit-split pre-tokenizer, exact bank sizing, cache-status tests, clang docs).
- Quality: full NVFP4 parity modulo quant-level wording drift; 24-prompt battery 3/24 -> 20/24 on-topic after the contiguity fix (NVFP4 24/24), 6/6 arithmetic battery after the digit-split (100 mod 7 -> Friday, 2^10=1024). Residual-norm explosion 8.9->40k is architecture-normal mHC (both models).
- Perf vs NVFP4 on the real 147 GB file: prefill 264-268 vs 510-515 tok/s (0.52x), decode 14.67 vs 14.08 @64k, boot 115 vs 56s; final: prefill 264 tok/s, decode 13.8-15.6 tok/s. Production-usable for private prose chat.
- Phase 5 acceptance (run 5): boot 2m40s, 3 graphs, prefill 4213 tok, decode 14.3/15.6, no IMA.
- ONLY remaining deferred: kernel-level assert for non-contiguous activations (vendor discipline; the python-side construction guard in _build_fused_copy_plan suffices).

## Durable root causes
- Chat-quality defect (028f2d9): fused_mul_mat_gguf (layers/gguf.py) fed NON-CONTIGUOUS activations (f_a/g_a views, stride (24896,1)) from the in_proj split to row-major-assuming ggml GEMM kernels -> garbage gates -> KDA state decay from L0 -> prior-driven text. NOT tokenizer/vocab/template. Fix: +4 lines normalize non-contiguous x + failing-before regression test; perf parity (.contiguous() costs nothing). MoE exonerated three ways (incl. gguf-py reference IQ dequant, closing the IQ3_XXS tooling gap). Full chain: verification/phase6/bisect/ROOT-CAUSE.md.
- Digit-split (78115df): GGUFGPTConverter's GPT-2 regex glued digits in USER INPUT ("in 100 days" -> "[blank] days"); reference pre_tokenizer = Sequence(Split Regex \p{N}{1,3} chunking, Isolated + ByteLevel) ported natively as _GLM4_SPLIT_REGEX + _glm4_pre_tokenizer (models/gguf/tokenizer.py), glm5next-gated, gemma4 untouched.
- Phase 5 offload banks: per-signature OffloadMoeCache partitions (role-ordered gguf_types, layer-id remap proven by id()-identity test, byte-proportional budget with E floors + auto byte-envelope, graph.py resets per partition); pinned banks via LayerCompletionTracker + PinPipeline - the note-count trap lives in ft-offload-banks-pinned-host.
- Per-partition prefill_overlap DEGRADE (>= 2E only; minorities synchronous; do NOT floor at 2E): invariant at offload_cache.py:176, implemented by Engine._partition_prefill_overlap. Phase 6 tuned plan runs overlap OFF on all partitions (464/288/288 < 576); cross-group prefetch hop = HONEST SKIP (all tuned-plan partitions overlap-degraded -> hop would be dead code; contract-pinning test added).
- #186 clamp: max_extend_tokens 8192 -> 8191 at top_k=8 (the DEFAULT is 8192, not 4096).
- Exact sizing (78115df): _gguf_bank_bytes header-only scan wired into bank_bytes_estimate; real-distribution pin 134,404,374,528 B vs conservative 160,922,861,568 (+24.7 GiB); no-path fallback kept; benchbw untouched.
- Kernel status: IQ4_XS IS in kernel/csrc/gguf (dequant + MMVQ + moe_vec); iq lacks only MMQ; the real gap was python dispatch (2637714). CPU GEMV untouched (offload-first decision; cpu_executor must NOT claim gguf formats).
- cache_status aggregation was ALREADY in f801a08 (the deferred item was stale); that wave fixed 2 silently-broken fixtures + 3 partition tests (kvcache 19 passed, was 16 with 2 failing).
- Known conscious divergences: indexer_rope_interleave shim False vs HF True (inert); code-prompt tokenization diverges from NVFP4 (pre=glm4 split missing in the gpt2 converter) - compare decoded text, not ids.
- Tests: tests/models/test_glm5_next_gguf.py (59 env-set), tests/models/test_gguf_expert_banks.py (29), tests/kernels/test_gguf_quant.py (GPU parity + contiguity regression, dfloat contract), tests/moe/test_cpu_moe_gguf_iq.py (analytic IQ), engine 146. Toolchain: gguf kernel JIT needs clang++ host - see ft-gguf-kernel-jit-toolchain.
- Upstream GGUF issues for context: umbrella #34; gemma4 raw #358/#186/#194. NONE target glm5next.

## 2026-09-15: hybrid campaign (follow-on to THIS campaign) - functional PASS, perf MISS
Plan: .tasks/gguf-glm5next-hybrid/PLAN.md (Tasks 01-05+07 closed; Task 06 re-scoped). 8 commits: 5a04423 (CPU GEMV iq3_xxs/iq4_xs/q6_k, per-projection tables), bd02232 (hardening fix wave), 427a431 (negative-path tests), 2169baa (benchbw CPU leg; fraction 0.781; r=3.6-3.7), aad5d3a (per-cache executors, disjoint pools, capability-gated multi-partition), f6b94ad + ef83ee8 (engine gate accepts gguf+hybrid via header-scan capability check), eb7de4c (hardening ride-along), 1635ecd (CC/CXX JIT leak fix). Hardware (Task 05): hybrid boots/serves the real file, battery 24/24, fraction 78% active, VRAM noise - BUT decode 2.99 tok/s vs 14.25 offload control: per-layer GPU<->CPU handshake floor (1.9-3 ms x 42 layers) + dominant-pool thread starvation; see the gguf-hybrid-decode-handshake-floor memory and .tasks/gguf-glm5next-hybrid/verification/. Offload remains the RECOMMENDED strategy for this file until re-scoped Task 06 (handshake batching + SIMD + weighted affinities) lands.
