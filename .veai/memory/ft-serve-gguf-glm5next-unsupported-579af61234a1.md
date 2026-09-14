---
name: "ft-serve-gguf-glm5next-unsupported"
description: "GGUF glm5next tracker: off-topic defect RESOLVED (non-contiguous activation bug, fixed, battery 5/5); fix uncommitted"
type: project
lastUpdated: 2026-09-14T15:10
lastRecall: 2026-09-14T15:15
---

# GGUF glm5next (Path A): phase tracker - see PLAN.md for the full record

Private-use work (upstream #34 gate WAIVED by the user, see ft-gguf-glm5next-private-scope). The authoritative, always-current record is .tasks/gguf-glm5next-path-a/PLAN.md (+ research/ + verification/ artifacts); this memory holds only the durable anchors.

- On MAIN: boot dies at models/gguf/config.py:65 "GGUF architecture 'glm5next' is not supported" (whitelist GGUF_ARCH_TO_REGISTRY, only gemma4). Real file: /media/ai/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL.gguf (147.5 GB, arch glm5next, 1412 tensors; mmproj-BF16.gguf one level up).
- Phase status (2026-09-14): Phases 1-4 COMMITTED on vektory79 - 4162f7c (config shim + registry + AOT claim), 4443d7e (iter_gguf_weights + iter_gguf_expert_sources + dequant_q8_0), 2637714 (dispatch wiring + tokenizer + engine gate). Phase 5 DONE uncommitted (acceptance PASSED) and Phase 6 A/B DONE same day: prefill 0.52x vs NVFP4, decode parity, BLOCKING chat-quality defect in the GGUF conversation-rendering path - all details + tuned serve command in ft-gguf-glm5next-phase5-acceptance.
- Kernel status (supersedes old claims): IQ4_XS IS in kernel/csrc/gguf (dequant + MMVQ + moe_vec); iq lacks only MMQ; the real gap was python dispatch - landed in 2637714. CPU GEMV untouched (offload-first decision; cpu_executor must NOT claim gguf formats).
- Toolchain: gguf kernel JIT needs clang++ host (kernel/gguf.py:25-40, FREETOKEN_GGUF_HOST_CXX override) - see ft-gguf-kernel-jit-toolchain.
- Tests: tests/models/test_glm5_next_gguf.py (58 env-set), tests/models/test_gguf_expert_banks.py (25), tests/kernels/test_gguf_quant.py (21 GPU parity, dfloat contract), tests/engine 146. Known-good serve command: `ft serve --model-path <gguf> --moe-strategy offload --moe-cache-auto --port 18081` (positional path rejected).
- Known divergences (conscious): indexer_rope_interleave shim False vs HF True (inert); code-prompt tokenization diverges from NVFP4 (pre=glm4 split missing in the gpt2 converter) - Phase 6 quality A/B must compare decoded text, not ids.
- Upstream GGUF issues for context: umbrella #34; gemma4 raw #358/#186/#194. NONE target glm5next.

- RENDERING DEFECT RESOLVED 2026-09-14: NOT tokenizer/vocab/template (re-confirmed) - the root cause was a non-contiguous activation bug in fused_mul_mat_gguf (layers/gguf.py): f_a/g_a views (stride (24896,1)) from the in_proj split fed to row-major-assuming ggml GEMM kernels -> garbage gates -> KDA context loss -> prior-driven text. Fixed (+4 lines normalize non-contiguous x) + failing-before regression test; battery 5/5 on-topic vs 0/5 pre-fix. MoE exonerated three ways (incl. gguf-py reference IQ dequant). Full chain: verification/phase6/bisect/ROOT-CAUSE.md. Fix UNCOMMITTED (layers/gguf.py + test_gguf_quant.py).
