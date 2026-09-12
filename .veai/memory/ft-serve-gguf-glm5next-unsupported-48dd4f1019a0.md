---
name: "ft-serve-gguf-glm5next-unsupported"
description: "ft serve GGUF whitelist: only gemma4; glm5next rejected at models/gguf/config.py:65; no MR adds GLM GGUF"
type: project
lastUpdated: 2026-09-13T00:20
---

# ft serve GGUF support: only gemma4 arch; glm5next rejected

Verified 2026-09-12 on vektory79 (= main + #408 port) against /media/ai/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL.gguf (147.5 GB single file + mmproj-BF16.gguf; gguf python module reports arch `glm5next`, name "GLM 5.3 Flash", size_label 288x10B, 1412 tensors incl hc_* (mHC), ssm_* (KDA linear attn), MLA blk.N.attn_*, ffn_* experts).

- Boot attempt `ft serve --model-path <that gguf>` dies in seconds: ValueError at python/freetoken/models/gguf/config.py:65 build_gguf_shim: "GGUF architecture 'glm5next' is not supported (known: ['gemma4'])" (detection path: server/args.py:821 cached_load_hf_config -> utils/hf.py:218 build_gguf_shim). The app DOES detect .gguf files and has a config-shim layer - the whitelist is the blocker.
- Only GGUF-backed ModelSpec in the tree: "Gemma4GGUFForCausalLM" (models/register.py:192-197, parse_gguf_config + iter_gguf_weights). glm5_next/weight.py loads HF/FTW safetensors only.
- Upstream GGUF PRs/issues (open, 2026-09-12): #327 native gguf llama (llama family), #368 Q8_0 dequant, #315 GGUF token atomicity, #359 gemma4 dense GGUF, #268 Q4_0/Q6_0 KV. Issue #34 = umbrella [Feature] Native GGUF. NONE target glm5next.
- Even the supported gemma4 path fails on unsloth UD quants: issue #358 "Q8_0 declared in BLOCK_SHAPE/GGML_NAME/__all__ but missing from _DEQUANT, so unsloth UD-*_XL quants fail with NotImplementedError"; #194 heterogeneous expert-bank padding; #186 GGUF MoE GEMV crashes when tokens*top_k > 65535 (grid.z limit; GLM top_k=8 -> 4096-token chunks are 32768, 8192-token chunks hit exactly 65536).
- To serve glm5next GGUF the project needs a feature, not an MR pickup: new GGUF arch adapter (config key mapping -> Glm5NextArgs incl KDA linear_attn_config + kpool/DSA tensors), tensor-name translator (llama.cpp naming differs from HF), Q3_K-family dequant kernels wired into the MoE expert-bank/CPU-GEMV paths (only Q4_0/Q6_K exist; Q8_0 half-landed #368/#36), and ggml gpt2 tokenizer -> HF conversion. Dequant-to-bf16 conversion is not viable on this box: Q3 experts would expand to ~300+ GiB vs 188 GB RAM.

## Build-it-yourself feasibility for the glm5next GGUF (2026-09-12, scoping vs llama.cpp reference at /media/ai/src/llama-cpp-glm5next/)
- File quant inventory (read with gguf-py): experts = IQ4_XS x41 (ffn_down_exps, 1.2 GiB each), IQ3_XXS x82 (gate/up exps), Q6_K x4 + Q4_K x1 + Q3_K x2 (down exps of select layers); dense = Q8_0 x644; norms/biases/ssm scalars = F32 x638. DSA indexer tensors ARE present (blk.N.indexer.*, indexer_compressor_*).
- FreeToken vendored gguf CUDA kernel (csrc/gguf/gguf_kernel.cu, port of sgl-kernel, JIT via kernel/gguf.py) covers Q2_K/Q3_K/Q4_0/Q4_K/Q5_0/Q5_K/Q6_K/Q8_0 + iq2_xxs/iq2_xs/iq2_s/iq3_xxs/iq3_s dequant+MMQ symbols, but has NO IQ4_XS - the dominant expert format in this file. Porting IQ4_XS MMQ/MMVQ from llama.cpp ggml-cuda is the kernel-side blocker for a NATIVE gguf server.
- Two tiers: (A) native gguf serving = new family adapter (parse_gguf_config + iter_gguf_weights, ~56 tensor kinds incl mHC/ssm/indexer naming) + IQ4_XS kernel + gguf-quant expert banks through the hybrid/offload stack (CPU GEMV formats, benchbw, graphs) = multi-week feature; gemma4 adapter (474 lines, q4_0-only experts) is the in-tree pattern. (B) GGUF->NVFP4 FTW converter = reuses the whole tested hybrid stack, needs only gguf reader + tensor-name translator + streaming dequant (gguf-py has reference dequant for ALL these types incl IQ quants) + existing GPU nvfp4 repack; tokenizer can be copied from the RedHatAI HF checkpoint (same model). Days vs weeks.

## Path A plan written (2026-09-12)
Full phased plan for native glm5next GGUF serving lives at .tasks/gguf-glm5next-path-a/PLAN.md (goal, verified facts, phases 0-6 with acceptance criteria, estimation ~3-6 weeks, offload-first intermediate milestone, first-session starter checklist). Read it there; this memory only points to it.
