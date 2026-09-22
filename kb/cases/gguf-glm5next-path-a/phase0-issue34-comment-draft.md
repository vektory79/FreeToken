# Draft: scoping comment for upstream issue #34 ([Feature] Native GGUF)
# The USER posts this themselves (CONTRIBUTING: agents do not post comments).
# Trim freely; every fact below is backed by .tasks/gguf-glm5next-path-a/ research artifacts.

---

Scoping a `glm5next` GGUF adapter (GLM-5.3-Flash)

Hardware: RTX 5090, Linux x86_64, FreeToken on branch vektory79 (= main + #408 port).
Checkpoint: /media/ai/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL.gguf
(147.5 GB single file, gguf v3; sibling mmproj-BF16.gguf one level up).
Exact command today: `ft serve --model-path /media/ai/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL.gguf`
Result: dies in seconds with ValueError at python/freetoken/models/gguf/config.py:65 -
"GGUF architecture 'glm5next' is not supported (known: ['gemma4'])"
(detection chain server/args.py:821 -> utils/hf.py:218 build_gguf_shim).

File inventory (gguf-py, verified by dump): 1412 tensors.
- Experts, stacked per layer (ffn_*_exps.weight): IQ4_XS x41 (down), IQ3_XXS x82 (gate/up),
  Q6_K x4 + Q4_K x1 + Q3_K x2 (down of select layers) - per-layer heterogeneous banks.
- Q8_0 x644 dense, F32 x638 norms/biases/ssm scalars.
- Full GLM-5.3-Flash feature set: KDA linear attention (ssm_*), mHC (hc_*), DSA indexer
  (blk.N.indexer.*, indexer_compressor_{ape,gate}). Reference: llama.cpp glm5next support
  (src/models/glm5next.cpp, 56 LLM_TENSOR kinds).

What already exists in-tree:
- GGUF shim + arch whitelist (models/gguf/config.py, gemma4 only) and the gemma4 adapter
  pattern (models/gemma4/gguf.py: parse_gguf_config / iter_gguf_weights;
  Gemma4GGUFForCausalLM in models/register.py).
- Vendored CUDA gguf kernels cover Q2_K..Q8_0 + iq2/iq3 variants; IQ4_XS dequant + MMVQ
  (dense and grouped moe_vec) is present as well. iq formats lack MMQ only; python-side
  dispatch (layers/gguf.py) currently exposes Q4_0/Q8_0/Q6_K.
- CPU GEMV supports bf16 / nvfp4 / mxfp4_triton / ds_fp4 / q4_0.

Proposed phased approach (working plan, ~3-6 weeks part-time, offload-first milestone):
1. config shim: glm5next -> Glm5NextArgs (40 glm5next.* metadata keys; KDA/DSA layer split
   via per-layer attention.head_count_kv == 0; dense/MoE via leading_dense_block_count).
2. tensor-name translator + iter_gguf_weights (stacked ffn_*_exps, packed qkv/gate_up,
   KDA in_proj/conv fusion).
3. tokenizer: sibling tokenizer.json from the HF checkpoint of the same model (same tokenizer).
4. python-side dispatch wiring for Q3_K/Q4_K/Q6_K/Q8_0/IQ3_XXS/IQ4_XS; MMQ-for-IQ only if
   the dense path needs it (experts run moe_vec MMVQ).
5. expert banks + offload integration (per-layer heterogeneous dtype, #194-class; cap
   prefill chunks below 8192 tokens for top_k=8 per #186).
6. E2E A/B numbers vs main (boot, prefill tok/s at 64k fill, decode tok/s, quality battery).

Questions:
- Is a glm5next adapter interesting given #327 (llama) and #359 (gemma) are family-specific?
- Is per-layer heterogeneous expert bank dtype (#194) a precondition, or can the first
  milestone be offload-only serving (GPU-only decode, no CPU GEMV formats)?