# Phase 2 - llama.cpp LLM_TENSOR kinds used by glm5next

Source of truth: llama.cpp reference checkout `/media/ai/src/llama-cpp-glm5next`.

- Name resolution: `LLM_TENSOR_NAMES` is one FLAT map at src/llama-arch.cpp:426-699.
  There is NO per-arch case block for glm5next - it reuses the generic patterns.
- Loader: `llama_model_glm5next::load_arch_tensors`, src/models/glm5next.cpp:102-221.
- Class: src/models/models.h:1342-1345 (`llama_model_glm5next : llama_model_base`,
  overrides load_arch_hparams / load_arch_tensors); graph helpers inherit
  llama_model_deepseek4::graph (models.h:1347-1348).
- Layer split in the loader: `is_recr(il)` = KDA layer (attention.head_count_kv == 0);
  `il < n_layer_dense_lead` = dense FFN; `il >= n_layer` = NextN block.
- Dim locals (glm5next.cpp:105-118): head_dim = n_embd_head_kda; d_inner = head_dim * n_head
  (n_head = n_head_arr[0], llama-model.h:827-845); d_conv = ssm_d_conv; q_lora_rank =
  n_lora_q; kv_lora_rank = n_lora_kv; qk_head_dim = n_embd_head_k_mla(); v_head_dim =
  n_embd_head_v_mla(); n_embd_indexer = indexer_head_size; kpool = indexer_kpool;
  hc_dim = dsv4_hc_mult * n_embd; hc_mix_dim = (2 + dsv4_hc_mult) * dsv4_hc_mult.
- 56 distinct LLM_TENSOR kinds are referenced (file-loaded + optional merged QKV).

## Embedding / output head

| kind | gguf name | dims | notes |
|---|---|---|---|
| TOKEN_EMBD | token_embd.weight | {n_embd, n_vocab} | glm5next.cpp:130 |
| OUTPUT | output.weight | {n_embd, n_vocab} | NOT_REQUIRED; when absent duplicated from token_embd with TENSOR_DUPLICATED - tied head (glm5next.cpp:133-136) |
| OUTPUT_NORM | output_norm.weight | {n_embd} | glm5next.cpp:132 |

## Per-layer norms (all layers, including the NextN block)

| kind | gguf name | dims | notes |
|---|---|---|---|
| ATTN_NORM | blk.%d.attn_norm.weight | {n_embd} | glm5next.cpp:142 |
| FFN_NORM | blk.%d.ffn_norm.weight | {n_embd} | glm5next.cpp:143 |

## mHC tensors (trunk layers only, il < n_layer)

| kind | gguf name | dims | notes |
|---|---|---|---|
| HC_ATTN_FN | blk.%d.hc_attn_fn.weight | {hc_dim, hc_mix_dim} | glm5next.cpp:146; hc_dim = hc_mult*n_embd; hc_mix_dim = (2+hc_mult)*hc_mult (117-118) |
| HC_ATTN_BASE | blk.%d.hc_attn_base.weight | {hc_mix_dim} | 147 |
| HC_ATTN_SCALE | blk.%d.hc_attn_scale.weight | {3} | 148 |
| HC_FFN_FN | blk.%d.hc_ffn_fn.weight | {hc_dim, hc_mix_dim} | 149 |
| HC_FFN_BASE | blk.%d.hc_ffn_base.weight | {hc_mix_dim} | 150 |
| HC_FFN_SCALE | blk.%d.hc_ffn_scale.weight | {3} | 151 |

Head-level HC kinds (output_hc_fn/base/scale/norm, arch.cpp:521-526) are NOT loaded:
glm5next collapses the mHC head with an unweighted mean, unlike deepseek4's gated head
(glm5next.cpp:670-672; models.h:1347 comment). The initial hc streams are a repeat of the
embedding (600-603).

## KDA layers (is_recr: per-layer attention.head_count_kv == 0)

| kind | gguf name | dims | notes |
|---|---|---|---|
| ATTN_Q | blk.%d.attn_q.weight | {n_embd, d_inner} | via create_tensor_qkv (glm5next.cpp:155; llama-model.cpp:3204-3234): merged ATTN_QKV blk.%d.attn_qkv tried first with TENSOR_NOT_REQUIRED\|TENSOR_SKIP_IF_VIRTUAL, else separate q/k/v (+ optional per-matrix biases) |
| ATTN_K | blk.%d.attn_k.weight | {n_embd, d_inner} | same helper |
| ATTN_V | blk.%d.attn_v.weight | {n_embd, d_inner} | same helper |
| SSM_CONV1D_Q | blk.%d.ssm_conv1d_q.weight | {d_conv, 1, d_inner, 1} | glm5next.cpp:157 (4D); the graph reshapes to 2D and concatenates q\|k\|v into one fused conv weight at runtime (251-255) |
| SSM_CONV1D_K | blk.%d.ssm_conv1d_k.weight | {d_conv, 1, d_inner, 1} | 158 |
| SSM_CONV1D_V | blk.%d.ssm_conv1d_v.weight | {d_conv, 1, d_inner, 1} | 159 |
| SSM_F_A | blk.%d.ssm_f_a.weight | {n_embd, head_dim} | 161; gate f_a leg |
| SSM_F_B | blk.%d.ssm_f_b.weight | {head_dim, d_inner} | 162 |
| SSM_G_A | blk.%d.ssm_g_a.weight | {n_embd, head_dim} | 163; output gate g_a leg |
| SSM_G_B | blk.%d.ssm_g_b.weight | {head_dim, d_inner} | 164 |
| SSM_BETA | blk.%d.ssm_beta.weight | {n_embd, n_head} | 166; beta = sigmoid(beta(x)) |
| SSM_A | blk.%d.ssm_a | {n_head} | 167 - loaded WITHOUT a "weight" suffix; holds -exp(A_log) per head (kimi-k3 convention, glm5next.cpp:7; kind at arch.cpp:490) |
| SSM_DT | blk.%d.ssm_dt.bias | {d_inner} | 168 - dt is the BIAS side (kind name blk.%d.ssm_dt, arch.cpp:481) |
| SSM_NORM | blk.%d.ssm_norm.weight | {head_dim} | 170 - KDA o_norm (RMS, then plain-sigmoid gate from ssm_g_a/g_b, 305-306) |
| ATTN_OUT | blk.%d.attn_output.weight | {d_inner, n_embd} | 171 |

## DSA / MLA layers (per-layer attention.head_count_kv > 0)

| kind | gguf name | dims | notes |
|---|---|---|---|
| ATTN_Q_A | blk.%d.attn_q_a.weight | {n_embd, q_lora_rank} | glm5next.cpp:173 |
| ATTN_Q_A_NORM | blk.%d.attn_q_a_norm.weight | {q_lora_rank} | 174 |
| ATTN_Q_B | blk.%d.attn_q_b.weight | {q_lora_rank, n_head*qk_head_dim} | 175 |
| ATTN_KV_A_MQA | blk.%d.attn_kv_a_mqa.weight | {n_embd, kv_lora_rank} | 177 |
| ATTN_KV_A_NORM | blk.%d.attn_kv_a_norm.weight | {kv_lora_rank} | 178 |
| ATTN_K_B | blk.%d.attn_k_b.weight | {qk_head_dim, kv_lora_rank, n_head} (3D) | 179 |
| ATTN_V_B | blk.%d.attn_v_b.weight | {kv_lora_rank, v_head_dim, n_head} (3D) | 180 |
| ATTN_OUT | blk.%d.attn_output.weight | {n_head*v_head_dim, n_embd} | 182 (KDA layers use {d_inner, n_embd} instead) |

3D MLA projections are why llama-quant.cpp:505 special-cases has_3d_mla for GLM5NEXT.
No attn_q/attn_k/attn_v, no rope tensors: absorbed MQA, K and V share the latent row
(glm5next.cpp:500-512), kq_scale = 1/sqrt(qk_head_dim) over the MLA head size (477).

## DSA indexer (underscore spellings - NOT deepseek4's dot-named attn_compressor_*)

| kind | gguf name | dims | notes |
|---|---|---|---|
| INDEXER_K_NORM | blk.%d.indexer.k_norm.weight | {n_embd_indexer} | glm5next.cpp:184 - LayerNorm (LLM_NORM), not RMS |
| INDEXER_K_NORM (bias side) | blk.%d.indexer.k_norm.bias | {n_embd_indexer} | 185; graph asserts the bias exists (330) |
| INDEXER_PROJ | blk.%d.indexer.proj.weight | {n_embd, indexer_n_head} | 186; head weights run PREC_F32 (411-412) |
| INDEXER_ATTN_K | blk.%d.indexer.attn_k.weight | {n_embd, n_embd_indexer} | 187 |
| INDEXER_ATTN_Q_B | blk.%d.indexer.attn_q_b.weight | {q_lora_rank, indexer_n_head*n_embd_indexer} | 188; consumes the q LoRA output qr (406) |
| INDEXER_COMPRESSOR_WGATE | blk.%d.indexer_compressor_gate.weight | {n_embd, n_embd_indexer} | 190; a second, independent projection cached beside the key (336-338) |
| INDEXER_COMPRESSOR_APE | blk.%d.indexer_compressor_ape.weight | {n_embd_indexer, kpool} | 191; added PRE-softmax over the pool-slot axis (383-384) |

## FFN dense (il < n_layer_dense_lead)

| kind | gguf name | dims | notes |
|---|---|---|---|
| FFN_GATE | blk.%d.ffn_gate.weight | {n_embd, n_ff} | glm5next.cpp:195; n_ff from feed_forward_length |
| FFN_UP | blk.%d.ffn_up.weight | {n_embd, n_ff} | 196 |
| FFN_DOWN | blk.%d.ffn_down.weight | {n_ff, n_embd} | 197 |

Plain silu, no clamp on the dense path beyond the swiglu_clamp_* keys (build_layer_ffn
539-545).

## MoE layers (il >= n_layer_dense_lead)

| kind | gguf name | dims | notes |
|---|---|---|---|
| FFN_GATE_INP | blk.%d.ffn_gate_inp.weight | {n_embd, n_expert} | glm5next.cpp:199 - router weight |
| FFN_EXP_PROBS_B | blk.%d.exp_probs_b.bias | {n_expert} | 200 - noaux_tc selection bias; biases top-k only, weights stay unbiased (547) |
| FFN_GATE_EXPS | blk.%d.ffn_gate_exps.weight | {n_embd, n_ff_exp, n_expert} stacked | 202 |
| FFN_UP_EXPS | blk.%d.ffn_up_exps.weight | {n_embd, n_ff_exp, n_expert} stacked | 203 |
| FFN_DOWN_EXPS | blk.%d.ffn_down_exps.weight | {n_ff_exp, n_embd, n_expert} stacked | 204 |
| FFN_GATE_SHEXP | blk.%d.ffn_gate_shexp.weight | {n_embd, n_ff_shexp} | 206 - shared expert |
| FFN_UP_SHEXP | blk.%d.ffn_up_shexp.weight | {n_embd, n_ff_shexp} | 207 |
| FFN_DOWN_SHEXP | blk.%d.ffn_down_shexp.weight | {n_ff_shexp, n_embd} | 208 |

Shared expert is added unscaled; routed_scaling_factor applies to the routed weights only
(560-568). No FFN_NORM_EXPS for glm5next (that kind exists at arch.cpp:569 for other
archs). Per-layer expert dims can vary via the expert_feed_forward_length array.

## NextN / MTP block (il >= n_layer)

| kind | gguf name | dims | notes |
|---|---|---|---|
| NEXTN_EH_PROJ | blk.%d.nextn.eh_proj.weight | {2*n_embd, n_embd} | glm5next.cpp:212; concatenates [enorm(tok), hnorm(h)] (763-764) |
| NEXTN_ENORM | blk.%d.nextn.enorm.weight | {n_embd} | 213 |
| NEXTN_HNORM | blk.%d.nextn.hnorm.weight | {n_embd} | 214 |
| NEXTN_SHARED_HEAD_NORM | blk.%d.nextn.shared_head_norm.weight | {n_embd} | 216; graph falls back to output_norm when absent (786-788) |
| NEXTN_EMBED_TOKENS | blk.%d.nextn.embed_tokens.weight | {n_embd, n_vocab} | NOT_REQUIRED (217); falls back to tok_embd (718-722) |
| NEXTN_SHARED_HEAD_HEAD | blk.%d.nextn.shared_head_head.weight | {n_embd, n_vocab} | NOT_REQUIRED (218); falls back to output (798-800) |

The NextN layer reuses ATTN_NORM / FFN_NORM (142-143, loaded for il < n_layer_all) plus a
full DSA attention (KDA branches excluded: s_copy expanded but unused, 740-742) and FFN
kinds at layer index n_layer. It carries NO hc_* tensors (assert, 708).

## Kinds that exist but are NOT used by glm5next

- ATTN_KV_B (deepseek2 kind, arch.cpp:516): glm5next uses ATTN_K_B + ATTN_V_B instead.
- ATTN_QKV as a required tensor: only the optional merged form inside create_tensor_qkv.
- ATTN_COMPRESSOR_* (dot spellings, arch.cpp:547-550): deepseek4 only.
- SSM_IN / SSM_OUT / SSM_X / SSM_D / SSM_A_NOSCAN / SSM_BETA_ALPHA / SSM_ALPHA: mamba /
  rwkv / kimi-linear kinds, unused here.
- PLE_* (arch.cpp:541-546): ple dims stay 0 for glm5next (no PLE keys read).
- ATTN_SINKS, SHORTCONV_*, FFN_NORM_EXPS, altup/laurel/per-layer kinds (gemma3n),
  DFLASH_*, DEC_*/ENC_* (t5-style).

## Tensors llama.cpp CREATES at runtime (not read from the gguf)

- Tied output head: OUTPUT duplicated from TOKEN_EMBD with TENSOR_DUPLICATED when absent
  (glm5next.cpp:134-136).
- Fused KDA conv weight: concat of ssm_conv1d_q/k/v reshaped to {d_conv, d_inner} each
  (glm5next.cpp:251-255); conv output goes through silu (258).
- kpool indexer inputs, created by llm_graph_input_kpool (llama-graph.cpp:3556-3638):
  pool_cells I32 {kpool*n_pools, n_stream} (3590); pool_bias F32 {n_pools, n_tps,
  n_stream} plus an F16 cast when cparams.fused_lid (3594-3603); sel_mask and cand_mask
  F16 {n_kv, n_tps, 1, n_stream} (3607, 3611); pool_reps I32 (3624); new_pool_cells I32
  {kpool*n_new_max, n_stream} (3628); new_pool_reps I64 (3632).
- Indexer K cache: each cell stores 3 heads (key | gate | pooled) of n_embd_indexer;
  llama_memory_hybrid_idx builds it with hand-tweaked hparams (single MQA key head,
  n_embd_head_k_full = indexer_head_size, llama-memory-hybrid-idx.cpp:49-60; equivalent
  tweak in llama-kv-cache-dsa.cpp:38-53). Type is forced back to F16 if the requested
  type_k is quantized, because the gate lanes feed a softmax (llama-model.cpp:2517-2523).
- KDA recurrent state planes from llama-memory-recurrent.cpp:101-115: r_l conv state of
  width n_embd_r = 3 * (ssm_d_conv - 1) * d_inner; s_l state of width n_embd_s =
  n_embd_head_kda^2 * n_head; the p_l ple plane is dead for glm5next (ple_n_heads == 0,
  llama-hparams.cpp:268-275).
- hc_init streams: repeat of the embedding into hc streams (glm5next.cpp:600-603);
  hc_mean collapses streams at the head (671).

## Quantization notes (llama-quant.cpp)

- Full-precision whitelist when quantizing (llama-quant.cpp:351-368): hc_attn_fn,
  hc_ffn_fn, indexer_compressor_ape, indexer_compressor_gate, indexer.proj,
  indexer.attn_k, indexer.attn_q_b, ssm_f_a, ssm_f_b, ssm_g_a, ssm_g_b, ssm_beta.
  Note the mixed spellings: compressor tensors use an UNDERSCORE, projections a DOT.
- MXFP4 ftype path: 3D MLA projections excluded from expert matching via has_3d_mla
  (llama-quant.cpp:501-513).

## mmproj / vision

src/models/glm5next.cpp contains zero vision/mmproj/clip references. The vision tower is
a separate clip_graph_glm5next (subclass of clip_graph_glm4v) built in
tools/mtmd/models/glm5next-vision.cpp, wired through PROJECTOR_TYPE_GLM5NEXT
(tools/mtmd/clip-impl.h:490,556; clip.cpp:1101,1103,1775,2606,4077,4104,4186,4819,6034;
mtmd.cpp:870-875; video tokens unsupported, mtmd.cpp:872). mmproj tensors therefore exist
only in a separate mmproj GGUF file and never enter the glm5next LLM graph. (The
companion dump tensors-mmproj.txt in this task folder corroborates that this checkpoint
ships such a separate mmproj file.)

## Engine capability switches (context)

- llm_arch_is_hybrid: true for GLM5NEXT (llama-arch.cpp:1084) - hybrid attn+recurrent
  memory; rs_rollback true (1110); sm_tensor false (1149).
- ROPE_TYPE_NONE (llama-model.cpp:2872-2874); model dispatch llama-model.cpp:205-206.
- Indexer memory built only when indexer_head_size > 0 (llama-model.cpp:2507-2524).
