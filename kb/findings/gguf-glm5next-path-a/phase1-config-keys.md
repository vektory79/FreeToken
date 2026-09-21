# Phase 1 - glm5next GGUF config keys -> FreeToken Glm5NextArgs

Source of truth: llama.cpp reference checkout `/media/ai/src/llama-cpp-glm5next` (read-only).
FreeToken target: `/media/ai/src/FreeToken/python/freetoken/models/glm5_next/`.

## How key names resolve

- There are NO per-arch KV case blocks in this checkout. `LLM_KV_NAMES` is one FLAT map
  at src/llama-arch.cpp:161-424; `%s` is filled with the arch name at format time
  (src/llama-arch.cpp:994), so every key below is prefixed `glm5next.` (arch name
  registered at src/llama-arch.cpp:87 `{ LLM_ARCH_GLM5NEXT, "glm5next" }`; enum entry
  src/llama-arch.h:92).
- Generic reader (all archs): `llama_model_base::load_hparams`, src/llama-model.cpp:1200-1378.
- Arch override: `llama_model_glm5next::load_arch_hparams`, src/models/glm5next.cpp:22-100.
- "required" below means the get_key call omits the optional flag; a missing required key
  aborts the load. `key_or_arr` = scalar accepted, or per-layer array over n_layer_all.

## A. Arch-specific keys (read in load_arch_hparams, glm5next.cpp:22-100)

| gguf metadata key | llama.cpp hparam field | FreeToken Glm5NextArgs field | notes |
|---|---|---|---|
| glm5next.attention.layer_norm_rms_epsilon | hparams.f_norm_rms_eps | norm_eps | required; glm5next.cpp:23; key string arch.cpp:255 |
| glm5next.attention.layer_norm_epsilon | hparams.f_norm_eps | UNMAPPED (indexer k_norm LayerNorm eps; FreeToken hardcodes 1e-6, models/glm5_next/attention.py:63) | required; glm5next.cpp:25, warns when != reference 1e-6 (26-30); key string arch.cpp:254 |
| glm5next.attention.q_lora_rank | hparams.n_lora_q | q_lora_rank | required; assert >0 glm5next.cpp:32,36; arch.cpp:259 |
| glm5next.attention.kv_lora_rank | hparams.n_lora_kv | kv_lora_rank | required; glm5next.cpp:33; arch.cpp:260 |
| glm5next.attention.key_length_mla | hparams.n_embd_head_k_mla_impl | qk_nope_head_dim | required; glm5next.cpp:34; arch.cpp:275; NoPE so MLA k-dim == nope dims; llama asserts n_rot()==0 (37) |
| glm5next.attention.value_length_mla | hparams.n_embd_head_v_mla_impl | v_head_dim | required; glm5next.cpp:35; arch.cpp:276 |
| glm5next.ssm.conv_kernel | hparams.ssm_d_conv | linear_conv_kernel_dim (linear_attn_config.short_conv_kernel_size) | required; assert >1 glm5next.cpp:39-41; arch.cpp:335 |
| glm5next.kda.head_dim | hparams.n_embd_head_kda | linear_head_dim (linear_attn_config.head_dim) | required; assert >0 glm5next.cpp:40-42; arch.cpp:342 |
| glm5next.kda.gate_lower_bound | hparams.kda_gate_lower_bound | linear_lower_bound (linear_attn_config.gate_lower_bound) | required; assert <0 glm5next.cpp:44-45; arch.cpp:344 |
| glm5next.attention.indexer.head_count | hparams.indexer_n_head | index_n_heads | required; glm5next.cpp:47; arch.cpp:282 |
| glm5next.attention.indexer.key_length | hparams.indexer_head_size | index_head_dim | required; glm5next.cpp:48; arch.cpp:283; also sizes the indexer K cache (llama-kv-cache-dsa.cpp:45, llama-memory-hybrid-idx.cpp:52) |
| glm5next.attention.indexer.top_k | hparams.indexer_top_k | index_topk | required; top_k % kpool == 0 asserted glm5next.cpp:49,51-52; arch.cpp:284 |
| glm5next.attention.indexer.kpool | hparams.indexer_kpool | index_kpool (derive index_kpool_compress = kpool > 1; FreeToken keeps a separate bool, args.py:142-144) | required; assert >0 glm5next.cpp:50-51; arch.cpp:288; n_select = top_k + kpool - 1 = (top_k/kpool + 1)*kpool - 1 (glm5next.cpp:9-20) |
| glm5next.hyper_connection.count | hparams.dsv4_hc_mult | mhc_num_residual_streams (hc_mult) | required; assert >0 glm5next.cpp:59-62; arch.cpp:296 |
| glm5next.hyper_connection.sinkhorn_iterations | hparams.dsv4_hc_sinkhorn_iters | mhc_sinkhorn_iterations (hc_sinkhorn_iters) | required; glm5next.cpp:60; arch.cpp:297 |
| glm5next.hyper_connection.epsilon | hparams.dsv4_hc_eps | hc_eps | required; glm5next.cpp:61; arch.cpp:298 |
| glm5next.expert_feed_forward_length | hparams.n_ff_exp_arr (key_or_arr over n_layer_all) | moe_intermediate_size | required; glm5next.cpp:67; arch.cpp:200; per-layer arrays allowed |
| glm5next.expert_shared_feed_forward_length | hparams.n_ff_shexp | derived: moe_intermediate_size * n_shared_experts (UNMAPPED as a standalone key) | optional; llama derives identically when absent: n_ff_shexp = n_ff_exp() * max(1, n_expert_shared) glm5next.cpp:68,77-79; arch.cpp:201 |
| glm5next.expert_shared_count | hparams.n_expert_shared | n_shared_experts | required; glm5next.cpp:69; arch.cpp:209 |
| glm5next.leading_dense_block_count | hparams.n_layer_dense_lead | first_k_dense_replace (FreeToken derives it from mlp_layer_types, args.py:117-125; shim should map the key directly) | required; glm5next.cpp:70; arch.cpp:195 |
| glm5next.expert_weights_scale | hparams.expert_weights_scale | routed_scaling_factor | required; glm5next.cpp:71; arch.cpp:212; applied to routed weights only, shared expert added unscaled glm5next.cpp:555-568 |
| glm5next.expert_weights_norm | hparams.expert_weights_norm (bool) | norm_topk_prob | required; glm5next.cpp:72; arch.cpp:213; passed as weight-norm flag to build_moe_ffn (555) |
| glm5next.expert_gating_func | hparams.expert_gating_func (enum int) | UNMAPPED (FreeToken hardcodes sigmoid noaux_tc routing with exp_probs_b as selection bias; moe.py) | required; cast to llama_expert_gating_func_type at glm5next.cpp:557; arch.cpp:215 |
| glm5next.swiglu_clamp_exp | hparams.swiglu_clamp_exp (key_or_arr, optional) | swiglu_limit | optional; glm5next.cpp:74; arch.cpp:203; FreeToken uses it to select hidden_act="swiglu_clamp" (config.py:120-124) |
| glm5next.swiglu_clamp_shexp | hparams.swiglu_clamp_shexp (key_or_arr, optional) | UNMAPPED (FreeToken carries a single swiglu_limit for routed AND shared experts) | optional; glm5next.cpp:75; arch.cpp:204 |
| glm5next.nextn_predict_layers | hparams.n_layer_nextn | UNMAPPED (FreeToken has no MTP/draft support) | optional, absent = 0; read twice: generic llama-model.cpp:1230 and glm5next.cpp:81; n_layer() = n_layer_all - n_layer_nextn (llama-hparams.cpp:347-349) |

## B. Generic keys (read for every arch, llama-model.cpp:1200-1378)

| gguf metadata key | llama.cpp hparam field | FreeToken Glm5NextArgs field | notes |
|---|---|---|---|
| glm5next.context_length | hparams.n_ctx_train | max_position | required; llama-model.cpp:1223; arch.cpp:189 |
| glm5next.embedding_length | hparams.n_embd | hidden_size | required; llama-model.cpp:1224; arch.cpp:190 |
| glm5next.block_count | hparams.n_layer_all | num_layers = block_count - nextn_predict_layers | required; llama-model.cpp:1228; arch.cpp:194; the count INCLUDES the NextN block when present |
| glm5next.attention.head_count | hparams.n_head_arr (key_or_arr) | num_heads (DSA layers); ALSO the KDA head count and MLA head count | optional; llama-model.cpp:1297; arch.cpp:248; see "single head count" note below |
| glm5next.attention.head_count_kv | hparams.n_head_kv_arr (key_or_arr) | UNMAPPED directly, but it IS the layer-type split: 0 marks a KDA layer | optional; glm5next.cpp:87; arch.cpp:249; see split section |
| glm5next.expert_count | hparams.n_expert | num_experts (n_routed_experts) | optional; llama-model.cpp:1232; arch.cpp:207; also inferable from ffn_gate_inp dims |
| glm5next.expert_used_count | hparams.n_expert_used_arr (key_or_arr) | num_experts_per_tok | optional; llama-model.cpp:1234; arch.cpp:208; n_expert_used() consumed at glm5next.cpp:554 |
| glm5next.feed_forward_length | hparams.n_ff_arr (key_or_arr) | intermediate_size (dense layers) | optional; llama-model.cpp:1296; arch.cpp:199 |
| glm5next.attention.key_length | hparams.n_embd_head_k_full | UNMAPPED | optional, read only when n_head() > 0; llama-model.cpp:1336,1341; arch.cpp:252; MLA uses key_length_mla instead |
| glm5next.attention.value_length | hparams.n_embd_head_v_full | UNMAPPED | optional; llama-model.cpp:1344; arch.cpp:253 |
| glm5next.rope.dimension_count | hparams.n_rot | UNMAPPED (NoPE) | optional; llama-model.cpp:1349; glm5next asserts n_rot()==0 (glm5next.cpp:37); ROPE_TYPE_NONE (llama-model.cpp:2872-2874) |
| glm5next.rope.freq_base, glm5next.rope.scaling.* | rope params | UNMAPPED (FreeToken rope_theta is indexer-side only, args.py:206-209) | optional; llama-model.cpp:1316-1333; not consumed by the NoPE graph |
| glm5next.embedding_length_out | hparams.n_embd_out_impl | UNMAPPED | optional; llama-model.cpp:1225; glm5next forces it back to 0 so n_embd_out stays n_embd (glm5next.cpp:64-65) |
| glm5next.vocab_size | n_vocab (via vocab) | vocab_size | NOT read by the model loader; n_vocab comes from tokenizer.ggml.tokens count; vocab_size only read on the vocab dummy path (llama-vocab.cpp:1947); shim may use it when present |
| glm5next.attention.causal | hparams.causal_attn | (engine-level) | optional; llama-model.cpp:1226 |
| glm5next.expert_group_count / expert_group_used_count | n_expert_group / n_expert_group_used | UNMAPPED (FreeToken carries n_group/topk_group separately) | optional; llama-model.cpp:1235-1236; unused by the glm5next graph |

Note "single head count": LLAMA_LOAD_LOCALS binds `n_head = hparams.n_head()` =
`n_head_arr[0]` (llama-model.h:827-845; llama-hparams.cpp:50-56) and load_arch_tensors
uses that ONE scalar for KDA d_inner (glm5next.cpp:105-106), KDA ssm_beta width (166),
and MLA q_b/wo widths (175, 182). llama.cpp therefore assumes the MLA head count equals
the KDA head count. The FreeToken shim should assert linear_num_heads == num_heads and
fail loudly otherwise; a checkpoint where linear_attn_config.num_heads differs from
num_attention_heads would silently build wrong shapes in llama.cpp semantics.

## C. Defined but NOT read for glm5next (shim must not require)

- glm5next.kda.safe_gate (arch.cpp:343): hparams.kda_safe_gate (llama-hparams.h:192) stays
  false; absence selects the sigmoid-gate branch; presence of the key would flip kimi-k3
  style softplus (comment glm5next.cpp:43). UNMAPPED.
- glm5next.attention.indexer.block_size / .local_blocks (arch.cpp:285-286): fields
  indexer_block_size/local_blocks (llama-hparams.h:274-275) stay 0 for glm5next.
- glm5next.attention.indexer.types (arch.cpp:287): not read; is_indexer_full_impl is set
  structurally = !is_recr (glm5next.cpp:92-94). UNMAPPED; FreeToken shim should set
  indexer_types = ("full",) * n_dsa_layers (config.py:80-87 expects "full").
- glm5next.ssm.inner_size / .state_size / .time_step_rank / .group_count / .dt_b_c_rms
  (arch.cpp:336-340): mamba keys, unused.
- glm5next.ple.* (arch.cpp:301-309): no loader reads LLM_KV_PLE_*, so ple_n_heads stays 0
  and ple_conv_state() == 0 - the p_l recurrent plane is dead for glm5next
  (llama-memory-recurrent.cpp:110-115, llama-hparams.cpp:268-275).
- glm5next.attention.recurrent_layers (arch.cpp:294), glm5next.hyper_connection.low_rank
  (arch.cpp:299): not read.
- logit_scale / final_logit_softcapping: zero readers in llama-model.cpp for this arch.
- glm5next.attention.max_alibi_bias / clamp_kqv / sliding_window family: not read here.

## Dense vs MoE layer split

- Key-driven ONLY: `glm5next.leading_dense_block_count` -> n_layer_dense_lead
  (glm5next.cpp:70). Layers il < n_layer_dense_lead load dense FFN tensors
  ffn_gate/up/down (glm5next.cpp:194-197); the remaining layers get the router
  (ffn_gate_inp), exp_probs_b bias, stacked experts and shared experts (198-209).
  Tensor presence is not consulted for the decision; build_layer_ffn branches the same
  way at graph time (glm5next.cpp:539-558). Dense FFN and experts both use silu
  (LLM_FFN_SILU, 544/555); the clamp comes from the swiglu_clamp_* keys.
- FreeToken equivalent: mlp_layer_types ("dense"/"sparse") folded into
  first_k_dense_replace (args.py:107-125, config.py:106-112). GGUF has no
  mlp_layer_types; the shim derives it from leading_dense_block_count.

## KDA vs DSA layer split

- Per-layer `attention.head_count_kv` array (optional key, llama-model.cpp:1305):
  n_head_kv(il) == 0 -> KDA/recurrent layer; > 0 -> DSA/MLA layer
  (is_recr_impl[il] = n_head_kv(il) == 0, glm5next.cpp:87).
- is_indexer_full_impl[il] = !is_recr (glm5next.cpp:92-93): every DSA layer owns a full
  indexer (no shared-indexer concept for this arch).
- Validity: 0 < n_recr < n_layer asserted (glm5next.cpp:88-90) - KDA layers must be a
  strict minority; the split is read over n_layer_all but only trunk layers count (88).
- There is no first_k_dense-style key for this split; it lives entirely in the
  head_count_kv array. The shim should emit FreeToken layer_types as
  ("linear_attention" where head_count_kv == 0, "deepseek_sparse_attention" otherwise).

## MTP / NextN handling

- `glm5next.nextn_predict_layers` (optional, default 0) -> n_layer_nextn
  (llama-model.cpp:1230, glm5next.cpp:81). Assert n_layer_nextn < n_layer_all (82).
  n_layer_all = block_count; the trunk n_layer() = n_layer_all - n_layer_nextn
  (llama-hparams.cpp:347-349). So block_count in the GGUF INCLUDES the draft layer.
- The NextN layer occupies layer index n_layer (i.e. blk.{n_layer}) and is an ordinary
  DSA attention + FFN block with extra nextn.* tensors (glm5next.cpp:211-219) and NO
  hc_* mixers (assert !layer.hc_attn_fn, 708).
- File-shape probes (glm5next.cpp:120-124): `mtp_only` when blk.0.attn_norm.weight is
  absent (draft-only file) -> trunk tensors loaded with TENSOR_NOT_REQUIRED;
  `trunk_only` when blk.{n_layer}.nextn.eh_proj.weight is absent -> nextn tensors
  NOT_REQUIRED. `--no-mtp` sets TENSOR_SKIP on all nextn tensors (126-128).
- graph_mtp asserts n_layer_nextn == 1 (696) and requires eh_proj/enorm/hnorm (706-707);
  shared_head_norm/head fall back to output_norm/output when absent (786-788, 798-800).
- Draft-context cache routing: llama-model.cpp:2489-2524 - mtp_ctx is true when
  ctx_type == LLAMA_CONTEXT_TYPE_MTP && n_layer_all > n_layer(); then the attn/indexer
  caches cover only il in [n_layer, n_layer_all); the recurrent half is unused
  (s_copy still expanded, glm5next.cpp:740-742).
- FreeToken: UNMAPPED - no MTP implementation; the shim should record
  nextn_predict_layers and either require a trunk_only-shaped file or skip tensors
  blk.>= (block_count - nextn_predict_layers).

## Glm5NextArgs fields with no GGUF source (shim must default)

| FreeToken field | GGUF status | shim default |
|---|---|---|
| mla_nope | no key; implied: llama asserts n_rot()==0 (glm5next.cpp:37), ROPE_TYPE_NONE | true |
| indexer_types | no key (types key exists but unread, arch.cpp:287) | ("full",) * n_dsa_layers |
| indexer_rope_interleave | no key; indexer is rope-free ("no rope: n_rot() is 0", glm5next.cpp:405) | false |
| index_kpool_always_select_tail | no key; llama's n_select math always force-includes the tail pool (glm5next.cpp:14-17) | true |
| mhc_tau / mhc_post_mult_value / mhc_no_norm_weight | no keys (hyper_connection.* has only count/sinkhorn_iterations/epsilon/low_rank, arch.cpp:296-299) | mhc_tau 0.05, mhc_post_mult_value 2.0, mhc_no_norm_weight False (args.py:216-219) |
| swiglu_limit (shexp variant) | separate key swiglu_clamp_shexp exists but FreeToken has one scalar | alias to swiglu_limit |
| linear lower-bound default | FreeToken falls back to -5.0 (args.py:178-183); llama REQUIRES kda.gate_lower_bound | must come from the key |