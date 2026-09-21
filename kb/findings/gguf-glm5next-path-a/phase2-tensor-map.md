# Phase 2 - definitive 3-way tensor map: GGUF <-> FreeToken (HF/FTW) <-> llama.cpp

Checkpoint: `/media/ai/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL.gguf`
(arch `glm5next`, GGUF v3, 1412 tensors; mmproj is a separate file, see section 11).
FreeToken side of every target string below is exactly what
`python/freetoken/models/glm5_next/weight.py` yields (read in full for this map);
llama.cpp kinds come from `research/llamacpp-tensor-kinds.md` (glm5next.cpp).
Gemma4 adapter `python/freetoken/models/gemma4/gguf.py` is the structural pattern for
the Phase 2 Code wave (see section 9).

## 1. Layer taxonomy (ground truth from tensors.txt + metadata.txt)

- `block_count = 46` = **45 trunk layers (blk.0-44) + 1 MTP/NextN block (blk.45)**.
  `glm5next.nextn_predict_layers = 1`. Proof that blk.45 is the NextN block, not a
  trunk layer: it is the ONLY block without `hc_*` tensors (llama.cpp asserts the
  NextN layer carries none, glm5next.cpp:708), and the nextn glue tensors
  (`blk.45.nextn.*`) sit inside it. Matches `args.py` docstring: "34 of 45 decoder
  layers run KDA ... the remaining 11 run DSA".
- KDA layers (GGUF `attention.head_count_kv[i] == 0`): 34 layers =
  {0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30,32,33,34,36,37,38,40,41,42,44}.
- DSA/MLA layers (`head_count_kv > 0`): 11 trunk layers = {3,7,11,15,19,23,27,31,35,39,43}.
- Dense FFN layers: {0,1,2} (`leading_dense_block_count = 3`, mirrors HF
  `first_k_dense_replace = 3`). MoE layers: 3..44 (42 layers).
- Geometry (metadata): n_embd 4096, n_head 64, dense ffn 12288, expert_ffn 2048,
  n_expert 288, top_k 8, shared experts 1, q_lora 1536, kv_lora 512, qk_head_dim_mla 256,
  v_head_dim_mla 256, KDA head_dim 128 (d_inner = 64 heads x 128 = 8192), conv kernel 4,
  indexer: 32 heads x 128 = 4096, kpool 4, top_k 2048, mHC hc_mult 4
  (hc_dim 16384, hc_mix_dim 24), vocab 154880, NoPE (`rope.dimension_count = 0`).

Notation: shapes are GGUF ne-order (ne0 first = fastest-varying = the linear INPUT dim
for 2D weights). Torch/HF shape = reversed ne. "direct" = only ne-reversal, no transpose.

## 2. Coverage ledger (1412 = 3 + 1380 + 29)

| category | layers | tensors/layer | count | disposition |
|---|---|---|---|---|
| global (embd / output / output_norm) | - | 3 | 3 | mapped |
| KDA + dense FFN | 0,1,2 | 26 | 78 | mapped |
| KDA + MoE FFN | 31 layers (see 1) | 31 | 961 | mapped |
| DSA + MoE FFN | 3,7,...,43 | 31 | 341 | mapped |
| MTP block blk.45 (DSA + MoE, no hc) + nextn glue | 45 | 25 + 4 | 29 | SKIP-LISTED |
| **total** | | | **1412** | **mapped 1383 / skipped 29** |

Reverse-check summary: 0 gguf tensors without a mapping or skip entry; every
`weight.py`-expected param has exactly one gguf source except 5 constructed targets
(fused KDA in_proj, fused KDA conv1d, fused kv_b_proj, derived A_log, sliced expert
banks) - see sections 5, 6 and 9.

## 3. Global tensors (3)

| gguf name | type | shape (ne) | FreeToken target | llama.cpp kind | notes |
|---|---|---|---|---|---|
| token_embd.weight | Q8_0 | 4096,154880 | `model.embed_tokens.weight` | TOKEN_EMBD | direct; dequant to bf16 or native GGUF embedding (gemma4 yields packed `.qweight` + GGUFEmbedding - prefer that here, Q8_0) |
| output.weight | Q6_K | 4096,154880 | `lm_head.weight` | OUTPUT | direct; present, so head is NOT tied - no GGUFTiedLMHead duplication path needed (llama.cpp would tie only if absent) |
| output_norm.weight | F32 | 4096 | `model.norm.weight` | OUTPUT_NORM | direct, bf16 |

## 4. Per-layer common tensors (every trunk layer 0-44; 45x6 + 45x2 = 360 tensors)

| gguf name | type | shape (ne) | FreeToken target | llama.cpp kind | notes |
|---|---|---|---|---|---|
| blk.N.attn_norm.weight | F32 | 4096 | `model.layers.N.input_layernorm.weight` | ATTN_NORM | direct, bf16 |
| blk.N.ffn_norm.weight | F32 | 4096 | `model.layers.N.post_attention_layernorm.weight` | FFN_NORM | direct, bf16 |
| blk.N.hc_attn_fn.weight | Q8_0 | 16384,24 | `model.layers.N.hc_attn_fn` | HC_ATTN_FN | raw param (no `.weight` suffix in FT name); DEQUANT Q8_0 then cast fp32 (weight.py keeps mHC fp32; gguf quantized it anyway) |
| blk.N.hc_attn_base.weight | F32 | 24 | `model.layers.N.hc_attn_base` | HC_ATTN_BASE | raw param, fp32 |
| blk.N.hc_attn_scale.weight | F32 | 3 | `model.layers.N.hc_attn_scale` | HC_ATTN_SCALE | raw param, fp32 |
| blk.N.hc_ffn_fn.weight | Q8_0 | 16384,24 | `model.layers.N.hc_ffn_fn` | HC_FFN_FN | as hc_attn_fn |
| blk.N.hc_ffn_base.weight | F32 | 24 | `model.layers.N.hc_ffn_base` | HC_FFN_BASE | raw param, fp32 |
| blk.N.hc_ffn_scale.weight | F32 | 3 | `model.layers.N.hc_ffn_scale` | HC_FFN_SCALE | raw param, fp32 |

hc shapes: hc_dim = 4*4096 = 16384, hc_mix_dim = (2+4)*4 = 24 - matches
`hyper_connection.count = 4`. Head-level `output_hc_*` kinds do not exist in this file
and are unused by design (mHC head collapsed by unweighted mean).

## 5. KDA layer tensors (34 layers; base set 23 tensors/layer incl. common rows)

LLM_TENSOR kinds: ATTN_Q/ATTN_K/ATTN_V via create_tensor_qkv (optional merged
`blk.%d.attn_qkv` tried first - absent here, separate forms present), SSM_* kinds,
ATTN_OUT. `d_inner = 8192`, `head_dim = 128`, `n_head(KDA) = 64`, `d_conv = 4`.

FUSIONS REQUIRED (must mirror `Glm5NextKDA._in_proj_split`, weight.py:94 `_KDA_IN_PROJ`):

- `in_proj.weight` = cat on output axis (dim 0) of q|k|v|b|f_a|g_a pieces -> torch
  (24896, 4096) = 3x8192 + 64 + 2x128, row blocks [p,p,p,h,d,d]. (An earlier revision
  of this line said 24832 - it dropped ssm_beta's 64 rows; correct total is 24896.)
- `conv1d.weight` = cat on channel axis (dim 0) of q|k|v conv pieces -> torch (24576, 1, 4).

| gguf name (blk.N.*) | type | shape (ne) | FreeToken target | kind | notes |
|---|---|---|---|---|---|
| attn_q.weight | Q8_0 | 4096,8192 | fused piece q of `model.layers.N.self_attn.in_proj.weight` | ATTN_Q | direct -> torch (8192,4096); fusion slot 1 of 6 |
| attn_k.weight | Q8_0 | 4096,8192 | fused piece k of in_proj | ATTN_K | direct; slot 2 |
| attn_v.weight | Q8_0 | 4096,8192 | fused piece v of in_proj | ATTN_V | direct; slot 3 |
| ssm_beta.weight | Q8_0 | 4096,64 | fused piece b of in_proj | SSM_BETA | direct -> (64,4096); beta = sigmoid(b(x)); slot 4 |
| ssm_f_a.weight | Q8_0 | 4096,128 | fused piece f_a of in_proj | SSM_F_A | direct -> (128,4096); slot 5 |
| ssm_g_a.weight | Q8_0 | 4096,128 | fused piece g_a of in_proj | SSM_G_A | direct -> (128,4096); slot 6 |
| ssm_conv1d_q.weight | F32 | 4,1,8192 | fused piece q of `model.layers.N.self_attn.conv1d.weight` | SSM_CONV1D_Q | ne-reverse -> (8192,1,4); cat slot 1 of 3 |
| ssm_conv1d_k.weight | F32 | 4,1,8192 | fused piece k of conv1d | SSM_CONV1D_K | slot 2 |
| ssm_conv1d_v.weight | F32 | 4,1,8192 | fused piece v of conv1d | SSM_CONV1D_V | slot 3 |
| ssm_f_b.weight | Q8_0 | 128,8192 | `model.layers.N.self_attn.f_b_proj.weight` | SSM_F_B | direct -> (8192,128) |
| ssm_g_b.weight | Q8_0 | 128,8192 | `model.layers.N.self_attn.g_b_proj.weight` | SSM_G_B | direct |
| attn_output.weight | Q8_0 | 8192,4096 | `model.layers.N.self_attn.o_proj.weight` | ATTN_OUT | direct -> (4096,8192) |
| ssm_a | F32 | 64 | `model.layers.N.self_attn.A_log` | SSM_A | DERIVE: gguf holds -exp(A_log) per head (kimi-k3 convention, glm5next.cpp:7); A_log = log(-x), fp32, guard x < 0. Bare name, no `.weight` |
| ssm_dt.bias | F32 | 8192 | `model.layers.N.self_attn.dt_bias` | SSM_DT | rename + fp32; dt lives in a `.bias` tensor but maps to a plain param, not a Linear bias |
| ssm_norm.weight | F32 | 128 | `model.layers.N.self_attn.o_norm.weight` | SSM_NORM | KDA o_norm (RMS then plain-sigmoid gate); bf16 |

Per-KDA-block totals: 8 common + 15 rows here = 23; dense layers 0-2 add the 3 rows
of 7a -> 26; MoE layers add 5 rows of 7b + 3 rows of 7c = 8 -> 31. Both reconcile with
the ledger in section 2.

## 6. DSA/MLA layer tensors (11 trunk layers 3,7,...,43; 31 tensors each = 341)

Kinds ATTN_Q_A..ATTN_V_B (3D MLA - no attn_q/k/v, NoPE, absorbed MQA) + INDEXER_*.
`qk_head_dim = 256, v_head_dim = 256, kv_lora = 512, q_lora = 1536, n_head = 64,
indexer: 32 heads x 128 = 4096, kpool = 4`.

FUSION REQUIRED: `kv_b_proj.weight` = per-head [k|v] rows from the two 3D tensors:
k = attn_k_b ne-reversed (64,512,256) -> transpose last two axes -> (64,256,512);
v = attn_v_b ne-reversed (64,256,512) as-is; cat dim 1 -> (64,512,512) -> reshape
(32768, 512). Row order per head must match HF kv_b_proj.view(64, 512, 512) =
[k(256); v(256)] - UNIT-VERIFY against an HF checkpoint in Phase 2.

| gguf name (blk.N.*) | type | shape (ne) | FreeToken target | kind | notes |
|---|---|---|---|---|---|
| attn_q_a.weight | Q8_0 | 4096,1536 | `model.layers.N.self_attn.q_a_proj.weight` | ATTN_Q_A | direct -> (1536,4096) |
| attn_q_a_norm.weight | F32 | 1536 | `model.layers.N.self_attn.q_a_layernorm.weight` | ATTN_Q_A_NORM | direct, bf16 |
| attn_q_b.weight | Q8_0 | 1536,16384 | `model.layers.N.self_attn.q_b_proj.weight` | ATTN_Q_B | direct -> (16384,1536) |
| attn_kv_a_mqa.weight | Q8_0 | 4096,512 | `model.layers.N.self_attn.kv_a_proj_with_mqa.weight` | ATTN_KV_A_MQA | direct -> (512,4096) |
| attn_kv_a_norm.weight | F32 | 512 | `model.layers.N.self_attn.kv_a_layernorm.weight` | ATTN_KV_A_NORM | direct, bf16 |
| attn_k_b.weight | Q8_0 | 256,512,64 | fused piece k of `model.layers.N.self_attn.kv_b_proj.weight` | ATTN_K_B | 3D; torch (64,512,256); TRANSPOSE last two axes -> (64,256,512); slot 1 of 2 |
| attn_v_b.weight | Q8_0 | 512,256,64 | fused piece v of kv_b_proj | ATTN_V_B | 3D; torch (64,256,512) already [head, v_dim, kv_lora]; slot 2; NO transpose |
| attn_output.weight | Q8_0 | 16384,4096 | `model.layers.N.self_attn.o_proj.weight` | ATTN_OUT | direct -> (4096,16384) |
| indexer.attn_k.weight | Q8_0 | 4096,128 | `model.layers.N.self_attn.indexer.wk.weight` | INDEXER_ATTN_K | direct -> (128,4096), bf16 |
| indexer.attn_q_b.weight | Q8_0 | 1536,4096 | `model.layers.N.self_attn.indexer.wq_b.weight` | INDEXER_ATTN_Q_B | direct -> (4096,1536), bf16 |
| indexer.proj.weight | F32 | 4096,32 | `model.layers.N.self_attn.indexer.weights_proj.weight` | INDEXER_PROJ | direct -> (32,4096), bf16; head weights run PREC_F32 in llama.cpp |
| indexer.k_norm.weight | F32 | 128 | `model.layers.N.self_attn.indexer.k_norm.weight` | INDEXER_K_NORM | LayerNorm weight (NOT RMS), bf16 |
| indexer.k_norm.bias | F32 | 128 | `model.layers.N.self_attn.indexer.k_norm.bias` | INDEXER_K_NORM | bias side must be loaded (llama.cpp asserts it, glm5next.cpp:330) |
| indexer_compressor_gate.weight | Q8_0 | 4096,128 | `model.layers.N.self_attn.indexer.index_kpool_compress_gate` | INDEXER_COMPRESSOR_WGATE | direct -> (128,4096), bf16; RAW param name (no `.weight` suffix in FT); underscore spelling in gguf |
| indexer_compressor_ape.weight | F32 | 128,4 | `model.layers.N.self_attn.indexer.index_kpool_compress_ape` | INDEXER_COMPRESSOR_APE | ne-reverse -> (4,128) = [kpool, head_dim], fp32; RAW param; added pre-softmax over pool slots |

Per-DSA-block totals: 8 common + 15 rows here + 5 router/shared (7b) + 3 banks (7c) = 31.

## 7. FFN tensors

### 7a. Dense FFN, layers 0-2 only (9 tensors)

| gguf name | type | shape (ne) | FreeToken target | kind | notes |
|---|---|---|---|---|---|
| blk.N.ffn_gate.weight | Q8_0 | 4096,12288 | `model.layers.N.mlp.gate_proj.weight` | FFN_GATE | direct -> (12288,4096); Q8_0 native or dequant |
| blk.N.ffn_up.weight | Q8_0 | 4096,12288 | `model.layers.N.mlp.up_proj.weight` | FFN_UP | direct |
| blk.N.ffn_down.weight | Q8_0 | 12288,4096 | `model.layers.N.mlp.down_proj.weight` | FFN_DOWN | direct -> (4096,12288) |

### 7b. MoE router + shared expert, every MoE layer 3-44 (5 tensors x 42 = 210)

| gguf name | type | shape (ne) | FreeToken target | kind | notes |
|---|---|---|---|---|---|
| blk.N.ffn_gate_inp.weight | F32 | 4096,288 | `model.layers.N.mlp.gate.weight` | FFN_GATE_INP | router; direct -> (288,4096), bf16 |
| blk.N.exp_probs_b.bias | F32 | 288 | `model.layers.N.mlp.e_score_correction_bias` | FFN_EXP_PROBS_B | noaux_tc selection bias; RAW param; fp32 (weight.py: bf16 cast would perturb top-8) |
| blk.N.ffn_gate_shexp.weight | Q8_0 | 4096,2048 | `model.layers.N.mlp.shared_experts.gate_proj.weight` | FFN_GATE_SHEXP | shared expert; direct -> (2048,4096) |
| blk.N.ffn_up_shexp.weight | Q8_0 | 4096,2048 | `model.layers.N.mlp.shared_experts.up_proj.weight` | FFN_UP_SHEXP | direct |
| blk.N.ffn_down_shexp.weight | Q8_0 | 2048,4096 | `model.layers.N.mlp.shared_experts.down_proj.weight` | FFN_DOWN_SHEXP | direct -> (4096,2048) |

### 7c. Routed expert banks, every MoE layer 3-44 (3 tensors x 42 = 126)

NOT yielded by `iter_weights` on the HF path (asserted: routed experts serve only from
the offload cache). Mirror gemma4: skip them inside `iter_gguf_weights` (an
`_EXPERT_SUFFIXES`-style tuple: ffn_gate_exps.weight / ffn_up_exps.weight /
ffn_down_exps.weight) and feed the offload bank loader directly. Stacked layout:
gate/up ne (n_embd, n_ff_exp, n_expert) -> torch (288, 2048, 4096); down ne
(n_ff_exp, n_embd, n_expert) -> torch (288, 4096, 2048). Per-expert pieces are dim-0
slices, so the bank loader needs no repack - just slice.

| gguf name | type | shape (ne) | FreeToken target | kind | notes |
|---|---|---|---|---|---|
| blk.N.ffn_gate_exps.weight | IQ3_XXS / IQ4_XS | 4096,2048,288 | routed gate bank; per-expert gate_proj (2048,4096) | FFN_GATE_EXPS | IQ3_XXS on 41 layers; IQ4_XS on layer 11 only |
| blk.N.ffn_up_exps.weight | IQ3_XXS / IQ4_XS | 4096,2048,288 | routed up bank; per-expert up_proj | FFN_UP_EXPS | same layer split as gate |
| blk.N.ffn_down_exps.weight | IQ4_XS / Q6_K | 2048,4096,288 | routed down bank; per-expert down_proj (4096,2048) | FFN_DOWN_EXPS | IQ4_XS on 39 layers; Q6_K on layers 11, 12, 44 |

Per-layer expert quant map (trunk MoE only):

| layers | gate_exps | up_exps | down_exps |
|---|---|---|---|
| 3-10 and 13-43 (39 layers) | IQ3_XXS | IQ3_XXS | IQ4_XS |
| 11 | IQ4_XS | IQ4_XS | Q6_K |
| 12, 44 | IQ3_XXS | IQ3_XXS | Q6_K |

## 8. MTP / NextN resolution and skip-list (29 tensors)

**The file DOES carry the full MTP layer.** `glm5next.nextn_predict_layers = 1`; the
NextN block is `blk.45` (llama.cpp: n_layer = block_count - nextn_predict_layers = 45,
NextN at il = n_layer): a complete DSA attention + MoE FFN + indexer block (no hc_*),
plus the nextn glue tensors stored INSIDE blk.45 (`blk.45.nextn.*`, not a separate
blk.46). Trunk = 45 layers (blk.0-44); total blocks = 46.

Verdict: **IGNORE, do not fail.** FreeToken has no MTP/draft support, and the HF path
already ignores it (`weight.py` never reads the trailing MTP layer; docstring: "the
trailing MTP layer is never read"; `_layer_to_bank` maps layer >= num_layers to None).
Recommended handling: `parse_gguf_config` sets `num_layers = block_count -
nextn_predict_layers = 45`, SLICES per-layer arrays (`attention.head_count_kv`,
`swiglu_clamp_exp`, `swiglu_clamp_shexp` are len 46) to 45 entries, and
`iter_gguf_weights` skips everything under `blk.45.` (both the block tensors and the
`.nextn.` glue) with an explicit skip log.

| gguf name (blk.45.*) | type | shape (ne) | would-be llama.cpp kind | reason skipped |
|---|---|---|---|---|
| attn_k_b.weight | Q8_0 | 256,512,64 | ATTN_K_B (NextN layer) | MTP block weight; no MTP support |
| attn_kv_a_mqa.weight | Q8_0 | 4096,512 | ATTN_KV_A_MQA | MTP block weight |
| attn_kv_a_norm.weight | F32 | 512 | ATTN_KV_A_NORM | MTP block weight |
| attn_norm.weight | F32 | 4096 | ATTN_NORM | MTP block weight |
| attn_output.weight | Q8_0 | 16384,4096 | ATTN_OUT | MTP block weight |
| attn_q_a.weight | Q8_0 | 4096,1536 | ATTN_Q_A | MTP block weight |
| attn_q_a_norm.weight | F32 | 1536 | ATTN_Q_A_NORM | MTP block weight |
| attn_q_b.weight | Q8_0 | 1536,16384 | ATTN_Q_B | MTP block weight |
| attn_v_b.weight | Q8_0 | 512,256,64 | ATTN_V_B | MTP block weight |
| exp_probs_b.bias | F32 | 288 | FFN_EXP_PROBS_B | MTP block weight |
| ffn_down_exps.weight | Q4_K | 2048,4096,288 | FFN_DOWN_EXPS | MTP block weight (also: Q4_K bank kind nowhere else in file) |
| ffn_down_shexp.weight | Q8_0 | 2048,4096 | FFN_DOWN_SHEXP | MTP block weight |
| ffn_gate_exps.weight | Q3_K | 4096,2048,288 | FFN_GATE_EXPS | MTP block weight |
| ffn_gate_inp.weight | F32 | 4096,288 | FFN_GATE_INP | MTP block weight |
| ffn_gate_shexp.weight | Q8_0 | 4096,2048 | FFN_GATE_SHEXP | MTP block weight |
| ffn_norm.weight | F32 | 4096 | FFN_NORM | MTP block weight |
| ffn_up_exps.weight | Q3_K | 4096,2048,288 | FFN_UP_EXPS | MTP block weight |
| ffn_up_shexp.weight | Q8_0 | 4096,2048 | FFN_UP_SHEXP | MTP block weight |
| indexer.attn_k.weight | Q8_0 | 4096,128 | INDEXER_ATTN_K | MTP block weight |
| indexer.attn_q_b.weight | Q8_0 | 1536,4096 | INDEXER_ATTN_Q_B | MTP block weight |
| indexer.k_norm.bias | F32 | 128 | INDEXER_K_NORM | MTP block weight |
| indexer.k_norm.weight | F32 | 128 | INDEXER_K_NORM | MTP block weight |
| indexer.proj.weight | F32 | 4096,32 | INDEXER_PROJ | MTP block weight |
| indexer_compressor_ape.weight | F32 | 128,4 | INDEXER_COMPRESSOR_APE | MTP block weight |
| indexer_compressor_gate.weight | Q8_0 | 4096,128 | INDEXER_COMPRESSOR_WGATE | MTP block weight |
| nextn.eh_proj.weight | Q8_0 | 8192,4096 | NEXTN_EH_PROJ | MTP glue (concat of [enorm(tok), hnorm(h)]) |
| nextn.enorm.weight | F32 | 4096 | NEXTN_ENORM | MTP glue |
| nextn.hnorm.weight | F32 | 4096 | NEXTN_HNORM | MTP glue |
| nextn.shared_head_norm.weight | F32 | 4096 | NEXTN_SHARED_HEAD_NORM | MTP glue (falls back to output_norm when absent upstream) |

Not present (and NOT required): `nextn.embed_tokens.weight` and
`nextn.shared_head_head.weight` - upstream marks both NOT_REQUIRED with fallbacks to
token_embd / output.

## 9. Reverse check (zero unmatched is the acceptance criterion)

Every gguf tensor -> target: 1383 mapped via sections 3-7, 29 skip-listed via section 8.
Quant-type cross-foot (matches header): Q8_0 = 1 token_embd + 42 dense-KDA + 434 MoE-KDA
+ 154 DSA + 13 MTP = 644; F32 = 36 + 434 + 154 + 14 = 638; IQ3_XXS 82; IQ4_XS 41;
Q6_K 4 (output + 3 downs); Q3_K 2 and Q4_K 1 (MTP only). Sum 1412.

FreeToken-side tensors and how they are satisfied:

- 1:1 rename from gguf: embed_tokens, lm_head, model.norm, input/post layernorms,
  o_proj (both types), q_a/q_b/kv_a/q_a_norm/kv_a_norm, f_b/g_b, o_norm, indexer
  wk/wq_b/weights_proj/k_norm.{weight,bias}, hc_* (6), mlp.gate,
  e_score_correction_bias, shared_experts.{gate,up,down}_proj, dense mlp gate/up/down.
- FUSED (multiple gguf -> 1 FT tensor, no gguf equivalent):
  `self_attn.in_proj.weight` <- attn_q + attn_k + attn_v + ssm_beta + ssm_f_a + ssm_g_a
  (order q|k|v|b|f_a|g_a); `self_attn.conv1d.weight` <- ssm_conv1d_{q,k,v} (channel axis);
  `self_attn.kv_b_proj.weight` <- attn_k_b (transposed) + attn_v_b.
- DERIVED (no gguf source): `self_attn.A_log` = log(-ssm_a) elementwise, fp32.
- RENAMED-with-suffix-change: `self_attn.dt_bias` <- ssm_dt.bias.
- SLICED: per-expert bank pieces from the 3 stacked ffn_*_exps tensors (dim 0).
- GGUF tensors with NO FreeToken target: exactly the 29 section-8 tensors. Nothing else.
- FT params with NO gguf source: exactly the 3 fused + 1 derived above (plus bank
  slicing), all constructible in the adapter. No FT param is left unsourced.

## 10. Quant-type summary per category (feeds Phase 4/5 dispatch)

| category | quants | counts | dispatch note |
|---|---|---|---|
| routed expert banks | IQ3_XXS (gate/up), IQ4_XS (down), Q6_K (3 downs) | 82 + 41 + 3 (trunk) | heterogeneous PER LAYER (3 quant kinds mix); kernel status: IQ4_XS/IQ3_XXS dequant + MMVQ dense/moe_vec in-tree, but NO MMQ and NO python-side wiring (dequant.py lacks iq/Q3_K/Q4_K entries; layers/gguf.py exposes only Q4_0/Q8_0/Q6_K) |
| dense linears (all attn projections, dense ffn 0-2, shared experts, hc_*_fn, indexer.attn_k/attn_q_b/compressor_gate, token_embd) | Q8_0 | 644 total incl. embd + eh_proj | Q8_0 fully covered (dequant + MMQ); gemma4-style packed `.qweight` + GGUFLinear is the natural path |
| lm_head | Q6_K | 1 | dequant or native; covered |
| norms / ssm scalars / router / indexer F32 | F32 | 638 | cast per weight.py: layer norms + o_norm -> bf16; A_log, dt_bias, hc_*, e_score_correction_bias, ape -> fp32; indexer proj/wk/wq_b -> bf16 |
| MTP-only banks | Q3_K (gate/up), Q4_K (down) | 2 + 1 | skipped with the MTP block; do NOT treat Q3_K/Q4_K as required bank kinds |

## 11. mmproj out-of-scope note

348 vision tensors live in a SEPARATE file (`GLM-5.3-Flash-GGUF/mmproj-BF16.gguf`, arch
`clip`; dump: tensors-mmproj.txt in this folder). The main gguf contains zero vision
tensors; llama.cpp builds the tower in a separate clip_graph_glm5next. Text-only scope
for the Phase 2 adapter - nothing to map or skip here; just never open the mmproj file.

## 12. Naming quirks to carry into the translator

1. `ssm_a` is bare (no `.weight`) and holds **-exp(A_log)** per head -> derive
   `A_log = log(-x)` (fp32, guard x < 0). Do not copy it through as A_log.
2. `ssm_dt.bias` - dt is stored as a bias-suffixed tensor but maps to a plain fp32
   param `dt_bias` (no Linear behind it).
3. 3D MLA tensors: `attn_k_b` ne (qk_head, kv_lora, n_head) vs `attn_v_b` ne
   (kv_lora, v_head, n_head) - the axis ORDER DIFFERS between the two; k_b needs the
   last-two-axes transpose, v_b does not. llama.cpp has_3d_mla special-cases these.
4. `indexer_compressor_{ape,gate}` use UNDERSCORE spellings while the rest of the
   indexer uses dots (`indexer.*`) - regex must accept both; FT names are
   `indexer.index_kpool_compress_{gate,ape}` and are RAW params (no `.weight`).
5. `indexer.k_norm` has BOTH weight and bias (LayerNorm, not RMS); the bias must be
   loaded (llama.cpp asserts its existence).
6. Head-level `output_hc_*` HC kinds: absent from the file, unused by design (mHC head
   collapsed by unweighted mean; initial hc streams repeat the embedding). No mapping.
7. FT hc_* / e_score_correction_bias / index_kpool_compress_* / A_log / dt_bias are
   raw parameters: gguf names ending in `.weight`/`.bias` must have the suffix
   stripped when yielding.
8. `exp_probs_b.bias` -> `mlp.e_score_correction_bias` (keep fp32).
9. No rope tensors at all (NoPE main attention, `rope.dimension_count = 0`); nothing to
   skip, just absent - unlike gemma4 there is no `rope_freqs.weight` to filter.
10. Packing deltas: the only non-trivial packing work is (a) KDA in_proj 6-way concat,
    (b) KDA conv1d 3-way concat, (c) kv_b 2-way fusion with one transpose, (d) expert
    bank dim-0 slicing. Everything else is ne-reversal only. The optional merged
    `attn_qkv` / fused gate_up forms do NOT exist in this file (separate tensors only),
    so gemma4's packed-row qkv/gate_up fusion is not needed here - but the fusion-
    buffer pattern is, for in_proj/conv/kv_b.
11. `blk.45.nextn.*` sits INSIDE the last block index; a naive `blk.<N>.` regex split
    would misroute it. Skip by prefix `blk.45.` (or by checking `.nextn.` membership)
    and cut num_layers to 45.
12. output.weight is present -> head is NOT tied (no GGUFTiedLMHead duplication).

## 13. Implementation notes (mirroring gemma4/gguf.py)

`parse_gguf_config`: read `glm5next.*` keys (g(key) helper pattern), num_layers =
block_count - nextn_predict_layers; build layer_types from `attention.head_count_kv`
(0 = KDA / linear_attention, >0 = DSA / deepseek_sparse_attention) sliced to 45;
first_k_dense_replace = leading_dense_block_count; carry kda.head_dim, ssm.conv_kernel,
hyper_connection.*, indexer.* into Glm5NextArgs equivalents (mla_use_nope = true,
qk_rope_head_dim = 0).

`iter_gguf_weights`: TP=1 assert (`_require_tp1` pattern); skip `_EXPERT_SUFFIXES =
(ffn_gate_exps.weight, ffn_up_exps.weight, ffn_down_exps.weight)` for the bank loader;
strip/skip `blk.45.`; scalar map for norms/hc (with fp32 vs bf16 casts as in section
4/10 - note gemma4 casts everything bf16, glm5_next needs fp32 for hc/A_log/dt_bias/
e_score_correction_bias/ape); fusion buffers for in_proj (6 slots), conv1d (3 slots),
kv_b (2 slots, one transpose), emitted when complete with an `assert not buf` tail
(gemma4 pattern). Dense Q8_0 linears: yield packed `.qweight` + swap in GGUFLinear
(convert_*_to_gguf pattern) - same as gemma4.

## 14. Top risks for the Phase 2 Code wave

1. **kv_b_proj fusion axis**: attn_k_b requires a last-two-axes transpose and a strict
   per-head [k|v] row order to match HF kv_b_proj.view(64,512,512); a wrong axis
   corrupts every DSA layer SILENTLY. Add a unit test vs an HF checkpoint shard.
2. **Fusion order coupling**: in_proj order q|k|v|b|f_a|g_a and conv order q|k|v are
   contractually pinned to `Glm5NextKDA._in_proj_split` (weight.py:94) and the conv
   split; changing one side silently scrambles KDA.
3. **A_log derivation**: ssm_a holds -exp(A_log); a wrong sign or a non-fp32 cast moves
   every KDA decay rate. Guard x < 0 and test round-trip exp(-exp(A_log)).
4. **Config resolution off-by-one**: block_count 46 vs 45 trunk layers; unsliced
   len-46 arrays (head_count_kv, swiglu_clamp_*) will misalign the last layer and the
   KDA/DSA split; blk.45 must be excluded before any per-layer regex dispatch.
5. **Heterogeneous expert quant**: 3 quant kinds across the 42 MoE layers (IQ3_XXS /
   IQ4_XS / Q6_K downs) - bank loader and Phase 4/5 kernels must dispatch per layer;
   today python-side wiring exists only for Q4_0/Q8_0/Q6_K dense and IQ kernels are
   CUDA-MMVQ-only. The Q6_K downs (layers 11,12,44) must not fall into the IQ path.
