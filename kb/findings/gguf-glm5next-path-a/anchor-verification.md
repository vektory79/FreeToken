# Anchor verification: glm5next GGUF native serving (Path A)

Read-only re-verification of PLAN.md anchor facts against the tree at
`/media/ai/src/FreeToken` (read-only, nothing modified). All paths relative to
`python/freetoken/` unless noted. Verification is current-source, not memory.

Verdicts: 1-2 VERIFIED (lines unchanged), 3 DRIFTED (major), 4-6 VERIFIED, 7 DRIFTED.

---

## 1. models/gguf/config.py (96 lines) - VERIFIED

- `GGUF_ARCH_TO_REGISTRY` at **config.py:19-21**:
  ```python
  GGUF_ARCH_TO_REGISTRY: dict[str, str] = {
      "gemma4": "Gemma4GGUFForCausalLM",
  }
  ```
  Whitelist still has exactly one entry.
- `build_gguf_shim(model_path: str) -> GgufConfigShim` at **config.py:61-93**.
- ValueError raised at **config.py:65** (message spans 66-67):
  ```python
  raise ValueError(
      f"GGUF architecture {arch!r} is not supported "
      f"(known: {sorted(GGUF_ARCH_TO_REGISTRY)})"
  )
  ```
  -> renders as `GGUF architecture 'glm5next' is not supported (known: ['gemma4'])`.
  Line 65 unchanged from the plan.
- What it returns for a supported arch: `GgufConfigShim` (frozen dataclass, **config.py:24-43**)
  constructed at config.py:88-94 with
  `architectures=[registry_key]` (so the registry dispatches on `architectures[0]`,
  e.g. `"Gemma4GGUFForCausalLM"`), `model_path=model_path`, `model_type=arch`
  (the GGUF `general.architecture` string), `metadata=load_gguf_metadata(model_path)`,
  `vocab_size=_vocab_size(model_path)` (config.py:46-58: `token_embd.weight`
  `shape[-1]`, fallback `len(tokenizer.ggml.tokens)` for metadata-only GGUF),
  `tie_word_embeddings` = `"output.weight" not in names` (config.py:73) or the
  `OUTPUT_WEIGHT_PRESENT_KV` fallback for metadata-only GGUFs (config.py:74-86).
- `GgufConfigShim.to_dict()` (config.py:33-43) emits a minimal HF-like dict with
  `torch_dtype: "bfloat16"` - this is what trunk code (e.g. `--dtype auto`) reads.
- Arch-specific `parse_gguf_config` reads `shim.metadata` keys
  (`<arch>.block_count`, `embedding_length`, ...) to build the FreeToken ModelConfig.

## 2. Detection chain - VERIFIED, line numbers unchanged

- `python/freetoken/server/args.py:821`:
  `cfg = cached_load_hf_config(kwargs["model_path"]).to_dict()`
  (import at args.py:819). This sits inside `parse_args` (def at **args.py:132**,
  function spans 132-839), in the `"auto"` dtype-resolution block (args.py:813-830).
  Two earlier non-fatal callers exist (args.py:184-186, 233-235, tool-call /
  reasoning parser inference); 821 is the boot-path one.
- `python/freetoken/utils/hf.py:209` `def cached_load_hf_config(model_path)`;
  GGUF branch: line 212 imports `gguf_config_source`, line 214
  `if (gguf_src := gguf_config_source(model_path)) is not None:`, line 216 imports
  `build_gguf_shim`, **line 218 `return build_gguf_shim(gguf_src)`**.
- `python/freetoken/models/gguf/reader.py:42` `def gguf_config_source(model_path) -> str | None`
  decides that a path is GGUF.
- Chain: `server/args.py:821 -> utils/hf.py:214-218 -> models/gguf/config.py:61-93`
  (raise at 65). All three anchors match the plan exactly.

## 3. CUDA gguf kernels - DRIFTED: IQ4_XS is now PRESENT

`kernel/csrc/gguf/gguf_kernel.cu` is 848 lines (vLLM port + FreeToken pybind block
at gguf_kernel.cu:841-847 binding 6 ops). All dispatch is by ggml type id in
`switch(type)`. IQ4_XS is **not absent anymore** - it was added to the vendored
kernels (project-wide case-sensitive search for `IQ4_XS` finds nothing because the
vendored code spells everything lowercase `iq4_xs`):

- Dequant (`dequantize.cuh:540-583`, `ggml_get_to_cuda`): q4_0(2), q4_1(3),
  q5_0(6), q5_1(7), q8_0(8), q2_K(10), q3_K(11), q4_K(12), q5_K(13), q6_K(14),
  iq2_xxs(16), iq2_xs(17), iq3_xxs(18), iq1_s(19), iq4_nl(20), iq3_s(21),
  iq2_s(22), **iq4_xs(23)**, iq1_m(29); `default: nullptr`.
  IQ4_XS symbols: `block_iq4_xs` struct ggml-common.h:192;
  `vec_dot_iq4_xs_q8_1` vecdotq.cuh:2017;
  `dequantize_block_iq4_xs` dequantize.cuh:431; `dequantize_row_iq4_xs_cuda`
  dequantize.cuh:534 (dispatched at :577).
- MMVQ dense (`ggml_mul_mat_vec_a8` switch): case 2,3,6,7,8 (q4_0/q4_1/q5_0/q5_1/q8_0),
  10-14 (q2_K..q6_K), 16 iq2_xxs, 17 iq2_xs, 18 iq3_xxs, 19 iq1_s, 20 iq4_nl,
  21 iq3_s, 22 iq2_s, **case 23 iq4_xs at gguf_kernel.cu:179-181 ->
  `mul_mat_vec_iq4_xs_q8_1_cuda` (def mmvq.cuh:323, dispatch mmvq.cuh:334)**,
  29 iq1_m.
- MMQ dense (`ggml_mul_mat_a8` switch): cases 2,3,6,7,8,10-14 ONLY (last case 14 at
  gguf_kernel.cu:318; the switch ends there). No iq format of any kind in MMQ.
- MMQ grouped experts (`ggml_moe_a8` switch): cases 2,3,6,7,8,10-14 ONLY
  (last case 14 at gguf_kernel.cu:518). No iq formats.
- MMVQ grouped experts (`ggml_moe_a8_vec` switch, def at :541): cases 2,3,6,7,8,
  10-14 plus 16-23 and 29; **case 23 iq4_xs at gguf_kernel.cu:781-782 ->
  `moe_vec_iq4_xs_q8_1_cuda` (def moe_vec.cuh:378, dispatch moe_vec.cuh:392)**.
- `ggml_moe_get_block_size` (gguf_kernel.cu:812-835): cases 2,3,6,7,8,10-14 only;
  returns 0 for every iq type incl. 23 (iq4_xs) - matters only for the
  sorted-token MMQ-MoE path, not for the MMVQ path gemma4 uses.

Per-format coverage for the GLM-5.3-Flash-UD-Q3_K_XL expert formats:

| format | dequant | MMVQ | moe_vec (grouped GEMV) | MMQ | moe MMQ | moe_get_block_size |
|---|---|---|---|---|---|---|
| IQ4_XS (23) | yes | yes | yes | no | no | returns 0 |
| IQ3_XXS (18) | yes | yes | yes | no | no | returns 0 |
| Q3_K (11), Q4_K (12), Q6_K (14), Q8_0 (8) | yes | yes | yes | yes | yes | yes |

Conclusion: the "port IQ4_XS kernels" blocker from the plan is RESOLVED for the
decode/offload (GEMV) paths; what remains kernel-side is only iq-family MMQ
(large-batch prefill) which moe_vec does not need.

### kernel/gguf.py (142 lines) - CUDA symbol wrapping

- Ops are exposed generically with an int `quant_type` arg (no per-format Python
  functions): `_module()` compiles gguf_kernel.cu via
  `torch.utils.cpp_extension.load` into `freetoken_gguf_kernels` (kernel/gguf.py:50-73;
  host-compiler selection 25-48). Wrappers: `ggml_dequantize` (79-83),
  `ggml_mul_mat_vec_a8` (86-90), `ggml_mul_mat_a8` (93-97), `ggml_moe_a8`
  (100-115), `ggml_moe_a8_vec` (118-128), `ggml_moe_get_block_size` (131-132).

### layers/gguf.py (128 lines) - the torch-level quant ops and their gate

- `fused_mul_mat_gguf(x, qweight, qweight_type)` (layers/gguf.py:43-65): dispatches
  unquantized -> plain matmul; `x.shape[0] <= _MMVQ_SAFE(=6)` -> `ggml_mul_mat_vec_a8`;
  else MMQ `ggml_mul_mat_a8`; else dequant+matmul fallback.
- Format gate at **layers/gguf.py:33-40**:
  `_UNQUANTIZED = {GGML_F32, GGML_F16, GGML_BF16}`;
  `_MMVQ = _MMQ = _DEQUANT = {GGML_Q4_0(2), GGML_Q8_0(8), GGML_Q6_K(14)}`.
  So the Python surface exposes ONLY Q4_0 / Q8_0 / Q6_K (+ F32/F16/BF16) even
  though the CUDA switches cover more - new formats must be added here and to
  `models/gguf/dequant.py` (BLOCK_SHAPE / GGML_NAME / row_bytes).
  Note Q8_0 IS in all three sets now (layers/gguf.py:35-37) and Q8_0 has a torch
  BLOCK_SHAPE entry (dequant.py:36) - the old #358 "_DEQUANT missing Q8_0" state
  is no longer true in-tree; host-side `_DEQUANT` (dequant.py:118-121) still
  implements only q4_0 + q6_k reference dequant.
- `GGUFLinear` (68-88) and `GGUFEmbedding` (91-125): packed `[out, row_bytes]`
  uint8 weights, dequant inside the kernels; `GGUFEmbedding` gathers packed rows
  per lookup and calls `ggml_dequantize` (layers/gguf.py:114-125).
- How gemma4 uses it: `convert_gemma4_to_gguf` (models/gemma4/gguf.py:340-375)
  swaps `qkv_proj` / `o_proj` / shared-MLP `gate_up_proj` / `down_proj` to
  `GGUFLinear` (GGML_Q4_0) and `embed_tokens` to `GGUFEmbedding` (GGML_Q6_K);
  tied LM head `GGUFTiedLMHead` (gemma4/gguf.py:311-337) calls
  `fused_mul_mat_gguf` against the Q6_K embedding table.
  Routed experts (offload banks, Q4_0): `fused_experts_gguf_q4_0`
  (moe/fused_q4_0.py:22-51) calls `ggml_moe_a8_vec` twice (gate_up then down,
  qt=GGML_Q4_0), consumed from `layers/moe.py:423-426`; the CPU-GEMV twin is
  `CpuMoeExecutor._resolve_q4_0_banks` (moe/cpu_executor.py:407-432).

---

## 4. python/freetoken/models/glm5_next/ inventory - VERIFIED

Files: `__init__.py`, `args.py`, `attention.py`, `config.py`, `kda.py`, `mlp.py`,
`model.py`, `moe.py`, `weight.py` (9 modules; no gguf.py).

### Glm5NextArgs (models/glm5_next/args.py:32-96, frozen dataclass)

Field list (args.py:34-76):
- generic: `hidden_size` (34), `num_heads` (35)
- MLA (11 DSA layers): `q_lora_rank` (37), `kv_lora_rank` (38),
  `qk_nope_head_dim` (39), `qk_rope_head_dim` (40; 0 = NoPE main attention),
  `v_head_dim` (41), `mla_nope` (42), `norm_eps` (43), `max_position` (44)
- DSA indexer: `index_n_heads` (46), `index_head_dim` (47), `index_topk` (48),
  `indexer_types` (49), `indexer_rope_interleave` (50), `index_kpool` (55),
  `index_kpool_compress` (56), `index_kpool_always_select_tail` (57)
- KDA linear attention (34 layers): `linear_num_heads` (59), `linear_head_dim` (60),
  `linear_conv_kernel_dim` (61), `linear_lower_bound` (62)
- layout: `layer_types` (64), `mlp_layer_types` (65)
- mHC: `mhc` (67), `mhc_num_residual_streams` (68), `hc_eps` (69),
  `mhc_sinkhorn_iterations` (70), `mhc_tau` (71), `mhc_post_mult_value` (72),
  `mhc_no_norm_weight` (73)
- misc: `swiglu_limit` (75), `rope_theta` (76; indexer-side only)

Properties: `qk_head_dim` (78-80), `latent_dim` (82-85; kv_lora_rank + qk_rope_head_dim),
`kda_layer_ids` (87-89), `dsa_layer_ids` (91-93), `is_kda_layer` (95-96).
Constants: `KDA_LAYER = "linear_attention"` (args.py:28),
`DSA_LAYER = "deepseek_sparse_attention"` (args.py:29).
`_require_mhc` (99-109) raises NotImplementedError for mhc=False.
`load_args(hf_config)` (116-222) builds it from the checkpoint config: reads nested
`text_config`, validates `layer_types`, derives `mlp_layer_types` from
`first_k_dense_replace` when absent, folds aliases (`mla_use_nope`/`mla_nope`,
`hc_mult`/`mhc_num_residual_streams`, `hc_sinkhorn_iters`/
`mhc_sinkhorn_iterations`), unpacks the nested `linear_attn_config` dict
(num_heads / head_dim / short_conv_kernel_size / gate_lower_bound, args.py:147-177),
and takes `rope_theta` from `rope_parameters` (args.py:180-182).
A GGUF adapter must fill all of this from GGUF KV metadata (llama.cpp `glm5next`
keys: ssm.* for KDA, indexer.* / attention.* for DSA, hc_* for mHC).

### weight.py tensor-name patterns (models/glm5_next/weight.py:1-279)

- Prefix rename `_CKPT="model.language_model"` -> `_MODEL="model"` (weight.py:44-45);
  safetensors only (HF NVFP4 / RedHatAI compressed-tensors / zai-org block-fp8,
  docstring 1-31), TP=1 asserted (iter_weights, weight.py:171-181),
  `assert not include_moe_experts` (weight.py:163-166) - routed experts only ever
  go to the offload cache.
- KDA layer (weight.py:117-136, `_iter_kda_layer`): source
  `model.language_model.layers.N.self_attn.{q,k,v,b,f_a,g_a}_proj.weight` fused into
  `self_attn.in_proj.weight` (order `_KDA_IN_PROJ` weight.py:94),
  `{q,k,v}_conv1d.weight` fused into `self_attn.conv1d.weight`, plus
  `f_b_proj`/`g_b_proj`/`o_proj`, fp32 `A_log` / `dt_bias`, `o_norm.weight`.
- DSA layer (weight.py:139-157, `_iter_dsa_layer`): `q_a_proj`, `q_b_proj`,
  `kv_a_proj_with_mqa`, `kv_b_proj`, `o_proj`, `q_a_layernorm`, `kv_a_layernorm`,
  indexer `wq_b` / `wk` / `weights_proj` (weight), `indexer.k_norm.{weight,bias}`,
  `indexer.index_kpool_compress_gate` (bf16),
  `indexer.index_kpool_compress_ape` (fp32).
- mHC per layer (weight.py:197-199): `hc_attn_fn`, `hc_attn_base`, `hc_attn_scale`,
  `hc_ffn_fn`, `hc_ffn_base`, `hc_ffn_scale`, all fp32.
- norms: `input_layernorm` / `post_attention_layernorm` (weight.py:201-207);
  dense layers (< first_k_dense_replace): `mlp.{gate,up,down}_proj` (weight.py:208-211);
  sparse layers: `mlp.gate.weight` (bf16) + `mlp.gate.e_score_correction_bias`
  (fp32, weight.py:212-218) + `mlp.shared_experts.{gate,up,down}_proj`
  (weight.py:219-220); trailing `embed_tokens` / `norm` / `lm_head`
  (weight.py:222-227).
- Routed-expert sources (offload banks): NVFP4 ModelOpt regex weight.py:59-72,
  compressed-tensors regex weight.py:73-85, selector weight.py:87-91,
  `nvfp4_expert_spec` 113-114; zai-org block-fp8 regex `_FP8_EXPERT_RE`
  weight.py:97-100 with `iter_expert_pieces` 231-276.

## 5. models/gemma4/gguf.py (474 lines) - the adapter pattern; register.py - VERIFIED

- `parse_gguf_config(shim: GgufConfigShim) -> ModelConfig` (**gemma4/gguf.py:51-140**):
  reads `gemma4.*` GGUF KV keys via local `g(key)` helper (54-58; KeyError on
  missing `gemma4.<key>`), builds attention groups from
  `attention.sliding_window_pattern` (per-layer bool list -> SWA/full layer ids),
  two `RotaryConfig`s (full layers: partial-rotary "proportional" with rotated
  width recovered from the `rope_freqs.weight` divisor tensor via
  `_full_rotary_dim` 31-48; SWA layers: `rope.dimension_count_swa`), and returns a
  `freetoken.models.config.ModelConfig` with `expert_quant="q4_0"`,
  `moe_weight_format="q4_0"`, `k_eq_v=True` on full layers, `Final softcap`,
  `embedding_scale = sqrt(hidden)`. Same ModelConfig as the HF gemma4 parser -
  only the source is GGUF metadata. This is the function a Glm5Next adapter must
  mirror, but returning a config carrying `Glm5NextArgs` (KDA/DSA/mHC geometry).
- `_LAYER_SCALAR_MAP` (**gemma4/gguf.py:149-163**): gguf per-layer suffix ->
  FreeToken module-relative name for tiny F32 tensors dequantized to bf16
  (`attn_norm.weight` -> `input_layernorm.weight`, ...,
  `ffn_gate_inp.weight` -> `feed_forward.router.proj.weight`,
  `ffn_down_exps.scale` -> `feed_forward.router.per_expert_scale`).
- `_EXPERT_SUFFIXES` (gemma4/gguf.py:165): `("ffn_gate_up_exps.weight",
  "ffn_down_exps.weight")` - routed experts, skipped by iter_gguf_weights and
  handled by the offload bank loader.
- `iter_gguf_weights(model_path, device, *, include_moe_experts, include_non_moe)`
  (**gemma4/gguf.py:187-298**): asserts `not include_moe_experts` (offload contract,
  204-206) and TP=1 (`_require_tp1` 174-184); yields `(param_name, tensor)` where
  `token_embd.weight` -> `model.embed_tokens.qweight` (packed Q6_K),
  `output_norm.weight` -> bf16, `_LAYER_SCALAR_MAP` suffixes -> bf16, packed
  projections are fused by concatenating packed rows on dim 0 (`attn_q/k/v.weight`
  -> `self_attn.qkv_proj.qweight`, `ffn_gate/ffn_up.weight` ->
  `feed_forward.shared_mlp.gate_up_proj.qweight`; full layers reuse k as v),
  `attn_output.weight` -> `self_attn.o_proj.qweight`, `ffn_down.weight` ->
  `feed_forward.shared_mlp.down_proj.qweight`; anything else raises
  `ValueError("unmapped gemma4 GGUF tensor: ...")` (gemma4/gguf.py:296).
  `_to_bf16` (168-171) dequantizes via `models/gguf/dequant.dequantize`.
- `load_q4_0_expert_sources(model_path, config, *, layer_sink=None)`
  (**gemma4/gguf.py:391-450**): per-layer host banks of verbatim Q4_0 packed bytes
  - `gate_up` `[E, 2I, H//32*18]` uint8 and `down` `[E, H, I//32*18]` uint8 per
  layer (`_q4_0_expert_specs` 382-388, `alloc_layer_banks` / `LayerCompletionTracker`
  / `PinPipeline` from `freetoken.moe.host_banks`); `layer_sink=None` -> pin-as-you-
  go for serving, `layer_sink` given -> converter mode, nothing pinned.
  `dummy_q4_0_expert_sources` 453-464 for tests.
  `is_gguf_model(config)` 306-308: true iff `moe_weight_format == "q4_0"`.
- Registration (**models/register.py:193-197**; search hit at 193):
  ```python
  "Gemma4GGUFForCausalLM": ModelSpec(
      "freetoken.models.gemma4",
      "Gemma4ForCausalLM",
      parse_config="parse_gguf_config",
      iter_weights="iter_gguf_weights",
  ),
  ```
  - `ModelSpec` dataclass at register.py:11-24 with fields `module` (13),
    `model_cls` (14), `parse_config` (15), `iter_weights` (16),
    `checkpoint_roots` (18), `checkpoint_segments` (20),
    `packed_modules_mapping` (22), `unquantized_modules` (24).
  - Existing GLM HF specs for reference: `Glm5NextForConditionalGeneration`
    register.py:221-226 and `Glm5NextForCausalLM` register.py:228-233, both
    module `freetoken.models.glm5_next`, model class `Glm5NextForCausalLM`.
  - What a new Glm5Next GGUF spec needs: (a) `GGUF_ARCH_TO_REGISTRY["glm5next"]
    = "Glm5NextGGUFForCausalLM"` in models/gguf/config.py:19; (b) a registry
    entry shaped like the Gemma4 GGUF one: `ModelSpec("freetoken.models.glm5_next",
    "Glm5NextForCausalLM", parse_config="parse_gguf_config",
    iter_weights="iter_gguf_weights")` plus a `parse_gguf_config`/`iter_gguf_weights`
    pair (e.g. models/glm5_next/gguf.py) that produces `ModelConfig` with
    `glm5_args` (KDA/DSA/mHC) and a GGUF-name -> FT-name iterator incl. the expert
    bank loader analogue of `load_q4_0_expert_sources`.

## 6. No existing glm5 GGUF spec - VERIFIED

- `GGUFForCausalLM` appears exactly once in models/register.py (line 193, Gemma4).
- The string `glm5next` (the GGUF arch key) appears NOWHERE under python/freetoken/
  (project-wide search hits only `.veai/memory/` and `.idea/`).
- models/glm5_next/ has no gguf.py (file list in section 4); `GGUF_ARCH_TO_REGISTRY`
  has only `gemma4`. Confirmed: a glm5 GGUF spec must be built from scratch.

## 7. CPU GEMV expert formats (moe/cpu_executor.py) - DRIFTED vs plan expectation

- `_WFMT_IDS` (**moe/cpu_executor.py:72**): `{"bf16": 0, "nvfp4": 1,
  "mxfp4_triton": 2, "ds_fp4": 3, "q4_0": 4}` ("Weight-format ids must match
  WFmt in csrc/cpu_moe/cpu_moe_ext.cpp").
- `CpuMoeExecutor` docstring (cpu_executor.py:144-147): "(bf16, nvfp4, mxfp4_triton,
  ds_fp4 or q4_0 - see `_WFMT_IDS` / `_resolve_banks")".
- Resolvers: `_resolve_banks` 340-405 (bf16 inline, then nvfp4 inline at the tail
  of the method), `_resolve_q4_0_banks` 407-432, `_resolve_mxfp4_banks` 434-463,
  `_resolve_dsfp4_banks` 465-491.
- **fp8_block is NOT a CPU-GEMV format**: no branch in `_resolve_banks` and no
  entry in `_WFMT_IDS`. fp8_block routed experts are GPU-only (bank schema
  moe/offload_cache.py:42-44, kernels moe/fused_fp8_block.py:16-42 via
  `kernel/triton/fp8_blockscale_moe`; benchbw `_BUILDABLE_FORMATS`
  moe/benchbw.py:68 = bf16, nvfp4, fp8_block, mxfp4_triton, ds_fp4).
- The CPU nvfp4 path is W4A8, matching the GPU offload MMVQ path
  (comment at kernel/csrc/cpu_moe/cpu_moe_ext.cpp:1120).
- So the plan's "expected: nvfp4-w4a8 / bf16 / fp8_block only" is outdated. Today's
  surface: **bf16 / nvfp4(W4A8) / mxfp4_triton / ds_fp4 / q4_0** - the last one
  added by the gemma4 GGUF work and the template a Q*_K / IQ4_XS CPU-GEMV format
  would follow.

---

## Surprising findings (delta vs PLAN.md / memory)

1. **IQ4_XS kernels exist now.** The plan (and memory
   `ft-serve-gguf-glm5next-unsupported`, verified 2026-09-12) states the vendored
   kernels have NO IQ4_XS. Current tree: full dequant + dense MMVQ + grouped
   moe_vec MMVQ for iq4_xs (case 23), plus vecdotq/ggml-common support. This was
   evidently upstreamed since the memory was written. The kernel-side blocker for
   the dominant expert format of the target GGUF is GONE for decode/offload paths;
   iq-family MMQ (prefill large-batch) and `ggml_moe_get_block_size` support are
   still missing (returns 0 for iq types).
2. **Q3_K / Q4_K / Q8_0 are fully covered** in every CUDA switch (incl. MMQ and
   MoE-MMQ). The memory's "only Q4_0/Q6_K exist; Q8_0 half-landed" is stale -
   layers/gguf.py:33-40 and dequant.py BLOCK_SHAPE/GGML_NAME now include Q8_0, and
   the CUDA MMVQ/MMQ/MoE switches have Q3_K and Q4_K wired. The gap to close for
   this GGUF is Python-side metadata + dispatch wiring (BLOCK_SHAPE / GGML_NAME /
   row_bytes / _MMVQ / _MMQ / _DEQUANT / CPU-executor fmt) for IQ4_XS, IQ3_XXS,
   Q3_K, Q4_K - not CUDA kernel porting for the GEMV path.
3. **CPU executor supports 5 formats** (mxfp4_triton and ds_fp4 exist; fp8_block
   does not) - the plan's CPU-GEMV format list needs updating.
4. Everything else (shim signature, ValueError line 65, whitelist, detection chain
   lines 821/218, register.py:193-197 pattern, glm5_next args/weight names) matches
   the plan exactly.

### Memory corrections to relay

- `ft-serve-gguf-glm5next-unsupported`: the "NO IQ4_XS" and "only Q4_0/Q6_K exist"
  claims are stale as of this verification - iq4_xs dequant/MMVQ/moe_vec and Q8_0
  python-side dispatch are present in-tree. The whitelist/line-number facts in that
  memory remain correct.
