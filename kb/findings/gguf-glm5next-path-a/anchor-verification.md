# Anchor verification: glm5next GGUF native serving (Path A)

NOTE: the capability matrix (CUDA kernel switches, per-format coverage,
CPU-GEMV format surface) was removed as an outdated 2026-09-14 snapshot;
kernel/format state lives in the campaign reports, not here. What remains is
the stable GGUF-config detection chain and the ModelSpec adapter pattern -
both re-verified read-only against the tree at `/media/ai/src/FreeToken`.
All paths relative to `python/freetoken/` unless noted. Verification is
current-source, not memory.

---

## 1. models/gguf/config.py - the shim factory

- `GGUF_ARCH_TO_REGISTRY` at **config.py:19-21**: whitelist mapping GGUF
  `general.architecture` -> registry key. Adding an arch means adding one
  entry here (e.g. `"glm5next": "Glm5NextGGUFForCausalLM"`).
- `build_gguf_shim(model_path: str) -> GgufConfigShim` at **config.py:61-93**.
  Unsupported arch raises ValueError at **config.py:65**:
  ```python
  raise ValueError(
      f"GGUF architecture {arch!r} is not supported "
      f"(known: {sorted(GGUF_ARCH_TO_REGISTRY)})"
  )
  ```
  -> renders as `GGUF architecture 'glm5next' is not supported (known: ['gemma4'])`.
- `GgufConfigShim` (frozen dataclass, **config.py:24-43**) constructed with
  `architectures=[registry_key]` (registry dispatches on `architectures[0]`),
  `model_path=model_path`, `model_type=arch`, `metadata=load_gguf_metadata(model_path)`,
  `vocab_size=_vocab_size(model_path)` (config.py:46-58: `token_embd.weight`
  `shape[-1]`, fallback `len(tokenizer.ggml.tokens)` for metadata-only GGUF),
  `tie_word_embeddings` = `"output.weight" not in names` (config.py:73) or the
  `OUTPUT_WEIGHT_PRESENT_KV` fallback (config.py:74-86).
- `GgufConfigShim.to_dict()` (config.py:33-43) emits a minimal HF-like dict with
  `torch_dtype: "bfloat16"` - this is what trunk code (e.g. `--dtype auto`) reads.
- Arch-specific `parse_gguf_config` reads `shim.metadata` keys
  (`<arch>.block_count`, `embedding_length`, ...) to build the FreeToken ModelConfig.

## 2. Detection chain (boot path) - stable

- `python/freetoken/server/args.py:821`:
  `cfg = cached_load_hf_config(kwargs["model_path"]).to_dict()`
  (import at args.py:819). This sits inside `parse_args` (def at **args.py:132**),
  in the `"auto"` dtype-resolution block (args.py:813-830).
  Two earlier non-fatal callers exist (args.py:184-186, 233-235, tool-call /
  reasoning parser inference); 821 is the boot-path one.
- `python/freetoken/utils/hf.py:209` `def cached_load_hf_config(model_path)`;
  GGUF branch: line 212 imports `gguf_config_source`, line 214
  `if (gguf_src := gguf_config_source(model_path)) is not None:`, line 216 imports
  `build_gguf_shim`, **line 218 `return build_gguf_shim(gguf_src)`**.
- `python/freetoken/models/gguf/reader.py:42` `def gguf_config_source(model_path) -> str | None`
  decides that a path is GGUF.
- Chain: `server/args.py:821 -> utils/hf.py:214-218 -> models/gguf/config.py:61-93`
  (raise at 65).

## 3. ModelSpec adapter pattern (models/gemma4/gguf.py, models/register.py)

The template a new GGUF arch adapter must mirror:

- `parse_gguf_config(shim: GgufConfigShim) -> ModelConfig` (**gemma4/gguf.py:51-140**):
  reads `<arch>.*` GGUF KV keys via a local `g(key)` helper (54-58; KeyError on
  missing keys) and builds the same `freetoken.models.config.ModelConfig` the
  HF parser produces - only the source is GGUF metadata.
- `iter_gguf_weights(model_path, device, *, include_moe_experts, include_non_moe)`
  (**gemma4/gguf.py:187-298**): asserts `not include_moe_experts` (offload
  contract, 204-206) and TP=1 (`_require_tp1` 174-184); yields `(param_name,
  tensor)` with packed projections fused by concatenating packed rows on dim 0;
  unmapped tensors raise `ValueError("unmapped gemma4 GGUF tensor: ...")`
  (gemma4/gguf.py:296). `_to_bf16` (168-171) dequantizes via
  `models/gguf/dequant.dequantize`.
- `_LAYER_SCALAR_MAP` (**gemma4/gguf.py:149-163**): gguf per-layer suffix ->
  FreeToken module-relative name for tiny F32 tensors dequantized to bf16.
- `_EXPERT_SUFFIXES` (gemma4/gguf.py:165): routed experts, skipped by
  iter_gguf_weights and handled by the offload bank loader
  (`load_q4_0_expert_sources`, **gemma4/gguf.py:391-450**).
- Registration (**models/register.py:193-197**):
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
  - A new Glm5Next GGUF spec needs: (a) the `GGUF_ARCH_TO_REGISTRY["glm5next"]`
    entry in models/gguf/config.py:19; (b) a registry entry shaped like the
    Gemma4 GGUF one plus a `parse_gguf_config`/`iter_gguf_weights` pair that
    produces `ModelConfig` with the arch-specific args and a GGUF-name ->
    FT-name iterator incl. the expert bank loader analogue of
    `load_q4_0_expert_sources`.