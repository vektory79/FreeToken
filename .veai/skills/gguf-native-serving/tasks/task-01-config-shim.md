# Task 01: Config shim - gguf metadata to family Args

Type: Code | Agent: Code | Duration: ~2-4 hours

## Goal

Добавить архитектуру в GGUF_ARCH_TO_REGISTRY + реализовать parse_gguf_config,
маппящий gguf metadata -> family Args через СУЩЕСТВУЮЩИЙ config.parse_config.

## Instructions for the subagent

1. Add `<arch>: "<Family>GGUFForCausalLM"` to GGUF_ARCH_TO_REGISTRY
   (models/gguf/config.py GGUF_ARCH_TO_REGISTRY).
2. Add a ModelSpec to models/register.py (module, model_cls, parse_config,
   iter_weights="iter_gguf_weights" - even if stub for now).
3. Add an AOT arch_aliases claim in kernel/aot_models.py (REQUIRED by the registry
   invariant test tests/models/qwen4_exp/test_weight.py - the test computes
   set(_MODEL_REGISTRY) - claimed and asserts empty).
4. NEW models/<family>/gguf.py: parse_gguf_config(shim) -> ModelConfig.
   Strategy: translate gguf metadata keys into a text-config namespace, then
   delegate to <family>/config.py parse_config - this gives BIT-IDENTICAL behavior
   to the HF path (proven pattern: glm5next).
5. Key mapping: use the discovery dump (metadata.txt) + the llama.cpp LLM_KV map.
   Layer topology: per-layer arrays (head_count_kv etc.) are len=block_count;
   SLICE to num_layers = block_count - nextn_predict_layers (MTP/NextN is skipped).
   Dense/MoE split: leading_dense_block_count.
   KDA/DSA split: per-layer attention.head_count_kv == 0 (no dedicated key).
6. Guard philosophy: FAIL LOUDLY on variant quants that would silently mis-serve.
   Pattern: per-role, per-target checks at yield time (not late shape asserts).
7. Validation gate: resolve the config on the REAL file (probe script) and
   cross-check EVERY resolved field against the HF reference config.json.

## Subagent instructions

- Code wave: implement steps 1-6, run the probe, report the resolved field table.
- Test wave: synthetic gguf fixture (metadata-only + token_embd stub) covering
  the full battery + negative paths (missing key -> KeyError, array length
  mismatch -> ValueError, block_count <= nextn -> ValueError, scalar head_count_kv
  broadcast, uniform fold rejection, topk % kpool != 0).
- Review-A: mapping correctness vs the key table + consumer chain.
- Review-B: edge cases + fixture hygiene + import-cycle risk.

## Traps (from [TRAPS.md](../TRAPS.md))

- T04: parse_gguf_config must produce an Args equivalent to load_args(HF config)
  for the same model - field-identical, not just "close enough".
- T05: The shim's vocab_size comes from token_embd.weight shape[-1] (ggml ne-order,
  ne0 = hidden, ne1 = vocab) - do not confuse with the HF (vocab, hidden) order.
- T06: indexer_types must be per-LAYER ("full",)*num_layers, NOT per-DSA-count
  (the config.py:111 + dsa.py:154 index by layer id).
- T07: lm_head: check output.weight presence -> tie_word_embeddings=False if present.

## Acceptance criteria

- [ ] I've created a git commit for this task
- Runs under ORCHESTRATION.md (quality loop + hygiene + commit gate).
