# Task 02: Tensor translator - iter_gguf_weights + fusion

Type: Code | Agent: Code | Duration: ~4-8 hours

## Goal

Реализовать iter_gguf_weights: 3-way маппинг (gguf -> FreeToken -> llama.cpp),
включая все fusion-тензоры (in_proj, conv1d, kv_b, A_log и т.д.).
Плюс iter_gguf_expert_sources для stacked expert banks.

## Instructions for the subagent

1. Module-level maps (gemma4 pattern): _COMMON_LAYER_MAP, per-attention-class maps,
   _MOE_MAP (suffix -> FreeToken param + cast: "qw" packed / "bf16" / "fp32").
   Cast targets MUST match weight.py HF dtypes exactly per parameter.
2. Fusion slot tables: _IN_PROJ_SLOTS (e.g. q|k|v|b|f_a|g_a), _CONV_SLOTS,
   _KV_B_SLOTS - with pinned orders matching the CONSUMER's runtime split.
   Cross-check EVERY fusion order against the consumer code (kda.py, attention.py),
   not just the comments.
3. Fusion emission: buffers emit when complete; unknown tensor -> ValueError
   naming the tensor; incomplete groups at end-of-stream -> ValueError
   (NOT bare assert - dies under -O).
4. Derived tensors: A_log = log(-ssm_a) fp32 with x >= 0 guard; any other
   derivation (e.g. fused conv, per-head reshape) must cite the consumer.
5. Per-role geometry: each tensor's rows come from ITS OWN shape (down rows = H,
   gate/up rows = I). NEVER reuse one row count for all roles.
6. Role-ordered gguf_types: `tuple(slots[r].ggml_type for r in ("gate","up","down"))`
   - NEVER slots.values() (file encounter order).
7. Packed ("qw") yields: validate the ggml_type against the convert contract
   (e.g. {Q8_0, Q6_K} for dense linears, Q8_0 embed, Q6_K head). Variant files
   with deviating types must fail AT ITERATION time, not as late shape asserts.
8. iter_gguf_expert_sources: lazy generator yielding (ckpt_layer, {gate,up,down:
   GgufTensor}) per MoE layer; packed bytes untouched, ggml_type carried;
   dim-0 slice = one expert's contiguous rows. Yields CHECKPOINT layer ids.
9. Per-tensor debug logging: name, GGML_NAME[ggml_type], rows x row_bytes / cast.
10. MTP/NextN: skip every tensor under blk.{N} where N >= num_layers (both block
    tensors and .nextn.* glue) with an explicit skip log.

## Validation (MANDATORY before live boot)

- Static fusion validation: for EACH fused tensor, construct the EXPECTED result
  from the reference HF/FTW checkpoint weights (mmap per-tensor, no full load)
  and compare element-wise. Any order/transpose bug shows immediately.
- Probe: iterate the real gguf; assert yielded multiset == independently derived
  expected multiset; zero unmapped; MTP skipped; per-type histogram matches header.
- 3-way name diff: gguf names vs weight.py expectations vs llama.cpp kinds.

## Subagent instructions

- Code wave: implement steps 1-10 + the static fusion validation probe.
- Test wave: synthetic fixtures with hand-packed quant tensors (bounded scales
  per _FP16_SCALE_FIELDS convention) covering every fusion + negative paths.
- Review-A: fusion order vs consumer code (not comments) + cast correctness.
- Review-B: edge cases + per-projection heterogeneity + test fixture hygiene.

## Traps (from [TRAPS.md](../TRAPS.md))

- T08: ssm_a is a BARE name (no .weight suffix) holding -exp(A_log).
- T09: ssm_dt arrives as .bias but maps to a plain param.
- T10: attn_k_b/attn_v_b are 3D (head-major) - the transpose crosses the Q8_0
  pack axis, so the fusion MUST dequantize those two pieces (Q8_0-only).
- T11: Packed fusion is valid ONLY if all pieces share ne0 (= same row_bytes).
  Guard: pre-cat check naming the layer + differing pieces.
- T12: The reference checkpoint may have SEPARATE q/k/v (not pre-fused) - the
  fusion is FreeToken-side only; validate against the pieces, not a fused ref.

## Acceptance criteria

- [ ] I've created a git commit for this task
- Runs under ORCHESTRATION.md (quality loop + hygiene + commit gate).
