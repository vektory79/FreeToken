# Task 04: Kernel dispatch - python wiring + convert path

Type: Code | Agent: Code | Duration: ~4-8 hours

## Goal

Python-side диспетчеризация квантовых форматов: layers/gguf.py форматная таблица +
dequant.py BLOCK_SHAPE + convert path (is_gguf_model -> GGUFLinear/GGUFEmbedding).

## Instructions for the subagent

1. layers/gguf.py: extend _MMVQ/_MMQ/_DEQUANT sets for the file's formats.
   IQ formats are MMVQ-only (no MMQ in the vendored csrc) - loud NotImplementedError
   above the MMVQ cutoff (batch > ~6); do NOT silently fall back.
   Verify each format's case in the csrc switch BEFORE claiming support.
2. dequant.py BLOCK_SHAPE: entries for each new format (geometry-only, no python
   dequant unless needed for tests - the CUDA kernels dequantize on GPU).
   Each entry must equal gguf-py GGML_QUANT_SIZES (test pins this).
3. Convert path: is_gguf_model(config) + convert_<family>_to_gguf swapping exactly
   the .qweight-yielded params to GGUFLinear/GGUFEmbedding/GGUFLMHead.
   The untied head uses ParallelLMHead's prefill last-token slicing.
   Per-target packed-type guards: token_embd -> Q8_0, output -> Q6_K, linears ->
   Q8_0 (or whatever the file uses). Variant files with deviating types fail at
   ITERATION time.
4. moe_weight_format marker: set "gguf" on the gguf config path (engine/planning
   consumers read this field).
5. Engine gate: reject gguf + cpu/hybrid/fused strategies at config time (offload
   is the only wired path until the hybrid milestone); the #186 clamp (chunk =
   65535 // top_k) goes here too.
6. Kernel parity tests (tests/kernels/): dequant parity vs gguf-py within the
   dfloat precision contract (per-element bound 8*2^-11*factor*|d| + 1e-3 -
   bit-parity is IMPOSSIBLE for half-chain kernels; document why in the docstring).
   moe parity: kernel output vs a torch reference on a synthetic expert stack.

## Traps (from [TRAPS.md](../TRAPS.md))

- T17: iq formats lack MMQ - do not add them to _MMQ without verifying the csrc
  switch; ggml_moe_get_block_size returns 0 for iq (fine for moe_vec, breaks MMQ).
- T18: BLOCK_SHAPE and _DEQUANT must stay consistent (upstream #358 is the
  missing-entry class of bug).
- T19: The tied variant (no output.weight) needs GGUFTiedLMHead - if the file is
  untied, add a tie guard at convert time with a clear error.
- T20: The single-signature degenerate case must be byte-identical to the
  pre-partition behavior (the existing tests pin this - keep them green).

## Acceptance criteria

- [ ] I've created a git commit for this task
- Runs under ORCHESTRATION.md (quality loop + hygiene + commit gate).
