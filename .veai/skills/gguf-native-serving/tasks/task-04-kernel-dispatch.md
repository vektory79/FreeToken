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
- T39: reordering (token,expert) pairs by expert id is a throughput no-op for
  streaming GEMV/MoE kernels - resident CTAs execute neighboring pairs
  concurrently; only full-traffic reduction moves the BW-bound term
  (289.54->291.26 tok/s, noise) (.tasks/mmq-prefill-kernel/v0-ab.md).
- T40: audit vendored kernels for silent bound assumptions: moe.cuh exp_idx > 255
  silently dropped experts 256+ at E=288; moe_q ds-load read token_offs
  [threadIdx.y] OOB - audit index constants, per-thread array sizes, expert-id
  ranges vs model geometry.
- T41: iq tile glue is not vendored verbatim from llama.cpp onto the vendored
  pre-#8495 vLLM interface; the working source is vLLM PR #36226; keep
  VDR=4 + need_sum=true; attribution is DOUBLE (vLLM Apache-2.0 + llama.cpp MIT).
- T42: before porting vLLM helpers check freetoken for an existing producer -
  moe_align_block_size (moe/fused.py:47) already serves the sentinel/int32/
  post-pad trio; one trio serves gate/up (top_k=8) and down (top_k=1,
  tokens=M*top_k).

## Acceptance criteria

- [ ] I've created a git commit for this task
- Runs under ORCHESTRATION.md (quality loop + hygiene + commit gate).
