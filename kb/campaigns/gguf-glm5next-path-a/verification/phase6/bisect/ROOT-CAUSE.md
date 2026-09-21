# Live per-layer bisect: ROOT CAUSE FOUND AND FIXED - 2026-09-14

Verdict: the off-topic defect was a NON-CONTIGUOUS ACTIVATION bug in
`fused_mul_mat_gguf` (python/freetoken/layers/gguf.py). The KDA gate projections
`f_b_proj` / `g_b_proj` are the only consumers that feed non-contiguous
`torch.split` views of the in_proj output into the GGUF GEMM; the CUDA MMVQ/MMQ
kernels read the activation buffer assuming row-major strides and silently
returned garbage. The KDA forget-gate (g1) and output-gate (g2) logits were
garbage in ALL 34 KDA layers from layer 0 onward -> the delta-rule state decay
and the gated output norm were wrong -> context was destroyed progressively ->
prior-driven off-topic generation (0/24 on-topic). MoE / tokenizer / embeddings /
config / attention weights were all verified correct and are exonerated.

FIX (uncommitted, working tree): python/freetoken/layers/gguf.py,
`fused_mul_mat_gguf` now normalizes non-contiguous activations
(`x = x.contiguous()`) before dispatch. +4 lines.
REGRESSION TEST: tests/kernels/test_gguf_quant.py::
`test_dense_gguf_noncontig_activations[4|27]` (fails before, passes after).
Suite: tests/kernels/test_gguf_quant.py + tests/models/test_glm5_next_gguf.py +
tests/models/test_gguf_expert_banks.py = 105 passed, 1 skipped.
LIVE BATTERY (fixed model, greedy, 5 prompts): 5/5 ON-TOPIC, including an exact
"BANANA" reply. Before: 0/5 (image-math JSON, empty, <user>-tag confusion,
nezha JSON, pyannote diarization). See battery_after_fix.md.

## Evidence chain (all artifacts under /tmp/ft6_bisect/, run ids = boot tags)

B0 setup
- The FREETOKEN_PHASE6_BISECT instrumentation in models/glm5_next/{model,moe}.py
  was an UNCOMMITTED leftover patch (never live-tested); snapshot
  /tmp/ft6_bisect/ephemeral-bisect-patch.diff. It had a bug: ForCausalLM capture
  used logits[0, -1] (scalar on [T, V]) -> RuntimeError on first request; fixed
  in place during B1, all instrumentation REVERTED at the end (git checkout).

B1/B2 captures (identical 27-token prompt, render+ids verified identical
  GGUF vs NVFP4 via token_check.py; "What is 2+2?" differs by 1 token only)
- gguf (run0/1/2), nvfp4 (run0/1/2): bisect.pt (embed + per-layer in/out norms +
  full out_x/out_res [T,H]/[T,16384] fp32), bisect-moe.pt, logits.pt.
- Logits top-10: GGUF identical across all 3 prompts ([198, 2, 785, ...]),
  NVFP4 varies per prompt -> GGUF context-blind at the output.

B3 per-layer comparison (analysis.py, focus.py; tables in bisect-tables.md)
- Residual norm explosion (8.9 -> ~40k) happens in BOTH models -> architecture-
  normal (mHC), NOT the defect.
- NVFP4 healthy prompt-sensitivity: cos(last-token hidden, prompt A vs B) decays
  0.95 -> 0.1-0.5 by L44. GGUF stays 0.77-0.97 everywhere -> context ignored.
- GGUF vs NVFP4 on the identical prompt: L0 attention input cos 0.99998 but
  attention OUTPUT cos 0.9357 (row 0 = 0.9998, rows token-idiosyncratic 0.62-0.99,
  NOT cumulative) -> the break is inside L0's KDA forward.
- Per-row out_res cos decays with depth (L4 row26 0.59 -> L32 0.15): context
  component of the residual dies progressively in GGUF.

B4 functional MoE bisect (prescribed suspect #1) - MOE EXONERATED
- Router healthy: topk ids/weights vary across rows in 42/42 layers (moe_rows.py).
- Offline recompute of layer-3 routed output from the FILE's own bytes via gguf-py
  reference dequant (recompute_routed.py): cos(recompute, kernel) = 0.99994/0.99997,
  norm ratio 1.000 -> kernel+bank+gather+swiglu+weighted-sum faithful to the file.
- Expert replay through the NVFP4 pipeline on the identical captured hidden+topk
  (moedebug_gguf.pt -> moedebug_replay.pt): cos(routed)=0.996/0.973,
  cos(shared)=1.0000 -> expert content/order equivalent. MoE fully exonerated.

B5 root cause isolation (KDA)
- Config diff (config_diff.py): all Glm5NextArgs fields equal (only fp32
  representation noise + the known indexer_rope_interleave shim difference).
- Input path: engine input_ids == expected 27 ids; every embed row cos 1.000.
- Offline fp64 KDA replication (kda_sim.py) with GGUF-dequant vs reference bf16
  weights on the identical input: cos(simG, simR) = 1.0000 -> quant noise cannot
  explain the engine divergence.
- Stage-by-stage engine capture (kda.py FREETOKEN_PHASE6_KDA_CAP, kda_stage_diff.py):
    stage      cos(GGUF,NVFP4)   norms G/N
    in         0.999981          28.31 / 28.31
    proj       0.999985
    conv_in    0.999984
    mixed      0.999990
    b/f_a/g_a  0.99999+
    g1 (f_b)   0.082763 (!)      32.4 / 98.7
    g2 (g_b)   0.052562 (!)      42.9 / 257.1
    core_out   0.998314
    onorm_out  0.905788
    final      0.935661
- Kernel directly tested (fused_mul_mat_gguf, real f_b packed bytes):
    contiguous x, K=128, T=27 (MMQ) and T=4 (MMVQ): cos 0.99998 -> kernel+weights OK
    NON-contiguous x (stride (24896,1) slice): cos 0.033, norm ratio 0.98 -> BUG
- Root cause: f_a/g_a are non-contiguous torch.split views of the in_proj output;
  the gguf CUDA kernels assume row-major activations. g1/g2 garbage -> KDA gates
  broken in all 34 KDA layers -> context loss. In_proj/o_proj/MLP/lm_head inputs
  are contiguous, which is why every other GGUF GEMM was correct.

## Fix + validation
- python/freetoken/layers/gguf.py: contiguity guard at the dispatch boundary.
- Regression test: tests/kernels/test_gguf_quant.py::test_dense_gguf_noncontig_activations
  (verified failing with the guard removed, passing with it).
- 105 passed / 1 skipped across the three gguf-related suites.
- Live 5-prompt battery on the fixed model: 5/5 on-topic (battery_after_fix.md).

## Exact final state
- Working tree: ONLY python/freetoken/layers/gguf.py (+4 lines) and
  tests/kernels/test_gguf_quant.py (+27 lines) modified. All bisect
  instrumentation (model.py / moe.py / kda.py) REVERTED; no commits made.
- Ephemeral harness lives in /tmp/ft6_bisect/ (runner, clients, analysis scripts,
  captures ~1.5 GB); safe to delete after review.
- All servers shut down cleanly (exit 143); VRAM back to ~1.3 GiB baseline.

## Remaining risks
- The gemma4 gguf path shares fused_mul_mat_gguf and benefits from the guard;
  no other caller in the tree feeds non-contiguous activations today, but the
  guard makes the contract explicit.
- p1 tokenization nuance ("What is 2+2?" read as "What is +2?") matches the known
  pre=glm4/gpt2 converter divergence for arithmetic strings - cosmetic, separate
  from this defect.
- A follow-up hardening option (not done): assert contiguity inside the CUDA
  kernel wrappers instead of the python dispatch.
