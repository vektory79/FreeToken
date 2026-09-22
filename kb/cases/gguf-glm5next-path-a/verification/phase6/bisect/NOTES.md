# Bisect working notes - live per-layer capture

## VERDICT (2026-09-14): ROOT CAUSE FOUND AND FIXED - see ROOT-CAUSE.md
- Broken module: fused_mul_mat_gguf (python/freetoken/layers/gguf.py) consumed
  non-contiguous activations as garbage; KDA f_b_proj/g_b_proj are the only
  non-contiguous callers (torch.split views of the in_proj output) -> KDA gates
  garbage in all 34 KDA layers -> context destroyed -> off-topic generation.
- Fix: contiguity guard in the dispatch (+4 lines, uncommitted). Regression test
  tests/kernels/test_gguf_quant.py::test_dense_gguf_noncontig_activations
  (fails before / passes after). 105 passed 1 skipped across the gguf suites.
- Live 5-prompt battery on the fixed model: 5/5 on-topic (battery_after_fix.md).
- Final tree: ONLY layers/gguf.py (+4) and test_gguf_quant.py (+27) modified;
  all bisect instrumentation reverted; no commits made.

## GGUF capture (B1/B2 record)
- Boots: tuned cmd + FREETOKEN_PHASE6_BISECT=/tmp/ft6_bisect/capture (committed-in-tree
  instrumentation in models/glm5_next/model.py + moe.py was an UNCOMMITTED ephemeral
  patch; snapshot at /tmp/ft6_bisect/ephemeral-bisect-patch.diff; REVERTED at the end).
- Instrumentation bug found+fixed in-place during B1: ForCausalLM capture used
  `logits[0, -1]` (scalar on [T, V] layout) -> `RuntimeError: selected index k out of
  range` on first request. Fixed to `logits[-1]`.
- Runs: run0 essay 27 tok, run1 "2+2" 18 tok, run2 BANANA 19 tok per boot.
- NOTE run1/run2 have shorter prompts; radix prefix cache may hit on the ~10-token
  system prefix, so their captured prefill covers the cache-miss tail.
- Capture data (~1.5 GB) + harness live in /tmp/ft6_bisect/ - safe to delete.
