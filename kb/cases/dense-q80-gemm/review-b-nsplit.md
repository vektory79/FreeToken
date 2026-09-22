# Review-B - Replan wave 1: N-split of kda_in_proj (FREETOKEN_GGUF_KDA_NSPLIT)

Reviewed 2026-09-19, protocol Review-B (edge cases / integration / test integrity / operability).
Read-only; this file is the only artifact written.

Scope audited: uncommitted changeset on vektory79 (nominal HEAD 1706aae; actual HEAD cb8db23,
HEAD~1 2d64a58 - both .veai-only commits, so the python diff vs 1706aae is identical).

Method / evidence base:
- full diff of the 4 files (`git diff --numstat` + per-file diffs, read in full);
- /tmp/gate-nsplit.log (775 lines, full-run log, 17:12, tail: "16 failed, 2173 passed,
  206 skipped, 15 deselected, 4 warnings in 175.63s (0:02:55) EXIT=1") - read per-test;
- /tmp/baseline16.log (17:08, 16-id rerun, "16 failed in 2.93s", same 16 ids, order permuted);
- independent collection reproduce today: "2395/2410 tests collected (15 deselected)";
- engine/graph.py read in full (GraphRunner._capture_graphs, _determine_cuda_graph_bs);
- greps: single env read / single call site / collection-variance mechanisms / JIT evidence.

## B1 diff hygiene - CONFIRMED

- numstat: gguf.py 53/1, kda.py 6/1, tests/kernels/test_gguf_quant.py 143/0,
  tests/models/test_glm5_next_kda_op.py 39/0 = +241/-2 exactly, 4 files, python-only.
  No csrc/, no setup.py. Zero nvcc/clang/JIT/recompilation lines in the gate log (grep empty).
- __all__ adds only `kda_in_proj_forward` (the intended entry point); `_kda_nsplit_env`
  stays private. `FREETOKEN_GGUF_KDA_NSPLIT` is read at exactly one site (gguf.py:116)
  and consumed at exactly one site (kda.py:116) - single source of truth holds.
- The working tree also carries ~17 .veai/memory deltas + 2 untracked memory files -
  parallel-session churn, NOT part of the changeset. Stage by explicit path at STOP GATE
  (existing protocol; zero .veai entries).

## B2 63-nodeid collection delta (2473 vs 2410) - NEED-MORE-DATA (mechanism); classified harmless-but-unexplained (record)

- Run 2 is grounded independently: today's `--collect-only -q` reproduce = 2395/2410 with
  15 deselected, matching gate-nsplit.log's summary (16+2173+206 = 2395 executed) exactly.
- Eliminated mechanisms: pytest collection hooks (pytest_generate_tests /
  collection_modifyitems / pytest_collection: zero hits in tests/); env- or glob-driven
  parametrize lines anywhere except one file; no test files under benchmarks/ or scripts/.
- Grounded candidate: tests/models/test_quant_config.py builds its parametrize list at
  COLLECTION time by globbing HF-cache snapshots + /mnt/nvme/models
  (L546-547 `roots = glob.glob(...)`; L622 `DIRS = _candidate_dirs()`; L626
  `@pytest.mark.parametrize("ckpt", DIRS, ...)`, 57 nodeids today). Checkpoint-dir or
  mount churn between the two runs shifts collection by tens of nodeids. Unconfirmed -
  run 1 left no log, so the exact 63 cannot be reconstructed.
- Classification: harmless-but-unexplained (record). The failure sets are byte-identical
  across the two surviving logs (same 16 ids), and collection-size variance does not
  mutate results. A test-isolation hazard is NOT established: isolation hazards manifest
  as nondeterministic RESULTS, not collection counts; no in-tree mechanism exists.
- Run-2 + --collect-only authoritative: JUSTIFIED (count reproduced, failure list matches
  baseline classes per B3). Residual: run 1's own numbers rest on the implementer's word.

## B3 environment classification of the 16 failures - CONFIRMED

Verified per-test from the gate-nsplit.log tracebacks (not from the claim):
- 11x `ModuleNotFoundError: No module named 'tests'` - the documented console-pytest
  import class, here as IN-TEST lazy imports (tests collect fine; they fail at runtime
  because the repo root is not on sys.path):
  7x tests/attention/test_dsa_kpool.py [nvfp4] (via `_ref_attend` ->
  `from tests.kernels.test_kv_nvfp4 import _reference`, :175);
  2x tests/kvcache/test_dsa_pool.py::test_latent_budget_matches_allocations_and_rebuild
  [1-nvfp4]/[4-nvfp4] (:220, `_decode_latent`);
  tests/models/test_glm_dsa.py::test_backend_ragged_prefill_identity_and_selection[nvfp4]
  (:316, `_decode_latent`);
  tests/server/test_muse_glimmer_parsers.py::test_mixed_quant_groups_rejected_in_any_order
  (:872, `import tests.models.test_muse_glimmer`).
- pinned_tensor UVA: RuntimeError cudaHostGetDevicePointer failed - invalid argument
  (test_pinned_tensor.py:128). Known baseline.
- qsa_fp8 [1-1-64]: AcceleratorError CUDA error: invalid argument (test_qsa_fp8.py:48).
  Known baseline.
- qwen4_exp/test_ple.py::test_track_snapshot_equals_a_prefill_stopped_at_the_boundary
  :421 - `torch.equal` bitwise flake (assert False, compared tensors visually identical).
  Known baseline.
- 2x glm_dsa: triton OutOfResources shared memory 102400 > 101376
  (test_glm_dsa.py:141 / :223 via glm_dsa_sparse.py:487/:517). Known baseline.
Total: 11 + 4 + 2 = 16 = exactly the stated baseline composition; zero failures outside
the baseline set; no new failure family. The 3x test_prefill_hit_d2d baseline class did
NOT fire in this run - baseline drift, recorded (not a wave effect).
- Corroboration: baseline16.log (claimed stash-restored unmodified tree) reran the same
  16 ids -> 16/16 F in 2.93 s. Caveat: neither log carries a pytest header/invocation
  line, so the rerun's tree state is verbal-only (stash list empty, consistent with
  stash -> run -> pop; reflog has no contradicting entry). See F1.
- Note: the 11 import-class items are green under `python -m pytest` (repo root lands on
  sys.path). Optional noise reduction for future gates, not required.

## B4 mirror/placement rule - CONFIRMED

No new files (git status shows M only). Kernel-adjacent tests extended into the existing
tests/kernels/test_gguf_quant.py (file now 74 nodeids, 21 test fns):
- test_kda_nsplit_env_knob: default/1/2, PER-CALL flip with no module state, invalid
  stragglers ("12","0","-2","abc"," 2","02","+2") fail-fast - mirrors
  test_moe_mtile_env_knob as TASK.md required;
- test_kda_nsplit_bitwise_parity: production geometry N=24896/K=4096/batch 256,
  `torch.equal` (no tolerance) split vs single AND knob=1 vs plain GGUFLinear.forward;
- test_kda_nsplit_boundary_math: N=48/32 fall back single (ragged / < 2 tiles),
  N=64 -> [32,32], N=96 -> [32,64], all bitwise; non-GGUF layer passthrough;
- test_kda_nsplit_decode_batch_stays_mmvq: bs=4 with knob=2 stays on the vec kernel.
Call-site wiring test extended into the existing tests/models/test_glm5_next_kda_op.py
(real Glm5NextKDA op, non-GGUF in_proj, knob=2 -> torch.equal passthrough).
All 5 new tests are absent from the gate's FAILED list, i.e. green in the full gate.
CONTRIBUTING mirror rule honored.

## B5 "do not flip after boot" operability - CONFIRMED

- Documented in TWO durable places: kda_in_proj_forward docstring (gguf.py - "Do not flip
  the knob after boot: decode CUDA graphs capture the dense MMQ path for bs > 6
  (engine/graph.py GraphRunner._capture_graphs), baking whatever structure was live at
  capture time - selection happens across boots, not mid-boot") and the
  test_kda_nsplit_decode_batch_stays_mmvq comment.
- ACCURATE: in engine/graph.py, capture runs `model.forward()` inside
  `torch.cuda.graph(...)` once per capture bs; the python env read executes at capture
  time only, replays re-issue the baked launches. Eager prefill reads per call.
- Env-only is sufficient for the A/B wave: knob consumed inside the model forward; the
  ft serve worker inherits the launcher env; benchbw and e2e harnesses need nothing;
  no CLI plumbing required or present. Caveat (by design): a mid-boot flip would leave
  eager prefill on the new value while decode graphs keep the old - the docstring warns;
  the A/B wave uses one env value per boot, so the hazard is not exercised.

## B6 decode re-measure mandate - CONFIRMED (real, not vacuous)

engine/graph.py `_determine_cuda_graph_bs`: default candidates = [1, 2, 4] +
range(8, max+1, 8) with max 160 (<80 GB free) - capture sizes >6 exist. For each bs,
`model.forward()` with a decode batch of `bs` dummy reqs runs INSIDE the graph capture;
for bs > 6 the KDA in_proj takes the MMQ path (x rows = bs > _MMVQ_SAFE=6), so with
knob=2 the captured graph contains 2 sub-GEMM launches + 2 copies instead of 1 GEMM.
Graph contents DO change -> the A/B brief's "decode graphs re-captured + decode
re-measured" mandate is substantive. (Production A/B at max-running-requests=1 replays
the bs=1 graph - MMVQ vec path - so decode should stay flat in practice; the mandate is
still correct and cheap.)

## B7 copy-overhead accounting - CONFIRMED (arithmetic); NIT (durability)

Arithmetic (production chunk: n_tok=8128, n_out=24896, half=12448, bf16):
- per call: 2 copies, each 8128 x 12448 x 2 B = 202.35 MB payload -> 404.7 MB read+write;
- 34 kda_in_proj calls/chunk -> 68 copies -> 27.5 GB/chunk traffic;
- at step0's 1638 GB/s effective BW = 16.8 ms; at ~1.8 TB/s = 15.3 ms.
The implementer's "~15 ms/chunk over 34 launches" is right to first order (dst is a
strided view `out[:, :half]` - elementwise strided copy, still ~BW-bound; slight
efficiency loss vs contiguous only makes the estimate conservative).
- The 0.24-0.28 s dense-term target IS durable: TASK.md:106, step0-profile.md:23 and
  :226, review-a-step0.md:159 and :189, triage-step0.md:17 and :58.
- The ~15 ms overhead figure itself is NOT in any durable artifact (chat/brief only) -
  see F3. Note the 0.24-0.28 s target is GROSS; net of the verified 15.3-16.8 ms it is
  ~0.22-0.26 s. The A/B brief's "0.24-0.28 net of ~15 ms" phrasing slightly conflates
  gross/net; the target stays conservative either way.

## B8 wave hygiene - gaps flagged (NIT)

- Gate ran serial (sequential progress bars, no xdist markers) - consistent with the
  serialized-runs rule.
- Review-time re-census (performed during this review): VRAM 1431 MiB used (matches the
  claimed ~1.4-1.5 GB); zero pytest / ft serve / benchbw survivors; /dev/shm has 5
  sem.mp-* = the documented IDE-baseline count (attribute, never kill). Postflight state
  is clean.
- Gaps: the preflight/postflight census and the gate logs are not recorded under
  .tasks/dense-q80-gemm/ (the folder holds step0 artifacts only; `find *nsplit*` = none);
  both surviving logs are headerless (no pytest banner / args; gate-nsplit.log starts at
  the 3% progress bar, baseline16.log at the 100% bar); run 1 (2473) left no log at all.
  The STOP-GATE checklist item "artifacts and report recorded under .tasks/<campaign>/
  with exact paths" is still open - close it before the gate (F1).

## B9 out-of-scope cross-check - CONFIRMED (clean)

The changeset touches none of the campaign's out-of-scope paths: fetch copies
(`fast_index_copy_multi` machinery untouched - the split's two narrow `copy_` calls are
local out-assembly in the helper, not the expert-fetch path), prefill overlap, MoE
re-tuning (moe.cuh / moe.py / FREETOKEN_GGUF_MOE_MTILE untouched), KV/pool (kvcache/
untouched). Diff = 2 production files + 2 test files, nothing else.

## Findings

- F1 NIT (action required before STOP GATE): gate-log/artifact hygiene. Both surviving
  logs are headerless, run 1 (2473) left no log, and no wave artifacts exist under
  .tasks/dense-q80-gemm/. Action: record the gate logs WITH their invocation lines, a
  census record, and the ~15 ms copy-overhead figure under .tasks/dense-q80-gemm/
  (e.g. wave1-nsplit/) before the STOP GATE.
- F2 NIT (record): 63-nodeid collection delta - grounded candidate = collection-time
  dir-scan parametrization in tests/models/test_quant_config.py (HF-cache +
  /mnt/nvme/models globs); harmless; run-2-authoritative justified. If a future gate
  shows another collection delta, diff `--collect-only` nodeid lists, not failure counts.
- F3 NIT (record): ~15 ms/chunk copy-overhead estimate verified (15.3-16.8 ms at
  1.6-1.8 TB/s effective) but not durably recorded; the durable 0.24-0.28 s target is
  gross (net ~0.22-0.26 s).
- F4 NIT (record): no test pins the GGUF-path WIRING - a reverted kda.py call site
  would fail no test (the kda_op test covers the non-GGUF passthrough; the 4 helper
  tests call kda_in_proj_forward directly). Low risk: the A/B wave's T46 liveness proof
  covers it on hardware. Optional: a spy test in tests/models/test_glm5_next_gguf.py.
- F5 FP (dismissed, recorded): the env read precedes the isinstance check, so a garbage
  knob value also fail-fasts on the non-GGUF/HF path. Consistent with the knob's
  fail-fast design (v3a pattern); no action.

## Overall verdict: SHIP

No BLOCKER, no MAJOR. The changeset is python-only +241/-2 across exactly 4 files,
byte-identical at default (knob=1 returns layer.forward(x) unchanged; non-GGUF layers
pass through), all 5 new tests green in the full gate, single env read + single call
site, mirror rule honored, out-of-scope untouched, and the gate's 16 failures verified
per-test against the documented baseline classes with no new family. Conditions:
F1 artifact recording is blocking for the STOP GATE only (not for launching the A/B
wave); triage should treat the 5 non-import failures (pinned_tensor UVA, qsa_fp8
[1-1-64], qwen4_ple :421, 2x glm_dsa OOR 102400>101376) as the documented baseline and
the prefill_hit_d2d non-firing as drift.
