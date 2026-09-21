# Review-B - Replan wave 2: Candidate A, dense q8_0 m-tile template + FREETOKEN_GGUF_DENSE_MTILE

Reviewed 2026-09-19, protocol Review-B (integration / gate integrity / sweep readiness / operability).
Read-only; this file is the only artifact written.

Scope audited: uncommitted changeset on vektory79 @ HEAD 4505572 (5 paths, +215/-20 by numstat:
gguf_kernel.cu 84/10, mmq.cuh 12/8, kernel/gguf.py 6/0, layers/moe.py 5/2, tests/kernels/test_gguf_quant.py 108/0)
plus 4 worktree .veai/memory deltas (parallel-session churn, not the changeset). Zero staged changes; stash empty.

Evidence base: full per-file diffs read; git log / status / stash / numstat; diff 9f09596..4505572 read;
gate-nsplit-run2.log header + review-b-nsplit.md B3 (wave-1 baseline); wave2-candidate-a.md;
TASK.md; orchestration-notes.md (T44/T46, protocol).

## B1 diff hygiene - CONFIRMED

- numstat = exactly the claimed 5 paths, python/csrc-only + 1 test file. The "4 files + tests"
  phrasing reconciles as 4 code files (mmq.cuh, gguf_kernel.cu, kernel/gguf.py, layers/moe.py)
  + 1 test file; moe.py is docstring-only (its 2 deleted lines are rewritten docstring lines).
- moe.cuh and setup.py: ZERO diff (verified by explicit `git diff --` on both paths, empty output).
- MoE dispatch untouched: in gguf_kernel.cu the only dispatch change is `case 8` inside
  ggml_mul_mat_a8 (the dense path); ggml_moe_a8 / ggml_moe_a8_vec / ggml_moe_get_block_size bodies
  unchanged. The FT_DENSE_MMQ_ARGS macro reproduces the old case-8 argument list verbatim
  (col, row, batch, padded, row, stream) - byte-identical call semantics.
- pybind additions minimal: one probe fn ggml_dense_get_mtile + one m.def line. kernel/gguf.py
  adds one wrapper + exactly one __all__ entry. mmq.cuh = template param hoist + 2 static_asserts
  + constexpr need_check; nothing else. Single source of truth holds: dispatch and the test probe
  both read ft_gguf_dense_mtile(); ROCm branch pins MMQ_X_Q8_0 == 64 and bypasses the knob.

## B2 HEAD/tree - PARTIALLY REFUTED (MAJOR)

- HEAD 4505572; log chain 4505572 "Memory" -> 9f09596 "default FREETOKEN_GGUF_KDA_NSPLIT=2" ->
  5254ffa "split kda in_proj gemm" -> cb8db23 "Memory" -> 2d64a58 "Skill" -> 1706aae. Matches claims.
- diff 9f09596..4505572 = 41 files: 40 under .veai/memory + **pyproject.toml +1**. The wave report's
  ".veai-only Memory commit" claim is REFUTED. The hunk is `+    "ipykernel>=7.3.0",` in core
  dependencies - the exact line wave 1 classified as stray ("origin unknown, NOT part of the
  campaign") and deliberately left uncommitted. It is now committed inside a bookkeeping commit.
- The narrow claim that matters for the gate DOES hold: no python/ paths in the delta, so the
  python tree is identical 9f09596 -> 4505572, and the changeset worktree carries no pyproject mod.
- Gate-internal validity unaffected: pyproject.toml is committed at 4505572, so BOTH legs of the
  wave-2 gate (changeset tree and stash-repro at HEAD) include it; the wave-1 gate ran with the
  same line present as an uncommitted worktree mod. Cross-gate comparability intact.
- Stash-repro mechanics: plausible and coherent (stash the 5 paths at HEAD 4505572 -> run -> pop;
  stash list empty now, consistent; 2.93 s runtime and disjoint failing files imply no JIT rebuild).
  But the evidence is verbal-only: no stash-repro log exists (same caveat as wave-1 B3/F1).

## B3 gate classification - NEED-MORE-DATA (per-test nodeid identity)

- Arithmetic consistent: claimed classes 10x import (dsa_kpool x7, dsa_pool x2, muse_glimmer x1)
  + pinned_tensor UVA + qsa_fp8 [1-1-64] + 3x glm_dsa OOR + ple boundary = 16; zero gguf-path
  failures; the 4 new tests absent from the failed list. All 16 fall in the documented baseline
  families either way, so "all Environment" is plausible.
- Class composition SHIFTED vs the wave-1 baseline (gate-nsplit-run2.log / review-b-nsplit.md B3):
  import 11x -> 10x, glm_dsa OOR 2x -> 3x. The shifted id is almost certainly
  tests/models/test_glm_dsa.py [nvfp4] (import class at :316 in wave 1) re-attributed into the
  glm_dsa OOR class - both classes live in the same file - or a collection variant shift. This is
  NOT explained anywhere in the wave report (the report does not mention the shift), and the
  collection-drift memory mechanism (test_quant_config globs) explains count drift with IDENTICAL
  failure sets, not class re-attribution. Without a wave-2 gate log there is no nodeid list to
  diff; the "SAME 16 nodeids on the stash-restored tree" claim is likewise verbal (2.93 s, no log).
- Not gate-invalidating: both shifted classes are documented baseline families, the failing files
  are disjoint from all 5 changed paths, and the stash-repro isolates the changeset. But the
  wave-1 F1 condition (durable gate log with invocation line + per-test nodeids under
  .tasks/dense-q80-gemm/) was NOT applied to wave 2: no gate log, no stash-repro log, no census
  record exists in the folder (only gate-nsplit-run2.log from wave 1).

## B4 gate delta accounting - CONFIRMED

- 2173 -> 2179 = +6 with failed (16) and skipped (206) flat. Diff-verified: exactly 4 new test
  functions, none parametrized (4 single nodeids) => +4 new + 2 from the documented
  collection-drift band (drift nodeids land in passed; consistent with the known mechanism).
  The +2 attribution is verbal but the mechanism is established.
- tests/kernels/test_gguf_quant.py diff is 108/0 - PURE ADDITION. No test deleted, no assertion
  weakened; existing dense/MMVQ/mmvq-cutoff tests untouched (diff context shows the block
  appended after test_kda_nsplit_decode_batch_stays_mmvq).
- New-test quality: env-knob test covers default/empty->4, per-call flips 8/16/32/64, cross-knob
  isolation BOTH directions, 8 stragglers fail-fast; scope pin (bs=4 vec / bs=7 MMQ with both
  knobs at 32); bitwise invariance via torch.equal over 3 shapes covering both need_check
  branches + M tails + production (4096,24896,256); NSPLIT composition pins launch structure
  [12448, 12448] + bitwise. Mirror rule honored (no new files).

## B5 "do not flip after boot" - CONFIRMED (one NIT)

- C++: ft_gguf_dense_mtile header comment states the dense rationale accurately: "decode CUDA
  graphs capture dense MMQ launches for capture batches > 6, so the knob changes graph contents".
  Matches engine/graph.py capture mechanics verified in wave-1 B6 (capture sizes >6 exist;
  dense MMQ route at batch > _MMVQ_SAFE=6).
- moe.py docstring: documents BOTH knobs with separate meanings - "Dense projections have a
  separate m-tile knob (FREETOKEN_GGUF_DENSE_MTILE, default 4, dense q8_0 only) ... the two
  knobs never affect each other's paths." No conflation. NIT: the closing "Do not flip the knob
  after boot" sentence retains only the MoE trio rationale (buffer/kernel read at different
  moments); the dense knob's boot hazard (graph contents) lives only in the C++ comment - the
  python-facing docstring warns only implicitly. Cosmetic; both warnings are truthful.
- NIT: the rewritten docstring line "One moe_align trio is shared by all three projections..."
  lost its 8-space indentation (starts at column 0 mid-docstring) - pure formatting slip,
  no behavior impact.

## B6 sweep readiness - REFUTED AS WRITTEN (MAJOR, next-wave dependency)

- TASK.md Candidate A A/B line still says "dense launch counts stay ~312/chunk" - STALE twice
  over: the MTILE=32 census was 322/chunk (TASK.md Replanning), and since 9f09596 the NSPLIT=2
  default doubles kda launches: 34 calls -> 68/chunk. Corrected anchor: total dense MMQ
  ~356/chunk = 68 kda sub-N launches + ~288 other dense. A sweep T46 check run against the
  stale anchor would spuriously FAIL.
- The wave report carries NO corrected liveness anchors - its notes only say the A/B sweep is
  the next wave. The only durable launch-structure pin is the unit test ([12448, 12448]).
- Corrected sweep liveness anchors the orchestrator must carry into the sweep brief:
  - total dense MMQ launches ~356/chunk, CONSTANT across tiles 4/8/16/32/64 (tile changes grid
    dims, not launch count);
  - kda stays 68/chunk (2x34; NSPLIT is N-side, tile is M-side; each half-launch keeps
    block_num_x = 389 = 12448/32);
  - per-launch grid.y = ceil(M/tile) shrinks by the tile factor: M=8128 -> 2032/1016/508/254/127;
  - decode: at max-running-requests=1 the replayed bs=1 graph is MMVQ vec (batch <= 6) - neither
    knob enters it; decode expected flat (wave-1 measured 13.35 vs 13.61), re-measure stays
    cheap/mandated; graph re-capture at boot is still the supported pattern.

## B7 census / JIT hygiene - CONFIRMED in substance (two NITs)

- No-repeat rebuild: first targeted run 49.49 s, immediate full-file rerun 80 passed in 4.61 s -
  a repeated source-hash rebuild would add minutes, so the 4.61 s rerun is decisive evidence of
  hash stability. NIT: the parenthetical "(includes the ONE JIT rebuild)" inside 49.49 s sits
  poorly against the documented ~3 min single-TU rebuild (ft-gguf-kernel-jit-toolchain); most
  likely the rebuild happened in an earlier untimed invocation. Does not affect any verdict.
- Census clean (no ft serve / nsys / pytest strays, 1.5 GB GPU baseline before/after) is claimed
  but not recorded anywhere durable - same artifact gap as B3.
- NIT: the ptxas mini-TU evidence backing the tile-64 inclusion (/tmp/ptxas_dense.log) lives in
  /tmp (TTL-expiring), not archived under .tasks/dense-q80-gemm/.

## B8 out-of-scope cross-check - CONFIRMED (clean)

The changeset touches none of the campaign's out-of-scope paths: no fetch-copy machinery
(fast_index_copy_multi untouched), no prefill overlap, no MoE re-tuning (moe.cuh zero-diff; MoE
dispatch bodies zero-diff; FREETOKEN_GGUF_MOE_MTILE semantics unchanged and explicitly
quarantined in the knob comment + a cross-isolation test in BOTH directions), no KV/pool changes.
layers/moe.py is docstring-only. Caveat: pyproject.toml (B2) landed out-of-band but OUTSIDE this
changeset - a dependency-declaration change committed by the parallel "Memory" session, user to
keep or revert; it must not ride along with the eventual Candidate A commit.

## Findings

- F1 MAJOR (disclose to user; decision needed): commit 4505572 labeled "Memory" is NOT .veai-only -
  it commits pyproject.toml `+    "ipykernel>=7.3.0",` (core deps), the line wave 1 explicitly
  classified as stray/origin-unknown and left uncommitted. Gate validity unaffected (both legs
  shared it; wave-1 gate ran with it as a worktree mod). Actions: inform the user (keep vs revert
  is a user-owned dependency decision); correct the wave report's ".veai-only" claim; ensure the
  eventual Candidate A commit stages ONLY the 5 changeset paths.
- F2 MAJOR (action before the sweep wave): stale T46 liveness anchor. TASK.md "~312 launches/chunk"
  is stale (census 322; NSPLIT=2 default -> kda 68 -> total ~356). The wave report does not correct
  it. The orchestrator must write the corrected anchors (B6 list) into the sweep brief before the
  A/B wave, or T46 will spuriously fail.
- F3 NIT (action before STOP GATE, wave-1 F1 recurrence): no durable wave-2 gate log, no
  stash-repro log, no census record, ptxas probe in /tmp. Record gate log WITH invocation line +
  per-test nodeids + census under .tasks/dense-q80-gemm/ before closing the wave.
- F4 NIT (record): class-composition shift import 11x -> 10x, glm_dsa OOR 2x -> 3x vs the wave-1
  baseline is unexplained (candidate: the test_glm_dsa [nvfp4] id re-attributed between classes,
  or collection variant drift). F3's log resolves it; classification "all Environment" is not at
  risk (all 16 in documented baseline families, failing files disjoint from the changeset).
- F5 NIT (polish): moe.py docstring de-indented line; dense-knob boot rationale present only in
  the C++ comment. One-line fix wave or session-end polish.
- F6 NIT (record): "one JIT rebuild inside the 49.49 s targeted run" is doubtful vs the documented
  ~3 min rebuild; hash stability itself is proven by the 4.61 s rerun.

## Overall verdict: SHIP (changeset), with conditions

No BLOCKER. The changeset is minimal and exactly as advertised: dense q8_0-only template + knob,
moe.cuh/setup.py/MoE dispatch untouched, byte-identical default behavior (tile 4 == historical
macro pin), probe is the same single source of truth, mirror rule honored, pure-addition tests,
out-of-scope clean. Conditions: F1 (pyproject disclosure + user decision) before any commit
activity touches HEAD; F2 (corrected sweep anchors into the sweep brief) before launching the
A/B wave; F3 (artifact recording) before the STOP GATE. F4 closes with F3's log; F5/F6 record-only.
