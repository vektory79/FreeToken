# Merge round 3 adaptation log (origin/main 6410c97 -> vektory79 @ 1445fc1)

Date: 2026-09-18. BASE = cac247a. Target = origin/main (6410c97), 2 new commits:
- 6410c97 fix(engine): raise WeightLoadError when the checkpoint cannot be read (#518) -> python/freetoken/engine/engine.py (74 lines)
- 1d8a541 fix(kernel): pin tvm-ffi jit arch to the bound gpu (#521) -> python/freetoken/kernel/utils.py (89 lines)

## Textual conflicts: ZERO

`GIT_EDITOR=true git merge main --no-commit` -> ort auto-merged both files, no conflict
hunks, no conflict markers (grep over python/ tests/ docs/ empty, rc=1). No manual line
decisions were required; every change below is a VERIFIED AUTO-UNION, logged as the
"conflict decision" record the protocol requires (file:line refer to the merged tree).

## Semantic verification of the auto-union (merged engine.py, 2324 lines)

1. Union kept BOTH sides - engine.py:899-922 (`_init_offload_moe_cache`):
   - Branch side: "Fast path / Slow path" comment block (899-904), `expert_parallel`
     selection, `requested_residency` split-residency list, and the slot-floor gate call
     `self._gguf_auto_floor_gate(config, method, self._resolve_auto_moe_cache_size)` at
     :919 (gate def at :839, with ftw-index reads `gguf_signature_groups` :849 /
     `gguf_ftw_signature_groups` :853).
   - Main side: `with _weight_load_context():` wrap around the shared
     `banks = load_expert_banks(...)` call at :921-932.
   - Decision: UNION (both). Main's wrap covers our gguf fast-path bank load;
     our gate call is outside the wrap (gate is a header-only pre-load check, not a
     checkpoint read).
2. Engine init - engine.py:411 `self._load_weights(config)` + :440-461:
   - Main side: new `_load_weights`/:585-593 (`_weight_load_context` around
     `load_state_dict`, `finalize_quant` outside it, vision fail-fast gains
     `not config.use_dummy_weight`) and `load_host_tables` wrap at :459-461.
   - Branch side: multi-partition inits `self.moe_offload_caches: list` :452 and
     `self.cpu_moe_executors: list` :454 survived.
   - Decision: UNION.
3. `_adjust_ftw_quant_backend` - engine.py:1856-1875:
   - Main side: `with _weight_load_context(): fmt = ftw_quant_format(model_path)` :1859.
   - Branch side: gguf passthrough `if kind is None: return quant_backend` :1871-1872
     with its comment.
   - Decision: UNION.
4. kernel/utils.py: clean, no overlap ever. `_pin_tvm_ffi_arch_ctx` + `ARCH_LIST_ENV`
   pinning landed at :43-48, wired at :245 and :303. Decision: MAIN as-is.
5. `load_gguf_moe_expert_sources`: lives in python/freetoken/models/weight.py:283
   (untouched by main), consumed from python/freetoken/moe/expert_banks.py:211
   (gguf fast path inside load_expert_banks). Untouched; structurally independent.
   Decision: BRANCH as-is.

## Duplicates (main-wins): NONE

Main's 2 commits add only WeightLoadError classification and a kernel jit arch pin -
no overlap with any branch functionality (ftw-index signature groups, slot-floor gate,
gguf fast path, cpu executors). Nothing discarded.

## Review finding (accepted gap, deliberately NOT fixed)

Main's `_weight_load_context` does NOT cover the ftw-index reads inside our
`_gguf_auto_floor_gate` / `_resolve_auto_moe_cache_size` (:839, :919 and the
resolve implementation). CORRECTED BY REVIEW pass A (2026-09-20): an unreadable
ftw index there does NOT raise a raw error - the scans degrade silently to None by
contract (moe/expert_banks.py:285/:406/:517 each catch Exception and return None),
so the gate falls through to the late split check; behavior is byte-identical to
pre-merge branch. Per instruction: leave behavior as the union of both
sides; expanding their wrap around our pre-load planning reads would change
failure-classification semantics without a main-side counterpart and would
contradict main's own semantics (config failures keep their own type). Recorded
for review; no code action.

## Staged state after merge

`git diff --cached --stat`: engine.py 74, kernel/utils.py 89 (113 insertions / 50
deletions) - exactly main's delta over our branch tree, confirming our-side content
is untouched and main-side content is fully applied.

## Recovery provenance (2026-09-20, continuation session)

The 18:41 merge (--no-commit) was left uncommitted by an interrupted prior run of this round
(boot-smoke-r3.* artifacts and the original sections above are that run's work). The continuation
session re-verified instead of abort+redo:

- stash list held exactly one entry, `round3-veai-memory` - the mandated pathspec stash created by
  the prior run (not a foreign entry); 9 .veai/memory files re-dirtied by parallel memory sessions
  after the stash -> treated as newer content, stash reconciled at post.
- staged index tree == `git merge-tree --write-tree HEAD 6410c97` result (tree d1f2313d,
  blob-identical: engine.py 0059ebaf, kernel/utils.py cdd765c7) -> the pending merge is exactly a
  clean ort auto-union; no manual edits existed to audit.
- conflict-marker grep over python/ tests/ docs/ -> empty. Semantic markers re-checked in the
  merged tree: WeightLoadError :333, _weight_load_context :348 with wraps :444 (load_host_tables),
  :592 (load_state_dict in _load_weights), :921 (load_expert_banks), :1859 (ftw_quant_format);
  branch side intact: _resolve_auto_moe_cache_size :804, _gguf_auto_floor_gate :839 and call :919,
  Fast/Slow path comments :902-903; load_gguf_moe_expert_sources weight.py:283 untouched;
  kernel/utils.py ARCH_LIST_ENV :26 / _pin_tvm_ffi_arch_ctx :43 wired :245 and :303.
- Tests and boot-smoke re-run/verified -> test-results.md; commit finalized by this session.

## Finalization + post (2026-09-20 19:2x, continuation session)

- Commit: 5001504e21d1d62f5c7d830daae0497d4fca4b10, parents 1445fc1 + 6410c97, subject exactly
  "Merge branch 'main' into vektory79" (rc=0, no amend needed).
- Invariants: `git rev-list --count HEAD --not 1445fc1 origin/main` == 1; `git diff --name-only
  HEAD^1 HEAD` == exactly engine.py + kernel/utils.py, zero .veai paths (note: `git show
  --name-only` on a merge commit prints only engine.py because utils.py is identical to its
  second parent - blob cdd765c7 verified via ls-tree); no MERGE_HEAD, 0 unmerged paths.
- Stash reconciliation: the `round3-veai-memory` stash held ONLY `lastRecall:` metadata lines
  (12 files). 9 were superseded by newer parallel-session timestamps already in the worktree;
  the other 3 (ft-pytest-worktree-baseline-gotchas, lean-subagent-context,
  merge-main-into-vektory79-skill) existed only in the stash -> restored into the worktree via
  `git restore --worktree --source=stash@{0}` (index untouched), verified byte-equal against the
  stash, then `git stash drop` (lossless). Final: stash list empty; 12 .veai/memory files dirty
  (lastRecall churn), nothing else new, nothing staged.
- Final census: no pytest/ft serve/benchbw/ft_run processes; sem.mp- = 18 (standing baseline);
  GPU 1773/32607 MiB (desktop compute apps only). No push, no rebase, no branch deletions;
  backup branch backup/vektory79-pre-merge3-1445fc1 retained.

## Review triage (2026-09-20, orchestrator)

Review passes: A correctness (strong), B edge/invariants. Both PASS, no blocker/major
on the merge. Triage (TP = fix applied, FP = no action):

- A1 adaptation-log wording (ftw-index gap framed as "raises a raw error") -> TP,
  artifact-only: corrected in place above; code unchanged (scans are silent-None
  by contract). No amend needed (this log is not committed).
- A2/A3 finalize_quant outside wrap; ValueError reclassification -> FP: main's
  intended #518 scoping; verified no catcher depends on the old types.
- B1 minor: runner stdout (the "/ready 200 after Ns" verdict line) not archived;
  /ready evidence is indirect (runner exits rc=1 before client probes without a
  200 + archived chat POST 200). TP as a process lesson: future rounds tee runner
  stdout into an artifact. Provenance judged sufficient (blobs re-verified).
- B2 SIGINT-grace-expiry escalation in smoke shutdown -> FP: known background-job
  SIG_IGN harness gotcha, not a merge regression.
- B3/B4 #518/#521 ship without tests -> note-level gaps, no action this round
  (main's commits; candidate hosts: tests/engine/test_weight_load_error.py,
  tests/kernels/test_gguf_quant.py:930).
- B6 _pin_tvm_ffi_arch_ctx process-global env mutation -> FP: one-shot startup
  JIT build, info-level.
- B7 typo in test-results.md (log line count 52 vs 60) -> TP, artifact-only: fixed.

No code TPs -> merge commit 5001504 NOT amended; parents preserved.
