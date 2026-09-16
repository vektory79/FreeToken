---
name: "ft-pytest-worktree-baseline-gotchas"
description: "FreeToken pytest: uv worktree editable trap, import-mode baseline fails (python -m green), pgrep census match"
type: project
lastUpdated: 2026-09-15T23:32
lastRecall: 2026-09-16T17:09
---

# FreeToken pytest/worktree gotchas (merge wave 2026-09-15)

Discovered while merging main into vektory79 (artifacts: .tasks/rebase-onto-main-work/). Complements ft-serve-test-and-e2e-gotchas.

## uv + git worktree trap
`uv pip install` inside a git worktree silently re-points the MAIN .venv's editable freetoken to the worktree path (took ~1 min of confusing results to notice). **How to apply:** in worktrees run pytest with an explicit interpreter (`-p <main>/.venv/bin/python`) or a dedicated venv, and verify `freetoken.__file__` before trusting any test outcome.

## Import-mode false failures
11 of the ~16 "known-failing" tests are cwd import-mode artifacts, NOT code failures: console `pytest` does not put the repo root on sys.path, so `from tests.` imports fail with ModuleNotFoundError; `python -m pytest` makes them green. No `tests/__init__.py` in repo (adding one would fix the class).

**Why it matters:** naive failure counts overstate real breakage; triage must separate import-mode noise from code failures.

**How to apply:** when a test wave reports failures, first reproduce with `python -m pytest <ids>`; only non-import-mode failures count.

## Verified baseline (RTX 5090 box, 2026-09-15)
A/B worktree runs pre-merge 6e673ab vs merge f786d7c: byte-identical failure profile, no merge regressions. Real (non-import-mode) baseline failures: pinned_tensor UVA driver error, qsa_fp8 CUDA invalid argument :48, qwen4_exp/test_ple flaky bitwise :421, 2x glm_dsa OutOfResources shared-mem 102400>101376. test_offload::graph_capture was a pre-existing branch defect (fixed in merge commit via duck-typing in engine/graph.py).

## Hygiene census nuance
`pgrep -af 'pytest|ft serve|benchbw'` matches the census wrapper's own shell - harmless; filter it out before flagging leftover processes.
