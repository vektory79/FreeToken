# Discovery report: merge round 2 (origin/main -> vektory79)

Read-only discovery, repo /media/ai/src/FreeToken. Commands ran with --no-pager.
Executed: `git fetch origin` (succeeded, network OK). No merge/rebase/checkout/stash/commit/push performed.

## Current state

- Branch: `vektory79`, HEAD = **8c0f1e7 "Memory"** - NOT 8bcca7b as assumed in the task context.
  Topology: `f786d7c (merge main afd99cb)` -> `8ce2657` (guard hf_config setattr) -> `8bcca7b` (gguf slot-floor fail fast) -> `8c0f1e7 "Memory"`.
  8c0f1e7 touches ONLY `.veai/memory/*.md` (17 files) - benign, no conflict risk (new commits never touch .veai). 8bcca7b and 8ce2657 are both ancestors of HEAD.
- Working tree: CLEAN. `git status --porcelain=v2 --branch` printed only branch headers, zero dirty entries (not even .veai).
- No upstream configured for vektory79 (no `# branch.ab` line); local `main` = afd99cb, "[origin/main: behind 2]".
- Stash: **see State flags - a stash entry appeared then vanished mid-session.** Net current state: empty.
- In-progress ops: none. MERGE_HEAD / CHERRY_PICK_HEAD / REVERT_HEAD / BISECT_LOG / rebase-merge / rebase-apply all absent (verified twice: .git paths + git-native rev-parse). `.git` is a real directory (not a worktree link).

## Target resolution

- User said "origin/master", but that ref DOES NOT EXIST: `git rev-parse --verify origin/master` -> MISSING, `master` -> MISSING (also after fetch --prune).
- `origin/main` EXISTS = **cac247a**, advanced `afd99cb -> cac247a` (+2 commits, `rev-list --count afd99cb..origin/main = 2`).
- **Actual target = origin/main (cac247a)**. Unambiguous - only one candidate ref exists and it is exactly the one that moved. No NEEDS-USER-DECISION required; the "origin/master" wording was just stale (repo default branch is main).
- Fetch side effects to know about: new remote branch `origin/chore/release-0.1.3` and tag `v0.1.3` appeared.

## New commits (afd99cb..origin/main)

1. **cade1a9** `fix(checkpoint): serve and repair FTW checkpoints around the vision encoder (#486)` - 23 files:
   - docs/ftw-hotfix.md, docs/models.md
   - python/freetoken/checkpoint/ftw.py
   - python/freetoken/engine/engine.py
   - python/freetoken/models/{gemma4,glm5_next,minimax_m3,muse_glimmer,qwen3_5_moe,qwen3_vl,qwen4_exp}/__init__.py
   - python/freetoken/models/{gemma4,glm5_next,minimax_m3,muse_glimmer,qwen3_5_moe,qwen3_vl,qwen4_exp}/weight.py
   - python/freetoken/models/weight.py
   - scripts/ftw_hotfix.py
   - tests/checkpoint/test_ftw_hotfix_vision.py, tests/checkpoint/test_ftw_weights.py
2. **cac247a** `chore(release): 0.1.3 (#489)` - 1 file: python/freetoken/version.py ("0.1.2" -> "0.1.3"; our branch never touched version.py since afd99cb -> auto-merge trivial).

Context (log -15 origin/main): the range stops at cade1a9; 08d728d (the hf_config setattr commit from round 1) is below the base, already merged.

## Overlapping files (new-commit files INTERSECT our diff afd99cb..HEAD) + whose commits

Our full delta since base = 123 files (merged glm5next-gguf delta via f786d7c + our 2 fix commits). Intersection = 3 files:

| File | New side (cade1a9) | Our side - last 2 fixes (8ce2657+8bcca7b)? | Our side - older merged delta (f786d7c etc.) |
|---|---|---|---|
| python/freetoken/engine/engine.py | +7: import `ftw_lacks_vision` (~line 19) + fail-fast raise `if config.active_encoders and ftw_lacks_vision(...)` in the weight-materialization area (~501-511) | **YES - 8bcca7b touched engine.py** | also f786d7c merged-delta changes |
| python/freetoken/models/glm5_next/__init__.py | +3/-1: `.weight` import gains `iter_vision_weights`; `__all__` gains it mid-list | no | YES: our gguf-import block + 4 gguf `__all__` entries (content of 4162f7c/4443d7e/f801a08, merged via f786d7c) |
| python/freetoken/models/weight.py | +28/-3: load_weight keep-filter (skip vision keys for text-only), new `load_vision_weight` + `ftw_lacks_vision` funcs, 2 `__all__` entries | no | YES: our `load_gguf_moe_expert_sources` func + `__all__` entry (merged delta) |

Collision analysis (3-way):
- **engine.py**: new hunks (import line ~19, ~501-511) vs our 8bcca7b hunks (~726-870: `_split_moe_cache_budget` safety net, `_resolve_auto_moe_cache_size` signature, new `_gguf_auto_floor_gate`, its call in `_init_offload_moe_cache`). Regions distinct -> auto-merge likely, but watch the import block (our merged delta also edits imports).
- **glm5_next/__init__.py**: both sides edit the same top import block (ours inserts `.gguf` import after line 1, theirs modifies the `.weight` import line) and the same `__all__` (ours appends at end, theirs inserts mid-list). Insert-vs-modify at adjacent lines -> auto-merge probable but this is the most likely small conflict; resolution = union (keep gguf imports AND iter_vision_weights).
- **weight.py**: theirs rewrites load_weight body + inserts 2 new functions between load_weight and load_q4_0_moe_expert_sources; ours inserts load_gguf_moe_expert_sources between load_q4_0... and __all__, plus both edit __all__ at different anchors. Different anchor lines -> auto-merge probable; if it conflicts, union of functions + union of __all__.

## Risk zones (lesson from last merge)

- (a) **python/freetoken/engine/config.py or mm/config.py (hf_config mutation)**: **NOT touched.** Verified `diff afd99cb..origin/main --` over engine/config.py, engine/cache_budget.py, models/gguf, scheduler, moe -> empty; corrected path check for real `python/freetoken/mm/config.py` (models/mm/config.py does not exist) -> empty. New engine.py only READS `config.active_encoders` (raise condition) - no setattr anywhere in the new range. Frozen-shim hazard NOT re-triggered by source changes.
- (b) **models/gguf/***: **NOT touched.** New weight-loading changes are in models/weight.py + per-family weight.py; gguf boots read weights via the gguf reader, not `load_weight`/FTW. New engine.py raise fires only for `active_encoders` non-empty AND FTW checkpoint lacking vision tensors - gguf/FTW-textless path unaffected. Residual risk low, but the standing rule stays: GGUF boot smoke after the merge (imports in glm5_next/__init__.py are part of the gguf load path - a bad merge there WOULD break gguf boots).
- (c) **engine/cache_budget.py / MoE planner (our 8bcca7b slot-floor gate)**: **NOT touched.** cache_budget.py, scheduler, moe all absent from the new range. Our `_gguf_auto_floor_gate` / `check_partition_floors` chain is untouched by main; the only shared file is engine.py and the new hunks are far from the gate. Note the new fail-fast raise sits in a different method (~line 501) than the gate (~870) - no interaction.

Relevant new-side diff quotes:
- engine.py: `+from freetoken.models.weight import ftw_lacks_vision` and `+ if config.active_encoders and ftw_lacks_vision(config.model_path): + raise ValueError(... "holds no vision encoder tensors ... Reconvert it with `ft checkpoint`, add the encoder in place with scripts/ftw_hot.py (docs/ftw-hotfix.md), or start with --text-model-only")`
- weight.py: keep-filter `keep = None if include_vision else (lambda name: not name.startswith(VISION_KEY_PREFIXES))`, `iter_ftw_weights(model_path, keep=keep)` signature change (checkpoint/ftw.py also modified), new `ftw_lacks_vision()` and `load_vision_weight()`.

## State flags

- HEAD is 8c0f1e7 "Memory", not 8bcca7b - deviation from the assumed context, benign (.veai-only commit on top of 8bcca7b).
- Working tree clean; stash empty (see below); no in-progress git operations.
- STASH ANOMALY (transient, resolved): the first state check listed `stash@{0}: WIP on vektory79: 8c0f1e7 Memory`; seconds later `git stash show stash@{0}` -> "not a valid commit ref", `git rev-parse stash@{0}` -> "Needed a single revision", `git reflog stash` -> ambiguous/absent, `.git/refs/stash` and `.git/logs/refs/stash` do not exist, and `git stash list` is now empty. Interpretation: the entry referenced a missing/pruned commit object and the stale stash ref disappeared during the session (likely concurrent cleanup by another agent/IDE session - this is a multi-agent campaign). Nothing was touched by me (read-only). Net state matches expectation (empty), but re-verify `git stash list` immediately before the merge; if some stash WIP content mattered it is unrecoverable via stash - however the tree is clean, so nothing was lost from HEAD.
- Local `main` = afd99cb, behind origin/main by 2 (ff-only `git fetch origin main:main` will work later if desired).
- New remote refs after fetch: `origin/chore/release-0.1.3`, tag `v0.1.3`.

## merge-base

- `git merge-base HEAD origin/main` = **afd99cb** (as expected: f786d7c already merged afd99cb; 8c0f1e7 and 8bcca7b/8ce2657 sit on top of it). So the merge will replay exactly the 2 new commits as delta.

## Recommendations

1. Target = **origin/main (cac247a)**, 2 new commits, merge-base afd99cb -> `git merge origin/main` (user rule: merge, not rebase; history not rewritten).
2. Backup branch from current HEAD (8c0f1e7) before the merge, per the established protocol.
3. Conflict expectation: LOW. 3 overlapping files; engine.py hunks are in disjoint regions; glm5_next/__init__.py is the most likely (small) conflict - resolve by union (keep our gguf imports/__all__ AND their iter_vision_weights); weight.py same union principle if it conflicts. Per user rule: these new files are NEW main functionality (vision-encoder hotfix), not duplicates of ours -> keep both sides, no "main wins" discard needed.
4. No risk-zone source changes (a/b/c all clean) -> no mandatory code adaptations; still run the standard post-merge GGUF boot smoke (glm5next gguf serve boots through glm5_next/__init__.py which both sides edit) plus `tests/checkpoint/test_ftw_weights.py`, `tests/checkpoint/test_ftw_hotfix_vision.py` (new), `tests/models/test_gguf_expert_banks.py` (our gate pins), `tests/models/test_glm5_next_gguf.py` (our frozen-shim guard).
5. version.py will auto-merge to 0.1.3 - no action.
6. Before merging: re-check `git stash list` is empty and `git status --porcelain` clean (stash anomaly documented above).
