# Adaptation log: merge round 2 (main cac247a -> vektory79 @ 8c0f1e7)

## Preflight deviations from the task text

- Working tree was NOT clean at preflight: 8 dirty files, all under `.veai/memory/` (memory
  updates from other sessions of the multi-agent campaign; `.veai` is committed in the branch).
  Applied the documented protocol (memory: vektory79-main-rewrite-gotcha): pathspec-stash of
  only those files (`git stash push -m "merge2: veai-memory-wip" -- .veai/memory/`), then the
  tree was clean. `git stash list` BEFORE our stash was EMPTY (discovery's vanished-stash
  anomaly not reproduced). Our stash will be popped after merge finalization.
- Backup branch: `backup/vektory79-pre-merge2-8c0f1e7` -> 8c0f1e7.
- Baseline artifacts: `commits-before.txt` (57 commits afd99cb..HEAD), `pre-merge-our-delta.stat`.
- `git fetch origin main:main`: ff-only OK, local main afd99cb -> cac247a (+2 commits).
- First `GIT_EDITOR=true git merge main` auto-committed immediately (zero conflicts, so there
  was no manual finalization point). To honor the "tests BEFORE finalizing the merge commit"
  rule the auto-commit 30ad06d was rewound (`git reset --hard 8c0f1e7`) and the merge redone as
  `git merge --no-commit main` - identical auto-merge result, MERGE_HEAD = cac247a. The only
  history in the final result is ONE merge commit with parents (8c0f1e7, cac247a).

## Conflicts

ZERO conflicts. The 'ort' strategy auto-merged all 3 discovery candidates:
- python/freetoken/engine/engine.py (7 new main-side lines in regions disjoint from our
  slot-floor gate hunks)
- python/freetoken/models/glm5_next/__init__.py
- python/freetoken/models/weight.py

## Post-auto-merge semantic checks (per task step 4)

### python/freetoken/models/glm5_next/__init__.py - UNION, verified in tree
- Our `.gguf` import block intact (iter_gguf_expert_sources, iter_gguf_weights,
  load_gguf_expert_sources, parse_gguf_config).
- Main-side `.weight` import now carries `iter_vision_weights` alongside our
  iter_expert_pieces/iter_weights/nvfp4_expert_spec.
- `__all__` is the union: 4 gguf entries + `iter_vision_weights` + rest. Verified by read.

### python/freetoken/models/weight.py - UNION + CRITICAL semantic check: PASS
- Main-side rewrite of `load_weight`: keep-filter
  `keep = None if include_vision else (lambda name: not name.startswith(VISION_KEY_PREFIXES))`
  applies ONLY inside load_weight (FTW branch + per-family iter_weights branch).
- Our `load_gguf_moe_expert_sources` (line 283, resolver for the family
  `load_gguf_expert_sources` hook) is INTACT and structurally independent: gguf boots do not
  route through `load_weight` at all. Main did NOT replace/duplicate that functionality -
  the two sides are disjoint filters on different load paths -> union was the correct
  resolution (no "main wins" discard needed, no functionality disabled).
- New main functions `load_vision_weight`, `ftw_lacks_vision` present; `__all__` is the
  union of {load_weight, load_vision_weight, ftw_lacks_vision, load_q4_0_moe_expert_sources,
  load_gguf_moe_expert_sources, iter_expert_tensors_parallel}.

### python/freetoken/engine/engine.py - both sides present, verified by search
- Main-side import `from freetoken.models.weight import ftw_lacks_vision` at line 19 and the
  fail-fast `if config.active_encoders and ftw_lacks_vision(config.model_path): raise
  ValueError(...)` in the weight-materialization region (~line 562).
- Our 8bcca7b slot-floor gate intact: `_gguf_auto_floor_gate` def at line 807, call at 882.
  Regions disjoint -> no interaction. GGUF path: `active_encoders` empty for gguf/textless ->
  new raise never fires there (discovery check (b)).

## Duplicates ("main wins" rule)

NONE. cade1a9 is new main functionality (FTW vision-encoder serve/repair, #486), disjoint
from our gguf/ slot-gate work; no duplicated functionality found.
