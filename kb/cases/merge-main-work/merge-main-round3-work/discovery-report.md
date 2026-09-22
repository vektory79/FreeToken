# Discovery report: merge round 3 (origin/main -> vektory79)

Read-only discovery, repo /media/ai/src/FreeToken, 2026-09-18. Executed `git fetch origin` (succeeded, network OK, rc=0). No merge/rebase/checkout/stash/commit/push/reset performed.

## State flags

- Branch: `vektory79` (matches expectation). HEAD = **1445fc1 "Memory"** (a .veai-only commit on top of 8f05892 "Skill").
- `git status --porcelain`: 9 modified files, ALL under `.veai/memory/` (continuation of HEAD's memory commit by parallel sessions). Nothing staged, nothing untracked, nothing dirty outside .veai -> matches expectation.
- Stash: run 1 = empty, run 2 = empty -> stable, no vanishing-stash anomaly this round (unlike round 2).
- In-progress ops: none. `.git/rebase-apply`, `.git/rebase-merge`, `.git/MERGE_HEAD`, `CHERRY_PICK_HEAD`, `REVERT_HEAD` all absent.
- Worktrees: single worktree `/media/ai/src/FreeToken` on `[vektory79]` @1445fc1.
- No upstream configured for vektory79 (unchanged).
- Local `main` ref = 6410c97 = origin/main (`git rev-list --count main..origin/main` = 0) - a parallel session already fast-forwarded it; harmless.

## Fetch + remote refs

- Fetch OK. `origin/main` EXISTS, tip = **6410c97e15d789f38262083cd386404dccd9dd7a** "fix(engine): raise WeightLoadError when the checkpoint cannot be read (#518)".
- `origin/master` DOES NOT EXIST (`git rev-parse --verify origin/master` -> fatal). Same situation as round 2 ("master" wording is stale; repo default branch is main).

## Target resolution

- Round-2 merge commit 5d93a30 (2026-09-16) parents = [8c0f1e7, **cac247a**] - second parent cac247a is the main tip that was merged in round 2.
- `cac247a` IS an ancestor of `origin/main` -> origin/main is the unambiguous continuation of the main line.
- origin/master absent -> **no NEEDS-USER-DECISION**. Target = **origin/main @ 6410c97**.

## Merge-base and new commits

- BASE = `git merge-base HEAD origin/main` = **cac247a860e316e06580d05aeb05f2e647bde214** (= round-2 target, as expected: f786d7c/5d93a30 already merged it).
- New commits: **2** (cac247a..origin/main):
  1. **6410c97** `fix(engine): raise WeightLoadError when the checkpoint cannot be read (#518)` (2026-09-18) - engine.py: new module-level `WeightLoadError(RuntimeError)` + `_is_resource_failure()` (OOM/ENOMEM/DefaultCPUAllocator pass through, everything else re-wrapped) + `_weight_load_context()` context manager; Engine init refactored: `load_state_dict` + `finalize_quant` moved into new `Engine._load_weights()`; `load_host_tables`, slow-path `load_expert_banks`, and `ftw_quant_format` (in `_adjust_ftw_quant_backend`) each wrapped in `_weight_load_context()` so /health reports WeightLoadError for unreadable checkpoints; vision fail-fast check gains `not config.use_dummy_weight` condition.
  2. **1d8a541** `fix(kernel): pin tvm-ffi jit arch to the bound gpu (#521)` (2026-09-18) - kernel/utils.py: tvm-ffi JIT arch pinning to the bound GPU.
- New-side files: only **2** (113 insertions, 50 deletions total):
  - python/freetoken/engine/engine.py (74 changed lines, 7 hunks)
  - python/freetoken/kernel/utils.py (89 changed lines)

## Conflict candidates

New-commit files (2) INTERSECT our delta BASE..HEAD (159 files) -> intersection = **1 file**:

| File | Recent branch commits touching it (top of `log BASE..HEAD -- <file>`) | Older merged-delta commits |
|---|---|---|
| python/freetoken/engine/engine.py | **YES**: ef14efb (ftw index -> gguf signature groups/bank types); below it the already-merged-in-branch lineage: 5d93a30, 8bcca7b (slot-floor gate), f786d7c, 979e3fc/eb7de4c/aad5d3a (cpu executors), f6b94ad, 78115df, f801a08/2637714 (gguf dispatch), PR #408 kv-quant cluster | - |

kernel/utils.py: our delta NEVER touched it (`git log`/`diff --stat` BASE..HEAD on that file are empty) -> clean auto-merge.

Our engine.py delta since BASE is large (696 lines: +611/-85, 32 hunks). Hunk-region comparison (old-file coords cac247a):

- Their hunks: 1-6 (imports +contextlib/errno), 272-278 (WeightLoadError block after `_make_dummy_weight_state_dict`), 330-338 (init load_state_dict -> `_load_weights`), 357-364 (`load_host_tables` wrap), 497-513 (`_load_weights` + `_load_weight_state_dict` restructure), 635-652 (`load_expert_banks` wrap), 1368-1375 (`ftw_quant_format` wrap).
- Overlap candidates with our hunks (old coords):
  1. **Init block (MEDIUM, likely still auto-merge)**: their 330-338 load_state_dict replacement vs our @@ -351,7 (+2 lines: multi-partition `self.moe_offload_caches` / `self.cpu_moe_executors` list inits). Our insert ends at old 356 with context to 357; their load_host_tables wrap modifies old 359 -> 3+ lines apart, shared context unchanged -> auto-merge expected, watch it.
  2. **`load_expert_banks` region (HIGHEST risk, small)**: our @@ -634,6 +882,9 inserts the 3-line "Fast path / Slow path" comment block directly before their wrapped region (their old 635-652 wraps the `banks = load_expert_banks(...)` call at old ~638-650). Adjacent/overlapping context -> this is the most likely (small) conflict. Mechanical resolution: keep our comments + our `_gguf_auto_floor_gate` call (current ~885), apply their `with _weight_load_context():` wrap around the unchanged call. Our big @@ -675,38 +926,127 block is AFTER their region (old 675 > 652) -> no overlap.
  3. **`_adjust_ftw_quant_backend` (LOW, likely auto-merge)**: their wrap of `fmt = ftw_quant_format(...)` (old ~1371, line unchanged in our tree) vs our @@ -1375,6 +1830,8 (2-line edit later in the same function, the `if kind is None` gguf comment/return). 4+ lines apart, same function -> auto-merge expected; verify after merge.
- Imports: theirs touches lines 3-4, ours ~20-27 -> no overlap.
- Lines 389-390 of current tree (`load_state_dict` + `finalize_quant`) are untouched by us -> their `_load_weights` refactor should apply cleanly.

## Risk zones (boot-smoke trigger check)

- (a) hf_config mutations (python/freetoken/engine/config.py, python/freetoken/mm/config.py): **NO** - not in the new range; the new diff contains zero setattr/config mutation (read the full new-side engine.py diff to verify).
- (b) models/gguf/*: **NO** - not touched.
- (c) MoE scheduler budget (python/freetoken/engine/cache_budget.py): **NO** - not touched. Note: their `load_expert_banks` wrap sits inside `_init_offload_moe_cache` where our `_gguf_auto_floor_gate` lives, but the gate call line itself (current ~885) is not modified by them.
- (d) Shared load paths (python/freetoken/models/weight.py, python/freetoken/models/glm5_next/{__init__,args,attention,config,gguf,kda,mlp,model,moe,vision,weight}.py): **NO** - not touched by the 2 new commits.

Semantic note (not a conflict): their `_weight_load_context` wraps the whole `load_expert_banks` call, so our gguf/FTW fast-path bank loads are covered by WeightLoadError classification; but our ftw-index reads in `_gguf_auto_floor_gate` / `_resolve_auto_moe_cache_size` / ef14efb's signature-group derivation are NOT wrapped -> an unreadable ftw index there still raises the raw error. Acceptable gap; mention during review.

## New tests introduced by main

- **None.** `git diff --name-only BASE..origin/main -- tests/` is empty; no test files in either new commit (6410c97 body lists 2 sub-commits, both fix-only; 1d8a541 no body).

## Anomalies

- Dirt confined to `.veai/memory/*` (9 files, unstaged) - expected parallel-session memory churn, no stale-index problem this round (nothing staged at all).
- Foreign `.veai/memory` commits on the branch since round 2 (known-benign pattern, do not touch): 8d8fd52 "Memoty" (typo), d1c48d6/315b278/4505572/cb8db23/ebf071b/1a444e4/7ed25e0/1445fc1 "Memory", 8f05892/2d64a58 "Skill".
- 27 commits on vektory79 since 5d93a30; real-code ones: ftw-gguf fastpath (8568807, e9ad38d, ef14efb, 2e9fcf1), kernel perf defaults (1706aae, 5254ffa, 9f09596, c464273, 0d81cba, 8e2e4c7, 7f8c570), scheduler mamba fix 5b72aba, skill/docs commits.
- Local `main` already equals origin/main (parallel session ff'd it) - no action needed; per protocol use ff-only if refetching.

## Recommendations (for the merge wave)

1. Target = **origin/main (6410c97)**, 2 new commits, BASE = cac247a -> `git merge origin/main` (user rule: merge, never rebase; keep as exactly one merge commit).
2. Backup branch from current HEAD 1445fc1 before merging (established protocol).
3. Conflict expectation: LOW-MEDIUM, single candidate file (engine.py). Watch 3 sub-regions; only the `load_expert_banks` comment-adjacency is realistically a conflict. Union principle: this is new main functionality, not a duplicate of ours -> keep BOTH sides; no main-wins discard needed.
4. No risk-zone source changes (a-d all clean) -> no mandatory code adaptations. Still run the standard post-merge GGUF boot smoke (standing rule; engine.py is on the boot path for everything) plus a quick sanity: python -c import of freetoken.engine; targeted tests unchanged from round 2 (tests/models/test_gguf_expert_banks.py, tests/models/test_glm5_next_gguf.py, engine tests).
5. After merge, verify `_weight_load_context` wrap landed around `load_expert_banks` and `ftw_quant_format` while our `_gguf_auto_floor_gate` call and comment block survived.
6. Before merging: re-check `git stash list` empty + `git status --porcelain` only-.veai (stable this round; stash-anomaly from round 2 did not recur).
