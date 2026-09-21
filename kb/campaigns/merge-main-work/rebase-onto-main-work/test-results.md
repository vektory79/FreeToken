# Test results: merge main -> vektory79 (merge commit c8d5140)

All runs serialized (one at a time, each under `timeout`, postflight census after every
run). Census pattern: `pgrep -af 'pytest|ft serve|benchbw'`; semaphore census
`ls /dev/shm | grep -c "^sem\.mp-"`.

Preflight census (before any test): no pytest / ft serve / benchbw processes; semaphores 0.
Postflight census after every run below: no processes; semaphores 0 (twice a transient 5
during the first extend-kernel failures; back to 0 immediately after, never accumulated).

Invocation form: `uv run --extra dev pytest ...` (venv split gotcha: plain
`uv run pytest` collects with a python that lacks fastapi; `--extra dev` is required).

## Targeted runs (pre-finalization, on the conflicted working tree)

1. `timeout 1200 uv run --extra dev pytest tests/kernels/test_triton_attention.py -q`
   - run 1: 15 failed / 31 passed  -> root cause: `extend_paged_attention` signature
     missing k_scale/v_scale/kv_quant after the first edit pass (NameError in body).
   - fixed signature; run 2: 1 failed / 45 passed -> FakeKVCache stub of main's
     block_ends test lacked k_scale/v_scale (adapted, see adaptation-log).
   - run 3: 46 passed / 0 failed.
2. `timeout 600 uv run --extra dev pytest tests/models/test_glm5_next_config.py tests/models/test_quant_config.py -q`
   - 30 passed / 41 skipped (skips are conditional GPU/arch cases, pre-existing pattern).
3. `timeout 1200 uv run --extra dev pytest tests/models/qwen4_exp/test_qsa_backend.py tests/kvcache/test_qsa_pool.py tests/kvcache/test_qsa_pool_fp8.py -q`
   - run 1: 2 failed / 29 passed (model_is_mrope missing from branch test stubs).
   - fixed stubs; rerun of the same set + extras:
     `timeout 1200 ... tests/kvcache/test_qsa_pool_fp8.py tests/kvcache/test_qsa_pool.py tests/kvcache/test_cache_unit_bytes.py tests/engine/test_kv_quant_config.py tests/kvcache/test_mha_pool_fp8.py -q`
     -> 86 passed / 0 failed.
4. `timeout 300 uv run python -c 'import freetoken.engine, freetoken.server.args'`
   - IMPORT-SMOKE OK.

## Merge finalization

After 1-4 were green: `GIT_EDITOR=true git commit -m "Merge branch 'main' into vektory79"`
-> c8d514054eac6f397182c521dc790e2aae7219c1, parents 6e673ab + afd99cb, tree clean.

After the extended run triaged the test_offload failure and it was fixed, the fix was
amended into the merge commit:
`git add python/freetoken/engine/graph.py && git commit --amend -m "Merge branch 'main' into vektory79"`
-> FINAL merge commit 380e9eb9ba1c58697471980f94a7874c770a9893,
parents 6e673abfe5a92eef11668af7d43d77092042df21 + afd99cb3b7c118dce4a9835359bcbc33b70c3cb8.

Post-finalization working-tree entries (NOT part of the merge, left untouched):
`.veai/memory/MEMORY.md`, `.veai/memory/pr408-kv-nvfp4-1m-port-029031181f9f.md`
(modified) and two new untracked `.veai/memory/*.md` files - these are live
memory-bank writes from the agent framework that appeared after the preflight;
nothing under python/ or tests/ is dirty.

## Extended run (post-finalization)

`timeout 1800 uv run --extra dev pytest tests/ -m "not slow" -q --ignore=tests/models/test_glm5_next_kda_snapshot.py`
(log: extended-pytest.log; the ignored file aborts collection with the known
no-tests/__init__.py import problem, see below).

Result summary: see the tail of extended-pytest.log quoted below.

<EXTENDED-RESULT>
`17 failed, 2081 passed, 200 skipped, 15 deselected, 4 warnings in 187.64s (0:03:07)`, RC=1.

Failure classification (17):
- 16 match the documented baseline classes on this box (see list below):
  1x test_pinned_tensor (UVA), 1x qwen4_exp/test_ple (flaky bitwise),
  1x test_muse_glimmer_parsers (no tests/__init__.py import),
  7x tests/attention/test_dsa_kpool.py[nvfp4] (same import root cause),
  2x tests/kvcache/test_dsa_pool.py[nvfp4] (same),
  3x tests/models/test_glm_dsa.py (baseline had 2x triton OOR + 1x import),
  1x tests/kernels/test_qsa_fp8.py (CUDA invalid argument).
- 1x NEW: tests/moe/test_offload.py::test_graph_capture_reuses_warm_offload_cache_before_capture
  -> triaged: reproduced identically on pre-merge HEAD 6e673ab via a scratch worktree
  (same TypeError at branch graph.py) - a pre-existing branch-side defect (branch's
  gguf wave landed after the 09-12 baseline), NOT a merge regression.
  Fixed in the merge commit: graph.py normalization now duck-types the single-cache
  case (see adaptation-log "Post-merge integration fixes").
  Post-fix: tests/moe/test_offload.py 24 passed; tests/engine/test_cache_budget.py +
  tests/moe/test_hybrid_fetch.py 53 passed; amended into the merge commit.
- vs 09-12 baseline composition: 3x tests/moe/test_prefill_hit_d2d that were failing
  at baseline PASSED here (transient kernel-probe class, no action).
</EXTENDED-RESULT>

Known baseline failing classes on this box (documented 2026-09-04/09-12, NOT merge
regressions if they match): pinned_tensor UVA driver error; swiglu_clamp CUDA fallout;
qwen4_exp/test_ple flaky bitwise GPU assert; muse_glimmer tests-import
(ModuleNotFoundError: No module named 'tests'); 10x dsa/kpool tests-import from the
nvfp4-kv wave (same no-tests/__init__.py root cause); 2x glm_dsa triton
OutOfResources; 3x test_prefill_hit_d2d cudaMemcpyBatchAsync probe; 1x
test_qsa_fp8 CUDA invalid argument.

Any failure NOT in that list will be triaged as a merge regression and fixed in the
working tree with `git commit --amend` into the merge commit.

## Post-review re-runs (second amend)

Scope of the second amend: docs (models.md, cli.md) + args.py help string + a
one-line tuple branch in graph.py. No full-suite rerun per plan (docs + an
isinstance widening of the same defensive fix the extended suite already covered).

Preflight census: no pytest / ft serve / benchbw processes (pgrep matched only the
wrapper shell itself); `ls /dev/shm | grep -c "^sem\.mp-"` -> 0.

1. `timeout 900 uv run --extra dev pytest tests/moe/test_offload.py -q`
   -> 24 passed / 0 failed in 2.29s. Covers the GraphRunner duck-typed
   single-cache path the A3/B2 fix touches.
2. `timeout 300 uv run python -c "import freetoken.engine, freetoken.server.args"`
   -> IMPORT-SMOKE OK (args.py help-string rewrite verified at import time).

Postflight census: no processes; semaphores 0.

Amend: `git add docs/models.md docs/cli.md python/freetoken/server/args.py
python/freetoken/engine/graph.py` (only these four; the dirty .veai/memory/*
entries stay out) -> `GIT_EDITOR=true git commit --amend` -> parents re-verified
6e673ab + afd99cb; `git rev-list --count HEAD --not 6e673ab afd99cb` == 1; no
conflict markers in the tracked tree; no .veai paths in the commit.
