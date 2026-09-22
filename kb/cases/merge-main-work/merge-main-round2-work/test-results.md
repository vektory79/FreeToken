# Test results: merge round 2 (main cac247a -> vektory79 @ 8c0f1e7)

Merged tree tested in `git merge --no-commit` state (HEAD 8c0f1e7 + staged merge tree,
MERGE_HEAD cac247a). Interpreter: `uv run --extra dev python -m pytest` (import-mode-correct
form; repo-root on sys.path). Preflight census before the test wave: no pytest / ft serve /
benchbw processes, `sem.mp-` = 0. All runs serialized, each under `timeout 900`.

## Targeted batches (step 5b)

| # | Files | Result |
|---|---|---|
| 1 | tests/models/test_gguf_expert_banks.py + test_glm5_next_gguf.py + test_glm5_next_config.py + test_quant_config.py | **129 passed, 42 skipped** in 39.7s (slot-gate/shim pins from 8bcca7b/8ce2657 included, green) |
| 2 | tests/checkpoint/test_ftw_weights.py + test_ftw_hotfix_vision.py (NEW, cade1a9) + tests/models/test_qwen3_vl.py | **28 passed, 6 skipped** in 1.7s (new vision-encoder tests pass) |
| 3 | tests/models/qwen4_exp/test_qsa_backend.py + tests/kvcache/test_qsa_pool.py + test_qsa_pool_fp8.py | **31 passed** in 6.9s |
| 4 | tests/engine/ (incl. test_kv_quant_config.py, test_cache_budget.py) | **158 passed, 2 skipped** in 2.0s |

TOTAL: **346 passed, 50 skipped, 0 failed**.

## Import smoke (step 5b)

```
.venv/bin/python -c "import freetoken.engine, freetoken.models.weight"
-> "smoke ok: engine + models.weight"
```

## Conflict-marker grep (step 5a)

`grep -rnE '^(<<<<<<<|=======|>>>>>>>)' python/ tests/ docs/` -> no matches (exit 1).

## GGUF boot smoke (step 5c)

Command = the 786k reference config from the task (port 18081, GGUF UD-Q3_K_XL,
moe-cache-auto, kv-reserve 786432, nvfp4, memory-ratio 0.89, hybrid, 16 cpu threads).
Runner: /tmp/ft_boot_smoke_merge2.sh (trap-on-EXIT cleanup, SIGTERM->30s->SIGKILL,
FAILPAT='AssertionError|OutOfMemoryError|Backend worker is gone|backend worker .* exited|
Traceback|CUDA error|ValueError', hard timeout 900s). Server log: /tmp/ft_boot_smoke_merge2/server.log.

**VERDICT: BOOT-SMOKE PASS.**

- Preflight census: no foreign processes, sem.mp- = 0, VRAM 1148 MiB used (baseline band).
- /ready == 200 after **91s** (reference boot 132s; page-cache warm).
- Boot numbers vs эталон (task reference) - ALL deviations in the SAFE direction, driven by
  more free VRAM at start (29.71 GiB vs ~29.46 GiB in the 786k reference run):

| Metric | эталон | observed | delta |
|---|---|---|---|
| moe_cache_size | 1071 | **1091** | +20 slots (more free VRAM -> larger envelope) |
| slots [g0,g1,g2] | [298,288,288] | **[320,288,288]** | +22 on group 0, groups 1/2 at floor 288 |
| KV alloc | 788160 tok / 2.87 GiB | **789120 tok / 2.88 GiB** | +960 tok (12330 vs 12315 pages) |
| free after init | ~2.86 GiB | **2.90 GiB** | +0.04 GiB (MORE margin, safer for 4096-chunks) |
| free before model | ~29.46 GiB | 29.71 GiB | +0.25 GiB explains the envelope drift |

- **Slot-floor gate SILENT** (no fail-fast line from 8bcca7b) = expected, all groups >= 288.
- Signature-group lines [320,288,288] all < 2E=576 -> synchronous materialized prefill,
  known Phase 5 semantics (same as reference).
- CPU MoE executor pools present (same pool structure as reference).
- Smoke POST /v1/chat/completions (model field set): **HTTP 200**, body
  `"content":"", "reasoning_content":"The user just said \\"Hi\\"", finish_reason=length,
  usage {prompt 13, completion 7, total 20}` - empty content at max_tokens=8 is the known
  glm reasoning-parser behavior, NOT an error.
- Shutdown: SIGTERM -> exit **143** (normal uvicorn reap), no SIGKILL needed, no hang.
- Postflight census: processes none, sem.mp- = 0, VRAM back to **1148 MiB** (= pre-smoke).

## Invariants after finalization (step 6)

- Merge commit: **5d93a30** "Merge branch 'main' into vektory79", parents **(8c0f1e7, cac247a)**.
- `git rev-list --count HEAD --not 8c0f1e7 cac247a` = **1** (exactly one new commit).
- `.veai/` files in the merge commit: **0**.
- python/freetoken/version.py in the tree: **0.1.3** (from cac247a).
- Working tree: only pre-existing `.veai/memory/*.md` dirty entries (stashed before the
  operation with pathspec-stash "merge2: veai-memory-wip", popped back after finalization).
- No push, no gh, no amend needed (nothing failed after finalization).

## Post-merge test wave (post-commit sanity re-check)

None needed: the tested tree content is byte-identical to the committed tree (same index),
no files were modified between the test wave and the commit.
