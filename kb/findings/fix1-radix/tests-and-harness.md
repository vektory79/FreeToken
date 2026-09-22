# Test patterns and hardware harness for fix1-radix-track-seqlen

Read-only discovery on vektory79 @ main checkout /media/ai/src/FreeToken (2026-09-17).
Verified live: `.venv/bin/python -m pytest tests/scheduler/test_hybrid_cache_manager.py
tests/kvcache/radix/test_hybrid_radix.py -q` -> 29 passed in 2.13s (CPU, editable install).

## A. Existing tests to extend

### A1. Scheduler chunked-prefill admission (check 1)

Primary file: `tests/scheduler/test_hybrid_cache_manager.py` (170 lines, CPU-only).
It mirrors `python/freetoken/scheduler/cache.py` + exercises `PrefillAdder` from
`scheduler/prefill.py` - the module where the bug lives. Two relevant existing tests:

- `test_prefill_chunk_ends_on_a_page_boundary` - the admission stub pattern to copy:
  real `CacheManager(64, 1, pt, "hybrid_radix", linear_state_pool=pool)` (CPU
  `LinearStatePool` from `LinearGatedDeltaGroupConfig`), real `TableManager`, real
  `PrefillAdder(token_budget=..., reserved_size=0, cache_manager=cm, table_manager=tm)`,
  `PendingReq(0, torch.arange(300, dtype=torch.int32), SamplingParams(max_tokens=1))`,
  then `adder.try_add_one(pending)` and asserts on the returned `ChunkedReq.extend_len`.
- `test_hybrid_cache_manager_donate_then_hit` - the donate/match pattern: hand-built `Req`
  with `linear_slot_idx`/`mamba_ping_pong`/`mamba_next_track_idx`/
  `mamba_last_track_seqlen = 4` set by hand, `cm.lock(mr.cuda_handle)`,
  `cm.cache_req(reqA, finished=False)`, then `cm.match_req(_pend([1,2,3,4,9]))` asserting
  `cached_len == 4` and `mamba_value == pp[0]`.

The stub/fixture pattern (quoted from the file):

```python
def _pool(num_slots=16):
    g = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate="silu",
    )
    return LinearStatePool(group=g, num_slots=num_slots, dtype=torch.bfloat16,
                           device=torch.device("cpu"), tp_size=1)

def _pend(ids):
    # int32 to match production Req.input_ids dtype
    t = torch.tensor(ids, dtype=torch.int32)
    return SimpleNamespace(input_ids=t, input_len=len(ids))
```

Supporting pattern for "L is set by the x64 track, not by the test framework":
`tests/scheduler/test_abort_inflight_prefill.py::_launch_req(pool, cm, tm, prompt, *,
cls=Req, track_seqlen=None)` manually sets `req.mamba_last_track_seqlen = track_seqlen`
after a fake forward (`cm.allocate_paged([req]); req.complete_one()`). On CPU nobody runs
`attention/linear.py` tracking, so unit tests must inject L by hand - this is the accepted
in-repo convention.

Bug site for orientation: `python/freetoken/scheduler/prefill.py` `try_add_one`
(continuation branch, ~236-250) forwards `ping_pong`, `next_track_idx`, `restore_src`,
`swa_evicted_seqlen` from `pending_req.chunked_req` but NOT `mamba_last_track_seqlen`;
`_add_one_req` (~200-232) is where the continuation Req fields are assigned.

Secondary host (if a PrefillManager-loop shape is preferred):
`tests/scheduler/test_scheduler_chunked_prefill.py::_drive_chunked_prefill(cm, tm, pm,
n_chunks)` drives `pm.schedule_next_batch(CHUNK)` + `cm.allocate_paged` +
`r.complete_one()` + final `cm.cache_req(r, finished=False)` through N chunks - but it
builds a plain `"radix"` CacheManager (no GDN fields); a hybrid variant would need the
`linear_state_pool=` kwarg and manual L injection, so A1's primary file is the better fit.

### A2. kvcache-level hybrid radix suite (check 2)

The finish-frozen donate under fix lives in `python/freetoken/scheduler/cache.py`
`cache_req(finished=True)` (~343-362: `L = req.mamba_last_track_seqlen; if L is not None
and 0 < L <= req.cached_len and align_down(L, self.page_size) == L and
req.mamba_ping_pong is not None:` -> `prefix_cache.insert(req.input_ids[:L],
page_indices[:L], frozen)`). By the "tests mirror the module they protect" rule, its
regression test belongs in `tests/scheduler/test_hybrid_cache_manager.py`, NOT in
tests/kvcache. The existing `test_hybrid_cache_manager_donate_then_hit` IS the
"L page-aligned + ping_pong -> donate at L -> second identical request matches
cached_len == L" test (at page_size=1, L=4); extend it (e.g. a page_size>1 aligned-L
sibling) or add a test next to it.

Raw-cache suite (GPU-free, property harness):
`tests/kvcache/radix/test_hybrid_radix.py` (+ `adapters.py`/`driver.py`/`model.py`/
`conftest.py`, README in same dir). Existing donate/match scenarios:
- `test_insert_attaches_snapshot_and_match_restores_it` (hyb fixture: `Session(CacheSpec("hybrid", PAGE))`,
  `hyb.do_insert(ids(1,2), slots=...)` / `hyb.do_match(ids(1,2))`, `hyb.check()` invariant battery)
- `test_deepest_snapshot_wins_and_unmatched_suffix_is_dropped`
- `test_insert_stores_whole_pages_only(P)` (page-size matrix)

Note the raw `HybridRadixCache` has no L concept - L is scheduler state; a raw-cache-level
test can only re-cover insert/match, which the harness already pins. Recommended primary
insertion point stays A2-first-paragraph (CacheManager level). The nearest raw-cache
inverse case is `tests/scheduler/test_swa_pagesize.py::test_hybrid_chunk_donate_skips_unaligned_boundary`
(ps=8; unaligned L -> donate skipped, aligned L -> donate lands) - also a natural neighbor
for an aligned-L page_size>1 case.

### A3. Files touching mamba_last_track_seqlen / ping_pong / tool-call anchor

- `tests/models/qwen4_exp/test_ple.py` - `_tracked_req(table_idx, cached_len, tokens, *,
live, ping_pong)` sets `req.mamba_ping_pong`; asserts `req.mamba_last_track_seqlen ==
CHUNK_SIZE` (:411) from a REAL forward track. GPU, known-flaky baseline (:421) - do not
chase failures there.
- `tests/scheduler/test_swa_pagesize.py` - frozen-donate skip/passthrough (:145-157).
- `tests/scheduler/test_hybrid_cache_manager.py` - donate-then-hit (:47-56), etc.
- `tests/scheduler/test_abort_inflight_prefill.py` - `_launch_req(..., track_seqlen=)`
(:87-92), drain asserts (:116, :185).
- `tests/scheduler/test_commit_repoints_page_table.py` -
`test_hybrid_unfinished_commit_repoints_the_row_off_the_freed_pages` sets
`r.mamba_last_track_seqlen = len(PROMPT)` (:73-75).
- `tests/models/test_glm5_next_model.py:127` - `mamba_ping_pong=None` in a Req ctor.

Tool-call anchor: the anchor logic is `CacheManager.snapshot_toolcall_anchor`
(python/freetoken/scheduler/cache.py:148-173; consumed by the finish-frozen donate and
scheduler.py:917). There is NO dedicated unit test for it in tests/ - the only reference
is the `toolcall_anchor_id=None` stub field in `test_abort_inflight_prefill.py:62`
(Scheduler stub). So check (4) "anchor tests keep passing" degrades to: the five L-touching
files above are the anchor's regression surface (the anchor reuses
`mamba_last_track_seqlen` as its pending-donate mark).

### A4. e2e chunked-reuse host (check 3)

No existing CPU e2e chunked-reuse test. `tests/e2e/test_cache_rebuild.py` boots a real
server but needs `FREETOKEN_REBUILD_TEST_MODEL` (a small local model dir) and a GPU -
that is the only e2e file that could host a real chunked 65585-token repeat, and it is
env-gated. Recommended: keep the true 65585-token check hardware-only (measure.py, section
C); a small CPU proxy (admission chain -> hand-set L per chunk -> finish donate ->
second match cached_len > 0) fits as a new test in `tests/scheduler/test_hybrid_cache_manager.py`.

## B. Running these tests on this box

- `/media/ai/src/FreeToken` IS the main checkout: `.git` is a real directory (not a
  worktree file), branch `vektory79`; `.venv/` at repo root. Verified:
  `.venv/bin/python` = 3.14.7, pytest 9.1.1, freetoken editable at
  `python/freetoken/__init__.py`.
- Correct invocation: `.venv/bin/python -m pytest ...` FROM the repo root. Console
  `pytest`/`uv run pytest` puts no repo root on sys.path and `from tests.` imports
  fail (no `tests/__init__.py`) - import-mode false failures; always reproduce with
  `python -m pytest` before counting a failure.
- Markers (pyproject.toml `[tool.pytest.ini_options]`, testpaths=["tests"]):
  `slow` (tens-of-seconds tests) and `needs_weights` (env-gated real checkpoints).
  Deselect with `-m "not slow"`. GPU-dependent tests self-skip without CUDA.
- Suggested commands:

```bash
cd /media/ai/src/FreeToken
.venv/bin/python -m pytest tests/scheduler/test_hybrid_cache_manager.py -q
.venv/bin/python -m pytest tests/scheduler/ tests/kvcache/ -q -m "not slow"
.venv/bin/python -m pytest tests/ -m "not slow" --ignore=tests/models/test_glm5_next_kda_snapshot.py
```

(the `--ignore` is the known no-`tests/__init__.py` collection abortor.)

- Pre-existing baseline failures to NOT chase (verified 2026-09-15, non-import-mode):
  pinned_tensor UVA driver error; qsa_fp8 CUDA invalid argument :48;
  qwen4_exp/test_ple flaky bitwise :421; 2x glm_dsa OutOfResources shared-mem
  102400 > 101376. Also 10x `ModuleNotFoundError: No module named 'tests'` artifacts
  under console pytest are import-mode noise, not regressions.

## C. Hardware harness: [measure.py](../../harness/serve-measure/measure.py)

EXISTS (437 lines, mirrored into kb). [PHASE2.md](../../cases/ft-gguf-serve-tuning/PHASE2.md) EXISTS (chunk-size
disambiguation matrix + back-to-back decode A/B; notes the 0.89 fail-fast deviation
when the desktop ate VRAM -> fall back to 0.90).

CLI: `python measure.py <mode> --name NAME --log LOG [--extra "FLAGS"] [--env KEY=VAL]...`
- modes: `boot` (boot + log tail only), `prefill` (boot + one 65k prefill), `full`
  (boot + prefill + decode repeat), `ladder` (boot + incremental-context ladder;
  adds `--max-steps` (7), `--target-tokens` (393216), `--step-timeout` (900)).
- `--extra` is appended to `shlex.split` and passed to `ft serve` after BASE_ARGS:
  `--model-path .../GLM-5.3-Flash-UD-Q3_K_XL.gguf --moe-cache-auto --kv-reserve-tokens
  500000 --kv-cache-dtype fp8 --memory-ratio 0.89 --max-prefill-length 4096
  --moe-strategy hybrid --moe-cpu-threads 16` (PORT 18801; FT = `.venv/bin/ft`).
  Repeating a flag in --extra depends on ft arg parsing (phase2 saw duplicated
  --memory-ratio flags take the later value) - prefer overriding via --extra only.
- 65k prompt: `DIR/filler.txt` is posted to `/v1/chat/completions`
  (`do_prefill`, max_tokens=4, non-stream; request template `req_prefill.json`),
  then the SAME content is re-sent streaming (`do_decode`, req_decode.json,
  max_tokens=64) to test radix reuse.
- Verdict parsing: `parse_tok(lines, "cached-token")` greps the server log lines
  emitted AFTER the request start offset (`new_log_lines(logpath, off)`) for
  `#cached-token` / `#new-token`; usage.prompt_tokens is the fallback. Gauges
  (#mamba-slot, token usage) EXCLUDE evictable tree snapshots - always use the log
  `#cached-token` as the reuse verdict (expect ~65472-65536 on HIT where today 0).
- Known patches CONFIRMED PRESENT:
  - decode-steady skip of the pre-prefill SSE frame: `do_decode` comment "the first
    SSE frame can arrive before prefill (role/accept frame)" + `tail = times[1:]`
    (steady measured over the tail without the first frame).
  - anchored throughput regex: `parse_thrs` =
    `re.search(r"input throughput \(token/s\):\s*([0-9]+\.?[0-9]*)", l)`;
    `thr_summary` drops chunk 1 (autotune warmup) via `vals[1:]`.
- Safety rails built in: WATCH regex
  `AssertionError|OutOfMemoryError|Backend worker is gone|CUDA error|Traceback`,
  VramPoller thread, census_serve() (semaphore census), teardown via SIGTERM, JSON
  result printed at the end.
- Gotchas (TASK.md + memory): pkill needs the `'[f]t serve'` bracket pattern; serial
  runs only, census between boots; the FINAL full chunk's "input throughput" line is
  a bogus drain artifact (~5x fast) - take the median of full chunks EXCLUDING the
  last one, not thr_summary's mean as-is.
- Fix-validation recipe: `python measure.py full --name fix4096 --log fix4096.log
  --extra "--max-prefill-length 4096 --memory-ratio 0.90"` -> check
  out["decode"]["cached_token"] ~ 65536 (HIT) and out["prefill"] steady class ~262
  tok/s; repeat with 8191 chunks (must stay HIT); decode steady 13.5-14.7 tok/s class.

## D. Concrete insertion points

1. Check (1) - scheduler-level regression (the field drop):
   - File: `tests/scheduler/test_hybrid_cache_manager.py`.
   - Insert after `test_prefill_chunk_ends_on_a_page_boundary` (last prefill-side test,
     :114-138).
   - Reuse: module `_pool()`, `_pend()`; imports
     `freetoken.scheduler.prefill.ChunkedReq/PrefillAdder`,
     `freetoken.scheduler.table.TableManager`, `freetoken.scheduler.utils.PendingReq`
     (all already imported locally inside that test).
   - Shape: hybrid cm; `adder = PrefillAdder(token_budget=64, ...)`;
     `chunk1 = adder.try_add_one(pending)` (ChunkedReq, extend_len=64);
     `cm.allocate_paged([chunk1]); chunk1.complete_one()`;
     `chunk1.mamba_last_track_seqlen = 64` (hand-injected x64 track, per
     `_launch_req` convention); `pending.chunked_req = chunk1` (PendingReq field used
     by the continuation branch);
     `cont = PrefillAdder(token_budget=64, ...).try_add_one(pending)`;
     assert `cont.mamba_last_track_seqlen == 64` (fails before the fix: None) and
     `cont.mamba_ping_pong is chunk1.mamba_ping_pong` (continuity unchanged); then
     finish the chain: `cm.lock(cont.cache_handle)`,
     `cm.cache_req(cont, finished=True)`, second `cm.match_req` on the same prompt ->
     `cached_len > 0` (merges checks 1+2 into one CPU story; keep a separate pure (1)
     assert so a donate regression cannot mask the field-drop regression).
2. Check (2) - donate-then-hit at L:
   - Primary: extend `tests/scheduler/test_hybrid_cache_manager.py::
     test_hybrid_cache_manager_donate_then_hit` with a page_size>1 aligned-L sibling
     (cm `CacheManager(64, 4, pt, "hybrid_radix", linear_state_pool=pool)`, L=8
     aligned -> donate at 8 -> match cached_len == 8; unaligned L=7 -> no donate is
     already covered by `tests/scheduler/test_swa_pagesize.py::
     test_hybrid_chunk_donate_skips_unaligned_boundary` at ps=8).
   - Raw-cache level (optional): add a named scenario next to
     `tests/kvcache/radix/test_hybrid_radix.py::
     test_insert_attaches_snapshot_and_match_restores_it` using the `hyb()` Session
     fixture; note the raw cache has no L - L is scheduler state, so this only
     re-pins insert/match the harness already covers (low marginal value).
   - Also relevant neighbors: `tests/scheduler/test_commit_repoints_page_table.py::
     test_hybrid_unfinished_commit_repoints_the_row_off_the_freed_pages`.
3. Check (3) - 65585-token chunked e2e: keep HARDWARE-ONLY via measure.py (needs the
   real `attention/linear.py` x64 tracking inside model forwards, i.e. GPU + model).
   A small CPU proxy (chain in point 1) can assert donate->match>0 with hand-set L.
4. Check (4) - anchor: no dedicated anchor test exists; run the L-touching files as
   the keep-passing gate:

```bash
.venv/bin/python -m pytest tests/scheduler/test_swa_pagesize.py \
  tests/scheduler/test_hybrid_cache_manager.py \
  tests/scheduler/test_commit_repoints_page_table.py \
  tests/scheduler/test_abort_inflight_prefill.py -q
```

(`tests/models/qwen4_exp/test_ple.py` also consumes L from a real forward but is
GPU + known-flaky baseline - do not gate on it.)

## Risks

- Unit tests must hand-inject `mamba_last_track_seqlen` (no CPU tracking); the (1)
  regression therefore tests field forwarding only - matching the in-repo convention
  (`_launch_req`), acceptable, but it cannot catch a tracking-side regression.
- `tests/models/qwen4_exp/test_ple.py` looks CPU-ish but needs GPU and is a known
  flaky baseline (:421).
- thr_summary's `vals[1:]` drops the FIRST chunk only; the LAST full chunk's
  throughput line is a drain artifact - compute medians excluding it when reporting
  perf numbers.
- Boot with `--memory-ratio 0.89` can fail-fast on the slot-floor gate if the desktop
  ate VRAM (PHASE2 deviation note); fall back to 0.90 and record the deviation.
- A fused (1+2) test risks a donate-side regression masking the field-drop assert;
  keep the `cont.mamba_last_track_seqlen == 64` assert standalone first.
