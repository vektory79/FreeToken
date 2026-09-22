# TASK: fix-1 - radix/prefix reuse for long chunked prompts (persist mamba_last_track_seqlen)

Status: COMMITTED as 5b72aba on vektory79 2026-09-18 (no push; 3 files, 214+/5-). Implemented + hardware-verified 2026-09-18. Wave artifacts: research/code-anchors.md, research/tests-and-harness.md, verification/hardware-report.md. Verdicts: repeat 65k @4096 -> HIT #cached-token 65472 (pre-fix 0); 8191@0.85+mr1 -> HIT 65536; prefill medians 300.6/336.4 tok/s, decode 13.9-14.7 tok/s (no regression); boot deviations: 0.89 fail-fast -> 0.90 (Run A) / winner 0.85+mr1 (Run B). Follow-up wave 2026-09-18: A3 test test_final_prefill_commit_clears_track_seqlen_and_donates (tests/scheduler/test_abort_inflight_prefill.py:199-238, pins the finished=False L-clear invariant, revert-check proven, tests/scheduler 104 passed) + orphan-process sweep done (semaphores 11 -> 5, 2 stale PIDs reaped). Composed 2026-09-17 from a hardware-measured A/B campaign
(artifacts in the git-ignored ft-gguf-serve-tuning worktree, results mirrored under
[baselines/ft-gguf-serve-tuning/](../../baselines/ft-gguf-serve-tuning/), summary memory:
`ft-serve-gguf-tuning-campaign-2026-09`). Read CONTRIBUTING.md first - it is binding.
Line anchors below were verified on vektory79 @ 1a444e4 on 2026-09-17 and re-verified on HEAD ebf071b 2026-09-18 (diff vs 1a444e4 over target files: empty).

## Goal

Restore radix/prefix reuse for long prompts at ANY prefill chunk size. Today a 65k prompt
prefilled with `--max-prefill-length 4096` is NEVER reused: a byte-identical repeat request
re-prefills all ~65k tokens (#cached-token: 0), costing ~251 s TTFT per repeat on the
RTX 5090 rig. With 8128-token chunks the same repeat HITs (cached=65536) and costs
7.6-8.6 s. For the agentic-coding workload (context grows incrementally, cache-loss
re-prefills are the dominant latency spike) this is the highest-value single bug.

## Measured evidence (hardware, 2026-09-16/17)

- 65585-token prompt, `--kv-cache-dtype fp8`, `--moe-strategy hybrid`, `--moe-cache-auto`,
  `--kv-reserve-tokens 500000`:
  - chunk 4096 (16 chunks): repeat = FULL MISS (0/49 cached) at ratio 0.85/0.89/0.90,
    mr4 and mr1. Config-independent.
  - chunk 8128 (8 chunks): repeat = HIT 65536 (ratio 0.85, mr1).
  - prompts <= 9.7k tokens: HIT in every config (single/few chunks; final chunk tracks).
  - fp8 KV dtype is NOT the cause; nor mr, nor ratio, nor request timing (60 s gaps).
- Decode speed itself is unaffected (13.5-14.7 tok/s across configs) - the cost is purely
  the re-prefill TTFT.
- Full matrix: [PHASE2.md](../ft-gguf-serve-tuning/PHASE2.md).

## Root cause (verified in code, consistent with every observation)

Snapshot donation into the hybrid radix tree happens at finish via two paths, both of which
read `req.mamba_last_track_seqlen` (L):

1. `scheduler/cache.py` ~346-362 (finish-frozen donate):
   `L = req.mamba_last_track_seqlen; if L is not None and 0 < L <= cached_len and
   align_down(L, self.page_size) == L and req.mamba_ping_pong is not None:` -> insert
   the frozen ping-pong slot at prefix L.
2. final-batch commit `scheduler.py:398` (`cache_req(finished=False)`) - only the batch
   that completes prefill produces a commit.

L is computed per tracking batch in `attention/linear.py` ~116-131:
`boundary = cached_len + (extend_len-1)//64*64`; a chunk with c<1 `continue`s WITHOUT
setting L. The bug: `scheduler/prefill.py` `try_add_one` (~236-250) forwards to the
continuation Req `ping_pong`/`next_track_idx`/`restore_src`/`swa_evicted_seqlen` but NOT
`mamba_last_track_seqlen` -> every continuation Req starts with L=None (default in
`core.py:52`). For 65585 tokens at chunk 4096 the LAST chunk is 49 tokens -> c=0 ->
L is never set on the Req that actually finishes -> both donation paths see L=None ->
tree empty -> `HybridRadixCache.match_prefix` (kvcache/hybrid_radix_cache.py:76-88) climbs
to no live GDN snapshot -> cached_len=0. At chunk 8128 the final chunk (561 tokens) has
c=8 -> L survives on the finishing Req -> donation lands -> HIT.
Side effect of the same bug: finish-frozen donate is lost for ANY multi-chunk prompt whose
last chunk is < 64 tokens; the tool-call anchor (`cache.py` ~148-173) also consumes L -
retest anchors after the fix.

## The change

Forward `mamba_last_track_seqlen` from `pending_req.chunked_req` to the continuation Req in
`try_add_one` (and make sure `_add_one_req` / the continuation Req construction carries the
field through; `ChunkedReq` already stores it). Expected diff ~5-10 lines plus tests.
Do NOT change donation semantics, ping-pong ownership, page-size handling, or the alignment
conditions - they are correct by design (see the comment block at cache.py ~343-350).
Do NOT touch hf_config mutation patterns (frozen dataclass shim crashes on unguarded
setattr - lesson from 8ce2657).

## Tests (bug fixes come with a test that fails before and passes after)

Put them where the modules live (tests/ mirrors python/freetoken/; extend an existing file
before creating a new one):
1. Scheduler-level (CPU): a multi-chunk admission where the final chunk has c=0 must
   finish with `req.mamba_last_track_seqlen` set (currently None) - the direct regression
   test for the field drop.
2. kvcache-level (CPU): with L set to a page-aligned value and ping_pong present, the
   frozen donate inserts a node at L and a second identical request matches cached_len
   reaching L. If a similar test exists in tests/kvcache (hybrid radix suite runs without
   GPU), extend it.
3. Optional CPU end-to-end: chunked 65585-token prompt (chunk 4096) -> second identical
   request -> cached > 0. If too heavy for a unit test, mark it as the hardware check below.
4. Tool-call anchor tests still pass (L semantics unchanged when L is None-consumed).

## Hardware validation (GPU, after unit tests pass)

Harness: reuse [measure.py](../../harness/serve-measure/measure.py) (already patched: decode steady
skips the pre-prefill SSE frame; throughput regex is anchored). Gotchas: pkill needs the
'[f]t serve' bracket pattern; gauges (#mamba-slot, token usage) EXCLUDE evictable tree
snapshots - use the server log "#cached-token" lines as the verdict; serial runs only,
watchdog patterns AssertionError|OutOfMemoryError|Backend worker is gone, SIGTERM->SIGKILL,
census between boots.
- Boot: user's BASE config (`--memory-ratio 0.89` may fail-fast on the early floor gate if
  the desktop ate VRAM - hoptodesk cost ~492 MiB on 2026-09-17; fall back to 0.90 and note
  the deviation).
- Prefill 65k filler -> repeat identical request -> expect #cached-token ~65472-65536 (HIT)
  where today it is 0. Also repeat at 8128 chunks (must stay HIT).
- No perf regression: prefill median (server "input throughput", 4096-token full chunks,
  skip chunk 1 autotune warmup) stays ~262 tok/s class; decode ~13.5-14.7 tok/s class.

## Commits

Conventional Commits, lowercase, imperative: e.g.
`fix(scheduler): carry mamba_last_track_seqlen across prefill chunk transitions`.
Commit only when the user asks. Never push. One change per PR, link the measured evidence.
Committed 2026-09-18 as 5b72aba (single commit, 3 files: python/freetoken/scheduler/prefill.py,
tests/scheduler/test_hybrid_cache_manager.py, tests/scheduler/test_abort_inflight_prefill.py -
the last file carries both the B5 fix-1 regression assert and the A3 follow-up test at 199-238).
Never push (user rule).

## Out of scope (separate follow-ups, do not mix into this PR)

- DONE 2026-09-18 (emerged from fix-1 review triage A3, not one of the original items):
  scheduler-level pin of the finished=False final-commit L-clear invariant ->
  test_final_prefill_commit_clears_track_seqlen_and_donates. Review-hardened same day
  (protection asserts full_protected/mamba_ref_count with revert-check, structural
  prefix_cache.check_integrity, ps=8 finished=False unaligned-L skip test, __main__ moved
  to EOF; tests/scheduler 105 passed). Optional follow-up left open:
  L=None finished=False commit pin (2-line no-op, loud failure mode - not wave-worthy).
- Per-chunk donation of reuse points (partial-prefix resume at chunk granularity) - bigger
  change, interacts with ping-pong alloc + locked-handle eviction race (cache.py:398-401).
- LRU timestamp refresh on insert so deep snapshot nodes are not preferentially evicted
  (hybrid_radix_cache.py:168-237; this is why the ladder only ever reused exactly 65536).
- Tiled MMQ prefill kernel (the larger prefill-throughput lever).
