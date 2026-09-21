# TASK: fix-3 - keep hybrid radix reuse points alive (refresh snapshot LRU on validate/use) + verify the --linear-state-cache-ratio lever

Status: OPEN (composed 2026-09; anchors verified by direct file reads on vektory79 HEAD 5001504).
Read CONTRIBUTING.md first - it is binding. Bug fixes come with a test that fails before and passes after.
Context: user-observed periodic FULL cache loss in multi-turn agent conversations on GLM-5.3-Flash
GGUF hybrid (`--moe-cache-auto --kv-reserve-tokens 400000 --kv-cache-dtype fp8 --memory-ratio 0.82
--max-prefill-length 8191 --moe-strategy hybrid --moe-cpu-threads 16 --max-running-requests 1`).
Evidence artifacts: `.tasks/conversation/request1.json` + `request2.json` (two consecutive requests).
Diagnosis chat 2026-09; memory: ft-serve-cache-loss-midhistory-rewrite.

## Observed facts (byte-verified from the two request bodies)

- Messages 0..15 and 17..31 are byte-identical between the two requests; ONLY message 16 differs:
  5027 chars in request1 (question + injected blocks <general_delegation_checkpoint>,
  <memory_index_changes>, <recalled_memories>) vs 176 chars in request2 (the bare question; it is a
  substring of the request1 version). Request2 re-attaches the same injected blocks to its NEW last
  message (msg33, 4970 chars).
- So the client replays history NON-append-only: injected blocks ride the current user message and
  are stripped from it later (apparently when the message leaves a rolling verbatim window, ~17
  messages - exact client rule unknown, the byte fact is enough). Token divergence lands mid-history:
  here at ~84% depth (~34.2k of 40.7k prompt tokens).
- User report: on the last message the context was fully re-prefilled (cached=0). [observed by user;
  the per-request boundary-death sequence below is inferred - verify on hardware]

## Why reuse dies (root cause chain, code-verified anchors)

1. Client trigger (outside FreeToken): history rewrite moves the shared-token-prefix end into the
   middle of the prompt. The previous request's finish-donate snapshot sits at its very END
   (scheduler/cache.py:365-380, finish branch of _cache_req_hybrid) - BEHIND the divergence, hence
   unreachable: match_prefix climbs UP from the deepest matching node to the deepest node with a
   LIVE GDN snapshot (kvcache/hybrid_radix_cache.py:76-88) and never looks forward.
2. LRU ties (the Fix-3 defect): _walk refreshes node.timestamp for every node on the walked path but
   with ONE tic per call (hybrid_radix_cache.py:279 `tic = time.monotonic_ns()`, assignments at
   284-290; present since the initial release 3af9d90). The insert() dedup branch (108-110) and the
   match_prefix return add NO refresh beyond that tic. Therefore at any eviction moment every
   on-path snapshot carries the same timestamp; evict_mamba (168-189) orders candidates only by
   RadixTreeNode.__lt__ = timestamp (kvcache/radix_cache.py:107-108; timestamp born at :27, split_at
   preserves it via tic). heapq with all-equal keys keeps the DFS order of _snapshot_nodes() -> the
   victim among equals is arbitrary/biased, with no relation to reuse value.
3. Churn source: every request's finish-donate is a NEW unique node (prompt+generated length never
   repeats) -> +1 evictable-slot demand -> ~1 eviction per request. Evictable capacity is ~4 slots:
   pool = 4*mr + max(4, ceil(linear_state_cache_ratio*mr)) + 1 = 9 at mr=1, ratio 2.0
   (kvcache/linear_state_pool.py:280-289); 4 slots pinned by the running req (1 live + 2 ping-pong +
   1 locked committed snapshot) + 1 padding.
   Net: over a ~10-request agent loop the interior chunk-boundary snapshots 8128..32512 die
   stochastically; the next divergence then climbs from ~34.2k and finds NOTHING -> cached_len=0 ->
   full re-prefill. [inferred - confirm with the hardware validation below]

## The change (Fix-3)

Refresh node.timestamp at the two moments that PROVE a snapshot useful, so re-validated boundaries
become strictly younger than walk-tic nodes and LRU churn falls on one-off finish nodes:

1. insert() dedup branch (hybrid_radix_cache.py:108-110): before `return prefix_len, True` set
   `node.timestamp = time.monotonic_ns()` (+1-2 line comment: uniform walk tics make evict_mamba
   victims arbitrary; a dedup hit is the only proof the boundary is still a live reuse point).
2. match_prefix() (76-88): when returning a live snapshot, refresh `cur.timestamp` right before
   constructing the HybridMatch (+same style comment).

Expected diff ~4-6 lines plus comments. Do NOT change: _walk, split_at tic preservation, eviction
order logic, ref-count semantics, pool sizing, page-size/alignment conditions, check_integrity.

Review notes:
- evict_full (KV LRU) evicts unlocked LEAVES only; refreshing an internal boundary node does not
  affect it. Refreshing a leaf snapshot node also delays its KV eviction - desired (live reuse point).
- Steady state after the fix: interior boundaries immortal while re-validated every turn; per-request
  finish nodes rotate 1:1 (each new finish-donate evicts the previous consumed one). If boundaries
  outnumber evictable slots (prompt > ~4 boundaries x 8128 at 4 slots), the shallowest-validated die
  first (FIFO by dedup time) -> reuse degrades to the deepest surviving boundary, never to 0. Pin
  this in a test.

AMENDMENT NOTE (2026-09, wave-2 triage, both reviews): "FIFO by dedup time" is not literally
pin-able - every walk re-stamps the whole on-path with one tic (hybrid_radix_cache.py:279-297),
erasing earlier dedup refreshes; interior boundaries TIE at eviction and the victim among ties is
heap-order (model.py:355-368 treats ties as arbitrary). The property that actually holds: the
just-validated tip strictly survives; interiors are expendable. Additionally the production
refresh is neutralized at evict time: the refreshed node is mamba-locked before ensure_mamba_slots
can evict (prefill.py:88 -> :93-95; cache.py:394-406 -> :410), and other on-path nodes are
re-stamped by the next walk before they become evictable (unlock cache.py:397 after the insert
walk) - probe on 5001504: identical victim multisets and admission reuse pre/post-fix. Mechanism
decision pending with the user; see WAVES.md triage.

## Tests (fail before, pass after; extend existing files, do not create new ones)

Primary suite: tests/kvcache/radix/test_hybrid_radix.py (353 lines) with the model oracle
tests/kvcache/radix/model.py (evict_mamba semantics 719-745) + adapters.py:242. FIRST check whether
the oracle tracks timestamps; if yes, update it to the new semantics; if not, add ordering tests at
the adapter level. Determinism: RadixTreeNode(key_fn, tic=...) accepts a fixed timestamp
(kvcache/radix_cache.py:20) - use explicit tics so the pre-fix behavior is deterministic.

1. Dedup refresh: node K holds a snapshot at tic=T0; fill the evictable set with newer snapshots;
   re-insert K's ids (dedup, mamba_exist=True); force evict_mamba(1). Pre-fix: K is among the
   stalest -> evicted. Post-fix: K survives, a tie-class node dies.
2. Match-use refresh: same shape but the reuse goes through match_prefix() returning K's snapshot;
   force evict_mamba(1) -> K survives post-fix.
3. Steady-state churn: simulate 6+ request cycles (per cycle: boundary dedup inserts + one unique
   finish insert + a match) -> after N cycles boundaries 8128..32512 still hold live snapshots
   (pre-fix: stochastically dead).
4. Scheduler-level (tests/scheduler/test_hybrid_cache_manager.py): repeat chunked-prefill flow keeps
   boundary snapshots alive across N requests (CPU).
Run: `uv run pytest tests/kvcache/radix tests/scheduler -m "not slow"` (fix-1 baseline: scheduler +
kvcache/radix 244 passed).

## Part B - verify the --linear-state-cache-ratio theory (same wave, separate commit)

VERIFIED today: the CLI flag does NOT exist. engine/config.py:78-80 defines
`linear_state_cache_ratio: float = 2.0`; the only consumer is _linear_pool_num_slots
(kvcache/linear_state_pool.py:280-289, formula above); NO entry in server/args.py (nearest patterns:
--max-running-requests args.py:325 with dest mapping, --memory-ratio args.py:347; parser defaults
come from ServerArgs.<field>).

B1. Add the flag: ServerArgs field + `--linear-state-cache-ratio` (type=float) parser entry +
plumb into EngineConfig (trace the ServerArgs -> EngineConfig mapping first). ~10-15 lines.
B2. Static math for the probe (GLM-5.3-Flash, mr=1): ratio 2.0 -> 9 slots (4 evictable); ratio 8 ->
13 slots (8 evictable); +4 slots x 140.76 MiB (measured per-slot: 25 slots = 3.44 GiB, campaign
2026-09-16/17) ~= +563 MiB, subtracted from the pool envelope via state_pool_bytes
(engine/engine.py:483) -> planned MoE slots drop -> check_partition_floors
(engine/cache_budget.py:60-84) can fail-fast with "...lower --kv-reserve-tokens or raise
--memory-ratio". The 0.82/400k recipe plans ~1091-1093 slots vs floor 1059 (memory:
ft-gguf-serving-vram-headroom) - margin ~32-34 slots; whether +563 MiB fits must be MEASURED,
not extrapolated.
B3. Hardware probe (RTX 5090, serial runs, process-hygiene protocol: census, '[f]t serve' pkill,
watchdog, SIGTERM->SIGKILL): boot matrix {ratio 2.0 control, 4, 8} at (0.82, 400k, fp8, hybrid,
8191). Record: boot pass/fail + exact fail-fast text, peak VRAM (2 s phase-tagged sampler from
.tasks/dense-q80-gemm/). Then with Fix-3 in: replay the two conversation bodies for 5+ turns ->
read "#cached-token" per request in the server log (the verdict line; gauges #mamba-slot /
token-usage EXCLUDE evictable tree snapshots). Success: interior boundary hits (cached > 0,
~= divergence floored to the 8128 grid) on every turn at the smallest ratio that boots with
headroom.
Decision rule: if Fix-3 alone gives per-turn boundary hits at ratio 2.0, do NOT raise the ratio
(VRAM is razor-thin at 0.82/400k); the flag still lands as infrastructure for future probes.

## Hardware validation for Fix-3 itself (before the Part B probe)

- Pre-fix baseline (HEAD 5001504): replay the two conversation bodies alternately 5+ times; count
  the "#cached-token = 0" rate. Optional log-only debug dump of live snapshot boundaries at match
  time (do not commit the debug print).
- Post-fix: same replay -> no full misses; cached ~= divergence floored to the boundary grid; decode
  and prefill medians unchanged (reuse logic only).

## Commits

Two separate commits (one change per PR), Conventional Commits, lowercase, imperative:
- `fix(kvcache): refresh hybrid radix snapshot lru on validate and reuse`
- `feat(server): add --linear-state-cache-ratio flag`
Commit only when the user asks. Never push.

## Out of scope

- Client-side history stability (Veai strips injected blocks on replay) - the trigger, not fixable
  in FreeToken.
- Per-chunk donation changes, eviction-order redesign, automatic pool resizing.

## AMENDED MECHANISM (user decision, post wave-2 triage): independent snapshot-LRU stamp

The original Fix-3 constraint "do NOT change eviction order logic" is LIFTED for the evict_mamba
candidate ordering ONLY (see the Amendment Note in Review notes + WAVES.md triage A1: node.timestamp
differentiation is erased by walk re-stamps and lock ordering, making the pure-refresh fix a no-op
in production). New spec:

- New per-node field snapshot_lru (hybrid radix node): written ONLY at prove-use moments, so it is
  immune to _walk re-stamping. Initialize it wherever a live snapshot lands on a node (enumerate the
  acquisition sites: fresh-node insert/finish donate, split_at carrying the snapshot to the suffix
  node; the dedup branch keeps the existing stamp) - init value = time.monotonic_ns() at acquisition.
  WAVE-5 CORRECTION: in the actual split_at the fresh node is the PREFIX half and self keeps the
  suffix (and the snapshot and its stamp), so the copy lands on a snapshot-less prefix node - it is
  inert today and kept as defensive only; there is no snapshot move in split_at.
  Update it at the two existing Fix-3 refresh points: insert() dedup branch and match_prefix()
  live-snapshot return. Keep the Fix-3 node.timestamp refreshes (harmless; timestamp remains the
  KV-LRU key).
- evict_mamba candidate ordering: build the victim heap from ((node.snapshot_lru, node.timestamp),
  node) tuples per call, so ordering is FIFO by last validation with timestamp as tiebreak.
  RadixTreeNode.__lt__ and the evict_full (KV LRU) heap are UNTOUCHED. _walk, split_at logic,
  ref-count/lock semantics, pool sizing, check_integrity untouched.
- Expected behavior (now production-real): the just-validated boundary is strictly newest among
  candidates and survives while re-validated every turn; stale boundaries die FIFO by last
  validation; per-request finish nodes rotate 1:1; reuse degrades to the deepest surviving
  boundary, never to 0. "FIFO by dedup/validation time" is now literally pin-able.
- Tests (fails-before, pass after): T1/T2 MUST fill the evictable set and force evict_mamba(1)
  asserting K survives and a tie-class node dies; T3 asserts by boundary key (never by timestamp
  value guard) that interior boundaries 8128..32512 stay live; add the exhaustion pin: keep forcing
  evictions until one live snapshot remains - it must be the deepest boundary, match length > 0,
  victims never the just-validated tip; T4 (scheduler): size the pool so ensure_mamba_slots actually
  exhausts mid-turn in the real commit order and assert a re-validated boundary outlives a
  tie-class peer. Oracle model.py mirrors: stamp init at acquisition, stamp refresh at match/dedup,
  _pick_victim orders by (snapshot_lru, timestamp).
- Commits unchanged: fix(kvcache) + feat(server), only when the user asks.
- Out of scope still: client-side history, per-chunk donation changes, automatic pool resizing.
  evict_full/KV-LRU behavior and _walk semantics are HARD non-goals. split_at: exactly one allowed
  touch - copy snapshot_lru to the suffix node alongside the existing snapshot move (currency
  belongs to the snapshot, not the node); no other split_at change.

## AMENDMENT 2 (user decision, post wave-10): per-chunk ChunkedReq boundary donation
The "per-chunk donation changes" out-of-scope is LIFTED (evidence: interior snapshots never created; single
per-turn snapshot + alignment-gated finish-donate make every mid-history rewrite a full re-prefill; zero
evictions measured on hardware).
- Goal: each intermediate ChunkedReq chunk donates its boundary snapshot to the hybrid radix tree at the
  page-aligned chunk boundary (the frozen ping-pong track snapshot), so a mid-history divergence matches at
  the deepest live boundary below it instead of missing everything.
- Constraints: (1) donated pp slot becomes tree-owned - clear the req reference, alloc fresh pp (double-free
  hazard documented in the scheduler skip comment; reconstruct the old failure, add a regression test);
  (2) position consistency (chunk ends page-aligned: 8128 = 127*64); (3) final-chunk path + finish branch
  semantically unchanged, no double-donate (dedup insert path); (4) non-hybrid caches keep the skip;
  (5) evictable fund now load-bearing: check_integrity green under churn, _free_req_slots never frees
  tree-owned slots.
- Expected (CORRECTED by wave-12 review A3-1: L = deepest x64 STRICTLY inside the chunk, linear.py:120-127;
  end state = live slot): donation grid = k*8128 + 8064; turn-2 #cached-token = 48704 (deepest live <=
  divergence 52224; NOT 52224 and NOT 48768 - the split node has no snapshot); re-prefill = 11480; full
  misses 0/12 except cold turn 1; evictions > 0 (FIFO-by-snapshot_lru kills stalest-validated first);
  turn-2-class prefills improve massively.
- Tests (fails-before): 3-chunk ChunkedReq flow -> 3 boundary snapshots; divergence match hits a boundary
  not 0; double-free regression; fund-pressure eviction kills stalest-validated first (Fix-3 engagement);
  no double-donate; oracle mirrors the chunked flow if it models chunks.
- Hardware validation: post-change 12-request replay vs wave-7/8/10 baselines (turn-2 8128/0); no config
  change needed (ratio 2.0).
- Commit candidate (user asks): fix(scheduler): donate per-chunk boundary snapshots in chunked prefill.
