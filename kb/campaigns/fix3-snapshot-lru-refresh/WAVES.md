# Fix-3 wave log (DIRECT_EXECUTION, gguf-native-serving ORCHESTRATION.md)

Spec: TASK.md (this folder). Anchors authored on vektory79 HEAD 5001504.

## Wave 1 - Code (Fix-3 + Part B1 flag + fails-before tests)
- Status: DONE (subagent 39416f91-38a7-49dc-886d-37d80d5a72ba)
- HEAD verified: vektory79 @ 5001504, no anchor drift (all brief anchors re-checked by direct reads).
- Code: hybrid_radix_cache.py +7 (match_prefix live-snapshot refresh + insert() dedup refresh, ASCII comments);
  server/args.py +11 (--linear-state-cache-ratio after --memory-ratio; ServerArgs inherits SchedulerConfig(EngineConfig)
  so no extra plumbing needed).
- Tests: tests/kvcache/radix/test_hybrid_radix.py +61 (T1 dedup refresh, T2 match-use refresh, T3 steady-state churn;
  determinism via autouse deterministic_clock fixture = itertools.count patched over time.monotonic_ns; node lookup by
  node_end_path); oracle model.py +7 (stamp refresh on match-return + dedup, oracle DOES track stamps via MGroup.stamp /
  _pick_victim); tests/scheduler/test_hybrid_cache_manager.py +42 (T4 CPU chunk-commit dedup refresh, full 129 ids since
  match_req strips last token); tests/server/test_args.py NEW (no prior args test file existed; default 2.0, float parse,
  pool 9->13 slots).
- Fails-before: "5 failed, 36 passed" pre-fix (4 Fix-3 tests + args parse test). Post-fix: "251 passed, 1 skipped in 2.46s"
  (baseline 247 +1 skip; +4 new); args file "3 passed in 1.63s".
- B2 static math confirmed (linear_state_pool.py:280-289): mr=1 ratio 2.0 -> 9 slots (4 evictable), ratio 8 -> 13 (8);
  +4 x 140.76 MiB ~= +563 MiB via state_pool_bytes; fail-fast risk at 0.82/400k real, hardware-measure in B3.
- Census: preflight pgrep empty, GPU 1773/32607 MiB, 18 semaphores; postflight identical. Runs serialized, timeout 900,
  CPU-only. No commits. .veai/memory/* modified - orchestrator-owned, untouched by wave.
- Plan: anchor re-verify -> baseline pytest (tests/kvcache/radix tests/scheduler -m "not slow", expect ~244 passed)
  -> tests first (T1 dedup refresh, T2 match-use refresh, T3 steady-state churn, T4 scheduler-level;
  oracle decision in tests/kvcache/radix/model.py:719-745 / adapters.py:242; explicit tics)
  -> fails-before evidence -> fix (2 refresh points: insert() dedup branch + match_prefix live-snapshot return)
  -> Part B1 flag (--linear-state-cache-ratio, ServerArgs field + parser + EngineConfig plumbing, extend args tests)
  -> green run -> B2 static math -> census.
- Constraints: no _walk/split_at/eviction-order/pool-size changes; no GPU; no commit.

## Wave 2 - Review x2 (DONE)
- Review-A DONE (a034784f-206b-4503-a2ff-819d01bba647). Verdict: NON-BLOCKING: 0 runtime hazard; 2 major follow-ups.
  A1 [major] Fix is a NO-OP in production: refresh cannot change any evict_mamba decision - refreshed node is
  mamba-locked before ensure_mamba_slots can evict (prefill.py:88 lock -> :93-95 ensure(3); cache.py:394-406
  insert->unlock->match->lock -> :410 ensure(1)); other on-path snapshots re-stamped by next _walk one-tic before
  evictable again (unlock cache.py:397 happens after insert walk); insert dedup refresh re-stamped by the same
  commit's match walk. Dynamic probe (real vs pre-fix clone, 9-slot pool, 8 turns, 3 shapes): identical per-turn
  admission reuse + identical victim multisets; only dead-branch victim ORDER swaps. Only surviving refresh:
  same-node finish dedup (cache.py:349-380) -> dead-branch node strictly younger, delays reclamation one round
  (marginally adverse). A2 [major] Tests pin the refresh, not the spec'd eviction-outcome change: T1/T2 assert
  timestamp inequality only (no fill + force-evict survival), T3 evictions never assert survival, T4 pool=16 slots
  -> ensure_mamba_slots can never evict. A3 [minor] help says ceil, code int() truncates (linear_state_pool.py
  :284-285). A4 [minor] no ratio<=0 validation (sibling pattern engine.py:1972-1974). A5 [note] dead-branch
  reclamation delay. A6 [note] typing nits, pre-existing style. Verified OK: every live-snapshot return refreshes
  (hybrid:84-88, no missed path); dedup covers the only existing-snapshot branch (:111-116); refresh tic after walk
  tic -> strictly younger-or-equal; ref/lock untouched; split_at preserves; evict_full leaves-only claim true; heaps
  rebuilt per call; args end-to-end no drop-point; oracle faithful; det-clock pre-existing; scope minimal.
  41 tests green; probe artifacts in /tmp only.
- Review-B: DONE (a3034812-4ad2-4c06-bd7c-93d478761242). Verdict: production change correct and verified effective
  (hybrid_radix_cache.py:85-88/:111-115, args.py:356-367); BLOCKING: 2.
  B1 [BLOCKING/major] T3 not fails-before + vacuous order assert (test_hybrid_radix.py:400-413): probe showed T3 PASSED
  pre-fix and under both single-revert probes; `all(t < deepest ... if t != deepest)` filters by VALUE -> empty generator
  pre-fix -> vacuously True. Fix: filter by boundary key (t < deepest for every boundary != deepest).
  B2 [BLOCKING/major] Brief-required pin missing: shallowest-validated die first / reuse degrades to deepest surviving
  boundary, never 0. T3 never exhausts (7 boundaries, 3 forced evictions). Add test: keep do_evict_second(1) until one
  live snapshot remains; assert it is the deepest boundary, match length > 0, victims never the just-deduped tip.
  Caveat (record in TASK.md): "FIFO by dedup time" not literally pin-able - every walk re-stamps the on-path with one tic
  (hybrid_radix_cache.py:279-297), interior boundaries TIE at eviction, victim among ties is heap-order (model.py:355-368
  treats ties as arbitrary). Real property: just-validated tip strictly survives; interiors expendable.
  B3 [minor] args.py:356-367 help promises ceil, linear_state_pool.py:288 truncates (int) -> fractional ratio*mr > 4
  reserves one slot fewer than documented; fix math.ceil or reword help + fractional test case.
  B4 [minor] No validation on ratio <= 0 (silently clamps to 4-slot floor); sibling pattern swa_full_tokens_ratio
  validated engine.py:1972-1974; fail-fast ratio <= 0 or document clamp.
  B5 [minor] docs/cli.md "KV cache & memory" table (lists --memory-ratio at :70) lacks the new flag - add one row.
  B6 [note] T4 title says dedup refresh but commit-match walk erases the dedup refresh (cache.py:394 re-stamps after
  insert); assert satisfied by match refresh alone - dedup pinned by T1, coverage complete overall.
  B7 [note] T4 runs on real clock (det-clock fixture is package-autouse in tests/kvcache/radix only) - theoretical tie
  flake on b2.timestamp > b1.timestamp; port the 4-line deterministic_clock fixture into that test.
  B8 [note] deterministic_clock fixture is PRE-EXISTING (conftest.py:10-17, driver.py:33-43, function-scoped autouse,
  try/finally restore, fresh counter per test - no pollution risk); only tests/oracle stamps are new.
  B9 [note] Commit hygiene: two commits per TASK.md; test_args.py currently STAGED while rest unstaged - stage by
  explicit path list per commit, ZERO .veai/memory; commit only when user asks.
  B10 [note] TRAPS.md: T48 not violated (T4 sets field directly; forwarding covered at :258; replay signature
  65472@4096/65536@8128), T49 applied (isolated suite), T50/T43/T44 apply to hardware replay only; T45-T47/T51-T56,
  D01-D10 N/A.
  Verified-OK anchors: admission prefill.py:79 match before :94-96 ensure_mamba_slots(3); chunk-commit cache.py:394 match
  before :410 ensure_mamba_slots(1); refresh scope deepest live node/dedup node only, root excluded, no ref-count
  interaction; no non-GDN impact (separate classes radix_cache.py:253/257, swa_radix_cache.py:189/201/417 untouched);
  timestamp consumers only __lt__ + two heaps; T4 match_req strips last token (cache.py:98), CPU-only; oracle update
  required and correct (model.py:282-288/:680/:694, _pick_victim :355-368); battery does not compare stamps; args
  plumbing end-to-end (ServerArgs(SchedulerConfig) args.py:68, field engine/config.py:78-80, pool math pinned); suite
  254 passed, 1 skipped; "#cached-token" verdict line untouched (scheduler/status.py:84); IDE inspections: 6 warnings
  all pre-existing patterns.

## Triage (orchestrator)
- A1 [major, no-op mechanism] TP - corroborated by B1 probe (T3 passes pre-fix AND under both single-revert probes),
  B2 caveat, A1 static proof + dynamic probe. ESCALATED to user: amend brief (independent mamba-LRU stamp) vs
  keep-as-no-op + hardware-first vs revert refresh. Test-contract fixes (B1/B2/A2) are decision-DEPENDENT - deferred.
- B1/B2/A2 [blocking/major, test contract] TP (merged) - deferred to post-decision wave.
- A3=B3 [minor, ceil] TP - math.ceil in _linear_pool_num_slots to match help + config comment + brief math;
  fractional test case. Decision-independent.
- A4=B4 [minor, validation] TP - mirror swa_full_tokens_ratio fail-fast (engine.py:1972-1974). Decision-independent.
- B5 [minor, docs row] TP - docs/cli.md KV cache & memory table. Decision-independent.
- B7 [note, T4 real-clock tie flake] TP but decision-DEPENDENT (test file rewritten under any mechanism option).
- A5 folded into mechanism decision; A6/B6/B8 informational; B9 commit hygiene for STOP GATE (test_args.py currently
  staged - restage explicitly per commit, ZERO .veai/memory); B10 TRAPS verdicts recorded (T43/T44/T50 -> hardware
  replay; T48 not violated; T49 applied).

## Wave 3 - Polish (decision-independent TP: A3/B3, A4/B4, B5)
- Status: DONE (4c7eca69-1798-4776-995c-278adfa66ffa). ceil fix in _linear_pool_num_slots, ratio<=0 fail-fast
  (mirrors swa_full_tokens_ratio), docs/cli.md row + help reword (formula incl. floor+padding), +2 args tests
  (fractional, invalid). Suite green (135+5 passed). Census clean. Boundary files untouched. No commits.

## Decision (user): AMEND the brief - independent mamba-LRU stamp + fails-before tests; hardware validation after.
Recorded in TASK.md as "AMENDED MECHANISM (user decision, post wave-2 triage)". Key spec: per-node snapshot_lru
written only at prove-use moments (acquisition sites init, dedup + match refresh; split copies it alongside the
snapshot move - the single sanctioned split_at touch); evict_mamba victim heap = ((snapshot_lru, timestamp), node)
tuples; __lt__ / evict_full / _walk / ref-locks / pool sizing untouched. Just-validated boundary strictly newest;
stale boundaries die FIFO by last validation; reuse degrades to deepest surviving boundary, never 0 - now
literally pin-able.

## Wave 4 - Amended mechanism implementation (DONE 60b5b81b-ecb9-4808-a893-0428af2ea108)
- Engine: radix_cache.py RadixTreeNode.snapshot_lru (default -1); split_at copies it (mutates self into suffix ->
  copy inert today, defensive; snapshot never moves node objects). hybrid_radix_cache.py: attach site stamps
  snapshot_lru alongside node.mamba_value (ONLY acquisition site: fresh insert/finish-donate/tombstone refill);
  dedup + match re-stamps BOTH snapshot_lru and timestamp; evict_mamba heap = ((snapshot_lru, timestamp), node).
  __lt__/evict_full/_walk/ref-locks/pool-sizing/check_integrity untouched.
- Oracle model.py: MGroup.snap (default -1), init at attach, refresh at match/dedup tic-for-tic, PageTrie.split
  copies; _pick_victim key param, mamba pass keys (snap, stamp), exact ties accepted; evict_full/evict_swa
  stamp-only. Tests: 3 old inequality tests replaced, 4 new cache-level (dedup/match/churn/exhaustion) + scheduler
  test_pool_exhaustion_evicts_the_stale_validated_boundary (7-slot pool, off-path decoy) + det-clock port (B7).
- T4 findings: (1) LinearStatePool padding sink - num_free == num_slots-1 idle -> pools sized +1; (2) admission
  match_req always re-validates the deepest live on-path boundary -> shallow-boundary survival unpinnable at
  scheduler level (DFS-first among walk-tic ties coincides with FIFO-by-validation; matches wave-2 probe).
- Fails-before: pre-amendment batch "4 failed, 36 passed" (dedup, churn, exhaustion, scheduler-pool); T2 initially
  by construction only -> CLOSED by match-only single-revert probe: "1 failed, 25 passed", ONLY T2 failed
  (ModelMismatch: non-LRU victim evicted while strictly older candidates existed); dedup + exhaustion green under
  match-only revert (dedup-pinned). Restoration verified bit-for-bit (git diff sha256 f7fe440a... identical).
- Final: radix+scheduler "253 passed, 1 skipped"; + args "5 passed" -> full wave "258 passed, 1 skipped in 2.42s".
- Census clean (GPU 1769/32607 MiB, 18 semaphores, no leftovers). Files: hybrid_radix_cache.py, radix_cache.py,
  model.py, test_hybrid_radix.py, test_hybrid_cache_manager.py (+ wave-3 polish files). No commits.

## Wave 5 - Review x2 of amended mechanism (DONE both). VERDICTS: Review-A2 NON-BLOCKING: 0; Review-B2 NON-BLOCKING: 0.
- Review-B2 (6bcbfbe0-47bb-4cc4-a32d-0fb2311fe5c0) findings:
  B2-1 [minor] STALE staged test_args.py (git status AM; index bf5fa10 = 3 tests, worktree 5020fbd = 5) - committing
    feat(server) from the index would ship 3 of 5 tests. Fix: re-stage before commit.
  B2-2 [minor] two 3-line comments exceed the 1-2 line rule (hybrid:84-86, :113-115) - trim to 2 lines.
  B2-3 [note] split_at copy inert-by-direction: new_node is the PREFIX half; self keeps suffix+snapshot+stamp; copy
    lands on a snapshot-less prefix node that _snapshot_nodes never yields; any later attach re-stamps (insert:120).
    TASK.md wording corrected by orchestrator (wave-5 correction note added).
  B2-4 [note] 8 PyUnresolvedReferences on hyb.second.free (Optional SlotLedger) - matches file style (donated()
    helper); optional bind+assert pattern. No action this wave.
  B2-5 [note] T3/exhaustion victim tracking via slot->length dict inversion - deterministic (LIFO list allocator,
    linear_state_pool.py:130-136), seed-stable; optional cleaner boundary-key-set tracking. No action.
  B2-6 [note] real-clock full-tie edge: two separate monotonic reads (hybrid:88-89/:116-117) can fully tie within one
    tick -> victim among exact tie is heap-order; spec accepts exact-tie arbitrariness; just-validated tip locked
    before ensure -> no correctness hazard. Sound as designed.
  B2-7 [note] T4 7-slot hard-code immune to _linear_pool_num_slots refactors (pool built directly); coupled only to
    the slot-0 padding sink (linear_state_pool.py:126-127); math verified: peak hold 3 snapshots + 3 req4 slots = 6
    usable -> exactly one eviction; if sink removed the test fails LOUDLY. Acceptable.
  B2-8 [note] det-clock fixture duplicated into tests/scheduler (verbatim port of driver.py:23-36) - hygiene
    verified (function-scoped autouse, try/finally, fresh counter, no pollution); promote to shared conftest if a
    third consumer appears.
  B2-9 [info] eviction is silent (no logger in the three files) - do NOT add logging this wave; if ever needed,
    one logger.debug in CacheManager.ensure_mamba_slots (scheduler/cache.py:138-146).
  B2-10 [info] hardware-replay readiness: "#cached-token" untouched (scheduler/status.py:84); ceil fix cannot alter
    replay pool sizing at the default (ceil(2.0*1)=2 -> max(4,2)=4 -> pool 9, same as int()); baseline sequence =
    stash the 2 source files only (hybrid_radix_cache.py, radix_cache.py), boot 0.82/400k fp8 hybrid 8191, replay
    request1/request2 alternately 5+ turns, collect "#cached-token", pop + verify. Serial runs (T31/D07); boot may
    fail-fast per T28/T50 -> ladder rather than blame the change; baseline must reproduce the historical hit
    pattern before post-fix delta is interpretable (T44); prefill medians exclude the last full chunk (T43).
  TRAPS verdicts: T48 not violated (T4 sets mamba_last_track_seqlen directly; replay reads the verdict line);
    T49 respected; T43/T44/T50 apply to the replay; T27/T25/T26 replay mechanics; T31/T32/D07 serialized discipline;
    T45 superseded (stash-A/B instead of env kill switch); T46/T51-T58/D01-D06/D08-D10 N/A.
  Verified-OK: fixture hygiene (3 seeds + isolation runs green); assert-by-KEY everywhere (zero timestamp-value
    guards); exhaustion pin exact; no tiebreak hard-coding (strictly distinct keys under det clock; oracle accepts
    any tie-class victim); T4 decoy genuinely discriminates; integration blast radius zero (snapshot_lru read only
    at the evict_mamba heap key hybrid:186; scheduler never reads node.timestamp; swa has own _tick; check_integrity
    stamp-blind until wave-6 fix; no public API/repr change); oracle mirror 1:1; wave-2 B6 caveat RESOLVED (the
    dedup test genuinely pins the dedup site: req3 insert-walk re-stamps b1/b2 with one tic AFTER req3's
    admission-match refresh); commit split listed (fix(kvcache) = 5 files, feat(server) = 5 files, zero .veai/memory).
  Independent checks by reviewer: suite re-run "45 passed" over the 3 relevant files; 6 new tests green under
    PYTHONHASHSEED {0,1,12345} and in isolation; line-by-line trace of all 5 eviction-outcome orderings reproduces
    the claimed pre/post outcomes exactly.

## Wave-5 Triage (orchestrator)
- TP (wave-6 fix): A2-1 check_integrity sentinel pin (behavior-neutral assert; deviation from the original brief's
  "don't touch check_integrity" letter is intentional and recorded - the amendment's intent is behavior-preserving);
  B2-2 comment trims (+ "defensive" in the split copy comment per B2-3); B2-1 re-stage test_args.py.
- No action: A2-2/B2-3 (TASK.md wording fixed by orchestrator), B2-4..B2-9 (note/info, recorded above),
  B2-10 adopted into the hardware-wave plan.

## Wave 6 - Fix TP (DONE, resume 60b5b81b): integrity pin + comment trims + restage. Suite "258 passed, 1 skipped
in 2.36s"; test_args.py re-staged (A only, AM gone); census clean. No commits.

## STOP GATE (code phase) - PASSED (commit gate deferred to user per TASK.md "Commit only when the user asks")
- [x] brief validation gates: fails-before empirical for all 5 eviction-outcome tests -> passes-after
- [x] all findings triaged, no open TP (A2-1/B2-1/B2-2 fixed in wave 6; notes recorded)
- [x] test executed (multiple independent runs incl. reviewer re-run + seeds/isolation)
- [x] TRAPS.md checked (B10 + B2-10 verdicts)
- [ ] commit - USER-OWNED; ready-to-commit split: fix(kvcache) = kvcache/hybrid_radix_cache.py,
      kvcache/radix_cache.py, tests/kvcache/radix/model.py, tests/kvcache/radix/test_hybrid_radix.py,
      tests/scheduler/test_hybrid_cache_manager.py; feat(server) = server/args.py, kvcache/linear_state_pool.py,
      engine/engine.py, docs/cli.md, tests/server/test_args.py (staged, A). ZERO .veai/memory in either.
- [x] artifacts: WAVES.md + amended TASK.md in this folder
- [x] census clean

## Wave 7 - Hardware A/B replay (DONE 93c3a26a-4a7b-4e63-9b36-1423bc2e4e82). VERDICT: fix does NOT change
per-turn reuse - pre/post admission patterns BYTE-IDENTICAL, full misses 2/12 BOTH legs. Capacity-bound, not
ordering-bound.
- Recipe: /media/ai/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL.gguf (served name
  "GLM-5.3-Flash-UD-Q3_K_XL.gguf"); 0.82 SURVIVED both boots (no ladder); ServerArgs echo incl.
  linear_state_cache_ratio=2.0; boots 141 s / 76 s; 0 FAILPAT hits; graceful teardowns (rc 143 by design).
- Per-request (turn, body, #new/#cached): 1 r1 8128/0 | 2 r2 8128/0 | 3 r1 11/58304 | 4 r2 24/60160 |
  5-7 r1 11/58304 | 8,10,12 r2 24/60160 | 9,11 r1 11/58304 - IDENTICAL both legs; gen_deltas deterministic
  (1766,1364,1722,2766,1912,819,1530,2526,1547,1355,1641,3342).
- Mechanism: turn-1 chunked prefill (~58k tok, NOT ~40.7k as brief said) creates ~9-10 boundary snapshots vs
  ~3-4 evictable slots (gauge 3/8) -> divergence-floored boundary 32512 dies of CAPACITY during turn 1 -> the
  r1->r2 rewrite turn (divergence ~34.2k) full-misses regardless of victim SELECTION (all snapshot_lru changes
  is the victim). Turns 3-12 hit their own complete path via the deepest boundary (58304/60160), stable.
- Medians: prefill 809.10 vs 807.41 tok/s; decode 16.00 vs 15.85 - unchanged within noise (criterion PASS).
  TASK.md "post-fix no full misses" criterion FAIL at ratio 2.0. Fixed-body replay exercises the divergence
  exactly ONCE - the periodic real-world loss (growing history) is not reproducible with this replay design.
- Restoration proven: stash pop + sha256 both files match; git diff --stat identical (38 files, 330+/84-);
  stash list empty; HEAD 5001504. Census clean every check. Artifacts /tmp/fix3_logs/* + /tmp/fix3_*.py.
- COROLLARY (brief decision rule): Fix-3 alone does NOT give per-turn boundary hits at ratio 2.0 -> raising the
  ratio IS sanctioned; at mr=1 ratio 4 = 2.0 (max(4,.) floor) -> only ratio 8 (pool 13, 8 evictable) matters.
  Capacity math: 8 evictable fits the ~7-8 boundaries of a 58k prompt -> 32768 should survive -> turn-2 hit.
  VRAM cost +4 slots x 140.76 MiB ~= +563 MiB vs margin ~32-34 MoE slots - MEASURE in wave 8 (Part B3).

## Wave 8 - Part B3 boot matrix + ratio-8 replay (DONE b00dc313-7c6f-4a17-8430-26f2ee9a2f3f)
- BOOT MATRIX (Fix-3 active, mr=1, fp8/hybrid/8191): ratio 2.0 @0.82/400k BOOTED 111s (moe_cache_size=1102,
  groups [331,288,288], gauge 3/8 -> pool 9, boot peak 27210 MiB); ratio 4 @0.82/400k BOOTED 108s - IDENTICAL plan
  (control pair confirmed, mr=1 max(4,.) floor); ratio 8 @0.82/400k FAIL-FAST ~15s: "budget cannot fund the
  per-signature slot floors ... byte-weighted minimum of 1059 layer-0-width slots vs the planned 1048; lower
  --kv-reserve-tokens or raise --memory-ratio" (planned 1102->1048 = -54 slots = +563 MiB pool cost CONFIRMED);
  ratio 8 @0.81/350k FAIL-FAST (planned 1049 < 1059); @0.80/350k FAIL-FAST (1020 < 1059) - the brief ladder
  premise was computed without the ratio-8 pool cost and is measured-wrong; WINNER 0.82/350000 + ratio 8 (kept
  0.82, applied only the reserve cut): BOOTED 102s, moe_cache_size=1078 (floor 1059, margin 19), groups
  [307,288,288], KV 350016 tok/2.08 GiB, gauge 3/12->4/12 -> pool 13, boot peak 27231 (free 4919), REPLAY peak
  31242, free-at-peak 907 MiB (thin; prior 0.80/400k winner had 2008).
- REPLAY at winner (12 req, r1/r2 alternating): 1 r1 8128/0 (cold) | 2 r2 8128/0 FULL MISS | 3 r1 11/58304 |
  4 r2 24/60160 | 5-12 self-hits - BYTE-FOR-BYTE the wave-7 ratio-2.0 pattern; full_miss_count 2/12 identical.
- DECISIVE GAUGE EVIDENCE: #mamba-slot never exceeded 4/12 during the whole replay (555x "4/12", 15x "3/12")
  -> the 12-gauge-slot pool was never filled past 4: POOL CAPACITY IS NOT THE BINDER. Interior boundary
  snapshots do NOT accumulate in the pool; the rolling ~4 = live + 2 ping-pong + committed (superseded+freed
  per commit, hypothesis pending code confirmation).
- VERDICT per B3: SUCCESS CRITERION NOT MET at any bootable ratio (2.0/4 -> pool 9 full-miss; 8 -> pool 13
  also full-miss). Decision rule: NO per-turn boundary hits; the flag is fundable infrastructure (measured
  +560 MiB vs +563 predicted, medians 806.53/15.77 - no cost) but the turn-2 rewrite-transition miss persists.
- Census: all 6 legs clean teardowns; postflight 27 sems (+9 = known-benign slot-floor fail-fast leak);
  tree untouched (no edits/commits); one poll cmd SIGKILLed by system (exit 137, no data lost).
- REVISED ROOT-CAUSE HYPOTHESIS (needs code confirmation before user escalation): the chunk-commit path FREES
  the superseded boundary snapshot's slot (rolling supersede) instead of retaining/donating it -> interior
  boundary reuse impossible BY POLICY, not by pool size or LRU order; if superseded snapshots were RETAINED
  while the evictable fund lasts, ratio 8 (8 evictable) + Fix-3 (deterministic FIFO churn) would give turn-2
  hits at the divergence-floored boundary. NOTE: the brief lists "per-chunk donation changes" OUT OF SCOPE
  -> material scope decision for the user.

## Wave 9 - Code research: slot lifecycle (DONE 93d3ca5b-af48-4abc-a4c5-393742d4a652) - INTERPRETATION FLIP
- HEADLINE: NO supersede-free exists - interior boundary snapshots are RETAINED (unlocked/evictable) until lazily
  evicted; commit attach = zero-copy MOVE of a pp slot, pp replaced by fresh alloc (cache.py:409-413); unlock of
  the previous boundary (cache.py:396) makes it EVICTABLE, not freed. Both retention hypotheses REFUTED by code;
  retention already exists -> "retain instead of free" would be a NO-OP. Per-chunk donation is unconditional
  (no flag controls it).
- GAUGE STRUCTURE: scheduler.py:457-466 + cache.py:114-117: used = total - (num_free + mamba_evictable) =
  live + 2pp + LOCKED only -> evictable tree snapshots CANCEL OUT of the gauge. The wave-8 "gauge never >4/12
  => pool unused" inference and wave-7 "capacity-bound" claim are UNSOUND.
- PREDICTION vs MEASUREMENT CONFLICT (unresolved): static lifecycle predicts turn-2 partial-hit >=32448 at pool 9
  (evictions kill stalest N1-N3, N4 alive) and no deaths at pool 13 -> cached_len >= 32448, not 0. Measured
  8128/0 twice. NEW leading hypothesis: LOG-LINE PARSING ARTIFACT - if the parser took the largest-#new line per
  request, r2's 8128/0 could be the POST-DIVERGENCE chunk while chunk 1 actually hit ~49k cached (also fits
  turn-1 8128/0 and turn-3 11/58304). Genuine-miss candidates: harness overlap timing; snapshot-eliminating path
  outside scheduler/kvcache.
- Retention-change sketch (if ever needed, currently NOT needed): no code change for retention; (a) observability
  +5 lines in status.py:131-137 (tree snapshot count); (b) optional keep-recent-K guard in evict_mamba
  (hybrid:177-201) via keep_k threaded through ensure_mamba_slots ~20-30 lines; pinned-4 invariant intact
  (candidates filtered mamba_ref_count==0 hybrid:181; _linear_pool_min_slots=4*mr+1).

## Wave 10 - Diagnosis (DONE 03cbae8e-b919-49aa-a970-761a2266d158). VERDICT: turn-2 "8128/0" is a GENUINE full
miss (raw logs airtight: 7x"8128/0"+"3288/0" = 60184, every line cached=0; emitter = one line per prefill batch,
status.py:70-97; analyzer picks chunks[0] = true admission; NOT a parsing artifact).
- REAL MECHANISM ([fix3dbg] replay r1,r2,r1 @0.82/400k, boot 157s, FAILPAT clean):
  (1) scheduler.py:329-336 `if isinstance(req, ChunkedReq): ... continue` - intermediate chunks NEVER call
  cache_req; per-chunk x64 track snapshots never donated; only the FINAL chunk commits -> ONE attach per turn
  (r1@58304, r2@60160; attach count 2). The wave-9 "per-chunk commits retained" premise FALSIFIED (it traced a
  path intermediate chunks never execute).
  (2) Finish-donate SKIPPED: align_down(60083/61653) != cached_len -> cache.py:383-394; slots [7,5,8]/[5,4,8]
  freed, keep_live=False.
  (3) Turn-2: tree = ONE live snapshot (58304); match splits at 52224 (page-aligned divergence ~89.6%; r2
  msg16=210 vs r1 5027 chars, NOT substrings - memory detail corrected), climbs from the fresh prefix node,
  58304 in the suffix child beyond divergence -> cached=0 -> full 60184 re-prefill. ZERO evict_mamba victims;
  pool size/LRU played no role. Body check: r1=32 msgs, r2=34; indexes 0..15 byte-identical; extra diffs at
  32,33 (trailing, irrelevant); volume depth 0.842/0.809.
- CONSEQUENCE: Fix-3/snapshot_lru and --linear-state-cache-ratio CANNOT help at <=2 snapshots/turn. Real fix =
  per-chunk ChunkedReq boundary donation (brief out-of-scope; double-free hazard documented in the skip comment)
  + then pool sizing matters; turn-2 floor would be 52224 (re-prefill ~7.9k, not 60.2k).
- Restoration: 3 files sha256 == pre-values; git diff sha256 bit-for-bit; suite "258 passed, 1 skipped in
  2.72s"; census clean (27 sem.mp- pre-existing). Artifacts: /tmp/fix3_logs/dbg_*.{log,txt}, /tmp/fix3dbg_backup/.
- Memories repaired: fix3-hardware-ab-verdict (final mechanism) + ft-serve-cache-loss-midhistory-rewrite
  (corrected substring detail + ChunkedReq skip mechanism + superseded-claims list).

## USER DECISION (post wave-10): continue in-session - implement per-chunk ChunkedReq donation; commits after.
Recorded in TASK.md as AMENDMENT 2.

## Wave 11 - Per-chunk ChunkedReq boundary donation (DONE e5924b77-eb2c-47aa-bd7e-fefc2e7f123b)
- HAZARD (reconstructed; comment from 3af9d90, never fixed): overlap-ordering, not cache logic. overlap_loop
  schedules the NEXT chunk (try_add_one inherits chunked_req.{cache_handle,mamba_ping_pong,...}) BEFORE the
  prior chunk's drain (_process_last_data): drain-point donation would (a) donate pp[frozen] zero-copy while
  the continuation's stale tuple still lists the slot -> _free_req_slots frees a tree-owned slot at finish;
  (b) stale admission handle cached_len=M + zero-copy page adoption [M,L_prior) -> next commit dedup free
  re-frees tree-canonical pages; (c) per-commit unlock decs the same admission node once per chunk for one
  inc -> underflow -> premature eviction. Corollary: no drain-point fix without new req state.
- DESIGN: donation at CONTINUATION CREATION (prefill.py try_add_one) - strictly after prior forward completes,
  BEFORE state copy; calls the existing _cache_req_hybrid chunk-commit branch verbatim (cache_req(
  chunked_req, finished=False)): insert at tracked xCHUNK boundary L, zero-copy donate frozen pp slot,
  unlock/re-lock rebound handle (one chain lock), fresh pp after ensure_mamba_slots, clear L. Hybrid-gated
  (cache_manager.is_hybrid, cache.py:46); scheduler.py skip STAYS (comment rewritten, code unchanged); final
  chunk + finish branches untouched; L consumed exactly once (dedup backstop); idempotent under budget-bounce
  retries; no new req fields; no cache.py/hybrid/oracle changes.
- DIFFS: prefill.py:237-246 (+10); scheduler.py:323-331 (comment 3->9 lines); test_hybrid_cache_manager.py
  +360 (helpers _chunked_setup/_forward_chunk/_snapshot_chain; 6 new tests + 1 rewritten
  test_prefill_continuation_forwards_mamba_last_track_seqlen).
- FAILS-BEFORE: "7 failed, 13 deselected in 2.04s" (all 7, right reasons). POST: "7 passed"; full wave
  "264 passed, 1 skipped in 2.45s" (baseline 258+1).
- POOL WALKTHROUGH (3-chunk cold, mr=1 ratio 2.0): admission free 8->5; donations free 4->3 (tree 1e/1l ->
  2e/1l with tracking final); finish frees to 5-6; floor never violated; conservation holds; no sizing change.
- NOTES: donation at continuation-creation (not drain) = the only overlap-safe point for a zero-copy move
  (documented in comments); rewritten test's chain-integration intent preserved and strengthened; T-B pins
  mechanism at test scale - production 48768 numbers = hardware wave.

## Wave 12b - Fix TP (DONE e5924b77): all 7 items. 265 passed, 1 skipped in 2.50s (+1 test). 4 tests
extended + 1 new (test_aligned_final_boundary_finish_live_donate), 4 comment/docstring items. Honest deviation
on B3-3: the named mechanics (finish live-donate hits mamba_exist within ONE chain) are geometrically
impossible - every chain boundary < cached_len, so an aligned cached_len's live-donate node is always fresh
(mamba_exist=False -> live DONATED, keep_live=True, +2); the test pins BOTH halves: phase A = literal geometry's
true behavior (drain donates L=128; aligned live-donate donates at 192, +2, conserved), phase B = dedup
mechanics via production-real re-prefill (191-token admission can't reach the 192 node's page key -> aligned
live-donate lands on the EXISTING 192 node -> keep_live=False, +3, tree untouched). Both phases end with
check_integrity. Two assert corrections caught by the run: chain lengths are SPANS; _free_req_slots always
clears linear_slot_idx. Census clean. TASK.md untouched by the wave.

## Wave 13 - Hardware validation of the donation (DONE 58ee51fc-7765-42b6-9ea6-a71ae3a9349c). VERDICT:
AMENDMENT-2 ACCEPTANCE PASSES. turn-2 admission cached = 48704 EXACTLY (acceptance key), re-prefill 11480
(exactly [48704,60184)), full misses 1/12 (cold only) vs baseline 2/12, medians unchanged (r1 prefill 807.3
vs 809; r2 prefill metric now VACUOUS - no full re-prefill chunks exist; decode 15.835/16.358 within noise).
- [fix3dbg] attribution: 10 attaches (baseline 2): t1 L={8064,16192,24320,32448,40576,48704,56832,58304}
  (8 chunk-attaches; final drain 58304 - grid correction vs the memory's 58240, includes 40576); t2 2 attaches
  L=56768 (prefix_len=52224 - the EXACT divergence point visible in the tree) + 60160; steady turns 0.
- 5 evictions STRICTLY FIFO-by-snapshot_lru: 8064(lru...0009) -> 16192(...0105) -> 24320(...0205) ->
  32448(...0306) -> 40576(...0406), all stalest-validated first, tip never a victim, all internal tombstones
  (KV kept); fire at attach-time ensure_mamba_slots; zero in steady turns. Fix-3 ordering LOAD-BEARING and
  hardware-verified.
- Turns 3-12: full self-hits 11/58304, 24/60160 (byte-identical to baseline steady state). Boot 121 s; gauge
  4/8; peak VRAM 31222 (~936 MiB free). Census clean; FAILPAT never fired; restoration sha256 bit-for-bit
  (hybrid_radix_cache.py, cache.py, prefill.py); final suite "265 passed, 1 skipped in 3.29s". Artifacts:
  /tmp/fix3_logs/postdon_{server,runner}.log + sha256 ledgers.
- MEASURED GRID CORRECTION: turn-1 donation grid = {8064,16192,24320,32448,40576,48704,56832,58304}.

## STOP GATE (wave-13) - PASSED (commit gate = user pre-authorized "commits after")
- [x] AMENDMENT-2 hardware acceptance: turn-2 48704, 0 full misses except cold, evictions FIFO-verified
- [x] suite green (265 passed, 1 skipped); reviews triaged (wave-12b TPs landed); TRAPS watched (T50 no
  fail-fast; T44 re-anchored; T48 superseded signature confirmed)
- [x] census clean; instrumentation removed bit-for-bit
- [ ] COMMIT - in flight (3 commits, see below)

## Commit wave (DONE ba0f6e2d-8718-4e17-ba70-1d5bba323750): 3 commits verified, suite green, zero .veai staged
- 9732be0 fix(kvcache): order mamba eviction by a per-node snapshot lru stamp (4 files)
- e5730e0 feat(server): add --linear-state-cache-ratio flag (5 files; test_args.py unstaged from the stale
  index first, re-added fresh)
- 7080824 fix(scheduler): donate per-chunk boundary snapshots in chunked prefill (3 files)
- Suite after: "265 passed, 1 skipped in 2.47s" (266 collected); git log: 3 commits atop 5001504; census
  clean (27 semaphores unchanged, no leftovers). No push, no Assisted-by.
- DEVIATION: 2 comment-only files (attention/linear.py +1, scheduler/cache.py +2 - the wave-12 TP comment
  fixes) were omitted from the 3 path lists and remain modified-unstaged -> 4th commit in flight.

## Wave 14 - 4th commit + skill self-update (RUNNING, background 661b2071-504a-4560-bea5-4115f44c1908)
- 4th commit: docs: pin chunk-donation slot invariants in code comments (linear.py + cache.py).
- Skill self-update (ORCHESTRATION sec. 11, files stay UNCOMMITTED): new traps (gauge excludes evictables;
  alignment gate silently kills finish-donate; ChunkedReq skip + continuation-creation donation point; L grid
  k*chunk+8064; per-batch log-line semantics), in-place T48/T45/T50 updates, ORCHESTRATION sec.7 hygiene
  bullet ([fix3dbg] attribution pattern), task-06 A/B replay expectation class; counter sync across TRAPS.md
  H1 + SKILL.md + ORCHESTRATION sec.10; ASCII only.
- C1 fix(kvcache): order mamba eviction by a per-node snapshot lru stamp - kvcache/hybrid_radix_cache.py,
  kvcache/radix_cache.py, tests/kvcache/radix/{model,test_hybrid_radix}.py
- C2 feat(server): add --linear-state-cache-ratio flag - server/args.py, kvcache/linear_state_pool.py,
  engine/engine.py, docs/cli.md, tests/server/test_args.py (staged A)
- C3 fix(scheduler): donate per-chunk boundary snapshots in chunked prefill - scheduler/prefill.py,
  scheduler/scheduler.py, tests/scheduler/test_hybrid_cache_manager.py
Staging hygiene: explicit path lists only; verify staged==worktree per file; ZERO .veai/memory staged;
fast suite re-run after all 3; no push; no Assisted-by trailer (user has not asked for attribution).
- Review-A3 (af1c179e) findings:
  A3-1 [major, ACCEPTANCE MATH not code]: turn-2 expected match = 48704 NOT 48768: L = chunk_start + 8064
    (linear.py:121-127 computes c=(extend-1)//CHUNK; kernel/fla/kda.py:1236-1242 h[i]=state at chunk START;
    extend-end state lives ONLY in the live slot glm5_next/kda.py:168-171). Grid {k*8128+8064}; deepest live
    <= 52224 = 48704 (761*64); re-prefill 11480. TASK.md Expected numbers corrected by orchestrator.
  A3-2 [minor, latent]: _build_track_metadata stamps L + flips at _prepare_batch BEFORE the forward writes
    the slot - a future non-fatal skip between prep and launch would donate an unwritten slot; today every
    failure propagates (scheduler.py:242-251, no try/except). Fix: 1 comment line (in fix-TP wave).
  A3-3 [minor test gap]: budget-bounce retry idempotency untested (mechanism sound by trace); fold into
    T-A1 assert (in fix-TP wave).
  A3-4 [note]: req.cached_len (8128) vs cache_handle.cached_len (8064) divergence is intended and load-bearing
    (excludes hazard (b)); swa-cap floor would understate hybrid+SWA continuation reclaim floor by <=1 page -
    no hybrid+SWA model exists; revisit if one lands.
  A3-5 [note]: "no cache.py changes" is true but fragile (three commit-branch invariants: L-consumed-once,
    lock-before-ensure, dedup backstop). Accept as-is.
  A3-6 [note]: replacement alloc can raise (linear_state_pool.py:126-130) - unreachable under current sizing;
    wave-11 multiplies commit frequency ~8x per long prompt; WATCH on hardware.
  Verified-OK (A3): overlap ordering (:242 before drain :259; normal_loop :280/:287); donation prefill.py:245-246
    strictly before _add_one_req :247-258; complete_one (engine.py:1449-1451) already ran; engine-stream
    ordering holds (donation = host-side bookkeeping); all three hazards faithful AND structurally excluded
    (incl. a WORSE variant of (a): pp alternation clobbers the tree-owned slot - excluded by pre-copy fresh
    replacement cache.py:411-413); L-content match (kda h[i]=start-of-chunk); L=None no-op; unaligned L
    cleared; one-snapshot-per-chunk = track granularity, satisfies Amendment-2 goal (48704 <= 52224); exactly
    one chain lock; check_integrity green under churn; no double-donate (L consumed once; dedup backstop real;
    final drain + finish live-donate don't collide); finish branches byte-identical (cache.py NOT in diff);
    scheduler.py comment-only; blast radius zero; test rewrite preserves+strengthens (no coverage lost);
    production trace: turn-1 donations {8064,16192,24320,32448,48704,56832,58240(final drain)}, steady state
    3 req + 4 unlocked + locked tip, FIFO kills oldest-validated, tip never a victim (T-D pins), turn-2 match
    = 48704, re-prefill 11480, snapshot_lru FIFO load-bearing, pinned-4 floor + page accounting intact.
- Wave-12 TRIAGE (orchestrator): TP -> fix-TP wave (running): B3-1 gate test; B3-2a L-clear no-op assert;
  B3-2b/A3-3 bounce+retry assert; B3-3 aligned-final test (192 = 64+128); A3-2 comment; B3-4 comment tweak;
  B3-5 docstring; B3-6 invariant comment. No action: A3-4/5/6 (notes, watch hardware), B3-7 (track stamping
  value contract verified by reading), B3-4/5/6 folded as cheap lines. TASK.md 48704 corrected by orchestrator.
- Review-B3 (679e2ef0) verdict + details: recorded above in the B3 block.
- Review-B3 (679e2ef0): VERDICT NON-BLOCKING: 7 (1 Warning, 2 WeakWarning, 4 Info). Probes run in throwaway
  worktree (removed after).
  B3-1 [Warning] is_hybrid gate unpinned by tests - removing it would reintroduce the ORIGINAL overlap
    double-free for radix/naive chunked prefill; recommend extending the naive test with a 2-chunk
    continuation flow (tree stays empty, handle not rebound until final chunk).
  B3-2 [WeakWarning] L-clear no-op / retry idempotency untested (clear cache.py:414, early return :366-367);
    probe: donation without L-clear fails only T0+T-A; recommend +3 lines in T-C (re-call cache_req(chunk1)
    -> tree/pool/refcounts unchanged).
  B3-3 [WeakWarning] aligned-final-boundary finish collision untested (tests use prompts 200/300/1280, none
    x64-terminated; at prompt 192 drain donates L then finish live-donate hits mamba_exist -> keep_live=False
    -> safe ONLY via post-commit handle rebound cache.py:396-400) - recommend one test (192 = 64+128).
  B3-4 [Info] comment wording: one sentence could be misread as covering the final chunk (which has no
    continuation). Optional tweak.
  B3-5 [Info] T-E2 forces an unreachable-in-new-design state (legacy dedup-backstop) - docstring should say
    "defensive legacy state".
  B3-6 [Info] misaligned-L skip branches (cache.py:357-360/:384-389) are DEAD in production (ctor asserts
    CHUNK_SIZE % page_size == 0; chunk starts page-aligned) - acceptable defense-in-depth; add invariant
    comment so nobody converts the skip into a donation.
  B3-7 [Info] real track stamping (_build_track_metadata attention/linear.py:93-137) never exercised at this
    level (helpers simulate L/flip); value contract verified by reading (deepest interior boundary + flip,
    host-side at launch -> visible at next try_add_one).
  Q7 CORRECTION: task premise "L = chunk end" is FALSE - linear.py:120-127 stamps L = cached_len +
    ((extend_len-1)//CHUNK)*CHUNK = deepest boundary STRICTLY INSIDE the extend (8064 for 8128); end state
    = live slot -> finish-donate. Tests' L values MATCH production; L != chunk-end IS tested; donation only
    needs L <= cached_len (holds). => Production donation grid = 8064 + 8128k; turn-2 expectation: deepest
    live boundary <= 52224 = 48704 (not 48768).
  Verified-OK: determinism (det-clock autouse; 20/20 under 3 seeds; 7 isolated; no real-clock asserts);
    fails-before reproduced empirically at HEAD+wave-4 (7 failed: T-C 462, T-D 488, T-E1 518, T-E2 544);
    single-revert probes (lock removed -> 4/7 fail; L-clear removed -> 2/7); hazard structurally excluded
    (drain no longer commits intermediates; single commit point; L-clear no-op; old ordering not
    unit-reproducible - covered by design argument + T-C + B3-2 test); T-E coverage (E1 L-None, E2 dedup
    backstop exercised); helpers use real try_add_one/allocate_paged/complete_one (divergence: simulated
    stamp, no drain/overlap loop/competing reqs); commit split detailed (C1 fix(kvcache) incl. PARTIAL
    scheduler-file hunks = imports+fixture+wave-4 tests; C2 feat(server) 5 files; C3 fix(scheduler)
    prefill.py + comment + remaining scheduler tests; full-suite run after C1 and C3; zero .veai/memory);
    TRAPS: T48 superseded (watch chunk-size-dependent MISS signature), T50 pool now load-bearing mid-prefill
    (watch slot-floor ladder), T44 re-anchor BEFORE vs wave-10 baseline, T46 prefill.py logger visible;
    recommend NO logging in wave-13 beyond existing lines (turn-2 admission cached = the signal), temporary
    [fix3dbg] counter patch for evict attribution if needed (two boots, not shipped).
  Memory staleness flagged: both fix3 memories say "out-of-scope / decision pending" - now implemented
    (orchestrator will update with the wave-13 verdict).
- Review-A2: DONE (095cb6bb-364d-4bf2-8c02-5c748171ee2f). VERDICT: NON-BLOCKING: 0. Mechanism correct,
  production-effective, minimal.
  A2-1 [minor] check_integrity (hybrid:220-224) does not pin the sentinel invariant - a future attach path
    forgetting the stamp would silently become always-evicted-first. Recommend adding `n.snapshot_lru >= 0`
    to the existing per-node loop.
  A2-2 [note] split_at copy confirmed inert (split mutates self into suffix; mamba_value never moves; fresh
    prefix half filtered out of evict_mamba; future attach overwrites stamp); no other split direction exists.
    TASK.md wording says "suffix", impl copies to the fresh prefix half - both inert. Comment rationale
    overstates; optional contract-test coverage for the per-field split rule (test_tree_and_harness.py:78-100
    pins every other field rule).
  A2-3 [note] evict_full byte-identical code, sanctioned input change: the two new timestamp refreshes make
    re-validated leaves younger for KV-LRU (delayed KV eviction) - explicitly allowed by the amended spec,
    recorded to avoid regression misreads.
  Verified OK: _walk writes only timestamp (hybrid:292-304) -> stamp survives walks - the wave-2 neutralization
    is broken at its decisive link; only evict sites admission prefill.py:79/:90/:95 + commit cache.py:386/:393/
    :396/:405/:410 (finish frees, never evicts); just-validated node lock-excluded but remaining candidates now
    have distinct stamps -> stale finish/dead-branch nodes die FIFO, victim differs whenever >=2 candidates have
    distinct last-validation times (the steady state); exactly one attach site (hybrid:119 stamped :120, :230 =
    None) - unstamped live snapshots impossible, -1 candidates impossible; tuple heap sound (exact ties fall to
    __lt__ timestamp, False both ways, heapq-safe, stale pops re-guarded :193); refresh completeness (live-return
    :87-89 stamps; fall-through :92 returns root/None; stamps after walk tic); steady state hand-walked 2-turn +
    N-turn (T3 victims [B0,P,B1], exhaustion [B0,B1,B2,P], deepest survivor derives from code); just-validated
    node survives next-turn eviction UNLOCKED on fresh stamp (T1/T2, scheduler b128 unlocked full-ref); blast
    radius zero (__lt__/_walk/ref-locks/counters untouched; RadixPrefixCache/SWARadixCache never read
    snapshot_lru); oracle tic-for-tic (model.py:99/:194/:682-685/:696-705/:746), transposed-key bugs cannot hide
    (all five scenarios have snap-order vs stamp-order disagreement); tests honest (boundary-key + identity
    asserts, no value guards; T4 decoy genuinely discriminates; 7-slot pool really exhausts: usable 6 = tree 3 +
    req 3); suite independently re-run "258 passed, 1 skipped". Out-of-scope: 1 pre-existing tests/server
    import-mode failure (muse_glimmer), unrelated.

## Decision pending (user)
Fix-3 mechanism: amend brief (mamba-LRU stamp) / keep no-op + hardware-first / revert refresh. Test-contract fixes
+ T4 clock port ride the post-decision wave. Hardware validation (pre-fix baseline replay -> Part B3 boot matrix)
is prescribed under ALL options.

## STOP GATE checklist
- [ ] brief validation gates (fails-before -> passes-after)
- [ ] all findings triaged, no open TP
- [ ] test executed (baseline + new)
- [ ] TRAPS.md relevance checked
- [ ] commit - ONLY when user asks (task spec overrides orchestration default)
- [ ] artifacts recorded here
- [ ] census clean

## Post-code waves (proposed)
- Hardware validation Fix-3 (RTX 5090): pre-fix baseline replay 5+ turns (#cached-token=0 rate) vs post-fix replay;
  harness per ft-serve-test-and-e2e-gotchas (port 18081, FAILPAT watchdog, trap-on-EXIT, serialized).
- Part B3 boot matrix: ratio {2.0, 4, 8} at 0.82/400k fp8 hybrid; record boot/fail-fast + peak VRAM; then Fix-3 replay verdict.
