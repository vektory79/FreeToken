---
name: "hybrid-interleave-snapshot-washout"
description: "HybridRadixCache cross-convo snapshot wash: evict_mamba FIFO; pool=4*mr+max(4,ceil(ratio*mr))+1; ratio flag; wash tests"
type: project
lastUpdated: 2026-09-22T20:42
---

# Cross-conversation snapshot wash-out (orchestrator/subagent interleave) - diagnosed 2026-09-22

**Verdict:** user's report (orchestrator builds context -> spawns subagent -> subagent runs -> orchestrator's prefix cache is GONE despite both contexts fitting the KV window) is NOT a fix-1/fix-3 regression: all 5 commits verified in the checkout (5b72aba, 9732be0, e5730e0, 7080824, a20519e). It is the two-currency eviction policy under interleaving.

## Mechanism (code-anchored)
- `match_prefix` resumes only at a node with a LIVE GDN snapshot (hybrid_radix_cache.py:67-88); no live snapshot on path -> cached_len=0 even with intact KV pages.
- Snapshot pool: `_linear_pool_num_slots` = 4*mr + max(4, ceil(ratio*mr)) + 1 (linear_state_pool.py). At user config (mr=1, default --linear-state-cache-ratio 2.0) -> 9 slots (~140.76 MiB each), only ~4-5 evictable; 5 locked to the running request (live + 2 ping-pong + locked committed + padding sink).
- Per-chunk donation (fix-3) commits ~8-9 snapshots per 58k-turn (grid 8064, 16192, ...). Each commit needing a slot fires `ensure_mamba_slots` -> `evict_mamba` - FIFO by snapshot_lru (scheduler/cache.py:140-146) with NO notion of a recently finished conversation. The other conversation's trace is strictly older -> its boundaries die first, tip included; a leaf victim's unlink cascades its tombstoned ancestors' KV away too (`_cascade_tombstone_leaves`), so the older conversation loses its KV chain entirely.
- Binding currency is GDN snapshot slots, NOT KV tokens -> "total size < context window" is irrelevant.

## Flag lever (partial)
`--linear-state-cache-ratio`: n_cache = max(4, ceil(ratio*mr)); measured fix-3: ratio 8 at 0.82/400k trips the slot-floor fail-fast (planned 1048 < floor 1059); 0.82/350k + ratio 8 boots pool 13 (~936 MiB free). Holding BOTH ~60k conversations needs ~17-18 evictable slots (ratio ~16-20, +1.7-2.2 GiB) - likely out of VRAM at 0.82. So the flag softens but does not solve the interleave.

## Code-fix candidates (separate task, NOT yet ordered by user)
pin the tip snapshot of a recently finished conversation (budgeted/timed); evict intermediate grid nodes before tips; CPU-spill snapshots (~140.76 MiB/slot copy cost). All need fails-before/passes-after + A/B on hardware.

## Baseline tests (CPU, in worktree 2026-09-22, uncommitted)
tests/kvcache/radix/test_hybrid_radix.py +2 scenarios (file 28 passed; package radix+scheduler 260 passed/1 skipped):
- `test_interleaved_conversations_wash_out_the_older_trace` - PINS CURRENT POLICY: victims leave FIFO A0 -> A1(tip) -> B0; A's resume cached=0/second=None; A's KV cascade-freed. A future protection policy must FLIP the A-resume assert to retention.
- `test_interleave_with_enough_snapshot_pool_keeps_both_traces` - no-pressure contrast: both tips retained (wash is a pool-size effect, not a match/insert defect).

## Hardware repro package (verdict PENDING as of 2026-09-22 evening)
.tasks/interleave-cache-repro/: README (2 arms), replay_interleave.py (sends Record/Case filler conversations ~58k tok alternately, sums `Prefill batch ... #cached-token:` from the server log per turn; verdict = final A turn cached==0 -> wash confirmed), run_arm.sh (census + trap + 3000s timeout). Payloads MUST carry `"model": "GLM-5.3-Flash-UD-Q3_K_XL.gguf"` (chat 422 otherwise). Arm1 = user config 0.82/400k default ratio (wash expected); Arm2 = 0.82/350k + ratio 8 (retention expected; if it still washes, flag lever insufficient -> code fix).
