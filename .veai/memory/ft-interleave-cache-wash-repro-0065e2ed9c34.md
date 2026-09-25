---
name: "ft-interleave-cache-wash-repro"
description: "Interleave wash HW-confirmed: ratio 2.0 = 11/11 full misses; ratio 8 + 350k = full hits ~20x"
type: project
lastUpdated: 2026-09-22T23:53
lastRecall: 2026-09-24T14:15
---

# Interleave cache wash (orchestrator/subagent): hardware-confirmed + flag-level fix

2026-09-22, RTX 5090, GLM-5.3-Flash UD-Q3_K_XL FTW, vektory79 (fix-1 5b72aba + fix-3 9732be0/e5730e0/7080824 all present). Package: `.tasks/interleave-cache-repro/` (run_arm.sh, replay_interleave.py, arm_arm{1,2}.{json,log}). Driver: two disjoint ~57k (A) / ~44k (B) token conversations, 5 alternating turns each + final A; verdict = summed `#cached-token` per turn from server log.

## Mechanism (root cause, code-verified)
GDN snapshot pool = `4*mr + max(4, ceil(ratio*mr)) + 1` slots (linear_state_pool.py `_linear_pool_num_slots`); default ratio 2.0 at mr=1 -> 9 slots, only ~4-5 evictable. `ensure_mamba_slots -> evict_mamba` is FIFO by snapshot_lru with no notion of "recently finished conversation": an interleaved conversation's per-chunk donation commits evict the peer's trace first (tip included); a leaf eviction cascades tombstoned ancestors -> peer's KV gone too. KV pool size is NOT the binding currency. Match needs a LIVE snapshot on path (hybrid_radix_cache.py match_prefix) -> cached=0 -> full re-prefill.

## Measured
- Arm 1 (user config, ratio 2.0 default, 0.82/400k): 11/11 turns full miss (cross-conversation AND same-conversation repeat after an intervening turn; wave-13 back-to-back HIT tip survives only when NO intervening commits). A turn ~75 s, B ~57 s prefill.
- Arm 2 (`--linear-state-cache-ratio 8 --kv-reserve-tokens 350000`, rest identical): turns 3-11 ALL full hits — A 57344/57345, B 44096/44145 (49-tok page tail), 3.6-3.9 s/turn = ~20x; no boot fail-fast at 350k; pool 13, gauge shows 0/12 (padding excluded).
- CPU pins: tests/kvcache/radix/test_hybrid_radix.py 28 passed (26 old + 2 new interleave tests; the wash test pins CURRENT policy baseline, contrast test proves pool-size effect).

## Recommendation (flag-level, no code change needed at this scale)
User's orchestrator rig: switch to `--linear-state-cache-ratio 8 --kv-reserve-tokens 350000`. Enough margin because only the peer's TIP must survive an interleaving turn (pool >= peer per-turn commits + tip). If agent contexts grow much beyond ~60k or >2 concurrent conversations, a tip-protection eviction policy (evict intermediate grid nodes first / pin recent tips) is the follow-up code task.

## Harness gotchas (this session)
- `/v1/models` answers BEFORE the engine finishes GGUF load -> chat returns 503; readiness probe must be a tiny chat POST until 200 (doubles as triton autotune warmup).
- `pkill -9 -f '<pattern>'` with the pattern text inside the invoking command kills the invoking shell itself (self-match; cost: one silently empty background run). Bracket the pattern.
- Stale zombie from old campaign (`while true` VRAM sampler, 26h old) made census FATAL; cleaned. Census pattern `'[f]t serve --port'` avoids both self-match and text-only wrappers.

## Addenda (2026-09-22, same session)
- Commit: tests landed as d82274c on vektory79 (test file only; .tasks package uncommitted).
- Measured from arm1 boot log: KV pool 400000 tokens = 2.38 GiB -> ~6.39 KiB/token (fp8 hybrid); a 57k session's KV path ~= 366 MiB. Snapshot slot 140.76 MiB. So a demote/restore unit (tip + KV path) ~= 0.5 GiB per 60k session.
- Gauge denominators: arm1 #mamba-slot 4/8 (pool 9 minus padding), arm2 0/12 (pool 13) - gauge excludes padding sink.
- Boot (FTW fast path): 9.06 GiB weights ~3 s + 125 GB expert banks 29 s (peaks 8-12 GB/s NVMe); free-after-init 4.86 GiB at 0.82/400k; CUDA-graph capture 51 s.

## USER CONSTRAINTS UPDATE (2026-09-22, evening) - supersedes the flag recommendation above
- Ratio 8 REJECTED by user: too expensive on VRAM (not accepted into production). VRAM pools stay tight (default ratio 2.0).
- Sessions are often >>64k: many >150k, some 200k. Only small subagent tasks fit 64k.
- Requirement: on server shutdown EVERYTHING demotes to L2 (disk) so nothing is lost at next boot.
- More than 2 agents; several sessions may run in parallel (mr may grow).
- Budgets user will donate: up to 30 GB RAM (pinned-cache candidate) and up to 100 GB NVMe @ ~5 GB/s.
- Direction chosen: tiered session cache (L0 VRAM -> L1 RAM 30GB -> L2 SSD 100GB) as the PRIMARY fix, kb/topics/session-cache-tiering.md is the design article. Grid retention policy open question (R=1 tip-only vs R=3 vs adaptive below-divergence).
