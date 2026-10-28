---
name: "ft-cachedrop-l2-restore-zero"
description: "CacheDrop case: L2 restores + landed commits da7b84e/9281151/fd22003, HW control pending"
type: project
lastUpdated: 2026-10-28T01:56
lastRecall: 2026-10-28T01:47
---

# CacheDrop case: session-tier L2 restores; self-evict + restamp + divergence log landed

2026-10-26/27 diagnosis + HW verification. Artifacts: .tasks/CahceDrop/ (run_cachehit_arm.sh, run_twosession_arm.sh, run_phase3_rebootb.sh, cache_drop_client.py, cache_drop_twosession.py, ACCEPTANCE.md, arm*_boot*.log, rec*_*.json).

## Root cause (HW-CONFIRMED)
Client (Veai) rewrites history: COMPRESSION 226393->121599 tokens; keys = rolling blake2b chain keys over 64-token pages (cache.py _chain_keys); hybrid probe is boundary-exact. Any rewrite above a boundary invalidates it; flushed L2 tips never match again -> restore=0(l2). HW arms: Arm1 H reboot+unchanged -> restore=1(l2), 5.2s vs 33.6s cold (mechanism works); Arm1 C summary-replacement -> cached=0 (reproduced). L2 at 100% cap evicts oldest-insert; boot compact 100 s.

## Arm2 capacity finding
2x250k alternating sessions on 400k pool: probe HITS boundary but budget gate refuses (reason=resources) -> ~200s full re-prefill per switch. CAPACITY ceiling, not a tier bug; operator lever = --kv-reserve-tokens.

## LANDED 2026-10-27 (vektory79, CPU waves)
- da7b84e self-evict restore: try_restore closes page deficit via _evict (same seam as _allocate); hybrid mamba-slot recheck post-evict; refusal only if still short; walked-prefix protected inc_lock/handle-lock in try/finally (review blocker); skip self-evict when prior_refusal present; wasted-self-evict logged on read/late-tail/resources gates. Red-before confirmed, 97+354 passed.
- Victim sort by (last_validation, seg_id) ALREADY EXISTED (d77d15ed; _demote_one_lru:822, evict_store:1520, _evict_oldest_l2:1529). Real hole: replay restored seg.last_validation = rec["ts"] (monotonic_ns not comparable across boots). 9281151 restamps replayed segments to 0 in shared replay record loop (slow + marker fast-path; "ts" written, not read back). Red-before on BOTH replay paths.
- fd22003 divergence log from match_req after try_restore (probe/restore deltas known only then; bookkeeping preserved); format per brief. freetoken loggers do not propagate - tests use tier_log_capture fixture, not capsys.
- GOTCHA: L2-only offer journals BEFORE seg.last_validation stamp (ts=0, no leak); leak reproduces only via demote/flush_live - use flush_live in replay tests.

## Harness traps (fixed in scripts)
Watchdog subshells held bash|tee pipe -> background never signaled; start_watchdog kills previous WATCH_PID. Restore counters lag one batch line -> tiny confirm turn, WARN not FAIL. --no-capture must exist on EVERY client variant.

## PENDING: HW control for all three commits
run_twosession_arm.sh (ALT turns cached>0, wall << 200s) + run_cachehit_arm.sh (PASS criteria same + grep: evictions take never-validated records first). Requires free VRAM.

## HW CONTROL 2026-10-28: both briefs PASS on HW (cloud-LLM window)
- run_twosession_arm.sh (da7b84e): 4 ALT switches restored from tier, 6.8-13.6s vs 196-214s baseline (~15-30x); cached ~100%, restore ok=1..4 (mostly l1 offers, l2 hit on first restore after flush); "restore refused"/"reason=resources" = 0; wasted-self-evict = 0; same-session grows 99.9-100%; ARM-2 PASS. Artifacts: rec2_*.json (rerun), arm2_boot1/2.log, baseline-20261027/rec2_alt*.json, runner_selfevict_20261028.log.
- run_cachehit_arm.sh (9281151+fd22003): H PASS (cached 99.9%, restore l2=1/2, wall 5.1s vs 32.7s cold), C PASS (cached=0, probe miss), restamp did not break replay fast-path restore. No overflow evictions in run (evict=0) -> victim-order check stays covered by CPU test only. ARM PASS.
- NEW GOTCHA (divergence-log blind spot, fd22003): the line fires only for same-chain0 rewrites (cached_len < st[0]). A rewrite INSIDE the first 64-token page changes chain0 -> setdefault creates a fresh session with st[0]=0 -> 0<0 false -> no line (confirmed in arm C: 0 divergence lines). Veai production compression replaces early history -> the largest real loss class is INVISIBLE to this grep; it catches mid/tail rewrites only. Possible follow-up (design decision, not started): cross-chain0 rewrite detection heuristic.
