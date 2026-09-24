---
name: "ft-tier-phase2-swave"
description: "Tier phase-2 S-wave committed 2348da5: mr>1 leak fix, QSA guard, log-line metrics, watermark compaction; briefs pending"
type: project
lastUpdated: 2026-09-24T13:01
---

# Session-tier phase-2 S-wave (metrics, scheduled compaction, safety guards)

2026-09-23, committed 2348da5 "feat(scheduler): tier metrics, scheduled compaction and restore-safety guards" on vektory79 (6 files, +375/-45; gate 303 passed/1 skipped, wide 452/1). Chain: d77d15e (phase-1 feat) -> 4b6aacb (HW hardening) -> f90b081 (kb docs) -> 2348da5.

## What landed
- mr>1 leak fix: _tier_pending_restore single slot -> dict keyed by id(handle) (value holds the handle so ids stay unique while present); lock()/abandon_restore pop only their own entry. Fails-before verified via stash probe.
- QSA activation guard: _tier_refusal_reason(pool) evaluated BEFORE SessionTierStore construction (cache.py) - a qwen4_exp/QSAKVCache model with tier flags now refuses loudly (warning via init_logger, tier fully off, no dir/fd/mlock churn). One removable function; remove when the QSA codec brief lands (see brief qsa-codec).
- Metrics: Counter in SessionTierStore (offers_ok/rej, probe_hit/miss, note_match, restore_l1/l2 = ATTEMPTS incl. later-discarded, evictions, demotions, discards) + snapshot()/stats_line(). Export = lazy tier_fn fragment appended to status.py batch log lines (computed only in the throttled branch) + shutdown final line. /v1/stats wire-up deliberately deferred (would thread through 4 layers of UserReply); harnesses grepping '#cached-token' unaffected. Off-mode: no store, empty fragment, bit-identical.
- Scheduled compaction: _dead_bytes ledger (padded spans in _discard, reset by compact, recomputed at boot replay); trigger = dead>25% of blob AND blob>256 MiB, checked on successful offer, end of evict_store, and Scheduler.run_when_idle (fires only on blocking receive - a fully busy server relies on the offer/evict_store check points); compact() holds the global RLock, restore blob reads are refs-guarded + read original offsets. Also fixed a dormant phase-1 bug: compact() never re-synced _blob_eof to the truncated file (runtime compaction would point the next journal offset above its data).
- Compaction record survival is IDENTITY-keyed {(path_key, off, n) in live}, not last-wins-per-path: after a SIGKILL crash a discarded segment is resurrected and two live segments can share one path_key - last-wins would hole the older LIVE segment's blob region -> zero-page restores (reviewer reproduced; regression test test_compact_keeps_both_resurrected_records_after_crash_replay).

## Open design points (accepted as-is at STOP GATE)
Watermark/floor constants (25% / 256 MiB); log-line metrics instead of /v1/stats; run_when_idle blocking-receive caveat.

## Pending follow-ups
Heavy briefs await execution: .tasks/session-cache-tiering/briefs/{async-prefetch,qsa-codec,generic-non-hybrid}.md; Qwen3.8-Flash-Next boot sanity (needs VRAM window); memory ft-session-cache-tiering-phase1 holds phase-1 history + operator decisions (L1 = 10 GiB, no ulimit raise).
