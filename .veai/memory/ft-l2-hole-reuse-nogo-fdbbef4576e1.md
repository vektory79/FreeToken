---
name: "ft-l2-hole-reuse-nogo"
description: "L2 hole reuse NO-GO hit 0.067 (afa5efd); P13/compact bugfixes; prod crc forensics + restamp hardening cb55c02"
type: project
lastUpdated: 2026-11-14T16:30
lastRecall: 2026-11-14T22:29
---

# L2 exact-fit hole reuse (session tier) - EXECUTED, HW verdict NO-GO (hit 0.067), committed as-is

2026-11-14, vektory79. Brief briefs/l2-exact-fit-hole-reuse.md. Commits: afa5efd (feat: hole reuse + order-dependent tomb filters, 594+/52-, gate 543->554 passed/1 skipped) -> a79a2dc (kb docs NO-GO section) -> 86fa3bb (kb: repair 7 stale links, 0 broken) -> cb55c02 (W3 restamp hardening). Feature ACTIVE when holes exist (operator decision; no off-by-default switch). No push.

## Verdicts
- Primary arm (arm27b protocol): VACUOUS - holes_reused=0/miss=38, ZERO holes born: deterministic fill re-extends same sessions, every re-offer exact-matches existing extents (c4 appends 0 bytes), supersede=0, compact never fired. Lesson: same-shape fill never births holes; a hit-rate arm is meaningless without a hole-birthing workload.
- FU arm (ssd cap 8 GiB -> overflow evictions birth holes): 3065.7 MiB holes born, hit rate 1/(1+14)=0.067, single reuse CROSS-identity. NO-GO per brief thresholds (<0.3). Compact fired exactly once (26.4% > 25%), eof dense 8138 MiB.
- Sim (0.917 arm-shape) diverged from HW structurally: (1) same-identity re-offers are tomb-guard-blocked BY DESIGN; (2) hybrid victims free only PRIVATE spans - shared GDN snaps stay live, so HW opportunity set << sim private-span model. Sim private-span assumption invalid for hybrid snapshot-bearing models.
- Mechanics proven honest end-to-end: ledger identity eof=appended-reused PASS per cycle, boot used= chain exact, integrity=0, gauges never negative, tombstone guard + compact lifecycle work.

## Pre-existing HEAD bugs fixed en route (valuable beyond the NO-GO)
- Replay tomb filter was order-independent: a sharer's journal identity == its victim's, so same-boundary re-offer after a sharer discard silently died at boot replay (measured on HEAD).
- compact() identity-keyed keep-set resurrected tombstoned victims (two live same-identity segments post-compact).
- Hole reuse + compact = permanent shrink-abort brick (dead record re-copied from reused span, crc mismatch). All three fixed by order-dependent tomb filters at replay AND inside compact (tomb_victims dropped pre-keep-set; no-op gate requires `not tomb_at`; dropped count = len(records)-len(keep)+len(tomb_victims)).

## PROD INCIDENT 2026-11-14 + W3 restamp hardening (cb55c02)
- User's GLM-5.3 cache dir: compact aborted every idle pass ("payload crc mismatch moving record at 55119660800") - forensics: record #267's 9th extent (snaps) had a WRONG journaled per-extent crc32c from an intermediate extents-era build; record-level LOGICAL crc verified bit-exact -> bytes good, field wrong. Class B legacy poison; invisible until first-ever compact (fast-path boots skip payload-crc pass; restore never verifies ext crcs).
- Hardening rule: per-extent crc mismatch defers to the record-level logical crc arbiter (_record_logical_crc_ok, concatenated extents from SOURCE blob, journaled alg semantics absent=zlib); certified -> restamp from disk + ext_crc_restamped counter (restamp=N), uncertified -> unchanged abort/drop. Compact persists restamp via dense rewrite; replay restamp is in-memory until next compact (no journal rewrite path).
- Adjacent latent bug fixed: fresh-crc accumulator used crc32c regardless of journaled alg -> legacy zlib records always false-aborted compact; now journaled alg (absent = zlib). Fails-before tested.
- Review round: KeyError fix on transient EIO (memoize miss -> boot crash; now uncertified->drop), zlib-alg fails-before test added. Gate 557 -> 559 passed / 1 skipped.
- Prod dir self-heals at next idle compact (~11.7 GiB dead reclaimed, one warning line); manual journal patch now optional. Orphan blob.bin.new (49.89 GiB prealloc from a killed compact) is cleaned by boot recovery.
- KEY LESSON: per-extent crc can be legacy-poisoned while the record-level logical crc certifies the bytes; fast-path boots and restores never verify per-extent crcs - only slow-path replay and compact do.

## Traps
- Engine logs `flushed %d live segments` ONLY when count>0 - saturated-replay boots legitimately print none; harness must accept final line + shutdown.marker.
- Harness anti-pattern: `TL=$(fill_once)` captures a helper's ENTIRE stdout incl. diagnostics -> multiline garbage downstream. Write results to a file.
- tier_journal_classify.py / tier_state.py crashed on the first-ever P13 tombstone-bearing journal (records without blen); fixed in .tasks 2026-11-14.
- ov.admitted is LOGICAL payload under extent dedup (27.2 GiB logical vs 8.4 GiB physical on c1) - do not misread as physical bytes.
- Orphaned-watchdog tee-pipe hang reproduced again (start_watchdog must kill previous WATCH_PID; see process-hygiene memory).
- Judge l2 by log "l2 used"/ledger, never blob.bin stat size (append-only with holes).
- "moving record at <off>" in compact aborts prints the extent SOURCE offset, not the record's primary span - do not misidentify the record during forensics.

## Closed follow-ups
- kb stale links: 7 fixed, committed 86fa3bb (two were format-illustration placeholders -> de-linked; five were wrong ../ depth -> ../../../). Full-kb walk 0 broken, table_lint OK.
