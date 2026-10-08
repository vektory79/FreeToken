---
name: "ft-l2-extent-dedup"
description: "L2 extent dedup for session tier: 23.81->8.17 GiB HW, commits 26e98c2..d3623ee, pin-census lessons T79"
type: project
lastUpdated: 2026-10-08T00:33
lastRecall: 2026-10-08T00:59
---

# L2 extent dedup (session tier) - executed 2026-10-07/08, HW verdict 23.81 -> 8.17 GiB (-65.7%, 2.91x)

Executed the ready brief .tasks/session-cache-tiering/briefs/l2-snapshot-dedup.md (direction (a) content/page-key-addressed extents + (d) compact lever; (c) flush-only-deepest was pre-refuted). Commits on vektory79: 26e98c2 (extents: `_plan_share`/`_plan_share_flush`, write seams on BOTH the flush pipeline and the runtime `_write_blob`, replay dual-read + refcounts derived from surviving record set, compact per-extent dedup copy + legacy split migration + remap) -> c73fe90 (flush-seam share-source pins + journal-failure failed_sids gating + corruption/cap gap tests) -> f0115a6 (runtime-seam pin + `_plan_share` caller census) -> d3623ee (kb docs). CPU gate 476 -> 493 passed/1 skipped.

## Review journey (why the census exists)
Parent review BLOCKED on a MAJOR accounting-drift bug: builder-thread eviction discarding a share source between plan and journal -> extent entry recreated with crc=None -> `_ssd_used` under-counts (negative-capable). First fix covered only the flush seam; delta review found the SAME mechanism on the runtime `_write_blob` seam. Lesson (skill trap T79, uncommitted): a pinning invariant is only complete with a caller census table (callers -> pin point -> release point) + deterministic fails-before per seam. `_plan_share` has exactly 2 callers, both pinned pre-reserve.

## HW verdict (27B arm @ f0115a6, ANCHOR comparison vs 23.81 GiB reproduced bit-identically twice)
Post-flush l2 used 8367.5 MiB; composition 4.87 GiB unique KV (2 chains) + 3.30 GiB snaps (brief's ~7 GiB assumed ONE chain). ext=78 == classifier-derived 78 (109 cross-boot), max 12 extents/record < cap 64. Reboot HIT 38592/24384/38592/30464, restore=3(l2), refused=0, integrity=0, gauges never negative/over-cap. Deepest restore 6.80 s vs 6.91-7.0 baseline (no read amplification).

## Honest caveats + residuals
Compact NEVER fired on HW (dead=0 - dedup starves the dead>25% gate): compact dedup path is CPU-test-covered only. Cap-pressure pin scenarios not exercised on HW (100 GiB cap, peak 9.28). Legacy-dir convergence deferred (no preserved pre-extent dir). Journal downgrade contract: old binaries DROP multi-extent records, single-extent boots. Residual: `_write_group` fdatasync-abort does not add group sids to failed_sids (boot converges via replay crc; cheap hardening candidate). Accepted: plan->pin window on both seams (builder thread sequential).

## Brief reconciliation (skill trap T78)
5 of 13 brief assumptions had changed by execution time (brief written 2026-09-25): P9 two-phase compact invalidated "copy at original offsets, journal verbatim" - the copy pass MUST dedupe by extent or it silently re-bloats L2. Cap 64 not 4 (chained sharing grows linearly). Reconciliation list: .tasks/session-cache-tiering/L2-DESIGN.md. Artifacts: WAVE-LOG.md, REPORT-hw-l2-dedup.md, arm_l2_dedup_hw.sh, l2_* logs/classifiers.
