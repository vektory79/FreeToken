---
name: "fix3-hybrid-cache-loss-donation"
description: "Fix-3 campaign LANDED+VALIDATED: donation fixes user's periodic full loss (5/5 -> 0/5 misses, 4.3x turns 2-6); 5 commits"
type: project
lastUpdated: 2026-09-21T19:06
lastRecall: 2026-09-23T01:22
---

# Fix-3: hybrid radix full cache loss on mid-history rewrite -> per-chunk donation LANDED

Campaign 2026-09 on vektory79 (GLM-5.3-Flash UD-Q3_K_XL hybrid, mr=1, 0.82/400k fp8, RTX 5090; waves 1-14). Wave log: .tasks/fix3-snapshot-lru-refresh/{TASK.md,WAVES.md}. COMMITTED: 9732be0 fix(kvcache) snapshot lru; e5730e0 feat(server) --linear-state-cache-ratio; 7080824 fix(scheduler) per-chunk donation; be57ee8 docs invariant comments. Suite "265 passed, 1 skipped". Skill files self-updated UNCOMMITTED (TRAPS T59-T63 + T45/T48/T50 in place).

## Client trigger (byte-verified)
Veai history is NOT append-only: injected blocks ride the current user message and are stripped on later replay. r1 msg16 = 5027 chars, r2 msg16 = 210 chars, NOT substrings; first differing message index 16; divergence depth ~84-90% of ~58-60k-token prompts (52224, page-aligned).

## Server mechanism (FINAL; supersedes LRU-churn / capacity-bound / gauge / retention theories)
scheduler.py:322-338: intermediate ChunkedReq chunks NEVER called cache_req (drain-point donation double-frees) -> only the FINAL chunk committed (one attach per turn at align-down(prompt end)); finish-donate silently skipped by the alignment gate (align_down != cached_len; the finish live-donate gate is cache.py:364-379). A mid-history divergence split that single node and the match climbed from the fresh prefix node - the old snapshot beyond the divergence -> cached=0 -> FULL re-prefill of every rewrite turn (raw logs airtight: turn-2 = 7x"8128/0"+"3288/0" = 60184, every line cached=0; emitter = one line per prefill batch, first chunk = admission match). Zero evictions; pool size and LRU irrelevant at <=2 snapshots. The #mamba-slot gauge structurally EXCLUDES evictable snapshots (used = live + 2pp + LOCKED; scheduler.py:464-473 + cache.py:114-117).

## Drain-point double-free hazard (why the skip existed; donation must be at CONTINUATION CREATION)
overlap_loop schedules the next chunk (try_add_one inherits cache_handle/pp) BEFORE the prior drain: a drain-point donation would (a) donate pp[frozen] zero-copy while the stale tuple still lists the slot -> _free_req_slots frees a tree-owned slot, and the pp alternation clobbers it; (b) stale admission handle -> the next commit's dedup free re-frees adopted pages; (c) per-commit unlock underflows the admission node refcount. Only overlap-safe point = CONTINUATION CREATION (prefill.py try_add_one), strictly after the prior forward completes and before state copy - the landed design: hybrid-gated (cache_manager.is_hybrid), calls the existing chunk-commit branch verbatim, L consumed exactly once, idempotent under budget bounce, final chunk + finish branches untouched, non-hybrid keeps the skip.

## L grid (review-A3 correction)
L = cached_len + ((extend_len-1)//CHUNK)*CHUNK = deepest x64 boundary STRICTLY INSIDE the chunk (8064 for an 8128 chunk; kda h[i] = state at chunk START; the extend-end state lives only in the live slot). Measured turn-1 grid {8064,16192,24320,32448,40576,48704,56832,58304(final drain)}.

## Hardware-validated (wave-13, 12-request replay @ ratio 2.0)
turn-2 #cached-token = 48704 EXACTLY, re-prefill 60184 -> 11480; full misses 2/12 -> 1/12 (cold turn only); turns 3+ full self-hits unchanged; 5 evictions STRICTLY FIFO-by-snapshot_lru (8064->16192->24320->32448->40576, just-validated tip never a victim) - the snapshot_lru ordering is load-bearing and hardware-verified; medians unchanged (r2 prefill metric now vacuous). Ratio flag: fundable infra (+563 MiB at ratio 8 measured; 0.82/400k fail-fast planned 1048 < floor 1059; 0.82/350k boots pool 13, ~936 MiB free) but moot at <=2 snapshots; with donation the pool is load-bearing mid-prefill (~8x commits per long prompt).

## Watch items
Replacement pp alloc can raise RuntimeError (unreachable under current sizing); hybrid+SWA swa-cap floor <=1 page over-admit if such a model ever lands; slot-floor fail-fast ladder if desktop VRAM moves; per-batch log semantics (first chunk = admission line; verify against raw lines before A/B interpretation).

## Durable gotchas
LinearStatePool padding sink (size pools +1); at mr=1 ratio 4 == 2.0 (max(4,.) floor); bare stdlib getLogger is boot-log-invisible (use init_logger); temporary [fix3dbg] attribution pattern (sha256 ledger, remove + bit-for-bit verify + suite re-run).

## STATUS UPDATE (2026-09-21): skill files COMMITTED as a20519e "docs(skill): add fix3 campaign knowledge (T59-T63, expected classes)" atop be57ee8 (4 files, 77+/5-; zero .veai/memory staged). The skill self-update is no longer user-gated. Follow-up waves 15-17 running: growing-history A/B replay (pre via 3-file stash vs post) + ratio-8 probe with donation active.

## FOLLOW-UPS COMPLETE (waves 15-17, user-ordered points 2-4)
Skill docs committed a20519e (5th commit). Growing-history A/B (6 turns, real Veai pattern, real model replies): PRE-fix full-miss rate 5/5 = 100% (every turn 8128/0, full re-prefill - the user's periodic loss MEASURED on the real pattern); POST-fix 0/5, t2 = 8128/48704 (exact acceptance key, divergence 52224 cross-checked), t3-t6 1743-2030 new tok @ 56768 cached (divergence climbs ~90/turn, trunk boundary immortal); turns 2-6 wall 426 s -> 99 s (4.3x). 10 FIFO evictions, matched nodes never victims. Ratio-8 probe (0.82/350k, pool 13): admissions byte-identical to ratio-2.0 (turn-2 48704 divergence-capped), evictions 1 vs 5 (fund retains ~4 more), NO fail-fast (margin 27), RuntimeError 0, peak 31314/free 835 MiB - ratio 8 = pure retention headroom, no reuse delta on this workload. Restoration bit-for-bit (reversal via git apply -R of the exact 3-file diff - stash cannot reverse committed fixes); suite 265 passed, 1 skipped.
