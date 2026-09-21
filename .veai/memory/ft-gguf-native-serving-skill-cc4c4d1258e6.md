---
name: "ft-gguf-native-serving-skill"
description: "gguf-native-serving skill ledger: TRAPS T01-T63; fix3 self-update committed a20519e; campaign validated"
type: project
lastUpdated: 2026-09-21T19:06
lastRecall: 2026-09-21T17:44
---

# GGUF native serving skill: reusable methodology for any model family

Single location: /media/ai/src/FreeToken/.veai/skills/gguf-native-serving/ (git-tracked; NO copies in .tasks - user directive 2026-09-15). Skill-file commits are user-gated; see Current state.

## Structure (as of 2026-09-21, fix3 session)
- SKILL.md (~153 lines): YAML preamble (never touched by self-updates), 7-phase pipeline, sibling links, counter line "63 ловушки" (:146).
- TRAPS.md: T01-T63 + recipes D01-D10; NO pre-close checklist section (the earlier "checklist" claim was stale - the file ends at D10). T57-T58 predate the fix3 session (provenance per on-disk sections, not re-verified); T59-T63 = fix3 hybrid-radix campaign 2026-09-21: T59 the #mamba-slot gauge EXCLUDES evictable snapshots (real span scheduler.py:464-473; never infer pool occupancy from it); T60 the finish live-donate gate cache.py:364-379 silently skips on unaligned ends (correct-but-skips; NOT the chunk-commit gate at cache.py:383-394); T61 ChunkedReq skip -> donation only at continuation creation (prefill.py try_add_one; drain-point double-frees); T62 L grid = deepest interior x64 boundary, k*chunk+8064; T63 one log line per prefill batch (first chunk = admission match). In-place supersessions: T45 (git-stash A/B + sha256 ledger), T48 (L consumed at continuation creation, commit 7080824), T50 (pool load-bearing mid-prefill, ~8x commits/prompt).
- ORCHESTRATION.md (11 sections): mode selection, phase=wave quality loop, STOP GATE, lossless handoff, process-hygiene protocol (incl. the [fix3dbg] temporary-attribution contract: init_logger, sha256 ledger, remove + bit-for-bit verify + suite re-run, never ship), sec.8 commit discipline (explicit-path staging, ZERO .veai/memory staged, measured-results body, no push), sec.10 counter sync (T01-T63), sec.11 session-end self-update.
- tasks/task-00..07: task-00 carries the working-reference question (ask for llama.cpp sources; extract vec_dot kernels, quantize_row_*, block layouts, ISA flags, sched pattern) + upstream-alignment-or-waive user decision; task-06 carries the expected-result classes incl. the hybrid-radix per-chunk donation A/B replay class (turn-2 key #cached-token 48704; re-prefill 60184 -> 11480; 2->1 full misses; FIFO stalest-first; T44 re-anchor; the eviction-gap bullet CLOSED via 9732be0).

## Campaign knowledge
Merged into TRAPS.md - do not duplicate here: hybrid/CPU-kernel T30-T38 + D06-D08; MMQ prefill T39-T47 + D09/D10; v3a m-tile T51-T52; fix-1 radix T48-T50; dense-q80 T53-T56; fix3 T59-T63.

## sec.11 Session-end self-update (user directive 2026-09-18)
At the END of every session that produced reusable knowledge (root causes with commit ids, measured expectation classes, new traps, hygiene refinements, stale-fact corrections) the Orchestrator MUST run a self-update pass: delegate integration to a Code subagent with the session-fact payload; update-in-place (TRAPS new-T-or-extension, task-06 expected classes, sec.7/8 bullets); counters synced across TRAPS H1 / SKILL.md / sec.10; blocking_status review pass + cheap polish. The pass NEVER commits/pushes itself; YAML preamble untouched; no README.

## Current state (2026-09-21, fix3 session)
- Campaign code COMMITTED on vektory79: ... -> 9732be0 fix(kvcache) snapshot lru -> e5730e0 feat(server) --linear-state-cache-ratio -> 7080824 fix(scheduler) per-chunk donation -> be57ee8 docs invariant comments (HEAD). Prior skill commits: e277d4b (dense-q80 knowledge) atop 8e2e4c7.
- Skill self-update from the fix3 session: 4 files modified-UNSTAGED (TRAPS.md T59-T63 + T45/T48/T50; ORCHESTRATION sec.7 attribution bullet + sec.10 counter; SKILL.md counter; task-06 donation A/B class + eviction-gap CLOSED) - user-gated. Self-update review: BLOCKING 1 (T60 wrong-gate anchor: cited cache.py:383-394, the chunk-commit gate, instead of the finish gate cache.py:364-379) + WeakWarning (T59 span 464-473) -> re-anchor fix-wave launched; counters/ASCII/YAML/no-commits confirmed OK.
- Reviews history: .tasks/skill-review-gguf-native-serving/review-1..4.md.

## UPDATE (2026-09-21): the fix3 skill self-update (4 files, 77+/5-) is COMMITTED as a20519e "docs(skill): add fix3 campaign knowledge (T59-T63, expected classes)" atop be57ee8 - the "modified-UNSTAGED/user-gated" status above is superseded. TRAPS T60 re-anchored to cache.py:364-379 and T59 to scheduler.py:464-473 (+padding-sink gloss) by the review fix-wave BEFORE the commit; T61 anchored 322-338 (full skip region superset of the :323-332 comment block).

## STATUS FINAL (2026-09-21): skill self-update COMMITTED a20519e (5th fix3 commit; 4 files 77+/5-). Growing-history A/B + ratio-8 probe complete (see fix3-hybrid-cache-loss-donation): pre-fix 5/5 full misses on the real pattern, post-fix 0/5; ratio 8 = retention headroom only.
