---
name: "ft-gguf-native-serving-skill"
description: "Self-sufficient Orchestrator skill: 7 GGUF phases, 38 traps, ORCHESTRATION.md; used-by Orchestrator; .veai/skills only"
type: project
lastUpdated: 2026-09-15T21:35
lastRecall: 2026-09-16T17:52
---

# GGUF native serving skill: reusable methodology for any model family

Single location: /media/ai/src/FreeToken/.veai/skills/gguf-native-serving/ (git-tracked, staged "A" - commits are the user's call). NO skill copies in .tasks (user directive 2026-09-15: skills live ONLY in .veai/skills; the former .tasks/gguf-native-serving-skill/ mirror was removed).

Structure (updated 2026-09-15 with the hybrid/CPU-kernel campaign, commits 11f1a80..63b9bff):
- SKILL.md: 7-phase pipeline (discovery -> config shim -> tensor translator -> tokenizer -> kernel dispatch -> expert banks -> E2E A/B offload baseline -> Phase 7 CPU compute tier). Mandatory inputs include the WORKING-REFERENCE question: ASK the user for the llama.cpp (or other) sources where the target gguf format is supported; extract vec_dot kernels (arch/<isa>/quants.c), quantize_row_* (ggml-quants.c), ggml-common.h block layouts, ISA build flags (CMakeCache.txt - microbench comparability), the CPU-resident-weights scheduling pattern (ggml_backend_sched). Static-fusion-validation rule and debug tool priority unchanged.
- TRAPS.md: T01-T38 + recipes D01-D08. CPU-tier additions: T30 env-override SIGILL (cap-down required), T31 single-dtype profile clobber, T32 concurrent-measurement corruption, T33 fetch-volume-as-floor misattribution, T34 even-split starvation, T35 projection without reference anchor, T36 stale profile without tier fingerprint, T37 CC/CXX JIT leak, T38 hardcoded capability gate; D06 standalone microbench, D07 serialized-run discipline, D08 reference-A/B decomposition.
- tasks/task-00..07: task-07-cpu-compute-tier.md is the self-contained CPU-tier brief (microbench gate -> port recipe -> weighted pools -> benchbw FULL rerun -> single-variable end-to-end A/B). task-00 discovery carries the reference-sources question and the CPU support-matrix row.

Scope: any gguf model family with HF/FTW support in python/freetoken/models/<family>/; phases parameterize from the discovery dump (geometry, format ids, block sizes); glm5next is the cited worked example (hybrid 16.67 tok/s vs offload 12.97 after the port).

How to use: run task-00 discovery (it asks the reference-sources question), then 01-07 in order with the quality loop; TRAPS.md is the pre-close checklist.

## 2026-09-15: self-sufficiency layer added
ORCHESTRATION.md (256 lines, 10 sections) now embeds the full orchestration contract specialized for gguf work: mode selection (the 7 SKILL.md phases ARE the default plan; briefs = task files), phase=wave quality loop (Review x2 mandatory for kernel/hardware/budget phases task-02/04/05/06/07), 8-item phase STOP GATE (incl. the per-phase commit gate), lossless handoff, process-hygiene protocol (serialized measurements, census, trap runners), commit discipline (conventional, Assisted-by: Veai, no push), escalation points. The skill is launched WITHOUT code-with-subagents. SKILL.md carries the Orchestration header + commit gate; all 8 briefs have uniform acceptance criteria with the commit gate. NOTE: git INDEX snapshots of the skill were stale vs worktree (staged TRAPS.md = old 29-trap edition) - re-stage before committing.

## 2026-09-15 (late): review cycle + explicit links
review-1 (checklist): 3 blockers - no YAML preamble, zero markdown links to siblings, orphaned Phase 0. Fix wave: SKILL.md now 138 lines linking ALL 10 sibling files explicitly; Phase 0 integrated into task-00 as the "Upstream alignment or waive" user decision step; counters unified to "discovery + 7"; phase-7 offload-only skip branch explicit; TRAPS.md named+linked in all briefs. OPEN ESCALATED ITEM: SKILL.md has NO YAML preamble BY DECISION - the user chose manual-only launch, but the required empty used-by list is UNCONFIRMED in the project schema (only a non-empty "Orchestrator" example exists; no schema/parser found; omitting used-by means any-agent launch) - resolves when the tooling owner confirms used-by: []; until then auto-launch is restricted by nothing. Counter scheme: "discovery + 7 phases".

## 2026-09-15 (final): preamble RESOLVED - manual-only confirmed empirically
The open item is CLOSED: a probe skill (probe-manual-only, used-by: [], agent General) was created and ACCEPTED by the IDE tooling - zero validation errors in idea.log and the user confirmed it loads cleanly in the skill list; probe deleted after the verdict. SKILL.md now carries the final preamble (name gguf-native-serving, schemaVersion v0.1, description with trigger phrases, agent: General, used-by: [] = manual-only). The skill is complete: ORCHESTRATION.md contract + 7 phases + 38 traps + 8 briefs, explicit sibling links, commit gates in every brief.

## 2026-09-15 (correction): agent type = Orchestrator (user clarification)
The user expected the skill to be of type Orchestrator and built accordingly. Preamble corrected: agent: Orchestrator, used-by: ["Orchestrator"] (mirrors code-with-subagents' preamble style; supersedes the empty-list manual-only path - the probe had already validated the parser mechanics). The skill body is Orchestrator-built by design: ORCHESTRATION.md contract + 7 phase waves with delegation to Code/Test/Review/Ask.
