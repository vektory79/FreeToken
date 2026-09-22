---
name: "ft-kb-knowledge-base"
description: "kb/ self-sufficient wiki on vektory79 (rework through a0a6064); kb-distill skill; index kb/README.md"
type: project
lastUpdated: 2026-09-22T18:26
lastRecall: 2026-09-22T18:06
---

# FreeToken knowledge base: kb/ (created 2026-09-20)

Committed on vektory79 as e2d07fc ("docs(kb): consolidate campaign artifacts
into versioned knowledge base", 607 files). The git-ignored `.tasks/` campaign
artifacts were mirrored into a versioned tree so they survive.

Structure (see kb/README.md for the full index with hooks):
- kb/methods/ - measurement methodology: orchestration digest (wave protocol +
  STOP GATE + traps), step0 nsys method, A/B protocol, quality battery,
  vram-headroom method.
- kb/harness/ - reusable scripts that exist nowhere else: measure.py (patched
  SSE parser), dense q8_0 mtile/nsplit A/B runners, MMQ v0/v2/v3 probes,
  quality battery, pcie-bw probes (pcie_bw binaries, ram_bw.py), repro
  runners, FTW conversion scripts. Harness gotchas are summarized in
  kb/README.md.
- kb/findings/ - durable code/upstream maps: gguf-glm5next hybrid + path-a
  research, upstream llama.cpp GLM5NEXT numbered snapshot (WARNING: upstream
  revision NOT recorded - kb/findings/llamacpp-glm5next-snapshot/README.md),
  mmq upstream tile-glue, fix1 radix anchors.
- kb/incidents/ - nvfp4-1m capacity (closed), crash-mr080-400k (OPEN: needs
  >=29.28 GiB free at boot), fix1 hardware report.
- kb/baselines/ - per-campaign result JSON/CSV; kb/campaigns/ - briefs/plans/
  reviews (templates for future plan packages); kb/raw/ - nsys .sqlite +
  .nsys-rep mirrors, git-ignored via kb/raw/.gitignore.

Why: .tasks/ is git-ignored (single /.tasks/ rule) and could be lost; the KB
preserves verdict-adjacent artifacts (scripts, research, raw numbers) that the
Memory Bank cannot hold.

How to apply: for any new campaign, consult kb/README.md first for methods and
harnesses instead of rebuilding measurement scripts; when a campaign in
.tasks/ finishes, mirror new durable artifacts into kb/ and commit on
vektory79. Originals stay in .tasks/ - re-copy, never edit the kb/ copy.

## REWORK COMPLETE (2026-09-21/22, phases P1-P8)

The initial mirror was reworked into a self-sufficient wiki per user-approved
triage (.tasks/kb-rework/TRIAGE-SUMMARY.md). Phase commits on vektory79:
e7e6218 (triage applied: 12 deletions, 11 merges, campaigns->cases rename),
a78e7f8 (RU wiki skeleton: top index, glossary.md 21 terms, 6 section READMEs,
4 distillates), 5d8f984 + a09d3a7 (7 topics articles: prefill-throughput,
vram-budget, radix-cache-reuse, moe-hybrid-cost-model, ftw-load-path,
gguf-upstream-map, merge-wave-procedure), 766056d (42 harness usage cards in
7 subsection READMEs), cabaa55 (14 nsys sqlite profiles distilled into
kb/reports/, kb/raw/ binaries removed - fresh clone now self-sufficient),
618b81d (self-sufficiency sweep: ~110 .tasks/.veai path refs relinked or
marked lost; fix3 TASK.md status fixed), a0a6064 (typos).

New skill .veai/skills/kb-distill/SKILL.md (review pass, uncommitted): reflexive
in-session skill that distills session insights into kb/ per kb rules; STOP GATE
of gguf-native-serving and merge-main-into-vektory79 now instruct calling it
(uncommitted). Fresh-clone verification: subagent answered 5/5 typical
questions from kb/ alone with 6 file reads, no dangling refs.

Deferred (user decision): commit of .veai/skills changes (new skill + STOP GATE
patches); optional per-cases/ summary table question->file; harness cards note
some scripts hardcode machine-local campaign paths (running kb copies as-is may
need path adjustment).

UPDATE (same day): user approved commit - 714cabd "docs(skill): add kb-distill skill and wire it into stop gates" (4 files: new kb-distill/SKILL.md + STOP GATE items in gguf-native-serving ORCHESTRATION.md/SKILL.md and merge-main-into-vektory79 SKILL.md). The "uncommitted" note above is superseded; the deferred-commit decision is resolved.

UPDATE (table lint, same day): broken tables found and fixed in kb/ (REPORT.md blank-line-split rows + separator cols; step0-profile.md unescaped q|k|v pipes) - commit e0936b8. Table linter table_lint.py bundled INTO the skill (.veai/skills/kb-distill/table_lint.py, param: kb dir arg, exit 0/1, respects \| escapes, catches blank-line breaks) and wired into kb-distill step 4 + acceptance checklist - commit 9dc22b5 (review-2 pass). Fresh kb edits must run: python3 .veai/skills/kb-distill/table_lint.py kb
