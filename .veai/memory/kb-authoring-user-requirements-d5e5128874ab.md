---
name: "kb-authoring-user-requirements"
description: "User requirements for kb/: git self-sufficiency, RU indexes, human-readable, markdown links, wiki granularity"
type: feedback
lastUpdated: 2026-09-21T20:49
lastRecall: 2026-09-25T01:35
---

# User requirements for the kb/ knowledge base (feedback, 2026-09-20)

Lead rules for anything authored into kb/ (FreeToken knowledge base):

1. Full git self-sufficiency: a fresh clone must contain everything an agent
   session needs. No references to paths outside the repo (.tasks/, .veai/tmp);
   an artifact is either distilled into the KB or explicitly marked lost.
2. Layered indexes: top kb/README.md + a README per section, so a session can
   tell what it needs WITHOUT reading whole directories (context economy).
3. Wiki granularity: one topic = one atomic article (<=~250 lines), details in
   linked files; primary navigation by topic, not by artifact type.
4. All cross-references in md files as markdown links [text](path), never bare
   file/dir names.
5. Human-readable texts: no bare commit hashes and no unexplained
   abbreviations; expand them ("fix-1" -> what it fixes); a glossary covers the
   rest. New indexes/explanatory docs in Russian; inherited artifacts stay in
   English.
6. Deletions only after per-file value triage with user-visible verdicts
   (keep/distill/merge/remove); never delete by blanket criteria - user fears
   losing valuable artifacts.
7. Established test harnesses are HIGH value (built over long time, actively
   reused by sessions); each needs a usage card: what it is for, how to run it,
   gotchas. Genuinely one-off wrapper scripts may be dropped.

Why: the first KB iteration was critiqued for empty kb/raw after clone, flat
index, dangling .tasks references and opaque jargon.
How to apply: any kb/ authoring, restructuring, or the kb-distill skill.
