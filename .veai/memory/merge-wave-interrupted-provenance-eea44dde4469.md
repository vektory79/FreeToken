---
name: "merge-wave-interrupted-provenance"
description: "Continue interrupted merge wave safely: git merge-tree bit-for-bit check + provenance rules + tee runner stdout"
type: project
lastUpdated: 2026-09-20T19:45
lastRecall: 2026-09-22T12:00
---

# Continue an interrupted merge wave instead of redoing it (round 3, 2026-09-20)

Fact: a subagent call can fail on result delivery AFTER the agent already did the work (round 3: "Access denied by security policy" arrived after the merge wave had started the merge and boot-smoke). The next wave then found the round's OWN pending state: `git merge main --no-commit` in progress, the mandated pathspec stash present, smoke artifacts on disk.

Protocol (validated in round 3):
- Do NOT abort+redo blindly. Verify the pending state bit-for-bit: staged index tree == fresh `git merge-tree --write-tree HEAD <target>` (tree OID + per-file blob OIDs), worktree == index, conflict-marker grep empty. If identical, CONTINUE and document provenance in adaptation-log.
- The mandated pathspec stash (e.g. "round3-veai-memory") from the interrupted run is NOT a foreign entry; reconcile at post (stash-only files -> `git restore --source=stash@{0}`, then drop).
- Boot-smoke executed by the interrupted incarnation against a blob-identical tree is acceptable evidence if merged-file blob OIDs == committed tree and files unmodified since (reviewer independently re-verified via rev-parse/ls-tree).
- Why: redo wastes a ~2 min GPU boot and duplicates work; bit-for-bit equality makes continuation exactly equivalent to a fresh run.
- How to apply: any merge round of the skill. Also: tee the boot-smoke runner stdout into an artifact so the "/ready 200 after Ns" verdict line is directly evidenced - uvicorn access log never prints /ready lines, so without the runner stdout only indirect proof exists (minor finding B1, round 3; fix for future rounds).
