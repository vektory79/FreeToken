---
name: "ft-gguf-glm5next-private-scope"
description: "GGUF glm5next Path A is private-use local work; user waived upstream issue #34 gate (2026-09-13)"
type: feedback
lastUpdated: 2026-09-13T18:47
lastRecall: 2026-09-14T20:27
---

# GGUF glm5next Path A is private-use local work; upstream gates waived

Rule (user, 2026-09-13): the glm5next GGUF serving work (plan .tasks/gguf-glm5next-path-a/PLAN.md) is purely personal - the user does NOT plan to publish it upstream. The CONTRIBUTING "features start as an issue / Phase 0 comment on umbrella issue #34" requirement is explicitly WAIVED by the user for this workstream ("сними это требование"). Do not draft or nag about issue comments, PR descriptions for upstream, or A/B-number PR gates for this work.

Why: the code is for the user's own serving on their RTX 5090 box; upstream alignment only matters for published contributions.

How to apply: keep local engineering discipline (tests, quality loops, conventional-commit messages, no pushes - the user still owns every line and every commit), but do not gate phases on upstream alignment and do not schedule upstream postings unless the user asks. If the user later says they want to publish, re-read CONTRIBUTING and reopen the Phase 0 question.
