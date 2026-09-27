---
name: "plain-language-final-reports"
description: "Final chat reports must be plain-language step-by-step; dense jargon stays in kb/memory, not user summaries"
type: feedback
lastUpdated: 2026-09-27T14:23
---

# Final reports to the USER must be plain human language

Rule: every final report / summary addressed to the user is written in plain, human-understandable language - step by step, short sentences, no unexplained jargon. Dense technical phrasing (pool/GB/s/anchors/lever names) stays in kb/ and memory, never in the final chat summary.

Structure that worked (user-approved 2026-09-27):
1. One sentence on how the relevant mechanism works (context first).
2. What we did (each step in one line, with the "stopwatch"/measurement framing instead of tool names).
3. What we found (the hypothesis that broke and the real cause, in cause-effect words).
4. Result: numbers before -> after, in user-visible terms (tokens/sec, ms, test pass counts).
5. What is NOT done and which decisions belong to the user (commit / default flip / next steps), as a short numbered list.

**Why:** 2026-09-27, after the T2 campaign I delivered a dense orchestrator-style final report (pools, GB/s, gates, anchor readings); the user replied "Я ничего не понял из финального отчёта" and only accepted the plain rewrite ("Вот так гораздо лучше. Запомни что финальные отчёты всегда должны быть сформулированы аналогичным образом").

**How to apply:** end of every task/campaign/wave: write the user-facing summary LAST, after kb/memory updates, following the 5-point structure above; expand any term a non-agent reader would not know on first use; keep exact numbers but give each one a "so what" (what it means for speed/quality); never end with unexplained verdict jargon like "gates MET" without translating it.
