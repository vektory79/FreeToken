---
name: "ru-style-no-calques"
description: "User style rule: no Russified calques like робастный in Russian texts; codified in AGENTS.md ## Language (e596bba)"
type: feedback
lastUpdated: 2026-10-24T20:39
lastRecall: 2026-10-25T18:43
---

# Language style: no Russified calques (робастный etc.)

Rule: in user-facing texts (chat replies, kb articles, reports) do NOT use half-translated calques like "робастный", "детерминировать", "метрика-конфиг" style hybrids. Write either normal Russian ("устойчивый", "надёжный", "определять") or an established English term left as-is ("robust", "validated").

**Why:** 2026-10-24, user reviewing kb/topics/gguf-serving-perf-matrix.md: "Что за слово `робастный`? Не надо применять такие слова. Нужно или на нормальном русском или на английском для устоявшихся терминов."

**How to apply:** when writing or editing any Russian kb/report text, replace calques with a normal Russian synonym or keep the English term verbatim; sweep existing docs when the user flags one. Established technical loanwords with no clean Russian equivalent are fine (e.g. "слоты", "транзиенты" in perf slang) - the target is awkward hybrid adjectives, not all borrowing.

**Codified in repo (2026-10-24):** user asked to mark the rule in AGENTS.md; new `## Language` section added (commit e596bba) - fresh sessions inherit it from AGENTS.md automatically, this memory stays as the why + sweep ledger. First sweep same day: kb/topics/gguf-serving-perf-matrix.md + kb/topics/gguf-vram-headroom-post-round5.md, 6 "робастн*" -> "устойчивый".
