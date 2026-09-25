---
name: "lean-subagent-context"
description: "Cloud LLM: do not resume big-context subagents (costly); prefer fresh narrow agents; resume only for tiny deltas"
type: feedback
lastUpdated: 2026-09-25T01:23
lastRecall: 2026-09-25T01:24
---

# Минимальный контекст для субагентов

Правило: не перезапускать подзадачи с большим контекстом. Запускать субагентов со СВЕЖИМ узким контекстом, содержащим только необходимую информацию.

**Why:** пользователь (2026-09-15, сессия gguf-glm5next-hybrid): большие контексты в субагентах дорого стоят в деньгах; перезапуск/relaunch агента с полным контекстом умножает расход токенов.

**How to apply:** при делегировании call_*_agent:
1. В `context` класть только конкретные пути, якоря (file:line) и решения - детали отдавать ТОЧНЫМ ПУТЁМ к артефакту (например, .tasks/.../research/*.md), а не пересказом текста.
2. Предпочитать свежий узкий агент вместо relaunch тяжёлого прогона; resume_id использовать только когда субагент уже держит в контексте нужные файлы и дельта задачи мала.
3. Не дублировать в промпте содержимое файлов, которые субагент обязан прочитать сам по пути; дублировать только то, без чего он не найдёт правильную точку входа.

## UPDATE 2026-09-23 (user decision, cloud-LLM cost)
When working on the CLOUD LLM (local rig LLM switched off), do NOT resume/continue subagents with a large accumulated context - resume replays the whole prior conversation and is expensive on provider credits. How to apply: on the cloud model, prefer a FRESH narrow subagent with only the needed context (exact file paths, anchors, decisions) over resuming a long-running one, even when the resumed agent "already knows" the files; resume only when the delta is tiny. On the LOCAL LLM (no per-token cost concern) the resume-vs-fresh tradeoff is softer, but keep the lean-context discipline anyway.
