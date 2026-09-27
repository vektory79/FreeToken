---
name: "kb-first-research-workflow"
description: "Research workflow: kb/ first, questions one-by-one; research+instrumentation in-session; only optimizations split out"
type: feedback
lastUpdated: 2026-09-26T22:42
lastRecall: 2026-09-26T23:50
---

# Research workflow preference (2026-09-26)

Правила работы над исследовательскими запросами в FreeToken:

1. База знаний kb/ - источник номер один: "При исследовании используй базу знаний
   kb/README.md. Предпочитай базу знаний прямому исследованию кода." Прямое чтение
   кода - только для сверки якорей после kb.
2. Уточняющие вопросы - по одному, каждый следующий после ответа на предыдущий;
   исчерпывающий ответ - когда данных достаточно.
3. Исследование (поиск решений, диагностика, измерения) выполняется в ТЕКУЩЕЙ сессии;
   в отдельные сессии выносятся только реальные задачи оптимизации, если найден
   способ оптимизировать (формулировка пользователя при запуске decode-исследования).

**Why:** kb/ строился именно как самодостаточная база кампаний (см.
ft-kb-knowledge-base); пользователь экономит контекст/кредиты облачной LLM
(см. lean-subagent-context) и хочет заземлённые, измеренные выводы
(см. measured-advice-over-paper).

**How to apply:** любой запрос "исследуй/изучи/можно ли улучшить": сначала
kb/README.md -> topics/findings/baselines/reports, цитировать kb-ссылки, сверить
якоря в коде точечно; финал - структурированный отчёт с ранжированными рычагами и
предложением отдельных брифов под реализацию.

4. Measurement instrumentation is part of research, not implementation (user, 2026-09-26: "T1 давай тоже тут проведём. Это часть исследования, а не реализация решения"). Harness/driver/analysis scripts under .tasks/<campaign>/ are fair game to build and run inside the research session; only behavioral fixes / kernel changes go to separate optimization sessions. The same directive extended the session scope to the offload strategy ("проверить поведение системы при --moe-strategy offload и можно ли получить улучшения").
