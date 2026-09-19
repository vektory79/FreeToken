---
name: "merge-main-into-vektory79-skill"
description: "Skill merge-main-into-vektory79 (.veai/skills): merge origin/main into vektory79; duplicates->main; boot-smoke"
type: reference
lastUpdated: 2026-09-16T19:29
lastRecall: 2026-09-19T18:14
---

# Skill merge-main-into-vektory79 (оркестрирующий)

Location: `.veai/skills/merge-main-into-vektory79/SKILL.md` (194 строки, git-tracked; закоммичен 2026-09-15 как 740b6b2 "chore(skills): add merge-main-into-vektory79 skill").

Purpose: полный регламент обновления ветки vektory79 мерджем origin/main - кодификация git-integration-preferences + опыт merge-раундов 1-2 (2026-09-15, f786d7c и 5d93a30).

Что кодифицирует:
- Волны: discovery (READ-ONLY + fetch; резолв origin/main vs origin/master - пользователь говорит "master", фактический ref origin/main) -> merge-волна (backup-ветка, ff-only `git fetch origin main:main`, merge --no-commit, тесты ДО финализации, GGUF boot-smoke) -> review (явные skip-критерии) -> triage/fix (amend в merge-коммит) -> STOP GATE -> Memory Bank update.
- Правило дубликатов main-wins; РОВНО ОДИН merge-коммит; `.veai/*` вне коммитов; без push/rebase; тесты до финализации.
- Process-hygiene (census + фильтр собственной census-обёртки, FAILPAT-watchdog, сериализованность, SIGTERM->SIGKILL); pytest-ловушки (`python -m pytest` import-режим, `uv run --extra dev`).
- Boot-smoke триггер по зонам риска (frozen-shim урок: hf_config setattr в engine/config.py + mm/config.py, models/gguf/*, планировщик MoE cache_budget, models/weight.py, glm5_next); boot-числа envelope-зависимы (инварианты вместо точных чисел: слот-гейт молчит, слоты >= 336, больше свободно = безопаснее); backend-death recovery (kill -9 дерева).
- Субагенты: call_code_agent (discovery fast / merge strong / triage-fix), call_review_agent (2 прохода A correctness + B edge), call_test_agent - у каждой роли задан формат результата; lean-context; лимит 3 quality-цикла.

Preamble: agent: Orchestrator, used-by ОТСУТСТВУЕТ (режим "все агенты" - выбор пользователя 2026-09-15; отличается от gguf-native-serving, где used-by: ["Orchestrator"]).

Review cycle: review-1 fail (1 блокер: у call_test_agent не был задан формат результата; + W/T/I доработки) -> 9 правок (делегированы call_code_agent - оркестраторский write_file не перезаписывает существующие файлы .veai, см. veai-skill-overwrite-trap) -> review-2 pass. Артефакты проверок: .tasks/skill-review-merge-main-into-vektory79/ (временные).
