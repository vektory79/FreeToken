---
name: "merge-main-into-vektory79-skill"
description: "Skill merge-main-into-vektory79: waves, split-round, all-remote discovery, GPU-preclass; 230 lines, commit 1a3bbe3"
type: project
lastUpdated: 2026-10-05T22:14
lastRecall: 2026-10-05T22:08
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

**Round 4 (2026-10-05) - первый мердж против upstream/main, merge aa0e71c (parents 41d3581+581fdca).** Ключевой урок: в репо ДВА remote (origin и upstream); origin/main стоял на 6410c97 и discovery вернул NOTHING_TO_MERGE, пока пользователь не подсказал, что изменения в upstream/main (581fdca, 13 коммитов). Правило: discovery обязан опрашивать ВСЕ remote (git remote -v + fetch + merge-base по каждому) ДО вердикта nothing-to-merge. Резолвы: ftw.py - gguf_types-валидация ветки перенесена verbatim в load_ftw_banks (ftw.py:684-714) при принятом upstream-хелпере _split_bank_entries; дублей нет (#601 fused NVFP4/MXFP4 banks сосуществует с gguf-цепочкой). Отложено в cloud-окно (VRAM занята): boot-smoke (задет engine/config.py #573), GPU-порции тестов, follow-up F1 (load_ftw_banks_to_device без gguf_types-валидации, ftw.py:851-915, сегодня недостижим - resident-ветка требует method!=None, expert_banks.py:611/:617-618). Детали: .tasks/merge-main-round4-work/, kb/topics/merge-wave-procedure.md (round 4 + секция "опрашивай ВСЕ remote").

**Актуализация 2026-10-05:** скилл после round-4 правок - 230 строк (было 198), закоммичен 1a3bbe3 "chore(skills): fold merge round 4 lessons into merge-main-into-vektory79" (родитель d1ce146 docs(kb) round-4 дистиллят). Новые режимы в скилле: all-remote discovery (nothing-to-merge только при пустой дельте по ВСЕМ remote), SPLIT_ROUND (занятая VRAM: CPU-only merge по одобрению + pending-handoff + amend при падении boot-smoke в cloud-окне), предклассификация GPU-only тестов, tee boot-smoke stdout, ревью A/B одним агентом при занятой VRAM, triage-класс deferred-followup (пример F1 -> 9ac8830), сверка git diff --cached при финализации, kb-shell-read и edit_file-heredoc заметки.
