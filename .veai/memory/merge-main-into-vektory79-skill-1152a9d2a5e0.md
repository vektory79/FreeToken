---
name: "merge-main-into-vektory79-skill"
description: "Skill merge-main-into-vektory79: waves, split-round, all-remote discovery, GPU-preclass; 230 lines, commit 1a3bbe3"
type: project
lastUpdated: 2026-10-25T00:52
lastRecall: 2026-10-24T23:01
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

**Round 5 (2026-10-17) - первый RE-PLATFORM раунд, merge f0eed55 (parents b8c3794+11c1adb, amend docs+comment; kb 8a62625).** Upstream/main пере-платформил GGUF (вендоренные ggml-ядра вместо csrc/gguf, per-format slot pools, layers/quantization/*). Новый стандартный форк: РЕШЕНИЕ О ГЛУБИНЕ до merge-волны (minimal green merge + deferred ports D1-D5 vs полная адаптация в merge-коммите) - пользователь выбрал minimal green. Skip-маркировка отложенных сюит (183+ тестов) легитимна при reason=именованный порт + probe-верификации честности ревьюером; счётчики skip в логах обязаны совпадать с фактическими маркерами (engine 18; test_cache_budget 9 D3 + 1 D4). Boot-smoke: старый рецепт может упасть на НОВЫХ assert-ах конфига main (cache-budget арифметика) - классифицировать как config-вопрос, адаптировать рецепт, девиацию документировать (r3: offload-стратегия, FTW NVFP4, /ready 87s). Stash-collision протокол: каждый путь в stash сверять с worktree (equal-or-newer), отсутствующее - в архив task-папки, потом drop - не盲-pop поверх пересозданных memory-файлов. engine.py auto-union оставляет висячие веточные хвосты с undefined именами - резолвить как "main-хунки + сплайс", затем grep каждого удалённого хелпера. All-remote discovery сработал как задуман (origin/main содержится в upstream/main). D1-D5 ledger: kb/topics/round5-ggml-replatform-deferred-ports.md + память ft-merge-round5-deferred-ports.

**Round 6 (2026-10-24/25) - второй re-platform upstream/main, merge 594934d -> amend 220b79e.** SPLIT-ROUND при занятой VRAM; pending-handoff в памяти ft-merge-round6-pending-handoff. Урок 1: main-wins-резолв МОЛЧА вернул round-5-регрессию (register.py потерял фикс _gguf_tensor_types, 48915b4; поймал test_ftw_quant_config_resolves_schemes) - при main-wins на файле с веточными фиксами grep-ить фиксы по недавним коммитам ветки. Урок 2: осиротелый хелпер после rewrite (engine.py _gguf_auto_floor_gate, 0 call-sites) - grep call-sites, юнит-тесты маскируют мёртвый прод-путь. Урок 3: счётчики test-results.md обязаны быть замеренными. Порт-долг: D2 выведен main-wins (#803/#810), D4 loader contract, D5 nvfp4 (2 stale-pin vs #737). kb: cases/merge-round6-upstream-main/TASK.md.
