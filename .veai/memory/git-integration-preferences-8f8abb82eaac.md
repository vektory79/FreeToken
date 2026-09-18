---
name: "git-integration-preferences"
description: "Branch-integration prefs: PR merges stay merge commits; duplicate functionality -> main wins; merge over rebase"
type: feedback
lastUpdated: 2026-09-16T19:29
lastRecall: 2026-09-18T23:05
---

# Предпочтения пользователя по интеграции веток (git)

Rule (user directive, сентябрь 2026, задача обновления ветки vektory79 на последний main):
1. Мерджи других веток (PR/MR, напр. merge ветки PR #408, коммит ba0f20f) должны ОСТАВАТЬСЯ merge-коммитами - не линеаризовать их простым rebase.
2. Если в main появилась функциональность, дублирующая функциональность ветки пользователя, - использовать функциональность ИЗ MAIN; дубль ветки отбрасывается (main - источник истины по пересекающимся фичам).
3. При рисках rebase пользователь выбирает `git merge main` в свою ветку вместо rebase.

**Why:** пользователь начал с запроса "rebase моей ветки на main", но после того, как узнал, что rebase сплющит его merge-коммиты и что история main когда-то переписывалась (см. vektory79-main-rewrite-gotcha), явно переключился на "смерджить main в ветку". Правило дубликатов продиктовано отдельно и подчёркнуто как "ВАЖНО!".

**How to apply:** любые операции интеграции main в пользовательские ветки (merge / rebase / cherry-pick / разрешение конфликтов):
- сохранять merge-топологию пользовательской ветки;
- при конфликте-дубликате брать сторону main и отбрасывать дубль ветки;
- каждый случай применения правила дубликатов (и неочевидные решения) фиксировать в логе адаптаций с точными путями и обоснованием - пользователь обязан смочь объяснить каждую строку (см. user rules из CONTRIBUTING: не пушить, не коммитить без запроса).

## Workflow codified as a skill (2026-09-15)
Регламент закреплён в .veai/skills/merge-main-into-vektory79/SKILL.md (коммит 740b6b2; см. memory merge-main-into-vektory79-skill): discovery-резолв origin/main vs origin/master, backup-ветка, ff-only main, merge --no-commit + тесты до финализации, правило дубликатов main-wins, GGUF boot-smoke по зонам риска, review/triage с amend. Merge-раунд 2 (cac247a -> 5d93a30, 2026-09-15) подтвердил правила на практике: 0 конфликтов, 0 дубликатов, semantic check weight.py PASS, 346 тестов зелёные.
