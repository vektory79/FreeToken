---
name: "git-integration-preferences"
description: "Branch-integration prefs: PR merges stay merge commits; duplicate functionality -> main wins; merge over rebase"
type: feedback
lastUpdated: 2026-09-20T19:45
lastRecall: 2026-09-20T20:29
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

## Round 3 (2026-09-20)
Merge commit 5001504e21d1 (parents 1445fc1 + 6410c97, main commits #518 WeightLoadError + #521 tvm-ffi arch pin): 0 conflicts, 0 duplicates, 383/0/4 tests green, boot-smoke PASS (gguf-786k recipe, boot ~118 s, slot-gate silent, KV 788992). Boot-smoke provenance: executed by an interrupted incarnation against a blob-identical merge tree, judged sufficient by review (see memory merge-wave-interrupted-provenance). Review A/B: no blocker/major on the merge; only artifact fixes (log wording, line-count typo). Backup backup/vektory79-pre-merge3-1445fc1; push left to the user.
