---
name: "vektory79-main-rewrite-gotcha"
description: "vektory79: main was rewritten (merge 3e5bbdd parent af71ba4 orphaned); rebase --rebase-merges replays stale commits"
type: project
lastUpdated: 2026-09-15T22:44
lastRecall: 2026-09-20T16:45
---

# vektory79: переписанная main - ловушка для rebase, правила git-операций

Факты (git discovery, сентябрь 2026, при подготовке обновления ветки на последний main):
- Рабочая ветка пользователя: `vektory79` (upstream НЕ настроен, push - только сам пользователь). Несёт ~50 коммитов вне main, включая 8 merge-коммитов: merge ветки PR #408 = ba0f20f, плюс мердж 3e5bbdd, второй родитель которого (af71ba4) недостижим из текущего main.
- История main когда-то ПЕРЕПИСЫВАЛАСЬ (и sync-инструмент пользователя уже переписывал саму vektory79 - см. pr408-kv-nvfp4-1m-port).
- Следствие: `git rebase --rebase-merges main` на vektory79 ОПАСЕН - stale-коммиты старой main (достижимые из HEAD, но не из нового main) попадают в диапазон rebase и перепроигрываются обратно в ветку, возвращая код, от которого main отказалась.

**Why:** перепроигрыш осиротевшей lineage удваивает конфликты и реанимирует отброшенный код; merge такой проблемы не имеет вообще - история не переписывается.

**How to apply:** интегрировать main в vektory79 - ТОЛЬКО через `git merge main` (решение пользователя, сентябрь 2026). Перед любыми history-операциями на ветке:
- backup-ветка от текущего HEAD (например `backup/vektory79-pre-merge-<sha>`);
- файлы `.veai/memory/*.md` ЗАКОММИЧЕНЫ в ветке и обычно грязные (агенты обновляют память между сессиями) - pathspec-stash только этих файлов перед операцией (`git stash push -m ... -- .veai/memory/<files>`) и `git stash pop` после, чтобы не плодить лишние коммиты;
- свежесть main: `git fetch origin` + ff-only (`git fetch origin main:main`); не-fast-forward - стоп и отчёт;
- merge-коммит финализировать только после успешных целевых тестов на разрешённом дереве.
Детальный discovery тех пор (26 конфликт-кандидатов, топ: engine/engine.py, server/args.py, glm5_next-кластер): .tasks/rebase-onto-main-work/discovery-report.md (артефакт, может устареть).
