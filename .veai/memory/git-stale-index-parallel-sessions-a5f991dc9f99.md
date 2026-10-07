---
name: "git-stale-index-parallel-sessions"
description: "Git index holds stale staged versions; parallel sessions commit .veai/memory between waves; re-stage before commit"
type: project
lastUpdated: 2026-09-27T22:09
lastRecall: 2026-10-07T17:26
---

# Stale git index + параллельные пользовательские коммиты

Правило: перед ЛЮБЫМ коммитом в этом репо - предний `git status --porcelain`, повторный `git add` ЯВНЫМИ путями и сверка staged-версии с рабочим деревом (`git diff --cached`).

**Why:** повторившийся паттерн (2 случая): (1) 2026-09-15, gguf-native-serving - staged TRAPS.md был старой редакцией (29 ловушек) против свежего worktree; (2) 2026-09-15, коммит скила merge-main-into-vektory79 - SKILL.md уже лежал в индексе старой 185-строчной версией (статус AM), тогда как в дереве была финальная 194-строчная после review-правок; старую версию чуть не закоммитили. Плюс между волнами появляются ЧУЖИЕ коммиты только с `.veai/memory/*` от пользователя/параллельных сессий (8c0f1e7 "Memory", 8d8fd52 "Memoty" - опечатка в сообщении, parent 5d93a30).

**How to apply:** в коммит-волне: preflight git status; staging только явными путями; после add - убедиться, что staged = актуальная версия файла, а не ранняя; не удивляться чужим .veai/memory-коммитам в истории - принимать как есть, не откатывать; ожидаемые счётчики файлов сверять с фактом до коммита.

**(3)-(4) 2026-09-28, два случая в decode-кампаниях:** в индексе снова лежали чужие staged-записи от параллельных сессий - в первый раз два kb-файла из другого коммит-вейва, во второй - файлы (lever3-microfix-ab.md, decode-chain-microfusion.md), которые вообще ещё нигде не были закоммичены (staged кем-то параллельно). Протокол сработал в обоих случаях: git reset -> явный add путями -> сверка `git diff --cached --stat` с ожидаемым набором -> коммит.
