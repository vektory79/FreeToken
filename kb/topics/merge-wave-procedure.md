---
title: "Регламент интеграции origin/main в приватную ветку: волны, прерванная волна, boot-smoke"
date: 2026-09-22
hardware: RTX 5090 32GB (boot-smoke)
branch-commits: "vektory79; слияния: раунд 1 - merge-коммит 380e9eb (родители 6e673ab + afd99cb, 8 конфликтных файлов / 16 ханков), раунд 2 - afd99cb -> cac247a (ноль конфликтов), раунд 3 - merge 5001504 (родители 1445fc1 + 6410c97, ноль текстовых конфликтов, продолжен после прерывания); rebase-подход отклонён в пользу merge"
status: validated
tags: [merge, main, waves, discovery, adaptation-log, test-results, boot-smoke, provenance, merge-tree, stash, duplicates]
---

# Интеграция origin/main в приватную ветку: регламент волн

**Вердикт (читайте это первым).** Слияние выполняется волнами: discovery ->
adaptation -> test -> boot-smoke, с артефактами discovery-report /
adaptation-log / test-results. Базовые правила: дубликат функциональности ->
main выигрывает; merge вместо rebase (rebase-эксперимент отклонён). Прерванную
волну НЕ переделывать вслепую: сначала побитовая верификация pending-состояния
через `git merge-tree --write-tree` - если дерево совпало, волна ПРОДОЛЖАЕТСЯ,
а история фиксируется в adaptation-log.

## Волны и артефакты

- **Discovery**: карта будущих конфликтов (кандидатные файлы, дивергенции
  сторон), delta ветки против main; отчёт - discovery-report.md раунда.
- **Adaptation**: разрешение конфликтов по правилам ниже + СЕМАНТИЧЕСКАЯ
  верификация каждого авто-юниона (ort может слить текст без конфликтов и
  получить бессмыслицу); журнал решений - adaptation-log.md.
- **Test**: сериализованные прогоны pytest под hard-timeout с цензурой
  процессов до/после каждой волны; сводка - test-results.md (пример: раунд 3,
  383 passed / 0 failed / 4 skipped).
- **Boot-smoke**: реальный бут ft serve на слияном дереве, обязателен когда
  merge трогает путь загрузки весов; ранер и вердиктная строка см. ниже.
- Примеры всех трёх артефактов: раунд 2 -
  [discovery-report.md](../cases/merge-main-work/merge-main-round2-work/discovery-report.md),
  [adaptation-log.md](../cases/merge-main-work/merge-main-round2-work/adaptation-log.md),
  [test-results.md](../cases/merge-main-work/merge-main-round2-work/test-results.md);
  раунд 3 - [discovery-report.md](../cases/merge-main-work/merge-main-round3-work/discovery-report.md),
  [adaptation-log.md](../cases/merge-main-work/merge-main-round3-work/adaptation-log.md),
  [test-results.md](../cases/merge-main-work/merge-main-round3-work/test-results.md).

## Правила разрешения конфликтов

- **Дубликат функциональности -> main выигрывает.** Если main пере-реализовал
  то же, что и ветка, веточная копия отпадает (примеры раунда 1: сайдкар
  hf_quant_config.json, устаревшая строка docs/models.md).
- **Merge вместо rebase**: rebase-подход был опробован и отклонён (приватная
  ветка vektory79 хранит свои merge-коммиты; rebase перепроигрывает их).
  Лог раунда 1, оформленного как merge, -
  [adaptation-log.md](../cases/merge-main-work/rebase-onto-main-work/adaptation-log.md),
  триаж ревью - [findings-triage.md](../cases/merge-main-work/rebase-onto-main-work/findings-triage.md).
- **Ноль текстовых конфликтов != ноль семантических решений.** Каждый
  авто-юнион верифицируется в слитом дереве: обе стороны на месте, guard-ы
  объединены (пример раунда 1: авто-гибрид guard объединил три независимых
  условия в AND; пример раунда 3: main-обёртка `_weight_load_context`
  вокруг общего вызова загрузки банков при веточном floor-gate вне обёртки).
- Обёртки/исключения main'а расширяют веточный код ТОЛЬКО если у main есть
  собственный аналог; менять классификацию сбоев веточных путей под main без
  парного кода main'а нельзя (принятый гэп раунда 3 в
  [adaptation-log.md](../cases/merge-main-work/merge-main-round3-work/adaptation-log.md)).

## Прерванная волна: продолжение вместо redo

Субагент может упасть НА доставке результата ПОСЛЕ того, как работа сделана
(раунд 3: ошибка доступа пришла после старта merge и boot-smoke). Следующая
волна обязана проверить pending-состояние побитово, а не abort+redo:

1. Проверить, что staged-дерево идентично свежему результату
   `git merge-tree --write-tree HEAD <target>`: OID дерева + per-file blob
   OID каждого затронутого файла; worktree == index; grep по конфликтным
   маркерам пуст.
2. Если идентично - ПРОДОЛЖИТЬ: финализировать merge, задокументировать
   провенанс в adaptation-log. Побитовое равенство делает продолжение
   эквивалентным свежему прогону; redo лишь тратит ~2 минуты GPU-бута и
   дублирует работу.
3. Pathspec-stash прерванного прогона (стянутый dirty-набор рабочих файлов) -
   НЕ чужой stash: reconcile на посте - файлы, живущие только в stash,
   восстановить (`git restore --source=stash@{0}`), затем drop; файлы,
   обновлённые параллельными сессиями позже стэша, считать новее стэша.
4. Boot-smoke, выполненный прерванной инкарнацией, - приемлемое доказательство,
   если blob OID слитых файлов совпадают с коммит-деревом и файлы не менялись
   после (независимо перепроверяется через rev-parse/ls-tree).

## Boot-smoke: ранер и ловушка улик

- Ранер - [boot-smoke-r3.sh](../harness/repro/boot-smoke-r3.sh): trap-on-EXIT
  уборка, FAILPAT-дозорный каждые 3 с (AssertionError / OutOfMemoryError /
  backend worker / CUDA error / ValueError), опрос /ready до 250x3 с,
  контрольный POST /v1/chat/completions с полем model.
- **Ловушка улик: uvicorn access log НИКОГДА не печатает строки /ready** -
  без сохранённого stdout ранера доказательство готовности остаётся только
  косвенным (ранер вышел rc=1 без 200 + архивный POST 200). Обязательно tee
  stdout ранера в артефакт, чтобы строка вердикта «READY after Ns» была
  прямой уликой (урок раунда 3, находка B1).
- Вердиктная строка и критерии PASS: slot-gate молчит, суммарные слоты >=
  рабочего набора, время бута в полосе, чистое завершение по SIGTERM, чистая
  цензура процессов; точные VRAM-числа не сравниваются (зависят от стартовой
  свободной VRAM). Пример разбора -
  [test-results.md](../cases/merge-main-work/merge-main-round3-work/test-results.md),
  секция boot-smoke.

## Инварианты финализации

- Ровно один коммит над родителями: `git rev-list --count HEAD --not <parents>`
  == 1; diff первого родителя - ровно дельта main (для merge-коммита `git show
  --name-only` печатает меньше файлов, чем реально вошло - файлы, идентичные
  второму родителю, скрыты; проверять через ls-tree).
- Тесты ДО финализации merge-коммита (если git сам закоммитил - откатить и
  переделать как `--no-commit`; прецедент раунда 2).
- Бэкап-ветка pre-merge HEAD, в merge-диффе нет ничего кроме дельты main
  (service-грязь параллельных сессий в коммит не попадает), нет MERGE_HEAD,
  цензура процессов чистая, без push и без rebase.
- Ревью-проходы по слитому дереву (A корректность / B граничные случаи) с
  триажем TP/FP: раунд 1 - [findings-triage.md](../cases/merge-main-work/rebase-onto-main-work/findings-triage.md);
  примеры юнион-решений 16 ханков - в логе раунда 1 (выше).

Смежное: волновой протокол кампаний и STOP GATE -
[orchestration-notes.md](../methods/orchestration-notes.md); почему длинные
рецепты могут fail-fast на буте - [vram-budget.md](vram-budget.md).