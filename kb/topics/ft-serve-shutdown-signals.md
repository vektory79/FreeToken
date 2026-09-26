---
title: "Graceful-остановка ft serve: SIGTERM группы процессов и session-tier flush"
date: 2026-09-25
hardware: RTX 5090 32GB; NVMe Samsung 990 EVO Plus (flush-трафик ~600 МБ/с)
branch-commits: "vektory79 @ f5e51fd (фикс graceful-остановки на SIGTERM, 2026-09-25)"
status: фикс реализован, закоммичен (vektory79 @ f5e51fd, 2026-09-25) и подтверждён железом 2026-09-25 (рука arm_sg)
tags: [ft-serve, shutdown, signals, sigterm, llama-swap, session-tier, lifecycle]
---

# Graceful-остановка ft serve: SIGTERM группы процессов и session-tier flush

**Вердикт.** Оркестраторы (llama-swap, systemd, docker) останавливают
`ft serve` SIGTERM-ом всей группы процессов, а у воркеров до 2026-09-25
graceful-путь был только на SIGINT (KeyboardInterrupt): group-SIGTERM убивал
их мгновенно, и session-tier flush при остановке молча не выполнялся. Фикс
(ветка vektory79, 2026-09-25, рабочее дерево): воркеры armed SIGTERM-обработчик,
который сводит SIGTERM к тому же KeyboardInterrupt-teardown. Железная рука
arm_sg: group-SIGTERM теперь даёт полный flush и HIT после перезагрузки;
Ctrl+C-путь не тронут. Терминальный Ctrl+C работал и раньше - SIGINT приходит
foreground-группе терминала, куда воркеры входят сами. Конфиг llama-swap
менять не нужно (см. "Руководство оператора" ниже).

## Проблема: остановка оркестратором бьёт по группе, а не по родителю

Процессная модель `ft serve`: uvicorn-родитель + scheduler-воркер (владелец
tier-стора) + tokenizer/detokenizer-воркеры (`python/freetoken/server/launch.py`).
Родитель при штатной остановке сам разливает SIGINT воркерам и ждёт
`FREETOKEN_WORKER_SIGINT_GRACE_S` (join-грейс, по умолчанию 10 с), затем
эскалирует до SIGTERM и SIGKILL; большие flush'и (~6-12 GiB) требуют
~120-180 с (см. [session-cache-tiering.md](session-cache-tiering.md)).

llama-swap v258 (проверено по исходникам тега 69cb75a) останавливает иначе:

- запускает команду напрямую, без шелла, с `SysProcAttr Setpgid` - ребёнок
  становится лидером собственной группы;
- stop = `syscall.Kill(-pid, SIGTERM)` всей группе (`runtime_unix.go:36-44`),
  SIGINT не используется никогда;
- после unloadTimeout (per-model, по умолчанию 10 с) - SIGKILL той же группе:
  это весь зазор SIGTERM -> SIGKILL;
- остановка самого демона жёстче: бюджет ~30 с, teardown моделей через
  ctx-cancel с `parentCancelGraceTimeout` = 1 с -> сразу SIGKILL группе
  (`llama-swap.go:470-527`, `process_command.go:78`);
- per-model knob cmdStop заменяет сигнал, например `/bin/kill -INT -- -${PID}`
  послал бы группе SIGINT.

До фикса у воркера не было обработчика SIGTERM: дефолтная диспозиция убивала
процесс до какого-либо cleanup, flush tier-стора не начинался. Измерено до
фикса (рука arm_sa, 2026-09-25, GLM-5.3 GGUF, флаги пользователя, дефолтный
грейс 10 с, L1 3221.9 MiB после ~169k-токенного fill): эскалация ровно на
+10.0 с, строк "session tier: flushed" / "session tier final:" в логе нет,
из 21 журнальной записи пережили 15 (6.10 GiB из 8.68 GiB контроля). Потеря
частичная, а не полная: blob пишется и fsync-ится ДО журнальной записи
(`session_tier.py:538-579`), поэтому kill между ними оставляет байты живыми,
а запись отсутствующей - сегмент молча теряется при replay; оборванная
запись отбрасывается как torn tail.

## Фикс: SIGTERM-обработчик воркеров (ветка vektory79, 2026-09-25, рабочее дерево)

- `python/freetoken/server/launch.py`, `_arm_sigterm_handler`: обработчик
  SIGTERM сначала ставит SIG_IGN на оба сигнала остановки (SIGTERM и SIGINT),
  затем бросает KeyboardInterrupt - teardown идёт через существующий
  except-блок. SIG_IGN ДО броска - one-shot защита: relayed SIGINT родителя,
  прилетающий посреди teardown, не может перебить уже идущий shutdown.
- Armed в `_run_scheduler` (все TP-ранги) и `_run_tokenize_worker` (роль
  detokenizer живёт в том же tokenize-процессе - отдельной точки входа нет).
- Закалены except-блоки teardown: scheduler (`launch.py`) и
  `python/freetoken/tokenizer/server.py` - в начале обработки
  KeyboardInterrupt оба сигнала остановки переводятся в SIG_IGN ("A relayed
  or second stop signal must not cut the teardown already in progress").
- Намеренно НЕ тронуто: обработчики родителя/uvicorn (uvicorn после
  "Application shutdown complete" перебрасывает stop-сигнал наверх - отсюда
  код выхода 143, by design), shell-mode `_detach_process_group`, логика
  демона llama-swap.
- Тесты: +3 в `tests/server/test_lifespan_shutdown.py` - юнит-тест маппинга
  SIGTERM -> KeyboardInterrupt с SIG_IGN обоих сигналов, subprocess-тест
  group-SIGTERM (graceful teardown выполняется), тест reentrancy
  SIGTERM + relayed SIGINT (ровно один teardown). Файловый CPU-гейт: 8 passed.

## Железная проверка (2026-09-25, рука arm_sg)

Конфиг руки: GLM-5.3 GGUF, флаги пользователя, env
`FREETOKEN_WORKER_SIGINT_GRACE_S=180`; arm_sg.sh поднимает сервер через
setsid (процесс = лидер группы) и останавливает `kill -TERM -pgid` -
эквивалент stop-семантики llama-swap.

- group-SIGTERM: flush полный - "session tier: flushed 22 live segments" и
  финальная строка "session tier final:"; журнал 22 записи, torn_tail=False,
  9.38 GiB.
- Перезагрузка: replayed 22 journal records; resume дал cached=95,616 токенов
  за 4.79 с.
- Регрессия Ctrl+C: group-SIGINT тоже даёт полный flush (exit 0, ~12 с) -
  прежний путь не сломан.
- Код выхода при SIGTERM-остановке 143 - by design (uvicorn re-raises stop
  signal после "Application shutdown complete"), не дефект.
- Остановка во время загрузки модели: exit 1 с traceback - совпадает с
  поведением Ctrl+C-during-load; возможное будущее закаливание, не блокер.

Оценка масштаба: типичный flush пользователя ~10-27 с при ~600 МБ/с NVMe
(Samsung 990 EVO Plus) - грейс 180 с берёт с многократным запасом.

## Руководство оператора

- При развёрнутом фиксе конфиг llama-swap менять не нужно:
  `FREETOKEN_WORKER_SIGINT_GRACE_S=180` + unloadTimeout 180 достаточны
  (flush ~10-27 с).
- Каветa демона: остановка самого llama-swap даёт моделям ~1 с
  (parentCancelGraceTimeout) - перед остановкой демона выгружайте модель
  через API.
- Для установок на старом ft (без фикса): per-model cmdStop заменяет сигнал,
  например `/bin/kill -INT -- -${PID}` - SIGINT группе запускает прежний
  KeyboardInterrupt-путь; и/или поднимайте
  `FREETOKEN_WORKER_SIGINT_GRACE_S` до 120-180 для больших L1.
- Смежное, клиентская сторона: обрывы декода через ~600 с - это дефолтный
  request-timeout клиентских SDK (openai/anthropic python), не сервер.
  Дискриминатор: ровно один WARNING "Aborting request for user" в логе
  сервера при живом сервере.

## Связанные статьи

- Механика session-tier flush, журнал и метрики:
  [session-cache-tiering.md](session-cache-tiering.md); числа железных рук
  tier-кампании - [session-cache-tiering-arms.md](../baselines/session-cache-tiering-arms.md).
