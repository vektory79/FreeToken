---
branch-commits: "vektory79 - фикс root cause 2 (is_disconnected мёртв, watcher на receive): fc5fd4a; бриф S1 (abort in-flight при смерти бэкенда/шатдауна) - отдельный PR, кода ещё нет"
status: "root cause 2 ЗАКРЫТ (фикс закоммичен fc5fd4a, железная приёмка ARM-B 2026-10-02); root cause 1 OPEN - правка конфига llama-swap за оператором; бриф S1 OPEN"
tags: [server, streaming, llama-swap, disconnect, starlette, orphan-decode, brief]
---

# Застревание стрим-ответа через llama-swap: два root cause, фикс disconnect-watcher и бриф S1

## Инцидент (2026-10-02)

Симптом: генерация завершается, статистика запроса в llama-swap есть, но ответ
клиента не закрывается: у uvicorn нет access-строки "200 OK" (пишется только по
завершении ответа), клиент ждёт секунды/минуты/час. Точная сигнатура: строка
"Scheduler is idle" уже напечатана, строка /v1/requests есть (пишется в finally
generate_events до закрытия ответа) - "статистика есть, ответ не закрыт".

## Root cause 1: эвикция mid-stream из конфига llama-swap (OPEN - оператор)

Конфиг llama-swap v259: GLM-5.3-Flash-gguf НЕ входит в matrix.vars
(q36b35|q38b27|gg4b4|q38fn) -> GLM "runs alone" -> любой запрос к другой модели
мгновенно эвиктит GLM mid-stream. Сама эвикция НЕ прерывает стрим: uvicorn
стоит в "Waiting for connections to close", воркеры (SIGTERM группе) умирают
раньше, "Backend worker is gone" через ~11 c, llama-swap форс-выход по
30-с таймауту, клиент получает закрытие стрима только через 47 c. Совпадает со
строками "Shutting down" из лога инцидента. Решение - правка конфига оператора
(добавить GLM в matrix или держать сессии в одной модели), в код ft не входит.
Окно 47 c закрывает бриф S1 ниже.

## Root cause 2: мёртвый is_disconnected в starlette 1.6 (ЗАКРЫТ фикс-волной)

per-chunk request.is_disconnected() в starlette 1.6 - мёртвый код: receive()
читается в pre-cancelled anyio-scope, сообщение http.disconnect не достаётся.
Прямой обрыв клиента при этом работал (starlette listen_for_disconnect, scope
spec "2.3" на uvicorn 0.52.4 dev / 0.53.0 prod + h11), но backpressure-парковка
генератора в flow.drain (send) выводит отмену из фрейма генератора ->
сирота-декод занимает max-running-requests=1, следующий запрос ждёт минуты.

Фикс: watcher-таск _watch_disconnect блокируется на request.receive(), при
http.disconnect -> abort_user (AbortMsg) + cancel таски-потребителя извне
фрейма генератора (парковка в send()-backpressure больше не спасает сироту),
abort-once флаг, watcher гасится в finally (cancel + await); мёртвый polling
удалён. В openai_api оставлена постоянная [stall]-телеметрия финализации
стрима (GenDone->finish/usage/[DONE] в мс + WARN >1 c).

- Тесты: tests/server/test_stream_disconnect.py, 3 теста; fails-before RED
  "DID NOT RAISE CancelledError"; регресс 883 passed (1 известный baseline
  muse fail), инвок тестов с --extra dev (venv-split).
- Железная приёмка ARM-B через llama-swap :18821: "Aborting request" <= 1 c,
  TTFB следующего запроса 2.4 мс (без очереди), [stall] finalize 0/0/0 мс.
- Закоммичено этим же PR (fc5fd4a).

## Задача S1: abort in-flight users при смерти бэкенда/шатдауне (бриф, OPEN)

**Цель.** Закрыть 47-секундное окно при эвикции/backend death: стрим клиента
должен закрываться сразу после смерти воркера, а не по таймауту оркестратора.

**База.** Волна stream-stall, ARM-C (реальная эвикция mid-stream через unload
API llama-swap): SIGTERM-эвикция не абортит in-flight стрим, закрытие через
47 c. Лог волны - boot-shutdown-io/stream-stall/plan.md, харнесс -
.tasks/stream-stall/ (оба вне git: в клоне отсутствуют).

**Якоря (перепроверены 2026-10-02 @ e45c592).**

- python/freetoken/server/api_server.py: FrontendManager.shutdown (:523),
  _on_failure (:1145), listen() (:344-358);
- python/freetoken/server/generation.py: wait_for_ack (:696, :802) - циклы
  потребления ack в стрим-генераторах.

**Спецификация.** При смерти воркера/шатдауне будить wait_for_ack всех
in-flight пользователей -> GenerationError/abort -> SSE error-чанк + закрытие
стрима. НЕ ломать graceful shutdown tier-кампании
(FREETOKEN_WORKER_SIGINT_GRACE_S=180, флаш тира должен успеть до смерти
процессов - см. связанное ниже).

**Приёмка.** Юнит-тесты (shutdown/_on_failure будят wait_for_ack) + железная
повтор ARM-C: эвикция mid-stream -> закрытие стрима клиента < 5 c (сейчас
47 c). Отдельный PR.

## Связанное

- Graceful-остановка, SIGTERM-грейс и флаш тира:
  [../../topics/ft-serve-shutdown-signals.md](../../topics/ft-serve-shutdown-signals.md).
- Контекст железного окна boot-shutdown-io кампании:
  [../boot-shutdown-io/P7-NOTES.md](../boot-shutdown-io/P7-NOTES.md),
  [../boot-shutdown-io/P8-NOTES.md](../boot-shutdown-io/P8-NOTES.md).
- План/лог stream-stall волны и харнесс (вне git):
  boot-shutdown-io/stream-stall/plan.md, .tasks/stream-stall/.
