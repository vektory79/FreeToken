---
name: "ft-stream-stall-llama-swap"
description: "Stream-stall: fix COMMITTED fc5fd4a + case/brief S1 f454715; config matrix = operator decision"
type: project
lastUpdated: 2026-10-02T20:39
lastRecall: 2026-10-02T20:28
---

# Stream-stall через llama-swap: root causes + fix (2026-10-02, инцидент)

Симптом: генерация завершается, статистика в llama-swap есть, но ответ не закрывается
(нет access-строки "200 OK" у uvicorn), клиент ждёт секунды/минуты/час.

## Root cause №1 (подтверждён логом окна 17:39-17:45 и ARM-C)
Конфиг llama-swap v259 (/media/ai/soft/llama-swap/config.yaml): GLM-5.3-Flash-gguf НЕ
входит в matrix.vars (q36b35|q38b27|gg4b4|q38fn) -> GLM "runs alone" -> ЛЮБОЙ запрос к
другой модели мгновенно эвиктит GLM mid-stream. При этом эвикция mid-stream НЕ прерывает
стрим: uvicorn "Waiting for connections to close", воркеры (SIGTERM группе) умирают
раньше, "Backend worker is gone" +11 c, llama-swap forced exit по 30 с таймауту, клиент
получает закрытие через 47 c. Совпадает со строками "Shutting down" из лога инцидента.
Рекомендация оператору: добавить GLM в matrix (или держать сессии в одной модели) -
правка конфига оператора, в код не входит.

## Root cause №2 (закрыт фиксом в api_server.py)
per-chunk request.is_disconnected() в starlette 1.6 - МЁРТВЫЙ код: receive() читается в
pre-cancelled anyio-scope, http.disconnect не достаётся. Прямой обрыв клиента при этом
работал (starlette listen_for_disconnect, spec "2.3"), но: backpressure-парковка
генератора в flow.drain (send) выводит отмену из фрейма генератора -> сирота-декод
занимает max-running-requests=1, следующий запрос ждёт минуты. Фикс: watcher-таск
_watch_disconnect блокируется на request.receive(), при http.disconnect -> abort_user
(AbortMsg) + cancel таски, abort-once флаг, гасится в finally (await + suppress).
Протокол фактический: prod uvicorn 0.53.0 + h11 (НЕ httptools), spec "2.3"; dev 0.52.4.

## Точная сигнатура инцидента
Застрявший запрос НЕ имеет access-строки "200 OK" (пишется по завершении ответа);
"Scheduler is idle" уже напечатан (другой процесс); /v1/requests-строка есть (пишется в
finally generate_events ДО закрытия ответа) -> "статистика есть, ответ не закрыт".

## Тесты/артефакты
- tests/server/test_stream_disconnect.py: 3 теста (mid-stream abort <=1 c - fails-before
  RED "DID NOT RAISE CancelledError"; no-pending-tasks после потребления; поздний
  disconnect не ломает завершённый стрим - regression-guard).
- [stall]-телеметрия в openai_api.py (GenDone->finish/usage/[DONE] в мс + WARN >1 c,
  гейт t_gendone>0) - оставлена постоянно.
- Регресс 883 passed (1 известный baseline muse fail). Тест-инвок требует
  --extra dev (venv-split).
- Артефакты: .tasks/stream-stall/ (start_ft.sh, arm_b/arm_c.sh, ls-config.yaml :18821,
  fake_model.py, логи), .tasks/boot-shutdown-io/stream-stall/plan.md.

## Открыто (не реализовано)
- NEXT WAVE: abort in-flight users при shutdown/backend-death (FrontendManager.shutdown/
  _on_failure будит wait_for_ack) - закроет 47-секундное окно при эвикции. Брифф
  зафиксирован в .tasks/boot-shutdown-io/stream-stall/plan.md (вне git).
- Конфиг llama-swap: matrix - решение оператора.

## Фикс закоммичен (2026-10-02)
- **fc5fd4a** fix(server): abort streaming requests on client disconnect via receive
  watcher (api_server.py +52, openai_api.py +38, 3 теста +170; 883 passed).
- **f454715** docs(kb): новый кейс kb/cases/llama-swap-stream-stall/TASK.md (инцидент +
  бриф S1 "abort in-flight users при смерти бэкенда/шатдауне", OPEN, старт по явному
  запросу) + статусы P7/P8 в kb скорректированы.
- Статус "uncommitted" в этой записи устарел. Некоммиченным остаётся только микро-хвост
  статусов (P8-NOTES.md frontmatter) из P9-кампании.
