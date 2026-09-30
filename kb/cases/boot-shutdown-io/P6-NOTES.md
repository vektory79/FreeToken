---
title: "Волна P6: env-override FREETOKEN_FLUSH_WRITERS / FREETOKEN_FLUSH_GROUP_BYTES (резолверы флаша L2)"
date: 2026-09-30
hardware: "CPU-прогоны pytest без VRAM (локальный LLM резидентен); железная рука не выполнялась"
commits: "волна НЕ закоммичена в vektory79 @ fc3396f (python/freetoken/scheduler/session_tier.py, tests/scheduler/test_session_tier.py); коммит - решение пользователя"
status: "CPU-волна завершена: реализация + тесты 79/79 + 70/70 + fails-before + ревью + STOP GATE READY; железная A/B - потребитель рукояток (идёт вместе с железной рукой P4, VRAM)"
tags: [session-tier, flush, env-var, override, validation, warning, ab-methodology, fails-before]
---

# Волна P6: env-override рукояток флаша L2

Реализация брифа задачи P6 из [TASK.md](TASK.md): тюнинг железной A/B числа
воркеров и размера группы флаша без правок кода. Пре-реквизит железной руки
P4 ([P4-NOTES.md](P4-NOTES.md)), узкое место которой - продюсер флаша. До
волны `_FLUSH_WRITERS` (8) и `_FLUSH_GROUP_BYTES` (1 ГиБ) были захардкожены
в `session_tier.py`. Волновой план: `.tasks/boot-shutdown-io/p6-plan/PLAN.md`
(вне git, после клона считается утраченным).

## Механика: резолверы в точке использования

- Два резолвера `_flush_writers(requested)` / `_flush_group_bytes(requested)`
  читают env (`FREETOKEN_FLUSH_WRITERS` / `FREETOKEN_FLUSH_GROUP_BYTES`) в
  точке использования, а НЕ при инициализации модуля. unset -> вернуть
  `requested` (модульную константу), поэтому monkeypatch-тесты, патчащие
  константы, продолжают работать без изменений (прецедент P1:
  `FREETOKEN_BANK_POOL_WORKERS`).
- Валидация без silent clamp: мусор (`non-integer`) или ниже пола
  (writers < 1, group_bytes < 1 МиБ) -> дефолт + WARNING через `init_logger`
  (урок T46); whitespace-only значение = unset (молча, задокументировано в
  docstring резолвера).
- Применение: `writers` - ширина ThreadPoolExecutor параллельных
  pwritev-буров в `_pwrite_group` (бурст на группу; после P4-конвейера
  билдер к рукоятке отношения не имеет); `group_bytes` - группировка
  сегментов по объёму в `flush_live`.

## Резолв ОДИН раз на флаш (урок ревью)

Ревью поймало TP: два независимых резолва на флаш (в `flush_live` и ниже)
давали при мусорном env до G+1 одинаковых WARNING (по одному на группу) и
риск разъезда залогированной конфиг-строки с фактически применённой.
Фикс: оба резолва выполняются один раз в `flush_live` до группировки и
передаются вниз параметром (`_flush_groups(groups, writers)` ->
`_write_group` -> `_pwrite_group`); warning ровно один на флаш.

## Видимость для железной A/B

Одна строка "session tier: flush pipeline: writers=N group_bytes=N" на
флаш-фазу (при непустых groups), а не на группу - атрибуция железной руки
идёт по этой строке, не по значению в шелле. Урок P1: silent clamp без
видимости = ложная атрибуция A/B (метод -
[env-config-hygiene.md](../../methods/env-config-hygiene.md)).

## Тесты и fails-before

- Батарея: 79/79 `tests/scheduler/test_session_tier.py` (6 новых) +
  70/70 `test_hybrid_cache_manager`. Fails-before: первые 6 тестов падали
  вербатим до реализации; после TP-фикса 3 теста-обёртки обновлены и
  fails-before перепроверен (3 == 1 вербатим).
- autouse-фикстура тест-файла изолирует оба env от шелла
  (`monkeypatch.delenv(..., raising=False)`) - урок 1 из
  [env-config-hygiene.md](../../methods/env-config-hygiene.md).
- Детерминированные env-тесты без таймеров: применение `writers`
  перехватом ThreadPoolExecutor (assert на `max_workers` при
  `FREETOKEN_FLUSH_WRITERS=1`); применение `group_bytes` - по фактической
  группировке сегментов (2x1 МиБ payload при env=1 МиБ -> ровно 2 группы).

## Незакрытое

- Коммит волны не сделан (решение пользователя); правки лежат в рабочем
  дереве vektory79 @ fc3396f.
- Железная рука P4 - потребитель рукояток: свип writers/group_bytes
  выполняется вместе с A/B конвейера флаша (VRAM; методика руки - в брифе
  P4 [TASK.md](TASK.md)).

## Связанное

- Бриф и приёмка: [TASK.md](TASK.md), "Задача P6".
- Конвейер флаша (потребитель рукояток): [P4-NOTES.md](P4-NOTES.md).
- База волны (параллельный флаш, fdatasync, CRC32C): [P3-NOTES.md](P3-NOTES.md).
- Метод env-knob гигиены: [env-config-hygiene.md](../../methods/env-config-hygiene.md).