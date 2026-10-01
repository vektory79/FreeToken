---
title: "Волна P8: record-based компакт-триггер и честный dead-учёт session tier"
date: 2026-10-01
hardware: "CPU-only прогоны pytest (без VRAM); HW-приёмка брифом не требуется - железный эффект (исчезновение но-оп парсингов на ок-офферах) верифицируется счётчиками вызовов теста (a)"
commits: "коммитов НЕТ (база волны vektory79 @ fc9a04a, DIRECT_EXECUTION, ревью после волны): python/freetoken/scheduler/session_tier.py (+44/-8), cache.py (+1/-1 docstring), tests/scheduler/test_session_tier.py (+116, 82 -> 87 тестов)"
status: "реализация + ревью STOP GATE READY (0 blocker/major; MINOR holes<0 исправлен display-clamp'ом с тестом, NIT - фраза в докстринге); батарея 157 passed (87 session_tier + 70 hybrid), 15.31 s; fails-before дословный (git stash A/B x3 + sha256-леджер)"
tags: [session-tier, compact, watermark, ledger, dead-records, stats-line, replay, crash-safety]
---

# Волна P8: record-based компакт-триггер и честный dead-учёт

Реализация брифа задачи P8 из [TASK.md](TASK.md) ("# Задача P8"); дизайн до кода
(PLAN.md) и отчёт волны лежали в git-игнорируемом каталоге
`.tasks/boot-shutdown-io/p8-plan/` и после клона считаются утраченными (суть -
ниже). База наблюдений - железная рука P5 ([P5-NOTES.md](P5-NOTES.md),
"Железная рука", флаги оператора из TASK.md): dead прода 72.29 GiB = 58.6% от
blob 132.4 GB при нуле мёртвых ЗАПИСЕЙ.

## Проблема: watermark != мёртвые записи

`maybe_compact` гейтился watermark-байтами (`_dead_bytes = blob_eof - live`),
а `compact()` но-оп при нуле мёртвых записей (`len(keep) == len(records)`).
На прод-тире watermark вечно "просрочен" (72-88 GiB > порога 25%): каждый
ок-оффер впустую платил полный `_parse_journal` (1.5 МиБ + 270 JSON-записей +
crc) и компакт-проверку; строка стопа "dead=90284 MiB" при нулевой работе
компакта дезинформировала оператора. Прогноз P5 "пауза 55-90 с на оффере"
опровергнут именно этой парой (гейт срабатывал, компакт - но-оп).

## Суть фикса: отдельный ledger того, что компакт РЕАЛЬНО дропнет

Новые поля `_dead_records` / `_dead_record_bytes` - та же метрика, что но-оп
условие компакта; `_dead_bytes` НЕ переопределяется (остаётся физической дырой
водяного знака и служит кап-аккаунтингу):

- гейт: `_dead_record_bytes <= _blob_eof * _COMPACT_DEAD_FRACTION`
  (floor 256 MiB и дробь не тронуты). Watermark-дыры больше НИКОГДА не
  триггерят компакт - их физическое сжатие сознательно за P9 (байтами, а не
  голым счётчиком записей, сохранена амортизация "не компактить здоровый тир":
  один discard 4 КиБ не поднимает 80-с паузу на 100+ ГиБ блобе);
- точки ведения ledger (инвариант: у каждого in_l2 сегмента ровно одна
  журнальная запись; сегмент никогда не бывает одновременно in_l1 и in_l2):
  - runtime: `_discard` при seg.in_l2 - +1 запись / +padded байт под тем же
    `_account_lock`; это единственное место смерти записи (supersede /
    evict_store / _evict_oldest_l2 проходят через него); L1-only discard
    (padded=0) не считается - журнальной записи не было;
  - бут fast-path: 0/0 (shutdown_tier компактит безусловно ДО маркера -
    журнал == live);
  - бут full-path: число/байты payload-crc дропов (они остаются в файле
    журнала - replay его не переписывает - и парсер компакта их видит);
  - compact(): сброс 0/0 (журнал переписан до live).

### Почему ledger точен в краш-окнах

- SIGKILL между эвикцией и компактом: P2-маркер сброшен -> бут по полному
  пути -> "выброшенная" запись воскресает live -> ledger 0 - точно.
- SIGKILL в окне rename `_rewrite_journal`: старый журнал на месте, регионы
  за усечённым blob_eof читаются как zero-дыры -> payload-crc дроп -> ledger
  точен.

## stats_line: dead= и holes= разделены

`dead=Nrec/X MiB` (рекордно удалимое) + `holes=Y MiB` (watermark-дыры, дез-
информирующая "dead=90284.3 MiB" ушла). holes обёрнут в display-clamp
max(0): crash-окно компакта оставляет мёртвые спаны выше усечённого blob_eof
и разность уходит в минус; ledger НЕ клампится - гейту нужны точные байты.
`snapshot()` дополнен dead_records/dead_record_bytes (аддитивно); строка
cache.py "session tier final:" печатает stats_line как есть - согласована
автоматически; парсеры железных раннеров p3/p4/p5 якорятся по префиксу -
не сломаны.

НЕ тронуто: P2-маркер, механика компакта (ключ выживания (path_key, off, n),
blob-before-journal, переоткрытие journal-fd), кап-аккаунтинг (reseeds
`_ssd_used`/`_dead_bytes`/`_blob_eof`), P4-конвейер, CRC32C, env-рукоятки P6.

## Тесты (fails-before дословный)

`tests/scheduler/test_session_tier.py` (+116 строк): хелперы
`_boot_with_watermark_holes` (P5-iron сценарий: watermark-дыры + 0 мёртвых
записей + fast-path бут) и `_counting_compact` (счётчики `_parse_journal`/
compact).

- (a) `test_watermark_holes_zero_dead_records_offer_skips_compact` - ок-оффер
  при watermark-dead и 0 мёртвых записей не парсит журнал и не зовёт compact;
  сегмент сохранён и пробивается. Fails-before: `AssertionError: assert
  {'parse': 1, 'compact': 1} == {'parse': 0, 'compact': 0}`.
- (b) `test_dead_records_still_gate_compact_and_drop_exactly_them` -
  регресс-гвард P3-семантики: реальные мёртвые записи -> компакт как раньше
  (ровно 2 дропа, ledger обнулён, живой сегмент restore байт-в-байт).
  Fails-before: `AttributeError: 'SessionTierStore' object has no attribute
  '_dead_records'`.
- (c) `test_stats_line_dead_is_record_honest_and_holes_separate`. Fails-
  before: строка содержит `dead=32.0MiB` вместо ожидаемого `dead=0rec/0.0MiB`.
- (d) `test_full_path_boot_counts_payload_dropped_records` - crash-бут:
  2 записи / 8192 байт сидятся в ledger, компакт дропает ровно их, перезагрузка
  - журнал чист. Fails-before: тот же AttributeError.
- (e) `test_stats_line_holes_never_negative` (fix-волна по ревью): crash-окно
  через публичный API (compact + старый журнал на месте = SIGKILL до journal
  rewrite + полный бут): pre-fix в строке `holes=-0.0MiB`, post-fix
  `holes=0.0MiB`.

Fails-before верифицирован git stash A/B x3 с sha256-леджером
(методика - [test-fails-before.md](../../methods/test-fails-before.md));
session_tier.py и cache.py ЦЕЛ после каждого pop. Батарея:
`uv run pytest tests/scheduler/test_session_tier.py
tests/scheduler/test_hybrid_cache_manager.py -q` -> 157 passed
(87 + 70), 15.31 s, CPU-only.

## Ревью и открытое

STOP GATE READY: 0 blocker/major; MINOR (holes уходил в минус в crash-окне)
исправлен display-clamp'ом с тестом (e); NIT - докстринг maybe_compact
дополнен: holes= сбрасывается в 0 после компакта, физические внутренние дыры
остаются - их убирает перезапись P9, не этот гейт.

Открытое (решение пользователя, отдельный бриф): P9 - физическое сжатие файла
(сейчас live копируется на свои офсеты; P5-iron: после компакта 133.15 GiB при
live 44.98 GiB) и темп компакта 0.60 ГБ/с QD1 (доминанта over-cap стопа).
Консерватизм гейта: дробь считается от blob_eof с хвостовым мусором - тир
с большим мусором и малым discard-трафиком компактит реже (тот же класс
консерватизма, что до фикса); на малых тирах доминирует floor 256 MiB.

## Связанное

- Бриф: [TASK.md](TASK.md), "# Задача P8"; там же бриф P9.
- Наблюдение-источник: [P5-NOTES.md](P5-NOTES.md), "Железная рука".
- Тема (место P8-пункта в дизайн-хронологии тиринга):
  [../../topics/session-cache-tiering.md](../../topics/session-cache-tiering.md).
