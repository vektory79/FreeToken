---
name: "ft-p8-record-compact-gate"
description: "P8 record-dead compact gate: committed 373505b + docs 1de8069; honest dead/holes stats, 157 passed; P9 landed ab52a33"
type: project
lastUpdated: 2026-10-02T22:46
lastRecall: 2026-10-05T10:54
---

# P8 record-based compact gate (boot-shutdown-io, 2026-10-01, UNCOMMITTED @ fc9a04a)

Проблема (P5-iron): maybe_compact гейтился watermark-байтами (_dead_bytes =
blob_eof - live), но compact() но-оп при нуле мёртвых ЗАПИСЕЙ -> прод-тир
(dead 72.29 GiB = 58.6% blob, мёртвых записей 0) платил _parse_journal на
каждом ок-оффере; строка "dead=90284 MiB" дезинформировала.

## Фикс (незакоммичен; STOP GATE READY, ревью 0 blocker/major)
- Ledger _dead_records/_dead_record_bytes = что compact РЕАЛЬНО дропнет.
  Точки: _discard при seg.in_l2 (под _account_lock; единственное место смерти
  записи); бут fast-path 0/0 (shutdown компактит до маркера), full-path =
  payload-crc дропы; SIGKILL между эвикцией и компактом -> маркер сброшен ->
  полный путь -> запись воскресает live -> ledger 0 - точно; SIGKILL в окне
  rename -> _rewrite_journal: crc-дроп за eof -> ledger точен. compact()
  сбрасывает 0/0.
- Гейт: _dead_record_bytes <= _blob_eof * _COMPACT_DEAD_FRACTION (floor
  256 MiB сохранён). Watermark-дыры больше НИКОГДА не триггерят компакт -
  их сжатие за P9 (сознательно).
- stats_line: "dead=Nrec/X MiB" + "holes=Y MiB" (watermark-дыры;
  display-clamp max(0) - crash-окно компакта даёт отрицательную разность;
  ledger не клампится). Парсеры железных раннеров (p3/p4/p5) якорятся по
  префиксу "session tier final:" - не сломаны.
- НЕ тронато: P2-маркер, компакт-механика (ключ выживания path_key,off,n),
  кап-аккаунтинг (:329 reseeds), P4-конвейер, CRC32C, P6 env.

## Тесты/приёмка
- 5 новых: (a) оффер при watermark-dead/0 записей НЕ парсит журнал
  (counting compact), (b) реальные мёртвые записи -> компакт как раньше,
  (c) stats_line честный, (d) full-path seed ledger, (e) holes never negative
  (crash-окно: holes=-0.0MiB -> holes=0.0MiB).
- 87/87 test_session_tier + 70/70 hybrid = 157 passed; fails-before дословный
  (stash A/B, sha256 ledger цел).

## Урок
Семантическое разделение двух "dead": watermark-байты (дыры прошлых
поколений, реальны для резерваций) vs record-dead (что компакт удалит,
реален для триггера). Смешивание давало но-оп-парсинг + дезинформацию.

## Открыто
- Коммит волны (session_tier.py + cache.py docstring + tests + kb P8-NOTES/
  TASK/README/session-cache-tiering) - решение пользователя.
- P9 (реальное сжатие файла компактом + темп) - OPEN, отдельный бриф.

## P8 закоммичен (2026-10-01)
- **373505b** perf(scheduler): gate l2 compaction on dead records not watermark bytes
  (session_tier.py +49, cache.py docstring, тесты +153; 157 passed перед коммитом).
- **1de8069** docs(kb): record p8 record-dead compaction gate wave (4 kb-файла).
- Статус "UNCOMMITTED @ fc9a04a" в этой записи устарел. P9 остаётся OPEN.

## Коррекция (2026-10-01): P9 больше не OPEN
P9 (сжатие блоба + темп) реализован волной 2026-10-01 (two-phase commit, file==live PASS,
1.40 ГБ/с PARTIAL, W10/W11/W14 уроки) - см. ft-p9-two-phase-compact-wave. Фраза
"P9 остаётся OPEN" в этой записи устарела. P8 сам закоммичен 373505b + 1de8069.
