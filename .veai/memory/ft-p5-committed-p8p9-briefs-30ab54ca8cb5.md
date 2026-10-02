---
name: "ft-p5-committed-p8p9-briefs"
description: "P5 committed 94e4783 + docs 512b511; P8 watermark-no-op and P9 file-shrink briefs fixated; campaign status"
type: project
lastUpdated: 2026-09-30T21:04
lastRecall: 2026-10-01T12:47
---

# P5 committed; P8/P9 briefs fixated (2026-09-30)

- **94e4783** perf(scheduler): log l2 flush overflow evictions - P5-минимум закоммичен
  (_FlushOverflow records/bytes/admitted, flush-локальная аккумуляция single-writer,
  лог "flush overflow: evicted N records / X.X MiB ..."; 82/82 + 70/70).
- **512b511** docs(kb): record p5 overflow wave and p8/p9 briefs (4 kb-файла).
- Статус "P5 НЕЗАКОММИЧЕНО @ 823d87e" в ft-p5-lru-research устарел.

## Новые брифы (kb/cases/boot-shutdown-io/TASK.md, якорная база 94e4783)
- **P8** (дешевле): maybe_compact сеется watermark'ом, но compact() но-оп при нуле
  мёртвых ЗАПИСЕЙ -> каждый ок-оффер впустую парсит журнал; строка "dead=90284 MiB"
  дезинформирует. Фикс: триггер по record-based dead; dead-строку убрать или разделить
  bytes/records. Якоря: maybe_compact :1691, но-оп :1736-1737, stats_line :1823.
- **P9** (крупнее): компакт копирует live на свои офсеты - файл НИКОГДА не сжимается
  (P5-iron: после компакта 133.15 GiB при live 44.98 ГиБ, физический dead 66%);
  темп 0.60 ГБ/с QD1 (81.3 с на 44.98 ГиБ). Спека: переписка live в НОВЫЙ blob
  (tmp+rename), офсеты журнала одним транзакционным шагом, крэш-безопасность (старый
  блоб до фиксации), параллельная переписка (прецедент P4-конвейера).
- Оба OPEN; P8 дешевле, P9 крупнее; один PR каждый.

## Кампания boot-shutdown-io на конец сессии
Выполнено и принято железом: P2, P3, P4 (throughput 2.18-2.46 ГБ/с), P5, P6.
Открыто: P1-гейт (io_uring -> P7), P4-util (стыки групп), P5, P8, P9.
Ветка vektory79 опережает origin; push не делался.
