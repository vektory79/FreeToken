---
name: "ft-p12-read-tempo-brief"
description: "P12 read-tempo CLOSED NO-GO: zero-fill 8.6% @ default pool=8 < 10% gate; premise refuted, no code; kb fixated"
type: project
lastUpdated: 2026-10-04T18:36
lastRecall: 2026-10-04T22:23
---

# P12 read-path tempo: ЗАКРЫТА NO-GO (wave-1 2026-10-04, no code @ 2d57b78)

Премиса брифа ("плато 5.2-5.4 ГБ/с фазы загрузки FTW-банков = zero-fill внутри preadv") ОПРОВЕРГНУТА cold-vs-warm A/B на реальных шардах. Кода нет, коммитов нет.

## Числа (ревью MEASUREMENT VALID, перепрогон сошёлся)
- Дефолт фазы = pool=8, НЕ 4: expert_banks.py:572 (workers=8) -> engine/engine.py:1022-1031 -> ftw.py:769 _bank_pool_workers(8); все 57 бутов p1/p4/p5/p7-iron дали "pool=8".
- Доля zero-fill = (T_cold-T_warm)/T_cold: 8.6% @ pool=8 spread (диапазон 8.0-8.8%, перепрогон 8.2%); 3.5% @ pool=4 (не дефолт); prefix 2.5/3.2%. Все < гейта волны 10%.
- Потолок ПОЛНОГО устранения (warm) = 25 x 0.914 = 22.85 с - формально касается гейта 23 с, но НЕРЕАЛИЗУЕМ: dest = свежий mmap.mmap(-1, asize) на банк (host_banks.py:112), first-touch zero неизбежен; MAP_POPULATE переносит ту же цену. Warm = переиспользование буфера между проходами = артефакт замера.
- Кандидаты брифа на дефолтном pool=8: (а) devzero MAP_PRIVATE ХУЖЕ cold (5.32 vs 5.59 ГБ/с; COW-fault каждую страницу; явных zero-областей на default-пути нет - padding читается с диска, raw.zero_() только в cuda-ветке host_banks:105); (в) MADV_HUGEPAGE не вступил (THP=madvise), cold_thp = cold; (г) ограничен warm-потолком; (б) born-pinned вне скоупа (p10-iron: бут хуже).
- Fault+zero цена реальна, но перекрыта IO: cold-дельта sys 195-215 ms/GiB (масштаб фазы ~33-36 с sys ~ p1-iron 32 с), дельта стенки 0.05-0.2 с. Стенка = диск (98-99% busy, P10-iron) + форма пути.

## Уроки
- Атрибуция лимитера ключится к ДЕФОЛТНОЙ конфигурации по цепочке из исходников, не по ярлыку (сначала заякорили на pool=4 - ревью поймало).
- Рычаг "X лимитирует стенку" гейтится потолком ПОЛНОГО устранения X ДО кода.
- O_DIRECT минует page cache - согрев кэша не конфаунд. Реплика "чище" фазы => её доля - верхняя граница.
- Measurement-only волна -> один фокусный ревью вместо полного x2 (оправдано: поймал 2 TP + 2 FP в отчёте).

## kb (зафиксировано, КОММИТА НЕТ - решение пользователя)
TASK.md P12 = ЗАКРЫТА NO-GO (числа инлайном); P1-NOTES P10-iron атрибуция = ОПРОВЕРГНУТО + коррекция; topics/ftw-load-path.md итог; НОВАЯ карточка kb/methods/limiter-attribution.md; methods/README + cases/README индексы. Артефакты волны: .tasks/boot-shutdown-io/p12-read-tempo/ (гит-игнор, нΕ переживает клон). kb-правки (5 изменённых + 1 новый файл) некоммичены.

**Why:** единственный открытый рычаг гейта P1 (23 с / 5.5 ГБ/с default-пути) исчерпан: гейт на default-пути недостижим известными рычагами; берут только born-pinned руки (22-23 с) ценой общего бута 80-85 с (не рекомендация).
**How to apply:** не реанимировать zero-fill-рычаги для FTW-загрузки; следующий шаг кейса - P13 durable-tombstone (отдельная сессия по явному запросу).
