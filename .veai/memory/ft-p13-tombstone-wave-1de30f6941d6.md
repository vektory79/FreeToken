---
name: "ft-p13-tombstone-wave"
description: "P13 tombstone COMMITTED (12623ff code + 91e53ff kb); KeyError accepted; sub-floor gate refined; case P1-P13 all closed"
type: project
lastUpdated: 2026-10-05T11:18
lastRecall: 2026-10-07T11:55
---

# P13 durable-tombstone wave (boot-shutdown-io, 2026-10-04/05, HEAD 9d91ec8 + wave, commit pending)

P13 кейса boot-shutdown-io РЕАЛИЗОВАНА полностью: реализация -> CPU-тесты -> ревью x2 (NOT BLOCKING) -> fix-TP -> HW-приёмка (все гейты взяты) -> kb-distill. Коммит НЕ сделан - решение пользователя, как и 2 открытых контракта.

## Что в волне (3 файла: session_tier.py +130, cache.py +9, tests/scheduler/test_session_tier.py +277)
- Новый тип журнальной записи {"t":1,"path_key","off","n"} (durable-tombstone); discard пишет fsync'ed томбстоун немедленно (_discard, _append_tombstone); неудача аппенда -> _marker_blockers фоллбек в pre-P13 поведение.
- replay_journal honors томбстоуны на ОБОИХ путях (fast-path включительно), сет-фильтр порядко-независим; жертвы без задвоения в P8-ledger; блокеры силятся.
- Новый shutdown_compact(): skip при dead>0 и blockers==0; cache.py shutdown_tier вызывает его; B3-гейт маркера - блокируют только не-томбстоуннутые мёртвые; компакт томбстоуны не переносит ("применён - не переносить"); формат старых записей не тронут.
- Fix-TP после ревью: 2 комментария (builder-thread evicts идут в _discard без _lock - корректность на атомарном O_APPEND + сет-фильтре; "no on-disk bytes" устарел), fsync-ассерт в survives-restart тесте (проверен в обе стороны), тест ledger на смешанных dead (dead==v+pdropped, blockers==pdropped).

## Числа
- Тесты: tier 122 passed; fails-before 3 красных на старом коде; батарея 2492 passed / 206 skipped / 5 известных Environment.
- HW A/B (кап 32 GiB, graceful стоп, прод-тир-копия как старт, fill 40 диалогов x 9.3k, порт 18821): стоп **16.3 с vs BASE 55.7 с (-71%)**; flush 4.0 с (2.21 ГБ/с); компакт-фаза ОТСУТСТВУЕТ (BASE 37.8 с, 47.07->31.95 GiB); teardown 11.3 vs 12.7; BASE сходится с P11-iron 51.7 с (8%).
- Воскрешений нет (35/35 эвицированных cached=0, 80 tombstoned stay dead); байт-точность 159/159 md5 сурвивор-спанов; resume HIT (d36-d40 cached=9280) + fast-path "honored 80 tombstones, 80 stay dead"; филл +1.3% (80 per-discard fsync - шум); второй skip-стоп стабилен 18 с; P8-гейт на следующем буте собрал дыры (рантайм-компакт 47->32 GiB при идентичных md5).

## Открытое (решения пользователя)
1. Даунгрейд-асимметрия: старый бинарник на журнале с томбстоунами = KeyError краш бута (жёстче crc32c-прецедента). Принять или мягкий skip unknown-type.
2. Тиры навсегда ниже P8-floor 256 MiB теряют пошаутдаунную реакламацию (skip-гейт вместо безусловного компакта; ~100 Б/дискрад в журнале до P8-гейта). Принять+документировано или уточнить гейт.
3. Коммит волны.

## Уроки
- Сценарий "большой live" НЕ воспроизводится чистым тиром (40 диалогов = ~15 GiB < капа 32) - нужна cp -a копия прод-тира (прод read-only). /media/ai переехал на nvme1n1 - вотчеры с авто-детектом.
- Даунгрейд-контракт новых типов журнальных записей - решение до мержа, не после (сильнее crc32c-прецедента).

kb: TASK.md P13 "Ход волны" + статусы (РЕАЛИЗОВАНА, hw-гейты взяты), topics/session-cache-tiering.md (tombstone-политика + числа), P9-NOTES.md B3-кросс-ссылка, cases/README.md. kb-правки НЕКОММИЧЕНЫ вместе с волной. Артефакты: .tasks/boot-shutdown-io/p13-tombstone/ (CHANGES.md, STATUS.md, hw/results.md, логи).

## Gate refine (2026-10-05, решение пользователя: тиры ниже floor)
Открытое (2) РЕШЕНО: skip-гейт shutdown_compact() теперь `dead>0 И blockers==0 И blob_eof >= _COMPACT_FLOOR_BYTES` (константа P8-гейта, :54, 256 MiB; метрика = watermark blob_eof, как в рантайм-гейте; новой env-ручки нет). Тиры ниже floor компактятся по-старому (перепись там ~мс - гигиена возвращена), skip остаётся только где экономит время. Фикстура _shrink_fixture(blob_above_floor) - физика+watermark из одного fstat. Fails-before пара (маленький тир: красный "skipped" -> зелёный shrink+reclaim), W1-W14 зелёные, tier 123 passed, батарея 2493 passed / 5 Environment. Фокусное ревью 6/6 PASS NOT BLOCKING (полярность, семантика константы, честная фикстура, деградаций нет); номинальный нит не-ASCII docstring исправлен (тест зелёный). HW не перегонялся: сценарий 32 GiB на 2 порядка выше floor, skip-путь не менялся. kb: TASK.md открытое (2) = РЕШЕНО, session-cache-tiering floor-оговорка. ОСТАЛОСЬ: KeyError-даунгрейд контракт + коммит волны.

## ЗАКОММИЧЕНО (2026-10-05, решение пользователя: KeyError-контракт принят как есть)
- **12623ff** feat(scheduler): make tier discard durable via journal tombstones (3 файла, +428/-80; тело коммита несёт HW-числа 55.7->16.3 с и принятый даунгрейд-контракт).
- **91e53ff** docs(kb): record p13 tombstone wave, gate refine and accepted contracts (4 kb-файла).
- Перед коммитом: tier-файл 123 passed контрольный прогон; индекс стартовал пустой, стейдж по явным спискам, residual dirty в python/tests/kb = 0. P13 кейс-фаза ЗАКРЫТА полностью (все P1-P13 кейса boot-shutdown-io закрыты: P12 NO-GO, P13 реализована). Хвост: temp tier-копии .tasks/boot-shutdown-io/p13-tombstone/hw/tier_* можно удалять (гиг-игнор, десятки ГиБ).
