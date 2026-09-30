---
name: "ft-p5-lru-research"
description: "P5 LRU already in prod; wipe was :329 degenerate; P5-minimum overflow observability committed 94e4783; iron accepted"
type: project
lastUpdated: 2026-09-30T22:41
lastRecall: 2026-09-30T22:35
---

# P5/LRU research: eviction already in prod; wipe degenerate closed by :329 (2026-09-30)

Read-only исследование по вопросу пользователя "насколько сложно реализовать LRU"
(P5-гейт, session_tier.py @ 823d87e; kb/cases/boot-shutdown-io/TASK.md P5).

## Главное
- LRU-вытеснение при over-cap УЖЕ в проде: _evict_oldest_l2 (session_tier.py:1067-1073)
  выбирает жертву min по (last_validation, seg_id) - настоящая LRU (last_validation
  освежают restore/note_match/consume_ticket, бут восстанавливает из ts журнала).
  Стоп-флаш идёт через тот же _reserve_blob_span (:735), который при капе зовёт эвикцию
  с pending - ровно столько старого L2, сколько нужно флашу.
- Исторический "wipe" (discard всех + truncate блоба до 0 за 0.6-2 с, ARM2/P3-iron)
  был ВЫРОЖДЕНИЕМ бага :329 (watermark-сеянный _ssd_used не снижался эвикциями -> все
  demote REFUSED -> компакт с пустым keep обтрунцивал блоб). Корень закрыт ab3a117
  (пересев на ОБОИХ путях, :1637) - ft-fix-329-cap-accounting; admission стал
  dead-независимым = детерминированным.
- Отказ "L2 cap reached, cannot demote" (:897) сегодня возможен только если
  pending+padded > кэпа ДАЖЕ после полной эвикции L2 (при кэпе 60 ГиБ и группах
  1-2 ГиБ - недостижимо).

## P5-минимум: реализован и ЗАКОММИЧЕН (решение пользователя: LRU как есть, опция (б) отклонена)
- Overflow-наблюдаемость: класс _FlushOverflow (records/bytes/admitted), слот
  self._flush_overflow (try/finally вокруг _flush_groups), _evict_oldest_l2 инкрементит
  после _discard (single-writer билдер), _write_group - admitted; лог "session tier:
  flush overflow: evicted N records / X.X MiB of oldest L2 to admit Y bytes" (один на
  флаш при эвикциях, тишина иначе). Аккумуляция flush-локальная (НЕ self._counters -
  урок P4). Подписи не менялись.
- Тесты 82/82 (+3: overflow-числа, тишина, drain-retry-eviction) + 70/70; fails-before
  tests-first + stash/md5; фокус-ревью чисто.
- Коммиты: **94e4783** perf(scheduler): log l2 flush overflow evictions; **512b511**
  docs(kb): record p5 overflow wave and p8/p9 briefs (P5-NOTES.md новый, TASK.md P5
  закрыт, README, deterministic-concurrency-tests). Не реализовано из исследования:
  компакт-как-рычаг (не брифицирован).

## P5-железная рука (2026-09-30, .tasks/boot-shutdown-io/p5-iron/) - гейты приняты
- dead прода: 72.29 GiB = 58.6% от blob 132.4 GB (270 records, live 51.03).
- Over-cap стоп (ssd-gib=45, live 51.03 = 113% кэпа): флаш 3.62 с / 2.91 ГБ/с (без
  эвикций - паритет P4), эвикции 84 records / 16.26 ГиБ старейших (~0 с), компакт
  81.3 с / 44.98 ГиБ live = 0.60 ГБ/с (QD1, util 86%) - доминанта стопа (83%), полный
  стоп 98 с. LRU-симметрия: 5/5 выживших HIT 99.5%, эвикнутая старейшая - MISS +
  репрефилл. Overflow-строка прод-масштаба; без кап-давления - тишина. Wipe: 0
  инцидентов. Fast-path бут 0 с.

## Вне-гейтовые находки -> брифы P8/P9 (детали в ft-p5-committed-p8p9-briefs)
- (P8) maybe_compact сеется watermark'ом, но compact() но-оп при 0 мёртвых ЗАПИСЕЙ ->
  каждый оффер впустую парсит журнал; строка "dead=90284 MiB" дезинформирует
  (водяной знак вечно "просрочен").
- (P9) компакт копирует live на их офсеты - файл НИКОГДА не сжимается (после компакта
  133.15 GiB при live 44.98 ГиБ, физический dead 66%); реальное сжатие - переписка в
  НОВЫЙ blob. Темп компакта 0.60 ГБ/с QD1 - кандидат на ускорение.

## Тайминги over-cap стопа из исследования (used 51.03 GiB live, кэп 60, флаш 9.83 ГиБ)
- Эвикция D~1 ГиБ: ~0 с (in-memory, ни лога, ни fsync; сотни эвикций < 0.1 с).
- ДОМИНАНТА - компакт на том же стопе (cache.py:944 безусловный; но-оп при dead=0):
  копирует только LIVE-спаны (QD1 0.6-1 ГБ/с), оценка 55-90 с при 51 GiB live
  (подтверждена железом: 81.3 с).
- Вариант (б) "кэп только runtime-гейт" (пропуск sacrifice/drain-retry по флагу,
  ~40-80 строк через flush_live->_flush_groups->_build_group_stage->_reserve_blob_span):
  0 эвикций -> 0 dead -> компакт но-оп; стоп = чистый флаш 4.1-4.7 с (темп P4-iron);
  цена - неограниченный рост блоба (уже 132.4 GB).
- Бут после LRU-стопа: маркер VALID -> fast-path ~0 с; SIGKILL -> полный реплей ~27 с
  оценка @ 51 GiB; dead пересеивается точно (ab3a117).
- Риски правок: не выйти из дисциплины "flush_live держит _lock весь конвейер", не
  тронуть drain-retry P4 (:880-896), pending-аккаунтинг, маркер.
