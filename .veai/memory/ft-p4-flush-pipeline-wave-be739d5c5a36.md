---
name: "ft-p4-flush-pipeline-wave"
description: "P4 zero-copy pwritev flush pipeline + P6 env overrides; iron accepted 2.18-2.46 GB/s; commits b99a103, dd349a6"
type: project
lastUpdated: 2026-09-30T22:40
lastRecall: 2026-10-07T22:45
---

# P4 flush-pipeline wave + P6 env-override + iron acceptance (boot-shutdown-io, 2026-09-30)

Single P4 ledger. Motivation: producer-узкое место P3-флаша (0.93-1.16 ГБ/с при util 20-40%).

## Что сделано (COMMITTED)
- ZERO-COPY: pwritev прямо из регионов L1-пула (memoryview-срезы pool.mm; страницы
  сегмента несмежны), pad из статического _ZERO_BLK (байт-идентичность блоба),
  чанкинг 512 iov, цепочечный CRC32C по need-байтам (crc32c освобождает GIL с
  32 КиБ - crc билдера перекрывается с записью).
- КОНВЕЙЕР ГРУПП: flush_live собирает ВСЕ группы списком; _flush_groups: билдер-поток
  (ThreadPoolExecutor(1)) строит группу N+1, консьюмер (main) пишет группу N
  (pwritev -> ОДИН fdatasync -> журнал). Инвариант durable-до-журнала и окно потери
  "группа" не тронуты; runtime _write_blob не тронут.
- Синхронизация: _PendingVolume (Condition) pending-объём; _account_lock (leaf-lock,
  нереентрантный) вокруг _ssd_used/_dead_bytes; правило "билдер не берёт _lock".
  P3-семантика жертв - bounded drain-retry (60 с, warning; fallback pending=0 может
  временно перекрыть кэп на объём группы - только shutdown-flush, задокументировано).
- BaseException-безопасность: _release_stage (идемпотентный флаг) на трёх путях;
  KI пробрасывается как KI.
- Тесты: test_session_tier 72/72 (6 новых: drain-retry, SystemExit-release, iov-чанкинг,
  общая дедуп-страница в разных группах), test_hybrid_cache_manager 70/70; fails-before
  точечным ревертом; два ревью + верификация: STOP GATE READY.
- Коммиты: **b99a103** perf(scheduler): zero-copy flush pipeline for l2 tier
  (session_tier.py +392, тесты +422), **fc3396f** docs(kb): record p4 flush pipeline
  wave notes (6 kb-файлов, вкл. methods/deterministic-concurrency-tests.md).

## P6 env-override (COMMITTED)
- FREETOKEN_FLUSH_WRITERS / FREETOKEN_FLUSH_GROUP_BYTES: резолверы в точке
  использования (unset -> константа; мусор/ниже пола -> дефолт + warning, без silent
  clamp; whitespace-only = unset); резолв один раз на флаш и передача параметром вниз
  (_flush_groups -> _write_group -> _pwrite_group); одна лог-строка конфигурации на флаш.
- Тесты 79/79 + 70/70; **dd349a6** perf(scheduler): env overrides for l2 flush workers
  and group bytes, **4c091a1** docs(kb): record p6 flush env override wave notes.

## Железная приёмка P4 - ПРИНЯТА (9.829 GiB / 52 сегмента; blob-кривая 20 Гц + /proc/diskstats 10 Гц; артефакты .tasks/boot-shutdown-io/p4-iron/, вне git)
- BASE (5f9abcc point-checkout, 8/1 ГиБ): 0.95 ГБ/с (write 1.15). DEFAULT (8/1 ГиБ):
  2.27/2.18 (write 3.07/2.84), util 53%. W4 (FREETOKEN_FLUSH_WRITERS=4): 1.92.
  G2 (group 2 ГиБ): 2.46 (write 3.10).
- ВЕРДИКТ: throughput-гейт (>=1.5-2 ГБ/с) ПРИНЯТ: x2.3-2.6 от базы. util>=60% НЕ взят
  в среднем (45-58%, пики 99-195%): диск простаивает в стыках групп (журнал + хэндофф
  билдера). Producer-паузы (>=0.5 с) ИСЧЕЗЛИ (n=0). Следующий рычаг при желании:
  группы 4 ГиБ / 16 райтеров.
- SIGKILL на новом дереве потоков PASS: t=+1.34 с после 2 журналированных групп ->
  маркера нет -> полный путь 270 records, НОЛЬ dropped/torn -> конвергенция -> resume
  HIT 9280/9324. used= аккумулирован бит-в-бит по всей волне (фикс :329 жив).
- Замерные ловушки: iostat only-since-boot на sysstat 12.6.1 + ru_RU - мерить
  blob-кривой/diskstats; exit=127 от setsid-хэндовера - косметика.
- Отклонения: restore-on-demand отдал базовые промпты из L2 (флаш 0 B - методика:
  свежие нонсы на руку); кэп поднят 50->60 ГиБ посреди волны (политика инертна,
  NO_CAP_EVENTS; при следующем прод-флаше штатно эвиктит сегменты волны);
  diskstats-семплер починен посреди волны. Постфлайт: VRAM 1178 МиБ, дерево бит-в-бит
  HEAD, tier 270 records / blob 132.4 GB, маркер VALID.
- Результаты дублированы в kb/cases/boot-shutdown-io/P4-NOTES.md; kb-правки приёмки
  на момент записи были некоммичены, закоммичены последующим docs-коммитом.
