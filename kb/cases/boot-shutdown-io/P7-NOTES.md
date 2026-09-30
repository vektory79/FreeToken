---
title: "Волна P7: io_uring ctypes-бэкенд чтения FTW-банков (opt-in)"
date: 2026-09-30
hardware: "RTX 5090 32GB; NVMe Samsung 990 EVO Plus 2TB (nvme0n1); i7-14700KF; железная A/B - 5 бутов на флагах оператора из TASK.md"
commits: "волна закоммичена (21f1525, база волны vektory79 @ 512b511): python/freetoken/checkpoint/ftw_uring.py (новый), ftw.py (+52 строки), tests/checkpoint/test_ftw_io_backend.py (новый); default-бэкенд: default = direct (решение пользователя 2026-09-30)"
status: "реализация + ревью x2 + fix + верификация (STOP GATE READY) + железная A/B: гейт фазы НЕ взят (гипотеза глубины очереди ОПРОВЕРГНУТА), CPU-гейт ВЗЯТ (sys фазы x1.45); батарея tests/checkpoint 53 passed / 6 skipped (16 новых тестов)"
tags: [ftw, banks, io-uring, ctypes, o-direct, queue-depth, thread-confinement, fallback, pin-contract, settle]
---

# Волна P7: io_uring ctypes-бэкенд чтения FTW-банков

Реализация брифа задачи P7 из [TASK.md](TASK.md) - вторая попытка поднять фазу
загрузки FTW-банков (125 ГиБ) выше плато ~5 ГБ/с после P1-пула
([P1-NOTES.md](P1-NOTES.md)): снять ограничение "QD=1 на поток" блокирующего
preadv глубокой очередью io_uring (прецедент формы: fio 1 поток x QD32 x 1 МиБ
= 6.45 ГБ/с на тех же шардах). Итог: реализация остаётся полезной (CPU-гейт
взят, открывает READV-цепочки), но центральная гипотеза волны "лимитит глубина
очереди" ОПРОВЕРГНУТА - см. "Железная A/B".

## Механика

- `python/freetoken/checkpoint/ftw_uring.py` (новый, ~300 строк): кольцо
  io_uring на raw ctypes-сисколлах io_uring_setup(425) / io_uring_enter(426),
  mmap SQ/CQ/SQE по офсетам из ответа ядра (io_uring_params) - БЕЗ liburing
  и без новых pip-зависимостей, раскладка колец не хардкожена. SQE = READV
  через struct.pack_into. ЕДИНЫЙ цикл submitter/reaper на вызывающем потоке
  (prep до QD -> enter(submit, min_complete=1, GETEVENTS) -> reap пачкой):
  thread confinement - никаких io_uring-вызовов из CUDA/пин-потоков; CDLL
  отпускает GIL на enter. Короткий read (0 < res < nbytes) -> переочередование
  остатка (терминальность: res>0 строго уменьшает nbytes); EOF (res==0) ->
  явная ошибка; первая ошибка поднимается после слива кольца.
  IoUringUnavailable поднимается ТОЛЬКО из инициализации -> это сигнал
  graceful fallback на пул-путь, а не ошибка чтения.
- Платформенный guard x86_64/amd64: на weak-памяти (ARM) plain stores кольца
  дают тихую гонку без барьеров liburing; на x86 TSO plain stores корректны и
  сериализуются границей io_uring_enter. Non-x86 -> IoUringUnavailable ->
  fallback (пул-путь остаётся верным).
- Частичный провал mmap при инициализации -> reverse-order unmap уже
  сделанных отображений.
- Lifetime буферов: буферы заякорены в _Rec/in_flight до CQE (export +
  memoryview) - кольцо не держит висячих указателей.
- `ftw.py` (+52 строки): резолвер `_resolve_io_backend`
  (`FREETOKEN_FTW_IO_BACKEND=io_uring|auto|direct`; unset -> direct - бэкенд
  OPT-IN по умолчанию; auto - алиас io_uring; мусор -> warning + direct),
  `_io_qd` (`FREETOKEN_FTW_IO_QD`, default 32, clamp 1..128 с warning),
  развилка в `_run_shared_reads`. Конфиг-строка "FTW bank reads: ..."
  дополнена: "backend=io_uring, qd=32, granted=32" - granted =
  params.sq_entries, фактически выданный ядром размер SQ (qd в regex
  фазового парсера остаётся запрошенным). mmap-режим FTW-ридера + io_uring ->
  явная warning "ignored under mmap fallback" (атрибуция A/B).
  read_into / iter_ftw_weights, born_pinned_default, FREETOKEN_BANK_CUDA_ALLOC
  и пин-контракт (tracker.note ровно один на слой) не тронуты.

## Главный урок: settle-порядок относительно failable init

Ревью поймало MAJOR: ring-путь сеттлил zero-job задачи (банки без sub-read)
ДО успешного создания кольца, в СМЕСИ с читающими. Пропущенный on_done ->
пропущенный pins.submit -> IMA на первом decode. Ловушка двойная: CPU-тесты
байт-точности это НЕ видят (чтение корректно), а нарушение контракта
"on_done ровно один на задачу" проявляется только на железе (note-count
ловушка P1).

Фикс: сеттл empty-jobs ПОСЛЕ успешного `_Ring(qd)`; не до - иначе при
IoUringUnavailable ftw.py уходит в пул-fallback и сеттлит те же задачи
ВТОРОЙ раз -> двойной on_done = двойной pins.submit (сломан пин-контракт
уже в fallback). Полностью-empty план (очередь пуста) сеттлится инлайн без
кольца - fallback там недостижим. Обобщение правила - "settle-порядок при
failable init": [../../methods/deterministic-concurrency-tests.md](../../methods/deterministic-concurrency-tests.md).

## Тесты

`tests/checkpoint/test_ftw_io_backend.py`, 16 тестов: байт-точность реального
кольца vs serial read_into; заворот SQ (16 jobs x QD4) побайтно; пин-контракт
(ровно один PinPipeline.submit на банк); fallback при падении _Ring; garbage
env; mmap-режим без кольца; EBADF-шард -> слив кольца и raise; путь
SystemExit; отсутствие утечки потоков; юниты _complete
(short-requeue/EOF/EIO/full); settle zero-job задачи в смеси с читающими.
autouse delenv обоих env (урок P1), детерминированные фикстуры без таймеров.

- Fails-before волны (дословно): против HEAD ftw.py + тест-файл волны -
  13 ERROR (AttributeError: no attribute '_IO_BACKEND_ENV'); без
  ftw_uring.py - ImportError на импорте ftw_uring.
- Fails-before фикса (дословно): 4 failed / 12 passed - settled теряет
  'empty' (assert ['read'] == ['empty', 'read']), потерян ok-settle, guard
  non-x86 не поднимал IoUringUnavailable, нет mmap-warning.
- Raw-ring тесты скипаются с явной причиной там, где io_uring недоступен:
  module-level probe один `_Ring(1)` на сессию.
- Батарея после фикса: tests/checkpoint 53 passed / 6 skipped (до волны
  37+6sk; волна +13, фикс +3).

## Железная A/B (5 бутов, сериализовано)

Конфиг: флаги оператора из [TASK.md](TASK.md), фаза = бар
"Loading expert banks (FTW)", 125 ГиБ, readiness = настоящий chat-POST
(usage.completion_tokens >= 1), форма очереди 126 tasks / 64089 jobs /
sub-chunk 2048 KiB на всех руках, готовность 60-61 с (P2-маркер: реплей
243 записей мгновенный). Артефакты руки (раннер, логи, summary) лежали в
git-игнорируемом каталоге .tasks/boot-shutdown-io/p7-iron/ и после клона
считаются утраченными; суть - таблица и вердикты ниже.

| рука | backend / QD | фаза, с | ГБ/с | CPU фазы, с | sys, с |
|---|---|---|---|---|---|
| b (база) | direct пул (P1) | 25 | 5.37 | 40.0 | 38.8 |
| u32 | io_uring QD32 | 26 | 5.16 | 27.2 | 26.6 |
| u64 | io_uring QD64 | 26 | 5.16 | 28.0 | 26.9 |
| garbage | env-мусор -> fallback | 26 | 5.16 | 40.3 | 38.8 |
| br | direct (повтор базы) | 26 | 5.16 | 40.7 | 39.8 |

- ГЕЙТ ФАЗЫ (<= 23 с / >= 5.5 ГБ/с) НЕ ВЗЯТ: io_uring QD32 = QD64 = пул
  (25-26 с / 5.16-5.37, разрыв с fio 6.45 ГБ/с сидит в конвейере). Гипотеза
  P1 "лимитит глубина очереди" ОПРОВЕРГНУТА: плато ~5.2-5.4 ГБ/с задают стыки
  конвейера загрузки (материализация новых HostBank, born-pinned cudaHostAlloc
  секции, обновление бара), не очередь и не syscall-оверхед. Всплески
  8-12.7 ГБ/с в хвостах бара - устройство способно быстрее средней.
- CPU-ГЕЙТ ВЗЯТ: sys 38.8 -> 26.6 с (x1.45): один submitter с io_uring_enter
  дешевле восьми потоков блокирующего preadv.
- Паритет: байт-точность через успешный chat-POST (пин-контракт жив) на всех
  руках; fallback-рука (мусор в env) - одна warning "ignoring unknown",
  поведение и CPU идентичны direct, конфиг-строка честно говорит backend=direct.
- Прод-тир не тронут (журнал 243 записи = бэкап бит-в-бит, blob mtime не
  изменился, shutdown.marker VALID); постфлайт чистый: VRAM 1272 МиБ,
  sem.mp- 28 = 28, дерево бит-в-бит волновое, коммитов нет.

## Открытое / отложенное (решение пользователя)

- Коммит волны и выбор default-бэкенда: сейчас opt-in (unset -> direct);
  wall-время фазы нейтрально, CPU-выигрыш реален. default = direct
  (решение пользователя 2026-09-30).
- libaio НЕ реализован: raw-syscall io_uring не зависит от библиотек, libaio
  дал бы второй ABI без новой гипотезы; брифом разрешена fallback-рука вместо
  libaio-руки.
- Short read (res < nbytes) покрыт юнитом _complete, на живом NVMe не
  воспроизводился (O_DIRECT-чтения возвращали полные длины); resubmit-путь
  сознательно простой.
- P1-фазовый гейт остаётся открытым с новым знанием: очередь не рычаг.
  Кандидаты следующей волны (не брифицированы): оверлап пин-конвейера с
  чтением, крупные READV-цепочки (несколько iovec на SQE, до 8 МиБ), метки
  t_submit/t_reap внутри `_run_shared_reads` для точного ответа, где фаза
  теряет 3-4 с.

## Связанное

- Бриф: [TASK.md](TASK.md), "Задача P7".
- Пул-путь P1 (база волны): [P1-NOTES.md](P1-NOTES.md).
- Тема: [../../topics/ftw-load-path.md](../../topics/ftw-load-path.md).
- Урок settle-порядка:
  [../../methods/deterministic-concurrency-tests.md](../../methods/deterministic-concurrency-tests.md).
- Env-гигиена (delenv, clamp-warning, конфиг-строка):
  [../../methods/env-config-hygiene.md](../../methods/env-config-hygiene.md).
