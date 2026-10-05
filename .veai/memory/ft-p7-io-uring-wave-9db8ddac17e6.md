---
name: "ft-p7-io-uring-wave"
description: "ftw_uring.py io_uring FTW read backend: committed 21f1525, opt-in default direct; QD refuted (5.16 GB/s), sys x1.45"
type: project
lastUpdated: 2026-10-02T22:46
lastRecall: 2026-10-02T22:42
---

# P7 io_uring ctypes wave (boot-shutdown-io, 2026-09-30, UNCOMMITTED @ 512b511)

io_uring-бэкенд чтения FTW-банков реализован, принят ревью x2 + fix + верификация,
прогнан железом (5 бутов). Незакоммичено; коммит и default-бэкенд - решение пользователя.

## Механика
- python/freetoken/checkpoint/ftw_uring.py (новый): raw ctypes-сисколлы io_uring
  (setup 425 / enter 426, mmap SQ/CQ/SQE по офсетам из kernel-ответа, БЕЗ liburing),
  READV-SQE, ЕДИНЫЙ цикл submitter/reaper на вызывающем потоке (thread confinement -
  никаких io_uring из CUDA/пин-потоков), short-read requeue терминален (res>0 строго
  уменьшает nbytes), EOF res==0 -> ошибка, IoUringUnavailable только из init ->
  graceful fallback + одна warning; platform guard x86_64/amd64 (ARM -> fallback:
  plain stores верны на TSO, на weak-memory тихая гонка); reverse-order unmap при
  частичном mmap-фейле; приёмные буферы заякорены в _Rec/in_flight до CQE; CDLL
  (не PyDLL) освобождает GIL на enter.
- ftw.py: _resolve_io_backend (io_uring|auto|direct; unset->direct = OPT-IN; garbage ->
  warning+direct), _io_qd (32, clamp 1..128); конфиг-строка "backend=io_uring, qd=32,
  granted=32" (granted = params.sq_entries); mmap-режим + io_uring -> явная warning.

## Главный урок (MAJOR fix)
Ring-путь терял settle zero-jobs задачи в СМЕСИ с читающими -> пропущенный on_done ->
пропущенный pins.submit -> IMA на первом decode (note-count ловушка, CPU-тестами
невидимо). Фикс: сеттл empty-jobs ПОСЛЕ успешного _Ring (НЕ до: иначе fallback дал бы
ДВОЙНОЙ on_done = двойной пин; fully-empty план сеттлится инлайн, fallback недостижим).
Урок: settle-порядок относительно failable init критичен для пин-контракта; явный
mixed-тест обязателен (запись в kb/methods/deterministic-concurrency-tests.md).

## Железо (5 бутов) - гейт фазы НЕ ВЗЯТ, гипотеза очереди ОПРОВЕРГНУТА
- io_uring QD32/QD64 = 26 с / 5.16 ГБ/с vs база direct 25 с / 5.37 (цель <=23 / >=5.5).
- Плато ~5.2-5.4 ГБ/с задают СТЫКИ конвейера загрузки (материализация HostBank /
  born-pinned / бар), не глубина очереди. Всплески 8-12.7 ГБ/с - устройство способно.
- CPU-гейт ВЗЯТ: sys фазы 38.8 -> 26.6 с (x1.45; один submitter дешевле 8 блокирующих
  preadv потоков).
- Следующий рычаг для P1-гейта (не брифицирован): оверлап пин-конвейера / крупные
  READV-цепочки. libaio не реализован (fallback-рука сделана). Short-read на живом NVMe
  не воспроизведён (юнит-покрытие).

## Тесты/артефакты
- tests/checkpoint/test_ftw_io_backend.py 16 тестов; tests/checkpoint 53+6sk;
  fails-before: 13 ERROR на HEAD (нет _IO_BACKEND_ENV), fix: 4 failed дословно
  (settled теряет 'empty').
- .tasks/boot-shutdown-io/p7-plan/{report.md,fix-report.md}, p7-iron/ (results.md,
  раннер, journal-backup). kb: P7-NOTES.md + 6 правок (TASK/README/ftw-load-path/
  methods/glossary/P1-NOTES) - НЕКОММИЧЕНЫ; P7-NOTES.md уже застейджен ловушкой.

## P7 закоммичен (2026-09-30, вечер)
- **21f1525** perf(checkpoint): io_uring read backend for ftw banks (3 файла, 873+).
- **fc9a04a** docs(kb): record p7 wave notes and p10 overlap-pipeline brief (7 kb-файлов).
- РЕШЕНИЕ ПОЛЬЗОВАТЕЛЯ: default = direct (opt-in подтверждён).
- Статус "UNCOMMITTED" в этой записи устарел. P10-бриф (оверклап пин-конвейера)
  зафиксирован в TASK.md - следующий рычаг P1-гейта, старт только по явному запросу.


