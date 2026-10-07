---
name: "ft-p1-ftw-pool-wave"
description: "P1 FTW bank-pool committed 94d5e4a; P10 seams refuted; P12 read-tempo CLOSED NO-GO (zero-fill 8.6% @ pool=8)"
type: project
lastUpdated: 2026-10-04T18:36
lastRecall: 2026-10-07T11:55
---

# P1 FTW bank-pool wave (boot-shutdown-io case, CPU part, 2026-09-29)

P1 из kb/cases/boot-shutdown-io/TASK.md реализован волной gguf-native-serving
(DIRECT_EXECUTION) на vektory79 @ 23b18a4, НЕ закоммичен. Локальная LLM резидентна ->
CPU-only; железная рука отложена.

## Что сделано
- Вилка 16x8 (до 128 конкурентных 8-МиБ чтений; замер: 9 пулов, peak 36 jobs) заменена
  ОДНИМ ThreadPoolExecutor(8): _plan_jobs/_read_job вынесены из read_into (контракт
  read_into не тронут - iter_ftw_weights ходит через него), _run_shared_reads с
  add_done_callback+счётчиком (on_done = progress bar + pins.submit РОВНО ОДИН раз на
  банк). Sub-read min(chunk, _BANK_SUB_CHUNK=2 МиБ) (~QD16x1M, форма fio-прецедента
  1xQD32x1M=6.45 ГБ/с). Задачи layer-major (layer_id, name).
- Env: FREETOKEN_BANK_POOL_WORKERS (1..16, clamp с warning через init_logger - T46).
  logger.info "FTW bank reads: N tasks, M jobs, pool=N, sub-chunk, backend=direct|mmap"
  - один call-site на бут (нет спама от read_into). Причина: без лога конфигурации
  железный A/B атрибутируется не той руке.
- НЕ тронуто: born_pinned_default, пин-контракт (tracker.note ровно один на слой -
  note-count ловушка: mismatch -> пиннинг молча пропускается -> IMA на первом decode,
  невидимо для CPU-тестов), FREETOKEN_BANK_CUDA_ALLOC.
- libaio/io_uring отложены за швом _run_shared_reads: системные libaio.so.1t64 /
  liburing.so.2 есть, python-обёрток нет; ctypes-путь реалистичен, но фаза идёт
  параллельно со строительством CUDA-контекста (thread-safety) + требует HW A/B.
- Тесты: tests/checkpoint/test_ftw_bank_pool.py 10/10 (новый файл); tests/checkpoint
  33+6sk+4 CUDA-env-errors (=базлайн). Fails-before: stash+md5, 7 failed дословно.
  Fix TP: delenv в autouse (тест читал реальный env var - экспорт из шелла железного
  прогона валит тест), тайминг-гард serial*6+1.0 (детерминированные тайминги),
  docstring-пачка, clamp-warning, logger.info.
- Верификационное ревью: STOP GATE READY; clamp/log спам рефутирован (один call-site);
  TRAPS T46 PASS.
- kb-distill: kb/cases/boot-shutdown-io/P1-NOTES.md (новый), TASK.md статус P1,
  kb/methods/env-config-hygiene.md (новая карточка: delenv-урок + clamp-атрибуция),
  topics/ftw-load-path.md (раздел пула). Некоммичены.

## Отложено
- Железная рука P1 (нужна VRAM): фаза <= 23 с / >= 5.5 ГБ/с (база 28-29 с / ~4.3),
  свип FREETOKEN_BANK_POOL_WORKERS 4/6/8, syscall-оверхед 2-МиБ sub-read (CPU-время
  фазы), mmap-parity на реальной ФС, layer-major vs name-major локальность.
- Коммит волны (ftw.py + тесты + kb) - решение пользователя; стейдж по явному списку.

## Коммиты (2026-09-29, поздний вечер)
- **94d5e4a** perf(checkpoint): consolidate ftw bank reads into one bounded pool
  (ftw.py +195, тесты +330; ветка дерева 23b18a4 -> HEAD b4fa210).
- **b4fa210** docs(kb): record p1 ftw pool wave notes (6 kb-файлов).
- Ветка опережает origin на 7; push не делался. Побочно: 4 бывших CUDA-env-error
  тестов прошли (VRAM освободилась - пользователь переключился на облако).

## Железная приёмка P1 (2026-09-29/30, артефакты .tasks/boot-shutdown-io/p1-iron/)
- A/B (5 бутов, фаза = бар "Loading expert banks (FTW)", 125 ГиБ): база 23b18a4 = 28 с /
  4.46 ГБ/с, CPU фазы 77.5 с (sys 76.4!); новый код pool=4: 25 с / 5.00 (повтор
  бит-в-бит), pool=6: 26 / 4.81, pool=8: 26 / 4.81; CPU 32.6-40.6 с (sys 32.6-39.5).
- ГЕЙТ ФАЗЫ НЕ ПРОЙДЕН: 1.12x вместо 1.4-1.5x; свип пула плоский 25-26 с - размер пула
  исчерпан как рычаг. CPU-гейт ПРОЙДЕН с запасом: sys-время 76.4 -> 32.6 с (2.3x меньше;
  2-МиБ sub-read дешевле по syscall-времени). Всплески 8-12.7 ГБ/с в хвостах бара -
  устройство способно на >5.5, лимитит пайплайн (глубина очереди). Следующий рычаг:
  ctypes io_uring/libaio за швом _run_shared_reads (системные библиотеки есть) или
  оверлап пин-конвейера.
- Паритет: конфиг-строка "FTW bank reads: 126 tasks, 64089 jobs, pool=N, sub-chunk=2048
  KiB, backend=direct" на всех руках; readiness chat-POST (пин-контракт работает);
  НАХОДКА: "MoE bank split residency" - УСЛОВНАЯ строка (только при unpinned-слоях),
  безусловный end-якорь - "expert banks: FTW fast path" (expert_banks.py:610); раннеры,
  грепающие residency, увидят её не всегда. iostat на sysstat 12.6.1 + ru_RU пишет
  only-since-boot снапшот - фазу мерить баром+таймстампами.
- Постфлайт: VRAM 1248 МиБ, ftw.py = HEAD бит-в-бит, tier не тронут, коммитов нет.

## Статус на конец сессии
Закоммичено: 94d5e4a (perf) + b4fa210 (docs); железная приёмка выполнена: гейт фазы
НЕ взят (25 с / 5.00 ГБ/с при цели <=23 / >=5.5; пул исчерпан), CPU-гейт взят (sys 2.3x).
Следующий рычаг: ctypes io_uring/libaio или оверлап пин-конвейера - ОТКРЫТО, решение
пользователя. kb-правки приёмки P1 (4 файла) НЕКОММИЧЕНЫ.

## P10 (2026-10-02): гипотеза стыков ОПРОВЕРГНУТА, волна отброшена
- P10-iron: диск занят 98-99% фазы уже на базе (idle 1.1%) - оверлапить нечего; плато
  5.2-5.4 ГБ/с = темп пути чтения (zero-fill внутри preadv). P10-машина = нон-регрессия,
  отброшена (ревью M1: окно троттлило билдер, оверлап 2-3%; M2/M3).
- ALLOC-замер: born-pinned руки берут гейт (22-23 с / 5.84-6.10), но общий бут ХУЖЕ
  (80-85 с vs 59 с готовности; аллокации +26..111 с CPU вне бара) - замер, НЕ рекомендация.
- Следующий рычаг: **P12** (бриф в TASK.md, OPEN): темп пути чтения - сначала замер доли
  zero-fill, потом код (кандидаты: pre-zeroed anonymous page как dest, GPU-side fill
  born-pinned only, large pages). Старт по явному запросу.

## P12 закрыт NO-GO (2026-10-04)
"Следующий рычаг P12" из хвоста этой записи закрыт волной-1: доля zero-fill 8.6% @ дефолт pool=8 (НЕ 4) < гейта 10%, warm-потолок 22.85 с нереализуем (first-touch zero на fresh mmap неизбежен), кандидаты ~0/отрицательны; кода нет, kb зафиксирована (карточка kb/methods/limiter-attribution.md). Детали: ft-p12-read-tempo-brief.
