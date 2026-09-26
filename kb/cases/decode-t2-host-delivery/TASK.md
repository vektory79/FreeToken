# TASK: T2 - совмещённая доставка host-DRAM в декоде гибрида (микро-простои пинг-понга)

Status: NOT STARTED. Composed 2026-09-26 из исхода decode-research кампании
(`.tasks/decode-research/`, вердикты в памяти `ft-decode-research-campaign` /
`ft-decode-lever-research`). Read CONTRIBUTING.md first - он обязателен. Якоря
строк проверены 2026-09-26 против HEAD 0ff8495 (vektory79); перед правками
перепроверить (дрейфуют). Private-use scope GGUF glm5next: no push, commit
только по запросу пользователя.

## Goal

Декод-шаг гибрида @64k = 61-62 ms и **полностью сериализован на стриме**
(span 59.1 ms из ~62 стены, CUDA-event замеры). Внутри шага две ноги -
CPU-пул (~1.89 GiB/шаг) и PCIe-фетч (~0.68 GiB/шаг) - суммарно качают
**2.57 GiB при ~41.5 GB/s эффективных**, тогда как benchbw overlapped-пара
показывает **72 GB/s** одновременной доставки (cpu_ov 49.97 + pcie_ov 22.12,
профиль `~/.cache/freetoken/benchbw/GPU-ec07f396-*.json`). Разрыв x1.7 - это
не скорость ног (burst CPU ~45-50 GB/s ~ бенч; gather ~34-36 vs 43.6 бенч),
а **пер-слойные микро-простои**: разгон/хвосты барьера пула, диспетч
doorbell, межслойные GPU-окна, где ни одна нога не работает.

Goal: поднять совмещённую доставку до 55-65 GB/s -> декод 16.3 -> **>=19 tok/s
@64k** (растяжка 21+). Любой принятый рычаг >= +5% декода без регресса
префилла (>1%) и без падения батареи качества.

## Verified background (не пере-выводить; числа с конфигом)

- Якорный конфиг = ежедневная команда пользователя: FTW-директория
  GLM-5.3-Flash-UD-Q3_K_XL-FTW/, `--moe-cache-auto --kv-reserve-tokens 400000
  --kv-cache-dtype fp8 --memory-ratio 0.82 --max-prefill-length 8191
  --moe-strategy hybrid --moe-cpu-threads 16 --max-running-requests 1
  --session-tier-ram-gib 10 --session-tier-dir .../cache
  --session-tier-ssd-gib 50`, порт 18801. Якорь сессии: **16.29 tok/s @64k**
  (61.4 ms/шаг); re-acceptance 16.67 (63b9bff) - тот же класс, дневной шум.
- Пулы [15,1,1] (веса 39/2/1 слоя): доминирующий = 8P+7E "cores 0..22",
  миноритарные по 1 E-core (23, 24). CPU i7-14700KF: P = cpu0-15 (SMT),
  E = cpu16-27; AVX-512 фьюз-off (потолок ISA = AVX2+VNNI, уже достигнут
  портом 11f1a80).
- Занятость пула в окне декода ~59% (P-воркеры ~62% @4.9 GHz, E ~55 %
  @3.85 GHz) - "26.9 GB/s эффективных" это idle-inflated; миноритарные пулы
  почти спят (7-16%) и критическим путём НЕ являются.
- Фиксированные ~17 ms CPU_exposed на шаг НЕ сжимаются объёмом
  (fetch3: -0.9% нейтрально; ov0: -23.5% => оверлап прячет ~19 ms).
- Опровергнутые рычаги (НЕ реанимировать): кеш-политика (oracle_hit 25.7%
  vs LRU 25.1%; WS 184/288, энтропия 0.83 - routing почти равномерен),
  перекалибровка сплита f, замена flag-sync (hostfunc -21.6%), ISA, GPU-ядра,
  FlashAttention (архитектура GDN+DSA - плотного attention нет), CUDA-графы
  (нейтральны при bs=1). Offload на этом конфиге 13.72 tok/s; закрытие его
  gather-разрыва (34-36 vs 43.6 бенч) даёт только паритет с гибридом.
- Handshake: memop-doorbell (flag-sync) + координатор; хуки декод-шага -
  `GraphRunner.replay` (шаги - CUDA-graph REPLAYS: хост-код модели, включая
  `OffloadMoELayer._decode_routed`, НЕ выполняется per step). stat-тензоры
  кешей выделяются безусловно; флаги collect_stats/collect_decode_freq
  обязаны быть выставлены ДО capture; routing-гистограмма честна только в
  eager (`--cuda-graph-max-bs 0`).

## Харнес (готов, пользоваться им)

`.tasks/decode-research/`: `run_arm.py` (boot точного конфига, fill 64k -
восстанавливается из session-tier L2 за ~1 с, radix-reuse декод 768 токенов,
per-CPU /proc/stat+MHz сэмплер, census- guard, SIGTERM-teardown),
`campaign.py` (серийные руки), `analyze.py` (сводка + busy/MHz по P/E),
`analyze_trace.py`, `driver/sitecustomize.py` v3 (GraphRunner.replay хук +
CUDA-event спаны шагов + collect_stats/freq; дамп `stats_<tag>.json`).
Дампы сессии 2026-09-26: `instr_hyb/stats_{hyb,off,eager}.json`,
`arm_*.json`, `cpu_*.csv`. Протокол декод-замера: fill -> ре-отправка того же
промпта (`#cached-token: 65536`), steady по SSE-дельтам, TTFT исключён.

## Step 0 - фазовая декомпозиция CPU-ноги (первым, до любых дизайнов)

1. Инструментация `cpu_moe_ext.cpp` (env-gated debug-патч, решение о
   коммите - отдельное): таймстемпы фаз на (слой, шаг) - [doorbell seen |
   dispatch done | first worker start | last worker done (tail) | done-flag
   written]. Плюс пер-воркерная сумма отработанных строк (для P/E-баланса).
2. Прогон base-руки харнесом; таблица фаз x 42 слоя: какая доля из ~40 ms
   wait-блока - wake/dispatch, чистый compute, хвост барьера, возврат флага.
   Сверка: сумма фаз + копии + ядра + простой ~ 61-62 ms (иначе модель
   неполна - расширять инструменты, не фантазировать).
3. GPU-сторона: кинето на этом боку ВАЛИТ планировщик (нативный abort при
   открытии окна, 2026-09-26 - FREETOKEN_DR_PROF держать unset); kernel-level
   сплит только через nsys (помнить trap тихого no-collection - верифицировать
   мини-пробой). Для Step 0 достаточно событий: копии vs ядра vs idle
   оцениваются из спанов + ov0/fetch3 производных.
4. Порог перехода к Wave 1: фазовая таблица опубликована (kb), рычаг №1
   назван числом.

## Wave 1 - рычаг №1 (кандидаты, порядок решает Step 0)

- (a) P/E-осознанное распределение работы: строк-чанки пропорционально
  throughput воркера (E-ядрам меньше строк / отложенные хвосты), вместо
  равного chunk stealing. Дискриминатор: хвост фазы (last worker - first
  worker) и busy-дисбаланс P vs E из сэмплера.
- (b) Персистентный поллинг воркеров / быстрый диспетч: убрать прослойку
  пробуждения на (слой, шаг). Дискриминатор: фаза wake+dispatch из Step 0.
- (c) Гранулярность работы: ~65k (expert,row) пар на слой с атомарным
  chunk stealing - попробовать крупные статичные чанки. Дискриминатор:
  атомарная конкуренция (счётчики) + фаза compute.
- Механика A/B: env-knob, per-call/fail-fast (домашний стиль), одна
  переменная на руку, серийные руки харнесом, >= 2 повторных рук на победителе
  (день/вечер шум до 10% - время суток рядом с числом).

## Gates и приёмка

- Рычаг принят: декод >= +5% (>= 17.1 tok/s) на >= 2 руках, префилл в
  пределах +/-1%, батарея качества 24/24 (kb/harness/quality/run_battery.py).
- e2e-bitwise НЕ гейт (boot-to-boot недетерминизм бимодален - урок
  arbitration); для scheduling-изменений математика строк не меняется.
- Каждый adopted-рычаг = отдельный conventional-commit, только по запросу
  пользователя; дефолт-флипы - отдельные микрокоммиты.
- Артефакты кампании: фазовая таблица + A/B JSON + сводка - в этот каталог
  (kb самодостаточен), вердикт - в память агента.

## Traps (обжжённые)

- kineto/torch.profiler убивает планировщик - профайлер не включать
  (FREETOKEN_DR_PROF unset); nsys - с верификацией сбора.
- pkill -f по паттерну, входящему в собственную командную строку, убивает
  себя (exit 137, 2026-09-26) - только bracket-паттерны `[f]t serve`.
- run_arm готовность = настоящий chat-запрос (200), не /v1/models.
- benchbw ПЕРЕЗАПИСЫВАЕТ полный профиль GPU - `ft bench bw` не запускать;
  если меняются ноги - пере-профилировать ВСЕ dtype.
- Руки печатают результат в конце (`tail` скрывает прогресс) - не паниковать
  раньше времени; аварийный сервер ловится WATCH-паттерном + census.
- /tmp-скрипты кампаний утрачиваются - всё жить в путях репозитория.
- Инструментация cpu_moe_ext.cpp требует rebuild: `python setup.py
  build_ext --inplace` (clang++ host, CC/CXX scoped - см. память
  ft-gguf-kernel-jit-toolchain).

## Точные команды

```
PY=/media/ai/src/FreeToken/.venv/bin/python
DR=/media/ai/src/FreeToken/.tasks/decode-research/driver
OUT=/media/ai/src/FreeToken/.tasks/decode-research/instr_t2
$PY .tasks/decode-research/run_arm.py --name base --log arm_base.log \
  --out .tasks/decode-research/arm_base.json \
  --env "FREETOKEN_DECODE_RESEARCH=$OUT" --env "PYTHONPATH=$DR" --env "FREETOKEN_DR_TAG=base"
$PY .tasks/decode-research/analyze.py base   # сводка + per-CPU P/E
# A/B рука: та же команда + --extra "<env-gated knob value как флаг>"
```
