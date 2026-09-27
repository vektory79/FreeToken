# T2 Step 0: фазовая декомпозиция CPU-ноги (host delivery)

Статус: Step 0 DONE 2026-09-26 (гейт пройден: таблица опубликована, lever #1
назван); Wave 1 NOT STARTED.

Бриф кампании: [TASK.md](TASK.md). Публикация гейта Step 0 - фазовая таблица
[слой x фаза] лежит рядом: [step0_phase_table.csv](step0_phase_table.csv)
(39 доминантных строк + заголовок, скопирована из исходника без изменений).

## Условия прогона

- Дата: 2026-09-26, инструментированный base arm.
- HEAD на момент прогона: рабочее дерево vektory79 с незакоммиченной
  инструментацией
  [cpu_moe_ext.cpp](../../../python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp)
  (env FREETOKEN_T2_PHASE_TRACE); отдельного коммита нет.
- Итог руки: 15.64 tok/s steady, 63.94 ms/step wall (p50 62.5, max 153.6),
  540 completion tokens; boot readiness 89.7 s, fill 65536 из L2 за 4.3 s.
- Якорный класс: 16.29 tok/s / 61.4 ms. Этот boot ~4% медленнее в целом:
  driver span mean 66.51 против 59.08 в прежних руках.

## Артефакты руки

- Arm: [arm_base_t2.json](../../../.tasks/decode-research/arm_base_t2.json),
  [arm_base_t2.log](../../../.tasks/decode-research/arm_base_t2.log),
  [cpu_base_t2.csv](../../../.tasks/decode-research/cpu_base_t2.csv).
- Трейс:
  [t2_phase_trace.csv](../../../.tasks/decode-research/instr_t2/t2_phase_trace.csv) -
  22,848 записей; decode-блок = 22,680 = 540 шагов x 42 вызова
  (pool0=39 / pool1=1 / pool2=2 на шаг).
- Статистика драйвера:
  [stats_base_t2.json](../../../.tasks/decode-research/instr_t2/stats_base_t2.json)
  (step_spans_ms n=10, mean 66.51).
- Анализаторы:
  [analyze_t2_phases.py](../../../.tasks/decode-research/analyze_t2_phases.py),
  [t2_supplement.py](../../../.tasks/decode-research/instr_t2/t2_supplement.py).

Заметка о самодостаточности: `.tasks/` в .gitignore, перечисленные
raw-артефакты в kb не зеркалились (локальные копии); в kb опубликована
фазовая таблица выше - именно она закрывает гейт Step 0.

## Инструментация

- Env-гейт FREETOKEN_T2_PHASE_TRACE в cpu_moe_ext.cpp (незакоммичено);
  при выключенном гейте сборка bitwise-идентична (проверено).
- Фазовые точки: t_door / t_disp / t_first / t_last / t_flag
  (steady_clock, ns), построчные записи per-worker; сброс буфера каждые
  256 записей и на выходе.
- Сборка: AOT-сборка setup.py ТРЕБУЕТ gcc - clang 18.1.3 отвергает
  __builtin_cpu_supports("avxvnni") (pre-edit строка 735). Для этой сборки
  CC/CXX=clang НЕ ставить (clang нужен только для runtime gguf JIT).

## Фазовое разбиение шага (доминантный pool 0, 15 воркеров, на шаг)

| фаза | окно | ms | доля блока ~40 ms |
|---|---|---|---|
| wake+dispatch | t_first - t_door | 0.303 | 0.76% |
| compute | t_last - t_first | 38.829 | 98.03% |
| flag return | t_flag - t_last | 0.477 | 1.21% |
| итого занятость доминанты | | 39.61 | |

Эффективная скорость доминанты внутри своего compute-окна:
1.89 GiB / 38.83 ms = ~48.7 GB/s ~= bench burst 45-50 GB/s - нога У
потолка внутри окна; idle-инфляция живёт в промежутках между окнами,
а не в самих окнах.

## Минорные пулы (сериально, по 1 E-core)

- pool1: 1 слой, 0.185 ms/вызов.
- pool2: 2 слоя, 4.138 ms/вызов = 8.28 ms/шаг = 13% wall за 2/42 слоя.
- Скорость одиночного E-core: ~11.7 GB/s на ~48.5 MiB/слой.

## Бюджет шага (wall 63.94 ms)

- Конверт (первый door -> последний flag): 63.51 ms.
- = pool busy 48.07 (доминанта 39.61 + миноры 8.46) + межвызовные гэпы
  15.44 (fetch-копии ~6-8 + ядра + idle живут здесь).
- Вне конверта: 0.43 ms.
- GPU span драйвера 66.51 (окно n=10, медленный boot) = busy 48.07 +
  non-pool stream 18.43; расхождение span vs wall +2.57 ms
  задокументировано.

## Wake latency

- t_first - t_door: mean 7.77 us, p50 7.14, p90 9.75, p99 15.0, max 66.6.
- Окно dispatch: mean 10.7 us.
- Безобидная гонка t_disp > t_first в 10,850/22,680 вызовов: p50 избытка
  0.38 us, p99 3.97 us. Инвариант
  t_door<=t_first<=t_last<=t_flag держится всюду.

## Баланс воркеров (доминанта, mean 20,488 строк/вызов)

- P-tier w0-w7: 8.2-10.4% каждый; E-tier w8-w14: 3.3-4.5% каждый;
  max/min 3.14 (w0 2165.5 против w8 690.2).
- Тировая структура знакоустойчива между половинами прогона; per-call
  argmax ротируется (w0 сверху только в 15% вызовов); per-call CV
  0.64-1.04 - динамический stealing работает.
- top-3: w0/w4/w6; bottom-3: w10/w9/w8.

## Разброс по слоям

compute/слой от 0.867 ms (L22) до 1.248 ms (L0); top-3 L0/L1/L2,
bottom-3 L13/L26/L22 (детали -
[step0_phase_table.csv](step0_phase_table.csv)).

## VERDICT: переупорядочивание рычагов Wave 1

- Lever (b) persistent polling / fast dispatch: REFUTED - wake+dispatch
  0.303 ms/шаг (0.76%), оптимизировать нечего.
- Lever (a) P/E-aware row split: THIN - распределение уже почти
  пропорционально (max/min 3.14 ~= отношению скоростей P/E), stealing
  динамический; остаточный запас только в разбросе по слоям
  (0.87-1.25 ms) - в лучшем случае единицы ms, и нужны tail-данные,
  которых схема не пишет.
- Lever (c) chunk granularity: THIN - измеренного сигнала атомарной
  контенции нет; доминирует тот же потолок окна.
- NEW lever #1 - minor-pool starvation: pool2 съедает 8.28 ms/шаг
  (13% wall) на ОДНОМ E-core при ~11.7 GB/s. Подъём минорных пулов до
  нескольких потоков / P-core (или ребаланс layer->pool сплита) имеет
  измеренный потолок ~6-7 ms/шаг (+10% decode-класса). Дискриминатор:
  A/B minority thread/pin knob.
- Lever #2 - перекрыть остаточный fetch в 15.44 ms гэпов (там ~6-8 ms
  fetch; нужен nsys kernel-level split для оценки idle-доли; kineto
  остаётся под запретом).

## Кросс-проверка

wall = envelope + outside выполняется точно: 63.94 = 63.51 + 0.43;
фазы + копии + ядра сходятся - необъяснённого остатка нет.

## Topology footnote (2026-09-26, post-Wave-1)

На этом ядре P-SMT-соседи - смежные пары (0,1)..(14,15), E = 16-27;
ядро-набор доминантного пула - [0,2,4,6,8,10,12,14,16..22], и бут печатает
"0..22" как min..max этого списка, а не непрерывный диапазон. Числа Step 0
не затронуты (дефолтное разрешение пулов всегда было корректным); сноска
появилась потому, что спеки рук Wave 1 сначала прочитали "0..22" как
непрерывный диапазон - разбор в
[wave1-ab-summary.md](wave1-ab-summary.md).
