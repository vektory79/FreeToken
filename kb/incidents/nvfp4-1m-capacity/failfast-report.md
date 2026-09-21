# Fail-fast отчёт: ранний честный reject несводимого бюджета gguf per-signature slot floors

Дата: 2026-09-16. Изменения в рабочем дереве, БЕЗ коммитов (поверх
незакоммиченного фикса engine/config.py + 2 тест-файла - не тронуты).

## 1. Точка вставки раннего check и почему именно она

`Engine._init_offload_moe_cache` (python/freetoken/engine/engine.py), вызов
`self._gguf_auto_floor_gate(...)` на строке 875 - СРАЗУ ПОСЛЕ блока
split-residency (который может погасить moe_prefill_overlap) и ПЕРЕД
`banks = load_expert_banks(...)`.

Почему здесь:
- Это самая ранняя точка, где расчёт ТОЧНЫЙ (тот же результат, что поздний
  сплит): все входы глобального авто-плана уже финальны - `self._baseline_free`
  и `self._weights_bytes` измерены (после загрузки плотных весов, ~+17s),
  `config.moe_prefill_overlap` принял финальное значение (flip от
  split-residency), `_pool_cls.kv_cost`, `state_pool_bytes`, `method` доступны.
  Раньше (до измерения весов) расчёт конверта был бы оценочным - нарушен бы
  инвариант "ранний check не срабатывает на влезающих сценариях".
- Единственный вход плана, которого ещё нет - `per_expert_bytes` (ширина слота
  слоя 0) - берётся из header-only gguf скана (`_gguf_bank_role_stacks`,
  миллисекунды, маппинг файла), который доказанно даёт ровно те же rows/rb,
  что материализует `load_gguf_expert_sources` (uint8 (E, rows, rb)); бридж-тест
  `test_gguf_signature_groups_match_loaded_banks` фиксирует равенство скана и
  реальных банков.
- Экономия: crash уходил на +85s (после ~68s serial build банков 9.87 GiB и
  resolve плана); теперь ValueError поднимается на ~+17s. Выигрыш ~68s (~80%
  времени до отказа). Плотные веса (+10s) по-прежнему грузятся раньше гейта -
  это цена точности `_weights_bytes`.

## 2. Дифф-сводка

- python/freetoken/engine/cache_budget.py (+33):
  - `partition_floor_slots(per_group_slot_bytes, num_experts)` - байтово-
    взвешенная сумма этажей групп в слот-0-эквивалентах:
    ceil(E * sum(p_i) / p_0). Сравнение `envelope >= need` эквивалентно
    байтовому `E*sum(p_i) <= envelope*p_0` бит-в-бит.
  - `check_partition_floors(groups, per_group_slot_bytes, num_experts, envelope)` -
    общий честный reject: число групп, слоёв в каждой, этаж E слотов/группу,
    взвешенная потребность, доступный конверт, remedy ("lower --kv-reserve-tokens
    or raise --memory-ratio"). Без магических чисел.
- python/freetoken/moe/expert_banks.py (+38): `gguf_signature_groups(model_path,
  model_config)` - header-only скан -> (группы, per-group slot bytes); None
  (поздний check остаётся единственным гейтом), если файл не bare .gguf,
  провайдер gguf не выбран, скан неполный. Dummy-веса и method-bound модели
  гейтятся в engine.
- python/freetoken/engine/engine.py:
  - `_resolve_auto_moe_cache_size(..., per_expert_bytes=None)` - новый
    override-параметр (скан-ширина до загрузки банков); дефолт - старый путь.
  - новый staticmethod `_gguf_auto_floor_gate` (строка 800): moe_cache_auto +
    не-dummy + method is None -> скан -> ранний resolve плана (та же чистая
    арифметика, что поздний resolve) -> check_partition_floors.
  - вызов гейта перед load_expert_banks (строка 875).
  - поздний auto-raise в `_split_moe_cache_budget` переведён на
    `check_partition_floors` (safety net; условие срабатывания эквивалентно
    прежнему `excess > 0` после цикла урезания - доказано достижимости:
    суммарная урезаемость >= excess_0 <=> need <= envelope). Explicit-ветка
    (--moe-cache-size) не тронута. Цикл пропорционального деления и урезания
    не тронут.

## 3. Сообщение об ошибке (ранний = поздний, один формат)

"--moe-cache-auto budget cannot fund the per-signature slot floors: 3 signature
groups (39/1/2 layers; groups [[0..7, 10..40], [8], [9, 41]]) each need 288
slots (one whole expert layer), a byte-weighted minimum of 1059 layer-0-width
slots vs the planned 974; lower --kv-reserve-tokens or raise --memory-ratio"
(для реального профиля 1M: E=288, p=[10.375, 15.0625, 12.6875] MiB ->
ceil(288*38.125/10.375) = 1059; на фикстуре - 15 vs 14).

## 4. Fails-before доказательство (Тест 1)

`uv run --extra dev python -m pytest tests/models/test_gguf_expert_banks.py -k
moe_auto_floor_gate -q` ДО правок:

```
AttributeError: type object 'Engine' has no attribute '_gguf_auto_floor_gate'
ImportError: cannot import name 'gguf_signature_groups' from 'freetoken.moe.expert_banks'
3 failed, 34 deselected in 1.92s
```

(ранней точки нет - именно ожидавшийся режим отказа). После правок: 4 passed.

## 5. Тесты (tests/models/test_gguf_expert_banks.py, +202; файл расширен,
новый не создан - там живут все сплит-тесты B3)

- test_moe_auto_floor_gate_fails_fast_before_bank_load (негативный, 1M-профиль:
  конверт 14 < потребность 15) - ValueError с честными числами на РАННЕЙ точке.
- test_moe_auto_floor_gate_boundary_envelope_passes (граничный: конверт 15 ==
  потребность 15) - гейт молчит, сплит проходит (все >= этажа, байтово впритык).
- test_moe_auto_floor_gate_fitting_profile_unaffected (786k-аналог: конверт 16,
  этажи 15 с запасом) - никаких новых исключений; скан == группировка/байты
  реальных шейпов; сплит даёт [5,4,4] (доминантная группа урезана чуть выше
  этажа, миноритарные на этаже - механизм [298,288,288]=1069).
- test_gguf_signature_groups_match_loaded_banks (бридж на файле, который
  принимает полный load-путь): группы и slot-bytes скана == то, что видят
  _group_bank_layers/expert_bytes_per_slot на материализованных банках.

## 6. Прогон сьютов (после фикса, сериализованно, под timeout)

- tests/models/test_gguf_expert_banks.py: 38 passed
- tests/engine: 158 passed, 2 skipped
- tests/models/test_gglm5_next_gguf.py + tests/models/test_qwen3_vl.py:
  70 passed, 1 skipped (незакоммиченный фикс engine/config.py цел)
- import-smoke freetoken.engine: ok
- Census: pre = post (0 чужих процессов, sem.mp- = 0)

## 7. Инварианты

- Ноль изменений поведения для влезающих конфигов: гейт только ДОБАВЛЯЕТ reject;
  для влезающих сценариев `envelope >= need` -> молчание; ранний resolve - та же
  чистая функция с теми же скалярами, что поздний (786k-кейс разрешается в те
  же числа; на фикстуре [5,4,4] и байт-эквивалент ≤ конверта).
- 1M-кейс: гейт поднимает тот же ValueError, но на ~+17s вместо +85s, с честным
  сообщением вместо "~8510 slots (the binding group's proportional share)".

## 8. Post-review fixes (2026-09-16, все 4 finding-а подтверждены TP)

- Finding 1 [minor]: позиция гейта в _init_offload_moe_cache (ДО
  load_expert_banks) запинена sentinel-тестом
  `test_init_offload_runs_floor_gate_before_bank_load`: monkeypatch
  `shared_offload_method`->None и `load_expert_banks`->sentinel (паттерн
  monkeypatch.setattr(module, ...) как с fixture_mod._EFF в этом же файле);
  gate-killer конфиг (конверт 14 < потребность 15); assert: ValueError поднялся
  И sentinel НЕ вызван. Reordering вызова теперь ломает тест.
- Finding 2 [nit]: `partition_floor_slots` получил assert'ы в начале (список
  групп непуст; slot_bytes[0] > 0) - битый заголовок умирает громко вместо
  IndexError/ZeroDivisionError.
- Finding 3 [nit]: reject-сообщение печатает per-group slot widths одним
  числом на группу в байтах/слот (для 1M-профиля: "... each need 288 slots
  (one whole expert layer) at <p0>/<p1>/<p2> bytes/slot ...", на фикстуре:
  "at 169984/246784/207872 bytes/slot") - число потребности теперь
  восстанавливается из текста; ассерт Теста 1 обновлен согласованно (добавлена
  ветка `at 169984/246784/207872 bytes/slot`).
- Finding 4 [nit, частично TP]: один компактный тест
  `test_moe_auto_floor_gate_silent_for_non_gguf` - гейт молчит (не бросает)
  при killer-конверте для (а) не-gguf формата экспертов (moe_weight_format=None,
  marlin/awq-путь) поверх .gguf-пути и (б) gguf-формата поверх не-.gguf пути
  (FTW-подобный). Полная матрица молчания не делалась (овер-инжиниринг).

### Прогон после фиксов

- tests/models/test_gguf_expert_banks.py: 38 passed, 2 failed - и это
  СОСЕДНИЕ CUDA-тесты (test_engine_partitions_three_groups_end_to_end,
  test_prefill_choreography_group_boundary_no_hop_contract) с torch.OutOfMemory:
  на карте чужая ЖИВАЯ волна ft serve (pid 474140, 786k-конфиг, порт 18081;
  backend 474277 держит 29.8 GiB VRAM) - по process-hygiene не тронута. Оба
  теста были зелеными в прогоне до review-фиксов (38 passed) до старта той
  волны; правки review - только CPU-пути (скан/гейт/сообщение/тесты), CUDA
  не трогают. К перепрогону когда VRAM освободится.
- tests/engine: 158 passed, 2 skipped.
- Census: семафоры (4) и VRAM - от чужой волны; собственных процессов нет.
- engine/config.py фикс (FrozenInstanceError guard) и его тест-файлы не тронуты.
