# Fix-отчёт: FrozenInstanceError на GGUF (engine/config.py:141)

Дата: 2026-09-16. Диагноз: .tasks/ft-serve-gguf-nvfp4-1m-crash-work/repro-report.md.
Изменения остаются в рабочем дереве, коммитов нет.

## 1. Дифф фикса (before/after)

Файл: python/freetoken/engine/config.py, EngineConfig.model_config (единственное
изменение в продукционном коде).

Before:

```python
        for key in set(ENCODER_SECTIONS) | {e.config_key for e in self.model_spec.encoders}:
            if key not in built:
                setattr(hf_config, key, None)
```

After:

```python
        for key in set(ENCODER_SECTIONS) | {e.config_key for e in self.model_spec.encoders}:
            # GGUF's frozen GgufConfigShim carries no encoder sections and raises
            # FrozenInstanceError on any setattr; sections the config never had
            # already read as None through getattr's default downstream.
            if key not in built and hasattr(hf_config, key):
                setattr(hf_config, key, None)
```

Почему hasattr-guard корректен (проверено перед правкой):
- models/gguf/config.py:25: GgufConfigShim - @dataclass(frozen=True) с полями
  (architectures, model_path, model_type, metadata, vocab_size,
  tie_word_embeddings); полей vision_config/audio_config НЕТ, поэтому guard
  превращает оба setattr в no-op для GGUF.
- Сгенерированный dataclass __setattr__ бросает FrozenInstanceError безусловно
  (даже для отсутствующих полей), поэтому copy.copy не спасает - нужен именно
  запрет setattr, а не смена способа копирования.
- Для HF-конфигов, несущих секцию (Qwen3-VL и др.), атрибут существует, guard
  истинен - поведение байт-в-байт как до фикса (зануление unbuilt-секций на
  копии; сам закешированный конфиг не мутируется - он копируется).
- Секции, которых у конфига никогда не было, читатель-парсер и так видит как
  None через getattr(..., None) - семантика "unbuilt == None" сохранена.

## 2. Регресс-тесты (fails before / passes after)

- Test A (регрессия): tests/models/test_glm5_next_gguf.py::
  test_engine_model_config_resolves_over_the_frozen_shim - строит синтетический
  glm5next GGUF (готовый fixture файла), прогоняет НАСТОЯЩИЙ путь
  EngineConfig.model_config (без monkeypatch: cached_load_hf_config ->
  build_gguf_shim -> parse_gguf_config) и проверяет, что шим не тронут.

  ДО фикса - УПАЛ с точной продакшн-сигнатурой:

  ```
  python/freetoken/engine/config.py:141: in model_config
      setattr(hf_config, key, None)
  ...
  name = 'audio_config', value = None
  E   dataclasses.FrozenInstanceError: cannot assign to field 'audio_config'
  =========================== short test summary info ============================
  FAILED ...::test_engine_model_config_resolves_over_the_frozen_shim
  1 failed in 1.76s
  ```

  ПОСЛЕ фикса - PASSED (2 passed in 1.75s вместе с Test B).

- Test B (golden семантики main на HF-пути): tests/models/test_qwen3_vl.py::
  test_unbuilt_encoder_sections_are_nulled_only_where_they_exist - hf-конфиг
  с vision_config (built) + audio_config (unbuilt), класс-Recorder фиксирует
  setattr'ы на копии, которую видит парсер.

  ДО фикса - PASSED (1 passed in 1.64s). ПОСЛЕ фикса - PASSED.
  Инвариант теста: `calls == ["audio_config"]` - зануляется ровно unbuilt-секция,
  built (vision_config) не тронут, исходный объект конфига не мутируется.

## 3. Прогон сьютов (после фикса)

```
uv run --extra dev python -m pytest \
  tests/engine tests/models/test_glm5_next_gguf.py tests/models/test_qwen3_vl.py \
  tests/models/test_qwen3_vl_vision.py tests/mm tests/scheduler/test_mm.py \
  tests/tokenizer/test_mm_tokenize.py tests/server/test_mm_input.py \
  tests/server/test_message_wire.py tests/kernels/test_mrope.py -q

284 passed, 12 skipped, 2 warnings in 4.16s
```

Покрывает: весь tests/engine (включая mm-тесты волны 08d728d
test_mm_encoder/test_mm_repeated_image), gguf-конфиг/шим glm5next, Qwen3-VL
(mm-волна), mm/scheduler/tokenizer/server mm-тесты 08d728d, mrope-тесты.

import-smoke: `uv run python -c "import freetoken.engine"` -> "import ok".

## 4. E2E ре-репро точной команды пользователя

Команда - точная команда пользователя (порт 18081), runner /tmp/ft_e2e_fix/runner.sh
(trap-on-EXIT + FAILPAT watchdog 5s + /ready-полл + hard timeout 900s).

ВЕРДИКТ E2E: исходный краш УСТРАНЕН (boot прошел точку FrozenInstanceError и
дошел до планирования MoE-кэша), но boot НЕ завершился - NEXT-FAILURE в другом
месте (см. 4.1). /ready не достигнут, smoke не выполнялся.

Таймлайн (лог /tmp/ft_e2e_fix/server.log):
- +0s (14:06:54) uvicorn up, ServerArgs распарсены (kv_quant='nvfp4',
  moe_cache_auto=True, kv_reserve_tokens=1048576).
- +7s: Resolved config: moe_strategy='hybrid', attention_backend='dsa',
  cache_type='hybrid_radix', page_size=64 (auto 1->64 для latent-KV).
  Free memory before loading model: 29.45 GiB.
- +17s: expert banks: slow path (serial build).
- +85s: gguf signature histogram: {(18,18,14): 2, (18,18,23): 39, (23,23,14): 1}.
- +85s: --moe-cache-auto RESOLVED: moe_cache_size=974 num_pages=16403
  (prefill_overlap=True). Страницы 16403 x 64 = 1,049,792 >= 1,048,576 reserve:
  глобальный plan_cache_budget assert (кандидат 1 из брифа) НЕ сработал - 1M
  nvfp4 KV в бюджет поместился.
- +85s: ValueError в _split_moe_cache_budget (engine.py:732) -> scheduler-воркер
  умер; supervisor корректно погасил сервер; watchdog поймал Traceback на +95s.

### 4.1 NEXT-FAILURE (не чинится по условию задачи)

Сигнатура:

```
ValueError: --moe-cache-auto budget cannot fund the per-signature slot floors
(288 slots per group; layers [[0..7,10..40], [8], [9, 41]]): the expert byte
envelope 9.87 GiB is below the partitioned minimum of ~8510 slots (the binding
group's proportional share)
  engine.py:419 __init__ -> _init_offload_moe_cache
  engine.py:906 -> _split_moe_cache_budget
  engine.py:732 -> raise ValueError
```

Числа и механика (из кода engine.py:640-740 + строк лога):

- Авто-план выдал byte_cap_slots=974 слот-0-эквивалента (ширина сигнатуры
  слоя 0 (IQ3_XXS, IQ3_XXS, IQ4_XS) = 9.87 GiB / 974 = 10.37 MiB/слот).
- 3 сигнатурные группы (39/1/2 слоя из 42 MoE). Относительные веса групп
  (bpw-единицы банков gate/up/down): w0 = 3.0625+3.0625+4.25 = 10.375;
  w1 (слой 8, IQ4_XS+IQ4_XS+Q6_K) = 15.0625; w2 (слои 9,41, IQ3+IQ3+Q6_K) =
  12.6875. W = 39*10.375 + 15.0625 + 2*12.6875 = 445.06.
- Пропорциональные доли от 974: [886, 33, 55] -> этаж 288 поднимает до
  [886, 288, 288]. Байтовая проверка: 886 + 288*1.452 + 288*1.223 = 1656
  слот-0-эквивалентов > 974. Цикл урезания срезает группу 0 до её этажа 288,
  остаток excess = 84 слот-0-эквивалента (~0.87 GiB) некуда девать -> raise.
- ФАКТИЧЕСКИЙ минимум под этажи (после цикла урезания): 288*(1 + 15.0625/10.375
  + 12.6875/10.375) = ~1058 слот-0-эквивалентов против конверта 974 - не хватает
  ~84 слота (~0.87 GiB).
- Число ~8510 в сообщении - другая граница: max(ceil(288*W/w_i)) =
  ceil(288*445.06/15.0625) = 8510 - минимум ПО пропорциональному правилу
  (binding group = слой 8, самый широкий). Сообщение использует формулу
  explicit-budget пути; операционно релевантен минимум ~1058 (этажи всех групп
  при байтовом учете).
- Почему у кампании 63b9bff (ре-акцептанс) такого не было: там дефолтный KV
  dtype и --num-tokens 70016 --memory-ratio 0.85 - KV-запас крошечный, конверт
  слотов 1234 >= 1058 - проходило. Здесь 1M nvfp4 KV (16403 страницы, ~3.83
  GiB) съедает конверт до 974 < 1058.

Справочный пересчет, что поместилось бы (НЕ рекомендация, конфиг не менялся):
- минус 262144 токена KV-резерва (-4096 страниц nvfp4 ~ -0.96 GiB ~ +95
  слот-0-эквивалента): 974+95 = 1069 >= 1058 - проходит;
- либо memory-ratio 0.92 (+0.88 GiB конверта ~ +85 слота) - впритык и съедает
  headroom CUDA-графов/активаций (0.08 x 29.45 = 2.36 GiB против 8192-chunk
  трансзиентов ~3.4 GiB) - рискованно.

Fix direction (без реализации): plan_cache_budget/resolve_moe_cache_auto не
знает о per-signature этажах - глобальный solve дает 974, сплит требует >=1058.
Минимальное направление: передать в глобальный план нижнюю границу MoE-слотов
суммой этажей в слот-0-эквивалентах (288*sum(w_i/w_0) по группам гистограммы
сигнатур), чтобы auto fail-fast'ился до чтения весов с честным числом, либо
включать этажи прямо в MoE-vs-KV solve. Это отдельное решение (код слит из
кампании gguf-гибрида, не из 08d728d).

## 5. Process hygiene

- Preflight census перед тестовой волной: чужих процессов нет (только wrapper),
  sem.mp- = 0, VRAM 1408 MiB used / 30741 MiB free (baseline).
- Каждый pytest-прогон под `timeout`, сериализованно.
- E2E: /tmp/ft_e2e_fix/runner.sh - trap-on-EXIT cleanup, FAILPAT- watchdog
  (5s), /ready-полл, hard timeout 900s, SIGTERM -> 30s -> SIGKILL.
- Postflight census после e2e: чужих процессов нет, sem.mp- = 0, VRAM
  1422 MiB used / 30727 MiB free - baseline; сервер погасился сам и cleanly
  (воркеры reaped pids=[376768, 376769], hang из известного паттерна не возник).
