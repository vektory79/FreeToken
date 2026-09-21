# Repro-отчёт: краш ft serve (GGUF GLM-5.3-Flash) на vektory79 @ f786d7c

Дата: 2026-09-16. Режим: только воспроизведение и диагностика. Код репозитория не менялся.

## 1. Точная команда

```bash
/media/ai/src/FreeToken/.venv/bin/ft serve --port 18081 --host 127.0.0.1 \
  --model-path /media/ai/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL.gguf \
  --moe-cache-auto --kv-reserve-tokens 1048576 --kv-cache-dtype nvfp4 \
  --memory-ratio 0.89 --max-prefill-length 4096 --moe-strategy hybrid \
  --moe-cpu-threads 16
```

Окружение: repo /media/ai/src/FreeToken, branch vektory79, HEAD f786d7c
(Merge branch 'main' into vektory79; parents: 6e673ab + afd99cb; по коду дерево чистое).
Модель: GLM-5.3-Flash-UD-Q3_K_XL.gguf (147.5 GB, arch glm5next), RTX 5090 32GB,
venv /media/ai/src/FreeToken/.venv (python 3.14).

## 2. Process hygiene

Preflight: pgrep 'pytest|ft serve|benchbw|ft_run' - только собственный wrapper-shell;
sem.mp- в /dev/shm = 0; VRAM baseline 1515 MiB used / 30634 MiB free.

Runner: /tmp/ft_crash_repro/runner.sh - trap-on-EXIT cleanup + FAILPAT- watchdog
(опрос 5s) + hard timeout 900s + SIGTERM -> SIGKILL через 30s, рекурсивное убийство
дерева. FATALPAT = 'AssertionError|OutOfMemoryError|Backend worker is gone|
backend worker .* exited|Traceback|CUDA error'. Осознанное отклонение от протокола:
'not supported' записывался в status, но НЕ убивал процесс - это слово встречается в
benign capability-логах, немедленный kill уничтожил бы доказательства реального краша.

Postflight: процессов нет, sem.mp- = 0, VRAM 1422 MiB used / 30727 MiB free -
вернулся к baseline. Shutdown прошёл чисто (hang из известного паттерна
"backend death -> uvicorn waiting" не случился: воркер умер рано, supervisor
успел корректно зарипать обоих воркеров).

## 3. Момент падения

- Старт 13:47:20 (Parsed arguments -> uvicorn up), краш 13:47:27 = +7 c от старта
  (watchdog зафиксировал FATALPAT на +15 c из-за 5s каденции опроса).
- Фаза: Engine.__init__ (engine.py:365) -> _adjust_config (engine.py:1840) ->
  первое обращение к cached_property EngineConfig.model_config (engine/config.py:134).
  Это происходит ДО загрузки весов, ДО выделения VRAM под пулы и ДО любого
  cache-budget / 2E-overlap планирования.
- Умер воркер freetoken-TP0-scheduler; FrontendAPI: "Backend supervisor:
  FrozenInstanceError: cannot assign to field 'audio_config'"; сервер погасился
  сам, воркеры reaped pids=[360819, 360820], главный pid 360671.

## 4. Полный traceback

```text
  File "/home/linuxbrew/.linuxbrew/opt/python@3.14/lib/python3.14/multiprocessing/process.py", line 320, in _bootstrap
    self.run()
  File "/home/linuxbrew/.linuxbrew/opt/python@3.14/lib/python3.14/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/media/ai/src/FreeToken/python/freetoken/server/launch.py", line 91, in _run_scheduler
    scheduler = Scheduler(args)
  File "/media/ai/src/FreeToken/python/freetoken/scheduler/scheduler.py", line 68, in __init__
    self.engine = Engine(config)
  File "/media/ai/src/FreeToken/python/freetoken/engine/engine.py", line 365, in __init__
    _adjust_config(config)
  File "/media/ai/src/FreeToken/python/freetoken/engine/engine.py", line 1840, in _adjust_config
    model_config = config.model_config
  File "/home/linuxbrew/.linuxbrew/opt/python@3.14/lib/python3.14/functools.py", line 1126, in __get__
    val = self.func(instance)
  File "/media/ai/src/FreeToken/python/freetoken/engine/config.py", line 141, in model_config
    setattr(hf_config, key, None)
  File "<string>", line 20, in __setattr__
dataclasses.FrozenInstanceError: cannot assign to field 'audio_config'
```

## 5. Verdict

**MERGE REGRESSION.** Регрессия union-мержа f786d7c: код, слитый из main
(коммит 08d728d "feat(mm): serve image input on the Qwen families (#454)"),
мутирует hf_config, который для GGUF-моделей - замороженный dataclass.
Краш не зависит ни от одного флага пользователя: ломается ЛЮБОЙ запуск
`ft serve` с .gguf моделью на HEAD f786d7c (kv dtype / reserve / strategy /
memory-ratio не успевают сыграть никакую роль). Гипотезы (1) fail-fast
plan_cache_budget, (2) 2E prefill_overlap, (3) gguf+nvfp4, (6) JIT clang++
- не опровергнуты, но ненаблюдаемы за этим крашем; гипотеза (4) регрессия
union-мержа - подтверждена (падающий файл engine/config.py).

## 6. Root cause (file:line + цитаты)

Падающая цепочка:

1. `python/freetoken/engine/config.py:134-141` (cached_property model_config,
   код слит из main в f786d7c):

```python
hf_config = copy.copy(self.hf_config)
built = {e.config_key for e in self.active_encoders}
for key in set(ENCODER_SECTIONS) | {e.config_key for e in self.model_spec.encoders}:
    if key not in built:
        setattr(hf_config, key, None)   # <- line 141, FrozenInstanceError
```

2. `python/freetoken/mm/config.py:11` (файл новый, тоже из 08d728d):
   `ENCODER_SECTIONS = ("vision_config", "audio_config")`.

3. Для GGUF-пути `cached_load_hf_config` (python/freetoken/utils/hf.py:247-251)
   возвращает шим: `build_gguf_shim(gguf_src)` ->
   `GgufConfigShim` в `python/freetoken/models/gguf/config.py:25`:

```python
@dataclass(frozen=True)
class GgufConfigShim:
    architectures: list[str]
    model_path: str
    model_type: str
    metadata: dict[str, Any]
    vocab_size: int
    tie_word_embeddings: bool
```

Механика отказа:

- У шима нет полей vision_config/audio_config, поэтому
  `active_encoders` пуст (getattr(..., None) is None), и цикл пытается
  занулить обе секции.
- `copy.copy` замороженного dataclass-экземпляра остаётся экземпляром
  замороженного класса: frozen-ность живёт в сгенерированном `__setattr__`
  класса, а не в состоянии экземпляра.
- Сгенерированный dataclass'ом `__setattr__` (в traceback: File "<string>",
  line 20) бросает FrozenInstanceError БЕЗУСЛОВНО для любого setattr на
  экземпляре - в том числе для отсутствующих полей. Первый же setattr цикла
  (порядок set-итерации: audio_config) роняет воркер.

Почему падает именно GGUF и почему это регрессия мержа:

- `git diff 6e673ab f786d7c -- python/freetoken/engine/config.py`: до мержа
  `model_config` использовал `self.hf_config` напрямую и НИКОГДА не мутировал
  его; setattr-цикл, active_encoders, model_spec, served_modalities и mm:
  MultimodalConfig добавлены мержем (+31 строка). `python/freetoken/mm/config.py`
  - целиком новый файл (+33 строки).
- `python/freetoken/models/gguf/config.py` мержем НЕ тронут (отсутствует в
  diffstat) - frozen-шим существовал до мержа и работал.
- Коммит-источник в main: 08d728d "feat(mm): serve image input on the Qwen
  families (#454)" (второй родитель мержа afd99cb).
- Контраст: на HF-пути `cached_load_hf_config` возвращает мутабельный
  transformers PretrainedConfig или `RawConfigShim` (utils/hf.py:139, обычный
  класс, не frozen) - там setattr легален. Единственный замороженный
  hf_config в дереве - GgufConfigShim, поэтому регрессия избирательно бьёт
  по GGUF-serving (не покрытому тестами mm-волны из main).
- Почему ре-акцептанс 2026-09-15 на 63b9bff проходил: тот HEAD предшествует
  мержу, мутирующего кода ещё не было в дереве.

Пересчёт "какая конфигурация поместилась бы" - N/A: краш арифметики VRAM не
касается; plan_cache_budget не вызывался (до него не дошло).

## 7. Fix direction (без реализации)

Минимальный и семантически точный: в `EngineConfig.model_config` занулять
только те секции, которые конфиг вообще несёт:

```python
if key not in built and hasattr(hf_config, key):
    setattr(hf_config, key, None)
```

- Для HF-конфигов семантика main сохраняется: скрываются реально существующие,
  но не собираемые секции башен.
- Для GgufConfigShim это no-op: его parse_config читает metadata и в
  атрибутах vision_config/audio_config не нуждается (доказано успешным boot
  этого файла на 63b9bff). `active_encoders` уже использует такой же
  defensive `getattr(..., None)` - стиль совпадает.

Альтернативы: (a) try/except FrozenInstanceError вокруг setattr - грубее,
глотает реальные ошибки; (b) сделать GgufConfigShim мутабельным - шире blast
radius, теряет иммутабельность шима; (c) object.__setattr__ - обходит контракт
frozen, не рекомендуется.

Тест: регресс-тест разрешения конфига GGUF (EngineConfig.model_config на
GgufConfigShim не бросает; парсер получает shim без изменений) в tests/
рядом с существующими gguf-config тестами; плюс golden-тест, что HF-путь
по-прежнему зануляет unbuilt-секции (защита от недо-фикса).

## 8. Контрольный A/B на пре-мерж базе 6e673ab

Не проводился: traceback + git diff однозначно указывают на код, слитый из
main (setattr-цикл отсутствовал в 6e673ab; gguf-шим мержем не менялся).
По инструкции задачи статического diff-анализа достаточно; worktree-прогон
добавил бы только время.

## 9. Что осталось за кадром

После фикса этого краша известные из memory-банка риски именно запрошенной
комбинации (1M nvfp4 KV + гибридный min-plan против бюджета 0.89 и 2E
per-partition overlap инвариант gguf-партиций) могут проявиться следующим
шагом - они в этом прогоне ненаблюдаемы. Для GGUF-файла авторитетная
известно-рабочая команда (без nvfp4 KV): base + --num-tokens 70016
--memory-ratio 0.85 --max-prefill-length 4096 (см.
.tasks/gguf-glm5next-hybrid/PLAN.md, Lessons).

## Appendix: server.log verbatim

Полный лог прогона (42 строки) - /tmp/ft_crash_repro/server.log - дописан
ниже дословно (включая полный traceback и строку ServerArgs с kv_quant='nvfp4').

```text
[1m[2026-09-16|13:47:20][0m [32mINFO    [0m Parsed arguments:
ServerArgs(model_path='/media/ai/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL.gguf', tp_info=DistributedInfo(rank=0, size=1), dtype=torch.bfloat16, max_running_req=4, attention_backend='auto', moe_strategy='hybrid', quant_backend=None, ple_backend='disk', expert_load='auto', moe_cache_size=0, moe_cache_rate=None, moe_cache_auto=True, kv_reserve_tokens=1048576, moe_cache_policy='lru', moe_prefill_overlap=True, moe_prefill_hit_d2d=False, moe_collect_stats=False, moe_cpu_threads=16, moe_cpu_layers=None, moe_hybrid_max_fetch=-1, cuda_graph_bs=None, cuda_graph_max_bs=None, page_size=1, kv_quant='nvfp4', memory_ratio=0.89, linear_state_cache_ratio=2.0, swa_full_tokens_ratio=0.2, swa_num_pages_override=None, distributed_timeout=60.0, use_dummy_weight=False, use_pynccl=True, max_seq_len_override=None, num_page_override=None, num_token_override=None, mm=MultimodalConfig(disabled_encoders=frozenset(), embed_cache_device='cpu', encoder_weights='host', image_min_tokens=None, image_max_tokens=None, processor_kwargs={}), max_extend_tokens=4096, cache_type='radix', offline_mode=False, decode_log_interval=40, special_token_ckpt=False, _unique_suffix='.pid=360671', server_host='127.0.0.1', server_port=18081, num_tokenizer=0, silent_output=False, shell_mode=False, served_model_name='GLM-5.3-Flash-UD-Q3_K_XL.gguf', tool_call_parser='glm47', reasoning_parser='glm', sampling_defaults='model', max_output_tokens=None, enable_cache_report=False, allowed_media_domains='', allowed_local_media_path='', cors_origins='tauri://localhost,http://tauri.localhost,http://localhost:1420', gpu=(), gpu_assigned=None)
[1m[2026-09-16|13:47:20|FrontendAPI][0m [32mINFO    [0m Default sampling config (source=model): temperature=1.0, top_k=-1, top_p=0.949999988079071
INFO:     Started server process [360671]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:18081 (Press CTRL+C to quit)
/media/ai/src/FreeToken/python/freetoken/engine/engine.py:1499: FutureWarning: torch.cuda._set_allocator_settings is deprecated. Use torch._C._accelerator_setAllocatorSettings instead.
  torch.cuda.memory._set_allocator_settings("expandable_segments:True")
[1m[2026-09-16|13:47:22|core|rank=0][0m [32mINFO    [0m Enabled expandable_segments (override via PYTORCH_ALLOC_CONF)
Process freetoken-TP0-scheduler:
[1m[2026-09-16|13:47:27|FrontendAPI][0m [31mERROR   [0m Backend supervisor: FrozenInstanceError: cannot assign to field 'audio_config'
Traceback (most recent call last):
  File "/home/linuxbrew/.linuxbrew/opt/python@3.14/lib/python3.14/multiprocessing/process.py", line 320, in _bootstrap
    self.run()
    ~~~~~~~~^^
  File "/home/linuxbrew/.linuxbrew/opt/python@3.14/lib/python3.14/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
    ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/media/ai/src/FreeToken/python/freetoken/server/launch.py", line 91, in _run_scheduler
    scheduler = Scheduler(args)
  File "/media/ai/src/FreeToken/python/freetoken/scheduler/scheduler.py", line 68, in __init__
    self.engine = Engine(config)
                  ~~~~~~^^^^^^^^
  File "/media/ai/src/FreeToken/python/freetoken/engine/engine.py", line 365, in __init__
    _adjust_config(config)
    ~~~~~~~~~~~~~~^^^^^^^^
  File "/media/ai/src/FreeToken/python/freetoken/engine/engine.py", line 1840, in _adjust_config
    model_config = config.model_config
                   ^^^^^^^^^^^^^^^^^^^
  File "/home/linuxbrew/.linuxbrew/opt/python@3.14/lib/python3.14/functools.py", line 1126, in __get__
    val = self.func(instance)
  File "/media/ai/src/FreeToken/python/freetoken/engine/config.py", line 141, in model_config
    setattr(hf_config, key, None)
    ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^
  File "<string>", line 20, in __setattr__
dataclasses.FrozenInstanceError: cannot assign to field 'audio_config'
INFO:     Shutting down
INFO:     Waiting for application shutdown.
[1m[2026-09-16|13:47:29|FrontendAPI][0m [32mINFO    [0m Graceful stop: reaping 2 backend worker(s), pids=[360819, 360820]
INFO:     Application shutdown complete.
INFO:     Finished server process [360671]
```
