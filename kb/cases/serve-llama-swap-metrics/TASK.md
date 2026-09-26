# Task: полная статистика запросов llama-swap для ft serve (Prompt/Generated/Cached/Prefill/Decode)

Status: W1+W5 ACCEPTED HW (2026-09-26, [REPORT.md](REPORT.md)); кодовая волна
W2+W3+W4 завершена и ревью-верифицирована, лежит в рабочем дереве
незакоммиченной. Исследование проведено в сессии 2026-09-26; бриф
фиксирует вердикты с якорями на код.

## Goal

В webui llama-swap для КАЖДОЙ секции с бэкендом `ft serve` все восемь колонок
заполняются корректными значениями:
Time / Status / Cached / Prompt / Generated / Prefill / Decode / Duration.
Цель покрывает секции `GLM-5.3-Flash-gguf`, `Qwen3.6-35B-A3B`,
`Qwen3.8-27B`, `Qwen3.8-Flash-Next`, а также любую будущую ft serve-секцию
(правило: любая секция, чей `cmd` использует `ft serve`).
Значения Prefill и Decode не содержат известных артефактов ft serve логов
(заниженный warmup-чанк в начале, завышенный последний чанк в конце), и логи
ft serve тоже показывают реальную скорость без этих искажений.

Для llama.cpp-секции (`GLM-5.3-Flash`) статистика уже отображается - там
llama-server сам дописывает в конец стрима финальный чанк с `usage` и `timings`.
Эталон - поведение llama.cpp; ft serve должен давать llama-swap те же данные.

## Контекст (верифицировано в сессии-исследовании, НЕ переоткрывать)

Живая конфигурация и процесс (верифицировано):

- Конфиг: `/media/ai/soft/llama-swap/config.yaml`; процесс: llama-swap v259,
  запущен как `./llama-swap -listen 0.0.0.0:18080 -config
  /media/ai/soft/llama-swap/config.yaml -watch-config`. Правки конфига
  подхватываются горячо через `-watch-config`, но проверка результата
  требует загруженной модели (VRAM protocol).
- `cmd` ft serve-секций в конфиге указывает на PROD-venv
  (`/media/ai/src/FreeTokenProd/.venv/bin/ft serve`), поэтому верификация
  W2/W3 должна явно поднимать сервер из dev-venv.

Как llama-swap получает статистику (бинарник пользователя: v259, сборка
2026-09-26; локальный чекаут /media/ai/src/llama-swap устарел и НЕ соответствует
запущенному бинарнику):

- llama-swap парсит ТЕЛО каждого проксированного ответа (metrics-монитор).
  В v259 это `internal/server/metrics.go`: для SSE-стримов сканируются чанки
  `data:` на объекты `usage`, `timings`, `metrics`; для не-стрима - корень JSON.
- `usage` (OpenAI/TabbyAPI/Anthropic-поля): Prompt = `prompt_tokens` /
  `input_tokens`, Generated = `completion_tokens` / `output_tokens`, Cached =
  `prompt_tokens_details.cached_tokens` / `input_tokens_details.cached_tokens` /
  `cache_read_input_tokens`.
- `timings` (llama.cpp-формат): `prompt_n`, `prompt_ms`, `predicted_n`,
  `predicted_ms`, `prompt_per_second`, `predicted_per_second`, `cache_n`,
  `draft_n`/`draft_n_accepted`. Имеет ПРИОРИТЕТ над `usage`; Duration при
  наличии timings = prompt_ms + predicted_ms.
- `metrics` (vLLM-формат): `time_to_first_token_ms`, `mean_itl_ms`,
  `tokens_per_second`, `speculative_decoding.*`.
- Prefill/Decode считаются ТОЛЬКО из timings / ставок TabbyAPI в usage /
  metrics. Если ничего из этого нет - колонки остаются пустыми. Wall-clock
  fallback в activity-логе отсутствует (он есть только в playground UI).
- Фильтр `setParamsByID` применяется к телу запроса через `sjson.SetBytes`
  (в устаревшем чекауте: proxy/proxymanager.go:732; в v259 то же поведение),
  вложенные JSON-объекты поддерживаются. llama-swap НЕ внедряет
  `stream_options.include_usage` в запросы к бэкенду автоматически.

Почему у ft serve колонок нет (якоря текущего кода FreeToken):

- `python/freetoken/server/openai_api.py`: usage-чанк в стриме эмитится только
  при клиентском `stream_options.include_usage` (chat: строки ~371-388;
  completions: ~503-517). Все клиенты пользователя стримят и НЕ просят usage.
- `--enable-cache-report` выключен по умолчанию (`server/args.py`:
  `enable_cache_report: bool = False`); без него `_reported_cached` гасит
  cached_tokens. При нулевом попадании `prompt_tokens_details` не сериализуется
  (sglang-конвенция) - llama-swap покажет "-" для Cached.
- Блока `timings`, ставок в `usage` и блока `metrics` ft serve не отдаёт
  НИГДЕ - поэтому Prefill/Decode невозможно получить любой конфигурацией
  llama-swap; нужно изменение FreeToken.
- Duration сейчас корректен (wall-clock прокси) и после появления timings
  станет точнее (prompt_ms + predicted_ms, без очереди/сети).

Артефакты измерений ft serve, которые НЕЛЬЗЯ воспроизвести в новых значениях:

1. Первый префил-чанк после старта движка занижен (JIT/autotune warmup).
   Правило сравнения логов: учитывать только steady-state чанки.
2. Последний ПОЛНЫЙ чанк префила завышен (~1552-1602 tok/s при 8128 токенов -
   аппаратно невозможно). Причина: пер-чанковая скорость считается как разность
   между временами ОТЧЁТОВ (`scheduler/status.py:_report_prefill`, поле gap),
   и отчёт последнего чанка эмитится, пока хвост ещё сливается. Правило:
   скорость префила = медиана полных чанков МИНУС последний.
   Root cause и метод: см. статью kb про prefill-throughput (kb/topics/) и
   эталонный отчёт в kb/cases/mmq-prefill-kernel/.
3. TTFT на полностью кешированном префиксе ~6 с - это overhead первого
   декод-шага (CUDA graph/sync), НЕ перезапуск префила. В знаменатель
   Decode-скорости TTFT включать нельзя.

## Work items (по зависимостям)

W2 и W3 - изменения уровня движка ft serve (usage/timings/логи эмитятся
независимо от модели), поэтому они автоматически действуют на каждую
модель; пер-модельной работы там нет.

### W1. Конфиг llama-swap (без изменений кода; даёт Prompt/Generated/Cached)

- Для КАЖДОЙ ft serve-секции (`GLM-5.3-Flash-gguf`, `Qwen3.6-35B-A3B`,
  `Qwen3.8-27B`, `Qwen3.8-Flash-Next`) в `filters.setParamsByID` добавить
  ключ `stream_options: {"include_usage": true}` для ВСЕХ вариантов ID этой
  секции: setParamsByID матчится по запрошенному ID точно, вариант =
  отдельная запись. У не-GLM секций вариантов `:think-*` нет - там
  достаточно базового `${MODEL_ID}`, если секция не определяет свои
  варианты. Состав ID-вариантов НЕ хардкодить: при исполнении перечислить
  фактические варианты каждой секции из актуального конфига.
- В `cmd` каждой ft serve-секции добавить `--enable-cache-report`.
- Проверить на реальном запросе: Prompt/Generated заполнились у стрим-запросов;
  повторный запрос с тем же префиксом даёт Cached > 0.
- Известные следствия (проверить, не баги ли для конкретных клиентов):
  финальный чанк с пустым `choices` (OpenAI-спека; нестандартные клиенты);
  на /v1/messages `--enable-cache-report` меняет семантику input_tokens
  (исключает кешированный префикс, Anthropic billing semantics) - проверить,
  есть ли Anthropic-клиенты у этой секции.
- Нестрим-запросы: stream_options ft serve игнорирует; usage в не-стриме
  ft отдаёт и так (openai_api.py ~232, ~456).

### W2. Отдача таймингов ft serve (без этого Prefill/Decode не появятся)

- Первым делом протрассировать цепочку ack: где scheduler знает факт начала/
  конца префила и декода конкретного запроса, и как это доезжает до
  GenDone/ack (`server/generation.py`, канал wait_for_ack). В ack уже есть
  prompt_tokens / completion_tokens / cached_tokens - тайминги добавить в тот
  же канал. ЭТА ЦЕПОЧКА В ИССЛЕДОВАНИИ НЕ ПРОТРАССИРОВАНА - начать с неё.
- Формат вывода: блок `timings` в llama.cpp-совместимом формате в терминальном
  usage-чанке стрима и в не-стримовом JSON: `prompt_n`, `prompt_ms`,
  `predicted_n`, `predicted_ms`, `prompt_per_second`, `predicted_per_second`,
  `cache_n`. Обоснование выбора против альтернатив (ставки TabbyAPI в usage,
  блок metrics в vLLM-стиле): llama-swap v259 даёт timings наивысший приоритет,
  формат совместим ещё и с llama.cpp-экосистемой, cache_n закрывает Cached
  единым механизмом.
- API-поверхность: /v1/chat/completions и /v1/completions (`server/openai_api.py`,
  оба usage-чанка + не-стрим). Решения по /v1/messages (`server/anthropic_api.py`,
  usage в message_start/message_delta, см. комментарий про "usage-chunk
  dependency") и /v1/responses (`server/responses_api.py`) - минимум:
  не ломать; желательный минимум - ставки там, где usage уже эмитится всегда.
- Семантика значений (фиксируется здесь, чтобы не расползлась):
  - prompt_n = полные prompt-токены запроса; cache_n = токены, взятые из кеша;
  - prompt_ms = суммарное время ВЫЧИСЛЕНИЯ forward-батчей префила ЭТОГО
    запроса (аккумуляция вокруг forward), НЕ разность времён отчётов;
  - prompt_per_second = (prompt_n - cache_n) / prompt_ms - по НОВЫМ токенам,
    т.к. llama-swap использует это значение как есть;
  - predicted_ms = от первого декод-шага до последнего (TTFT и первый шаг
    не входят); predicted_per_second = predicted_n / predicted_ms;
  - батч warmup (первый префил-батч после старта движка) исключается из
    per-request rate; политика пометки (флаг батча в логах + пометка в
    статистике) - решить при реализации, но НЕ выкидывать батч из вычислений
    самого запроса незаметно для лога.
- Не-стрим и стрим должны давать согласующиеся значения для одного промпта.

### W3. Честные логи ft serve (user: "в идеале в логах тоже реальная скорость")

- `_report_prefill` (`scheduler/status.py:83-95`): считать input throughput
  из измеренной длительности forward батча, а не из gap между отчётами.
  Это устраняет и last-chunk артефакт (отчёт эмитится до окончания слива
  хвоста), и warmup-искажение (см. семантику в W2).
- `_report_decode` (`scheduler/status.py:119-129`): проверить, что gap-метод
  по окнам decode_log_interval не даёт выбросов на первом окне после
  префила/перехода; если да - выровнять семантику с W2.
- Не менять формат строк логов, на которые завязаны харнессы сравнения
  (медиана чанков в kb/methods/): артефакт должен ИСЧЕЗНУТЬ, а не сменить
  место в строке.

### W4. Тесты

- Формат timings: юнит-тест на сериализацию usage+timings чанка (стрим и
  не-стрим), включая cached=0 (prompt_tokens_details отсутствует, timings
  присутствует) и кейс с --enable-cache-report.
- Фикстурная дисциплина: аналитические фикстуры и сидированные толерансы
  (методика: kb, трап "unseeded = latent flakes").
- Тест лога: последний полный чанк префила больше не даёт завышенной скорости
  (регрессия на root cause gap-отчёта).
- Полный gate: uv run --extra dev pytest tests/ -m "not slow" с поправкой на
  известный baseline (классы известных падений; см. память ft-pytest-
  worktree-baseline-gotchas и ft-gate-nodeid-collection-drift).

### W5. Hardware acceptance (требует освобождения VRAM - VRAM protocol)

- Бут одной ft serve-секции (например `GLM-5.3-Flash-gguf`) с новым флагом;
  остальные секции - точечным реальным запросом. Готовность проверять
  реальным запросом, НЕ /v1/models (см. kb/methods/, A/B protocol).
- Через llama-swap: все 8 колонок заполнены; повторный запрос с тем же
  префиксом: Cached > 0 и Prefill заметно выше (кеш-хит).
- Сверка с логами: per-request prompt_per_second из API против медианы
  полных чанков из лога (минус последний) - совпадение в пределах погрешности;
  последний полный чанк в логе без "разгона" до ~1600 tok/s.
- Anthropic-клиент (если используется): сверить input_tokens после
  --enable-cache-report.
- Обновить: kb/cases/<этот каталог>/REPORT.md, запись в kb/cases/README.md,
  статью kb про prefill-throughput (если методика логов изменилась).

## Acceptance

- [x] В webui llama-swap для каждой ft serve-секции заполнены все
      8 колонок (в т.ч. Prefill/Decode/Cached у стрим-запросов). Верифицировано
      на тест-инстансе v259 (:18099, dev-venv) для фактических ft-секций
      `GLM-5.3-Flash-gguf` (10 строк) и `Qwen3.8-Flash-Next` (4 строки);
      состав ft-секций в живом конфиге изменился - Qwen3.6-35B-A3B и
      Qwen3.8-27B ушли на llama.cpp (REPORT.md, D1). Live-инстанс не бутали
      (VRAM protocol) - D8.
- [x] Значения Prefill/Decode согласуются с логами (сверка из W5) и не
      содержат warmup/last-chunk артефактов. На уровне значений API - да
      (806.63 tok/s на 24013 токенов; сумма окон = wall - 75 ms). Оговорка:
      хвостовой НЕполный чанк в логе всё ещё завышен (D4).
- [x] Логи ft serve: последний полный чанк префила показывает честную скорость
      (834.29 tok/s; старые ~1552-1602 не воспроизвелись);
      warmup-чанк помечен (`, warmup: true`, ровно 2 на 2 бута) и исключён
      из rate (prompt_ms=0 у warmup-запроса).
- [x] W1 не меняет поведение llama.cpp-секции; W2-W3 покрыты тестами;
      полный gate зелёный относительно известного baseline.
      (W4-файлы кампании: 70 passed; полный gate в волне W1+W5 не прогонялся;
      llama.cpp-секции в конфиге не тронуты - diff в REPORT.md.)
      (2325 passed, 16 fails = documented box baseline classes:
      import-mode ModuleNotFoundError + hardware, failure set
      byte-identical to 2026-09-19 baseline; 1 collection-excluded file
      tests/models/test_glm5_next_kda_snapshot.py)
- [x] REPORT.md + индексные записи kb обновлены.

## Deviations (кратко; полностью - REPORT.md)

- D1: ft serve-секций в живом конфиге теперь 2 (GLM-5.3-Flash-gguf,
  Qwen3.8-Flash-Next); Qwen3.6-35B-A3B / Qwen3.8-27B / gemma-4-E4B -
  llama.cpp. W1 применена к фактическим секциям, всем 8 вариантам ID.
- D2: тест-инстанс на :18099, startPort тест-конфига 18100 (живой инстанс
  сам раздаёт дочерние порты с 18081).
- D3: GLM-бут 165 s (выше полосы 52-130 s, на результат не влияет).
- D4: хвостовой неполный чанк мультичанкового префила в логе завышен
  (16851.49 tok/s на 7757 токенов): overlap + lag дрейна схлопывают
  own-window; сумма окон честна (= wall - 75 ms), запрос-левел ставка
  806.63 tok/s в валидированной полосе. Пер-чанковые ставки лога на
  hybrid-offload - окна конвейера, не изолированные стоимости.
- D5: cache_n>0 показан на повторе 833 токенов; повторы короче страницы
  (page_size=64) дают честный ноль.
- D6: Anthropic-проверка input_tokens не выполнялась (клиента нет).
- D7: полный pytest-gate в этой волне не прогонялся.
- D8: вебуи живого инстанса не проверялась бутом модели (VRAM protocol);
  проверен тест-инстанс с тем же бинарём v259.

## Ограничения и предупреждения

- В исследовании НЕ протрассирована цепочка ack/GenDone - W2 начинается с
  трассировки, якоря могут сместиться.
- Локальный чекаут /media/ai/src/llama-swap (ветка model_listing, старая
  структура proxy/) НЕ соответствует запущенному v259 (internal/server/):
  не сверять поведение по нему, только по GitHub-исходникам тега v259.
- Пользователь работает через разные клиенты, все - streaming через
  llama-swap; изменение финального чанка (пустой choices + usage) должно быть
  проверено минимум на двух клиентах.
- VRAM protocol: бут-проверки только после того, как пользователь переключился
  на облачную модель; один сервер за раз; регистрация смерти бэкенда в
  раннере обязательна (см. kb/methods/).