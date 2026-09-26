# REPORT: llama-swap stats для ft serve - hardware acceptance W1+W5

Дата: 2026-09-26. Волна: W5 (hardware acceptance) + W1 (конфиг llama-swap).
Кодовая волна W2+W3+W4 к этому моменту была завершена и ревью-верифицирована,
лежала в рабочем дереве dev-tree незакоммиченной. Бриф кампании:
[TASK.md](TASK.md).

Вердикт: W5 PASSED на обеих ft serve-секциях (GLM-5.3-Flash-gguf,
Qwen3.8-Flash-Next). W1 применена к живому конфигу. Одно отклонение по логам
(хвостовой неполный чанк префила) - см. [Deviations](#deviations).

## Окружение

- Rig: RTX 5090 32 GiB, VRAM свободна на старте волны (1662/32607 MiB,
  только hoptodesk). Облачная модель активна, локальной LLM в VRAM нет.
- Live-стек НЕ тронут процессуально: llama-swap v259, PID 1598375,
  `-listen 0.0.0.0:18080 -config /media/ai/soft/llama-swap/config.yaml
  -watch-config`, аптайм на момент проверки 3ч23м.
- Тест-инстанс: тот же бинарник `/media/ai/soft/llama-swap/llama-swap`,
  `-listen 127.0.0.1:18099`, конфиг
  [.tasks/ls-metrics-test/config.yaml] - копия живого
  с заменами (см. [Тест-инстанс](#тест-инстанс)).
- ft в тест-конфиге: DEV-venv `/media/ai/src/FreeToken/.venv/bin/ft`
  (W2/W3 код есть только в dev-дереве). Live-cmd остаётся на PROD-venv.
- Клиент: `.tasks/ls-metrics-test/client.py`, httpx,
  read timeout 1800 s (SDK-дефолт 600 s рвёт декоды - см. память
  ft-serve-600s-abort-client-timeout).
- Эндпоинт статистики v259 найден по исходникам тега v259 (GitHub) и JS
  вебуи: `GET /api/metrics/activity` (таблица Requests вебуи), агрегаты
  `GET /api/metrics/stats`. Разбор тела ответа - `internal/server/metrics.go`:
  `timings` имеет приоритет над `usage`; `prompt_per_second` ->
  колонка Prefill, `predicted_per_second` -> Decode, `cache_n` -> Cached,
  `Duration = max(wall, prompt_ms + predicted_ms)`, отсутствующие ключи ставок
  парсятся как 0.0.

## Тест-инстанс

Отличия тест-конфига от живого (полный diff - в артефактах
`artifacts/config-work.diff`):

1. `startPort: 18081 -> 18100`: живой инстанс раздаёт дочерним ft порты
   начиная с 18081, поэтому слушающий порт тест-инстанса взят 18099, а
   дочерний диапазон - 18100+, чтобы не пересекаться.
2. В обеих ft serve-секциях путь `ft` заменён на DEV-venv.
3. В обеих ft serve-секциях в `cmd` добавлен `--enable-cache-report`.
4. В `filters.setParamsByID` обеих секций для ВСЕХ вариантов ID добавлен
   `stream_options: {include_usage: true}` (GLM: 4 варианта, Qwen3.8-Flash-Next:
   4 варианта; состав вариантов перечислен из актуального конфига, не
   захардкожен).

Состав ft serve-секций в актуальном живом конфиге изменился относительно брифа:
теперь их ДВЕ (`GLM-5.3-Flash-gguf`, `Qwen3.8-Flash-Next`); секции
`Qwen3.6-35B-A3B`, `Qwen3.8-27B`, `gemma-4-E4B` перешли на llama-server и в
scope кампании не входят (правило брифа: секция = чей cmd использует ft serve).

Харнесс: census-gate перед стартом (`[f]t serve` + порт), FAILPAT-watchdog
(OutOfMemoryError/backend-death/panic -> `FAILPAT.hit`), teardown SIGTERM ->
SIGKILL, postflight census. Все скрипты в `.tasks/ls-metrics-test/`.

## W5 результаты: GLM-5.3-Flash-gguf

Бут через реальный chat-запрос (не /v1/models): **165 s** до первого 200
(полоса валидированного рецепта 52-130 s - чуть выше, health-check llama-swap
+ первый прогрев; см. Deviations). Признаки бута в логе: ServerArgs с
`enable_cache_report=True`, `cache_type='hybrid_radix'`, `page_size=64`.

Батарея (все ответы HTTP 200, модель `GLM-5.3-Flash-gguf`):

| Шаг | Вид | prompt_n | pred_n | cache_n | Prefill tok/s | Decode tok/s | wall, ms | Duration_ms |
|-----|-----|----------|--------|---------|---------------|--------------|----------|-------------|
| glm_boot | non-stream | 19 | 15 | 0 | (warmup, нет ключа) | 16.27 | 164649 | 164626 |
| glm_b_no_usage_flag | stream БЕЗ stream_options | 21 | 95 | 0 | 6.30 | 15.86 | 9358 | 9339 |
| glm_c_explicit_usage | stream, явный include_usage | 24 | 255 | 0 | 7.24 | 16.61 | 18691 | 18672 |
| glm_chunked | non-stream, 24013 токенов | 24013 | 15 | 0 | 806.63 | 17.14 | 30720 | 30700 |
| glm_d1 / d2 | stream, тот же промпт 21 ток | 21 | 47 | 0 / 0 | 6.36 / 6.39 | 16.01 / 15.37 | 6294 / 6364 | 6246 / 6348 |
| glm_d3 / d4 | stream, тот же промпт 833 ток | 833 | 47 | 0 / **832** | 204.97 / **0.27** | 15.02 / 15.48 | 7227 / 6782 | 7210 / 6765 |
| glm_e1 | completions non-stream | 8 | 47 | 0 | 2.46 | 18.04 | 5880 | 5863 |
| glm_e2 | completions stream | 8 | 47 | 0 | 2.47 | 17.02 | 6023 | 6006 |

Проверки по телам ответов:

- Блок `timings` присутствует во всех ответах; у не-warmup запросов - все
  7 ключей (prompt_n, prompt_ms, predicted_n, predicted_ms,
  prompt_per_second, predicted_per_second, cache_n). У warmup-запроса
  (glm_boot) `prompt_ms=0.0` и ключ `prompt_per_second` корректно опущен
  (нулевой знаменатель; llama-swap парсит как 0.0).
- Инъекция setParamsByID работает: в шаге b клиент НЕ просил stream_options,
  финальный usage-чанк с usage+timings в стриме присутствует.
- `predicted_ms` исключает TTFT: у glm_c wall-ttft = 15355.6 ms против
  predicted_ms 15349.89; у glm_d4 3039 против 3036.67.
- Кеш-хит: d4 (повтор d3) - `cache_n=832` из 833, в usage появился
  `prompt_tokens_details.cached_tokens: 832` (--enable-cache-report работает).
  `prompt_per_second=0.27` = 1 новый токен / 3.72 s - ставка считается по
  НОВЫМ токенам, как зафиксировано в брифе.
- Важно: повторы d1/d2 (21 и 8 токенов) дали cache_n=0 - это честный ноль
  движка, а не баг проводки: page_size=64, промпт короче страницы радикс не
  удерживает. Лог подтверждает: во всех префилах `#cached-token: 0` ровно до
  d4, у d4 кеш-матч есть.

### Сверка с логами ft serve (W3)

- Warmup помечен: во всём логе ровно ДВЕ строки с `, warmup: true` - по одной
  на бут каждой секции; warmup-запрос получил prompt_ms=0 (исключён из rate),
  в логе его честная скорость 0.36 tok/s.
- Последний ПОЛНЫЙ чанк префила честный: у glm_chunked чанки 8128/8128/7757
  дали 415.39 (первый, холодный: MoE-fetch) / **834.29** / 16851.49 tok/s.
  Старый артефакт ~1552-1602 tok/s на полном чанке НЕ воспроизвёлся.
  Трейлинг-артефакт на ХВОСТОВОМ (неполном) чанке остался - Deviations D4.
- Сумма measured-окон = запрос-левел честности: prompt_ms 29769.69 +
  predicted_ms 875.12 = 30644.81 ms против wall 30719.6 (расхождение 75 ms -
  HTTP/токенизация). Запрос-левел ставка 806.63 tok/s попадает в валидированную
  полосу рецепта (~800-805).
- Decode-окна без выбросов после префила: 15.25 -> 15.28 (GLM),
  40.69 -> 47.70 -> 47.75 (Qwen) - gap-выброс первого окна не наблюдается.

### Колонки llama-swap (api/metrics/activity, тест-инстанс)

Все 10 GLM-строк заполнены полностью, примеры (id = строка activity):

- id 4 (glm_chunked): Prompt 24013, Generated 15, Cached 0, Prefill 806.63,
  Decode 17.14, Duration 30700 (wall 30719.6, расхождение 20 ms).
- id 10 (d4): Prompt 833, Generated 47, Cached 832, Prefill 0.27, Decode 15.48,
  Duration 6765 (wall 6782, расхождение 17 ms).
- id 1 (boot): Duration 164626 = wall (сумма timings 921.87 < wall; v259 берёт
  max - корректно).

## W5 результаты: Qwen3.8-Flash-Next

Бут: **29 s** (тёплый tier-dir). Секция сессионного яруса активна
(session-tier-*), в логе `probes=0/N`, offers 0 - повторов в батарее не было
(минимальный объём по ТЗ: шаги a + c).

| Шаг | Вид | prompt_n | pred_n | Prefill tok/s | Decode tok/s | wall, ms | Duration_ms |
|-----|-----|----------|--------|---------------|--------------|----------|-------------|
| qwen_boot | non-stream | 19 | 2 | (warmup) | 42.49 | 29067 | 29050 |
| qwen_a_nonstream | non-stream | 19 | 2 | 12.37 | 66.86 | 1587 | 1570 |
| qwen_c_explicit_usage | stream, явный include_usage | 24 | 85 | 13.69 | 46.41 | 3607 | 3590 |
| qwen_b_no_usage_flag | stream БЕЗ stream_options | 21 | 58 | 13.89 | 50.37 | 2684 | 2668 |

Все timings-ключи на месте, инъекция для Qwen-вариантов работает, Duration
сходится с wall (расхождение 16-40 ms), decode 46-67 tok/s правдоподобен
для Qwen3.8-Flash-Next NVFP4. Колонки activity (id 11-14) заполнены полностью.

## Teardown тест-инстанса

SIGTERM -> уход за ~2 s -> postflight: нет `ft serve`/llama-swap процессов,
VRAM 1654 MiB (baseline), `sem.mp` census 3 (чисто). Артефакты сессии:
`.tasks/ls-metrics-test/artifacts/` (SSE-тела, JSON-ответы, сводки,
llamaswap_instance.log, FAILPAT-статусы).

## W1: живой конфиг

1. Проверка PROD-бинарника (предусловие): `/media/ai/src/FreeTokenProd/.venv/
   bin/ft serve --help` содержит `--enable-cache-report` ("Return the number
   of prefix-cached prompt tokens in each response's usage block") - флаг
   добавлять в live-cmd МОЖНО.
2. Патч применён скриптом (идемпотентным):
   `.tasks/ls-metrics-test/make_work_config.py live`. Бэкап до правки:
   `/media/ai/soft/llama-swap/config.yaml.bak-w1-20260926`. Итоговый diff
   (backup -> live): 8 блоков `stream_options/include_usage` (4 варианта GLM +
   4 варианта Qwen3.8-Flash-Next) + 2 строки `--enable-cache-report` в cmd ft
   serve-секций. PROD-путь ft не тронут, startPort не тронут, llama.cpp-секции
   (Qwen3.6-35B-A3B, Qwen3.8-27B, gemma-4-E4B) байт-в-байт не изменились.
3. Горячая перезагрузка живого инстанса: PID 1598375 жив (аптайм без рестарта),
   `/v1/models` отдаёт все 13 моделей. Stdout живого процесса - pipe, его лог
   недоступен из сессии, поэтому строку reload-а подтвердить нечем; структуру
   же конфига принял тот же бинарь v259 на тест-инстансе (identical schema).
   Бутов моделей на живом инстансе не делал (VRAM protocol) - первая же
   реальная стрим-заявка пользователя должна дать заполненные Prompt/
   Generated/Cached; если этого не случится - смотреть лог live-инстанса.

## Deviations

- D1. Состав секций: бриф и ТЗ исходили из 4 ft serve-секций; в актуальном
  живом конфиге их 2 (Qwen3.6/Qwen3.8-27B/gemma ушли на llama-server).
  W1 применена к фактическим ft-секциям; шаг ТЗ "по трём Qwen-секциям"
  выродился в одну Qwen3.8-Flash-Next. Для Qwen добавлены ВСЕ 4 варианта ID
  (ТЗ говорило "base variant only" из устаревшего состава секций; бриф -
  контракт - требует все варианты секции; иначе think-варианты остались бы
  без статистики).
- D2. Порт тест-инстанса 18099 (ТЗ предлагало 18081) + startPort 18100 в
  тест-конфиге: живой инстанс уже раздаёт с 18081.
- D3. GLM-бут 165 s - выше валидированной полосы 52-130 s. На результат не
  влияет (healthCheckTimeout 500), вероятная причина - холодный page cache
  checkpoint-файлов.
- D4. Хвостовой НЕполный чанк мультичанкового префила в ЛОГЕ всё ещё завышен
  (16851.49 tok/s на 7757 токенов). Механизм: overlap-планировщик + lag дрейна
  схлопывают own-window последнего чанка; сумма окон при этом честна
  (= wall - 75 ms), запрос-левел ставка попадает в валидированную полосу.
  Критерий буквы брифа ("последний ПОЛНЫЙ чанк честен") выполнен: 834.29.
  Рекомендация: пер-чанковые ставки лога на hybrid-offload конвейере -
  окна пропускной способности конвейера, а не изолированные стоимости батчей;
  для сравнений использовать запрос-левел timings (как это и делает
  llama-swap).
- D5. cache_n>0 продемонстрирован на промпте 833 токена; более короткие
  повторы (21/8 токенов) дают честный ноль из-за page_size=64.
- D6. Пункт брифа W5 про Anthropic-клиента (input_tokens после
  --enable-cache-report) не выполнялся: Anthropic-клиента у секций в этой
  волне не было.
- D7. Полный pytest-gate в этой волне не прогонялся (не входит в W1+W5);
  прогнаны 4 файла кампании W4: 70 passed за 2.52 s.
- D8. Полный стакан статистики в вебуи ЖИВОГО инстанса не проверялся бутом
  модели (VRAM protocol, "не бутать на live") - проверен на тест-инстансе с
  тем же бинарём v259 и той же схемой конфига.

## Acceptance (по TASK.md)

- Выполнено и тикнуто: 8 колонок у обеих ft-секций (на тест-инстансе v259);
  Prefill/Decode без warmup/last-chunk артефактов на уровне значений API и
  полного чанка лога (оговорка D4); логи: последний полный чанк честный,
  warmup помечен и исключён из rate; REPORT.md + индекс обновлены.
- Не тикнуто: "полный gate зелёный" (не прогонялся, D7); поведение
  Anthropic-клиента (D6).

## Артефакты

- `.tasks/ls-metrics-test/` - make_work_config.py, client.py,
  verify_activity.py, start_instance.sh, stop_instance.sh,
  failpat_watchdog.sh, config.yaml (work), артефакты ответов в artifacts/.
- Бэкап живого конфига: `/media/ai/soft/llama-swap/config.yaml.bak-w1-20260926`.
