# TASK: round-5 - порты D1-D5: восстановление функциональности после ре-платформы ggml

Status: DONE 2026-10-21 - все пять фаз закрыты. Фаза 1 (D4) готова 2026-10-21
(коммит 9314c9b: boot-smoke PASS, decode медиана 21.00 tok/s, кросс-валидация
38/38); фаза 2 (D3) готова 2026-10-21 (коммит 393b682: 786k гибридный рецепт бут
x2 без лестницы, гибрид 27.71 tok/s против offload 21.00 на том же стеке, Fix-3
e2e 5568 cached / 0 полных промахов); фаза 3 (D1) закрыта NO-GO 2026-10-21 (все
4 веточных выигрыша ядер перекрыты ggml нативно или опровергнуты замерами:
[round5-d1-kernel-wins-no-go](../../topics/round5-d1-kernel-wins-no-go.md));
фаза 4 (D2) готова 2026-10-21 (коммит 7bc68b0: verify 347/347 exact, e2e-картинка
PASS, boot --mmproj READY 142 с; см. раздел 6); фаза 5 (D5) готова 2026-10-21
(коммит 660c814: -p no:skipping 8 failed -> 36 passed, A/B Qwen3.8 3.46x без
деградации, greedy 11/12 byte-identical; см. раздел 7).
Контекст слияния и карта портов -
[round5-ggml-replatform-deferred-ports](../../topics/round5-ggml-replatform-deferred-ports.md);
процедура интеграционных волн -
[merge-wave-procedure](../../topics/merge-wave-procedure.md). Якоря
файлов/символов/строк проверены против f0eed55 (vektory79); строчные якоря
дрейфуют - перед правками перепроверить. Каждый порт - отдельная кампания со
своими волнами, приёмкой и серией коммитов; push - только по явной команде
пользователя.

## 0. Контекст: что произошло

- 2026-10-17 выполнен round-5 merge: upstream/main 11c1adb -> vektory79,
  merge-коммит f0eed55 (parents b8c3794 + 11c1adb; следом docs-коммит
  8a62625). Режим: "минимальный зелёный merge" по решению пользователя -
  merge-коммит даёт зелёные сборку+тесты на новых механиках main, всё
  остальное отложено в порты D1-D5 (карта:
  [round5-ggml-replatform-deferred-ports](../../topics/round5-ggml-replatform-deferred-ports.md)).
- Backup-ветка со СТАРЫМ стеком (все выигрыши ядер, старый загрузчик gguf,
  старый offload-кэш): `backup/vektory79-pre-merge5-b8c3794`. Это основной
  источник для портов: git show/diff против неё.
- Что main сделал (коммиты #690 e5d3bd3, #689/#700, #670/#662/#704):
  * удалил kernel/csrc/gguf/ (moe.cuh, mmq.cuh, mmvq.cuh, moe_vec.cuh,
    vecdotq.cuh, dequantize.cuh, dispatch.h, ggml-common.h, gguf_kernel.cu)
    и python-клей: kernel/gguf.py, layers/gguf.py, models/gguf/dequant.py,
    moe/fused_q4_0.py;
  * вендорил ggml/ggml-cuda + ggml-cpu; появились пакеты
    layers/quantization/{configs,linear,moe,embedding}/, kernel/ggml.py +
    kernel/ggml_cpu/, models/gguf/{experts,tensors}.py;
  * offload-кэш экспертов переведён на per-format slot pools (новые ручки
    EngineConfig: moe_cache_mem / moe_cache_scope / moe_cache_split;
    layer_slot_bytes в engine/cache_budget.py);
  * CPU-MoE executor переписан (kernel/csrc/cpu_moe/cpu_moe_ext.cpp +
    moe/cpu_executor.py; один пул resolve_threads_and_affinity, бюджет
    потоков глобальный, не per-partition);
  * deps: torch 2.14.1, triton 3.8, sglang-kernel 0.4.9, flashlib 0.4.0,
    transformers 5.18;
  * новые архитектуры GGUF: DeepSeek V4/V4.1, линейка Qwen GGUF
    (2/3/3.5/3-MoE/4_exp), Gemma 4, MiniMax-M2/M3, Muse-Glimmer.
- Что ПЕРЕЖИЛО merge (восстанавливать НЕ надо): tier-хранилище сессий
  (SessionTierStore) и QSA/KpoolDSA кодеки, L2 extent dedup, FTW-волны
  P1-P13, kv_quant гейты, mmproj-kwargs, gguf strategy gates, ftw
  kind-None guard, _cpu_moe_executor_viable, vision-реестры в
  models/gguf/config.py (сам vision-путь - нет, это порт D2). W1-скаффолд
  спек-декода никогда не был закоммичен - он не потерян, его нет
  (кейс: [glm53flash-spec-decode](../glm53flash-spec-decode/TASK.md)).
- Тест-базлайн merged-дерева: ~2441 passed / 6 failed (все известные
  baseline/latent классы) / ~333 skipped (skip-марки называют отложенный
  порт). 6 известных падений: 2x glm_dsa OutOfResources, qwen4_ple bitwise
  (tests/models/qwen4_exp/test_ple.py, ~строка 421), pinned_tensor UVA,
  qsa_fp8 флак, offload FakeGraph raw_cuda_graph_exec (upstream-латентный,
  тест+стаб байт-идентичны main). Это фон: новые волны сравнивать по
  FAILURE SET против него, а не по абсолютным числам.

## 1. Опорные точки и базлайны

- Производительность ДО мерджа (цели для сравнения, не для слепого
  воспроизведения - числа надо заново мерить на новом стеке):
  * hybrid glm5next-GGUF: 16.67 tok/s (эталон 63b9bff);
  * offload: 12.97 tok/s;
  * round-4 boot-smoke (offload, kv-reserve 786k): /ready 200,
    decode 18.67 tok/s;
  * round-5 smoke (offload, kv-reserve 262k): /ready 200 за 87 с,
    11.21 tok/s - это базлайн УЖЕ нового стека.
- Рабочий boot-smoke рецепт на merged-дереве (проверен; порт 18081,
  т.к. дефолтный 18080 бывает занят; модель GLM-5.3-Flash-NVFP4-FTW):

  ```
  .venv/bin/ft serve --moe-strategy offload --moe-cache-auto \
    --disable-moe-prefill-overlap --kv-reserve-tokens 262144 \
    --kv-cache-dtype nvfp4 --memory-ratio 0.85 --max-prefill-length 4096 \
    --max-running-requests 1
  ```

- Старый 786k гибридный рецепт (цель после фаз 1-2):
  `--moe-cache-auto --kv-reserve-tokens 786432 --kv-cache-dtype nvfp4
  --memory-ratio 0.89 --max-prefill-length 4096 --moe-strategy hybrid
  --moe-cpu-threads 16`. ВНИМАНИЕ: на merged-дереве он падает на новом
  cache-budget assert из main - это арифметика конфига, ожидаемо требует
  пересчёта под slot pools (фаза 2), а не чинения assert-а.
- Готовность сервера проверять реальным запросом, а не дешёвым эндпоинтом:
  /v1/models отвечает до того, как движок дочитал веса (паттерн
  wait-for-ready с реальным chat-запросом - в старых runner-скриптах
  кампаний ft serve).

## 2. Правила исполнения (обязательны для каждой фазы)

- VRAM-протокол: GPU-волны (boot-smoke, CUDA-тесты, бенчи) - только в
  cloud-LLM окне; попросить пользователя освободить VRAM; сериализованно,
  по одному серверу/прогону. CPU-only работа (юнит-тесты без VRAM, разбор
  логов, статанализ) - в любое время.
- Process-hygiene: census до/после каждого прогона
  (`pgrep -af 'pytest|ft serve|benchbw|ft_run'` с фильтром собственной
  обёртки; `ls /dev/shm | grep -c "^sem\.mp-"`); runner-скрипты с
  trap-on-EXIT + FAILPAT-часовой

  ```
  AssertionError|OutOfMemoryError|Backend worker is gone|backend worker .* exited|Traceback|CUDA error|ValueError
  ```

  hard timeout на каждый прогон; SIGTERM -> 30 с -> SIGKILL; чужие процессы
  не убивать; stdout runner-а ОБЯЗАТЕЛЬНО tee в артефакт.
- Тесты: ТОЛЬКО `uv run --extra dev python -m pytest` (консольный pytest
  ломает import-mode); перед доверием результатам проверить
  `freetoken.__file__` (uv editable-ловушка); сравнивать по FAILURE SET,
  а не по числу собранных тестов (collection drift в test_quant_config.py
  от glob-параметризации на этапе сбора).
- Git: коммитить только по явным путям; перед коммитом сверять
  `git diff --cached` с рабочим деревом (stale-index между параллельными
  сессиями); conventional commits - одна строка, императив, lowercase,
  без точки в конце; один PR = одно изменение; НЕ делать push; не трогать
  служебные каталоги (.veai/). Полная процедура волн -
  [merge-wave-procedure](../../topics/merge-wave-procedure.md).
- Каждая фаза = отдельная кампания: измерение до/после (A/B против
  f0eed55), обновление статуса порта в
  [round5-ggml-replatform-deferred-ports](../../topics/round5-ggml-replatform-deferred-ports.md)
  ("не начато" -> "в работе" / "готово <дата>, коммит <sha>, число
  <результат>").
- Порядок: Фаза 1 (D4) -> Фаза 2 (D3) -> Фаза 3 (D1). Микробенчи D1 можно
  начинать параллельно с фазой 1 - они не требуют GGUF-модели. Фазы 4 (D2)
  и 5 (D5) независимы - в любом свободном окне.

## 3. Фаза 1 - порт D4: загрузчик glm5next GGUF (приоритет 1, блокирует весь GGUF-сервинг)

Цель: ветка снова открывает и обслуживает glm5next в GGUF-формате.

Что потеряно: в main загрузчика glm5next-GGUF НИКОГДА не было (это чисто
веточная функциональность). Старая реализация ссылалась на удалённые файлы
(models/gguf/dequant.py, kernel/gguf.py и др.), поэтому сейчас выбор
glm5next-GGUF падает с сырым ImportError.

Шаги:

1. Изучить новый GGUF-конвейер на поддерживаемой main архитектуре
   (ближайший аналог - линейка Qwen GGUF: тензоры и эксперты в
   models/gguf/{tensors,experts}.py, регистрация в models/gguf/config.py;
   moe-путь через layers/quantization/moe).
2. Составить карту тензоров/весов glm5next по старой реализации:
   `git show backup/vektory79-pre-merge5-b8c3794:python/freetoken/models/glm5_next/<файл>`
   (там: __init__.py, args.py, attention.py, config.py, gguf.py) и старые
   gguf-файлы той же ветки; маппинг перенести на новые
   models/gguf/{experts,tensors}.py.
3. Восстановить защиту gguf_types: эталон - `git show 9ac8830` (guard в
   load_ftw_banks + зеркало в load_ftw_banks_to_device, контракты ошибок
   "re-convert" / "does not match its config"); найти эквивалентные точки
   нового конвейера.
4. Заменить сырой ImportError при выборе неподдерживаемой gguf-архитектуры
   на внятное сообщение (список поддерживаемых, hint про конвертацию).
5. Тесты: юнит-тесты маппинга тензоров (фикстуры по методологии старых
   gguf-тестов - аналитические фикстуры, сеяные допуски) + boot-smoke
   glm5next-GGUF конфигом.

Критерии приёмки:

- boot-smoke glm5next-GGUF (offload-стратегия для начала): /ready 200,
  чат 200 + usage, SIGTERM -> rc 143, VRAM к baseline, census чист;
- тест-набор gguf-файлов зелёный; новые тесты падают-до/проходят-после
  (bug-fix правило);
- слот-гейт молчит на влезающем конфиге.

Риски: маппинг тензоров поименован - ошибка в одном имени даёт молчаливую
порчу весов, сверять поимённо; quant-форматы ветки (q8_0 / iq-линейка)
могут отсутствовать в новых layers/quantization/configs - тогда
минимальный набор форматов для старта модели + расширение в D1.

## 4. Фаза 2 - порт D3: гибрид/offload поверх slot pools - ГОТОВО 2026-10-21, коммит 393b682

Статус: готово 2026-10-21, коммит 393b682 (волны 8-17). Порт не понадобился -
main УЖЕ вёз гибрид на новом стеке; вернулись три fail-fast гейта (format
screen CPU-executor, header-отказ hybrid на непроверяемом gguf, pre-load auto
floor gate), T2 НЕ возвращён (однопульный executor, глобальный бюджет).
Приёмка (RTX 5090, UD-Q3_K_XL): 786k рецепт - `--moe-strategy hybrid
--moe-cache-auto --kv-reserve-tokens 786432 --kv-cache-dtype nvfp4
--memory-ratio 0.85 --max-prefill-length 4096 --max-running-requests 1
--moe-cpu-threads 16 --disable-moe-prefill-overlap` - бут x2 без лестницы
(ready 119/135 с, 3 пула 29-1103 слотов, гейты молчат); decode гибрид 27.71
tok/s медиана против offload 21.00 на том же стеке (+32%), против старого
стека 16.67 (+66%); cap-0 CPU-лег ~49.6 GB/s (~1.8x старого); Fix-3 e2e
5568 cached / 0 полных промахов; g.3 (test_mixed_expert_formats.py) 132
passed; g.2 отложен (D1-deferred - сидит на удалённом models/gguf/dequant).
Уроки - [hybrid-moe-slot-pools-port](../../topics/hybrid-moe-slot-pools-port.md).

Цель (исходная): гибридный MoE-режим (CPU+GPU сплит по парам (слой, эксперт)) снова
работает на новом кэше; хост-доставка T2 возвращена.

Что потеряно: main переписал offload-кэш на per-format slot pools и
переписал CPU-executor. Из engine.py удалены веточные хелперы: partition
gate, gguf hybrid rejection, T2 helpers, plural-cache state -
каждый нужно ре-имплементить в новом виде или признать неактуальным;
решение зафиксировать в kb. (NS-knob в этот список не входит: он жил в
kernel/triton/mhc.py, db0f6b3, не в engine.py.)

Шаги:

1. Карта нового API: layer_slot_bytes (engine/cache_budget.py), пулы слотов
   по форматам, один глобальный поток-бюджет executor-а.
2. Переадаптировать сплит гибрида по парам (слой, эксперт) (механика
   маркера -1, не по слоям), LRU-пул (~3 шага) и донесение кэша Fix-3
   (кампания [fix3-snapshot-lru-refresh](../fix3-snapshot-lru-refresh/TASK.md):
   не терять периодические полные промахи при повторных ходах - это была
   главная боль, 5 коммитов фикс-волны) к пулам слотов.
3. Вернуть T2-хост-доставку (кейс
   [decode-t2-host-delivery](../decode-t2-host-delivery/TASK.md)): коммиты
   227e7a3..d751511 в backup-ветке, ручка FREETOKEN_T2_POOL_THREADS;
   проверить взаимодействие с новым однопульным executor-ом (бюджет
   глобальный).
4. NS-knob mhc (NS=64, db0f6b3, +3.4% paired) - вернуть как опцию, мерить
   на новом стеке (см. фазу 3).
5. Пересчитать арифметику старого 786k гибридного рецепта под slot pools
   (с учётом moe_cache_mem / moe_cache_scope / moe_cache_split), чтобы
   прошёл cache-budget assert.

Критерии приёмки:

- старый 786k гибридный рецепт проходит boot-smoke (после пересчёта
  параметров);
- decode tok/s гибрида сравним с базлайном 16.67 (замер A/B против
  f0eed55-offload; точное число не гарантировано - фиксировать факт);
- кэш-донесение: сценарий периодических повторов ходов не даёт полных
  промахов (тест-паттерн Fix-3: 5/5 -> 0/5);
- тесты moe/hybrid-набора зелёные.

Риски: инварианты Fix-3 завязаны на старую структуру кэша - не переносить
слепо, переписать поверх пулов; конкуренция за глобальный тред-бюджет с T2.

## 5. Фаза 3 - порт D1: перенос выигрышей ядер на ggml-стек - ЗАКРЫТО NO-GO 2026-10-21, замеры 415b84e

Статус: закрыта NO-GO 2026-10-21 (волна 18, RTX 5090, пробы d1-stage*): grouped MMQ и
m-tile перекрыты вендоренным ggml нативно (MoE-проба: все форматы 241-572 GB/s эффективных
при 169-227 TF/s int8-экв. - ALU-крыша, не DRAM); dense q8_0 N-split медленнее целого
везде (kda 24896x4096: -1.8% при M=8128 и -7.1% при M=2048; 4096x4096: -16.5%;
2048x4096: -10.1%; целое ~306 TF/s = 73% int8-пика; bitwise-equal только на kda) -
FREETOKEN_GGUF_KDA_NSPLIT не воскрешать; mhc NS=64 не переносится - текущий дефолт
SPLIT=128 оптимален (T=512: 64 -29% / 32 -84%; T=8128: -40% / -107%; T=1 - латентный пол
<=1.8%; дрейф редукции <=1.2e-7). Числа и уроки -
[round5-d1-kernel-wins-no-go](../../topics/round5-d1-kernel-wins-no-go.md).

Цель: вернуть измеренные победы производительности ветки в новом виде.

Что потеряно (все жили в удалённом kernel/csrc/gguf/):

- grouped MMQ +16.4% @8128 (7f8c570);
- m-tile sweep MoE +67.2% (1706aae; прод-дефолт MOE_MTILE=32 из 8e2e4c7);
- q8_0 GEMM N-split +2.61% и tile64 +40.46% (кампания dense-q80-gemm,
  HEAD 8e2e4c7);
- mhc NS=64 +3.4% (db0f6b3).

Шаги (по каждому ядру отдельно - своя кампания/PR):

1. Взять реализацию из backup-ветки (`git show <sha>` или файл из
   backup-дерева).
2. Выбрать точку интеграции в новом стеке: либо внутри вендоренного ggml
   (если структура допускает), либо параллельный путь (triton/CUDA) с
   подключением в новый диспетчер. Решение фиксировать в kb (почему так).
3. A/B-замер против f0eed55 ДО включения: микробенч (benchbw / микро-харнес
   из старых кампаний) + сквозной декод после фаз 1-2. СТАРЫЕ ПРОЦЕНТЫ НЕ
   ПЕРЕНОСЯТСЯ - считать заново.
4. Включать дефолт (например MOE_MTILE=32) только при подтверждённом
   выигрыше на новом стеке.

Критерии приёмки (на каждое ядро): замер до/после на том же железе/конфиге,
выигрыш >= порога шума (как в старых кампаниях), нет регрессий на остальных
форматах, тесты ядра зелёные.

Риски: ALU-bound потолок старого стека (~758 tok/s MMQ) мог сместиться;
новый torch/triton меняют ландшафт; часть выигрышей может перекрыться
оптимизациями main - решает замер.

## 6. Фаза 4 - порт D2: возврат mmproj-зрения - ГОТОВО 2026-10-21, коммит 7bc68b0

Статус: готово 2026-10-21, коммит 7bc68b0 (волна 20; 4 файла:
checkpoint/convert.py, models/gguf/reader.py, verify_mmproj_tensors.py,
test_glm5_next_gguf.py). Приёмка: verify 348 тензоров 347/347 параметров exact
(max_abs_diff 0); boot raw GGUF + --mmproj READY 142 с (мультимодальность ON,
text chat 200, decode 17.63 tok/s); e2e-картинка - осмысленный ответ; 6
carrier-тестов разморожены и зелёные (fails-before: 5x TypeError extra_kvs +
1x ImportError _pack_kv); контрольная батарея 2 failed (базлайн-пара) /
2066 passed / 825 skipped. Уроки -
[round5-d2-mmproj-vision-port](../../topics/round5-d2-mmproj-vision-port.md).

Цель (исходная): мульти-модальный путь (изображения) снова работает.

Что потеряно: веточный mmproj-конвейер трогался удалённых файлов
(dequant.py и др.); реестры визионных архитектур уцелели в
models/gguf/config.py, но кодовая база под ними ушла.

Шаги:

1. Эталон - кампания ветки e35577d..f67449b (кейс
   [gguf-glm5next-vision](../gguf-glm5next-vision/TASK.md), файлы в
   backup-ветке): конвертация mmproj-весов, сверка 347/347 тензоров точно,
   аппаратный PASS; методология тест-фикстур и ловушки той кампании
   применимы снова.
2. Перенести конвертацию и обработку изображений на новые
   layers/quantization/* и models/gguf/{experts,tensors}.py.
3. e2e: запрос с изображением через chat API.

Критерии приёмки: поэлементно точная загрузка mmproj-тензоров (сверка как
в проверке 347/347), e2e-запрос с картинкой возвращает осмысленный ответ,
юнит-тесты конвертации зелёные.

## 7. Фаза 5 - порт D5: nvfp4-KV для generic attention-бэкендов - ГОТОВО 2026-10-21, коммит 660c814

Статус: готово 2026-10-21, коммит 660c814 (волны 22-23; 8 файлов:
kernel/triton/attention.py, attention/triton.py, attention/__init__.py,
tests/engine/test_kv_quant_config.py, tests/kernels/test_kv_nvfp4.py,
tests/kernels/test_triton_attention.py, benchmarks/bench_kv_quant.py, docs/cli.md).
Приёмка: ре-ленд load_nvfp4 (decode/extend/split/paged) НА текущую структуру ядер
(sinks/sliding-window/split-caps сохранены, None-шкалы байт-в-байт),
supports_fp8_kv+nvfp4_kv flipped вместе; test_kv_quant_config.py -p no:skipping
8 failed -> 36 passed; GPU: test_kv_nvfp4 115 passed (включая +32
sliding-window-комбо и 2 валидационных негатив-теста из ревью),
test_triton_attention 49 passed, test_kv_fp8 12 passed (волна 22); контрольная
батарея 2 failed (базлайн-пара) / 2079 passed / 850 skipped. Boot A/B:
RadixArk/Qwen3.8-27B-NVFP4-FTW (qwen3_5, pool FULL, head_dim 256): triton+nvfp4
бут без reject-а, KV 231954 ток (4.09 GiB) против bf16 67049 ток = 3.46x; greedy
12 задач temp 0 max_tokens 768: 11/12 byte-identical, 1 argmax-flip на открытой
перефразе (оба ответа корректны); needle @30% глубины найден идентично
(ZX-7412-KM). fa/fi/trtllm - отдельные PR (внешнему kernel нужна наша
layout-схема). Уроки -
[round5-d5-nvfp4-kv-generic](../../topics/round5-d5-nvfp4-kv-generic.md).

Цель: упакованный NVFP4 KV-кэш снова декодируют не только dsa/qsa_sparse.

Текущее состояние merged-дерева: только dsa и qsa_sparse имеют
supports_nvfp4_kv=True (attention/__init__.py); enforcement в engine
(_backend_supports_kv_quant, _validate_attention_backend_choice); triton /
fa / fi / trtllm отвергают nvfp4 на старте.

Шаги:

1. Ре-ленд квантованных KV-ядер для generic-бэкендов (реализация - в
   backup-ветке; на merged-дереве store-стек частично жив:
   kernel/triton/kv_nvfp4.py, kernel/triton/kv_quant.py, kvcache/mha_pool -
   но без активного покрытия из-за blanket-skip в tests/kernels/test_kv_nvfp4.py).
2. Реактивировать bsa-половины тестов: skip-маркеры в
   tests/engine/test_kv_quant_config.py (строки ~176 и ~248,
   параметризация kind="dsv4"|"bsa") - bsa-половины УЖЕ проходят на
   merged-дереве (проверено прямым прогоном при ревью merge); падают
   только dsv4-половины через duck-typing _adjust_dsv4_config.
3. Закрыть дыру покрытия store-стека (сейчас 0 активных тестов).
4. По мере готовности каждого бэкенда включать supports_nvfp4_kv точечно,
   с тестом.

Критерии приёмки: nvfp4-KV стартует на triton (затем остальных) без
reject-а; качественный A/B (перплексия/ответы) без деградации; тесты
kv_nvfp4/kv_quant зелёные и активные.

## 8. Приложение: гигиена вне функциональности

- 11 до-существующих битых markdown-ссылок в kb: ПОЧИНЕНО 2026-10-21 (docs-коммит).
  Фактический линк-чек по kb/ нашёл 9 битых: 5 ссылок из
  [lever3-chain-anatomy](../decode-t2-host-delivery/lever3-chain-anatomy.md)
  на исходники, удалённые ре-платформой f0eed55 (layers/gguf.py,
  kernel/csrc/gguf/*) - заменены текстом с пометкой об удалении;
  4 ссылки из [glm53flash-spec-decode/TASK.md](../glm53flash-spec-decode/TASK.md)
  имели неверную глубину пути - перенацелены на существующие файлы.
  Оставшиеся 2 из изначального списка (kb/README.md и
  [review-1.md](../skill-reviews/gguf-native-serving/review-1.md))
  уже резолвятся после коммита 618b81d (self-sufficiency sweep).
  Линк-чек после: 0 битых из 1119 внутренних ссылок.
- Upstream-латентный тест offload FakeGraph (raw_cuda_graph_exec,
  engine/graph.py) - кандидат на upstream issue (по правилам репо - сначала
  issue, обсуждение с мейнтейнерами).

## 9. Определение готовности всего кейса (свернуто 2026-10-21)

Сделано:

- старый 786k гибридный рецепт проходит boot-smoke (фаза 2, 393b682: бут x2 без
  лестницы);
- гибрид даёт измеренный декод: 27.71 tok/s против offload 21.00 на том же стеке
  (+32%; старый стек 16.67 = +66%);
- ядра D1 измерены: NO-GO - не нужен ни один порт (фаза 3, статус в разделе 5);
- mmproj-зрение e2e: PASS (фаза 4, 7bc68b0: e2e-картинка осмысленная,
  verify 347/347 exact, boot --mmproj READY 142 с - статус в разделе 6);
- nvfp4-KV доступен минимум на triton (фаза 5, D5, коммит 660c814: -p no:skipping
  8 failed -> 36 passed, A/B Qwen3.8 3.46x без деградации, 11/12 byte-identical).

Верификации закрытия (2026-10-21, облачное окно, HEAD eeceaf1, прогоны
сериализованы; числа - из артефактов .tasks/round5-deferred-ports/
{verification,audit}/):

- SSD/IO пути (базлайны FTW P-волн и r3-смоука) - деградаций нет: FTW boot
  /ready 66 с против 87 с; raw GGUF boot 133 с против 142-145 с (банки
  42 layers / 125.2 GiB за 106 с; холодный page-cache 1.9 GB/s против
  3.3-3.5 тёплых - состояние кэша, не путь); io_uring-путь чтения FTW
  не изменился (252 tasks / 81816 jobs, pool=8, sub-chunk 2048 KiB,
  backend direct); session tier жив (demote 21, restore из l1/l2,
  shutdown flush 5 live-сегментов, l2=3254.5 MiB, ext=20, dead=0,
  blob.bin сошёлся байт-в-байт со статистикой); replay fast-path 21 record
  ~2 с против 22 с полного скана (READY 64 с == бут 63 с); L2 dedup жив
  (физический blob = сумме уникальных экстентов; полный no-dedup A/B -
  вне волны);
- префилл (27692 ток x3, медиана полных чанков без последнего) -
  деградаций нет: GGUF offload 966.1 tok/s против 487.1 старого стека =
  x1.98 быстрее (69% fetch-потолка offload); FTW nvfp4 774.8 tok/s -
  первый записанный FTW-префилл-базлайн;
- аудит мёртвого кода (статанализ): уверенно мёртвых символов 13 -
  кластеры benchmark/client.py (benchmark_one_batch / benchmark_trace /
  read_qwen_trace / read_mooncake_trace / scale_traces / BenchmarkTrace),
  fla (L2Norm / chunk_kda / fused_recurrent), perf.py
  compare_memory_kernel_perf, osproc.py now_monotonic, session_tier.py
  _calc_payload_crc_iovecs; мёртвые env-ручки FREETOKEN_GGUF_KDA_NSPLIT /
  FREETOKEN_GGUF_MOE_MTILE / FREETOKEN_GGUF_DENSE_MTILE + 4 commented-out
  autotune-блока fla; CLEAN: legacy_format.py, коллизий имён нет, nvfp4
  read-путь единственный;
- аудит мёртвых тестов: 9 тестов к разморозке (4x tests/models/
  test_quant_config.py - D2-deferred контракт жив; tests/engine/
  test_cache_budget.py:1196 - причина ложна; 4x tests/checkpoint/
  test_ftw_gguf_banks.py - mmproj_path жив), 3 к удалению (тесты удалённой
  partition-machinery в test_cache_budget.py), причины 3 честных скипов
  переписать (FTW-index scan не ре-ленднут); module-skip вердикты:
  test_gguf_expert_banks.py держать с переносом test_moe_vec_chunk_cap,
  test_cpu_moe_gguf_iq.py - цель жива (g.2), tests/kernels/
  test_gguf_quant.py (27 тестов) - цель мертва (D1 NO-GO), кандидат на
  удаление/надгробие; гигиена: pip gguf нужен активным тестам, но не
  объявлен в dev-группе pyproject; collection 2921/2935, errors=0.

Остатки (вне D1-D5) после волны починки 2026-10-23 (fix(checkpoint) 48915b4,
fix(moe) 739336f, chore 039bbf9; kb - docs-коммит этой волны):

ЗАКРЫТО волной починки:

- gguf->FTW boot residual (OPEN residual ниже) - ЗАКРЫТ (48915b4): источник
  dense-типов ветвится по носителю - точные per-tensor типы из мета
  gguf_tensor_types (новые конверсии), легаси-носители через хук
  ftw_tensor_types (контракт конверсии + gguf_types тройки); добавлены
  qweight replay, row padding, gate/up fold и громкая ошибка re-convert
  вместо KeyError. HW-доказательство: до-реплатформенный 136g ftw-vision
  носитель бутится без реконверсии (/ready 87 с, chat 200);
- g.2 ре-ленд test_cpu_moe_gguf_iq - ЗАКРЫТ (739336f): тесты ре-ленднуты на
  живой third_party-деквант-референс (merged gate_up банки, per-layer
  форматы) и гоняют восстановленный per-layer num_experts guard банка
  (короткий банк на любом слое группы сигнатур - громкий fail при
  конструировании, был тихий oob-read в C++ ядре);
- чистка мёртвого кода по аудиту - ВЫПОЛНЕНА (039bbf9): 13 символов + кластер
  benchmark (-575 строк), мёртвые env-ручки KDA_NSPLIT / MOE_MTILE /
  DENSE_MTILE, always-dead layers/gguf import, байт-дубль _untile слит в
  models/gguf/tensors.py, offload_expert_method удалён (0 вызывающих, тесты
  патчили его вакуумно);
- тест-гигиена по аудиту - ВЫПОЛНЕНА (739336f + 039bbf9): 8 разморозок, 2
  теста partition-machinery удалены, 3 причины module-skip переписаны,
  tests/kernels/test_gguf_quant удалён (25 мёртвых, 2 живых переехали);
- pip gguf объявлен в dev-группе pyproject (039bbf9);
- kb/methods/v3a-surface-map.md помечен устаревшим (описанная поверхность -
  _gguf_grouped_*, tests/kernels/test_gguf_quant.py - удалена re-platformом
  и волной починки).

ОСТАЮТСЯ ОТКРЫТЫМИ:

- fa / fi / trtllm nvfp4-KV: внешним kernel'ам нужна наша layout-схема
  (row fp32 + block e4m3-u8/16 + low-nibble first) - отдельные PR с тестом и A/B.

Исторический список до волны починки (каждый пункт - отдельная задача/PR):

- g.2: ре-ленд test_cpu_moe_gguf_iq (сидит на удалённом models/gguf/dequant);
- чистка мёртвого кода по аудиту выше (кластеры символов - одна задача);
- разморозка/удаление тестов по аудиту выше (одна задача: 9 разморозить,
  3 удалить, 3 причины module-skip переписать);
- pip gguf объявить в dev-группе pyproject (одна строка);
- ревизия подозрений аудита: offload_expert_method (engine.py:1860 - прод
  не вызывает) и байт-дубль _untile (qwen3_5_moe/gguf.py:138 против
  qwen4_exp/gguf.py:154);
- OPEN residual (новый пункт порт-долга, вне D1-D5): gguf->FTW boot - ЛЮБОЙ
  конвертнутый FTW не поднимается: checkpoint_quant_config
  (models/register.py:458) строит таблицу ggml-типов из
  gguf_tensor_index(hf_config.model_path) (строка 469), но для FTW model_path =
  carrier source_metadata.gguf (metadata-only, tensor_count=0) -> пустая
  таблица -> KeyError "GGUF has no tensor 'token_embd'"
  (layers/quantization/configs/gguf.py:42, вызов из scheme_for :58).
  Pre-existing (воспроизводится на любом конвертнутом FTW, включая text-only);
  D4-boot-smoke не проявил - гнался только raw GGUF. Направление фикса:
  FTW-ветвящийся источник dense-типов (мета FTW-индекса) либо docs-запрет
  gguf->FTW до фикса; reconversion не поможет. Фикс - отдельной задачей.

Статусы всех портов обновлены в
[round5-ggml-replatform-deferred-ports](../../topics/round5-ggml-replatform-deferred-ports.md);
каждый порт - своя серия коммитов по правилам репо; push - только
пользователь.

Статус кейса: DONE 2026-10-21 - все пять портов закрыты (D4 9314c9b, D3 393b682,
D1 NO-GO 415b84e, D2 7bc68b0, D5 660c814); остатки зафиксированы в этом
списке и в kb-статьях кампании.
