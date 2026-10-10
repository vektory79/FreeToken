# TASK: спекулятивный декод для GLM-5.3-Flash GGUF в ft serve (MTP NextN blk.45)

Status: W0 EXECUTED 2026-10-10, вердикт NO-GO - кейс ЗАКРЫТ (полный отчёт волны:
[RESULTS.md](RESULTS.md)). Кейс создан
research-сессией: варианты изучены, экономика прикинута, W0-гейт сформулирован.
Scope-решение пользователя 2026-10-09: **только FreeToken (ft serve); llama.cpp-путь
НЕ трогать** (он остаётся референсом для чтения, не для запуска). Целевой профиль
пользователя: большие контексты 64k+ с radix-reuse. Read CONTRIBUTING.md first -
он обязателен. Якоря строк проверены 2026-10-09 против HEAD ce1ab86 (vektory79);
перед правками перепроверить (дрейфуют). Private-use scope GGUF glm5next: no push,
commit только по запросу пользователя.

## Goal

Декод ft serve на GLM-5.3-Flash-UD-Q3_K_XL @64k radix-reuse стоит ~53.6 ms/шаг
(18.65 tok/s, HEAD после T2 + C-L1; якоря ниже). Спекулятивный декод может
умножить токены на шаг (acceptance), но на этом риге verify-батч дорожает по
числу УНИКАЛЬНЫХ пар (слой, эксперт) в CPU-ноге гибридного MoE - выигрыш НЕ равен
acceptance. Цель кейса: (1) W0-гейтом измерить реальную цену verify-батча из M
токенов; (2) при положительном вердикте реализовать спекуляцию: сначала
verify-каркас с n-gram-драфтером, затем MTP self-draft из blk.45 того же GGUF.
Гейт приёмки: >= +12-15% декода @64k radix-reuse в парном окне без регресса
префилла и батареи качества. NO-GO на W0 закрывает кейс с числами.

## Current state (verified 2026-10-09 - do NOT re-derive)

### Модель и файл
- GLM-5.3-Flash = 321.3B гибридная MoE (upstream PR [ggml-org/llama.cpp#27754](https://github.com/ggml-org/llama.cpp/pull/27754)):
  45 транковых слоя = 34 KDA (linear attention, `attention.head_count_kv[i] == 0`)
  + 11 DSA/MLA (слои {3,7,11,15,19,23,27,31,35,39,43}) + 3 dense FFN (0,1,2);
  42 MoE-слоя, 288 экспертов, top_k 8, H=4096, I=2048.
- Файл `/media/ai/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL.gguf`
  (147.5 GB) СОДЕРЖИТ полный MTP/NextN блок `blk.45`:
  [phase2-tensor-map.md](../../findings/gguf-glm5next-path-a/phase2-tensor-map.md),
  раздел 8. `glm5next.nextn_predict_layers = 1`, `block_count = 46`.
  Состав blk.45: ПОЛНЫЙ DSA-attention (attn_q_a/q_b/kv_a_mqa/k_b/v_b/output,
  indexer-набор) + MoE (ffn_gate/up_exps Q3_K, ffn_down_exps Q4_K - ЕДИНСТВЕННЫЕ
  Q3_K/Q4_K банки в файле, ~3.6 GB суммарно) + nextn-клей (`nextn.eh_proj` 8192->4096,
  `enorm`, `hnorm`, `shared_head_norm`; embed и lm_head переиспользуются из транка,
  отдельных `nextn.embed_tokens`/`nextn.shared_head_head` в файле НЕТ).
- Конфиг пользователя: `/media/ai/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/config.txt`
  (llama-swap entry) указывает на llama.cpp-билд; для ft serve - ежедневный конфиг
  из памяти `ft-decode-lever-research`: FTW-директория `.../GLM-5.3-Flash-UD-Q3_K_XL-FTW/`,
  `--moe-cache-auto --kv-reserve-tokens 400000 --kv-cache-dtype fp8 --memory-ratio
  0.82 --max-prefill-length 8191 --moe-cpu-threads 16`, tier 10/50.

### FreeToken сегодня (HEAD ce1ab86)
- blk.45 ПРОПУСКАЕТСЯ целиком: [models/glm5_next/gguf.py](../../python/freetoken/models/glm5_next/gguf.py)
  (~:863 skip + debug-лог, ~:1011 continue для MTP-банков, ~:462 комментарий
  "FreeToken has no MTP"); `num_layers = 45`. HF-путь игнорирует аналогично.
- Draft/speculative-инфраструктуры НЕТ: ни draft-загрузчика, ни verify-цикла в
  scheduler, ни спекулятивного сэмплирования, ни метрик acceptance.
- Заделы-рудименты: [kernel/fla/fused_sigmoid_gating_recurrent.py](../../python/freetoken/kernel/fla/fused_sigmoid_gating_recurrent.py)
  уже принимает `intermediate_states_buffer` (пошаговый чекпоинт рекуррентного
  состояния GDN) и `retrieve_parent_token` (`HAS_EAGLE_TREE_CUSTOM_ATTN_MASK`,
  EAGLE-маска дерева) - порт-источник (sglang/fla) поддерживал спекуляцию, примитивы
  на месте; [kvcache/qsa_pool.py](../../python/freetoken/kvcache/qsa_pool.py)
  `ring_capacity_for(num_speculative_tokens=...)` (qwen4_exp, vLLM-сайзинг);
  [scheduler/decode.py](../../python/freetoken/scheduler/decode.py) `DecodeManager`
  = continuous batching (running_reqs/inflight_tokens) - батч декода >1 существует.
- Стоимость шага (числа с конфигом):
  [moe-hybrid-cost-model.md](../../topics/moe-hybrid-cost-model.md),
  [decode-chain-microfusion.md](../../topics/decode-chain-microfusion.md):
  - @64k radix-reuse, GGUF гибрид, temp 0, 768 токенов: 16.67 tok/s = 60 ms/шаг
    (кампания 63b9bff); после T2 (e541423/227e7a3) ~17.4; после C-L1 (db0f6b3,
    mhc NS=64) парное окно 2026-09-28: **18.65 vs контроль 18.04 tok/s**.
  - Шаг сериализован; CPU-нога гибрида качает уникальные пары (слой, эксперт):
    bs=1 -> 42 x top_k 8 = 336 пар/шаг; миссы @64k ~63-72%; fetch f=30.7%.
  - Декод bs=1: CUDA-графы ~= eager (15.82 vs 15.84 tok/s) - verify-шаг можно
    гонять eager без налога.

### VRAM-рамка
- Boot-порог рецепта пользователя 29.28 GiB свободных (инцидент
  [crash-mr080-400k](../../incidents/crash-mr080-400k/README.md), OPEN);
  greedy-fill планировщика: урезание KV не создаёт headroom
  ([vram-budget.md](../../topics/vram-budget.md)).
- Draft-слой blk.45 нативно ~3.6 GB (Q3_K/Q4_K). Варианты размещения:
  (a) VRAM-резидентно ~3.6 GB - прямая атака на boot-порог (либо жертвовать
  reserve/чанком по меню vram-budget); (b) CPU/offload-машиной - нужны Q3_K/Q4_K
  в банках и CPU-исполнителе (сейчас `_WFMT_IDS` = bf16/nvfp4/mxfp4_triton/ds_fp4/q4_0);
  (c) dequant-at-load в поддерживаемый формат: q8_0 ~7.7 GB RAM, bf16 ~14.5 GB
  (VRAM не лезет, RAM ок). Выбор - часть W2.

## Evidence из экосистемы (числа с конфигом; 2026-10-09)

- ik_llama.cpp PR [ikawrakow/ik_llama.cpp#1513](https://github.com/ikawrakow/ik_llama.cpp/pull/1513)
  (GLM-5 MTP, `-mtp`, draft-max 10, p-min 0.85, GLM-5 smol-IQ2_KS): acceptance
  50.8-62.2%, но CPU-оффлоад-риг: **25.01 -> 11.85 tok/s (2x МЕДЛЕННЕЕ)** при
  acceptance 0.573 (786/1371); собственные замеры автора тоже отрицательны
  (8.18 -> 5.1-6.9 tok/s). Урок: глубокий draft + verify через CPU-MoE = яд.
  PR закрыт.
- mainline llama.cpp `--spec-type draft-mtp` (выбор: draft-simple/eagle3/mtp/
  ngram-simple/ngram-mod/ngram-cache/...): qwen35 (гибрид 48 GDN + 16 full-attn)
  MTP **~1.7x на RTX 5090** при полной GPU-оффлоаде
  ([ggml-org/llama.cpp#28196](https://github.com/ggml-org/llama.cpp/issues/28196));
  Qwen3.8-Flash-Next 1.3-2x ([#28243](https://github.com/ggml-org/llama.cpp/pull/28243));
  **Gemma4 MoE - ускорения НЕТ** (dense 2x)
  ([#23398](https://github.com/ggml-org/llama.cpp/pull/23398)) - MoE-verify
  съедает выигрыш и на GPU, если экспертные банки DRAM-bound.
- Референс-реализация MTP-графа для glm5next: `src/models/glm5next.cpp` `graph_mtp`
  (локальная читальня: `/media/ai/src/llama-cpp-glm5next`, unslothai fork,
  build 10836 commit 629b50552 - флаги `--spec-type draft-mtp` в help есть).
  Снапшот старой ревизии с номерными строками:
  [llamacpp-glm5next-snapshot/README.md](../../findings/llamacpp-glm5next-snapshot/README.md)
  (WARNING: ревизия не записана). ЧИТАТЬ как спецификацию; НЕ запускать (scope).

## Варианты (вердикт research-сессии)

1. **MTP NextN self-draft из blk.45** - основной кандидат. Веса уже в файле
   (не надо вторую модель), draft-слой DSA+MoE (переиспользует
   Glm5NextDSABackend и MoE-машину), shared токенайзер/lm_head/embed.
   Draft-шаг: 1 DSA-слой + 1 MoE-слой + lm_head - при VRAM-резидентных банках
   это ~1/40 стоимости транкового шага.
2. **n-gram / prompt-lookup drafter** - training-free, ноль VRAM, ноль новых
   весов; drafter = n-gram-таблица последних токенов (у FreeToken есть radix-кэш,
   но драфтеру достаточно хэш-таблицы). Низкий, workload-зависимый потолок, но
   ИДЕАЛЬНАЯ первая волна: строит verify-каркас и метрики acceptance без
   загрузки blk.45.
3. **Отдельная draft-модель (классика)** - ОТКЛОНИТЬ: мелкого GLM-5.x с тем же
   токенайзером в распоряжении нет, движок одно-модельный, двойной engine -
   непропорциональная работа.
4. **Sidegrade на llama.cpp `--spec-type draft-mtp`** - вне scope (решение
   пользователя); упомянут как единственный способ БЫСТРО померить acceptance
   GLM-5.3-Flash чужими руками, если пользователь передумает.

## Work items (ordered by dependency; W0 - ГЕЙТ всего кейса)

### W0. Зонд цены verify-батча (БЕЗ кода движка)
- Вопрос: во сколько раз дорожает шаг гибридного MoE при M токенах на шаг
  (уникальные пары растут сублинейно от M x 8, hits в VRAM-слотах бесплатны,
  GEMM на CPU эффективнее GEMV на байт)?
- Метод: N параллельных декод-потоков через ft serve (continuous batching
  разделяет MoE-кэш) - per-token cost vs N in {1,2,4,6} на якорном конфиге
  (harness-паттерн [.tasks/decode-research/run_arm.py] + серийные руки;
  контексты 32k и 64k на поток; следить за KV-бюджетом 400k). N потоков -
  верхняя оценка verify-батча (маршруты независимых потоков пересекаются
  меньше, чем M продолжений одной цепочки: реальный verify будет ДЕШЕВЛЕ
  замера - консервативный гейт). ДВЕ руки стратегии: `--moe-strategy hybrid`
  И `offload` - вопрос "offload+MTP без CPU-ноги" закрывается тем же зондом.
  Опорные числа offload @64k: шаг fetch-bound - ~2.14 GiB/шаг при ~28.4 GB/s
  эфф. = 77.1 ms (12.97-13.72 tok/s против 16.29-18.65 у гибрида;
  moe-hybrid-cost-model.md, decode-research 2026-09-26); на 2k разрыв сжимается
  до 1.13x (14.32 против 16.12).
- Вердикт-формула: приёмка M=4 требует `step_cost(M)/step_cost(1) <
  ожидаемый acceptance x запас`. Ожидаемый acceptance для MTP GLM-класса:
  1.8-2.6 ток/шаг (acceptance 50-62% при глубоком draft из PR #1513; на
  коротком draft ниже, но и M меньше). Явно недостижимый гейт -> NO-GO,
  кейс закрывается с таблицей чисел (валидный исход по правилам kb).
- Ветка offload+MTP (исследована 2026-10-09, chat): "убрать CPU-ногу" НЕ
  выигрыш - CPU-нога гибрида это вторая труба пропускной способности, а не
  накладной расход; в offload её работу делает PCIe по ОДНОЙ трубе с меньшей
  эффективной полосой (~28.4 GB/s против ~41.5 GB/s суммарных у пары гибрида).
  Шаг offload fetch-bound: объём фетча растёт с g(M) (те же уникальные пары),
  acceptance его делит. Брейк-ивен против СЕГОДНЯШНЕГО гибрида:
  A >= (77.1/53.6) x g(M) = 1.44 x g(M); при потолке MTP A <= 2.6 это требует
  g(4) <= 1.8 - на грани оптимизма; значит offload+MTP в лучшем случае
  ДОГОНЯЕТ сегодняшний гибрид, и никогда не обгоняет hybrid+MTP (разрыв
  bs=1 1.25-1.35x сохраняется при любом g, у обеих стратегий объём растёт
  одинаково). Отдельной стратегии НЕТ - только рука W0 выше.
- Замеры: парные окна, p50 ms/шаг первична, last-chunk артефакт не трогать
  (память `ft-last-chunk-throughput-artifact`), сериальные руки (VRAM-протокол).

### W1. Verify-каркас + n-gram drafter (код; только при PASS W0)
- Draft/verify цикл в scheduler: K токенов черновика -> один verify-шаг на
  K+1 позиций -> acceptance по аргмаксу (greedy-фастпат первым).
- Откаты на reject: KV-страницы/radix trim (механика abort/fini уже есть),
  GDN-состояние: снапшот h0 34 KDA-слоёв перед циклом (GPU-копия, размер
  состояния уточнить при реализации; примитив `intermediate_states_buffer`
  в ядре уже принимает пошаговые состояния) ИЛИ восстановление из
  intermediate-буфера принятого префикса; DSA-kpool: watermark/откат хвостов
  черновика.
- Drafter W1 = n-gram таблица (глубина K=1..4, adaptive depth по hit-rate
  оставить на W3). Метрики: accepted/draft counts в status.
- Тесты: fixture-модель (маленькая), интеграционные - по тест-традиции
  tests/ (зеркало scheduler/); fails-before для каждого багфикса.

### W2. MTP draft-слой blk.45 (код; параллельно/после W1)
- Анскип загрузки: `models/glm5_next/gguf.py` (~:863, ~:1011) - отдельная
  draft-ветка модели, `num_layers` транка остаётся 45; FTW-конверсия обязана
  эмитить draft-тензоры в ОБА пути чтения (урок T66
  [decode-chain-microfusion.md](../../topics/decode-chain-microfusion.md));
  бут-тест на raw GGUF И на FTW обязателен.
- Q3_K/Q4_K банки draft-слоя - выбрать: (a) CPU executor форматы (ggml-порт
  W4A8K по образцу кампании 11f1a80; референс деизоляции
  `kernel/csrc/gguf/dequantize.cuh`, parity-модель
  `tests/kernels/test_gguf_quant.py`: 8*2^-11*factor*|d| + 1e-3);
  (b) dequant-at-load в q8_0 RAM-банки (~7.7 GB RAM, ноль новых ядер) -
  РЕКОМЕНДОВАНО для первой итерации.
- eh_proj/enorm/hnorm/shared_head_norm + shared lm_head/embed; собственный
  мини-кэш draft-слоя (KV/kpool только для il=45 по образцу mainline
  LLAMA_CONTEXT_TYPE_MTP).
- VRAM-план: recipe с учётом +3.6 GB (либо RAM-вариант), проверка boot-порога
  методом [vram-headroom.md](../../methods/vram-headroom.md); при жёстком
  пороге - деградация `--max-prefill-length` 8191 -> 4095-класс (+2 GiB, цена
  префилла ~-26% по vram-budget).

### W3. Сэмплирование и адаптивная глубина
- Спекулятивное сэмплирование (rejection sampling) для temp>0 - иначе
  распределение ответов меняется (у пользователя temp 1.0/top-p 0.95 в
  llama-swap-конфиге; ft serve путь уточнить). Greedy-фастпат уже в W1.
- Adaptive depth: K по скользящему acceptance (llama.cpp-опыт jukofyork:
  фиксированный p-min субоптимален, стоимость шага прыгает по бэкендам).

### W4. CUDA graphs
- Draft-шаг (bs=1) и verify (bs=K+1) - графы либо eager (kb: eager ~= graph,
  налога нет; начать с eager, графы - только если nsys покажет launch-налог).

### W5. Hardware acceptance
- Парный A/B в одном окне против якоря HEAD (18.04-18.65 tok/s @64k
  radix-reuse, тот же промпт/заполнение 64k из filler, 768 max_tokens,
  temp 0 - метод re-acceptance 63b9bff): гейт **>= +12-15%** e2e декода.
- Батарея качества 24/24 ([harness/quality/](../../harness/quality/README.md));
  префилл не хуже -1%; boot на raw GGUF и FTW; VRAM floor задокументирован
  (инцидент crash-mr080 не ухудшать); сериальные руки, реальная readiness-
  проверка запросом, не /v1/models.

## Constraints

- Только FreeToken; llama.cpp-билд не трогать и не запускать (решение
  пользователя 2026-10-09).
- VRAM-протокол: сериальные руки, пользователь переключается на cloud-LLM
  на время GPU-экспериментов; CPU-волны (тесты, синтаксис) - в любой момент.
- Один change per PR, Conventional Commits, commit только по запросу.
- NO-GO - валидный исход: закрыть кейс таблицей чисел и вердиктом в
  cases/README.md (прецедент: tier-staging-admission-trigger).

## Acceptance criteria

- W0: таблица step_cost(M) для M in {1,2,4,6} x {32k, 64k} на якорном конфиге
  + вердикт-формула применена явно. PASS/NO-GO записан в cases/README.md.
- W1-W5 (при PASS W0): декод @64k radix-reuse >= 21 tok/s в парном окне
  (>= +12-15% к 18.65), батарея 24/24, префилл > -1%, boot обоих путей,
  acceptance-метрики в status, тесты зелёные (`uv run pytest tests/ -m "not slow"`).

## Ссылки

- Стоимость декода: [moe-hybrid-cost-model.md](../../topics/moe-hybrid-cost-model.md);
  якоря скорости: [baselines/README.md](../../baselines/README.md),
  [RESULTS.md](../../baselines/decode-research/RESULTS.md).
- Карта тензоров/конфигов GGUF: [phase2-tensor-map.md](../../findings/gguf-glm5next-path-a/phase2-tensor-map.md),
  [phase1-config-keys.md](../../findings/gguf-glm5next-path-a/phase1-config-keys.md)
  (раздел MTP/NextN handling - hparams/кэш-роутинг mainline).
- Харнесы: [harness/decode-research/README.md](../../harness/decode-research/README.md),
  [harness/serve-measure/](../../harness/serve-measure/README.md),
  [harness/quality/](../../harness/quality/README.md).
- Upstream: PR #27752 / #27754 (glm5next), #28196 (qwen35 MTP 1.7x), #23398
  (Gemma4 MoE no-gain), #28243 (Qwen3.8 1.3-2x), ik_llama.cpp #1513 (CPU-риг
  отрицательный результат).
