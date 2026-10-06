# TASK: поддержка зрения (mmproj) для GGUF glm5next

Status: ALL PHASES DONE (1-4 CPU + 5 HW), 2026-10-06, commits e35577d..2b91236
+ kb d8dee13; HW-волна - [phase-5-hw.md](phase-5-hw.md). Composed 2026-09-27
по итогам исследовательской сессии
(исследование проводилось на ветке vektory79, HEAD edd6309; якоря строк
проверены против HEAD - перед редактированием перепроверить, они дрейфуют).
Железо: локальный риг RTX 5090 (VRAM-протокол CONTRIBUTING; HW-волна -
отдельная фаза 5).
Private-use scope GGUF glm5next: без push и upstream PR (решение
пользователя 2026-09-13, требование issue #34 отменено). CONTRIBUTING.md
читать первым - он обязателен.

## Goal

Добавить приём изображений в GGUF-путь FreeToken для
GLM-5.3-Flash-GGUF: визуальная башня из отдельного mmproj GGUF-файла,
конфиг и image-процессор из его метаданных, ноль внешних HF-файлов.
Оба пути загрузки: raw .gguf и GGUF-FTW. Референс качества - уже
работающий NVFP4-путь.

## User decisions (зафиксированы 2026-09-27, не переоткрывать)

1. Покрыть ОБА пути загрузки: raw .gguf и GGUF-FTW.
2. Для raw .gguf бута путь к mmproj задаёт оператор явным CLI-флагом
   `ft serve` (например `--mmproj <path>`); никакого автоматического
   поиска.
3. Конфиг image-процессора выводится из GGUF-метаданных mmproj
   (ключи `clip.*`) в коде; внешние preprocessor_config.json и пр. не
   требуются.
4. (добавлено 2026-09-27, та же сессия) `ft checkpoint` включает веса
   зрения в FTW: визуальные тензоры mmproj упаковываются в чекпойнт
   как `visual.*` вместе с метаданными башни, GGUF-FTW бут
   самодостаточен по башне и флаг --mmproj НЕ требует. Флаг остаётся
   способом задать mmproj только для raw .gguf бута.

## Verified background (не переоткрывать; перепроверить якоря перед правкой)

### Как устроен референс (NVFP4-путь, работает)

- `../../../python/freetoken/models/glm5_next/vision.py` - полная визуальная башня
  на PyTorch: `Glm5NextVisionModel` (Conv3d patch-embed, pre-RMSNorm
  блоки с q/k-нормами + 2D rope через
  `apply_rope_with_cos_sin_cache_inplace`, bidirectional attention в
  пределах каждой картинки, 2x2 Conv2d downsample, clamped-SwiGLU
  merger). Имена параметров: `visual.patch_embed.proj.{weight,bias}`,
  `visual.blocks.N.{norm1,norm2}.weight`,
  `visual.blocks.N.attn.{qkv,proj}.{weight,bias}`,
  `visual.blocks.N.attn.{q_norm,k_norm}.weight`,
  `visual.blocks.N.mlp.{gate,up,down}_proj.{weight,bias}`,
  `visual.post_layernorm.weight`, `visual.downsample.{weight,bias}`,
  `visual.merger.proj.weight`, `visual.merger.post_projection_norm.*`,
  `visual.merger.{gate,up,down}_proj.*`. Класс Multi-модели:
  `Glm5NextForConditionalGeneration` строит `self.visual` при
  `config.is_multimodal` (`model.py:194-204`) + `place_encoder_weights`.
- Референсные имена весов: `iter_weights(include_vision=True)` отдаёт
  `visual.*` из HF-safetensors (`weight.py:161-164`), отдельный хук
  `iter_vision_weights` (`weight.py:241-250`).
- Реестр: спека `Glm5NextForConditionalGeneration`
  (`models/register.py:303-311`) несёт `mm_processor=_GLM5_NEXT_PROCESSOR`
  и `encoders=_GLM5_NEXT_ENCODERS` (`EncoderSpec("vision",
  "vision_config", ("image",))`). `is_multimodal == (vision_config is not
  None)` (`models/config.py:368-369`).
- Image-процессор: `Glm5NextMMProcessor`
  (`mm/processors/glm5_next.py`) берёт
  `AutoImageProcessor.from_pretrained(model_path)`
  (`mm/processor.py:83-90`) - требует preprocessor_config.json в
  директории модели; в GGUF-директории его НЕТ (проверено ls
  2026-09-27). Из изображения собирает `MMItem` (pixel_values bf16 +
  grid_thw), placeholder - последовательность ` идентиtoken_id`;
  begin/end обёртки рендерит сам chat-шаблон.
- Boot энкодера в движке обобщён: `config.active_encoders` -> модель
  обязана реализовать `SupportsMultimodal` -> `place_encoder_weights`
  (`engine/engine.py:524-527`). mm-пайплайн (encoder cache, серверный
  API, scheduler) не завязан на формат весов.

### Как устроен GGUF-путь сейчас (текст-only)

- Реестр: `Glm5NextGGUFForCausalLM` (`models/register.py:315-320`) ->
  module glm5_next, model_cls `Glm5NextForCausalLM` (ТЕКСТ-only класс,
  без `visual`), `parse_config="parse_gguf_config"`,
  `iter_weights="iter_gguf_weights"`, БЕЗ mm_processor/encoders.
- Шим: `GgufConfigShim` (`models/gguf/config.py`),
  `GGUF_ARCH_TO_REGISTRY["glm5next"] = "Glm5NextGGUFForCausalLM"`.
- `parse_gguf_config` (`glm5_next/gguf.py:60-253`) не строит
  `vision_config` -> `is_multimodal` False.
- `iter_gguf_weights` (`glm5_next/gguf.py:413-581`) читает только
  основной .gguf; параметра `include_vision` НЕТ. Важно: общий лоадер
  `models/weight.py:load_weight` передаёт `include_vision` в семейство
  ТОЛЬКО если `spec.encoders` непуст (`weight.py:230-233`) - после
  добавления encoders в GGUF-спеку `iter_gguf_weights` обязан принять
  этот kwarg, иначе boot падает.
- Хук encoder-only перегрузки: `load_vision_weight`
  (`models/weight.py:246-252`) ищет в модуле семейства функцию с именем
  `iter_vision_weights` (`_model_override`). У glm5_next такая уже есть
  (HF-версия) - GGUF-версию придётся диспетчеризовать по формату
  чекпойнта внутри неё.
- Модель: `Glm5NextForCausalLM.__init__` вызывает
  `convert_glm5_next_to_gguf` при `is_gguf_model(config)`
  (`model.py:176-183`) - мультимодальный сиблинг наследует это через
  super().
- FTW: `ftw_lacks_vision` (`models/weight.py:255-263`) и
  `VISION_KEY_PREFIXES` (`models/config.py`) фильтруют vision-тензоры в
  FTW-чекпойнте.

### mmproj-файл (само исследование)

- Файл: `/media/ai/models/unsloth/GLM-5.3-Flash-GGUF/mmproj-BF16.gguf`,
  arch `clip`, GGUF v3, 348 тензоров (224 F32 + 124 BF16). Основной
  .gguf vision-тензоров не содержит (kb
  [phase2-tensor-map.md, §11](../../findings/gguf-glm5next-path-a/phase2-tensor-map.md)).
- Полный дамп имён: [tensors-mmproj.txt](../../findings/gguf-glm5next-path-a/tensors-mmproj.txt).
- Upstream-контекст: llama.cpp строит башню в отдельном
  clip_graph_glm5next (наследник clip_graph_glm4v) через
  PROJECTOR_TYPE_GLM5NEXT; video-токены не поддерживаются (kb
  [llamacpp-tensor-kinds.md, "mmproj / vision"](../../findings/gguf-glm5next-path-a/llamacpp-tensor-kinds.md)).
- Кандидатский маппинг llama.cpp -> FreeToken `visual.*` (уточнить по
  дампу и сверить численно, см. Спецификацию):
  - `v.blk.N.attn_qkv.{weight,bias}` -> `visual.blocks.N.attn.qkv.*`
  - `v.blk.N.attn_out.{weight,bias}` -> `visual.blocks.N.attn.proj.*`
  - `v.blk.N.attn_q_norm.weight`, `v.blk.N.attn_k_norm.weight` (F32, 64)
    -> `visual.blocks.N.attn.{q,k}_norm.weight`
  - `v.blk.N.ln1.weight`, `v.blk.N.ln2.weight` (F32, без bias)
    -> `visual.blocks.N.{norm1,norm2}.weight`
  - `v.blk.N.ffn_gate/ffn_up/ffn_down.{weight,bias}`
    -> `visual.blocks.N.mlp.{gate,up,down}_proj.*`
  - `mm.post_norm.{weight,bias}` -> `visual.post_layernorm.weight`
    (ВНИМАНИЕ: в файле есть и bias - у FreeToken RMSNorm без bias;
    выяснить, что это за bias, прежде чем маппить - см. Риски)
  - `mm.patch_merger.{weight,bias}` -> `visual.downsample.{weight,bias}`
    (Conv2d out,in,kh,kw; GGUF ne-порядок обратный: ne=(2,2,1024,4096))
  - `mm.model.fc.weight` -> `visual.merger.proj.weight`
  - `mm.gate/mm.up/mm.down.weight` -> `visual.merger.{gate,up,down}_proj`
    (ne-порядок: ne0 = in-dim строки, чтение даёт (ne1, ne0) - проверить
    транспонирование численно)
- Считыватель GGUF: `models/gguf/reader.py` (`iter_gguf_tensors`,
  `load_gguf_metadata`) - переиспользовать, отдельного clip-ридера нет.
- Директории моделей (проверено ls 2026-09-27):
  - raw GGUF: `/media/ai/models/unsloth/GLM-5.3-Flash-GGUF/` содержит
    ТОЛЬКО mmproj-BF16.gguf и подкаталог UD-Q3_K_XL/ - mmproj лежит на
    уровень выше основного gguf, что и мотивировало решение про флаг.
  - GGUF-FTW: `UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL-FTW/` содержит FTW
    шарды + freetoken_weight.json + source_metadata.gguf (metadata-only,
    без таблицы тензоров); processor-конфигов и chat_template.jinja нет.
  - Референс NVFP4-FTW: `RedHatAI/GLM-5.3-Flash-NVFP4-FTW/` содержит
    config.json, processor_config.json, chat_template.jinja.

## Specification

Фазы по возрастанию риска; каждая фаза закрывается тестами до перехода.

### Фаза 1: флаг и мультимодальная спека GGUF

1. `--mmproj <path>` в `server/args.py` (валидация: файл существует,
   arch == `clip`, иначе fail-fast с внятной ошибкой). Прокинуть в
   конфиг-сборку. Если флага нет - GGUF обслуживается текст-only как
   сейчас (никакого молчаливого поиска mmproj).
2. Новая спека реестра, например `Glm5NextGGUFForConditionalGeneration`:
   module glm5_next, model_cls `Glm5NextForConditionalGeneration`,
   `parse_config="parse_gguf_config"`, `iter_weights="iter_gguf_weights"`,
   `mm_processor=_GLM5_NEXT_PROCESSOR`, `encoders=_GLM5_NEXT_ENCODERS`.
   Точка ветвления реестра - открытый дизайн-вопрос сессии: шим строится
   из пути (`build_gguf_shim`), а флаг живёт в serve-аргах; кандидаты -
   передавать mmproj-факт в `build_gguf_shim`/`GGUF_ARCH_TO_REGISTRY`
   или выбирать спеку в `parse_gguf_config` по наличию vision-конфига.
   Требование: без флага путь реестра не меняется ни на бит.
3. `parse_gguf_config`: при заданном mmproj открыть его метаданные,
   собрать `VisionConfig` (`glm5_next/config.py:40-54`: depth, hidden_size,
   num_heads, intermediate_size, projection_intermediate_size,
   out_hidden_size, patch_size, temporal_patch_size, in_channels,
   spatial_merge_size, rms_norm_eps, swiglu_limit, attention_bias) из
   ключей `clip.*` mmproj-метаданных; сверить каждое поле со значениями
   NVFP4-референса (см. Acceptance) и fail-fast при расхождении.
4. `iter_gguf_weights`: добавить kwarg `include_vision=True` (см.
   background); при include_vision=False mmproj не открывается вовсе.
5. `iter_vision_weights` в glm5_next: диспетчер по формату чекпойнта -
   GGUF -> чтение mmproj, иначе текущий HF-ридер.

### Фаза 2: ридер и маппинг mmproj-тензоров

1. Функция-итератор (например `iter_gguf_vision_weights(mmproj_path,
   device)`) в `glm5_next/gguf.py`: отдаёт `visual.*`-тензоры с
   маппингом из Verified background. Касты: BF16 -> bf16 как есть, F32
   нормы/скаляры - по правилам референсного `_iter_vision` (bf16) с
   оговорками ниже.
2. Числовая верификация: каждый из 348 тензоров сравнить с
   соответствующим тензором NVFP4-референса
   (`RedHatAI/GLM-5.3-Flash-NVFP4-FTW` или исходного HF-чекпойнта;
   чтение тензоров - CPU-only, VRAM не нужно). Формат: таблица имя ->
   max_abs_diff; порог TBD сессией (башня BF16, ожидание ~1e-2 для
   зашумлённых, ~0 для точных совпадений). Несовпавшие - кандидаты на
   перепроверку маппинга, не на "и так сойдёт".
3. Порядок осей: каждая свёрточная/линейная форма читается с учётом
   GGUF ne-порядка (ne0 - fastest-varying); верификация из п.2 ловит
   транспонирование.
4. Слияние attn_qkv: убедиться, что llama.cpp хранит qkv в том же
   row-порядке, что ожидает `vision.py` (q|k|v по выходной оси);
   верификация п.2 это ловит.

### Фаза 3: image-процессор из метаданных mmproj

1. Fallback в `Glm5NextMMProcessor`/`mm.processor.py`: когда в
   model_dir нет preprocessor_config.json и задан mmproj - собрать
   конфиг image-процессора из ключей `clip.*` (patch_size,
   temporal_patch_size, in_channels, norm mean/std если есть в
   метаданных). Если какого-то поля в метаданных НЕТ - использовать
   значение NVFP4-референса, зафиксировав его в коде как
   GLM-5.3-Flash-константу с проверкой против файла при возможности.
2. Проверить наличие и достаточность `image_token_id` и chat-шаблона в
   GGUF-токенизаторе (`models/gguf/tokenizer.py`): шаблон обязан
   рендерить begin/end-обёртки вокруг `	prev \\
   placeholder вставляет только последовательность токенов (см. комментарий
   в `mm/processors/glm5_next.py:15-16`). Если шаблона нет или он не
   содержит image-секции - fail-fast с диагностикой, не молчаливый
   битый промпт.
3. Токен-бюджет/мин-токены: `get_mm_processor_kwargs` уже передаёт
   min/max_image_tokens - проверить, что fallback-конфиг их принимает.

### Фаза 4: GGUF-FTW путь

1. `ft checkpoint` для GGUF-источника упаковывает зрение в FTW
   (USER DECISION 4, 2026-09-27): визуальные тензоры mmproj записываются
   в чекпойнт как `visual.*` (bf16), а метаданные башни (clip.* ключи
   для VisionConfig и image-процессора) попадают в метаданные FTW;
   носитель KVs (freetoken_weight.json / source_metadata.gguf / индекс
   FTW) - решение сессии. После этого GGUF-FTW бут строит башню из
   чекпойнта и флаг --mmproj НЕ требует; флаг остаётся только для raw
   .gguf бута.
2. Выяснить в сессии, какой код писал freetoken-000xx.ftw шарды для
   GGUF-источника (конвертер `checkpoint/convert.py` для single-file
   gguf пишет только metadata-only копию; шарды появились откуда-то
   ещё) - точка врезки упаковки mmproj в конвертацию.
3. `ftw_lacks_vision` и VISION_KEY_PREFIXES: убедиться, что FTW с
   упакованными visual.* грузит башню штатным iter_ftw_weights-релеем,
   а FTW без vision (чекпойнты, конвертированные до этой кампании) -
   работает как сейчас (текст-only; оператор переконвертирует при
   желании).
3. E2E-проверка GGUF-FTW и raw-gguf boot - HW-волна, отдельная фаза
   (см. ниже).

### Фаза 5 (отдельная HW-волна, VRAM)

Boot + e2e image-запрос на реальном железе: raw .gguf (с флагом
--mmproj) и GGUF-FTW (башня из чекпойнта, без флага), одна картинка,
сравнение ответа с NVFP4-путём на том же изображении
(субъективная валидация + sanity чисел). По VRAM-протоколу: сначала
упаковать сценарии и критерии, потом попросить пользователя освободить
VRAM. VRAM-интеракцию с expert offload (башня host-placement по
умолчанию, `encoder_weights=host`) проверить на
[kb/methods/vram-headroom.md](../../methods/vram-headroom.md).

## Tests

- Зеркало: `../../../tests/models/test_glm5_next_gguf.py` (расширять его, не
  плодить файлы) + возможно `tests/mm/`.
- Фикстуры: синтетический mmproj-gguf (методика
  [kb/harness], аналитические фикстуры, сидированные толерансы - см.
  уроки `ft-gguf-test-fixture-crafting`): полная карта имён, порядок
  осей, касты, отрицательные случаи (arch != clip, отсутствие ключа,
  лишний тензора).
- Конфиг: parse_gguf_config строит VisionConfig из метаданных; без
  mmproj - текст-only и реестр не меняется.
- Диспетчер iter_vision_weights: GGUF vs HF выбор ридера.
- Bug-fix правило CONTRIBUTING: тест, падающий до и проходящий после,
  на каждый исправленный дефект.

## Risks / open questions (решить в сессии, не тащить дальше)

1. `mm.post_norm.bias` (F32 4096) при том, что FreeToken post_layernorm
   - RMSNorm без bias. Варианты: у llama.cpp clip пост-норма LayerNorm;
   bias нулевой и это артефакт конвертера; bias принадлежит другой
   ноде графика. Решить сверкой с HF-референсом до маппинга.
2. F32/BF16 сплит 224/124: норм/скаляры F32 - ожидаемо; но mm.gate/up
   BF16 при BF16-ммproj - сверить, что merger-MLP в референсе тоже bf16.
3. Точка ветвления реестра по флагу (Фаза 1 п.2) - минимальный дизайн.
4. Место fallback-конфига процессора: в `mm/processor.py` (общий) или
   только в GLM-процессоре - выбрать по количеству правок.
5. VRAM: башня ~1.1 GB BF16-файла; при host-placement стримится, но
   patch_embed/downsample/merger резидентны - проверить бюджет в
   связке с hybrid-expert конфигом пользователя.

## Acceptance

- [x] CPU: все новые/расширенные тесты зелёные; существующие
  `../../../tests/models/test_glm5_next_gguf.py` и `tests/mm/` не сломаны.
- [x] CPU: отчёт числовой верификации всех 348 тензоров против
  NVFP4-референса (таблица рядом с брифом в kb/cases/gguf-glm5next-vision/), порог
  согласован.
- [x] CPU: parse_gguf_config собирает VisionConfig из синтетического
  mmproj; без флага - точное воспроизведение текущего поведения
  (текст-only, текущая спека реестра).
- [x] CPU: `ft checkpoint` GGUF-источника записывает в FTW все 348
  `visual.*` тензоров и метаданные башни; конфиг-путь бута из этого
  FTW строит башню без флага (HW e2e - фаза 5).
- [x] CPU: старый GGUF-FTW без vision не регрессирует (текст-only
  boot как сейчас).
- [x] HW (отдельная волна): boot + e2e image-запрос на raw .gguf (с
  флагом --mmproj) и на GGUF-FTW (башня из чекпойнта, без флага);
  текст-only boot без флага не изменился. PASS 2026-10-06: image-e2e OK
  на обоих путях, ответы совпали с NVFP4 - [phase-5-hw.md](phase-5-hw.md).
- [x] Коммиты: Conventional Commits, по фазе; никакого push.

## Чек-лист сессии реализации

1. Прочитать CONTRIBUTING.md.
2. Перепроверить якоря строк против текущего HEAD.
3. Прочитать kb: [findings/gguf-glm5next-path-a/phase2-tensor-map.md
   §11](../../findings/gguf-glm5next-path-a/phase2-tensor-map.md),
   [tensors-mmproj.txt](../../findings/gguf-glm5next-path-a/tensors-mmproj.txt),
   [llamacpp-tensor-kinds.md](../../findings/gguf-glm5next-path-a/llamacpp-tensor-kinds.md).
4. Решить открытые вопросы из Risks 1, 3, 4 ДО кода.
5. Фазы 1-4 с тестами; HW-волна 5 - только после упаковки сценариев и
   согласования с пользователем.