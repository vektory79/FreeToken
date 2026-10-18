---
title: "Порт D4: glm5next GGUF-загрузчик на вендоренном ggml-стеке"
date: 2026-10-21
hardware: "RTX 5090 32GB (boot-smoke, кросс-валидация); CPU-волны (pytest)"
branch-commits: "vektory79 9314c9b (порт D4 поверх merge f0eed55); гард-эталон 9ac8830 (раунд 4)"
status: validated
tags: [gguf, glm5next, ggml, replatform, deferred-ports, loader, ftw-convert, mmq, cross-validation]
---

# Порт D4: glm5next GGUF-загрузчик на вендоренном ggml-стеке

**Вердикт (читайте это первым).** Порт D4 закрыт 2026-10-21 коммитом 9314c9b:
glm5next снова открывает и обслуживает .gguf после re-platform раунда 5
([round5-ggml-replatform-deferred-ports.md](round5-ggml-replatform-deferred-ports.md)).
GPU-приёмка: boot-smoke READY 145 s, chat 200, decode медиана 21.00 tok/s
(UD-Q3_K_XL, offload r3-рецепт), SIGTERM rc=143; кросс-валидация против
независимого NVFP4-FTW-референса - 38/38 пар без перестановок. Статья несёт
уроки порта; исполняемый план и приёмочные критерии фазы -
[../cases/round5-deferred-ports/TASK.md](../cases/round5-deferred-ports/TASK.md).

## Состав порта (коммит 9314c9b, 18 файлов)

- Загрузчик: [models/glm5_next/gguf.py](../../python/freetoken/models/glm5_next/gguf.py)
  (перепривязан на квант-пакет [layers/quantization/](../../python/freetoken/layers/quantization/)),
  model.py, __init__.py, регистрация в [models/register.py](../../python/freetoken/models/register.py).
- Чтение GGUF: [models/gguf/tensors.py](../../python/freetoken/models/gguf/tensors.py),
  experts.py, tokenizer.py, config.py - плотные тензоры и банки экспертов
  через вендоренный ридер, с сохранением per-layer ggml-типа.
- Конвертация GGUF->FTW: [checkpoint/convert.py](../../python/freetoken/checkpoint/convert.py),
  [checkpoint/ftw.py](../../python/freetoken/checkpoint/ftw.py) - gguf-ветка
  конвертера, layer_sink-стриминг, гард gguf_types при загрузке банков.
- Арифметика форматов: pad_rows/MMQ-хвост в
  [layers/quantization/ggml_types.py](../../python/freetoken/layers/quantization/ggml_types.py);
  потребители - linear/gguf.py, embedding/gguf.py.
- Схемы offload-кэша: дублирующийся ключ gguf в _BANK_SCHEMAS
  [moe/offload_cache.py](../../python/freetoken/moe/offload_cache.py) устранён.
- Тесты: [tests/models/test_glm5_next_gguf.py](../../tests/models/test_glm5_next_gguf.py)
  (пересажен на новый API), [tests/checkpoint/test_ftw_gguf_banks.py](../../tests/checkpoint/test_ftw_gguf_banks.py);
  старый набор [tests/models/test_gguf_expert_banks.py](../../tests/models/test_gguf_expert_banks.py)
  помечен module-skip до ре-ленда expert-banks API (отдельная волна).

## Урок 1. Загрузчик пережил re-platform: ампутация импортов, не потеря

Основной файл glm5_next/gguf.py пережил merge f0eed55 целиком - re-platform
ампутировал только импорты на удалённые файлы (models/gguf/dequant.py,
kernel/gguf.py). Не надо было переписывать загрузчик - надо было перепривязать
его на новые поверхности layers/quantization/*. Вывод для будущих портов:
прежде чем планировать переписывание, сравнить размер выжившего файла с
диффом merge - "сырой ImportError при выборе модели" означал здесь ровно
семь сломанных импортов, а не отсутствие функциональности.

## Урок 2. Дублирующийся ключ dict молча затирает схему (_BANK_SCHEMAS)

_BANK_SCHEMAS в moe/offload_cache.py содержал два ключа "gguf": один описывал
merged-раскладку gate_up+down, другой - three-separate. Python молча оставляет
последний, и обе схемы по отдельности валидны, поэтому юнит-тесты это не ловили:
импорт модуля уже затёр первую схему до любого теста. Boot нашёл за секунды на
реальном чекпойнте (волна 14: boot-smoke заблокирован B1). Урок: словари схем со
строковыми ключами формата проверять на дубли ключей - статически (флейк-тест на
модуль) или assert'ом при объявлении; "тесты зелёные" не гарантирует, что схема
та, которую вы думали.

## Урок 3. GPU-ветка плотных кастов: .to(cuda) до op_dequantize (kDLCUDA)

Дефолтное чтение плотных GGUF-тензоров даёт CPU-mmap массивы. Перед вызовом
деquantизатора [kernel/ggml.py](../../python/freetoken/kernel/ggml.py) тензор
должен лежать на CUDA: op_dequantize требует буфер с kDLCUDA. Первая версия
делала каст по типу устройства входа (CPU вход - CPU выход) и GPU-ветка была
мёртвой (Review-B, major). Фикс: явный .to(запрошенное устройство) до вызова
деquantизации; рантайм-проверка ветки выполнена в GPU-окне приёмки. Урок:
device-полиморфные хелперы проверять на фактическом устройстве-потребителе, а
не на устройстве-источнике.

## Урок 4. pad_rows нативного MMQ-хвоста: ceil(tail/row_bytes), не одна строка

Нативный MMQ требует, чтобы хвост матрицы был добит до целых строк. Правильная
формула паддинга: ceil(tail_bytes / row_bytes) строк, а не фиксированная одна
строка (символ pad_rows в
[layers/quantization/ggml_types.py](../../python/freetoken/layers/quantization/ggml_types.py),
константа MATRIX_ROW_PADDING рядом). Заниженный паддинг сдвигал чтение хвоста
за пределы банки и проявлялся только на реальном чекпойнте, где хвосты не
выравнены удачно (волна 15, B4). Потребители формулы: linear/gguf.py,
embedding/gguf.py, models/gguf/{tensors,experts}.py - все обязаны считать
паддинг одной функцией, не локальной арифметикой.

## Урок 5. GGUF->FTW: merged-имена, gguf_types-тройки, layer_sink-стриминг

Конвертер GGUF->FTW ([checkpoint/convert.py](../../python/freetoken/checkpoint/convert.py))
должен сохранять: fused-имена и piece-order GGUF (in_proj как q|k|v|beta|f_a|g_a,
kv_b как транспонированную конкатенацию k/v), тройки gguf_types на банки
экспертов - по ним гард в load_ftw_banks_to_device (семантика коммита 9ac8830,
урок раунда 4 - [merge-wave-procedure.md](merge-wave-procedure.md)) отличает
банки от чужих payload'ов. Слой-приёмник (layer_sink) наполняется стримингом
тензор за тензором: torch.cat требует единые dtype/форму и падает на
разнотипных входах - стриминг снимает требование (волна 15, B2: gguf-ветка
конвертера падала с LEGACY_FORMAT KeyError и собиралась cat'ом). Контракт
тихих сканов FTW при загрузке - [ftw-load-path.md](ftw-load-path.md).

## Урок 6. pip-пакет gguf==0.19.0 в venv нужен только фикстурам

Рантайм читает GGUF вендоренным ридером (модули models/gguf/). Пакет gguf
0.19.0, установленный в venv, нужен только тестовым фикстурам -
генерации синтетических GGUF-файлов в тестах загрузчика и конвертера. Без
пакета фикстуры падают на импорте, рантайм - нет; наоборот, обновление пакета
не меняет поведение прод-чтения. Полезно помнить при разборе "почему тесты
падают у одного из команды".

## Урок 7. Кросс-валидационная методика: corr-диапазоны по битности

Метод: пары имён GGUF<->FTW деquantизируются в float32 независимо (GGUF -
вендорным quants.dequantize, NVFP4-FTW - LUT e2m1 + per-16 scales + per-row
fp16 globals), сравниваются форма / Pearson corr / maxabs / rmse. На реальной
паре чекпойнтов (unsloth GLM-5.3-Flash-GGUF UD-Q3_K_XL против RedHatAI
GLM-5.3-Flash-NVFP4-FTW, 1412 GGUF-тензоров / 1607 FTW-записей) проверено
38 пар, форма сошлась везде. Интерпретация порогов:

- байт-идентичные F32 (нормы, роутер, conv1d, indexer-проекции): corr 1.000000,
  maxabs 0 - доказательство побайтового маппинга имён;
- плотные кванты (Q8_0 линейные, Q6_K lm_head): corr >= 0.9998 - сегментный
  путь чтения корректен;
- эксперт-банки: corr 0.980 (gate/up, IQ3_XXS ~3.3 bpw) и 0.9927-0.9929 (down,
  IQ4_XS ~4.5 bpw), монотонно по битности; e_score_bias 0.9866 - 288-вектор
  Q8_0 с почти нулевой дисперсией, corr гиперчувствителен.

Под-0.99 значения - это допуск ДВУХ независимых квантований одного bf16
источника, а не перестановка: ошибка оси, nibble-порядка или роли куска рушит
corr до ~0, потому что веса позиционно некоррелированы. Полный разбор -
verification-отчёт кампании (волна 14, GPU-окно).

## Приёмка

- Boot-smoke (RTX 5090, волна 15): рецепт - --moe-strategy offload
  --moe-cache-auto --disable-moe-prefill-overlap --kv-reserve-tokens 262144
  --kv-quant nvfp4 --memory-ratio 0.85 --max-prefill-length 4096
  --max-running-requests 1, порт 18081. READY 145 s; POST /v1/chat/completions
  200 (digit-split ответы корректны); decode медиана 21.00 tok/s (медиана
  полных decode-интервалов); SIGTERM rc=143; процесс-цензура чистая.
- Ориентиры старого стека: offload 12.97 tok/s, гибрид 16.67 (эталон 63b9bff).
  Сравнение не apples-to-apples (другой стек), но факт: offload на новом стеке
  быстрее старого базлайна.
- Батарея: CUDA-visible - 6 failed известного союза / 2665 passed;
  CPU-hidden - 2 failed базлайн-пары / 2059 passed. Срез STOP GATE
  (D3-файлы застэшены): trio loader+convert+cache-budget 163 passed /
  24 skipped, 0 failed; смок cpu_moe_gguf - skipped без ошибок.

## Открытое

- D3 (гейты гибрида/offload): CPU-готов, ждёт GPU-приёмки и своего коммита;
  docs-строки о гейтах (docs/cli.md, docs/models.md) едут с D3-коммитом.
- _fuse_kv_b геометрия конвертера отложена за pinned-конвертер.
- tests/models/test_gguf_expert_banks.py в module-skip до ре-ленда
  expert-banks API.
- D1-микробенчи (mhc NS, dense q8_0 N-split) - фаза 3 кампании.
