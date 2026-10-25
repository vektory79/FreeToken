---
title: "\"Round 6 порт-долг: ре-имплементация D2/D4/D5 поверх main + честный A/B D3 после D5\""
date: 2026-10-25
hardware: "\"GPU-волны (boot-smoke, CUDA/vision-тесты, A/B decode) - в cloud-окне; CPU-подготовка и тесты - в любое время\""
commits: "\"база - d66f6e3 (round-6 merge, parents 23c1f02 + 32056cf); источники веточной функциональности - коммиты round-5 кампании: D2 7bc68b0, D3 393b682, D4 9314c9b, D5 660c814\""
status: "\"ОТКРЫТА - фазы не начаты\""
tags: [merge, upstream, round6, deferred-ports, D-PORT, mmproj, vision, gguf-loader, nvfp4, kv-quant, stale-pin, no-dual-logic]
---

# TASK: round 6 - порт-долг: ре-имплементация D2/D4/D5 поверх main + честный A/B D3 после D5

Status: ОТКРЫТА 2026-10-25 - фазы не начаты. Три фазы порт-долга после round-6
merge: D2 (mmproj-зрение), D4 (контракт gguf-лоадера), D5 (nvfp4-KV generic
read-path) и фаза D3 - честный A/B decode строго ПОСЛЕ D5. Рамка всего кейса -
директива пользователя 2026-10-25 о запрете двойных логик (раздел 0): веточный
код - референс-спека, каждый портированный элемент ре-имплементируется ПОВЕРХ
main-примитивов, тесты - без семантических дублей. Кейс волны слияния и
findings F1-F5 - [round 6 merge](../merge-round6-upstream-main/TASK.md);
прецедент порт-долга и полные правила исполнения -
[round-5 порт-долг](../round5-deferred-ports/TASK.md), карта портов round-5 -
[round5-ggml-replatform-deferred-ports](../../topics/round5-ggml-replatform-deferred-ports.md);
процедура интеграционных волн -
[merge-wave-procedure](../../topics/merge-wave-procedure.md). Каждый порт -
отдельная кампания со своими волнами, приёмкой и серией коммитов; push - только
по явной команде пользователя.

## 0. Контекст: что произошло

- Директива пользователя 2026-10-25 (рамка всего документа): "Ты хочешь оставить
  ДВЕ логики одновременно? Из main и из моей? Так не пойдёт!". Правила: если
  механика уже есть в main - остаётся ТОЛЬКО логика main (main-wins тотален);
  веточная функциональность либо ИСПОЛЬЗУЕТ новую логику main, либо
  РЕ-ИМПЛЕМЕНТИРУЕТСЯ поверх неё как дельта; веточные коммиты - референс-спека,
  не перенос verbatim; тесты - тоже без дублей, дедуп СЕМАНТИЧЕСКИЙ (по
  утверждениям/гардам/кейсам, не по названиям) - см. раздел 3.
- 2026-10-24/25 выполнен round-6 merge: upstream/main 32056cf (38 коммитов) ->
  vektory79; merge 594934d -> amend 220b79e -> amend d66f6e3 (parents 23c1f02 +
  32056cf); backup со старым стеком: `backup/vektory79-pre-merge6-23c1f02`.
  Режим по директиве пользователя: минимальный зелёный merge + deferred-ports -
  merge-коммит даёт зелёные сборку+тесты на механиках main, глубокая адаптация
  отложена в порт-долг. Детали -
  [round 6 merge](../merge-round6-upstream-main/TASK.md).
- Раунд re-platform-подобный: main пере-платформировал NVFP4-квант-механику
  (#732/#739 BREAKING, #737 w4a4 sm_120), hybrid fetch (#778 chunked eviction,
  #742 pinned staging) и принёс собственные mmproj-пути (#803/#810) и
  gguf-лоадер (#721), заменившие веточные порты round-5. Выведенная веточная
  функциональность помечена skip-маркерами с именованным портом: сюиты зелёные,
  долг учтён в этом файле.
- Источники кода: веточные коммиты round-5 кампании (7bc68b0 / 393b682 /
  9314c9b / 660c814) и backup-ветка (`git show`) - как РЕФЕРЕНС-СПЕКА, не как
  переносимый код. Артефакты волн round-6
  (`.tasks/merge-main-round6-work/`) в kb не зеркалированы; суть дистиллирована
  в [кейсе round 6](../merge-round6-upstream-main/TASK.md).
- Тест-базлайн merged-дерева (CPU, d66f6e3): tests/checkpoint 69 passed /
  13 skipped; tests/engine/test_cache_budget 51 passed / 2 skipped;
  tests/models/test_glm5_next_gguf 35 passed / 91 skipped (замер grep-ом:
  D-PORT-D2 x44, D-PORT-D4 x37); консолидированный прогон 155 passed /
  16 skipped / 0 failed. GPU-порции: 411 passed / 2 failed (2x glm_dsa
  OutOfResources - baseline с 2026-09-12, не долг) / 33 skipped (см. "Не
  входит"). Новые волны сравнивать по FAILURE SET против этого базлайна,
  не по абсолютным числам.

## 1. Опорные точки и базлайны

- Boot-smoke round 6 (матричный дефолт, дерево d66f6e3): GLM-5.3-Flash-UD-Q3_K_XL
  (FTW), fp8-KV, memory-ratio 0.82 / max-prefill-length 4095 / kv-reserve 400k,
  hybrid; READY 51 с, decode медиана 22.21 tok/s - верх полосы матрицы
  [gguf-serving-perf-matrix](../../topics/gguf-serving-perf-matrix.md),
  регрессии нет. Базлайн A/B всех фаз.
- Эталон nvfp4-KV стека (round-5, коммит 393b682): hybrid 786k nvfp4-рецепт
  (kv-reserve 786432, kv-cache-dtype nvfp4, memory-ratio 0.85,
  max-prefill-length 4096, moe-strategy hybrid, moe-cpu-threads 16),
  decode 27.71 tok/s. Сейчас этот стек невоспроизводим: nvfp4-KV read-path
  ампутирован (порт D5), hybrid-механика заменена main-ом (порт D3).
- 22.21 (fp8-KV) против 27.71 (nvfp4-KV) напрямую несравнимы - разные KV-типы
  И разные hybrid-механики; честное сравнение возможно только после D5
  (фаза D3).
- Готовность сервера проверять реальным chat-запросом, а не дешёвым эндпоинтом
  (/v1/models отвечает до того, как движок дочитал веса).

## 2. Правила исполнения (кратко; полные - в прецеденте round-5)

- VRAM-протокол: GPU-волны (boot-smoke, CUDA-тесты, A/B) - в cloud-окне,
  сериализованно, по одному серверу/прогону; CPU-работа - в любое время.
- Тесты только `uv run --extra dev python -m pytest`; доверие результатам
  проверять `freetoken.__file__` (uv editable-ловушка); сравнение по
  FAILURE SET, счётчики skip-маркеров замерять grep-ом на актуальном дереве,
  не по памяти (урок round-6: заявка 30/51 против факта 44/37).
- Каждая фаза начинается с шага 0 - классификации её skip-маркеров по правилу
  недублирования (раздел 3), до любых правок кода.
- Skip-маркеры D-PORT снимаются ТОЛЬКО при исполнении соответствующей фазы
  и только вместе с зелёными тестами (счётчики совпадают).
- Git: conventional commits (одна строка, императив, lowercase, без точки);
  коммитить только по явным путям, перед коммитом сверять `git diff --cached`
  со рабочим деревом; НЕ делать push; служебные каталоги (.veai/) не трогать.
- Строчные якоря дрейфуют - перед правками перепроверять.

## 3. Правило недублирования и работа с тестами

Обязательный шаг 0 каждой фазы - классификация её skip-маркеров пробным
прогоном на текущем дереве. Пять исходов для каждого маркера:

- (a) тест зелёный на main-механике -> снять маркер (он пиннел main-поведение);
- (b) семантический дубль main-теста (та же семантика: утверждения, гарды,
  кейсы - не имена) -> веточный тест УДАЛИТЬ, остаётся main; пару задокументировать;
- (c) тест пиннит веточную семантику вопреки осознанному main-дизайну ->
  ПЕРЕПИСАТЬ тест под main;
- (d) кейс ветки, НЕ покрытый в main -> ДОИМПЛЕМЕНТИРОВАТЬ (адаптировать на
  main-механику), затем снять маркер;
- (e) тест пиннит отсутствующую дельту -> остаётся до реализации дельты
  соответствующей фазой.

Итог фазы - ПОЛНАЯ карта маркеров: каждый учтён (un-skipped / удалён /
переписан / доимплементирован / отложен), счётчики grep-ом совпадают с картой.
Критерий "нет файлов с двумя логиками" - grep-ревью каждой фазы: в диффе фазы
не появляется вторая реализация того, что main уже делает, и не остаётся
закомментированной веточной механики. Дедуп - семантический, по кластерам
(мм-config-гарды / reader name-map + axis / fails-fast гарды / engine-config
резолв / mm-процессор / iter_vision проводка / FTW-carrier packing /
image-serving); кластерные карты в этом файле - стартовые гипотезы,
финальные карты фиксируются в kb-статье фазы.

## 4. Фаза D2 - mmproj-зрение (main-wins #803/#810; ветка 7bc68b0 - референс)

Цель: мультимодальный путь работает НА main-механике; веточные дубли удалены;
единственная настоящая дельта - gguf->FTW vision packing, ре-имплементированная
поверх main.

Факт-база дерева (проверена, якоря по состоянию d66f6e3; перед правками
перепроверять): main mmproj-флоу жив ПОЛНОСТЬЮ -
`python/freetoken/models/gguf/mmproj.py` (Mmproj:32, `_find`:80 - override
--mmproj, `find_mmproj`:121 + auto-discovery);
`python/freetoken/models/glm5_next/gguf.py` (`gguf_mm_config`:284 с гвардами
projector / второй patch-эмбеддинг / пропорций, `_vision_name`:354,
`iter_mmproj_weights`:362); `iter_gguf_weights`:403-417 уже стримит vision
после транка (`yield from iter_mmproj_weights(find_mmproj(model_path))`), kwarg
`include_vision` в glm5_next подтверждён (:409);
`python/freetoken/engine/config.py` (mmproj_path:126, трэдинг в hf_config:149,
vision_config как реальное поле шима:181);
`python/freetoken/engine/engine.py`:536 (`get_mm_processor(..., mmproj_path=)`);
`python/freetoken/mm/processors/glm5_next.py` (76 строк, image-процессор читает
hf_config.vision_config / image_token_id). Мёртвая точка проводки:
`python/freetoken/models/glm5_next/weight.py` - `iter_vision_weights`:241
хард-гвардит mmproj_path (253-262), затем 263: RuntimeError "gguf vision serving
is not wired in this build (deferred with the D2 mmproj port)"; HF-путь
(`_ShardReader`/`_iter_vision`) жив. Диспетчер - models/weight.py:258.

Классификация веточного кода по правилу раздела 3:

- ДУБЛИРУЕТ main (не ре-имплементировать): веточные `iter_gguf_vision_weights`
  и reader (name map `v.blk.N.*`/`mm.*`, гарды unknown-tensor / non-clip /
  missing-block) - семантически покрываются main `iter_mmproj_weights` +
  `gguf_mm_config` + гардами mmproj.py. weight.py:263 - не функциональность,
  а отсутствующая проводка на main-механизм.
- ОТСУТСТВУЕТ в main (единственная настоящая дельта D2): gguf->FTW vision
  packing. `python/freetoken/checkpoint/convert.py`:249-252 (main #803):
  `include_vision = gguf_config_source(model_path) is None` -> для GGUF-источника
  vision в FTW НЕ пакуется ("an FTW keeps only a GGUF's text metadata");
  `convert_checkpoint` (convert.py:184) без kwarg mmproj_path. Инфраструктура
  кэрриера в main ЕСТЬ: `python/freetoken/models/gguf/reader.py`:85-109,
  `_pack_kv`:112 (appends без проверки коллизий исходных ключей). Остаток
  ветки: коммент convert.py:343-345 ("merge the mmproj's clip.* tower facts...
  USER DECISION 4", blame 7bc68b0) без кода под ним - закрыть кодом или удалить.
- Решение пользователя по gguf->FTW vision packing: РЕ-ИМПЛЕМЕНТИРОВАТЬ ПОВЕРХ
  MAIN (дополнительная функциональность, её в main нет).

Шаги:

1. Шаг 0: классификация 44 skip-маркеров D-PORT-D2 (tests/models/
   test_glm5_next_gguf.py) + 4 skip-маркеров D-PORT-1
   (tests/checkpoint/test_ftw_gguf_banks.py: test_ftw_gguf_vision_tensors_
   roundtrip:476, test_ftw_gguf_vision_metadata_carrier:498,
   test_ftw_gguf_vision_boot_without_mmproj:520,
   test_convert_mmproj_non_clip_fails_fast:711; фикстура ftw_gguf_vision_ckpt:458
   вызывает convert_checkpoint(..., mmproj_path=mmproj)) - по правилу раздела 3.
   Стартовая гипотеза (кластерная карта, не по-тестно все 44): ~24 теста
   (mm_config-гарды / reader name-map + axis orders / fails-fast гарды /
   engine-config spec-резолв / mm-процессор) семантически ложатся на живую
   main-механику -> исходы (a)/(b); 4 D-PORT-1 + 2 carrier-теста D2
   (test_ftw_carrier_boots_vision_without_mmproj:2776,
   test_ftw_carrier_partial_vision_metadata_fails_fast:2827) пиннят
   отсутствующий packing -> исход (e) до шага 3; test_iter_gguf_weights_
   include_vision_contract:2087 / streams_vision_after_trunk:2444 /
   include_vision_requires_mmproj:2464 - kwarg include_vision в main glm5_next
   подтверждён (gguf.py:409), ожидание исхода (a) на пробном прогоне. Сопоставление
   с main-тестами mmproj-флоу (пришли с #803/#810/#721; среди немаркированных:
   test_mmproj_boot_carries_image_token_and_processor_path:2647,
   test_mm_processor_fallback_builds_image_processor_from_mmproj:2680,
   test_mm_processor_mmproj_boot_wraps_image_into_mm_item:2723 и сюита
   tests/checkpoint/test_ftw_hotfix_vision.py) - пары фиксировать в карте фазы.
2. weight.py:263 - заменить RuntimeError на проводку: GGUF-ветка
   `iter_vision_weights` делегирует main `iter_mmproj_weights(find_mmproj(...))`
   через mmproj_path (делегирование на main-примитив, НЕ возврат веточного
   reader/`iter_gguf_vision_weights`). Снять исходы (a)/(b)/(c) из шага 0
   зелёным прогоном.
3. Дельта packing - ре-имплементация поверх main: тензоры - из main
   `iter_mmproj_weights`; кэрриер - через main `_pack_kv` (reader.py:112);
   точка включения - kwarg mmproj_path в `convert_checkpoint` + пересмотр гейта
   include_vision (convert.py:249-252); вопрос коллизий исходных ключей решить
   в main-примитиве `_pack_kv` (веточный writer имел collision-гвардию - ценный
   кейс для доимплементации, исход (d)). Референс-спека: веточный 7bc68b0
   (writer-семантика, collision-гарды, verify 347/347 exact); уроки -
   [round5-d2-mmproj-vision-port](../../topics/round5-d2-mmproj-vision-port.md),
   кейс [gguf-glm5next-vision](../gguf-glm5next-vision/TASK.md). Коммент
   convert.py:343-345 закрыть кодом или удалить.
4. MMProj vision smoke #810: boot с --mmproj + e2e-запрос с картинкой.
   Прецедент round-5 D2: verify 347/347 exact, boot --mmproj READY ~142 с,
   e2e-ответ осмысленный.

Критерии приёмки: карта тестов полная (44 + 4, пять исходов, счётчики grep-ом
совпадают); семантические дубли удалены (остаётся main); packing работает
поверх main - FTW-vision carrier конвертируется (convert_checkpoint c
mmproj_path) и бутится без --mmproj; vision-smoke PASS; контрольная батарея по
FAILURE SET против базлайна раздела 0; grep-ревью "нет файлов с двумя логиками".

## 5. Фаза D4 - контракт gguf-лоадера (main-wins #721/#810; ветка 9314c9b - референс)

Цель: гарды и контракты, добавляющие ценность СВЕРХ main-лоадера, ре-
имплементированы на main-механике; всё, что семантически покрывается main-гардами
и main-тестами, - удалено без замены.

Что произошло: веточный glm5next GGUF-лоадер (9314c9b) заменён main-овским
(#721); в tests/models/test_glm5_next_gguf.py размечено 37 skip-маркеров
D-PORT-D4 (loader/convert contract). Частично уже исполнено при merge:
семейный контракт ftw_tensor_types + 5 layer maps возвращены фиксом F1 в
register.py / glm5_next/gguf.py (см.
[кейс round 6](../merge-round6-upstream-main/TASK.md), finding F1) - фрагмент
считается исполненным, в карте фазы пометить un-skipped/возвращён.

Шаги:

1. Шаг 0: классификация 37 маркеров по правилу раздела 3 - семантический дедуп
   с main-тестами лоадера #721: гарды/контракты, которые main уже даёт
   (семантически), -> исход (b) удалить веточный тест; пин веточной семантики
   там, где main ведёт себя иначе осознанно, -> исход (c) переписать под main;
   непокрытые в main кейсы, добавляющие ценность, -> исход (d) доимплементировать
   на main-примитивах; зелёные -> (a).
2. Возвращать поэтапно: каждый отобранный контрак-тест размораживать с зелёным
   прогоном (падает-до / проходит-после), снимая маркеры D-PORT-D4 по мере
   покрытия.
3. Boot-smoke glm5next-GGUF (raw GGUF) + кросс-валидация весов по методике
   round-5 D4 (38/38 non-permuted, плотные corr >= 0.9998, банки в
   квант-допуске).

Критерии приёмки: карта тестов полная (37 маркеров, пять исходов, счётчики
grep-ом совпадают); возвращены только гарды/контракты сверх main-гардов,
каждый сидит на main-примитивах; тесты зелёные; boot-smoke PASS; остаточных
расхождений контракта нет (итог diff-ревью зафиксирован в kb-статье фазы);
grep-ревью "нет файлов с двумя логиками".

## 6. Фаза D5 - nvfp4-KV generic read-path (ветка 660c814 - референс-спека)

Цель: nvfp4-KV read-path ре-имплементирован на новой квант-механике main
(#732/#739: fp32 global scales, weight scales); nvfp4-KV стек 786k снова
воспроизводим (эталон 27.71 tok/s; текущий fp8-KV дефолт - 22.21 tok/s на
0.82/4095; рецепты и полосы decode -
[gguf-serving-perf-matrix](../../topics/gguf-serving-perf-matrix.md)).

Что произошло: main #732/#739 (BREAKING) пере-платформировал NVFP4-квант-механику;
read-path аргументы (pool_scale / kv_quant / pool_block_scale) на
glm_dsa_sparse_attn / dsa.py ампутированы (kv-механика и e4m3-хелперы живы,
wiring - нет); веточный generic read-path (660c814) на новое дерево не
перенесён.

Шаги:

1. Ре-имплементация nvfp4-KV read-path НА новой квант-механике main #732/#739
   (fp32 global scales, weight scales): decode / extend / split / paged;
   ветка 660c814 - референс-спека (None-шкалы байт-в-байт как контракт);
   НЕ ре-ленд verbatim - каждый элемент сидит на main-квант-примитивах.
   Сохраняя main-фичи: sinks, sliding-window, split-caps. Прецедент round-5 D5
   (None-шкалы байт-в-байт); уроки -
   [round5-d5-nvfp4-kv-generic](../../topics/round5-d5-nvfp4-kv-generic.md).
2. Перекалибровать 2 stale-pin теста tests/models/test_quant_config.py:
   test_hybrid_explicit_kernel_error_names_the_usable_alternative и
   test_nvfp4_auto_selection_still_lands_on_triton. Веточные пины ожидали
   "usable here: triton"; main #737 добавил nvfp4 w4a4 на sm_120 -> корректное
   ожидание "usable here: w4a4, triton". Разморозка - с новым пином и зелёным
   прогоном.
3. Включать supports_nvfp4_kv точечно по бэкендам, каждый - с тестом.
4. Качественный A/B без деградации (прецедент round-5: greedy 11/12
   byte-identical на Qwen3.8-27B-NVFP4, 1 argmax-flip задокументирован).

Критерии приёмки: nvfp4-KV стартует без reject-а на целевом бэкенде; оба
stale-pin теста зелёные с новой формулировкой; тесты kv_nvfp4 / kv_quant
зелёные; boot nvfp4-KV конфига PASS; read-path не дублирует main-квант-механику,
а сидит на ней (grep-ревью "нет файлов с двумя логиками").

## 7. Фаза D3 - честный A/B decode (только ПОСЛЕ D5)

Цель: решить замером, нужен ли ре-порт веточной hybrid-механики (393b682:
slot pools, Fix-3 донесение, fail-fast гейты) на main-механику (#778 chunked
eviction, #742 pinned staging), или NO-GO. Прецедент round-5 D1: старые
выигрыши не переносятся через re-platform - решает замер
([round5-d1-kernel-wins-no-go](../../topics/round5-d1-kernel-wins-no-go.md)).

Почему сейчас сравнивать нельзя: fp8-KV 22.21 tok/s (round 6) против nvfp4
27.71 tok/s (round-5 эталон) - разные KV-типы и разные hybrid-механики;
разница не атрибуируется.

Шаги:

1. После D5 поднять nvfp4-KV конфиг (786k-рецепт из раздела 1, насколько
   влезает в текущую арифметику cache-budget) на дереве round-6.
2. A/B против обеих точек: 22.21 tok/s (fp8-KV, текущее дерево) и 27.71 tok/s
   (nvfp4, round-5 эталон). Если main-механика на nvfp4-KV >= эталона -
   ре-порт D3 не нужен (NO-GO); если провал - картировать, какие веточные
   элементы возвращать (механика ветки описана в
   [hybrid-moe-slot-pools-port](../../topics/hybrid-moe-slot-pools-port.md)).
3. Если вердикт за ре-порт: ре-имплементация отобранных элементов (slot pools /
   Fix-3 донесение / fail-fast гейты) НА main-механике #778/#742, НЕ возврат
   393b682 - каждый элемент сидит на main-примитивах, по правилу раздела 3.
4. Вердикт зафиксировать в kb: статья фазы + обновление
   [gguf-serving-perf-matrix](../../topics/gguf-serving-perf-matrix.md).

Критерии приёмки: замер nvfp4-KV конфига на дереве round-6 зафиксирован
(конфиг + числа); вердикт ре-порт/NO-GO с обоснованием замером; при ре-порте -
карта элементов (каждый - дельта на main-примитивах) и grep-ревью "нет файлов
с двумя логиками"; матрица обновлена.

## 8. Не входит (не-долг round 6)

- 2x glm_dsa OutOfResources - известный baseline с 2026-09-12, не регрессия
  round 6.
- test_gguf_expert_banks module-level skip - наследие round-5 ре-платформы
  ggml, не round 6.
- 33 skip в nvfp4_w4a4_linear / backends - CPU-skip новых main-тестов
  (GPU-only параметризация), не порт-долг.

## 9. Определение готовности кейса (по образцу round 5)

- Полная карта маркеров каждой фазы: все skip-маркеры классифицированы по
  пяти исходам раздела 3 (un-skipped / удалён / переписан / доимплементирован /
  отложен), счётчики замером совпадают: D-PORT-D2 44 -> 0, D-PORT-1 4 -> 0,
  D-PORT-D4 37 -> 0, 2 stale-pin теста перекалиброваны.
- Семантический дедуп завершён: нет пар веточный-тест/main-тест с одинаковой
  семантикой (дедуп по утверждениям/гардам/кейсам, не по названиям); пары
  (удалён -> остался main) зафиксированы в картах фаз.
- Нет файлов с двумя логиками: каждый портированный элемент сидит на
  main-примитивах, grep-ревью каждой фазы без находок.
- Соответствующий smoke PASS: D2 - MMProj vision smoke #810 (boot --mmproj +
  e2e-картинка) + FTW-vision carrier бутится без --mmproj; D4 - boot-smoke
  glm5next-GGUF; D5 - boot nvfp4-KV конфига.
- A/B decode зафиксированы против 22.21 tok/s (fp8-KV, round 6) и 27.71 tok/s
  (nvfp4-KV, round-5 эталон); по D3-механике вынесен вердикт ре-порт/NO-GO.
- Статусы фаз обновлены в этом файле по мере исполнения; каждый порт - своя
  серия коммитов по правилам репо; push - только пользователь.