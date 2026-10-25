---
title: "Round 6 порт-долг: D2 mmproj-зрение, D4 контракт gguf-лоадера, D5 nvfp4-KV + честный A/B D3 после D5"
date: 2026-10-25
hardware: "GPU-волны (boot-smoke, CUDA/vision-тесты, A/B decode) - в cloud-окне; CPU-подготовка и тесты - в любое время"
commits: "база - d66f6e3 (round-6 merge, parents 23c1f02 + 32056cf); источники веточной функциональности - коммиты round-5 кампании: D2 7bc68b0, D3 393b682, D4 9314c9b, D5 660c814"
status: "ОТКРЫТА - фазы не начаты"
tags: [merge, upstream, round6, deferred-ports, D-PORT, mmproj, vision, gguf-loader, nvfp4, kv-quant, stale-pin]
---

# TASK: round 6 - порт-долг: D2/D4/D5 + честный A/B D3 после D5

Status: ОТКРЫТА 2026-10-25 - фазы не начаты. Три фазы порт-долга после round-6
merge: D2 (mmproj-зрение, выведен main-wins), D4 (контракт gguf-лоадера),
D5 (nvfp4-KV generic read-path) и фаза D3 - честный A/B decode строго ПОСЛЕ D5.
Кейс волны слияния и findings F1-F5 -
[round 6 merge](../merge-round6-upstream-main/TASK.md); прецедент порт-долга и
полные правила исполнения - [round-5 порт-долг](../round5-deferred-ports/TASK.md),
карта портов round-5 -
[round5-ggml-replatform-deferred-ports](../../topics/round5-ggml-replatform-deferred-ports.md);
процедура интеграционных волн -
[merge-wave-procedure](../../topics/merge-wave-procedure.md). Каждый порт -
отдельная кампания со своими волнами, приёмкой и серией коммитов; push - только
по явной команде пользователя.

## 0. Контекст: что произошло

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
  9314c9b / 660c814) и backup-ветка (`git show`). Артефакты волн round-6
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
- Skip-маркеры D-PORT снимаются ТОЛЬКО при исполнении соответствующей фазы
  и только вместе с зелёными тестами (счётчики совпадают).
- Git: conventional commits (одна строка, императив, lowercase, без точки);
  коммитить только по явным путям, перед коммитом сверять `git diff --cached`
  со рабочим деревом; НЕ делать push; служебные каталоги (.veai/) не трогать.
- Строчные якоря дрейфуют - перед правками перепроверять.

## 3. Фаза D2 - mmproj-зрение (выведен main-wins #803/#810; ветка 7bc68b0)

Цель: мультимодальный путь GGUF glm5next (изображения через --mmproj) снова
работает.

Что произошло: main принёс собственный mmproj-путь (#803/#810: Mmproj/hf_fields/
env, find_mmproj override); веточная конверсия и vision-serving gguf->FTW
(7bc68b0: checkpoint/convert.py, models/gguf/reader.py,
verify_mmproj_tensors.py) выведены как main-wins. Исключение, уже исполненное
в merge (порту не подлежит): веточный стриминг-sink gguf->FTW (layer_sink,
гетерогенные ширины слоёв) перепортирован на per-layer цикл main.

Шаги:

1. Вернуть веточную vision-механику на текущую структуру и снять 44 skip-маркера
   D-PORT-D2 в tests/models/test_glm5_next_gguf.py (mmproj/vision/image/
   mm_processor). Эталоны: round-5 D2 (7bc68b0: verify 347/347 exact) и кейс
   [gguf-glm5next-vision](../gguf-glm5next-vision/TASK.md); уроки -
   [round5-d2-mmproj-vision-port](../../topics/round5-d2-mmproj-vision-port.md).
2. Вернуть gguf->FTW vision packing: снять 4 skip-маркера D-PORT-1 в
   tests/checkpoint/test_ftw_gguf_banks.py (фикстура ftw_gguf_vision_ckpt:
   convert_checkpoint без mmproj_path; ImportError iter_gguf_vision_weights;
   test_convert_mmproj_non_clip_fails_fast).
3. Убрать громкий RuntimeError в models/glm5_next/weight.py:263: путь
   iter_vision_weights(gguf + mmproj_path) снова читает визуальную башню из
   --mmproj (сейчас: raise RuntimeError "deferred with the D2 mmproj port";
   в проде при main #810/#803 недостижим - потому merge и остался зелёным).
4. MMProj vision smoke #810: boot с --mmproj + e2e-запрос с картинкой.
   Прецедент round-5 D2: verify 347/347 exact, boot --mmproj READY ~142 с,
   e2e-ответ осмысленный.

Критерии приёмки: 44 + 4 маркеров сняты, счётчики замером совпадают, тесты
зелёные; vision-smoke PASS; контрольная батарея по FAILURE SET против базлайна
раздела 0.

## 4. Фаза D4 - контракт gguf-лоадера (main-wins #721/#810; ветка 9314c9b)

Цель: контракт загрузчика/конверсии glm5next-GGUF соответствует веточной
семантике (гарды, имена тензоров, типы), а не только зелёному буту.

Что произошло: веточный glm5next GGUF-лоадер (9314c9b) заменён main-овским
(#721); в tests/models/test_glm5_next_gguf.py размечено 37 skip-маркеров
D-PORT-D4 (loader/convert contract). Частично уже исполнено при merge:
семейный контракт ftw_tensor_types + 5 layer maps возвращены фиксом F1 в
register.py / glm5_next/gguf.py (см.
[кейс round 6](../merge-round6-upstream-main/TASK.md), finding F1).

Шаги:

1. Инвентаризация: diff ветки (9314c9b) против main-лоадера - какие гарды,
   имена и тип-контракты потеряны сверх уже возвращённого ftw_tensor_types.
2. Возвращать поэтапно: каждый контрак-тест размораживать с зелёным прогоном
   (падает-до / проходит-после), снимать маркеры D-PORT-D4 по мере покрытия.
3. Boot-smoke glm5next-GGUF (raw GGUF) + кросс-валидация весов по методике
   round-5 D4 (38/38 non-permuted, плотные corr >= 0.9998, банки в
   квант-допуске).

Критерии приёмки: 0 маркеров D-PORT-D4 (счётчик замером), тесты зелёные;
boot-smoke PASS; остаточных расхождений контракта нет (итог diff-ревью
зафиксирован в kb-статье фазы).

## 5. Фаза D5 - nvfp4-KV generic read-path (ветка 660c814)

Цель: вернуть read-путь nvfp4-KV и с ним nvfp4-KV стек 786k (эталон 27.71 tok/s;
текущий fp8-KV дефолт - 22.21 tok/s на 0.82/4095; рецепты и полосы decode -
[gguf-serving-perf-matrix](../../topics/gguf-serving-perf-matrix.md)).

Что произошло: main #732/#739 (BREAKING) пере-платформировал NVFP4-квант-механику;
read-path аргументы (pool_scale / kv_quant / pool_block_scale) на
glm_dsa_sparse_attn / dsa.py ампутированы (kv-механика и e4m3-хелперы живы,
wiring - нет); веточный generic read-path (660c814) на новое дерево не
перенесён.

Шаги:

1. Ре-ленд load_nvfp4 (decode / extend / split / paged) НА текущую структуру,
   сохраняя main-фичи: sinks, sliding-window, split-caps. Прецедент round-5 D5
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
зелёные; boot nvfp4-KV конфига PASS.

## 6. Фаза D3 - честный A/B decode (только ПОСЛЕ D5)

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
   элементы возвращать (slot pools / Fix-3 донесение / гейты; механика ветки
   описана в
   [hybrid-moe-slot-pools-port](../../topics/hybrid-moe-slot-pools-port.md)).
3. Вердикт зафиксировать в kb: статья фазы + обновление
   [gguf-serving-perf-matrix](../../topics/gguf-serving-perf-matrix.md).

Критерии приёмки: замер nvfp4-KV конфига на дереве round-6 зафиксирован
(конфиг + числа); вердикт ре-порт/NO-GO с обоснованием замером; матрица
обновлена.

## 7. Не входит (не-долг round 6)

- 2x glm_dsa OutOfResources - известный baseline с 2026-09-12, не регрессия
  round 6.
- test_gguf_expert_banks module-level skip - наследие round-5 ре-платформы
  ggml, не round 6.
- 33 skip в nvfp4_w4a4_linear / backends - CPU-skip новых main-тестов
  (GPU-only параметризация), не порт-долг.

## 8. Определение готовности кейса (по образцу round 5)

- Все снимаемые маркеры реально сняты, тесты зелёные, счётчики замером
  совпадают: D-PORT-D2 44 -> 0, D-PORT-1 4 -> 0, D-PORT-D4 37 -> 0,
  2 stale-pin теста перекалиброваны.
- Соответствующий smoke PASS: D2 - MMProj vision smoke #810 (boot --mmproj +
  e2e-картинка); D4 - boot-smoke glm5next-GGUF; D5 - boot nvfp4-KV конфига.
- A/B decode зафиксированы против 22.21 tok/s (fp8-KV, round 6) и 27.71 tok/s
  (nvfp4-KV, round-5 эталон); по D3-механике вынесен вердикт ре-порт/NO-GO.
- Статусы фаз обновлены в этом файле по мере исполнения; каждый порт - своя
  серия коммитов по правилам репо; push - только пользователь.
