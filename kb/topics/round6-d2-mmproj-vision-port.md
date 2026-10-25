---
title: "Порт D2 round 6: mmproj-зрение ре-имплементировано поверх main - CPU-часть готова"
date: 2026-10-25
hardware: "RTX 5090 32GB (GPU-приёмка - в cloud-окне); CPU - контрольная батарея и верификация фикстур"
branch-commits: "vektory79: 523452b + e1de1df (HEAD) поверх d66f6e3 (round-6 merge, parents 23c1f02 + 32056cf); референс-спека - веточный 7bc68b0"
status: "cpu-done - GPU-smoke #810 и verify 347/347 ожидают cloud-окно"
tags: [gguf, mmproj, vision, ftw, carrier, round6, deferred-ports, main-wins, no-dual-logic]
---

# Порт D2 round 6: mmproj-зрение поверх main - CPU-часть готова

**Вердикт (читайте это первым).** Фаза D2 кампании [round 6 порт-долг](../cases/round6-deferred-ports/TASK.md)
закрыта в CPU-части 2026-10-25 двумя коммитами: 523452b ("feat(models): serve
gguf mmproj vision through the main mmproj path"; 8 файлов, +337/-242) и
e1de1df ("feat(checkpoint): pack gguf mmproj vision into ftw carriers on
convert"). Из 48 skip-маркеров (44 D-PORT-D2 + 4 D-PORT-1) не осталось ни
одного: 38 переписаны под main, 2 доимплементированы, 8 сняты после
ре-имплементации суставов; пар семантических дублей с main-тестами нет.
Финальная батарея двух файлов сюит: 99 passed / 47 skipped / 0 failed.
Единственная настоящая дельта D2 - gguf->FTW vision packing - сидит на
main-примитивах; веточные reader и iter_gguf_vision_weights удалены, остаётся
только main-логика. Прецедент round-5 - [round5-d2-mmproj-vision-port.md](round5-d2-mmproj-vision-port.md);
карта раунда - [round5-ggml-replatform-deferred-ports.md](round5-ggml-replatform-deferred-ports.md).

## Что произошло

Round-6 merge принёс upstream-овские собственные mmproj-пути (#803 - mmproj-модель
и convert-проводка, #810 - image serving) и gguf-лоадер, заменившие веточный
порт D2 round-5 (7bc68b0). По директиве о запрете двойных логик веточная
функциональность была выведена и помечена skip-маркерами D-PORT-D2 (44 в
tests/models/test_glm5_next_gguf.py) и D-PORT-1 (4 в
tests/checkpoint/test_ftw_gguf_banks.py). Факт-база merged-дерева показала:
main mmproj-флоу жив полностью (модель, find_mmproj, image-процессор,
iter_mmproj_weights, конфиг-гварды), а веточный код дублировал его
семантически. Уникальный невосстановимый из main элемент - только
gguf->FTW vision packing: main пакует в FTW-носитель метаданные GGUF без
тензоров и без vision ("an FTW keeps only a GGUF's text metadata").

## Что ре-имплементировано поверх main (суставы)

Все стыки посажены на main-примитивы, без возврата веточных API:

- **Проводка iter_vision_weights.** GGUF-ветка делегирует main
  iter_mmproj_weights через find_mmproj(model_path, mmproj_path)
  ([models/glm5_next/weight.py](../../python/freetoken/models/glm5_next/weight.py));
  RuntimeError "gguf vision serving is not wired in this build" убран, добавлен
  guard против двойного стриминга (повторный вход после того, как vision уже
  отдан, возвращает пусто, а не дублирует тензоры).
- **Kwarg mmproj_path в iter_gguf_weights.** Диспетчер main
  ([models/weight.py](../../python/freetoken/models/weight.py)) уже прокидывал
  mmproj_path вниз - отсутствие kwarg было латентным TypeError при первом же
  вызове с mmproj; добавлен и заповедан.
- **Named-гарды башни в iter_mmproj_weights.** Отсутствующие блоки тензоров и
  блок за пределами depth - именованные ошибки; отсутствующий
  clip.vision.block_count даёт осмысленный ValueError вместо сырого KeyError.
- **gguf->FTW packing.** Kwarg mmproj_path в convert_checkpoint; arch==clip
  fail-fast; clip.* KV-факты носителя идут через СУЩЕСТВУЮЩУЮ collision-гвардию
  write_metadata_gguf в [models/gguf/reader.py](../../python/freetoken/models/gguf/reader.py)
  (та самая, что была восстановлена в round-5 как _pack_kv-урок); устаревший
  коммент "USER DECISION 4" (convert.py, строки 343-345: "merge the mmproj's
  clip.* tower facts...") закрыт кодом под ним.
- **Бут carrier без --mmproj.** Новый семейный хук gguf_carrier_mm_config
  ([models/gguf/config.py](../../python/freetoken/models/gguf/config.py)),
  вызываемый build_gguf_shim ТОЛЬКО для encoder-семейств на metadata-only
  FTW-носителях: сборка mm-конфига из KV-метаданных носителя. Header-выводимые
  поля, которые metadata-only носитель пересчитать не может, берутся из
  задокументированного _TOWER_REFERENCE - пиннеда фактов башни
  GLM-5.3-Flash (proj_inter 10240, in_channels 3, attention_bias True),
  значений из верификации round-5 (347/347 тензоров поэлементно точны,
  max_abs_diff 0). Варианты значений фейлятся громко при загрузке, не молча.
- [scripts/verify_mmproj_tensors.py](../../scripts/verify_mmproj_tensors.py)
  адаптирован под main API; прогон на реальном
  чекпойнте - в остатке фазы.

## Карта маркеров и метод классификации

Метод шага 0: пробный прогон `pytest -p no:skipping` над маркированными
тестами - снимает skip-маркеры и показывает, что падает, без правок кода.
Итог из 48: a=0 (зелёных на main-механике как есть нет) / b=0 (семантических
дублей нет) / c=38 (переписаны под main) / d=2 (доимплементированы) / e=8
(сняты после суставов). Счётчики сверены grep-ом на актуальном дереве:
D-PORT-D2 44 -> 0, D-PORT-1 4 -> 0; D-PORT-D4 37 не тронуты (следующая фаза).

Стартовая гипотеза кейса (~24 маркера лягут в исходы a/b как
"семантически покрываются main") опровергнута прогоном. Причины:

- upstream 32056cf вообще не имеет mmproj-тестов: test_glm5_next_gguf.py -
  round-5 веточная сюита, значит пар "веточный тест -> остался main-тест"
  (исход b) взяться неоткуда;
- три теста, которые кейс считал "немаркированными main-тестами"
  (test_mmproj_boot_carries_image_token_and_processor_path:2647,
  test_mm_processor_fallback_builds_image_processor_from_mmproj:2680,
  test_mm_processor_mmproj_boot_wraps_image_into_mm_item:2723), сами были
  помечены D-PORT-D2 - гипотеза опиралась на память о дереве до merge.

Урок: карту маркеров строит только пробный прогон на актуальном дереве;
кластерные гипотезы кейса - стартовые, не финальные.

## Уроки

1. **Key-shape реального файла.** main _mm_config требовал
   clip.vision.swiglu_clamp и бюджетные ключи image_min_pixels / image_max_pixels;
   реальный артефакт mmproj-BF16.gguf несёт clip.vision.swiglu_limit,
   clip.vision.image_size=448 и НЕ несёт бюджетных ключей - boot/convert на
   реальном файле падал бы KeyError ещё до сегрегации весов. Исправлено в
   main-примитиве: swiglu_clamp фоллбэчится на swiglu_limit (именованная
   ошибка, если нет ни того, ни другого); отсутствие бюджетных ключей даёт
   референсный бюджет GLM-5.3-Flash (min 16 / max 8000 soft-токенов, семантика
   ветки); частичное присутствие ключей по-прежнему fail-fast. Фикстура
   present-keys формы (_CLIP_METADATA) сохранена, добавлена
   real_shape_mmproj_gguf, зеркалящая реальный файл, + 3 новых теста.
   Урок уровня round-5 ("verify 347/347 не заменяет boot на реальном
   артефакте"): тестовые фикстуры копируют форму ключей из головы автора, а
   реальный GGUF - из writer-а llama.cpp; сверяйтесь с артефактом.
2. **Вакуумный тест.** Немаркированный
   test_mmproj_pinned_reference_mismatch_fails_fast матчил "reference" против
   имени tmp-каталога внутри ЛЮБОГО текста ошибки - проходил вакуумно даже на
   нерелевантном падении. Перепиннен на три реальных кросс-чека main (ширина
   проекции, silu, пиксельный бюджет) с реальными текстами ошибок. Урок:
   fail-fast-тест проверяйте на падении НЕ на том пине.
3. **Фикстура glm5next -> glm5v.** _CLIP_METADATA пиннила projector_type
   "glm5next", а main _PROJECTOR - "glm5v"; одна строка фикстуры была корнем
   ~15 фейлов. Урок: при переписывании сюиты под main сначала сверяйте
   константы фикстур с main-константами, потом чините код.
4. **Ревью-проход окупился.** Один проход нашёл 1 блокер (key-shape из урока 1,
   проверен против реального артефакта) и 1 минор (raw KeyError на отсутствующем
   block_count) - оба исправлены до финальной батареи. Сомнение
   "_TOWER_REFERENCE - это двойная логика?" разрешено как допустимое: это
   документированный фоллбэк с пиннедом верифицированных значений, не вторая
   реализация main-механики.

## Что осталось (cloud-окно)

GPU-часть фазы, по VRAM-протоколу в cloud-окне, сериализованно:

- MMProj vision smoke #810: boot raw GGUF с --mmproj + e2e-запрос с картинкой
  (прецедент round-5: READY ~142 с, осмысленный ответ).
- Бут FTW-vision carrier без --mmproj (новый путь gguf_carrier_mm_config).
- verify 347/347 на реальном чекпойнте адаптированным
  [verify_mmproj_tensors.py](../../scripts/verify_mmproj_tensors.py)
  (методика кампании [gguf-glm5next-vision](../cases/gguf-glm5next-vision/TASK.md)).

Статус, карта маркеров и шаги фазы - в разделе 4
[кейса round 6 порт-долг](../cases/round6-deferred-ports/TASK.md); процедуры
волн - [merge-wave-procedure.md](merge-wave-procedure.md). Смежные порты:
D4 - [glm5next-gguf-loader-port.md](glm5next-gguf-loader-port.md),
D5 - [round5-d5-nvfp4-kv-generic.md](round5-d5-nvfp4-kv-generic.md).