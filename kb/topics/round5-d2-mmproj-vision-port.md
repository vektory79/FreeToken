---
title: "Порт D2: возврат mmproj-зрения на вендоренный ggml - готово"
date: 2026-10-21
hardware: "RTX 5090 32GB (boot raw GGUF + --mmproj, e2e-картинка); CPU - контрольная батарея"
branch-commits: "vektory79: 7bc68b0 (код, 4 файла) поверх 415b84e; волна 20 кампании round5-deferred-ports"
status: done
tags: [gguf, mmproj, vision, ftw, carrier, round5, deferred-ports]
---

# Порт D2: возврат mmproj-зрения - готово

**Вердикт (читайте это первым).** Фаза 4 (D2) кампании round5-deferred-ports готова
2026-10-21, коммит 7bc68b0. Приёмка (RTX 5090): verify 348 тензоров, 347/347
визуальных параметров поэлементно точны (max_abs_diff 0); boot raw GGUF +
--mmproj READY 142 с (мультимодальность ON, text chat 200, decode 17.63 tok/s);
e2e-запрос с картинкой возвращает осмысленный ответ; 6 carrier-тестов разморожены
и зелёные; контрольная батарея (uv run --extra dev pytest): 2 failed (известная
базлайн-пара) / 2066 passed / 825 skipped. Карта раунда -
[round5-ggml-replatform-deferred-ports.md](round5-ggml-replatform-deferred-ports.md);
план фазы - [../cases/round5-deferred-ports/TASK.md](../cases/round5-deferred-ports/TASK.md).

## Ампутация ре-платформы оказалась точечной

Опасение было, что mmproj-конвейер потерян целиком (старый код ссылался на
удалённые файлы). Фактически re-platform round-5 ампутировала только три стыка,
всё остальное пережило merge:

| Компонент | Судьба на re-platform |
|---|---|
| флаг --mmproj, vision-реестры/шим, ридер, image processor | пережили; реестры карт тензоров - в [models/gguf/config.py](../../python/freetoken/models/gguf/config.py) |
| merged-carrier writer: write_metadata_gguf(extra_kvs) + _pack_kv | потерян, восстановлен в 7bc68b0 |
| convert-проводка mmproj_path (EngineConfig + load_weight) | потеряна, восстановлена |
| CPU-обход convert_checkpoint (dev.type == "cpu", для тест-фикстур) | потерян, восстановлен |

Урок: при re-platform нельзя списывать подсистему целиком по факту ссылок на
удалённые файлы - ампутация может быть точечной; сначала инвентаризация стыков,
потом вердикт.

## Уроки и методика

1. **Collision-гвардия в _pack_kv обязательна.** Дубликат KV-ключа в GGUF
   репарсится молча (читается первый экземпляр), а столкновение extra-KV с
   зарезервированным фактом output-weight молча ЗАМЕНИЛО бы его. Поэтому
   write_metadata_gguf делает fail-fast на столкновении extra_kvs с собственной
   KV-секцией источника; негативный тест -
   test_metadata_gguf_extra_kv_collision_fails_fast (test_glm5_next_gguf.py).
2. **Verify-методика переиспользуется как есть.** Сверка
   [scripts/verify_mmproj_tensors.py](../../scripts/verify_mmproj_tensors.py)
   (методика кампании [gguf-glm5next-vision](../cases/gguf-glm5next-vision/TASK.md)):
   347/347 визуальных параметров, max_abs_diff 0; два temporal-среза (исходный
   mmproj и конверт) стекаются в 348 тензоров.
3. **Fails-before честные:** на HEAD до 7bc68b0 - 5x TypeError extra_kvs +
   1x ImportError _pack_kv; после - зелёные (test_glm5_next_gguf.py).
4. Пин изображения (154854 / _MMPROJ_PINNED) пережил волны без изменений.

## ЭСКАЛАЦИЯ: FTW-carrier не несёт тензоры - gguf->FTW чекпойнт не бутится

Найдено волной 20 (не специфика D2 и не D4-банки), pre-existing: ЛЮБОЙ
сконвертированный gguf->FTW чекпойнт не поднимается на новом стеке, включая
text-only. Цепочка: checkpoint_quant_config
([models/register.py:458](../../python/freetoken/models/register.py)) строит
таблицу ggml-типов вызовом gguf_tensor_index(hf_config.model_path) (строка 469),
но для FTW model_path указывает на carrier source_metadata.gguf - metadata-only,
tensor_count=0 -> пустая таблица -> KeyError "GGUF has no tensor 'token_embd'"
из _part_types
([layers/quantization/configs/gguf.py:42](../../python/freetoken/layers/quantization/configs/gguf.py)),
вызов из scheme_for (строка 58).

- D4-boot-smoke это не поймал: гнался только raw GGUF.
- Направление фикса: FTW-ветвящийся источник dense-типов (мета FTW-индекса)
  либо docs-запрет gguf->FTW до фикса. Reconversion не поможет - carrier не
  содержит тензоров по построению.
- Статус: открытый пункт порт-долга; фикс отдельной задачей.

## Артефакты

Логи boot-smoke и e2e остались в git-ignored рабочей папке кампании
round5-deferred-ports и не версионированы; числа переживают в этой статье и в
[../cases/round5-deferred-ports/TASK.md](../cases/round5-deferred-ports/TASK.md).
Смежное: порт D4 (загрузчик glm5next GGUF) -
[glm5next-gguf-loader-port.md](glm5next-gguf-loader-port.md).
