---
title: "Фазы 3-4 mmproj-зрения GGUF glm5next: image-процессор из mmproj-метаданных и упаковка визуальной башни в GGUF-FTW"
date: 2026-10-06
hardware: "CPU-only (pytest + mmap-чтение GGUF/mmproj; VRAM не затронута); контекст стенда - RTX 5090 риг, VRAM занята локальной LLM"
commits: "d864578 feat(mm): build the gguf image processor from mmproj metadata; 2b91236 feat(checkpoint): pack the gguf vision tower into ftw checkpoints"
status: "фазы 1-4 закрыты (CPU); фаза 5 (HW-волна e2e) впереди, см. [TASK.md](TASK.md)"
tags: [gguf, mmproj, vision, glm5next, image-processor, chat-template, ftw, source-metadata, checkpoint]
---

# Фазы 3-4 mmproj-зрения: image-процессор и FTW-упаковка башни

Бриф - [TASK.md](TASK.md); контекст фаз 1-2 (точка выбора vision-спеки и
числовая верификация маппинга) - [phases-1-2.md](phases-1-2.md);
сопроводительная заметка Фазы 3 об осознанном отклонении от брифа -
[phase-3-notes.md](phase-3-notes.md). Фаза 3 (коммит d864578) строит
image-процессор и image_token_id из GGUF-метаданных; фаза 4 (2b91236)
упаковывает визуальную башню в GGUF-FTW, так что FTW-бут самодостаточен
(без `--mmproj` и без доступа к исходному .gguf).

## Фаза 3: image_token_id и chat-шаблон из ОСНОВНОЙ GGUF

Ключевой факт, скорректированный против ресёрч-заметки: chat-шаблон живёт в
ОСНОВНОЙ GGUF-файле (KV `tokenizer.chat_template`; в unsloth UD-Q3_K_XL
обёртки begin/end image рендерятся макросом `emit_image`), а в mmproj
токенизаторных KV нет вообще. `load_gguf_tokenizer`
(`models/gguf/tokenizer.py`) подхватывает этот KV в
`PreTrainedTokenizerFast.chat_template`.

- `resolve_gguf_image_serving` (`models/glm5_next/gguf.py`) выводит
  image_token_id из vocab-массива: обязательны шаблон, рендерящий
  9-символьный ASCII-глиф image-placeholder, и сам глиф в
  `tokenizer.ggml.tokens`; выведенный id (154854) должен совпасть с пином
  NVFP4-референса, иначе fail-fast - переупорядоченный vocab это другой
  чекпойнт, для которого башня не верифицирована. Отсутствие шаблона или
  глифа тоже fail-fast: иначе image-промпт молча теряет картинку.
- Пины процессора: только token budget (`min_image_tokens=16`,
  `max_image_tokens=8000`) - clip.* KV его не несёт, и он совпадает с
  NVFP4 `processor_config.json`.

## Фаза 3: fallback image-процессор (GLM-специфичный override)

У raw-GGUF бута нет `preprocessor_config.json` рядом с .gguf, поэтому
`Glm5NextMMProcessor._image_processor` (mm/processors/glm5_next.py)
переопределён: при `mmproj_path` процессор строится лениво (тот же
single-instance контракт под локом, что и базовый загрузчик) из
`parse_mmproj_image_processor_config` - patch/merge размеры и CLIP-нормализация
из `clip.*` метаданных, `temporal_patch_size` пин со сверкой числа срезов
`v.patch_embd.weight`.

- Отклонение от брифа: отсутствующий обязательный `clip.*` ключ - boot-time
  ValueError, а НЕ подстановка NVFP4-пина. Пин никогда бы не сработал на
  хорошем файле и единственно маскировал бы дрейф будущего конвертера,
  уронившего ключ; неверные mean/std молча отравляют нормализацию пикселей.
  Подробности - [phase-3-notes.md](phase-3-notes.md).
- `image_size` в kwargs процессора НЕ нужен: референсный image-процессор
  ресайзит динамически и поля `image_size` не имеет; значение читается
  только для sanity-проверки кратности `image_size % patch_size`.

Ловушка гейта исключений: `get_mm_processor` (mm/processor.py) ранее глотал
ЛЮБОЕ исключение загрузки конфига в `None` (контракт "нет мультимодального
входа"). Fail-fast ValueError валидации `--mmproj` (нет файла, не clip-арх)
проглотился бы той же сеткой, и мультимодальный бут молча стал бы
text-only. Гейт: `except ValueError` ре-райзит при выставленном
`mmproj_path`, старый `None`-контракт для чужих/отсутствующих HF-конфигов
сохранён.

## Фаза 4: башня в GGUF-FTW через source_metadata.gguf

Носитель фактов башни - уже существующий metadata-only артефакт конвертера
`source_metadata.gguf`: конвертер (`checkpoint/convert.py`) при
`--mmproj` мержит все `clip.*` KVs mmproj в его KV-секцию через
`extra_kvs` в `write_metadata_gguf` (`models/gguf/reader.py`).

- `_pack_kv` пишет bool / int32 / float32 / string / плоские
  однотипные массивы. Гетерогенный массив - fail-fast: GGUF-массивы
  однотипны, упаковка `[1, 2.0]` по типу первого элемента молча обрезала бы
  хвост. Пустой массив тоже fail-fast.
- Гвард коллизий: `extra_kvs` не пересекаются с собственными KVs источника и
  с резервным `OUTPUT_WEIGHT_PRESENT_KV` (дубликат KV репарс-проверка
  молча дедупнула бы, а коллизия с резервом молча ЗАМЕНИЛА бы факт).
- `image_processor_kwargs` НЕ персистятся: на буте они пересчитываются из
  тех же carrier `clip.*` KVs - нет двойного источника истины.

Raw-путь и FTW-путь правомерно расходятся в источнике kwargs процессора:
raw-GGUF парсит mmproj-файл лениво при первом изображении; FTW запекает
kwargs в конфиг-шим на буте (файла mmproj рядом нет). Оба источника - одни и
те же `clip.*` KVs.

## Фаза 4: vision-ветка build_gguf_shim и parse-хуки

`build_gguf_shim` (`models/gguf/config.py`) выбирает источник башни: mmproj
файл при `--mmproj` ИЛИ `clip.has_vision_encoder` собственных метаданных
(GGUF-FTW бут). Raw GGUF с clip-KVs и непустой таблицей тензоров - fail-fast
("pass --mmproj"): его таблица тензоров это LLM, а не башня, и serving молча
остался бы без visual весов.

- Parse-хуки переведены на ядро `(metadata, shapes, source)`:
  `vision_config_from_tower_metadata` и
  `image_processor_config_from_tower_metadata`; файловые обёртки
  `parse_mmproj_*` остаются точкой входа raw-пути. `source` - имя
  носителя для сообщений об ошибках (mmproj-файл или merged
  source_metadata.gguf).
- Shape-кросс-чеки (число temporal-срезов, input channels, ширина mm.gate)
  гейтятся на непустой `shapes`: metadata-only carrier не имеет таблицы
  тензоров, и числовым контролем остаются фазовые верификации
  ([phases-1-2.md](phases-1-2.md)). Пустой `shapes` отключает только
  кросс-чеки, обязательные clip.* KVs проверяются всегда.

## Фаза 4: ftw_hotfix и старый текст-only FTW

`gguf_ftw_tower_reconvert_needed` (scripts/ftw_hotfix.py) распознаёт
GGUF-происхождение FTW по наличию `source_metadata.gguf`: у GGUF-FTW башня
либо уже лежит как visual.*, либо её нет и HF-rename конвейер не произведёт
её - единственное средство реверсия `ft checkpoint --mmproj`, а план
скачивания HF-файлов, печатаемый дальше, вводил бы в заблуждение. Гвард
ставится ВЫШЕ dry-run ветки чтения башни. Старый путь исправления текст-only
HF-FTW (`ftw_lacks_vision`) не тронут.

Попутно (тот же коммит): `convert_checkpoint` допустил `device="cpu"` для
тестовых фикстур; production-конвертация всегда на GPU.

## Уроки

- Fail-fast бьёт пин по умолчанию, когда пин недостижим на валидном файле:
  единственный наблюдаемый эффект пина - маскировка дрейфа конвертера
  ([phase-3-notes.md](phase-3-notes.md)).
- Metadata-only carrier (source_metadata.gguf) оказался естественным местом
  для фактов башни: формат уже есть, чтение дешёвое, конвертеру нужен один
  merge KVs, а не второй файл. Гварды коллизий/типов нужны потому, что
  ручная упаковка KVs идёт мимо валидатора gguf-библиотеки.
- Ловушки фаз записаны в TRAPS.md скилла gguf-native-serving (вне kb):
  T72-T75.