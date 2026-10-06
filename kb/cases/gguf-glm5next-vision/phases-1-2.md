---
title: "Фазы 1-2 mmproj-зрения GGUF glm5next: точка выбора vision-спеки и числовая верификация маппинга 347 параметров"
date: 2026-10-06
hardware: "CPU-only (mmap-чтение mmproj и NVFP4-FTW, VRAM не затронута); контекст стенда - RTX 5090 риг, VRAM занята локальной LLM"
commits: "e35577d feat(models): add --mmproj flag and gguf vision registry spec; fae88c8 feat(models): add gguf mmproj vision tensor reader with numeric verification; 747f658 fix(models): harden gguf vision reader guards and verification report"
status: "фазы 1-2 закрыты (STOP GATE обеих фаз); фазы 3-5 (image-процессор, FTW-упаковка, HW-волна) впереди, см. [TASK.md](TASK.md)"
tags: [gguf, mmproj, vision, glm5next, config-shim, tensor-verification, clip]
---

# Фазы 1-2 mmproj-зрения: точка ветвления спеки и числовая верификация маппинга

Бриф - [TASK.md](TASK.md). Фаза 1 (коммит e35577d) ввела флаг `--mmproj`
(`mmproj_path` в EngineConfig) и мультимодальную GGUF-спеку; фаза 2
(fae88c8 + 747f658) добавила ридер vision-тензоров mmproj и числовую
верификацию против NVFP4-референса. Отчёт верификации -
[tensor-verification-report.md](tensor-verification-report.md); дамп имён
тензоров mmproj -
[tensors-mmproj.txt](../../findings/gguf-glm5next-path-a/tensors-mmproj.txt).

## Фаза 1: выбор vision-спеки возможен только в build_gguf_shim

Риск-3 брифа решился вариантом A: поле `mmproj_path` в EngineConfig ->
`cached_load_hf_config` -> `build_gguf_shim` выбирает мультимодальную спеку
(`Glm5NextGGUFForConditionalGeneration`) в vision-ветке `GGUF_ARCH_TO_REGISTRY_VISION`.

- Вариант B ("выбрать спеку в parse_gguf_config") НЕ работает: реестр
  диспетчеризуется раньше - `model_spec` читает `architectures[0]` шима ДО
  вызова `parse_config`, а `parse_config` возвращает `ModelConfig` и спеку
  менять не может.
- `GgufConfigShim` - frozen dataclass (FrozenInstanceError при попытке
  присвоить поле). Поэтому `VisionConfig` собирается хуком в
  `build_gguf_shim`, а `parse_gguf_config` только потребляет
  `shim.vision_config`; сброс секции конфига делается через
  `dataclasses.replace`.
- `ServerArgs` наследует `SchedulerConfig(EngineConfig)`, поэтому новое поле
  EngineConfig само становится аргументом `ft serve` без ручного прокидывания
  (добавлен только `parser.add_argument` с валидацией: файл существует и
  `gguf_architecture == "clip"` - fail-fast).

## Фаза 2: числовая верификация маппинга (методика)

Объект: mmproj-BF16.gguf (arch clip, 348 тензоров: 224 F32 / 124 BF16) против
визуальной башни NVFP4-FTW референса (`visual.*`, bf16). Результат: все 347
параметров exact, max_abs_diff == 0 (fp32-сравнение обеих bf16-сторон,
CPU-mmap). Обе стороны происходят из одного HF-источника и bf16-cast
идентичен, поэтому эталон здесь вырожденно строгий: ЛЮБОЙ diff > 0 - это
подозрение на маппинг (axis / transpose / segment order), а не cast-шум.

Методика `scripts/verify_mmproj_tensors.py` (коммит fae88c8, гварды 747f658):

- CPU-mmap обеих сторон, никакого VRAM.
- Обратный индекс mmproj-имя -> visual-параметр: два temporal-среза
  `v.patch_embd.weight` / `.weight.1` стекуются в один параметр
  `visual.patch_embed.proj.weight`, dtype объединяется через "/".
- Таблица параметровая: UNMATCHED считается по union имён обеих сторон, а не
  по строкам одной стороны.
- Ловушка ключёвки: колонки, ключёванные именами одной стороны
  (например source_dtype по mmproj-именам при строках по visual.*), молча
  дают дыры - обратный индекс строится явно, а не выводится из порядка строк.

## Формы и порядок осей (что подтвердилось diff == 0)

- GGUF-ридер отдаёт torch-порядок осей (reversed ne): `_cast().reshape(t.shape)`
  даёт матричный (out, in) автоматически, ручной transpose не нужен.
- qkv row-порядок q|k|v по выходной оси подтвердился diff == 0 на
  `visual.blocks.N.attn.qkv.*`.
- Conv3d patch-embed: два temporal-среза собираются `torch.stack(dim=2)` ->
  (1024, 3, 2, 14, 14). Не cat: cat по последней оси дал бы (1024, 3, 14, 28, 14)
  и разнёс patch-сетку.
- `mm.post_norm.{weight,bias}` (F32, 4096, с bias) - это
  `visual.merger.post_projection_norm.{weight,bias}` (torch LayerNorm),
  НЕ `visual.post_layernorm` (RMSNorm 1024 без bias). `v.post_ln.weight`
  (F32, 1024, без bias-тензора) - это `visual.post_layernorm.weight`.

## Гварды ридера (747f658)

Гварды живут внутри ридера, а не выше по стеку: проверка arch == clip;
дубликаты и дыры среди patch-срезов; блок с индексом за depth; неизвестное
имя тензора -> ValueError с именем тензора в сообщении.

## Урок про бриф

Кандидатская таблица маппинга брифа содержала 2 неверные строки
(`mm.post_norm` -> `visual.post_layernorm` вместо
`visual.merger.post_projection_norm`; `v.post_ln` -> неверный таргет) и
пропуски (`v.patch_embd.*` с двумя temporal-срезами, `mm.patch_merger` - F32,
в брифе dtype не указан). Вывод: якорные таблицы маппинга верифицировать
численно, брифу не доверять; dim/bias-подсказки из кода референса
(shape vs поле конфига, наличие/отсутствие bias) находят такие ошибки до
замера.

## Ловушки скилла

Инсайты фаз записаны ловушками T68-T71 в TRAPS.md скилла gguf-native-serving
(вне kb): T68 точка выбора vision-спеки + frozen shim, T69 верификация
бриф-маппинга, T70 формы/ne-порядок/temporal-stack + гварды, T71 ключёвка
верификационной таблицы обратным индексом.