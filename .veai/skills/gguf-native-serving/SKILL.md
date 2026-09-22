---
name: "gguf-native-serving"
schemaVersion: "v0.1"
description: "Добавляет поддержку нативного gguf-сервинга для поддерживаемых моделей FreeToken: discovery, шим конфига, переводчики тензоров, токенизатор, диспетчер ядер, банки экспертов, offload-базлайн и процессорный вычислительный тир (гибрид). Используй, когда пользователь просит поддержку нового gguf формата, нативный gguf-сервинг, или модель сообщает 'architecture X is not supported'."
agent: Orchestrator
used-by:
    - "Orchestrator"
---

# Skill: Native GGUF Serving for Arbitrary Model Families (FreeToken)

Реализуй нативный GGUF-сёрвинг для произвольной архитектуры модели в FreeToken.
Скилл обобщает завершённые кампании GLM-5.3-Flash-UD-Q3_K_XL: offload
(коммиты 4162f7c..78115df) и hybrid + CPU-kernel port (11f1a80..63b9bff) - и
работает для ЛЮБОЙ gguf-модели, семейство которой уже поддержано через HF/FTW
safetensors в python/freetoken/models/<family>/.

## Orchestration

Run via [ORCHESTRATION.md](ORCHESTRATION.md) (self-sufficient skill) - each
phase is a wave: Test -> Review -> Triage -> STOP GATE -> commit. Режимы
исполнения, handoff, process-hygiene и commit-дисциплина описаны там; этот
файл - доменная часть (discovery + 7 фаз), брифы задач:

В конце сессии скилл самообновляется: новые знания сессии (причины с
коммит-идами, замеренные классы ожиданий, новые ловушки, правки протокола)
интегрируются в TRAPS.md / брифы / ORCHESTRATION.md (см. раздел
"Session-end self-update" в [ORCHESTRATION.md](ORCHESTRATION.md)); проход
никогда не коммитит сам - коммит остаётся решением пользователя.

- [tasks/task-00-discovery.md](tasks/task-00-discovery.md)
- [tasks/task-01-config-shim.md](tasks/task-01-config-shim.md)
- [tasks/task-02-tensor-translator.md](tasks/task-02-tensor-translator.md)
- [tasks/task-03-tokenizer.md](tasks/task-03-tokenizer.md)
- [tasks/task-04-kernel-dispatch.md](tasks/task-04-kernel-dispatch.md)
- [tasks/task-05-expert-banks.md](tasks/task-05-expert-banks.md)
- [tasks/task-06-e2e-ab.md](tasks/task-06-e2e-ab.md)
- [tasks/task-07-cpu-compute-tier.md](tasks/task-07-cpu-compute-tier.md)

## Когда применять

- Пользователь даёт .gguf-файл модели, семейство которой уже работает через HF/FTW.
- Boot умирает на models/gguf/config.py build_gguf_shim: "architecture X is not supported".
- Цель: тот же файл сервится нативно (без bf16-материализации экспертов).

## Обязательные входы (проверь до старта)

1. Путь к .gguf (архив; mmproj может лежать уровнем выше).
2. HF/FTW reference-чекпоинт ТОЙ ЖЕ модели (для кросс-валидации весов).
3. Рабочая reference-реализация, где формат gguf файла УЖЕ поддержан (llama.cpp
   или другая). СПРОСИ пользователя путь ЯВНО на discovery
   ([tasks/task-00-discovery.md](tasks/task-00-discovery.md)). Если
   формата нет в upstream - проси любую существующую reference: проекшны
   обязаны быть заякорены на рабочую реализацию, никогда на бумажной математике.
   Что извлекать: vec_dot-ядра (ggml-cpu/arch/<isa>/quants.c), quantize_row_*
   (ggml-quants.c), layout'ы блоков (ggml-common.h), ISA-флаги референс-сборки
   (build/CMakeCache.txt - сопоставимость микробенчей), паттерн шедулинга
   CPU-resident весов (ggml_backend_sched splits, --n-cpu-moe).
4. RTX 5090 (или аналогичная NVIDIA), clang++ на PATH, uv venv с --extra dev.

## Архитектура конвейера (discovery + 7 фаз, строго последовательно)

```
Phase 0: upstream alignment (или waive для private-use) - шаг
         внутри task-00 discovery
Phase 1: config shim (metadata -> Args)
Phase 2: tensor translator (gguf -> HF/FTW имена + fusion)
Phase 3: tokenizer (embedded vocab + pre-tokenizer + specials)
Phase 4: kernel dispatch (python wiring + convert path)
Phase 5: expert banks -> offload (pinned banks, partitions)
Phase 6: E2E bring-up + A/B (offload baseline)
Phase 7: CPU compute tier (hybrid/offload decode) - опциональна:
         микробенч-gate -> порт ядер -> weighted pools ->
         end-to-end A/B вердикт hybrid vs offload
```

Каждая фаза = отдельная code-wave + quality loop (Test -> Review x2 -> Triage ->
Fix TP -> STOP GATE -> commit); acceptance criteria каждой фазы включают gate
"I've created a git commit for this task" (см. брифы [task-00](tasks/task-00-discovery.md),
[task-01](tasks/task-01-config-shim.md), [task-02](tasks/task-02-tensor-translator.md),
[task-03](tasks/task-03-tokenizer.md), [task-04](tasks/task-04-kernel-dispatch.md),
[task-05](tasks/task-05-expert-banks.md), [task-06](tasks/task-06-e2e-ab.md),
[task-07](tasks/task-07-cpu-compute-tier.md) и [ORCHESTRATION.md](ORCHESTRATION.md)).
Не перескакивай фазы: каждая строит инварианты для следующей.

## Фаза 7: CPU compute tier (hybrid/offload decode)

Нужна, только когда файл должен сервиться hybrid или CPU-only decode. Пререквизит -
offload-базлайн фазы 6 (это A/B-якорь). Подробный бриф:
[tasks/task-07-cpu-compute-tier.md](tasks/task-07-cpu-compute-tier.md).

1. CPU-GEMV матрица форматов: для КАЖДОГО формата из discovery-дампа - какой ярус
   существует в cpu_moe_ext.cpp (scalar fp32-LUT / integer W4A8 / нет) и какая пара
   "веса-активации" нужна. K- и IQ-кванты берут q8_K, q4_0 - q8_0, nvfp4 - pg16/VNNI;
   пару смотри в reference-реализации, не угадывай.
2. Microbench gate ДО порта: standalone-харнесс компилирует СОБСТВЕННЫЕ ядра reference
   с теми же ISA-флагами, что референс-сборка (CMakeCache.txt + flags.make), и меряет
   sustained GB/s на синтетических byte-exact блоках; парити reference-vs-scalar на
   случайных строках. Gate: CPU-leg >= эффективной PCIe gather полосы бокса.
3. Порт: verbatim-транскрипция ядер (per-function target attributes, provenance,
   машинная LUT-equality, static_assert'ы layout'ов); ISA-ярус с env-override,
   КАПЛЕННЫМ поддержкой CPU (некапнутый = SIGILL); scalar остаётся fallback +
   correctness-reference; quantize-once-per-projection на top_k экспертов; scratch
   пре-сайз до capture; ноль pageable аллокаций в hot path.
4. Weighted pools: треды по партициям - по числу слоёв (even-split голодает
   доминантную); floor-at-one + disjointness; wrap-уголок явного пути под guard'ом.
5. Вердикт hybrid vs offload - только end-to-end A/B (тот же движок, тот же fill,
   единственная переменная --moe-strategy). CPU/PCIe ratio < 2.0 держит строку
   вердикта benchbw "offload" - ОЖИДАЕМО; выигрыш меряется end-to-end, не строкой.

Стоимостная модель (и анти-паттерн): per-layer hybrid cost = max(CPU leg, f x PCIe
fetch) + fixed sync (~0.6-1.0 ms/layer). Fetch-объём ЭНДОГЕНЕН скорости CPU-leg:
быстрый leg -> меньше f -> меньше байт через PCIe. Фиксированный "handshake floor",
измеренный при высоком f, - мисатрибуция fetch-объёма как sync и ложный NO-GO
(оценка Task 06 кампании glm5next ошиблась ровно так; NVFP4 A/B того же движка её
опровергла). Правило: изолируй sync при f->0 (cap-0 прогон); проекшны заякорены на
A/B рабочей reference, никогда на бумажной математике.

Бюджетные грабли: single-dtype bench-прогон КОРОБИТ per-GPU профиль (другие форматы
молча деградируют в cap-1) - FULL реран всех форматов; у профиля нет kernel-tier
fingerprint - репрофиль после любого изменения ядер; верифицируй boot-лог (fetch
fraction не cap-1, pool split, isa tag); capability gates - по capability set
исполнителя, без хардкода type id.

Worked example: glm5next IQ3_XXS/IQ4_XS/Q6_K, H=4096 I=2048 E=288 top_k=8 ->
microbench 53-66 GB/s -> hybrid 16.67 tok/s vs offload 12.97 @64k.

## Правило качества, проверенное crashing'ом

**Каждый статический вес и каждая сшивка валидируются против reference-чекпоинта
ДО того, как модель запускается на живом prompt.** Пропуск этой проверки = IMA или
prior-driven текст после 2-минутной загрузки - худший debug-профиль. Порядок:
1. Static weight correlation (мmap reference, per-tensor)
2. Fusion validation (каждый cat/transpose/derive против reference-derived expected)
3. Live per-layer bisection (только если statics прошли, а forward всё равно ломается)

## Инструменты отладки (приоритет)

1. compute-sanitizer --tool memcheck (точный первый bad-launch)
2. CUDA_LAUNCH_BLOCKING=1 (синхронизирует асинхронный IMA)
3. compute-sanitizer --tool racecheck (для stream-семантики)
4. compute-sanitizer --tool initcheck (для неинициализированных буферов)

## Матрица ловушек (проверь КАЖДУЮ перед закрытием фазы)

См. [TRAPS.md](TRAPS.md) - 63 ловушки, каждая из которых стоила реального
отладочного времени.

## Приёмка кампании

Кампания завершена, когда discovery и все фазы прошли свои STOP GATE вместе с
commit gate каждого брифа (контракт и единственное исключение - фаза 7 при
offload-only - в [ORCHESTRATION.md](ORCHESTRATION.md)). На STOP GATE каждой фазы
вызывай kb-distill (навык [kb-distill](../kb-distill/SKILL.md)): инсайты фазы
дистиллируются в kb/, дайджест записанного и пропущенного приложен к закрытию фазы.
