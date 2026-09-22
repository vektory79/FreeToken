---
title: "Карта upstream llama.cpp GLM5NEXT: что читать для GGUF-портов"
date: 2026-09-22
hardware: не применимо (карта исходников)
branch-commits: "снапшот upstream снят 2026-09-13; git-ревизия upstream НЕ записана - перед новым портом обязательна сверка имён с текущим upstream"
status: validated
tags: [llamacpp, upstream, glm5next, gguf, snapshot, kpool, indexer, converter, tensor-names, kv-names]
---

# Upstream llama.cpp GLM5NEXT: карта для GGUF-портов

**Вердикт (читайте это первым).** Для порта архитектуры GLM5NEXT (GGUF-путь
FreeToken) источник правды по upstream - нумерованный снапшот исходников
llama.cpp в kb/findings/llamacpp-glm5next-snapshot/. **ВНИМАНИЕ: git-ревизия
upstream, с которого снят снапшот, НЕ записана** - перед любым новым портом
имена тензоров и KV-ключей нужно сверить с текущим upstream (могли уйти с
2026-09-13). Смежные карты FreeToken-стороны - в
[../findings/gguf-glm5next-path-a/](../findings/gguf-glm5next-path-a/anchor-verification.md).

## Состав снапшота

Снапшот снят 2026-09-13 во время reverse-engineering прохода gguf-glm5next-path-a;
описание и список файлов - [README.md](../findings/llamacpp-glm5next-snapshot/README.md):

- [glm5next_model_full_numbered.txt](../findings/llamacpp-glm5next-snapshot/glm5next_model_full_numbered.txt)
  (814 строк) - полная реализация модели glm5next: выбор селектора
  glm5next_n_select, загрузка hparams, построение графа, конвертерные проверки.
- [llama-arch.cpp](../findings/llamacpp-glm5next-snapshot/llama-arch.cpp)
  (1158 строк) + извлечённые срезы:
  [kv_names_L161_L424.txt](../findings/llamacpp-glm5next-snapshot/kv_names_L161_L424.txt)
  (LLM_KV_NAMES),
  [tensor_names_L426_L699.txt](../findings/llamacpp-glm5next-snapshot/tensor_names_L426_L699.txt)
  (LLM_TENSOR_NAMES),
  [glm5next_cases_L1075_L1160.txt](../findings/llamacpp-glm5next-snapshot/glm5next_cases_L1075_L1160.txt)
  (список arch-кейсов).
- [kpool.h.txt](../findings/llamacpp-glm5next-snapshot/kpool.h.txt) и
  [kpool_grep.txt](../findings/llamacpp-glm5next-snapshot/kpool_grep.txt) -
  механика пула индексатора llama-kv-cache-kpool.
- [lm_hparams_generic.txt](../findings/llamacpp-glm5next-snapshot/lm_hparams_generic.txt),
  [lm_create_tensor_qkv.txt](../findings/llamacpp-glm5next-snapshot/lm_create_tensor_qkv.txt),
  [lmh_create_qkv.txt](../findings/llamacpp-glm5next-snapshot/lmh_create_qkv.txt) -
  generic-чтение hparams и создание QKV-тензоров в llama-model / llama-memory-hybrid.

## Ключевые факты upstream-реализации

- **Селектор n_select считается из индексаторных hparams**: используются
  `indexer_top_k` и `indexer_kpool`; утверждения в коде: `GGML_ASSERT top_k >=
  kpool` и `top_k % kpool == 0`; n_select = top_k + kpool - 1 =
  (top_k/kpool + 1)*kpool - 1 (начало
  [glm5next_model_full_numbered.txt](../findings/llamacpp-glm5next-snapshot/glm5next_model_full_numbered.txt)).
- **Гибридная память = recurrent + kv-cache-kpool**: слои KDA несут
  рекуррентное состояние, а слой внимания - индексаторный kpool-кэш.
  Тензоры пула индексатора: `pool_cells` I32, `pool_bias` F32, `pool_reps`;
  число пулов n_pools = ne[0]/r
  ([kpool.h.txt](../findings/llamacpp-glm5next-snapshot/kpool.h.txt)).
- **Знак ssm_a: -exp(A_log)** - соглашение lineage kimi-k3; вариант +exp -
  у bailingmoe3. Конвертер проверяет знак; при порте не перепутать.
- **Карты имён плоские, не per-arch**: и LLM_KV_NAMES, и LLM_TENSOR_NAMES -
  одна плоская карта в llama-arch.cpp (KV: строки 161-424, тензоры: 426-699);
  `%s` в ключах заполняется именем архитектуры при форматировании, поэтому
  все ключи получают префикс `glm5next.`; отдельного case-блока для glm5next
  в картах имён нет.
- Слои делятся в загрузчике по `attention.head_count_kv == 0` (KDA-слой),
  `il < leading_dense_block_count` (dense FFN), `il >= n_layer` (блок NextN).

## Смежные карты FreeToken-стороны (gguf-glm5next-path-a)

Исследовательский проход Path A оставил четыре карты, связывающие upstream
с полями FreeToken - использовать их вместе со снапшотом:

- [phase1-config-keys.md](../findings/gguf-glm5next-path-a/phase1-config-keys.md) -
  полная таблица GGUF-мета-ключей glm5next -> hparams llama.cpp -> поля
  Glm5NextArgs FreeToken (арх-специфичные и generic), включая unmapped-ключи
  и что FreeToken хардкодит.
- [llamacpp-tensor-kinds.md](../findings/gguf-glm5next-path-a/llamacpp-tensor-kinds.md) -
  56 видов LLM_TENSOR, используемых glm5next, с gguf-именами и размерностями;
  содержит срез про mHC-тензоры и KDA-слои.
- [phase2-tensor-map.md](../findings/gguf-glm5next-path-a/phase2-tensor-map.md) -
  карта тензоров чекпоинта (что реально лежит в GGUF-файле).
- [anchor-verification.md](../findings/gguf-glm5next-path-a/anchor-verification.md) -
  проверка якорей исходников; [metadata.txt](../findings/gguf-glm5next-path-a/metadata.txt) -
  параметры исходного чекпоинта.

Как FreeToken-сторона эти карты потребляются - см. бриф кампании
[PLAN.md](../cases/gguf-glm5next-path-a/PLAN.md); отображение гибридной
механики на движок FreeToken -
[hybrid-machinery-mapping.md](../findings/gguf-glm5next-hybrid/hybrid-machinery-mapping.md).

## Как использовать / проверить

- Перед новым портом: сверить имена KV/тензоров с текущим upstream llama.cpp
  (файлы снапшота нумерованные - удобно для diff-сверки). Снапшот НЕ заменяет
  свежую ревизию, а даёт зафиксированную точку отсчёта.
- Для будущих снапшотов: ВСЕГДА записывать URL репозитория и хэш коммита в
  момент снятия (правило вынесено из этого снапшота -
  [README.md](../findings/llamacpp-glm5next-snapshot/README.md)).
- Терминология GGUF/конвертации - [glossary.md](../glossary.md); чем GGUF-путь
  отличается от FTW-загрузки - [ftw-load-path.md](ftw-load-path.md).