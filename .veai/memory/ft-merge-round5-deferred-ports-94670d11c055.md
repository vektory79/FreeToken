---
name: "ft-merge-round5-deferred-ports"
description: "Merge round5 f0eed55: D1-D5 DONE; residuals fixed; kb links repaired (8bba8e5); only fa/fi/trtllm open"
type: project
lastUpdated: 2026-10-23T21:53
lastRecall: 2026-10-24T14:40
---

# Round 5 merge (2026-10-17): ggml re-platform, D1-D5 deferred ports campaign

Merge f0eed55 = upstream/main 11c1abd -> vektory79; режим "минимальный зелёный merge" (binding для будущих ре-платформ: спрашивать scope ДО волны). Backup старого стека: backup/vektory79-pre-merge5-b8c3794. План: kb/cases/round5-deferred-ports/TASK.md; журнал волн и спеки (d2/d3/d4/d5-spec.md, d1-kernel-map.md): .tasks/round5-deferred-ports/.

## Итог кампании (2026-10-21, все фазы закрыты)

- D4 glm5next GGUF loader: 9314c9b + kb 9f66db9. Loader пережил merge (ампутация импортов). Boot-smoke PASS (READY 145s, decode 21.00 tok/s offload UD-Q3_K_XL), кросс-валидация 38/38 non-permuted (плотные corr>=0.9998, банки 0.98-0.993 = квант-допуск). Уроки: .to(CUDA) до kernel.ggml.dequantize (op_dequantize требует kDLCUDA); дубль ключа gguf в _BANK_SCHEMAS затирал merged-схему (boot нашёл, тесты не могли); pad_rows = ceil(tail/row_bytes); GGUF->FTW merged-имена + gguf_types + layer_sink-стриминг; kb: glm5next-gguf-loader-port.md.
- D3 hybrid на slot pools: 393b682 + kb 1deebb1. Main УЖЕ имел гибрид - вернулись 3 fail-fast гейта (format screen, header rejection, auto floor gate). 786k рецепт бутится без лестницы; hybrid 27.71 tok/s vs offload 21.00 (+32% same stack); cap-0 CPU-leg ~49.6 GB/s (~1.8x старого); Fix-3 e2e 5568 cached / 0 промахов; T2 helpers НЕ возвращены (однопульный executor). kb: hybrid-moe-slot-pools-port.md.
- D1 ядра: 415b84e kb (статья round5-d1-kernel-wins-no-go.md). ВСЕ 4 старых выигрыша NO-GO на ggml (ALU/тензорная крыша, не DRAM; mhc SPLIT=128 оптимален; N-split хуже целого). Старые проценты не переносятся.
- D2 mmproj-зрение: 7bc68b0 + kb d9b72a2 (статья round5-d2-mmproj-vision-port.md). Ампутация была точечной (writer extra_kvs/_pack_kv с collision-гвардией, mmproj_path-проводка, CPU device-обход). Verify 347/347 exact (max_abs 0), boot --mmproj READY 142s, e2e-картинка PASS.
- D5 nvfp4-KV generic: store-стек пережил merge; потерян был только read-путь generic-triton. Ре-ленд load_nvfp4 (decode/extend/split/paged) НА текущую структуру (sinks/sliding-window/split-caps сохранены, None-шкалы байт-в-байт); supports_fp8_kv+nvfp4 flipped вместе; -p no:skipping 8 failed -> 36 passed; test_kv_nvfp4 81 passed GPU. Quality A/B: Qwen3.8-27B-NVFP4-FTW (dense GDN+FULL), 3.46x KV ужатие, greedy 11/12 byte-identical (1 argmax-flip задокументирован), needle ok. fa/fi/trtllm - отдельные PR (нужен внешний kernel с нашей layout-схемой).

## Открытое (после кампании)

- OPEN residual (новый пункт порт-долга в TASK.md сек.9): gguf->FTW чекпойнт НЕ бутится: register.py:458/469 строит таблицу ggml-типов из gguf_tensor_index(model_path), для FTW это metadata-only (tensor_count=0) -> пустая таблица -> KeyError token_embd (layers/quantization/configs/gguf.py:42/:58). Любой конвертнутый FTW (включая text-only). Фикс: FTW-ветвящийся источник dense-типов или docs-запрет; отдельная задача.
- 11 pre-existing битых kb-ссылок (список в TASK.md сек.8) - при первом docs-коммите.
- g.2: tests/moe/test_cpu_moe_gguf_iq.py сидит на удалённом models/gguf/dequant - ре-ленд отдельной задачей.

## Уроки-ловушки кампании

Двухфазный дифф в одном дереве - расщепление фазовых коммитов stash-проверкой бисект-среза; пип-gguf нужен только фикстурам (рантайм - вендорный ридер); не смешивать dequant-пути; старые проценты/победители-геометрии не переносятся через ре-платформу - решает замер; вакантные модули тестов сидят на удалённых API (проверять importorskip-и).

**Финал 2026-10-21:** D5 закрыт (660c814 код 8 файлов + eeceaf1 kb, статья round5-d5-nvfp4-kv-generic.md) - кейс round5-deferred-ports СТАТУС DONE в TASK.md: D4 9314c9b, D3 393b682, D1 NO-GO 415b84e, D2 7bc68b0, D5 660c814. Полная цепочка: f0eed55 -> ... -> eeceaf1 (11 коммитов кампании), push не делался. Скилл-левиафан: TRAPS T81-T88 добавлены (не закоммичены - 4 файла скилла Modified UNSTAGED, решение за пользователем). Остатки в TASK.md сек.9: gguf->FTW boot residual (register.py:458/469), fa/fi/trtllm (нужен внешний kernel), g.2 ре-ленд test_cpu_moe_gguf_iq, 11 битых kb-ссылок.

**Починка остатков (2026-10-21, вечер):** 4 коммита 48915b4 (gguf->FTW ветвление dense-типов: старый 136G FTW-vision бутнулся БЕЗ реконверсии, /ready 87s chat 200; legacy qweight replay/pad/fold; громкий re-convert вместо KeyError), 739336f (num_experts гард per-layer + гигиена тестов: 8 разморожено, 2 честно пере-заморожены, 3 удалено, g.2 ре-ленд 17 passed GPU, test_gguf_quant удалён), 039bbf9 (мёртвый код -575: benchmark/, perf.py, osproc, session_tier, NSPLIT/MTILE-ручки, untile консолидирован, offload_expert_method удалён, pyproject gguf>=0.19), 55c0062 (kb закрытие). Ревью A/B: 0 blocker/0 major, 4 minor пофикшены (rpartition[1]->[2], race _pair_on_done -> Lock, num_experts rep->все слои, ложный docstring). FTW decode "9.6 vs 11.21" - МЕТОДОЛОГИЧЕСКИЙ АРТЕФАКТ (8-токенный зонд r3 vs 256-стрим; f0eed55 на 256-стриме 9.74/9.65 - код не менялся); бисект завершён на точке 2. Открыто: fa/fi/trtllm (внешние kernel), 11 битых kb-ссылок, v3a-surface-map устарел (баннер добавлен). TRAPS T89-T91 (не закоммичены). Цепочка: ...624ac64 -> 48915b4 -> 739336f -> 039bbf9 -> 55c0062; push не делался.

**kb-ссылки починены (2026-10-21, поздний вечер):** коммит 8bba8e5 - линк-чек нашёл 9 битых (не 11: 2 псевдо-ссылки kb/README.md и review-1.md уже резолвились после self-sufficiency sweep 618b81d), 0/1119 после; lever3-chain-anatomy.md - 5 удалённых-ре-платформой целей переведены в текст с пометкой, glm53flash-spec-decode/TASK.md - 4 ссылки перенацелены (глубина пути ../../ вместо ../../../). Сек. 9 кейса: открыт только fa/fi/trtllm (внешние kernel). Цепочка: ...55c0062 -> 8bba8e5; push не делался.
