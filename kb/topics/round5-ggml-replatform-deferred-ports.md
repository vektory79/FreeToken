---
title: "Раунд 5 слияния: re-platform GGUF-стека main'а и порт-долг D1-D5"
date: 2026-10-17
hardware: "RTX 5090 32GB (boot-smoke); CPU-волна (pytest)"
branch-commits: "vektory79; merge f0eed55 (родители b8c3794 + 11c1adb = upstream/main, один аменд после ревью); бэкап - ветка backup/vektory79-pre-merge5-b8c3794"
status: validated
tags: [merge, round5, ggml, replatform, slot-pools, deferred-ports, nvfp4-kv, glm5next, mmproj]
---

# Раунд 5: re-platform GGUF-стека main'а и порт-долг D1-D5

**Вердикт (читайте это первым).** upstream/main, влитый раундом 5, пере-платформировал
весь GGUF-стек: собственное CUDA-дерево FreeToken и python-клей удалены, вместо них -
вендоренные ggml-ядра (ggml-cuda / ggml-cpu) и квант-пакет `layers/quantization/`.
Веточные победы, сделанные на старом стеке, на новом НЕ работают. Рамка раунда -
**minimal green merge**: merge-коммит делает зелёными только сборку и тесты на новых
механиках main'а; всё глубокое отложено в порт-долг D1-D5; кейс ЗАВЕРШЁН 2026-10-21 -
все пять портов закрыты (D4, D3, D2, D5 готовы, D1 закрыт NO-GO); открытый остаток -
gguf->FTW boot (раздел «Открыто» ниже). Эта статья - durable-карта того, ЧТО пере-реализовывать на новых механиках;
процедурные уроки волны - в [merge-wave-procedure.md](merge-wave-procedure.md).

## Что поменял main (дельта upstream, влитая раундом 5)

- **Свой GGUF CUDA-стек удалён** (e5d3bd3, PR #690): csrc-дерево `kernel/csrc/gguf`
  (moe.cuh, mmq.cuh, mmvq.cuh, moe_vec.cuh, vecdotq.cuh, dequantize.cuh, dispatch.h,
  ggml-common.h, gguf_kernel.cu) и python-клей - kernel/gguf.py, layers/gguf.py,
  models/gguf/dequant.py, moe/fused_q4_0.py.
- **Вместо них вендоренные ggml-ядра**: ggml-cuda и ggml-cpu; python-слой -
  [kernel/ggml.py](../../python/freetoken/kernel/ggml.py) с каталогом kernel/ggml/,
  квант-пакет [layers/quantization/](../../python/freetoken/layers/quantization/)
  (configs / linear / moe / embedding, плюс scheme.py, registry.py, ggml_types.py),
  GGUF-модели - [models/gguf/](../../python/freetoken/models/gguf/) (experts.py,
  tensors.py, config.py).
- **Offload-кэш экспертов - per-format слот-пулы** (b416f85 PR #689, e8c6bf1
  PR #700): новые knobs в [engine/config.py](../../python/freetoken/engine/config.py)
  - moe_cache_mem / moe_cache_scope / moe_cache_split; layer_slot_bytes в
  планировании cache-budget.
- **CPU-MoE executor перепилен** (2010271 PR #670, 95ac86f PR #662, c8559fd
  PR #704): разделение на weight-format kernels и gpu-sync backends;
  cpu_moe_ext.cpp переписан практически целиком.
- **Бамп зависимостей**: torch 2.14.1, triton 3.8, sglang-kernel 0.4.9,
  flashlib 0.4.0, transformers 5.18. **Новые семейства моделей**: DeepSeek V4/V4.1,
  Qwen2/3/3.5/3-MoE/4_exp GGUF, Gemma 4, MiniMax-M2/M3, Muse-Glimmer.

## Рамка раунда: minimal green merge

До волны пользователь зафиксировал развилку для re-platform раунда: минимальный
зелёный merge (один merge-коммит только чинит сборку и тесты на механиках main'а)
либо полная адаптация ветки внутри merge-коммита. Раунд 5 прошёл по первому пути;
глубокая адаптация вынесена в кампании-последователи по порт-долгу ниже.
Бухгалтерию скип-маркеров и остальные процедурные уроки см. в
[merge-wave-procedure.md](merge-wave-procedure.md), раздел «Раунд 5».

## Порт-долг D1-D5 (кейс завершён 2026-10-21: все пять портов закрыты)

### D1. Веточные победы ядра на вендоренном ggml

Статус: закрыт NO-GO 2026-10-21: все 4 выигрыша перекрыты ggml нативно или опровергнуты
замерами волны 18 (пробы d1-stage*); порты не нужны, дефолты не тронуты. Числа и уроки -
[round5-d1-kernel-wins-no-go.md](round5-d1-kernel-wins-no-go.md).

| Победа ветки (старый стек) | Якорное число (конфиг измерения) | Что пере-реализовать на новом стеке |
|---|---|---|
| v2 grouped MMQ (GGUF-префилл) | +16.4% @ батч 8128 (battery PASS) | grouped-MMQ механика поверх вендоренных ggml-ядер |
| v3 m-tile grouped MoE | +67.2% (свип m-tile, префилл); prod-дефолт MOE_MTILE=32 | m-tile-нарезка MoE-пути; дефолт тайла пере-мерить |
| dense q8_0 GEMM: N-split + tile64 | N-split +2.61%; tile64 +40.46% (A/B dense q8_0) | N-split и тайл 64 в dense q8_0 GEMM |
| mhc-цепочка NS=64 | +3.4% (paired A/B декода) | NS-нарезка mhc-цепочки (NS-knob живёт в kernel/triton/mhc.py, не в engine.py) |

Вердикты замеров (2026-10-21, RTX 5090): MoE-проба - все форматы в ALU-крыше
169-227 TF/s int8-экв. при эффективных 241-572 GB/s (DRAM ~1.7 TB/s не узкое место);
dense N-split - медленнее целого везде (-1.8%..-16.5%), целое ~306 TF/s = 73% int8-пика;
mhc - текущий SPLIT=128 оптимален (64 теряет 29-40% на префилле), T=1 - латентный пол.

Все числа измерены на УДАЛЁННОМ стеке csrc/gguf - на вендоренном ggml они
обнуляются до порта. Методика и полные конфиги: [prefill-throughput.md](prefill-throughput.md);
машиночитаемые якоря - [../baselines/README.md](../baselines/README.md).

### D2. mmproj-видение на поверхностях layers/quantization/* - готово 2026-10-21, коммит 7bc68b0

Статус: готово 2026-10-21, коммит 7bc68b0: verify 348 тензоров 347/347 параметров
exact (max_abs_diff 0), e2e-картинка PASS (осмысленный ответ), boot raw GGUF +
--mmproj READY 142 с (мультимодальность ON, decode 17.63 tok/s); 6 carrier-тестов
разморожены и зелёные (fails-before: 5x TypeError extra_kvs + 1x ImportError
_pack_kv). Уроки и эскалация -
[round5-d2-mmproj-vision-port.md](round5-d2-mmproj-vision-port.md).

Веточная mmproj-работа опиралась на models/gguf/dequant.py, который main удалил;
видение надо переложить на новый квант-пакет. Реестры vision-тензорных карт
пережили re-platform в [models/gguf/config.py](../../python/freetoken/models/gguf/config.py).
Исходный кейс и проверенные карты тензоров - [../cases/gguf-glm5next-vision/TASK.md](../cases/gguf-glm5next-vision/TASK.md).

### Открыто: gguf->FTW boot residual (новый пункт порт-долга)

Найдено волной D2 (2026-10-21), pre-existing - не специфика D2 и не D4-банки:
ЛЮБОЙ сконвертированный gguf->FTW чекпойнт не бутится на новом стеке, включая
text-only. checkpoint_quant_config ([models/register.py:458](../../python/freetoken/models/register.py))
строит таблицу ggml-типов из gguf_tensor_index(hf_config.model_path) (строка 469),
но для FTW model_path = carrier source_metadata.gguf (metadata-only,
tensor_count=0) -> пустая таблица -> KeyError "GGUF has no tensor 'token_embd'"
(_part_types, [layers/quantization/configs/gguf.py:42](../../python/freetoken/layers/quantization/configs/gguf.py),
вызов из scheme_for, строка 58). D4-boot-smoke не проявил: гнался только raw GGUF.
Направление фикса: FTW-ветвящийся источник dense-типов (мета FTW-индекса) или
docs-запрет gguf->FTW до фикса; reconversion не поможет. Фикс - отдельной задачей.

### D3. Гибрид/offload на per-format слот-пулах - готово 2026-10-21, коммит 393b682

Статус: готово 2026-10-21, коммит 393b682. Порт не понадобился: main УЖЕ вёз
гибрид на новом стеке, вернулись только три fail-fast гейта конфига (format
screen CPU-executor, header-отказ hybrid на непроверяемом gguf, pre-load
auto floor gate); T2-хелперы НЕ возвращены (однопульный executor с глобальным
бюджетом). Приёмка: 786k гибридный рецепт бут x2 без лестницы (ready 119/135 с,
3 пула 29-1103 слотов), гибридный декод 27.71 tok/s против offload 21.00 (+32%)
на том же стеке. Уроки порта -
[hybrid-moe-slot-pools-port.md](hybrid-moe-slot-pools-port.md).

Исходный план (до диспозиции): в [engine/engine.py](../../python/freetoken/engine/engine.py) веточные хвосты
сброшены по правилу «main выигрывает» как отложенные: partition gate ветки,
gguf hybrid rejection, хелперы T2 (host delivery), plural-cache state. NS-knob
в этот список не входит - он жил в kernel/triton/mhc.py (db0f6b3), не в engine.py.
Оставлено в слитном дереве: kv_quant-гейты, mmproj-kwargs, gguf strategy-гейты.
Экономика гибрида, под которую это затачивалось, - [moe-hybrid-cost-model.md](moe-hybrid-cost-model.md),
кейс T2 - [../cases/decode-t2-host-delivery/TASK.md](../cases/decode-t2-host-delivery/TASK.md).

### D4. glm5next GGUF-загрузчик - готово 2026-10-21, коммит 9314c9b

Статус: готово 2026-10-21, коммит 9314c9b: boot-smoke PASS (READY 145 s, chat
200, decode медиана 21.00 tok/s, rc 143), кросс-валидация 38/38 без перестановок.
Уроки порта и приёмка - [glm5next-gguf-loader-port.md](glm5next-gguf-loader-port.md).

У main его НИКОГДА не было - glm5next приезжал только веткой; HF-вариант glm5next
работает и после слияния. До посадки D4 GGUF-сервинг glm5next (включая гибридные
рецепты прежних кампаний, [../cases/gguf-glm5next-hybrid/PLAN.md](../cases/gguf-glm5next-hybrid/PLAN.md))
был НЕЗАГРУЖАЕМ в принципе. Сопровождающие шаги при посадке выполнены:
семантика gguf_types-защиты коммита 9ac8830 возвращена (гард в
load_ftw_banks_to_device, урок раунда 4 - см.
[merge-wave-procedure.md](merge-wave-procedure.md)), сырой ImportError заменён
внятным диагностическим сообщением.

### D5. Re-land generic-triton fp8/nvfp4-KV - готово 2026-10-21, коммит 660c814

Статус: готово 2026-10-21, коммит 660c814: store-стек пережил re-platform
байт-в-байт, потерян был только read-путь generic-triton; load_nvfp4
(decode/extend/split/paged) ре-ленднут НА текущую структуру ядер
(sinks/sliding-window/split-caps сохранены, None-шкалы байт-в-байт),
supports_fp8_kv + supports_nvfp4_kv у triton flipped вместе. Метрика спеки:
[test_kv_quant_config.py](../../tests/engine/test_kv_quant_config.py)
`-p no:skipping` 8 failed -> 36 passed; GPU: test_kv_nvfp4 115 passed,
test_triton_attention 49 passed. Качественный A/B: Qwen3.8-27B-NVFP4-FTW (dense
GDN+FULL; GLM-5.3 не годится - dsa): 3.46x ужатие KV без деградации, greedy
11/12 byte-identical (1 argmax-flip на открытой перефразе), needle @30% найден
идентично. fa / fi / trtllm - отдельные PR (внешнему kernel нужна наша
layout-схема). Уроки -
[round5-d5-nvfp4-kv-generic.md](round5-d5-nvfp4-kv-generic.md).

## Происхождение

Рабочие артефакты волны (discovery-report, adaptation-log со skip-бухгалтерией,
test-results) остались в git-ignored рабочей папке волны
(.tasks/merge-main-round5-work/adaptation-log.md) и переживут клон только в виде
этой статьи: она несёт durable-копию порт-долга, kb самодостаточен.
