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
механиках main'а; всё глубокое отложено в порт-долг D1-D5; статус: D4 готов
(2026-10-21), остальные - не начато. Эта статья - durable-карта того, ЧТО пере-реализовывать на новых механиках;
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

## Порт-долг D1-D5 (D4 готов, остальные - не начато)

### D1. Веточные победы ядра на вендоренном ggml

| Победа ветки (старый стек) | Якорное число (конфиг измерения) | Что пере-реализовать на новом стеке |
|---|---|---|
| v2 grouped MMQ (GGUF-префилл) | +16.4% @ батч 8128 (battery PASS) | grouped-MMQ механика поверх вендоренных ggml-ядер |
| v3 m-tile grouped MoE | +67.2% (свип m-tile, префилл); prod-дефолт MOE_MTILE=32 | m-tile-нарезка MoE-пути; дефолт тайла пере-мерить |
| dense q8_0 GEMM: N-split + tile64 | N-split +2.61%; tile64 +40.46% (A/B dense q8_0) | N-split и тайл 64 в dense q8_0 GEMM |
| mhc-цепочка NS=64 | +3.4% (paired A/B декода) | NS-нарезка mhc-цепочки (NS-knob живёт в kernel/triton/mhc.py, не в engine.py) |

Все числа измерены на УДАЛЁННОМ стеке csrc/gguf - на вендоренном ggml они
обнуляются до порта. Методика и полные конфиги: [prefill-throughput.md](prefill-throughput.md);
машиночитаемые якоря - [../baselines/README.md](../baselines/README.md).

### D2. mmproj-видение на поверхностях layers/quantization/*

Веточная mmproj-работа опиралась на models/gguf/dequant.py, который main удалил;
видение надо переложить на новый квант-пакет. Реестры vision-тензорных карт
пережили re-platform в [models/gguf/config.py](../../python/freetoken/models/gguf/config.py).
Исходный кейс и проверенные карты тензоров - [../cases/gguf-glm5next-vision/TASK.md](../cases/gguf-glm5next-vision/TASK.md).

### D3. Глубокая адаптация гибрида/offload к per-format слот-пулам

В [engine/engine.py](../../python/freetoken/engine/engine.py) веточные хвосты
сброшены по правилу «main выигрывает» как отложенные: partition gate ветки,
gguf hybrid rejection, хелперы T2 (host delivery), plural-cache state. NS-knob
в этот список не входит - он жил в kernel/triton/mhc.py (db0f6b3), не в engine.py.
Оставлено в слитном дереве: kv_quant-гейты, mmproj-kwargs, gguf strategy-гейты.
Порт должен заново связать эти механизмы с per-format слот-пулами #689/#700;
экономика гибрида, под которую это затачивалось, - [moe-hybrid-cost-model.md](moe-hybrid-cost-model.md),
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

### D5. Re-land generic-triton fp8/nvfp4-KV

После переписывания attention в main'е упакованный NVFP4-KV декодируют только
dsa и qsa_sparse ([attention/__init__.py](../../python/freetoken/attention/__init__.py):
supports_nvfp4_kv=True только у них; решение проводит engine через
_backend_supports_kv_quant / _validate_attention_backend_choice); triton / fa / fi /
trtllm отклоняют nvfp4 при старте. Re-land - это: вернуть generic-triton путь к
уцелевшему nvfp4-KV store-стеку ([kernel/triton/kv_nvfp4.py](../../python/freetoken/kernel/triton/kv_nvfp4.py),
[kernel/triton/kv_quant.py](../../python/freetoken/kernel/triton/kv_quant.py),
[kvcache/mha_pool.py](../../python/freetoken/kvcache/mha_pool.py)) и в том же
порте: (а) реактивировать два bsa-half теста в
[test_kv_quant_config.py](../../tests/engine/test_kv_quant_config.py) (строки 176
и 248, D5-skip'ы - пробным запуском подтверждено, что тесты проходят без скипа);
(б) закрыть активным покрытием уцелевший nvfp4-KV store-стек, пока у него нет
пользователей среди бэкендов.

## Происхождение

Рабочие артефакты волны (discovery-report, adaptation-log со skip-бухгалтерией,
test-results) остались в git-ignored рабочей папке волны
(.tasks/merge-main-round5-work/adaptation-log.md) и переживут клон только в виде
этой статьи: она несёт durable-копию порт-долга, kb самодостаточен.
