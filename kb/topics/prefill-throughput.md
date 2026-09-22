---
title: "Пропускная способность префилла: чем задаётся, как её поднимали и как измерять"
date: 2026-09-22
hardware: RTX 5090 32GB
branch-commits: "vektory79; цепочка ядерных кампаний 2026-09: base (293 tok/s) -> v2 grouped MMQ -> v3 stationary m-tile 1706aae -> dense q8_0 tile64 + N-split 8e2e4c7 -> MOE_MTILE=32 production default"
status: validated
tags: [prefill, throughput, chunk-size, mmq, dense-q80, m-tile, gguf, nvfp4, step0, measurement]
---

# Пропускная способность префилла

**Вердикт (читайте это первым).** Для serving'а с гибридным MoE главный рычаг
префилла — размер чанка (`--max-prefill-length`): каждый префилл-чанк делает
полный круг PCIe-копий банков экспертов по всем 42 MoE-слоям, поэтому
удвоение чанка вдвое сокращает число кругов на токен. На NVFP4-FTW это
измеренные ~510 tok/s при чанке 4096 против ~693-695 tok/s при 8192 (+36%).
Текущий якорь «голого» GGUF glm5next после всех ядерных кампаний —
792-808 tok/s при чанке 8128 (победитель тюнинга serving-флагов — тоже
8128-класс чанка). Потолок для ядра MMQ-класса — ~758 tok/s, после dense
tile64 он пройден.

## Механизм: почему размер чанка — рычаг

- Модель GLM-5.3-Flash имеет 42 routed-слоя с top_k 8. На каждый префилл-чанк
  движок материализует банки экспертов на GPU: по одному kernel'у
  `fast_index_copy_multi` на слой, читающему хост-банки по PCIe
  (~42 GB/s, ~3.15 s на 8128-чанк в step-0-разборе).
- Плюс фиксированные на-чанк накладные: прогрев/автотюн первого чанка,
  хвостовой чанк, синхронизации планировщика. Чем крупнее чанк, тем больше
  токенов амортизируют один круг копий.
- Оборотная сторона: крупный чанк ест больше transient-VRAM в пике заполнения
  и меняет сетку переиспользования radix-кэша — см. [vram-budget.md](vram-budget.md)
  и [radix-cache-reuse.md](radix-cache-reuse.md).

## Числа с конфигурациями

### GGUF glm5next (GLM-5.3-Flash-UD-Q3_K_XL), путь кампаний

Победитель тюнинга serving-флагов: `--memory-ratio 0.85
--max-prefill-length 8191` + radix mr1 (детали и матрица —
[REPORT.md](../cases/ft-gguf-serve-tuning/REPORT.md)).

| этап | префилл @8128 | что изменилось | где зафиксировано |
|---|---|---|---|
| база до ядерных кампаний | ~293 tok/s | ядро до правок | [baselines/README.md](../baselines/README.md) |
| v2 grouped MMQ | +16.4% | веса читаются один раз | [mmq-prefill-kernel](../baselines/mmq-prefill-kernel/README.md), метод [v2-ab.md](../methods/v2-ab.md) |
| v3 stationary m-tile | +67.2% (= 556.43 tok/s, тайл 32, коммит 1706aae) | weight-stationary расписание | [baselines/mmq-v3](../baselines/mmq-v3/v3a-ab-results.json), метод [v3a-ab.md](../methods/v3a-ab.md) |
| dense q8_0 tile64 | +40.46% | dense GEMM-тайл | [ab-mtile.md](../methods/ab-mtile.md) |
| N-split (kda_in_proj) | +2.61% | NSPLIT=2 | [ab-nsplit.md](../methods/ab-nsplit.md) |
| итог, production | **792-808 tok/s** | MOE_MTILE=32 (production-дефолт, 8e2e4c7), DENSE_MTILE=64, NSPLIT=2 | [baselines/README.md](../baselines/README.md) |

step-0-разбор чанка 8128 (метод [step0-profile-mmq.md](../methods/step0-profile-mmq.md)):
MoE GEMM 63% чанка, копии банков 11%, dense q8_0 GEMM 21%, остальное ~5%;
GPU никогда не ждёт CPU (idle 0.05%). Ядро MMQ-класса — ALU/BW-ограниченное,
его измеренный потолок ~758 tok/s (при 60% эффективной полосы: MoE 131.5 GB
за ~0.13 s на чанк); после dense tile64 и N-split этот потолок пройден —
дальнейшие рычаги лежат в другом (см. [moe-hybrid-cost-model.md](moe-hybrid-cost-model.md)
для декода).

### NVFP4-FTW (GLM-5.3-Flash-NVFP4-FTW), после iommu-фикса

| чанк | префилл | конфиг |
|---|---|---|
| 4096 | ~507-510 tok/s | user 1M-команда, hybrid, threads 16, `--disable-moe-prefill-overlap`, mr 0.89 ([nvfp4-ab-measurements.md](../cases/gguf-glm5next-hybrid/verification/nvfp4-ab-measurements.md)) |
| 8192 | ~691-695 tok/s | рецепты меню ниже в [vram-budget.md](vram-budget.md); 512k-рецепт `--memory-ratio 0.85 --max-prefill-length 8192` даёт ~693 tok/s |

Отношение +36% за удвоение чанка — прямое следствие механизма «круг PCIe-копий
на чанк». Prefill не зависит от MoE-стратегии (hybrid == offload, измерено в
[A/B-кампании](../cases/gguf-glm5next-hybrid/verification/nvfp4-ab-measurements.md)).

## Ловушки измерения (не читайте лог «в лоб»)

1. **Первый полный чанк — прогрев** (JIT/autotune): в логах это 76-145 tok/s
   класс. Всегда исключайте его из медианы.
2. **Последний ПОЛНЫЙ чанк — ложная строка**: его `input throughput`
   показывает невозможные ~1552-1602 tok/s (против измеренных 18 s GEMM на
   чанк). Это артефакт учёта финального чанка, не ускорение. Медиана считается
   по полным чанкам БЕЗ последнего (и без хвостового недополного).
3. **Авторитетная метрика** — строки серверного лога
   `input throughput (token/s)` по чанкам; клиентские измерения —
   [measure.py](../harness/serve-measure/measure.py) (патченный парсер SSE,
   декод-режим пропускает префилл-кадр).
4. Разброс день-в-день ~3% (282-283 против 291-293 tok/s на одной конфигурации);
   дневной шум однажды перекрыл реальные -10% — время суток пишите рядом с
   числом (правило из [baselines/README.md](../baselines/README.md)).
   При «мигающих» A/B — [arbitration-battery.md](../methods/arbitration-battery.md).

## Как измерить / проверить

- Живой сервер: [harness/serve-measure/measure.py](../harness/serve-measure/measure.py)
  + payload'ы `req_prefill.json`/`req_ladder.json`; общие гочвы прогонов — в
  [корневом README kb](../README.md) и [harness/README.md](../harness/README.md).
- A/B одной переменной (один boot на значение): шаблон
  [ab-nsplit.md](../methods/ab-nsplit.md); свип по тайлу —
  [ab-mtile.md](../methods/ab-mtile.md); лестница чанков — [v2-ab.md](../methods/v2-ab.md).
- «Есть ли вообще что выигрывать в ядре»: step-0-профиль —
  [step0-profile.md](../methods/step0-profile.md) и его MMQ-инстанс
  [step0-profile-mmq.md](../methods/step0-profile-mmq.md).
- Качество при смене ядра не должно деградировать: батарея —
  [quality-ab.md](../methods/quality-ab.md).
- Протокол волн (сериализация, census, watchdog) —
  [orchestration-notes.md](../methods/orchestration-notes.md).
- Машиночитаемые якоря — [baselines/README.md](../baselines/README.md).

Смежное: цена чанка в VRAM и выбор рецепта serving — [vram-budget.md](vram-budget.md);
декод-стоимость гибридного MoE — [moe-hybrid-cost-model.md](moe-hybrid-cost-model.md);
термины — [glossary.md](../glossary.md).