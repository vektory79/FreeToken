---
title: "Стоимость гибридного MoE в декоде: модель слоя, синхронизация и пределы рычагов"
date: 2026-09-22
hardware: RTX 5090 32GB
branch-commits: "vektory79; гибридная кампания GGUF: ggml integer-kernel порт 11f1a80 + 979e3fc + фикс 63b9bff (re-acceptance), кампания закрыта на 63b9bff"
status: validated
tags: [hybrid, offload, moe, decode, pcie, cpu-gemv, fetch-fraction, benchbw, nvfp4, gguf]
---

# Модель стоимости гибридного MoE (декод)

**Вердикт (читайте это первым).** Стоимость декод-шага гибридного MoE:

```
шаг = max(CPU-нога, f x misses x PCIe) + ~0.6-1.0 ms фиксированной синхронизации
```

**Фиксированного «пола хэндшейка» 1.9-3 ms/слой НЕ существует** — ранняя
бисект-оценка ошибочно записала объём PCIe-фетча в «синхронизацию». Финальный
GGUF-гибрид: **16.67 tok/s @64k = 1.43 ms/слой all-in** при f=30.7% и
CPU-ноге 26.9 GB/s эффективных — гибрид РЕКОМЕНДОВАН против offload
(12.97 tok/s). NVFP4-гибрид быстрее своего offload в 1.39-1.43x.

## Механизм

- На каждый декод-шаг маршрутизатор просит 42 x top_k 8 = 336 пар
  (слой, эксперт); при 64k контекста миссы ~63-72% пар. Гибрид делит миссы
  по соответствию полос: долю `f = pcie_ov/(pcie_ov+cpu_ov)` тянут по PCIe,
  остальное считает CPU-нога (GEMV в DDR5) с перекрытием.
- Пары: доля f и объём фетча эндогенны — БЫСТРАЯ CPU-нога одновременно
  считает миссы дешевле И сокращает объём, который надо тянуть по PCIe
  (планировщик снижает f). Поэтому «пол хэндшейка» не фиксирован: NVFP4
  фетчит 1.06 эксперта/слой (~15 МБ) — его полный шаг 72.6 ms = 1.73 ms/слой,
  НИЖЕ заявленного «поля»; GGUF-класс до порта фетчил ~5.4-6 экспертов/слой,
  и копия ~1.7-2.5 ms/слой записывалась в «хэндшейк». Разбор противоречия:
  [nvfp4-ab-measurements.md](../cases/gguf-glm5next-hybrid/verification/nvfp4-ab-measurements.md), раздел 5.

## Числа с конфигурациями

### GGUF (GLM-5.3-Flash-UD-Q3_K_XL), после ggml-порта, REAL файл

Команда A/B (одна переменная `--moe-strategy`): `ft serve --moe-cache-auto
--num-tokens 70016 --memory-ratio 0.85 --max-prefill-length 4096
--moe-strategy {hybrid|offload} --moe-cpu-threads 16` (полная команда и
метод — [re-acceptance.md](../cases/gguf-glm5next-hybrid/verification/re-acceptance.md)).

| метрика | hybrid | offload |
|---|---|---|
| декод @64k (767 токенов) | **16.67 tok/s** | 12.97 tok/s |
| шаг @64k (на слой) | 60.0 ms (**1.43 ms/слой**) | 77.1 ms (1.84) |
| fetch-доля | f=30.7% | n/a |
| PCIe-трафик/шаг (расчёт) | ~0.56 GiB (~9.5 GB/s) | ~2.14 GiB (~28.4 GB/s) |
| CPU-нога/шаг (расчёт) | ~1.61 GiB (**26.9 GB/s эфф.** при 50.0 бенчед) | - |
| короткий декод 2k / burst | 16.12 / 15.42 | 14.32 / 14.21 |

Три гейта re-acceptance пройдены (>=14.3, паритет-обгон offload, burst);
батарея качеств 24/24. Вердикт кампании: hybrid RECOMMENDED
([re-acceptance.md](../cases/gguf-glm5next-hybrid/verification/re-acceptance.md)).

### NVFP4 (GLM-5.3-Flash-NVFP4-FTW), 1M-команда, hybrid vs offload

| метрика | hybrid | offload | выигрыш |
|---|---|---|---|
| декод 2k / @64k | **14.46 / 13.77 tok/s** | 10.12 / 9.91 | 1.43x / 1.39x |
| ms/слой @64k | 1.73 | 2.40 | - |
| fetch-доля | f=24.9% | n/a | - |
| PCIe H2D / CPU-нога на шаг | 0.64 GiB / 2.85 GiB | 3.36 GiB / 0 | - |

Почему выигрывает: целочисленная W4A8-нога AVX2+VNNI (55.6 GB/s overlap)
снимает ~75% миссов с PCIe на DDR5 (f=24.9%), DDR5 при 40 GB/s не узкое
место — шаг упирается в GPU-dense + копию + синхронизацию, а не в PCIe.
Полная декомпозиция и объяснение разрыва: [nvfp4-ab-measurements.md](../cases/gguf-glm5next-hybrid/verification/nvfp4-ab-measurements.md).

### GGUF scalar-нога: почему f было 78% и что с этим делать

- fp32-LUT scalar-нога GGUF: 11.3 GB/s бенчед; в production 3
  пер-сигнатурных партиции с ровным сплитом потоков голодали её до
  **2.96 GB/s** эфф. -> планировщик поднимал f до 78% -> ~6 экспертов/слой
  тянулось по PCIe. Не аппаратный предел — отсутствующий tier: точка
  диспетчеризации уже есть (`select_ggufdot`), сравнение путей —
  [nvfp4-ab-measurements.md](../cases/gguf-glm5next-hybrid/verification/nvfp4-ab-measurements.md), раздел 6.
- Рычаг — портирование целочисленного ggml-ядра для GGUF-форматов с гейтом
  микробенчмарка **>= 40 GB/s устойчивых** (изучение ядер —
  [ggml-cpu-kernel-study.md](../findings/gguf-glm5next-hybrid/ggml-cpu-kernel-study.md),
  пере-объём задачи — [task06-rework-estimate.md](../findings/gguf-glm5next-hybrid/task06-rework-estimate.md)).
  Быстрая нога заодно снизит f и объём фетча (эндогенная связка).
- После порта (текущее состояние) ноги GGUF: CPU-нога 26.9 GB/s эфф.,
  дальше на флаговом уровне рычагов нет — `--moe-cpu-threads` 16 == 20
  (шум после iommu-фикса); остаточный разрыв = пер-слойная синхронизация
  (~0.6-1.0 ms) + полоса DRAM CPU-ноги.

## Как измерить / проверить

- Декод: ре-отправка того же 64k-промпта (`#cached-token: 64512` в логе),
  стрим, TTFT исключён; метод и таблицы —
  [decode-numbers.md](../cases/gguf-glm5next-hybrid/verification/decode-numbers.md),
  клиент — [measure.py](../harness/serve-measure/measure.py).
- Декомпозиция шага: collect-stats дамп на SIGTERM (misses/fetched/cpu на
  слой по пулам) + сэмплер `/proc/stat` 1 Гц; вывод banner `moe_collect_stats=False`
  не читать — дамп действителен.
- Бенчмарк ног: benchbw-профиль и микробенчи —
  [task-03-benchbw-cpu-moe-gguf.md](../cases/gguf-glm5next-hybrid/task-03-benchbw-cpu-moe-gguf.md);
  зонд шин — [pcie_bw_probe.py](../harness/probes/pcie_bw_probe.py) и
  [ram_bw.py](../harness/probes/ram_bw.py). Осторожно: прогон benchbw
  переписывает ПОЛНЫЙ профиль на GPU целиком — однотипный прогон затирает
  доли других форматов; после замены ног пере-профилируйте все форматы.
- Машиночитаемые якоря кампании — [baselines/README.md](../baselines/README.md).
- Смежное: префилл — [prefill-throughput.md](prefill-throughput.md);
  VRAM-план гибрида — [vram-budget.md](vram-budget.md); термины —
  [glossary.md](../glossary.md).