---
title: "VRAM-бюджет serving'а: headroom, greedy-fill планировщика и меню рецептов"
date: 2026-09-22
hardware: RTX 5090 32GB
branch-commits: "vektory79; VRAM-headroom волна 2026-09-20 @ 315b278 (код идентичен 8e2e4c7); инцидент crash-mr080-400k (OPEN); nvfp4-1m-capacity (CLOSED)"
status: validated
tags: [vram, headroom, memory-ratio, kv-reserve, num-tokens, moe-cache-size, fail-fast, recipes, nvfp4, gguf]
---

# VRAM-бюджет serving'а на 32 ГБ

**Вердикт (читайте это первым).** Рецепт-победитель GGUF-тюнинга
(0.80/400k fp8, чанк 8191) требует **>= 29.28 GiB свободной VRAM на старте**
и падает при меньшем headroom (инцидент OPEN). Ни `--kv-reserve-tokens`, ни
один `--moe-cache-size` НЕ освобождают VRAM: планировщик greedy — урезанный
бюджет KV он наливает обратно в слоты MoE. Освободить VRAM у KV можно только
явным потолком `--num-tokens` (это CLI-флаг; `--num-token-override` — НЕ
CLI-флаг, есть соседний `--num-pages`). Проверенный рычаг без потери скорости
— размер чанка: 8191 -> 4095-класс даёт +2.0 GiB в пике ценой ~-26% префилла.

## Механизм: что занимает VRAM и что решает планировщик

- Пул кэша масштабируется от `--memory-ratio`: **1500 MiB на каждый шаг
  0.05 ratio** (измерено: пик 32103 -> 30603 MiB при 0.85 -> 0.80). Пик
  «дышит» внутри ratio-конверта, а не по байтам KV.
- Рабочий набор гибридного MoE: 42 serving-слоя x top_k 8 = **336 пар
  (слой, эксперт)**; пул слотов должен накрывать его, иначе fail-fast.
- Транзиенты префилла растут с чанком: 8191 -> 4095-класс дал **+2020 MiB**
  в пике при идентичном плане (1148 слотов / 7830 страниц) — чистое снижение
  префилл-транзиентов. Для 8192/8191-чанков нужен ratio 0.85: пик транзиентов
  с triton do_bench (256 MiB) доходит до **~3.4-3.5 GiB при 64k-заполнении**
  (старая оценка ~2.4 GiB занижена).
- Greedy-fill: **урезание `--kv-reserve-tokens` НЕ создаёт headroom** —
  освобождённые KV-байты планировщик наливает в новые слоты MoE (измерено:
  500k -> 350k reserve = 0 MiB выгоды, слоты 1148 -> 1236). Пик остаётся
  прижат к ratio-конверту.
- Явный `--moe-cache-size` тоже сам по себе НЕ освобождает VRAM: план
  заливает сэкономленное в больший KV. Работает только пара
  «ограничить кэш экспертов + ограничить KV потолком `--num-tokens`»
  (поведение уровня PR #198). `--num-token-override` не существует в CLI;
  соседний флаг страниц — `--num-pages`.

## Boot-порог рецепт-победителя и desktop-налог

- 0.80/400k fp8 (GGUF, чанк 8191) грузится при 29.54 GiB свободных до
  загрузки (замер 2026-09-16, llama-swap выключен; пик заполнения оставляет
  1.96 GiB). Порог выживания — **29.28 GiB**; ниже — падение на старте
  ([инцидент crash-mr080-400k](../incidents/crash-mr080-400k/README.md), OPEN).
- Desktop-налог переворачивает исход: remote-desktop ~492 MiB + idle-обёртка
  llama-swap ~1.27 GiB — каждый из них способен закрыть маржу. Перед
  сравнением boot'ов вычитайте их из free-after-init (замечание в
  [REPORT.md](../cases/ft-gguf-serve-tuning/REPORT.md)).
- Fail-fast gate: план сверяется с минимальным фондом слотов по сигнатурным
  группам; для winner-рецепта план ~1091-1093 слота против пола ~1059
  слот-эквивалентов (маржа 32-34) — отсюда чувствительность к каждому
  украденному MiB. Тексты ошибок gate — в
  [failfast-report.md](../incidents/nvfp4-1m-capacity/failfast-report.md) и
  [error-report.md](../incidents/crash-mr080-400k/error-report.md).

## Меню рецептов (NVFP4-FTW, threads 16, после iommu-фикса)

| цель | рецепт | префилл | декод | замечания |
|---|---|---|---|---|
| 1M безопасный | `--moe-cache-size 340 --max-prefill-length 6144` | ~618 tok/s | ~нейтрально | устойчивый вариант |
| 1M макс-префилл | `--moe-cache-size 288 --num-tokens 1048576 --max-prefill-length 8192` | ~691 tok/s | -4.5% | deep-fill хрупкий; workspace DSA-индексера ~0.62 KB на контекст-токен растёт в тот же headroom |
| 512k | `--memory-ratio 0.85 --max-prefill-length 8192` | ~693 tok/s | 15.5 tok/s @64k | простой и быстрый |

Контекст 1M: верифицированный максимум 786432 токена, потолок заполнения
~678k, защита — fail-fast gate
([verification-786k.md](../incidents/nvfp4-1m-capacity/verification-786k.md),
[verification-deepfills.md](../incidents/nvfp4-1m-capacity/verification-deepfills.md),
инцидент CLOSED). Сам boot 1M-nvfp4 был fail-fast лишь один раз (2026-09-14,
VRAM-сдвиг рабочего стола) — транзиент: 09-15 тот же command бутится
(341 слот >= 336 рабочего набора;
[nvfp4-ab-measurements.md](../cases/gguf-glm5next-hybrid/verification/nvfp4-ab-measurements.md)).

GGUF-сторона (реальные файлы Q3_K_XL): измеренная матрица
ratio/reserve/chunk и таблица knob-эффектов —
[vram-headroom.md](../methods/vram-headroom.md); там же подтверждено, что
radix-хиты не деградируют вплоть до 351k-токеных KV-пулов.

## Как измерить / проверить

- Матрица «knob -> FREE VRAM в пике -> цена в tok/s»:
  [vram_headroom_run.py](../harness/ab-runner/vram_headroom_run.py) по методу
  [vram-headroom.md](../methods/vram-headroom.md); машиночитаемые итоги —
  [vram-headroom-results.json](../baselines/dense-q80-gemm/vram-headroom-results.json).
- Метрика = peak-free на пике заполнения (сэмплер 2 s с фазовыми тегами) и
  free-before-load на буте; сравнивайте только при одинаковом desktop-фоне.
- Воспроизведение crash-mr080: [crash-mr080-repro.sh](../harness/repro/crash-mr080-repro.sh).
- Связки: цена чанка в скорости — [prefill-throughput.md](prefill-throughput.md);
  slot-floor против desktop-налога и его роль в radix-переиспользовании —
  [radix-cache-reuse.md](radix-cache-reuse.md); термины —
  [glossary.md](../glossary.md).