---
name: "ft-decode-lever-research"
description: "GGUF glm5next hybrid decode lever study: R1-R3 hypothesis closures, FA N/A for GDN+DSA arch, user-config gotcha"
type: project
lastUpdated: 2026-09-29T23:01
lastRecall: 2026-09-30T22:35
---

# Декод GGUF glm5next hybrid: рычаги оптимизации (research, 2026-09-26)

Cloud-LLM сессия; режим - только исследование (пользователь: фиксы не реализовывать,
реальные задачи оптимизации выносятся в отдельные сессии). Вердикты основаны на
kb/topics/moe-hybrid-cost-model.md, re-acceptance.md, ggml-cpu-kernel-study.md + коде HEAD 0ff8495.

## Вердикт: где находится узкое место декода
- Шаг @64k: 16.67 tok/s = 60 ms = 1.43 ms/слой all-in; bound на CPU-ноге
  (1.61 GiB/шаг; пул занят ~59% окна декода, burst ~45 GB/s ~= бенч 50).
- GPU-цепочка (dense q8_0 MMVQ, moe_vec-хиты, Triton kpool-DSA attention, GDN fla)
  = единицы ms/шаг, полностью перекрыта - НЕ рычаг. Лишних копий нет (per-слой
  KB-масштаб pinned D2H/H2D + необходимый PCIe gather в _decode_hybrid).
- Фиксированные ~17 ms CPU_exposed/слой не сокращаются объёмом (fetch3 flat) -
  это per-layer фиксированные расходы (doorbell dispatch, pool wake, barrier tails),
  не compute volume.

## Судьба гипотез (измерено 2026-09-26/27)
- R1 "26.9 GB/s эффективных" - idle-inflated framing; атаковать фиксированные расходы
  (T2), не BW ноги. Итог T2: minor-pool starvation на 1 E-core, t2d-кноб +7.9/+10.4%,
  committed - ft-t2-host-delivery-wave1.
- R2 flag-sync swap: ЗАКРЫТ - FREETOKEN_CPU_MOE_FLAG_SYNC=0 = -21.6% (+17 ms
  hostfunc-штраф); memop-doorbell в проде прав.
- Гипотеза сплита: ЗАКРЫТА - дискриминатор E4 (--moe-hybrid-max-fetch 3) = -0.9%,
  нейтрально; объём сплита не рычаг.
- R3 frequency-aware LRU: REFUTED оракулом (oracle_hit 25.7% vs LRU 25.1% -
  cache policy dead; миссы режет только больше VRAM-слотов).
- Полная 5-рукавная A/B, аддитивная модель шага, T1-декомпозиция блока 44 ms и chain
  исходов кампании: ft-decode-research-campaign; инструменты T1 -
  ft-decode-stats-instruments-unwired; offload gather (33.5-36 GB/s vs 43.6 benched,
  closing only ties hybrid) - ft-offload-decode-gather-gap.

## Flash Attention - неприменим архитектурно
У glm5next НЕТ полных attention-слоёв: GDN/KDA (linear attention) + sparse MLA с
DSA-индексером, nope-only (rope_dim==0 ассерт в models/glm5_next/gguf.py).
Декод-внимание = Glm5NextDSABackend (attention/dsa_indexer_kpool.py): кастомный
Triton gathered-KV кернел (gather->score->top-k->attend) с kpool-сжатием индекса 4x -
дешевле FA по полной истории. fa.py (sgl_kernel), fi.py (FlashInfer), trtllm.py -
для плотных архитектур.

## Математика оптимальна
ggml AVX2 W4A8K порт (11f1a80) - ISA-потолок этого CPU (AVX-512 fused off,
см. rtx5090-pcie-gen5-bw-cap); ядра DRAM-bound. Ускорять надо доставку, не арифметику.

## Готча конфигов
Ежедневная команда пользователя ОТЛИЧАЕТСЯ от якоря 16.67 (63b9bff): FTW-директория
+ --memory-ratio 0.82 + --kv-reserve-tokens 400000 + tier 10/50 + 8191 против
якорных raw GGUF + 0.85 + 4096 без tier. Сверять осторожно.

## Харнес кампании
.tasks/decode-research/: run_arm.py (boot точного конфига пользователя, порт 18801,
fill 64k из .tasks/ft-gguf-serve-tuning/filler.txt, radix-reuse декод 768 токенов,
per-CPU /proc/stat+MHz сэмплер, SIGTERM-teardown), campaign.py (сериальные руки:
base / fetch3 / ov0 / flagsync0 / fetch0), analyze.py (сводная таблица + разбор
busy%/MHz по P/E ядрам). Якорь 16.67 воспроизведён (base 16.29, дневной шум).
