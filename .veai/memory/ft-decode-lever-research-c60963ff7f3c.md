---
name: "ft-decode-lever-research"
description: GGUF glm5next hybrid decode lever study; A/B verdicts + T1 decomposition; T3 refuted by oracle; offload measured
type: project
lastUpdated: 2026-09-26T23:12
lastRecall: 2026-09-26T23:45
---

# Декод GGUF glm5next hybrid: рычаги оптимизации (research, 2026-09-26)

Cloud-LLM сессия; режим - только исследование (пользователь: фиксы не реализовывать,
реальные задачи оптимизации выносятся в отдельные сессии). Вердикты основаны на
kb/topics/moe-hybrid-cost-model.md, re-acceptance.md, ggml-cpu-kernel-study.md + коде HEAD 0ff8495.

## Вердикт: где находится узкое место декода
- Шаг @64k: 16.67 tok/s = 60 ms = 1.43 ms/слой all-in; bound на CPU-ноге
  (1.61 GiB/шаг при 26.9 GB/s эффективных ~ весь шаг).
- GPU-цепочка (dense q8_0 MMVQ, moe_vec-хиты, Triton kpool-DSA attention, GDN fla)
  = единицы ms/шаг, полностью перекрыта - НЕ рычаг. Лишних копий нет (per-слой
  KB-масштаб pinned D2H/H2D + необходимый PCIe gather в _decode_hybrid).
- Фиксированная синхронизация уже хорошо перекрыта (1.43 < банд 0.6-1.0 ms + нога).

## Ранжированные рычаги
- R1 (главный): production CPU-нога 26.9 GB/s vs бенч 50 vs STREAM 75.84. Суммарный
  host-DRAM трафик ~2.17 GiB/шаг (нога 1.61 + фетч 0.56, фетч тоже читает host RAM)
  = ~36 GB/s - DRAM НЕ насыщен. Подозреваемые: контеншн ноги и фетч-копий за DDR5,
  асимметрия пулов [15,1,1] (миноритарные по 1 потоку на E-ядрах 23/24), P/E-микс
  (8P+7E), governor/частоты.
- R3: host-трафик = миссы x 10.88 MB и НЕ зависит от сплита f. Ручки: hit-rate
  (LRU -> frequency-aware; оракул - decode_routing_stats), VRAM-бюджет кеша.
- R2: 2 host round-trip'а/слой (flag-sync memop doorbell по умолчанию;
  FREETOKEN_CPU_MOE_FLAG_SYNC=0 -> hostfunc) - резерв мал, исследовать после R1.
- Гипотеза сплита: production эффективные скорости ног почти равны (CPU 26.9 ~
  PCIe 28.4), а f=30.7% выведен benchbw из бенч-скоростей -> оптимум в проде может
  быть ~50%. Дискриминатор: --moe-hybrid-max-fetch 3 (E4).

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

## Харнес кампании (создан 2026-09-26, результаты на момент сохранения pending)
.tasks/decode-research/: run_arm.py (boot точного конфига пользователя, порт 18801,
fill 64k из .tasks/ft-gguf-serve-tuning/filler.txt, radix-reuse декод 768 токенов,
per-CPU /proc/stat+MHz сэмплер, SIGTERM-teardown), campaign.py (сериально 4 руки:
base / fetch3 / ov0=FREETOKEN_HYBRID_OVERLAP=0 / flagsync0), analyze.py (сводная
таблица + разбор busy%/MHz по P/E ядрам).

## Итог A/B (2026-09-26, decode-research кампания) - пересмотр гипотез

Полные результаты, 5-рукавный A/B и аддитивная модель шага - в ft-decode-research-campaign + .tasks/decode-research/. Что изменилось относительно разделов выше:
- R1 переформулирован: "26.9 GB/s эффективных" - idle-inflated (пул занят ~59% окна декода, burst ~45 GB/s ~= бенч 50); фиксированные ~17 ms CPU_exposed на слой не двигаются объёмом - атаковать фиксированные расходы (T2), а не BW ноги.
- Гипотеза сплита ЗАКРЫТА: дискриминатор E4 (--moe-hybrid-max-fetch 3) = -0.9%, нейтрально; объём сплита не рычаг.
- R2 ЗАКРЫТ: FREETOKEN_CPU_MOE_FLAG_SYNC=0 -> -21.6% (+17 ms hostfunc-штраф); memop-doorbell в проде прав.
- R3 (frequency-aware LRU) остаётся кандидатом (T3); GPU-цепочка/математика/FA - подтверждённые не-рычаги.
- Харнес: кампания завершена (base 16.29 tok/s @64k, якорь 16.67 воспроизведён в пределах дневного шума); следующий шаг - декомпозиция блока 44 ms, блокирована отсутствием CLI-инструментации (см. ft-decode-stats-instruments-unwired).

## 2026-09-26 evening: T1 executed - block decomposed, T3 REFUTED, offload measured
Instrumentation works: driver v3 (.tasks/decode-research/driver/sitecustomize.py) hooks GraphRunner.replay (decode steps are CUDA-graph REPLAYS - model host code does NOT run per step; _decode_routed only during capture; replay hook + CUDA-event step spans + collect_stats/collect_decode_freq flipped pre-capture). kineto torch.profiler CRASHES the scheduler (native abort at window open) - events-only mode is the only working profiler path on this box. Eager run (--cuda-graph-max-bs 0) needed for the routing histogram (host scatter is not replayed under graphs).
Findings: hybrid stream span 59.1 ms of ~62 wall (offload 73.6/72.9) - step fully serialized; the block = stream-wait-for-CPU-pool (~40 ms) + fetch copies (~6-8 ms) + kernels. Combined host delivery 2.57 GiB/step at ~41.5 GB/s vs benchbw overlapped pair 72 GB/s concurrent-capable -> the gap is per-layer ping-pong micro-gaps (pool ramp/tails, doorbell dispatch) = the live T2 lever. T3 REFUTED by oracle: oracle_hit 25.7% vs actual LRU 25.1% (dominant 39-layer partition, 8.6 slots/layer, working set 184/288, norm entropy 0.83) - cache policy is dead; only more VRAM slots cuts misses. Offload on user config: 13.72 tok/s (72.9 ms), gather ~33.5-36 GB/s effective vs 43.6 benched - closing it only ties hybrid; offload improvements deprioritized. Eager decode 15.84 ~= graphed 15.82 (CUDA graphs ~neutral at bs=1); miss stats identical across modes. Driver bugs that crashed two boots: KeyError on _e1 (fixed + try/except armor - any driver error now disables the driver, never the engine).
