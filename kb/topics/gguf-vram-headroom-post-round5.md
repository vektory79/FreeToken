---
title: "VRAM-headroom GGUF-сервинга после re-platform раунда 5: почему рецепт сентября перестал держать пик"
date: 2026-10-24
hardware: "RTX 5090 32607 MiB (desktop-фон волны 796 MiB)"
commits: "vektory79 @ ddefb5e; стек тега 2026.10.17 (b8c3794) как база сравнения"
status: "verified (замеры железа)"
tags: [vram, headroom, memory-ratio, max-prefill-length, slot-floor, triton, mmq, session-tier, round5, gguf]
---

# VRAM-headroom после re-platform раунда 5

**Вердикт (читайте это первым).** Сентябрьский профиль-победитель
(`--memory-ratio 0.82 --kv-reserve-tokens 400000 --max-prefill-length 8191`)
на пост-round5-стеке (HEAD ddefb5e) держит всего **641 MiB** свободных в пике
8k-префилла (пик 31966 MiB из 32607) - обычный дрейф desktop-фона роняет
генерацию в OOM. Бут при этом выглядит здоровым: план 1127 слотов / 6256
страниц, "Free memory after initialization: 5.00 GiB", после CUDA graphs
4.97 GiB. Планировщик не виноват: формула бюджета байт-в-байт прежняя, выросли
внеплановые потребители вокруг плана. Устойчивый рецепт: **ratio 0.82 +
`--max-prefill-length 4095`** - запас 1.2-2.1 GiB на любой глубине ценой
-53% префилла. Полная скорость: **ratio 0.80 + чанк 8191** (запас 1.2 GiB) -
только на лёгком десктопе: бут-маржа +10 слотов над полом 1059.

Модель волны: GLM-5.3-Flash-UD-Q3_K_XL-FTW-vision-new (носитель, конверт
2026-10-24), fp8 KV, hybrid, moe-cpu-threads 16, max-running-requests 1,
session tier L1 10 GiB / L2 50 GiB.

## Симптом и воспроизведение

- Конфиг пользователя: ratio 0.82, kv-reserve 400k, чанк 8191.
- Бут без аномалий: план 1127 слотов / 6256 страниц, free-after-init
  5.00 GiB, 4.97 GiB после CUDA graphs.
- Замер 8k-префилла (unique-nonce промпт, параллельный 2 с сэмплер - метод
  [../methods/vram-peak-measurement.md](../methods/vram-peak-measurement.md)):
  пик 31966 MiB занято, свободно 641 MiB. OOM приходит на генерации, когда
  desktop-фон подрастает.
- На стеке тега 2026.10.17 (b8c3794) тот же профиль даёт ~1.4 GiB свободных
  (интерполяция по замерам сентябрьской волны,
  [../methods/vram-headroom.md](../methods/vram-headroom.md)). Разница - не
  в конфиге, а в стеке.

## Разбор причин: план не изменился, выросли внеплановые потребители

Статический diff b8c3794..HEAD: машинерия планировщика байт-в-байт та же -
бюджет = ratio * baseline_free - weights - fixed_cache
(python/freetoken/engine/cache_budget.py:39-45); остаток (1-ratio) -
транзиентный headroom, который план не бронирует. OOM поджимает именно
остаток: новые потребители тратят его в пике.

Новые внеплановые потребители на HEAD:

- **Triton split-KV decode scratch**: `_MAX_KV_SPLITS` 8 -> 64 с масштабом
  по числу SM (python/freetoken/kernel/triton/attention.py:44 и 72-74;
  upstream #729 "split triton decode attention by context length and sm
  count"). fp32-буферы attn_logits захватываются в CUDA graphs и живут
  постоянно.
- **ggml-cuda MMQ scratch**: fp32-де-квантование x32 на каждый GEMM -
  транзиент, пик на большом батче префилл-чанков (csrc/ggml, ggml_ops.cu).
- **Сдвиг baseline-библиотек**: torch 2.14.1 / triton 3.8 / flashlib 0.4.0
  (merge-пункты #630, #669).
- **Новый носитель (carrier)**: другая раскладка тензоров - другие байты
  на слот.

Что НЕ является причиной (проверено): vision-энкодер (дефолт
`--mm-encoder-weights=host` стримит блоки башни с хоста, ~60 MiB резидентно -
python/freetoken/mm/config.py:21), CUDA graphs (захвачен только bs=1,
~30 MiB), `--enable-cache-report` (только отчётность), session tier
(RAM/SSD, не VRAM).

Замер транзиентов над пулом на 8k-префилле: ~4.8 GiB на HEAD против
~3.9-4.1 GiB на старом стеке (~+0.8 GiB) - это и есть съеденная маржа.

## Измеренная матрица (HEAD, десктоп 796 MiB, unique-nonce промпты)

Конфиг = `--memory-ratio / --kv-reserve-tokens / --max-prefill-length`
(эти три менялись, остальное как в шапке статьи):

| ratio/reserve/chunk | пик-free | префилл | декод | вердикт |
|---|---|---|---|---|
| 0.82/400k/8191 | 641 MiB | 2129 tok/s | 14-20 tok/s | OOM-хрупкий: дрейф десктопа убивает |
| 0.80/400k/8191 | 1239 MiB | 2129 tok/s | 15-21 tok/s | полная скорость, но бут-маржа +10 слотов (пол 1059) - бритва |
| 0.82/400k/4095 | 2097 MiB | 997 tok/s (-53%) | 21 tok/s | устойчивый рецепт |

Рычаги (HEAD):

- ratio: ~300 MiB пик-free на каждый шаг 0.01 (не изменилось с сентября).
- Пол ratio на новом стеке - **0.80 при reserve 400k**: 0.80 бутится 1069
  слотами против пола 1059; 0.79 slot-floor gate не пропустит.
- Чанк 8191 -> 4095: **+1456 MiB** пик-free.
- `--kv-reserve-tokens` - по-прежнему no-op для пика VRAM: освобождённые
  KV-байты планировщик наливает обратно в слоты MoE (подтверждено снова
  этой волной).

## Лестница глубины (0.82/400k/4095, растущий radix-префикс)

Промпт растёт шагами ~22k токенов через общий radix-префикс (метод -
[../methods/vram-peak-measurement.md](../methods/vram-peak-measurement.md)),
все шаги HTTP 200; свободная VRAM по факту запроса:

| prompt_tokens | свободно, MiB |
|---|---|
| 22k | 1817 |
| 89k | 1618 |
| 200k | 1622 |
| 267k | 1520 |
| 311k | 1380 |
| 356k | 1240 |
| 378.6k | 1180 (94.6% пула 400384 токенов) |

Эрозия глубины ~640 MiB от мелкого к глубокому: workspace DSA-индексера плюс
NT-масштабирование чанк-транзиентов. Итог: рецепт держит >= 1.18 GiB на всей
глубине.

## Рецепты запуска

- **Устойчивый дефолт**: `--memory-ratio 0.82 --max-prefill-length 4095`
  (запас 1.2-2.1 GiB на любой глубине; префилл 997 tok/s - всё равно быстрее
  сентябрьских 805 tok/s на чанке 8191).
- **Полная скорость**: `--memory-ratio 0.80 --max-prefill-length 8191`
  (запас 1.2 GiB) - только при лёгком десктопе: бут-маржа +10 слотов, это
  ~100-150 MiB толерантности к desktop-сдвигу.
- **Fail-fast на буте** ("budget cannot fund the per-signature slot floors",
  ~17 с, до загрузки весов) = ratio занижен: шагните назад вверх (0.81/0.82).
- `--kv-reserve-tokens` ради headroom не трогать - не работает (см. рычаги).

## Семантика L2-капа session tier: LRU-вытеснение первое, отказ - крайний случай

Вопрос "почему нет LRU-вытеснения" снят чтением кода - вытеснение есть.
`_reserve_blob_span` (python/freetoken/scheduler/session_tier.py:981) при
переполнении капа сначала вызывает `_evict_oldest_l2` (:1526): кандидаты -
сегменты in_l2 с refs==0, жертва = min по (last_validation, seg_id), цикл,
пока padded-потребность не поместится. Отказ ("L2 cap reached; offer
dropped" :711 / "cannot demote seg" :834) - только когда вытеснение не может
освободить достаточно: сегменты pinned посреди restore, span больше всего
вытеснимого объёма, дедуп-общие экстенты возвращают меньше номинала.

Блоб L2 append-only: discard'ы оставляют дыры и durable tombstones (P13);
автоматический компакт возвращает объём позже. Наблюдение волны: каталог L2
49.94/50 GiB - старые сессии ротируются молча; поднять кап - значит лишь
держать больше истории, а не "освободить место".

## Побочные находки

- **benchbw-профиль устарел для нового движка** ("version 4, need 5;
  ignoring it, rerun ft bench bw"): гибридный fetch-кап залипает в 1, пока
  профиль не перегенерирован `ft bench bw` - микробенчмарки fetch до
  рефреша неверны.
- **kv-reserve-tokens - no-op для пика VRAM** (подтверждено снова): рычаг
  плана и декода, не headroom.

## Как измерить / связанные статьи

- Метод честного замера пика:
  [../methods/vram-peak-measurement.md](../methods/vram-peak-measurement.md).
- Сентябрьская матрица старого стека:
  [../methods/vram-headroom.md](../methods/vram-headroom.md); меню рецептов
  и boot-пороги старого стека: [vram-budget.md](vram-budget.md).
- Харнесс матрицы:
  [../harness/ab-runner/vram_headroom_run.py](../harness/ab-runner/vram_headroom_run.py).
- Откуда рост внеплановых потребителей: re-platform раунда 5 -
  [round5-ggml-replatform-deferred-ports.md](round5-ggml-replatform-deferred-ports.md).
- Устройство session tier и его L2:
  [session-cache-tiering.md](session-cache-tiering.md).
- Термины (slot floor, benchbw, headroom):
  [../glossary.md](../glossary.md).
