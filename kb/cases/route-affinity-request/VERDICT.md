---
title: "Вердикт кейса route affinity на запрос: Ф0 готова, Ф1 PASS (per-request работает на bs=1), цена лестницы при bs=[1] ~10 MiB и ~0 с"
date: 2026-11-16
hardware: "RTX 5090 32GB; GGUF GLM-5.3-Flash-UD-Q3_K_XL (FTW-vision-new), moe-strategy hybrid, прод-графы, memory-ratio 0.81 / kv-reserve 800k fp8, tier 10/50, chunk 4095"
branch-commits: "vektory79: db8dffb (feat per-request route affinity), 12eb11d (kb verdict), e4c859b (kb index done), 8805b41 (memory sync) - не запушено"
status: "PHASES DONE - Ф0/Ф1/Ф2 выполнены, вердикт Ф1/Ф2 PASS (Ф2: лестница почти бесплатна при bs=[1]); дефолт широкой лестницы - решение владельца; коммиты созданы, не запушены"
tags: [route-affinity, per-request, cuda-graphs, ladder, decode, hybrid, verdict]
---

# Вердикт кейса route affinity на запрос

## Ф0: статус реализации

Полная цепочка реализована и закоммичена в vektory79 (db8dffb, не запушено), точки по файлам:

- CLI (`server/args.py`): флаг `--route-affinity-ladder` (comma-separated
  float); на буте лестница = дедуп `лестница + {server_default}`, значения >= 0.
- `engine/config.py`: `resolve_route_affinity_ladder` - единая точка
  эффективной лестницы (её едят и GraphRunner, и DecodeManager; гейт:
  не-offload/hybrid стратегия MoE или отсутствие `set_route_affinity` ->
  пустая лестница + boot-warning) + `batch_route_affinity` - резолв
  `None -> серверный дефолт`, снап к ближайшему значению лестницы
  (ties - меньшее), разнобой в батче -> 0.0 + однократный warn.
- `engine/engine.py`: инъекция лестницы в ModelConfig рядом с
  `route_affinity` + eager-хук: перед `model.forward()` вне графа
  печёт батчевое значение affinity.
- `engine/graph.py`: ключ графа `(padded_bs, affinity)`; захват по всей
  лестнице на каждый bs с повторным сбросом offload-кэша на каждом
  значении; `apply_affinity`: phase == "prefill" печёт дефолт,
  decode - батчевое значение.
- `models/glm5_next/model.py`: `set_route_affinity(value)` - сеттер,
  переприсваивает `route_affinity` MoE-слоям; движок зовёт через
  hasattr-проверку, другие модели не затронуты.
- scheduler: DecodeManager резолвит снап при сборке декод-батча.
- `core.py`: поля `Req.route_affinity` (None = серверный дефолт),
  `Batch.route_affinity`, `SamplingParams.route_affinity`.
- API: все три ручки (OpenAI / Anthropic / Responses) принимают
  `route_affinity` сырым значением по образцу `temperature`;
  `generation.py::resolve_sampling` прокидывает его в Req.
- docs: `docs/cli.md` - строка флага лестницы (отдельных API-доков
  полей запросов в docs/ нет - ограничение учтено ниже).

Ревью: 2 прохода. Блокер (prefill-контаминация: eager-декод оставлял
чужое значение в блоках навсегда, prefill роутил не по дефолту) закрыт
фазовой печатью в `apply_affinity` + регресс-тест
`test_eager_decode_then_prefill_restores_the_server_default`. Нит про
дублирующий гейт закрыт единой точкой `resolve_route_affinity_ladder`.
CPU-хвост: 563 passed (`test_args`, `test_graph`, 3 API-харнеса,
`tests/scheduler`); полный not-slow чист в GPU-зонах против
stash-бейзлайна.

## Ф1: приёмочные пробы per-request - PASS

Один бут с лестницей {0.0, 0.3}; fill 65.6k за 4.7 с (session-tier L2
restore), greedy decode 768 ignore_eos; `--max-running-requests 4` -
единственное отклонение от гейт-конфига, только чтобы F1c собрал
смешанный декод-батч. Одна секция захвата до READY,
`captures_after_ready == 0`. Все 9 бэндов из EXPECTED.md зелёные.

### F1a - серия одиночных route_affinity=0.3

| проба | steady tok/s | p50 мс | hit % |
|---|---|---|---|
| F1a_1 (декод #1) | 43.44 | 22.6 | 78.96 |
| F1a_2 | 44.61 | 22.1 | 80.03 |
| F1a_3 | 43.44 | 22.8 | 79.62 |
| гейт 0.3 (медиана, [../route-affinity/VERDICT.md](../route-affinity/VERDICT.md)) | 45.29 | 21.8 | 80.9 |

Декод #1 - единственный напрямую сравнимый с гейт-рукой (у гейта декод
один на бут) - в полосах [38,52] tok/s / [18,28] мс / [72,90] hit;
дрейфа по серии нет (повторы - та же 64k-проба, LRU-состояние
переносится).

### F1b - чередование 0.0 <-> 0.3 между запросами (6 запросов)

| группа | hit % | steady tok/s | p50 мс |
|---|---|---|---|
| 0.3 (3 запроса) | 77.2-80.1 | 43.5-44.2 | ~22.5 |
| 0.0 (3 запроса) | 32.4-33.7 | 21.1-21.6 | ~46 |
| гейт: 0.3 / база | 80.9 / 35.9 | 45.29 / 21.35 | 21.8 / 46.4 |

Разрыв групп по hit ~44 п.п. (полоса >= 25). `captures_after_ready = 0`:
переключение значения - это выбор графа на каждом декод-шаге, рекапчура
нет, качество не портится (веса unbiased в обоих графах).

### F1c - два одновременных запроса (0.3 и 0.0) -> fallback 0.0

В логе РОВНО ОДИН warn (warning_once); оба запроса досчитали 768 ток
без ошибок, по ~14.4 tok/s каждый (14.43 / 14.46; bs=2 - шаг вдвое
дороже одиночного). Hit окна 32.29% = уровень базы (полоса [25,45]):
батч декодирует граф 0.0, а не 0.3 - fallback сработал как задуман.

### Вердикт Ф1

**PASS.** Per-request семантика работает на bs=1: значение запроса
снапится к лестнице и выбирает заранее захваченный граф, чередование
мгновенное, смешанный батч честно падает в fallback 0.0.

## Ф2: цена лестницы (A/B бутов, bs=[1])

boot A - дефолт (без флага), boot B - `--route-affinity-ladder
0.0,0.05,0.3`; сериально, сервер снесён между ногами. Первичный
источник - строки лога `Free GPU memory before/after capturing CUDA
graphs`; nvidia-smi - кросс-чек.

| метрика | boot A (дефолт) | boot B (лестница x3) | дельта B-A |
|---|---|---|---|
| ready_s (реальный чат) | 70 | 64 | -6 с (шум) |
| дельта секции захвата (before - after) | ~41 MiB | ~51 MiB | +10 MiB на 2 доп. графа ~ 5 MiB/граф |
| nvidia-smi free at ready | 3560 MiB | 3546 MiB | -14 MiB |
| nvidia-smi free после teardown+20s | 30786 | 30786 | teardown чистый, утечки нет |

Вывод: цена лестницы при bs=[1] - ~10 MiB VRAM и ~0 с бута (в пределах
шума). Оценка владельца "единицы-десятки MiB на граф" подтвердилась на
нижней границе (гранулярность лога 0.01 GiB). Цена растёт линейно как
|bs| x |лестница| захватов. Дефолтить лестницу шире одного значения
можно; решение за владельцем.

## Ограничения

1. Смешанный декод-батч (разные значения после снапа) -> fallback 0.0 +
   однократный warn. Честный per-request в смешанном батче одним
   форвардом невозможен: бонус применяется к fused-topk на слой и батч,
   резидент-маска общая. Разрезание декод-батча по группам affinity
   вне скоупа кейса.
2. Спекулятивный декод (spec/mtp): verify-графы захватываются другим
   кодом и hook не получают - там affinity всегда 0.0. В прод-конфиге
   GGUF hybrid спека отключена.
3. Лестница игнорируется (пустая + boot-warning) на не-offload/hybrid
   стратегиях MoE и у моделей без `set_route_affinity`; гейт сидит в
   единой точке `resolve_route_affinity_ladder` - иначе был бы
   молчаливый eager-регресс для запросов с полем.
4. Prefill инертен: affinity влияет только на декод; `apply_affinity`
   на prefill печатает серверный дефолт (закрытый блокер ревью).
5. Поле запроса задокументировано только в `docs/cli.md` - отдельных
   API-доков полей запросов в docs/ нет.

## Артефакты

- [TASK.md](TASK.md) - бриф кейса: спека правок по файлам,
  owner-решения, чеклист фаз.
- `.tasks/route-affinity-request-f1f2/` (live, git-ignored):
  RUNBOOK.md (порядок ног), EXPECTED.md (полосы допуска, утверждены до
  измерений), ARTEFACTS.md (F2-таблица + ServerArgs обоих бутов),
  logs/f1_result.json (сырьё Ф1, все бэнды зелёные).
- [../route-affinity/VERDICT.md](../route-affinity/VERDICT.md) - гейт
  значений, опорные числа: 0.3 -> 45.29 tok/s / 80.9% hit, база ->
  21.35 tok/s / 35.9%.
