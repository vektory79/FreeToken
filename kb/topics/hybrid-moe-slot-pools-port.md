---
title: "Порт D3: гибридный MoE на per-format слот-пулах ggml - три гейта конфига"
date: 2026-10-21
hardware: "RTX 5090 32GB (boot-smoke x2, A/B decode, Fix-3 e2e); CPU-волны (pytest)"
branch-commits: "vektory79 393b682 (порт D3; поверх merge f0eed55 и D4 9314c9b)"
status: validated
tags: [gguf, ggml, hybrid, moe, slot-pools, deferred-ports, offload, cache-budget, fix3]
---

# Порт D3: гибридный MoE на слот-пулах - три гейта конфига

**Вердикт (читайте это первым).** Порт D3 закрыт 2026-10-21 коммитом 393b682,
и это самый дешёвый порт кампании round-5: глубокая адаптация гибрида/offload
к per-format слот-пулам не понадобилась - main УЖЕ вёз гибрид на новом стеке.
Вернулись только три fail-fast гейта конфига; T2-хелперы признаны
неактуальными. Приёмка (RTX 5090, unsloth GLM-5.3-Flash-GGUF UD-Q3_K_XL):
786k гибридный рецепт бут x2 без лестницы (ready 119/135 с, 3 пула 29-1103
слотов, гейты молчат); decode гибрид 27.71 tok/s медиана против offload
21.00 на том же стеке (+32%) и против старого стека 16.67 (+66%); cap-0
CPU-лег ~49.6 GB/s (~1.8x старого стека); Fix-3 e2e 5568 cached-токенов /
0 полных промахов. Карта порт-долга -
[round5-ggml-replatform-deferred-ports.md](round5-ggml-replatform-deferred-ports.md);
план и приёмочные критерии фазы -
[../cases/round5-deferred-ports/TASK.md](../cases/round5-deferred-ports/TASK.md).

## Состав порта (коммит 393b682, 8 файлов, 737+/42-)

- Три гейта в [engine/engine.py](../../python/freetoken/engine/engine.py):
  `_cpu_moe_format_rejection` - capability-screen явных cpu/hybrid-выборов
  по форматам слоёв (хук в `_init_offload_moe_cache`, отказ называет
  offending-форматы до загрузки банков); `_gguf_hybrid_rejection` -
  header-скан .gguf в `_adjust_config`, явный hybrid требует доказанной
  серверабельности каждого типа банков, непроверяемый файл отклоняется,
  offload остаётся безопасным фолбэком; `_gguf_auto_floor_gate` -
  pre-load проверка floor-ов пулов `check_partition_floors` для
  мультисигнатурного gguf, отказ за ~секунды вместо минутной загрузки.
- `cpu_spec_for_types` в
  [layers/quantization/moe/gguf.py](../../python/freetoken/layers/quantization/moe/gguf.py) -
  спека CPU-executor по типам (gate_up, down), общая для загруженных слоёв
  и header-скана, чтобы формат файла и слой не могли разойтись.
- Тесты: [tests/engine/test_cache_budget.py](../../tests/engine/test_cache_budget.py),
  [tests/moe/test_offload.py](../../tests/moe/test_offload.py),
  [tests/scheduler/test_hybrid_cache_manager.py](../../tests/scheduler/test_hybrid_cache_manager.py),
  новый g.3 - [tests/moe/test_mixed_expert_formats.py](../../tests/moe/test_mixed_expert_formats.py);
  семантика гейтов задокументирована в
  [docs/cli.md](../../docs/cli.md) и [docs/models.md](../../docs/models.md).

## Урок 1. Сначала диспозиция, потом порт: main УЖЕ имел гибрид

Спека фазы (волна 8 кампании) исходила из списка удалённых в merge хвостов
engine.py - partition gate, gguf hybrid rejection, T2 helpers, plural-cache
state - и планировала "глубокую адаптацию". Исследование показало: гибрид
на новом стеке жив (`_decode_hybrid`, `ensure_experts_hybrid`, per-format
пулы внутри OffloadMoeCache). Реально вернулись 3 гейта; T2 и plural-cache
- неактуальны. Урок для остальных портов кампании: список удалённых
символов - это НЕ список потерянной функциональности; сначала ответить,
что из этого main уже реализовал иначе, потом планировать объём порта.

## Урок 2. T2 не возвращать: однопульный executor убил пер-пульную грамматику

T2-хост-доставка (кейс
[decode-t2-host-delivery](../cases/decode-t2-host-delivery/TASK.md)) была
пер-пульной: свой пул потоков на пул слотов. Новый CPU-executor main'а
однопульный - глобальный бюджет потоков, пулы делят его динамически,
поэтому пер-пульная грамматика (FREETOKEN_T2_POOL_THREADS и хелперы)
теряет смысл. Ручка на пул CPU-ядер имеет смысл только при доказанном
голоде CPU-пула; голод не доказан: cap-0 дал ~49.6 GB/s CPU-лег (~1.8x
старого стека на том же чекпойнте). Возвращать - только после замера,
показавшего голод.

## Урок 3. Спек-оценка слотов не сошлась: доверять resolve, не оценкам

Спека пересчитала 786k-рецепт аналитически: ~1470 слотов
((21.0 - 5.0) / 0.0109, ~4.4 шага глубины на 336 пар) и ноль headroom.
Факт при буте: "MoE slot pools: unified, 3 pools of 29-1103 slots (equal)"
- суммарно 1189 слотов: футпринт весов выше спек-допущения, но рецепту
хватает, и floor-гейт молчит. Урок: арифметика спеки - прикидка порядка,
источник истины - resolve на реальном чекпойнте; рецепт проверять бутом,
а не уравнением.

## Урок 4. Старые проценты не переносятся: +32% гибрида - другая механика

Веточные проценты старого csrc-стека (+67.2% m-tile, +16.4% grouped MMQ)
перекрыты вендоренным ggml (карта D1), и сравнивать их с новым числом
нельзя: +32% гибрида относительно offload того же стека получены из совсем
другой механики (overlap fetch по PCIe с CPU-легом на ggml-cpu), а не из
настройки тех же ядер. Каждое число живёт только со своим конфигом:
27.71 tok/s - медиана декода, UD-Q3_K_XL, kv-reserve 786k, nvfp4,
memory-ratio 0.85, disable-overlap, moe-cpu-threads 16, RTX 5090;
21.00 tok/s - тот же стек и чекпойнт, offload.

## Урок 5. Нет benchbw-профиля для gguf - fetch-fraction auto недоступен

Автоподъём offload -> hybrid и авто-доля fetch требуют кэшированного
профиля `ft bench bw`; для .gguf-чекпойнтов профиля нет, поэтому кап
фиксируется (1). Кандидат на отдельную мелкую задачу: научить benchbw
снимать профиль gguf-модели, тогда fetch-fraction auto оживёт и для
гибридных gguf-рецептов.

## Урок 6. Честная бухгалтерия g-тестов: g.3 пинит контракт, g.2 - D1-deferred

g.3 ([tests/moe/test_mixed_expert_formats.py](../../tests/moe/test_mixed_expert_formats.py))
пинит контракт -1-лэйна смешанных пулов: 132 passed / 1 deselected.
g.2 (test_cpu_moe_gguf_iq.py, iq-линейка) отложен честно: модуль сидит на
удалённом models/gguf/dequant - это D1-deferred, а не долг D3; skip-маркер
называет настоящий порт (дисциплина skip-маркеров раунда 5, см.
[merge-wave-procedure.md](merge-wave-procedure.md)).

## Приёмка

- Рецепт бута x2 (порт 18081): `ft serve --moe-strategy hybrid
  --moe-cache-auto --kv-reserve-tokens 786432 --kv-cache-dtype nvfp4
  --memory-ratio 0.85 --max-prefill-length 4096 --max-running-requests 1
  --moe-cpu-threads 16 --disable-moe-prefill-overlap`, модель unsloth
  GLM-5.3-Flash-GGUF UD-Q3_K_XL. Ready 119/135 с; 3 пула 29-1103 слотов;
  все три гейта молчат на влезающем конфиге.
- Decode A/B (тот же стек, тот же чекпойнт): гибрид 27.71 tok/s медиана,
  offload 21.00 tok/s (+32%). Ориентир старого стека: гибрид 16.67 tok/s
  (эталон 63b9bff) - формально +66%, сравнение не apples-to-apples.
- cap-0 (moe_cache_size=0, чистый CPU-лег): ~49.6 GB/s, ~1.8x старого
  стека; голод CPU-пула не подтверждён (см. урок 2).
- Fix-3 e2e: префилл 4096 + повторный ход - 5568 cached-токенов, 0 полных
  промахов: донесение кэша пережило переезд на пулы без порта (паттерн
  Fix-3 - [../cases/fix3-snapshot-lru-refresh/TASK.md](../cases/fix3-snapshot-lru-refresh/TASK.md)).
- Тесты: g.3 132 passed / 1 deselected; g.2 skipped (D1-deferred); батарея
  CPU-волн кампании - 2059 passed / 831 skipped при 2 failed известной
  базлайн-паре, сравнение по FAILURE SET.

## Открытое

- benchbw-профиль для gguf (fetch-fraction auto) - мелкая отдельная задача.
- g.2 (iq-линейка CPU-MoE) - D1-deferred, не долг D3.
- D1 микробенчи (mhc SPLIT 128vs64 + quality A/B, N-split A/B, базовый
  probe) - фаза 3 кампании, RUN/QUEUED.
- Экономика гибрида, под которую всё затачивалось -
  [moe-hybrid-cost-model.md](moe-hybrid-cost-model.md); соседний порт D4 -
  [glm5next-gguf-loader-port.md](glm5next-gguf-loader-port.md).
