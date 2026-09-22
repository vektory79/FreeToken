---
title: "Интерлив-промывка: как чередование диалогов вымывает GDN-снапшоты и что закрыто флагом"
date: 2026-09-22
hardware: RTX 5090 32GB
branch-commits: "vektory79; fix-1 5b72aba + fix-3 9732be0/e5730e0/7080824 присутствуют; тест-пины d82274c"
status: validated
tags: [interleave, snapshot-eviction, evict_mamba, snapshot_lru, hybrid, cached-token, linear-state-cache-ratio]
---

# Интерлив-промывка кеша сессий (оркестратор <-> субагент)

**Вердикт.** При чередовании двух разных диалогов на гибридной модели каждый ход
полностью теряет кеш: суммарный размер контекстов роли не играет, ограничивающая
валюта - пул GDN-снапшотов, а не KV-окно. Железно подтверждено 2026-09-22:
при дефолтном `linear_state_cache_ratio=2.0` - 11/11 ходов с `#cached-token: 0`;
при `--linear-state-cache-ratio 8 --kv-reserve-tokens 350000` - 9/9 ходов полными
попаданиями (3.6-3.9 с на ход против 57-75 с, ~20x). Флагового фикса достаточно
для двух агентов ~44-60k контекстов каждый.

## Механизм (якоря в коде)

- Пул снапшотов: `4*mr + max(4, ceil(ratio*mr)) + 1` слотов
  (`python/freetoken/kvcache/linear_state_pool.py`, `_linear_pool_num_slots`).
  При mr=1 и ratio 2.0 это 9 слотов (~140.76 MiB/слот), из них ~4-5 вытесняемых;
  ratio 8 даёт пул 13. Гейдж `#mamba-slot` исключает padding-слот: 4/8 (пул 9)
  и 0/12 (пул 13) в логах прогонов сходятся с этой формулой.
- Донат снапшота на каждом префилл-чанке (per-chunk donation, см.
  [radix-cache-reuse.md](radix-cache-reuse.md)) создаёт ~8-9 boundary-узлов
  на ход при чанке 8128 и промпте ~57k. Каждый коммит, которому нужен слот,
  вызывает `ensure_mamba_slots` -> `evict_mamba`
  (`python/freetoken/scheduler/cache.py`), а тот упорядочен FIFO по
  `snapshot_lru` (`python/freetoken/kvcache/hybrid_radix_cache.py`) - без
  понятия "недавно завершившийся диалог".
- Весь след чужого диалога строго старше любых границ текущего хода, поэтому
  вымывается первым, tip включительно. Вытеснение листа каскадно удаляет и
  tombstone-предков с их KV - чужой диалог исчезает из дерева целиком.
  `match_prefix` матчится только до живого снапшота -> `cached=0` ->
  полный re-prefill.
- Внутри одного диалога tip выживает собственную промывку (омолаживание
  стампа на матче), поэтому wave-13 fix-3 видел HIT на повторе 65k подряд.
  Промывка начинается, как только между ходами вклинивается чужой диалог.

## Железо: две руки реплея (2026-09-22)

Драйвер: два детерминированных раздельных филлер-диалога (A ~57345 токенов,
B ~44145), чередование 5 ходов каждый + финальный ход A; вердикт - сумма
`#cached-token` по строкам `Prefill batch` серверного лога. Пакет:
`.tasks/interleave-cache-repro/` (run_arm.sh, replay_interleave.py,
arm_arm1/arm_arm2 .json+.log; не закоммичен, числа дистиллированы сюда).

- Arm 1: конфиг пользователя `--port 18801 --moe-cache-auto
  --kv-reserve-tokens 400000 --kv-cache-dtype fp8 --memory-ratio 0.82
  --max-prefill-length 8191 --moe-strategy hybrid --moe-cpu-threads 16
  --max-running-requests 1` (ratio дефолтный 2.0). Ходы: 11/11 полный промах;
  A = 57345 new ~75 с, B = 44145 new ~57 с. `wash_confirmed=true`.
- Arm 2: те же флаги, но `--kv-reserve-tokens 350000
  --linear-state-cache-ratio 8`. Ходы 1-2 холодные, ходы 3-11 - полные
  попадания: A 57344/57345, B 44096/44145 (хвост 49 токенов до границы
  страницы), 3.6-3.9 с/ход. Бут без fail-fast; free-after-init 4.86 GiB
  (Arm 1 @ 0.82/400k); KV-пул: `400000 tokens = 2.38 GiB` -> ~6.39 KiB/токен,
  KV-путь сессии 57k ~ 366 MiB.
- CPU-пины: `tests/kvcache/radix/test_hybrid_radix.py` (d82274c) -
  `test_interleaved_conversations_wash_out_the_older_trace` пиннит текущую
  политику как базовую линию (жертвы FIFO A0->A1(tip)->B0, матч A = 0, KV A
  вычищен каскадом), контрастный
  `test_interleave_with_enough_snapshot_pool_keeps_both_traces` доказывает,
  что промывка - эффект размера пула, не дефект match/insert.

## Почему флага хватает и когда перестанет

ОБНОВЛЕНИЕ 2026-09-22: `ratio 8` ИЗМЕРЕН, но пользователем НЕ принят - дорог по
VRAM. Эксплуатационные ограничения: сессии часто 150-200k, агентов может быть
больше двух, параллельные сессии; VRAM-пулы остаются тесными (дефолтный
ratio 2.0). Поэтому флаговый путь снят с повестки, основной механизм - ярусный
кеш сессий RAM/SSD ([session-cache-tiering.md](session-cache-tiering.md)).
Технический критерий ниже сохранён как справка.

Держать нужно только tip чужого диалога (промежуточные grid-точки нужны для
переписанных историй - см. дивергенцию в [radix-cache-reuse.md](radix-cache-reuse.md)).
Условие выживания: вытесняемых слотов >= коммитов чужого хода + 1 tip.
При ratio 8 (8-9 вытесняемых) это выполняется для двух диалогов ~60k.
Перестаёт выполняться при 3+ параллельных сессиях, контекстах заметно за
~100k (141 MiB/снапшот не помещается в VRAM пачками) или при желании
сохранять кеш между рестартами сервера.

## Как измерить / проверить

- Реплей: `bash .tasks/interleave-cache-repro/run_arm.sh <arm> [флаги ft serve]`,
  затем `replay_interleave.py` печатает таблицу ходов и JSON-вердикт
  `wash_confirmed`. Serial only; census `'[f]t serve --port'`.
- Готовность сервера: `/v1/models` отвечает ДО конца загрузки GGUF (наблюдалось:
  chat 503 mid-load) - проб готовности должен быть tiny chat POST до 200,
  он же прогревает triton autotune.
- Юнит: `uv run pytest tests/kvcache/radix/test_hybrid_radix.py -q` (CPU).
- Смежное: переиспользование и fix-1/fix-3 - [radix-cache-reuse.md](radix-cache-reuse.md);
  VRAM-торг при подъеме ratio - [vram-budget.md](vram-budget.md) и
  [vram-headroom.md](../methods/vram-headroom.md).