# coverage-curve/ - харнесс офлайн-кривой "слоты -> покрытие" (гейт ёмкостных рычагов)

Байт-в-байт копия скриптов кейса
[../../cases/offline-coverage-curve/](../../cases/offline-coverage-curve/)
(TASK.md, PHASE0.md, VERDICT.md, results_trace5.*). Живой рабочий каталог -
`.tasks/offline-coverage-curve/` (git-ignored: routes/stats/arm json, логи).
Вердикт кампании (2026-11-15): KV-стриминг NO-GO (+300 слотов = +3.85 п.п.
C(S) < +5 порога; потолок всех ёмкостных рычагов 82% покрытия).

## Состав

- [run_trace.py](run_trace.py) - Ф1: один boot точного конфига пользователя
  (GGUF GLM-5.3-Flash-UD-Q3_K_XL, FTW-vision-new, 0.81 / kv-reserve 800k fp8,
  tier 10/50, chunk 4095, порт 18801) в eager (`--cuda-graph-max-bs 0`),
  fill 64k + radix-reuse + greedy-декод 768 с `ignore_eos`; через
  FREETOKEN_DR_ROUTES=1 T1-драйвера пишет routes_<tag>.json (770 шагов x 42
  слоя маршрутов + мета кеша: pool_sizes, slot_bytes, фракции fetch) и
  stats_<tag>.json (телеметрия missing/fetched/active для валидации);
- [sim_coverage.py](sim_coverage.py) - Ф2/Ф3: numpy-реплика
  `_ensure_experts_hybrid_kernel` (timestamp-LRU, BY_RECENCY, min-wait,
  кап fetch, overflow -> CPU), валидация при S0 против телеметрии, sweep
  {S0, +100..+600, 2*S0, все пары}, таблица C(S) + cpu_gib_per_step(S) +
  вердикт по гейту; CPU-only, без GPU;
- [test_sim_synth.py](test_sim_synth.py) - оффлайн-проверки симулятора:
  аналитические кейсы политики, кросс-чек split_equal против
  `pool_policy.split_slots`, end-to-end plumbing.

## Запуск

```bash
# Ф1 (GPU, ~5-8 мин): boot + трейс
python .tasks/offline-coverage-curve/run_trace.py --tag traceN
# Ф2/Ф3 (CPU, секунды): валидация + sweep + вердикт
python .tasks/offline-coverage-curve/sim_coverage.py \
    --routes routes_traceN.json --stats stats_traceN.json \
    --out results_traceN.md --csv results_traceN.csv
# проверки симулятора
python .tasks/offline-coverage-curve/test_sim_synth.py
```

Готчи: скрипты несут машинно-локальные абсолютные пути (REPO, чекпоинт
FTW-vision-new, filler-промпт, порт 18801) - на другой машине поправить;
копии в kb/ не редактируются - правится оригинал в .tasks и зеркалится сюда;
greedy-декод без `ignore_eos` обрывается по EOS (наблюдалось 545 из 768);
eager обязателен (per-step маршруты существуют только вне CUDA-графов);
ready-пробы оставляют 1-2 шумовых шага в трейсе - реплеются симулятором
согласованно с телеметрией.

Зависимости Ф1: T1-драйвер с режимом routes -
[../decode-research/driver/sitecustomize.py](../decode-research/driver/sitecustomize.py)
(загружается через PYTHONPATH; обычные boot'ы не затронуты).
