# quality/ — 24-промптовая батарея качеств и разбор дивергенции

Карточки использования. Методология: [../../methods/quality-ab.md](../../methods/quality-ab.md),
[../../methods/arbitration-battery.md](../../methods/arbitration-battery.md).
Числа и итоги: [../../baselines/mmq-prefill-kernel/](../../baselines/mmq-prefill-kernel/).

### run_battery.py
[run_battery.py](run_battery.py)
- Назначение: stage-драйвер quality-battery (v2 MMQ A/B): census -> boot `ft serve` с FREETOKEN_GGUF_GROUPED_PREFILL=0/1 -> 24-промптовая greedy-батарея (reasoning+content verbatim, p00..p23.json) -> SIGTERM 30 s -> SIGKILL teardown -> census.
- Запуск: `run_battery.py <env0|env1>`; порт 18801, serial; env стадии включает `PYTHONPATH=<probe_sitecustomize>` для видимости bare-logger-маркера; коды выхода 0/2/3/4/5.
- Готчи / требования: боевой набор флагов `ft serve` и путь к модели захардкожены; FAILPAT-лист `AssertionError|OutOfMemoryError|Backend worker is gone`; серийность обязательна — перед boot чужой `ft serve` убивается (килл-лист только ft serve).

### analyze_divergence.py
[analyze_divergence.py](analyze_divergence.py)
- Назначение: анализатор дивергенции env0 vs env1 + исторический контекст (battery-post-fix GGUF, t05/t08 hybrid): первая дивергенция по символам (common-prefix), delta длин, flag task-flip (div < 5), digit-проверки p20 (80 km/h), p21 (Friday), p22 (6 apples), p23 (1024); пишет `divergence.json`.
- Запуск: `analyze_divergence.py` (без аргументов); ожидает `env0/` и `env1/` с p00..p23.json рядом, контекстные стадии — по захардкоженным путям.
- Готчи / требования: сравнение идёт по конкатенации reasoning+content в unicode-символах; контекстные стадии обязательны (скрипт падает без них); markdown-таблица печатает 110-символьные головы для ручного чтения.

### ft_phase6_battery.py
[ft_phase6_battery.py](ft_phase6_battery.py)
- Назначение: replay 24 промптов NVFP4-baseline на живом сервере (post-fix GGUF-дерево): greedy temp0/topk1, non-stream, артефакты в том же schema pXX.json, что у baseline.
- Запуск: `ft_phase6_battery.py --port 18081 --baseline-dir <каталог baseline> --outdir <куда писать>`; сервер должен уже быть поднят отдельно.
- Готчи / требования: промпт и max_tokens берутся из baseline-артефактов, ответы сравнимы 1:1; HTTP-ошибки считаются failed и дают rc=1; schema артефактов совместима с [analyze_divergence.py](analyze_divergence.py).