# mmq/ — A/B-машинерия префилл-ядер MoE (MMQ)

Карточки использования. Методология: [../../methods/step0-profile-mmq.md](../../methods/step0-profile-mmq.md),
[../../methods/v0-ab.md](../../methods/v0-ab.md), [../../methods/v2-ab.md](../../methods/v2-ab.md),
[../../methods/v3a-ab.md](../../methods/v3a-ab.md), [../../methods/v3a-surface-map.md](../../methods/v3a-surface-map.md).
Числа и итоги: [../../baselines/mmq-prefill-kernel/](../../baselines/mmq-prefill-kernel/), [../../baselines/mmq-v3/](../../baselines/mmq-v3/).

### step0_run.py
[step0_run.py](step0_run.py)
- Назначение: драйвер step-0 — один boot `ft serve` (winner-флаги) под nsys-сессией `step0cap` с отложенным захватом, окно = полные префилл-чанки (старт внутри чанка 3, стоп внутри чанка 7).
- Запуск: `step0_run.py` (без аргументов; `--extra` и сессия зашиты); переиспользует measure.py, подменяя `FT` на nsys-обёртку `ft_nsys.sh`; пишет `step0_run.json` и `<name>.nsys-rep`.
- Готчи / требования: nsys обязателен; забытая подмена обёртки = plain boot и молчаливо сорванная коллекция; watchdog FAILPAT включает `backend worker .* exited`. Методология: [../../methods/step0-profile-mmq.md](../../methods/step0-profile-mmq.md).

### step0_analyze.py
[step0_analyze.py](step0_analyze.py)
- Назначение: разбор step-0 — сплит GEMM/копии/остальное из nsys sqlite + серверного лога, нормализация на chunks_eq, расчёт implied v0/v2 потолков (одиночный проход по expert-весам).
- Запуск: `step0_analyze.py` (без аргументов); сам находит новейший `report*.nsys-rep`, при отсутствии экспортирует sqlite через `nsys export`.
- Готчи / требования: T (steady-медиана) берётся из `step0_run.json`; MoE-ядро распознаётся по подстроке `moe` — dense `ggml_mul_mat_a8` остаётся в «rest» намеренно.

### v0ab_probe.py
[v0ab_probe.py](v0ab_probe.py)
- Назначение: liveness-проба v0 — boot с root-лог-хендлером, инжектируемым через `PYTHONPATH`, один мелкий префилл, grep маркера `expert-sorted pair order active` в логе.
- Запуск: `v0ab_probe.py` (без аргументов; переиспользует measure.py; env: `PYTHONPATH=<probe_sitecustomize>`).
- Готчи / требования: сам маркер логируется bare stdlib logger'ом без хендлеров — без sitecustomize (см. [sitecustomize.py](sitecustomize.py)) INFO-записи падают в logging.lastResort и невидимы; production-код не трогается.

### v0ab_analyze.py
[v0ab_analyze.py](v0ab_analyze.py)
- Назначение: разбор v0 A/B stage-выводов measure.py: per-stage метрики (prefill-медиана с исключением c1 и последнего полного чанка, decode, boot, JIT/liveness-следы) + финальный comparison-блок.
- Запуск: `v0ab_analyze.py <имя_стадии> <имя_лога> [<имя_стадии2> <имя_лога2> ...]` — пары читаются из `.out`-файлов measure.py.
- Готчи / требования: парсер находит JSON по якорю `'{\n "name"'` — формат вывода measure.py менять нельзя; liveness-флаг ищет строку `expert-sorted pair order active` в сыром логе. Методология: [../../methods/v0-ab.md](../../methods/v0-ab.md).

### v2ab_probe.py
[v2ab_probe.py](v2ab_probe.py)
- Назначение: liveness-проба v2 (попытка 2) — capture-from-launch nsys-сессия (`--cuda-graph-trace=node`), boot с `FREETOKEN_GGUF_GROUPED_PREFILL=1`, 65,585-токеновый префилл + radix-hit decode, потом stop.
- Запуск: `v2ab_probe.py` (без аргументов; env: `FREETOKEN_GGUF_GROUPED_PREFILL=1`, `PYTHONPATH=<probe_sitecustomize>`); перед стартом подчищает stale-сессию.
- Готчи / требования: попытка 1 стартовала mid-prefill и eager-активности не появились — коллекция обязана стартовать ДО первого префилл-ядра; маркер liveness — строка `grouped mmq prefill active`. Методология: [../../methods/v2-ab.md](../../methods/v2-ab.md).

### v2ab_probe_analyze2.py
[v2ab_probe_analyze2.py](v2ab_probe_analyze2.py)
- Назначение: liveness-ассерты v2 по kernel-таймлайну: фазовый сплит по GPU-idle gap > 1.5 s (boot -> prefill -> decode) и P1-P4 (grouped MMQ в префилле, moe_vec* в префилле = 0, moe_vec* в decode > 0, moe_align присутствует).
- Запуск: `v2ab_probe_analyze2.py` (без аргументов); сам экспортирует новейший `report*.nsys-rep` в sqlite; exit 3 при провале ассертов.
- Готчи / требования: фазовая идентификация эвристична (boot = первая фаза, prefill = максимум grouped, decode = последняя) — при изменении сценария проба пересматривается.

### v2ab_stage.sh
[v2ab_stage.sh](v2ab_stage.sh)
- Назначение: stage-раннер v2 A/B: census до/после, одна сериализованная measure.py-прогонка на стадию.
- Запуск: `v2ab_stage.sh <name> <log> <prefill|full> "<extra-flags>" "<ENV=V>"`; serial only.
- Готчи / требования: measure.py отказывается стартовать при живом `ft serve`; timeout 1800 s; выход measure.py целиком пишется в `<name>.out` — его потом разбирает [v2ab_analyze.py](v2ab_analyze.py).

### v2ab_analyze.py
[v2ab_analyze.py](v2ab_analyze.py)
- Назначение: разбор v2 A/B по трём точкам чанкинга (4096/6144/8128): медиана полных чанков без c1 и последнего полного чанка, decode-сравнение, boot-тайминги; пишет `v2-ab-results.json`.
- Запуск: `v2ab_analyze.py` (без аргументов); ожидает `v2ab_{before,after}_{4096,6144,8191}.out` рядом со скриптом.
- Готчи / требования: суффикс `8191` именует 8128-чанковые прогоны; campaign-reference константы (262/281/292.30 tok/s) зашиты для сверки. Методология: [../../methods/v2-ab.md](../../methods/v2-ab.md).

### v3a_stage.sh
[v3a_stage.sh](v3a_stage.sh)
- Назначение: stage-раннер v3a: census -> одна сериализованная `measure.py full` -> census; trap-on-EXIT cleanup + shell-watchdog с FAILPAT и SIGKILL по pidfile measure.py.
- Запуск: `v3a_stage.sh <name> <log> "<ENV=V>"` (пустой третий аргумент = baseline boot); timeout 2400 s.
- Готчи / требования: watchdog подхватывает pid сервера через pidfile measure.py — лог-файл должен совпадать; FAILPAT-лист включает OOM/AssertionError/Backend worker/`backend worker .* exited`.

### run_battery_v3a.py
[run_battery_v3a.py](run_battery_v3a.py)
- Назначение: v3a quality-battery stage-драйвер: census -> boot с FREETOKEN_GGUF_MOE_MTILE стадии -> 24-промптовая батарея -> SIGTERM/SIGKILL teardown -> census.
- Запуск: `run_battery_v3a.py <base|win> [tile]` (base = no env, tile 4; win = FREETOKEN_GGUF_MOE_MTILE=<tile>); коды выхода 0/2/3/4/5.
- Готчи / требования: это кампаний-специфичный вариант [quality/run_battery.py](../quality/run_battery.py) (env-flip там — GROUPED_PREFILL, здесь — MTILE); объединение в один обобщённый драйвер — возможная будущая работа; PYTHONPATH-проба sitecustomize ставится для видимости bare-logger-маркера.

### v3a_battery_analyze.py
[v3a_battery_analyze.py](v3a_battery_analyze.py)
- Назначение: разбор дивергенции v3a-батареи base vs win: pair_stats (первая дивергенция по символам, delta длин, identical) + digit-проверки p20-p23; пишет `v3a_divergence.json`, печатает markdown-таблицу.
- Запуск: `v3a_battery_analyze.py` (без аргументов); читает `<stage>/p00..p23.json` двух стадий.
- Готчи / требования: кампаний-специфичный вариант [quality/analyze_divergence.py](../quality/analyze_divergence.py) (та же методология pair_stats/DIGIT, указатели на v3a-директории); возможное будущое объединение с качественным анализатором.

### v3a_probe_analyze.py
[v3a_probe_analyze.py](v3a_probe_analyze.py)
- Назначение: liveness + per-format разбор v3a-пробы: tile-16 захват против tile-4 baseline (v2probe2), ассерты A1-A5 (grouped-ядра по формату, moe_vec в префилле = 0, moe_vec в decode, launch-count math 126/42 на chunk-equiv, grid.y shrink ~tile).
- Запуск: `v3a_probe_analyze.py` (без аргументов); экспортирует sqlite из новейшего `report*.nsys-rep`; exit 3 при провале ассертов.
- Готчи / требования: baseline-sqlite и пути v3a-директории захардкожены; 9 chunk-equivalents = 8 полных чанков + 561-хвост — пересчитывать при другом чанкинге. Методология: [../../methods/v3a-ab.md](../../methods/v3a-ab.md), [../../methods/v3a-surface-map.md](../../methods/v3a-surface-map.md).

### sitecustomize.py
[sitecustomize.py](sitecustomize.py)
- Назначение: проба видимости bare-logger'а: `logging.basicConfig(level=INFO)` на импорте — добавляет root-handler, чтобы one-shot маркеры, логируемые bare stdlib logger'ом в production-коде (ни один серверный хендлер их не покрывает), перестали попадать в logging.lastResort и отбрасываться.
- Запуск: не запускается — кладётся в отдельный probe-каталог и подключается к boot через `PYTHONPATH=<dir с этим файлом>`; Python сам импортирует sitecustomize при старте процесса.
- Готчи / требования: использовать только harness-side, production-код не трогается; без него liveness-пробы ([v0ab_probe.py](v0ab_probe.py)) не увидят маркер — симулированная «мертвая» фича. Диагностический приём описан в [../../README.md](../../README.md).
