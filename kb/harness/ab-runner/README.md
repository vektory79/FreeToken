# ab-runner/ — A/B-машинерия dense q8_0 GEMM

Карточки использования. Методология: [../../methods/ab-mtile.md](../../methods/ab-mtile.md),
[../../methods/ab-nsplit.md](../../methods/ab-nsplit.md), [../../methods/quality-ab.md](../../methods/quality-ab.md),
[../../methods/step0-profile.md](../../methods/step0-profile.md), [../../methods/vram-headroom.md](../../methods/vram-headroom.md).
Числа и итоги кампании: [../../baselines/dense-q80-gemm/](../../baselines/dense-q80-gemm/).

### ab_mtile_run.py
[ab_mtile_run.py](ab_mtile_run.py)
- Назначение: драйвер m-tile свипа dense q8_0 — один boot на значение `FREETOKEN_GGUF_DENSE_MTILE`, якорь `FREETOKEN_GGUF_MOE_MTILE=32` на каждом boot, `FREETOKEN_GGUF_KDA_NSPLIT` никогда не ставится (committed default 2).
- Запуск: `ab_mtile_run.py --name mtile4|mtile8|mtile16|mtile32|mtile64|mtile4b` (mtile4 = knob unset, mtile4b = явный `=4`). Требует nsys, запущенный сервер живёт под обёрткой [ft_nsys_nsplit.sh](ft_nsys_nsplit.sh); окно захвата nsys = чанки 3..7 филла (старт после 2-й, стоп после 6-й строки `input throughput`).
- Готчи / требования: nsys в `/usr/local/bin/nsys`; забытая подмена `measure.FT` на обёртку даёт plain boot — start/stop молча не работают (см. [../../methods/step0-profile.md](../../methods/step0-profile.md)); порт 18801, serial; результат — `<name>.nsys-rep` + `<name>_run.json` в рабочем каталоге кампании.

### ab_nsplit_run.py
[ab_nsplit_run.py](ab_nsplit_run.py)
- Назначение: драйвер N-split A/B — один boot на значение `FREETOKEN_GGUF_KDA_NSPLIT` (nsplit1 = unset -> 1, nsplit2 = `2`), якорь `FREETOKEN_GGUF_MOE_MTILE=32` на обоих boot.
- Запуск: `ab_nsplit_run.py --name nsplit1|nsplit2`; сценарий boot'а идентичен [ab_mtile_run.py](ab_mtile_run.py) (fill -> decode re-send с ожидаемым radix HIT -> 6 greedy-промптов -> teardown).
- Готчи / требования: те же, что у mtile-драйвера; env-pins фактического сервера пинятся через `/proc/<pid>/environ` — проверять `server_env_pins` в отчёте перед разбором.

### ab_mtile_analyze.py
[ab_mtile_analyze.py](ab_mtile_analyze.py)
- Назначение: анализатор m-tile свипа — читает `mtile{4,8,16,32,64}.sqlite` (nsys sqlite-экспорты steady-окна) + одноимённые `_run.json` и пишет `ab-mtile-results.json` с таблицей свипа и побитовым diff'ом промптов.
- Запуск: `ab_mtile_analyze.py` (без аргументов); детерминированный, CPU-only, sqlite открывается read-only.
- Готчи / требования: sqlite и `_run.json` должны лежать рядом со скриптом; liveness-якоря T46 (356 dense MMQ launches/chunk, 68 kda halves, gridY = ceil(8128/tile)) зашиты в скрипт — при смене чанкинга обновить. Методология: [../../methods/ab-mtile.md](../../methods/ab-mtile.md).

### ab_nsplit_analyze.py
[ab_nsplit_analyze.py](ab_nsplit_analyze.py)
- Назначение: анализатор N-split A/B — читает `nsplit{1,2}.sqlite` + `_run.json`, пишет `ab-nsplit-results.json`; вердикт `judge_on_net` оценивает восстановление dense-терма в окне 0.22-0.26 s/chunk (net) против 0.24-0.28 (gross).
- Запуск: `ab_nsplit_analyze.py` (без аргументов); CPU-only, sqlite read-only.
- Готчи / требования: как у mtile-анализатора; медиана считается по steady c2..c7 (исключены warmup-чанк и последний полный чанк). Методология: [../../methods/ab-nsplit.md](../../methods/ab-nsplit.md).

### ab_nsplit_stage.sh
[ab_nsplit_stage.sh](ab_nsplit_stage.sh)
- Назначение: serial stage-раннер для обоих A/B-драйверов по протоколу process-hygiene: census до/после, trap-on-EXIT cleanup, timeout 2400 s, SIGTERM -> 10 s -> SIGKILL.
- Запуск: `NSPLIT_NAME=nsplit1|nsplit2 ./ab_nsplit_stage.sh` либо `MTILE_NAME=mtile4|mtile8|mtile16|mtile32|mtile64 ./ab_nsplit_stage.sh`; после прогона сам экспортирует `<name>.nsys-rep` -> `<name>.sqlite` через `nsys export --type sqlite`.
- Готчи / требования: отсутствие `.nsys-rep` после стопа = молчаливо сорванная коллекция (rc=4); баннер инъекции не является дискриминатором; `report1.nsys-rep` в CWD убирается как побочный артефакт abort-итерации.

### ft_nsys_nsplit.sh
[ft_nsys_nsplit.sh](ft_nsys_nsplit.sh)
- Назначение: nsys-обёртка запуска сервера для A/B-захватов (рецепт D09: capture-from-launch).
- Запуск: не вызывается вручную — подменяется в `measure.FT` драйверами; имя сессии берётся из `NSYS_SESSION` (по умолчанию `nsplitcap`).
- Готчи / требования: nsys 2026.1.3 — `launch` отвергает `-o`/`--sample`, поэтому захват с момента старта процесса + `--cuda-graph-trace=node`; запускает `.venv/bin/ft` по абсолютному пути.

### arbitration_run.py
[arbitration_run.py](arbitration_run.py)
- Назначение: арбитражная батарея недетерминизма: 4 plain boot'а (t4a/t64a/t4b/t64b), строго сериализованных и перемежённых, чтобы boot-дрейф не маскировался под эффект конфига; NO nsys нигде (одиночный режим инструментации).
- Запуск: `arbitration_run.py [--only t4a|t64a|t4b|t64b]`; коды выхода: 0 ok, 2 FAILPAT, 3 boot/prefill, 4 батарея, 5 teardown-остатки, 6 стоп по чужим процессам.
- Готчи / требования: quiet-box gate останавливает волну при чужом VRAM-держателе (никогда не убивает его); idle llama-swap proxy на 18080 считается безвредным и не останавливает волну; prefill-якоря tile4=563.90 / tile64=792.08 tok/s — sanity-check каждой boot'ы. Методология: [../../methods/quality-ab.md](../../methods/quality-ab.md), [../../methods/arbitration-battery.md](../../methods/arbitration-battery.md).

### arbitration_analyze.py
[arbitration_analyze.py](arbitration_analyze.py)
- Назначение: разбор арбитражной батареи: same-config пары (t4a~t4b, t64a~t64b) дают базу boot-недетерминизма, cross-config (t4a~t64a и др.) — пере-тест эффекта tile на гранулярности 24 промптов; пишет `arbitration-results.json` + человекочитаемый `arbitration-battery.md`.
- Запуск: `arbitration_analyze.py` (без аргументов); CPU-only, читает `<boot>/p00..p23.json`.
- Готчи / требования: классификация divergent-промптов (whole-answer / prefix-only / mid-answer / trailing-only) требует полных reasoning+content артефактов; вердикт честен в обе стороны — «cross <= same» означает tile-invariance e2e.

### vram_headroom_run.py
[vram_headroom_run.py](vram_headroom_run.py)
- Назначение: волна проверки VRAM-headroom — какие knobs (`--memory-ratio` / `--kv-reserve-tokens` / `--max-prefill-length`) дают >= 1.5 GiB free VRAM на пике при большом контексте и какой ценой в throughput.
- Запуск: `vram_headroom_run.py [--only a|b|c|d|e] [--no-auto-fallback]`; матрица a..e зафиксирована в скрипте (a = control 0.85/500k/8191, e = chunk-рычаг 0.85/500k/4095), бюджет ровно одного extra boot с фолбэком 0.70/300000/8191; коды выхода 0/2/3/4/6.
- Готчи / требования: env санитизируется до committed kernel-knob defaults (никаких `FREETOKEN_*`); VRAM-сэмплер пишет per-boot csv с фазовыми пиками; quiet-box gate как в [arbitration_run.py](arbitration_run.py). Методология: [../../methods/vram-headroom.md](../../methods/vram-headroom.md); контекст: [../../topics/vram-budget.md](../../topics/vram-budget.md).

### microbench_dense.py
[microbench_dense.py](microbench_dense.py)
- Назначение: standalone зонд режима dense q8_0 без serve boot — точная реплика запуска `ggml_mul_mat_a8` (q8_0, N=24896, K=4096, gridY = ceil(M/4)) и timing-sweep по M.
- Запуск: `microbench_dense.py` (без аргументов, нужен CUDA + собранный `freetoken.kernel.gguf`).
- Готчи / требования: MS timing через CUDA events, 20 итераций; сам по себе не разделяет traffic-vs-issue — для этого ncu (см. [../../methods/step0-profile.md](../../methods/step0-profile.md)); фиксаторы-файкстуры (fp16 scale 0x3C) — см. [../../methods/step0-source-read.md](../../methods/step0-source-read.md).

### microbench_format_ab.py
[microbench_format_ab.py](microbench_format_ab.py)
- Назначение: дискриминатор формата — q8_0 против q6_K на одинаковой форме/MAC: time ratio ~0.772 => kernel weight-traffic-bound, ratio >= 1 => issue/ALU-bound.
- Запуск: `microbench_format_ab.py` (без аргументов, CUDA).
- Готчи / требования: выводит итоговый ratio сравнением с модельным 0.8203/1.0625; аналитические q6_K-фикстуры валидированы только на корректность декода (NaN-чек), не на качество весов.

### header_scan.py
[header_scan.py](header_scan.py)
- Назначение: step-0 скан реального GGUF-хедера (gguf-py): все dense-тензоры с размерами/квантом/байтами, агрегаты routed-expert типов по блокам, ключевые KV-пары — авторитетные шейпы для traffic-модели.
- Запуск: `header_scan.py` (без аргументов); путь к GGUF захардкожен, вывод `denseq80_header.json` + печатная таблица.
- Готчи / требования: требует `gguf` (gguf-py) в окружении; read-only по файлу модели; ориентация rows-first/ne-first определяется пробой по `ffn_down.weight`.

### final_split.py
[final_split.py](final_split.py)
- Назначение: единственный генератор step-0 deliverable — per-class атрибуция (медианы x census), regime-модель и проекции; регенерирует `step0-split.json` и `step0-projection-table.md`.
- Запуск: `final_split.py` (без аргументов); читает `denseq80.sqlite` read-only, GPU не трогает.
- Готчи / требования: sqlite-экспорт шага step-0 должен лежать рядом; константы T=14.87 s и W=57.519 s зафиксированы из лога той камеры — это не переисчисляемая модель, а воспроизводимая развёртка. Методология: [../../methods/step0-profile.md](../../methods/step0-profile.md).