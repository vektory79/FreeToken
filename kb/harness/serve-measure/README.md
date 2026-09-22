# serve-measure/ — измерение throughput живого сервера

Роль: measure.py — патченный клиент измерения: пропускает первый SSE-кадр
(он приходит до префилла), якорит строку `input throughput (token/s):` и
чистит stdout перед json.loads; рядом payload-файлы req_decode.json,
req_ladder.json, req_prefill.json, req_probe.json.

## measure.py — базовый A/B-раннер `ft serve`

[measure.py](measure.py)
- Назначение: запуск + измерение живого `ft serve` (GGUF GLM-5.3-Flash на RTX 5090) в одном из четырёх режимов, teardown с census-верификацией; общий каркас для всех A/B-кампаний.
- Запуск: `python measure.py <boot|prefill|full|ladder> --name <имя> --log <файл.log> [--extra "<флаги ft serve>"] [--env KEY=V] ...`; для `ladder` дополнительно `--max-steps 7 --target-tokens 393216 --step-timeout 900`. Порт захардкожен 18801; строго serial — отказывается стартовать, если живой `ft serve` уже поднят. Рядом лежат payload-файлы [req_decode.json](req_decode.json), [req_ladder.json](req_ladder.json), [req_prefill.json](req_prefill.json), [req_probe.json](req_probe.json); `filler.txt` и `blocks/` шага ladder живут в рабочем каталоге кампании.
- Готчи / требования: `measure.WATCH` — общий FAILPAT для всех наследников; путь `FT` — модульная переменная, которую обёртки подменяют на nsys-wrapper перед стартом; парсер SSE с skip первого кадра и якорь `input throughput (token/s):` описаны в [../../README.md](../../README.md) и [../../topics/prefill-throughput.md](../../topics/prefill-throughput.md); последний полный чанк исключается из медианы (артефакт tail-отчёта) — [../../README.md](../../README.md). Пути к рабочим каталогам и модели захардкожены: копия в kb/ — зеркало, исполнять её нужно из того же окружения, где лежат `filler.txt`, `blocks/` и логи.

Наследники, переиспользующие measure.py как библиотеку (`import measure`):
[ab-runner/ab_mtile_run.py](../ab-runner/ab_mtile_run.py),
[ab_nsplit_run.py](../ab-runner/ab_nsplit_run.py),
[arbitration_run.py](../ab-runner/arbitration_run.py),
[vram_headroom_run.py](../ab-runner/vram_headroom_run.py),
[mmq/step0_run.py](../mmq/step0_run.py),
[v0ab_probe.py](../mmq/v0ab_probe.py),
[v2ab_probe.py](../mmq/v2ab_probe.py).
Как CLI его вызывают обёртки [ab-runner/ab_nsplit_stage.sh](../ab-runner/ab_nsplit_stage.sh),
[mmq/v2ab_stage.sh](../mmq/v2ab_stage.sh), [mmq/v3a_stage.sh](../mmq/v3a_stage.sh),
[repro/run_one.sh](../repro/run_one.sh). [ftw/run-wave2-measure.py](../ftw/run-wave2-measure.py)
не импортирует его, а зеркалит методологию (те же census/prefill/battery/teardown), чтобы
числа были сопоставимы с bare-GGUF референсами.



## Снапшот кампании

Здесь ранее лежал CAMPAIGN.json — снапшот тюнинговой кампании ft serve
(конфиги boot'ов, env-pins, логи). Его результаты дистиллированы: числа и
победитель — в [../../baselines/ft-gguf-serve-tuning/](../../baselines/ft-gguf-serve-tuning/),
отчёт и план — в [../../cases/ft-gguf-serve-tuning/REPORT.md](../../cases/ft-gguf-serve-tuning/REPORT.md).
Сам снапшот удалён как дубликат этих данных.