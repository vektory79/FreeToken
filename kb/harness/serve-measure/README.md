# serve-measure/ — измерение throughput живого сервера

Роль: measure.py — патченный клиент измерения: пропускает первый SSE-кадр
(он приходит до префилла), якорит строку `input throughput (token/s):` и
чистит stdout перед json.loads; рядом payload-файлы req_decode.json,
req_ladder.json, req_prefill.json, req_probe.json.

Подробная карточка использования (аргументы, типовые прогоны, гочвы) —
отдельная более поздняя фаза.

## Снапшот кампании

Здесь ранее лежал CAMPAIGN.json — снапшот тюнинговой кампании ft serve
(конфиги boot'ов, env-pins, логи). Его результаты дистиллированы: числа и
победитель — в [../../baselines/ft-gguf-serve-tuning/](../../baselines/ft-gguf-serve-tuning/),
отчёт и план — в [../../cases/ft-gguf-serve-tuning/REPORT.md](../../cases/ft-gguf-serve-tuning/REPORT.md).
Сам снапшот удалён как дубликат этих данных.