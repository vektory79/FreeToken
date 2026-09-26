# decode-research/ - харнесс декод-кампании гибридного MoE (A/B рук + T1-инструментация)

Байт-в-байт копия харнесса кампании decode-research (2026-09-26, ветка vektory79,
HEAD 0ff8495); живой рабочий каталог кампании - каталог .tasks/decode-research/
(git-ignored: логи arm_*.log, сэмплы cpu_*.csv, свежие arm_*.json, instr_hyb/).
Харнесс загружает `ft serve` с точной production-командой пользователя
(GLM-5.3-Flash-UD-Q3_K_XL GGUF через FTW-директорию, 0.82/400k, kv fp8, tier
10/50, 8191, hybrid, cpu-threads 16, порт 18801), заполняет префикс 65536
токенов, повторно отправляет тот же промпт (radix-reuse) и стримит декод 768
токенов; steady tok/s считается по дельтам SSE-фреймов; параллельно сэмплер
/proc/stat+MHz (1 Гц) пишет per-CPU busy/MHz окна декода. Числа кампании -
[../../baselines/decode-research/RESULTS.md](../../baselines/decode-research/RESULTS.md),
модель стоимости шага -
[../../topics/moe-hybrid-cost-model.md](../../topics/moe-hybrid-cost-model.md).

## Состав

- [run_arm.py](run_arm.py) - одна рука: boot + fill 64k + radix-reuse декод +
  сэмплер CPU + SIGTERM-teardown; пишет arm_<name>.json и cpu_<name>.csv;
- [campaign.py](campaign.py) - сериальный прогон рук (строго один сервер за
  раз), пишет campaign_summary.json;
- [analyze.py](analyze.py) - сводная таблица по arm_*.json + per-CPU busy/MHz
  разбивка по P/E ядрам;
- [analyze_trace.py](analyze_trace.py) - декомпозиция chrome-trace: window
  wall, GPU busy/idle, классы кернелов (fetch_copy / moe / dense_quant_gemm /
  attention / gdn / sampler_head / memcpy), top kernels и cpu ops;
- [driver/sitecustomize.py](driver/sitecustomize.py) - T1-драйвер статистики,
  загружается в дерево ft serve через PYTHONPATH.

## Запуск

```bash
# одна рука: boot точного конфига пользователя + добавка флагами/env
python .tasks/decode-research/run_arm.py --name myarm --log arm_myarm.log \
    --out arm_myarm.json [--extra "--moe-hybrid-max-fetch 3"] \
    [--env FREETOKEN_HYBRID_OVERLAP=0]
# серия рук сериально, имена - фильтр (встроенные руки: base fetch3 ov0 flagsync0)
python .tasks/decode-research/campaign.py [имена рук...]
# разбор результатов
python .tasks/decode-research/analyze.py [имена рук...]
python .tasks/decode-research/analyze_trace.py <trace.json> [tag]
```

Готчи: скрипты содержат машинно-локальные абсолютные пути (REPO/DIR, путь
filler-промпта, порт 18801) - запуск как есть работает только на машине
кампании, на другой машине поправить пути; копии в kb/ не редактируются -
правится оригинал в .tasks и зеркалится сюда; готовность сервера проверяется
реальным fill-запросом, а не дешёвым эндпоинтом.

## T1-драйвер ([driver/sitecustomize.py](driver/sitecustomize.py))

Активируется ТОЛЬКО при FREETOKEN_DECODE_RESEARCH=<outdir> (run_arm.py
прокидывает её и PYTHONPATH каталога драйвера в дерево ft serve; обычные boot'ы
не затронуты). Хуки: Engine._init_offload_moe_cache (включает collect_stats и
collect_decode_freq до capture), GraphRunner.replay (счётчик шагов + CUDA-event
пары: декод-шаги - это реплеи CUDA-графов, host-код модели на шаг НЕ
выполняется), OffloadMoELayer._decode_routed (нужен только eager-прогону
--cuda-graph-max-bs 0, где реплеев нет).

env-крючки:

- FREETOKEN_DECODE_RESEARCH=<outdir> - активация драйвера и каталог вывода;
- FREETOKEN_DR_TAG - тег выходных файлов (по умолчанию run);
- FREETOKEN_DR_SSTART / FREETOKEN_DR_SEND - окно шагов инструментария
  (по умолчанию 150..160);
- FREETOKEN_DR_PROF=1 - открыть окно torch.profiler (kineto). ВНИМАНИЕ: на этой
  машине kineto нативно убивает планировщик; по умолчанию режим events-only
  (CUDA-event спэны шагов без профайлера).

Выход в outdir: stats_<tag>.json пишется ВСЕГДА (aggregate + per_layer + routing
по партициям + step_spans_ms); trace_<tag>.json - только с FREETOKEN_DR_PROF=1.
Маркеры в логе serve: "driver armed" (x4 процесса), "events-only window OPEN" /
"trace window OPEN", "window CLOSED", "DUMPED stats". Броня: тело каждого хука
обёрнуто в try/except - баг драйвера деградирует в no-op и никогда не валит
движок.

Готчи драйвера:

- окно, привязанное к шагам, может молча не открыться: длина декода при temp0
  недетерминирована boot-to-boot (наблюдалось 467..763 токенов) - держи окно
  низким и перед интерпретацией проверяй, что outdir непуст и есть маркер
  "DUMPED stats";
- гистограмма маршрутизации честна только в eager (--cuda-graph-max-bs 0):
  host-scatter не переигрывается под CUDA-графами;
- mean(step_spans_ms) против wall ms/step = разделение шага на
  GPU-stream-занятую и host/idle части без kineto.
