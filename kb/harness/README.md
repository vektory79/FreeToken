# harness/ — переиспользуемые скрипты измерений

Скрипты, которых нет больше нигде в репозитории: они живут только здесь и
переиспользуются между кампаниями. **Карточки использования (что запускается,
с какими аргументами, гочвы) — отдельная более поздняя фаза; ниже пока только
роль каждого подкаталога и ключевых входов.**

| Каталог | Роль |
|---|---|
| [serve-measure/](serve-measure/README.md) | measure.py (патченный парсер SSE) + payload-файлы req_*.json: throughput-измерение живого сервера |
| [ab-runner/](ab-runner/) | A/B-машинерия dense q8_0: свипы mtile/nsplit, арбитраж, микробенчмарки, VRAM-headroom, параметризованный stage-раннер |
| [mmq/](mmq/) | A/B и анализ MMQ: v0/v2 пробы и анализаторы, step0-запуск/разбор, v3a stage и батарея, проба живости |
| [quality/](quality/) | Батарея качеств: запуск прогонов и разбор дивергенции |
| [probes/](probes/) | Зонды железа RTX 5090 / NVMe: PCIe BW, RAM/DRAM BW, size-scaling, NVMe-термика |
| [repro/](repro/) | Воспроизводители: crash-mr080, boot-smoke мердж-волн, волновой раннер |
| [ftw/](ftw/) | GGUF->FTW: конвертация wave1, верификация конвертации, measure/parity wave2 |

Ключевые входы (по одному на типовую задачу):

- «измерить tok/s живого сервера» — [serve-measure/measure.py](serve-measure/measure.py);
- «прогнать A/B свип dense» — [ab-runner/ab_mtile_run.py](ab-runner/ab_mtile_run.py),
  [ab_nsplit_run.py](ab-runner/ab_nsplit_run.py), разбор — одноимённые
  ab_*_analyze.py;
- «измерить headroom конфига» — [ab-runner/vram_headroom_run.py](ab-runner/vram_headroom_run.py);
- «батарея качества» — [quality/run_battery.py](quality/run_battery.py) +
  [analyze_divergence.py](quality/analyze_divergence.py);
- «воспроизвести crash-mr080» — [repro/crash-mr080-repro.sh](repro/crash-mr080-repro.sh),
  детали инцидента — [../incidents/crash-mr080-400k/README.md](../incidents/crash-mr080-400k/README.md);
- «зонд шины» — [probes/pcie_bw_probe.py](probes/pcie_bw_probe.py),
  [ram_bw.py](probes/ram_bw.py).

Правила: скрипты в kb/ не редактируются как копии — правится оригинал и
зеркалится сюда; каждая кампания дописывает сюда только то, что переиспользуется
дальше. Общие гочвы прогонов (skip times[0] SSE, якорь строки
`input throughput (token/s):`, исключение последнего полного чанка из медианы,
pkill через `[f]t serve`) перечислены в [../README.md](../README.md).