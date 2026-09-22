# baselines/ — измеренные числа кампаний

Машиночитаемые результаты (JSON/CSV): по подкаталогу на кампанию. Число без
строки конфигурации — неверное число; якоря ниже приведены с их конфигом.
Конвенции измерений: медианы считаются по полным префилл-чанкам без последнего
(его строка throughput искажена) и без первого (прогрев); каждому методу
соответствует машинночитаемый близнец здесь и методика в
[../methods/README.md](../methods/README.md).

## Кампании

- [dense-q80-gemm/](dense-q80-gemm/) — dense q8_0 GEMM: A/B свипы mtile и
  nsplit, арбитраж качеств, посухи mtile4..64, step0-сплит, VRAM-headroom
  (CSV + boot-инвентари), качество арбитража в
  [quality-arbitration/](dense-q80-gemm/quality-arbitration/).
  Якоря: префилл 792-808 tok/s (GLM-5.3-Flash-UD-Q3_K_XL, MOE_MTILE=32,
  DENSE_MTILE=64, NSPLIT=2, winner-флаги, порт 18801); dense tile64 +40.46%,
  N-split +2.61% против базовой 8e2e4c7; для конфига 0.80/400k нужно
  >=29.28 GiB свободной VRAM.
- [ft-gguf-serve-tuning/](ft-gguf-serve-tuning/) — тюнинг serving-флагов:
  матрица results_*.json, калибровка, PHASE2-сводка, vram-логи.
  Якоря: победитель mr1+8191+0.85 (`--memory-ratio 0.85 --max-prefill-length
  8191` + radix mr1); radix L-drop — корневая причина потерь; база ~293 tok/s
  до dense-q80.
- [ftw/](ftw/) — GGUF->FTW fast path: parity и результаты wave2, census.
  Якорь: boot 52.1 s против 94-132 s на GGUF-пути (wave1/wave2 конвертация).
- [mmq-prefill-kernel/](mmq-prefill-kernel/README.md) — MMQ префилл v0/v2:
  step0-сплит, A/B результаты, пробы, quality-этап.
  Якоря: v2 grouped MMQ +16.4% @8128 (префилл, UD-Q3_K_XL, winner-флаги);
  v0 — выигрыша нет.
- [mmq-v3/](mmq-v3/) — v3a stationary MoE: A/B свип, дивергенция, проба,
  stage-raw, quality-этапы.
  Якорь: тайл 32, +67.2% @8128 = 556.43 tok/s (коммит 1706aae);
  production-дефолт MOE_MTILE=32.

## Исторические якоря serving'а

Гибридная стратегия MoE: 16.67 tok/s @64k против 12.97 у offload (реальный
файл, re-acceptance) — вердикт и конфиг в
[../cases/gguf-glm5next-hybrid/TASK.md](../cases/gguf-glm5next-hybrid/TASK.md).
Дневные-vs-ночные декоды на этой машине несравнимы напрямую: дневной шум один
раз перекрыл реальные -10% — всегда указывайте время суток рядом с числом.