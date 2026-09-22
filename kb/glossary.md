# Глоссарий kb

Алфавитный глоссарий терминов, реально используемых в документах kb. Для
каждого термина — короткое определение и ссылка на документ, где он раскрыт
наиболее полно.

## A

- **A/B** — сравнение двух конфигураций движка, отличающихся ровно одной
  переменной (один boot на значение), с medians-by-chunk метриками; базовый
  протокол всех измерений kb — [ab-nsplit.md](methods/ab-nsplit.md).

## B

- **benchbw** — микробенчмарк пропускной способности банков MoE (CPU/GPU),
  использовался для отбора гибридной стратегии —
  [task-03-benchbw-cpu-moe-gguf.md](cases/gguf-glm5next-hybrid/task-03-benchbw-cpu-moe-gguf.md).

## C

- **chunked prefill** — разбиение длинного промпта на чанки фиксированной длины
  (`--max-prefill-length`), главный источник багов переиспользования кэша —
  [TASK.md](cases/fix3-snapshot-lru-refresh/TASK.md).

## F

- **fail-fast gate** — проверка свободной VRAM на старте, обрывающая загрузку
  раньше OOM при заведомо неподъёмной конфигурации —
  [failfast-report.md](incidents/nvfp4-1m-capacity/failfast-report.md).
- **FTW** — родной формат весов FreeToken; GGUF->FTW конвертация ускоряет
  загрузку модели с диска — [TASK.md](cases/ftw-gguf-fastpath/TASK.md).

## G

- **GDN** — gated delta net, слои линейного внимания модели GLM-5.3; в step-0
  разборе профиля — отдельная статья расходов GEMM —
  [step0-profile.md](methods/step0-profile.md).
- **GGUF** — формат весов llama.cpp, обслуживаемый приватным gguf-путём
  движка на ветке vektory79 — [TASK.md](cases/gguf-glm5next-hybrid/TASK.md).
- **GEMV / GEMM** — матрично-векторное (декод) и матрично-матричное (префилл)
  умножение; базовые операции всех ядерных кампаний —
  [ggml-cpu-kernel-study.md](findings/gguf-glm5next-hybrid/ggml-cpu-kernel-study.md).

## H

- **headroom** — свободная VRAM в пике serving'а; главный лимит больших
  контекстов на 32 ГБ карте — [vram-headroom.md](methods/vram-headroom.md).
- **hybrid / offload MoE** — две стратегии MoE: hybrid (PCIe-подтягивание
  банков, перекрывающееся с CPU GEMV) и offload (полный offload в CPU);
  hybrid признан рекомендуемой — [TASK.md](cases/gguf-glm5next-hybrid/TASK.md).

## K

- **KDA** — блок kda_in_proj (проекция gated-внимания), объект N-split
  оптимизации — [wave1-notes.md](methods/wave1-notes.md).

## M

- **MMQ** — квантованный матричный мультипликатор gguf-ядра (multi-material
  quantized matmul), основа grouped MMQ префилла —
  [TASK.md](cases/mmq-prefill-kernel/TASK.md).
- **mr (memory-ratio)** — флаг `--memory-ratio`, доля VRAM под пул кэша;
  главный рычаг headroom — [vram-headroom.md](methods/vram-headroom.md).
- **MTILE** — env-knob `FREETOKEN_GGUF_MOE_MTILE` / `FREETOKEN_GGUF_DENSE_MTILE`,
  шаблонный m-tile MoE/dense GEMM ядра — [ab-mtile.md](methods/ab-mtile.md).

## N

- **nsys** — профилировщик NVIDIA; step-0-метод опирается на его экспорт —
  [step0-profile.md](methods/step0-profile.md).
- **NVFP4** — 4-битный FP-формат KV/весов; лимитировал 1M-контекст на 32 ГБ —
  [failfast-report.md](incidents/nvfp4-1m-capacity/failfast-report.md).

## R

- **radix cache** — префиксный radix-кэш повторного использования KV;
  баги трекинга длины при chunked prefill — чейны fix-1/fix-3 —
  [TASK.md](cases/fix1-radix-track-seqlen/TASK.md).

## S

- **sqlite-профиль** — экспорт nsys-профиля в SQLite, запросимый SQL без
  повторного GPU-прогона; исходные .nsys-rep/.sqlite следы оставались в
  git-ignored рабочих каталогах кампаний и не зеркалировались —
  [step0-profile.md](methods/step0-profile.md).
- **step-0** — декомпозиция первого префилл-чанка по профилю nsys: доля GEMM,
  копий и прочего, классификация режима BW/ALU —
  [step0-profile.md](methods/step0-profile.md).
- **STOP GATE** — точка в протоколе волн, где оркестратор обязан остановиться
  и дождаться вердикта пользователя —
  [orchestration-notes.md](methods/orchestration-notes.md).

## T

- **TP/FP (триаж)** — в ревью мердж-волн: true-positive / false-positive
  классификация находок ревью —
  [review-2.md](cases/skill-reviews/merge-main-into-vektory79/review-2.md).