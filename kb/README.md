# FreeToken knowledge base (kb/)

`kb/` — версионированная база знаний кампаний оптимизации 2026-09 (GGUF glm5next
serving на RTX 5090): методики измерений, переиспользуемые скрипты-харнессы,
измеренные числа, карты кода, брифы кампаний и разборы инцидентов. Всё, что
нужно сессии для работы, лежит здесь; тяжёлые бинарные профили дистиллированы
в текстовые отчёты [reports/](reports/README.md), а одноразовые выводы
дистиллированы в документы.
Правило слоёв: краткие вердикты живут в банке памяти агента, развёрнутые
обоснования и данные — здесь (kb/), сырые оригиналы вне репозитория считаются
утраченными — kb самодостаточен после чистого клона.

## Правила ведения

- Каждый документ начинается с шапки: дата, железо, коммит ветки, статус
  (`validated` / `superseded by ...`).
- Одна тема = один атомарный документ (желательно <=~250 строк); детали — в
  связанных файлах.
- Все перекрёстные ссылки — только markdown-ссылки `[текст](путь)`, никогда
  «голые» имена файлов или каталогов.
- Число без своей конфигурации — неверное число: рядом всегда строка конфига
  (флаги `ft serve`, knob-переменные, размер промпта, время суток).

## Секции

- [methods/](methods/README.md) — как измерять: A/B-протоколы, step-0 профилирование,
  батареи качества, метод VRAM-headroom, протокол оркестрации волн. **Читать
  первым:** перед любой новой A/B-волной — [orchestration-notes.md](methods/orchestration-notes.md);
  «почему падает префилл / OOM» — [vram-headroom.md](methods/vram-headroom.md);
  «как поставить A/B» — [ab-mtile.md](methods/ab-mtile.md) и [ab-nsplit.md](methods/ab-nsplit.md)
  как шаблоны.
- [harness/](harness/README.md) — готовые скрипты, которых нет больше нигде:
  измерение throughput через SSE, A/B-раннеры dense/MMQ, зонды PCIe/NVMe/DRAM,
  скрипты repro. **Читать первым:** карточку нужного скрипта в
  [harness/README.md](harness/README.md), затем раздел «Гочвы» там же — до
  запуска первого прогона.
- [findings/](findings/README.md) — прочные технические карты с якорями в код:
  устройство ggml-ядер CPU, тензоры и конфиг-ключи llama.cpp, снапшот
  upstream GLM5NEXT. **Читать первым:** при работе над ядром или конвертацией —
  [hybrid-machinery-mapping.md](findings/gguf-glm5next-hybrid/hybrid-machinery-mapping.md)
  и [llamacpp-tensor-kinds.md](findings/gguf-glm5next-path-a/llamacpp-tensor-kinds.md).
- [baselines/](baselines/README.md) — машиночитаемые результаты кампаний (JSON/CSV)
  с якорными числами и их конфигами. **Читать первым:** сводку якорей в
  [baselines/README.md](baselines/README.md) — «на какую скорость равняться».
- [cases/](cases/README.md) — брифы, планы и ревью кампаний; двойное назначение —
  шаблоны для будущих план-пакетов (Goal / Scope / Specification / Acceptance /
  Checklist). **Читать первым:** при планировании новой кампании —
  [TASK.md](cases/dense-q80-gemm/TASK.md) и [PLAN.md](cases/gguf-glm5next-hybrid/PLAN.md)
  как образцы.
- [incidents/](incidents/README.md) — разборы отказов с корневой причиной и
  статусом OPEN/CLOSED. **Читать первым:** при загрузке больших контекстов —
  [README.md](incidents/crash-mr080-400k/README.md) (статус OPEN) и
  [failfast-report.md](incidents/nvfp4-1m-capacity/failfast-report.md) (CLOSED).
- [topics/](topics/README.md) — атомарные статьи по темам (вердикт -> механизм ->
  числа -> как измерить): префилл ([prefill-throughput.md](topics/prefill-throughput.md)),
  VRAM-бюджет ([vram-budget.md](topics/vram-budget.md)), переиспользование
  radix-кэша ([radix-cache-reuse.md](topics/radix-cache-reuse.md)), стоимость
  гибридного MoE ([moe-hybrid-cost-model.md](topics/moe-hybrid-cost-model.md)),
  graceful-остановка ft serve ([ft-serve-shutdown-signals.md](topics/ft-serve-shutdown-signals.md)).
- [glossary.md](glossary.md) — глоссарий терминов kb (mr, MTILE, step-0, MMQ,
  fail-fast gate, ...) с ссылками на основной документ по каждому термину.
- [reports/](reports/README.md) - текстовые выжимки nsys-профилей кампаний
  (dense q8_0, MMQ-префилл, v3a m-tile): топ ядер по GPU-времени и числа запусков
  на каждый профиль; тяжёлые бинарники (.sqlite/.nsys-rep) удалены после
  дистилляции, после клона эти отчёты - единственный след профилей.
  **Читать первым:** перед интерпретацией любой таблицы - метод в
  [step0-profile.md](methods/step0-profile.md) и
  [step0-profile-mmq.md](methods/step0-profile-mmq.md).

## Что дистиллировано

Тяжёлые сырые профили (nsys-дампы, полные выводы анализаторов) умышленно
сжаты: в kb остались только вердикты, сводные таблицы и машиночитаемые итоги
в [baselines/](baselines/README.md). Ссылок на рабочие каталоги вне репозитория
в документах нет: артефакт либо дистиллирован сюда, либо помечен как утраченный.
Бинарные зеркала профилей после дистилляции удалены совсем - их текстовые
выжимки по каждому профилю лежат в [reports/](reports/README.md).
