# cases/ — брифы, планы и ревью кампаний

Каждая кампания — свой подкаталог: бриф TASK/PLAN, обзоры волн, верификация.
Брифы и план-пакеты — шаблоны для будущих кампаний: Goal / Scope /
Specification / Acceptance / Checklist.

## Кампании

- [dense-q80-gemm/](dense-q80-gemm/) — полный цикл кампании: бриф, триаж
  nsplit/step0, ревью волн A/B и свипа (review-a-*/review-b-*). Учит, как
  ревью фиксируют отклонения от брифа. Вход: [TASK.md](dense-q80-gemm/TASK.md).
- [fix1-radix-track-seqlen/](fix1-radix-track-seqlen/) — баг-фикс с тестом
  «падает до / проходит после» и hardware-верификацией. Вход:
  [TASK.md](fix1-radix-track-seqlen/TASK.md). Статус: COMMITTED (5b72aba).
- [fix3-snapshot-lru-refresh/](fix3-snapshot-lru-refresh/) — диагностика
  периодической полной потери кэша по двум телам запросов; WAVES-план волн.
  Вход: [TASK.md](fix3-snapshot-lru-refresh/TASK.md). Статус: OPEN.
- [session-cache-tiering/](session-cache-tiering/) — бриф ярусного кеша сессий
  (L0 VRAM -> L1 RAM 30 GiB -> L2 SSD 100 GiB): pull-on-evict с адаптивной
  селекцией, контент-адресный стор, журнал-персистентность; фазы 1/2, приёмка
  на фазу 1. Вход: [TASK.md](session-cache-tiering/TASK.md). Статус: PLANNED.
- [ft-gguf-serve-tuning/](ft-gguf-serve-tuning/) — тюнинг serving-флагов:
  REPORT с победителем и PHASE2-план как шаблон фазировки. Вход:
  [REPORT.md](ft-gguf-serve-tuning/REPORT.md).
- [ftw-gguf-fastpath/](ftw-gguf-fastpath/) — формат брифа с задачей-строкой S:
  ускорение загрузки GGUF->FTW. Вход: [TASK.md](ftw-gguf-fastpath/TASK.md).
  Статус: DONE.
- [gguf-glm5next-hybrid/](gguf-glm5next-hybrid/) — эталонный план-пакет:
  TASK + PLAN + task-01..07 + verification/ (A/B-таблицы, батарея, bisect,
  re-acceptance). Учит, как измерением разворачивают NO-GO вердикт. Входы:
  [PLAN.md](gguf-glm5next-hybrid/PLAN.md), [TASK.md](gguf-glm5next-hybrid/TASK.md).
- [gguf-glm5next-path-a/](gguf-glm5next-path-a/) — Path A (offload-этап): PLAN
  с фазами 0..6, boot-smoke-отчёты, разбор root-cause в phase6. Вход:
  [PLAN.md](gguf-glm5next-path-a/PLAN.md).
- [merge-main-work/](merge-main-work/) — процедура интеграции origin/main:
  discovery/adaptation/test-waves в трёх раундах (round2, round3, rebase).
  Шаблон discovery-этапа:
  [merge-main-round2-work/discovery-report.md](merge-main-work/merge-main-round2-work/discovery-report.md);
  триаж находок — [rebase-onto-main-work/findings-triage.md](merge-main-work/rebase-onto-main-work/findings-triage.md).
- [mmq-prefill-kernel/](mmq-prefill-kernel/) — бриф v0->v2: как нулевой
  эксперимент (сортировка пар) уточнил цель следующей итерации. Вход:
  [TASK.md](mmq-prefill-kernel/TASK.md).
- [mmq-v3-stationary-moe/](mmq-v3-stationary-moe/) — бриф v3 (weight-stationary
  расписание): закрытие read-once-разрыва. Вход:
  [TASK.md](mmq-v3-stationary-moe/TASK.md).
- [skill-reviews/](skill-reviews/) — ревью скиллов с TP/FP-классификацией
  находок — образец структуры ревью. Вход:
  [merge-main-into-vektory79/review-2.md](skill-reviews/merge-main-into-vektory79/review-2.md).
- [serve-llama-swap-metrics/](serve-llama-swap-metrics/) — бриф: полная
  статистика запросов llama-swap для ft serve (usage/timings, честные скорости
  префила и декода без warmup/last-chunk артефактов). W1 (конфиг) + W5
  (hardware acceptance, обе ft-секции) приняты; код W2-W4 в рабочем дереве.
  Входы: [TASK.md](serve-llama-swap-metrics/TASK.md),
  [REPORT.md](serve-llama-swap-metrics/REPORT.md). Статус: W5 DONE.

## Навигация

- Планируете новую кампанию — берите шапку статуса, Goal/Scope/Acceptance из
  [gguf-glm5next-hybrid/TASK.md](gguf-glm5next-hybrid/TASK.md).
- Числа кампаний — в [../baselines/README.md](../baselines/README.md);
  методики волн — в [../methods/README.md](../methods/README.md).