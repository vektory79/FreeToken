---
name: "ft-spec-decode-glm53-case"
description: "Spec-decode GLM-5.3-Flash case CLOSED NO-GO: W0 R4 3.76, W0.5 cW 0.50, W1 ngram acceptance=0 (stage0 method flaw)"
type: project
lastUpdated: 2026-10-17T12:00
lastRecall: 2026-10-24T02:09
---

# Спек-декод GLM-5.3-Flash: кейс ЗАКРЫТ NO-GO окончательно (W0+W0.5+W1, 2026-10-10)

## W0 (цена verify-батча)
N параллельных декод-потоков = верхняя оценка verify M токенов: hybrid R_2=1.90-1.96
(32k/64k), R_4=3.76, R_6=6.10; offload R_2=2.40-2.45, R_4=4.89, R_6=6.72.
Гейт +12-15% требует R <= A/1.12: M=2 (K=1, потолок A=2.0) - порог 1.79 не достигнут
даже при идеальном acceptance (greedy A=1.6-1.75 -> замедление 0.83-0.91x); M=4 (K=3)
порог 1.61-2.32, замер выше в 1.6x. Why: гибридная MoE-нога дорожает по уникальным
парам (слой,эксперт) быстрее, чем платит acceptance; для опровержения нужен дисконт
коррелированных маршрутов >=1.62x (M=4) - приоры против (Gemma4 MoE no-gain #23398
даже на GPU; ik_llama.cpp #1513 2x LOSS при acceptance 0.573).

## W0.5 (draft-токены в уже кешированные эксперты)
bs=1 eager hybrid, env-инструментация `_route` (живёт только с --cuda-graph-max-bs 0:
CUDA-graph replay обходит host-роутинг): cW med 0.50 (p25 0.25 / p75 0.625), c1 med
0.25-0.375, одинаково @64k и @32k. Глубокий draft мёртв (0.50 < порога 0.65-0.78);
K=1 на грани полосы 0.40-0.54, но с ценой draft-шага ниже гейта.
Caveat: алиасинг layer_id слоёв 39-41 в route-статах (похоже, модули шарят
experts-объект) - возможный отдельный follow-up.

## Побочные находки (независимо от вердикта)
1. Slot-floor gate moe-cache-auto зависит от --max-running-requests: план слотов
   1167@N1 (>= min 1059, OK) -> 934@N6 (ValueError). Якорный конфиг 0.82/400k/8191
   жив только для N<=2.
2. Scheduler WEDGE (материал для issue, 2 части): max-running-requests>=4 AND
   multi-chunk prefill -> воркер TP0-scheduler КРАШИТСЯ с torch.OutOfMemoryError
   в DSA sparse-префилле (glm_dsa_sparse.py:471: запрос 256 MiB при 287 MiB свободных)
   после ровно одного чанка; фронтенд не сообщает о смерти бэкенда и держит
   стрим-коннекты -> вечное зависание (gotcha backend-death hang). N=1/2 и
   single-chunk работают. Драфт: .tasks/glm53-spec-decode-w0/issue-wedge-draft.md
   (gh issue create запускает пользователь сам).
3. offload N=1 @64k 14.18 tok/s против якоря 12.97-13.72 (+3-8%) - дрейф эры
   T2/C-L1, offload-якоря TASK.md устарели.
4. Структурно: verify M>=4 на 64k в 32 GiB нереализуем - целевой профиль 64k+
   недоступен глубокому draft.

## Stage 0 + W1-narrow (урок метода)
Офлайн n-gram симуляция по filler-потоку дала p(K=1) 0.91-0.95, A_2 2.73-2.89
(лучший N=2) и оправдала W1-narrow - но W1 A/B на живом декоде дал acceptance = 0
(ноль verify-шагов в ON-руках, k1/k2 в шуме пары +-1.7%) -> NO-GO, кейс закрыт
окончательно. Stage-0 мерил самопересечение потока filler, а не согласие модели
с n-грамм-таблицей: живой greedy-декод не воспроизводит шаблон. Урок: перед
drafting-инфраструктурой мерить model-vs-table agreement на живом декоде;
высокий radix-reuse ПРЕФИКСА не предсказывает повторяемость ВЫВОДА.
Каркас (n-gram drafter + verify + GDN-откат через LinearStatePool.copy_from,
флаги --spec-* default OFF) собран и бутится на GLM-5.3-Flash, но НЕ закоммичен
(14 файлов ~625+ строк; commit default-OFF или откат - решение пользователя).
Детали волны: ft-w1narrow-ngram-wave.

## Артефакты и статус
kb/cases/glm53flash-spec-decode/RESULTS.md (+ шапка TASK.md, строка в
cases/README.md); .tasks/glm53-spec-decode-w0/ и -w1/ (вне git).
Коммиты 2026-10-10: b361268 (kb closure), b9bb3d8 (memory sync) на vektory79.
Scope: только ft serve, llama.cpp не трогать.
