---
name: "ft-spec-decode-glm53-case"
description: "Spec-decode GLM-5.3-Flash case closed NO-GO by W0 probe (R4 3.76 vs gate 2.32); wedge = DSA OOM + frontend hang"
type: project
lastUpdated: 2026-10-10T22:01
lastRecall: 2026-10-10T21:49
---

# Спек-декод GLM-5.3-Flash: кейс ЗАКРЫТ NO-GO по W0-замеру (2026-10-10)

W0-зонд цены verify-батча (N параллельных декод-потоков через ft serve = верхняя
оценка verify M токенов; независимые потоки -> некоррелированные маршруты):
- hybrid: R_2=1.90-1.96 (якорный конфиг, 32k/64k), R_4=3.76, R_6=6.10 (2k-пара)
- offload: R_2=2.40-2.45, R_4=4.89, R_6=6.72
Гейт кейса +12-15% e2e требует R <= A/1.12. M=2 (K=1, потолок A=2.0): порог 1.79
не достигнут даже при идеальном acceptance; при greedy A=1.6-1.75 это замедление
0.83-0.91x. M=4 (K=3): порог 1.61-2.32, замер выше в 1.6x при нулевом запасе.
ВЕРДИКТ NO-GO, кейс закрыт по правилу TASK.md (валидный исход).

Why: гибридная MoE-нога дорожает по уникальным парам (слой,эксперт) быстрее, чем
платит acceptance. Для опровержения NO-GO нужен дисконт коррелированных маршрутов
>=1.62x (M=4) - не измерен, приоры против (Gemma4 MoE no-gain #23398 даже на GPU;
ik_llama.cpp #1513 2x LOSS при acceptance 0.573).

Находки независимо от вердикта:
1. Slot-floor gate moe-cache-auto зависит от --max-running-requests: план слотов
   1167@N1 (>= min 1059, OK) -> 934@N6 (< 1059, ValueError). Поэтому якорный
   конфиг 0.82/400k/8191 жив только для N<=2 - "якорь не бутится" бывает только
   при больших N.
2. Scheduler WEDGE (баг движка, материал для issue): max-running-requests>=4 AND
   multi-chunk prefill -> ровно один чанк и вечная тишина без Decode batch; 4x
   repro (overlap on/off, stagger 12s, холодные L2); N=1/2 и N=6+single-chunk
   работают.

УТОЧНЕНИЕ механизма wedge (вскрыто при подготовке issue, 2026-10-10): в логах ВСЕХ
репродукций воркер freetoken-TP0-scheduler КРАШИТСЯ с torch.OutOfMemoryError в DSA
sparse-attention префилле (glm_dsa_sparse.py:471, запрос 256 MiB при 287 MiB свободных)
после ровно одного префилл-чанка; фронтенд не сообщает о смерти бэкенда и держит
стрим-коннекты открытыми -> клиент видит вечное зависание (известный gotcha
backend-death hang). Баг двухчастный: (1) OOM DSA-префилла при cap>=4 + multi-chunk,
(2) fail-fast фронтенда. "Тишина без traceback" в артефактах не существует - это
поздняя интерпретация. Драфт issue: .tasks/glm53-spec-decode-w0/issue-wedge-draft.md
(gh issue create пользователь запускает сам).

3. offload N=1 @64k 14.18 tok/s против якоря 12.97-13.72 (+3-8%) - дрейф эры
   T2/C-L1, якоря TASK.md по offload устарели.
4. Структурно: verify M>=4 на 64k в 32 GiB нереализуем - целевой профиль 64k+
   недоступен глубокому draft.

Артефакты: kb/cases/glm53flash-spec-decode/RESULTS.md (+ шапка TASK.md, строка в
cases/README.md); сырые руки и харнес .tasks/glm53-spec-decode-w0/ (вне git).
Scope-решение прежнее: только ft serve, llama.cpp не трогать.
Коммиты 2026-10-10: b361268 (kb closure), b9bb3d8 (memory sync) на vektory79.
