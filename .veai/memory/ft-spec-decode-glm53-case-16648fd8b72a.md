---
name: "ft-spec-decode-glm53-case"
description: "Spec-decode GLM-5.3-Flash case CLOSED NO-GO (W0 R4 3.76; W0.5 cW 0.50); stage0 ngram p 0.95 -> W1-narrow justified"
type: project
lastUpdated: 2026-10-11T18:56
lastRecall: 2026-10-12T14:05
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

## W0.5 (2026-10-10): гипотеза "draft-токены в уже кешированные эксперты" проверена измерением
- Метод: bs=1 eager hybrid, якорный конфиг; env-инструментация `_route` мерила c1/cW на 42 MoE-слоях (патч временный, откачен; CUDA-graph replay обходит host-роутинг - хук живёт только с --cuda-graph-max-bs 0).
- Числа: cW med 0.50 (p25 0.25 / p75 0.625), c1 med 0.25-0.375, одинаково @64k и @32k; шум независимых маршрутов 0.028/0.11 - корреляция реальна, но умеренная.
- Вердикт: глубокий draft подтверждён мёртвым (0.50 < порога 0.65-0.78); K=1 ровно на грани полосы 0.40-0.54, но с ценой draft-шага медиана ниже гейта (+2..10% CPU-q8_0 банки, +7..17% VRAM-банки; p75-кварталь +18-30%). Кейс остаётся CLOSED NO-GO (W0+W0.5), запись в kb RESULTS.md + cases/README.md (незакоммичено).
- Дешёвая живая проверка грани при желании пользователя: W1-narrow (n-gram drafter K=1..2 + verify-каркас, ноль VRAM/весов).
- Caveat: алиасинг layer_id слоёв 39-41 в route-статах (keys слились в 0/1; похоже, модули шарят experts-объект) - возможный отдельный follow-up.
- Артефакты: .tasks/glm53-spec-decode-w0/w05_{64k,32k}(_route).json, W05_SUMMARY.md, run_w05.py, summarize_w05.py.

## Stage 0 (2026-10-10): офлайн n-gram acceptance - PASS, W1-narrow оправдан
- Метод: симуляция по токенстриму filler (lookup до вставки, прогрев 1000, 64572 позиций); токенизатор канонический ft serve: freetoken.models.gguf.tokenizer.load_gguf_tokenizer(); filler -> 65573 токенов (ожидалось 65585, 0.018%).
- Числа (N - порядок ключа): p(K=1)=0.907/0.954/0.933/0.912 для N=1/2/3/4; A_1 1.91-1.95; A_2 2.73-2.89 (лучший N=2); A_4 до 4.69; полные каскады N=2 0.954/0.933/0.912/0.891; c_pass(K=1)=0.199-0.244 - гейт проходит при c вдвое меньше измеренной 0.50.
- Пороги: K=1 p>=0.65 при c_W=0.50 (все N выше в 1.4-1.5x); K=2 A>=2.16 (в спеке опечатка 2.09) - min A_2=2.731 проходит.
- Оговорки: filler повторяющийся - верхняя оценка этого контента (совпадает с профилем A/B; acceptance на реальной нагрузке неизвестен); R(c) - экстраполяция, реальный R мерит только W1 A/B; temp 1.0 потребует W3. У n-gram НЕТ цены draft-шага - оценка +30% для K=1 на целевом профиле. Wedge не мешает: verify K+1 = один запрос, cap=1.
- Артефакты: .tasks/glm53-spec-decode-w0/{ngram_offline.py, ngram_offline_results.json, NGRAM_OFFLINE.md, NGRAM_STAGE0_SPEC.md}.
- Состояние: ждётся решение пользователя о запуске W1-narrow (2026-10-10). Волна: verify-каркас в scheduler (K+1 в один forward, argmax, откат GDN-состояния через проводку intermediate_states_buffer + KV/radix trim + DSA-kpool watermark, метрики в /status), fixture-тесты, парный A/B против 18.04-18.65 @64k temp 0, гейт +12-15%. Главный риск - проводка GDN-отката.
