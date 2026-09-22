# Review-1: gguf-native-serving skill (authoring checklist)

Review target: `.veai/skills/gguf-native-serving/` (12 files: SKILL.md 116 lines, ORCHESTRATION.md 256, TRAPS.md 153, tasks/task-00..07 8 briefs). Все файлы прочитаны полностью; файлы не изменялись. Проверка вёлась по чеклисту A-G (blocking = любой FAIL).

## Матрица PASS/FAIL

| Пункт | Критерий | Вердикт |
|---|---|---|
| A | YAML-преамбула SKILL.md (---, name, schemaVersion v0.1, description, agent, used-by) | **FAIL (blocking)** — преамбула ОТСУТСТВУЕТ ПОЛНОСТЬЮ |
| B | Тело SKILL.md: императивные шаги, ветвления, входы/выходы, acceptance criteria, <400 строк, без воды | PASS (116 строк; замечания не блокируют) |
| C | R1: SKILL.md явно ссылается на КАЖДЫЙ sibling-файл markdown-ссылками | **FAIL (blocking)** — ни одной markdown-ссылки |
| D | Кросс-ссылки: фазы/ловушки/рецепты/worked-example консистентны между файлами | **FAIL (blocking)** — Phase 0 осиротевшая, счётчик фаз 7 vs 8 |
| E | Требования к субагентам: роли, формат результата, стратегия разбиения, cap циклов <=4 | PASS (cap 3 цикла, ORCHESTRATION.md:134-135) |
| F | Safety/ops: нет README.md, секретов, абсолютных путей, деструктива без подтверждения | PASS |
| G | Блокирующи только ссылки SKILL.md (ORCHESTRATION.md/TRAPS.md — нет) | Учтено; информация в замечаниях |

## Блокирующие проблемы

### 1. [A] YAML-преамбула SKILL.md отсутствует полностью
- **Type/Severity:** Documentation / Error (blocking)
- Файл начинается сразу с `# Skill: Native GGUF Serving for Arbitrary Model Families (FreeToken)` (SKILL.md:1). Нет открывающего `---`, нет закрывающего `---`; следовательно нет ни одного обязательного поля: `name` (должен совпадать с папкой `gguf-native-serving`, <=50 chars), `schemaVersion: "v0.1"`, `description` (что делает + триггер-фразы, <=1024 chars, без `<`/`>`), `agent`, `used-by`.
- Чеклист A прямо требует: преамбула отсутствует целиком => blocking FAIL. Все подпункты A не выполнены по причине отсутствия преамбулы (не частичный дефект, а полное отсутствие).
- Минимальный фикс: добавить преамбулу в начало SKILL.md: `name: gguf-native-serving`, `schemaVersion: "v0.1"`, description с триггерами (".gguf файл", "architecture X is not supported", "нативный gguf-сёрвинг"), `agent` валидным значением; `used-by` можно опустить.

### 2. [C/R1] SKILL.md не содержит ни одной markdown-ссылки на sibling-файлы
- **Type/Severity:** Documentation / Error (blocking)
- Поиск `]` по SKILL.md даёт 0 совпадений => markdown-ссылки формата `[text](path)` отсутствуют целиком. Все 10 sibling-файлов упомянуты только инлайн-текстом/глобами:
  - ORCHESTRATION.md — SKILL.md:11 ("Run via ORCHESTRATION.md"), SKILL.md:54 — plain text;
  - TRAPS.md — SKILL.md:116 ("См. TRAPS.md") — plain text;
  - tasks/task-00..07 — SKILL.md:14 ("tasks/task-00..07", диапазонная запись), SKILL.md:53 ("tasks/task-0N-*.md", глоб), SKILL.md:60 (только task-07 назван по имени) — plain text.
- НЕ ссылаются (полный список, требующий ссылок): `ORCHESTRATION.md`, `TRAPS.md`, `tasks/task-00-discovery.md`, `tasks/task-01-config-shim.md`, `tasks/task-02-tensor-translator.md`, `tasks/task-03-tokenizer.md`, `tasks/task-04-kernel-dispatch.md`, `tasks/task-05-expert-banks.md`, `tasks/task-06-e2e-ab.md`, `tasks/task-07-cpu-compute-tier.md` — т.е. ВСЕ sibling-файлы.
- Минимальный фикс: в SKILL.md заменить упоминания на явные ссылки относительно местоположения SKILL.md: [ORCHESTRATION.md](../../../../.veai/skills/gguf-native-serving/ORCHESTRATION.md), [TRAPS.md](../../../../.veai/skills/gguf-native-serving/TRAPS.md), и по одной ссылке на каждый бриф: [tasks/task-00-discovery.md](../../../../.veai/skills/gguf-native-serving/tasks/task-00-discovery.md) ... [tasks/task-07-cpu-compute-tier.md](../../../../.veai/skills/gguf-native-serving/tasks/task-07-cpu-compute-tier.md) (в т.ч. раскрыть диапазонные записи "task-00..07" / "task-0N-*.md" в список из 8 ссылок; здесь ссылки даны относительно самого ревью, в SKILL.md они sibling-relative).

### 3. [D] Фаза 0 осиротевшая: счётчик фаз 7 vs 8, у Phase 0 нет волны/брифа/STOP GATE
- **Type/Severity:** Design (consistency) / Error (blocking в рамках чеклиста D)
- SKILL.md:36 объявляет конвейер "(7 фаз, строго последовательно)", но блок SKILL.md:39-48 перечисляет 8 фаз: Phase 0..Phase 7.
- ORCHESTRATION.md повторяет ошибку счёта и расписания: ORCHESTRATION.md:71 "The 7 domain phases of SKILL.md ARE the default plan"; ORCHESTRATION.md:14 default cycle "task-00 discovery -> phase waves 1..7 (SKILL.md order)". Итог: Phase 0 (upstream alignment / waive для private-use, SKILL.md:39) объявлена частью строгой последовательности, но:
  - у неё нет волны в default cycle (waves 1..7);
  - у неё нет брифа: tasks/ содержит ровно task-00..07, где task-00 = discovery (inventory/anchors/reference question), шага upstream-alignment/waive в нём нет (task-00:11-76, шесть шагов инструкций);
  - ORCHESTRATION.md вообще не оперирует понятием waive/upstream alignment (слово "waive" в файле отсутствует).
- Исполнитель, буквально следующий контракту ORCHESTRATION.md, никогда не выполнит Phase 0 — пропадает обязательный шаг (roadmap/upstream alignment либо осознанный waive), заявленный в SKILL.md как строго обязательный этап конвейера. Дополнительно ORCHESTRATION.md:3 "Do not skip or reorder phases without the user" конфликтует с существованием Phase 0, у которой нет пути исполнения.
- Минимальный фикс (один из двух): (а) перенести upstream-alignment/waive в task-00 discovery как явный шаг + добавить его в default cycle ORCHESTRATION.md и исправить счётчики ("7 фаз" -> фактическая нумерация, например "фазы 1-7 + discovery + upstream-alignment gate"); либо (б) убрать "Phase 0" из блока фаз и описать alignment/waive как предусловие discovery. Счётчик "7 фаз" в SKILL.md:36 и "7 domain phases"/"waves 1..7" в ORCHESTRATION.md:71/:14 привести к одному варианту.

## Не блокирующие замечания

- [B] Acceptance criteria всего скилла делегированы брифам (SKILL.md:51-54); в самом SKILL.md финального критерия приёмки нет — допустимо, т.к. каждый бриф содержит "## Acceptance criteria" с commit-gate, но можно добавить одну строку-итог.
- [B/D] Phase 7 опциональна (SKILL.md:46, 59), а ORCHESTRATION.md:14 включает её в waves 1..7 безусловно; ветка "пропустить фазу 7, если файл сервится offload-only" в ORCHESTRATION.md не прописана явно (косвенно покрывается "Do not skip or reorder phases without the user", ORCHESTRATION.md:3). Стоит добавить явное ветвление.
- [D] Только task-00 называет файл TRAPS.md ("Traps (from TRAPS.md)", task-00:66); брифы task-01..07 цитируют ID ловушек без имени файла (task-01:42, task-02:47, task-04:39 и т.д.). Сами диапазоны совпадают с TRAPS.md точно (см. Доказательства), но ссылка на файл в каждом брифе была бы полезна.
- [G] ORCHESTRATION.md:249-254 и TRAPS.md тоже ссылаются на sibling'ы plain-текстом (без markdown-ссылок) — по R1 не блокирует, но та же правка подняла бы навигируемость.
- [F] Весь хардкод железа в worked-example (RTX 5090, >=40 GB/s gate, 53-66 GB/s, 16.67/12.97 tok/s) подан как пример/анти-урок, а не как требование — корректно.

## Доказательства (file:line)

Пути относительно корня проекта; все строки проверены чтением полного файла / поиском.

- SKILL.md:1 — `# Skill: Native GGUF Serving...` (первая строка, нет `---`; преамбула отсутствует; файл прочитан целиком, 116 строк).
- SKILL.md:11 — "Run via ORCHESTRATION.md ..." (plain text, не ссылка); SKILL.md:54 — "ORCHESTRATION.md)." (plain text).
- SKILL.md:116 — "См. TRAPS.md - 38 ловушек..." (plain text, не ссылка).
- SKILL.md:14 — "tasks/task-00..07" (диапазонная запись, не ссылки); SKILL.md:53 — "tasks/task-0N-*.md" (глоб); SKILL.md:60 — "tasks/task-07-cpu-compute-tier.md" (plain text).
- SKILL.md:36 — "## Архитектура конвейера (7 фаз, строго последовательно)"; SKILL.md:39-48 — блок с "Phase 0: upstream alignment (или waive для private-use)" .. "Phase 7: CPU compute tier ... опциональна" (8 записей).
- ORCHESTRATION.md:14 — "task-00 discovery -> phase waves 1..7 (SKILL.md order)".
- ORCHESTRATION.md:71 — "The 7 domain phases of SKILL.md ARE the default plan".
- ORCHESTRATION.md:134-135 — "Limit: 3 full cycles per wave" (cap циклов 3 <= 4, пункт E PASS); ORCHESTRATION.md:241 — "loop limit reached (3 cycles)".
- ORCHESTRATION.md:222-227 — commit discipline: "NO git push, no gh pr create..."; скилл-файлы не коммитятся агентами (пункт F PASS).
- ORCHESTRATION.md:15-21 — роль/границы, делегирование Code/Test/Review/Ask (пункт E PASS); ORCHESTRATION.md:152-176 (§6 payload minimums) — форматы результатов Ask/Code/Test/Review/Triage.
- TRAPS.md:1-153 — T01-T38 (включая T12b на строках 34-36) + D01-D08; группировка по задачам Task 00..07 совпадает с цитатами в брифах.
- task-00:3 — "Type: Ask (read-only) | Agent: Ask or Code (measurement)"; task-00:66 — "## Traps (from TRAPS.md)"; task-00:11-76 — шаги 1-6 без upstream-alignment/waive (подтверждение осиротевшей Phase 0).
- task-01:3, task-02:3, task-03:3, task-04:3, task-05:3, task-06:3, task-07:3 — "Agent: Code" (явные роли, пункт E PASS).
- Ловушки по брифам == TRAPS.md: task-00 (T01-T03, T35), task-01 (T04-T07), task-02 (T08-T12), task-03 (T13-T16), task-04 (T17-T20), task-05 (T21-T24), task-06 (T25-T29), task-07 (T30-T38); D07/D08 в task-07:85-99; всё сходится (пункт D, частично PASS).
- Worked example консистентен: SKILL.md:109-110 (microbench 53-66 GB/s; hybrid 16.67 vs offload 12.97 @64k) == ORCHESTRATION.md:250-254 (commits 11f1a80 + 979e3fc + 63b9bff, gate >= 40) == task-07:13-16 (H=4096 I=2048 E=288 top_k=8, 42 MoE layers).
- Поиски по всей папке скилла: `]` в SKILL.md — 0 совпадений (нет markdown-ссылок); `/media/` — 0; `/home/` — 0; `vektor` — 0 (пункт F PASS: нет абсолютных машинных путей и персональных данных).
- Список каталога `.veai/skills/gguf-native-serving/` — 3 файла + tasks/ (8 брифов); README.md отсутствует (пункт F PASS).

Вердикты по пунктам: A FAIL, B PASS, C FAIL, D FAIL, E PASS, F PASS; G — только SKILL.md блокирует (учтено).

## Итоговый blocking_status: fail
