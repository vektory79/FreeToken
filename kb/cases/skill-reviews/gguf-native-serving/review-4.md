# Review 4: session-end self-update rule (gguf-native-serving)

Проверка свежей правки: ORCHESTRATION.md `## 11. Session-end self-update` (строки 302-324) и
русский указатель в SKILL.md (строки 25-29), поверх ранее существовавшего 6-файлового
дельта-набора против коммита 6e673ab.

## Матрица PASS/FAIL

| # | Пункт чек-листа | Статус | Комментарий |
|---|---|---|---|
| A1 | Триггер покрывает ВСЕ классы переиспользуемых знаний (корневые причины с коммит-идами, замеренные классы ожиданий/числа, новые ловушки, правки протокола/гигиены, правки устаревших фактов) | PASS | ORCHESTRATION.md:304-307 перечисляет все пять классов явно |
| A2 | Цели интеграции явные: TRAPS.md (новый T-номер или расширение существующей), брифы (классы ожиданий), §7/§8; update-in-place + cross-reference-over-repeat | PASS | ORCHESTRATION.md:309-314, включая «update-in-place over duplication, cross-reference instead of repeating» |
| A3 | Гейты консистентности: синк счётчиков по 3 точкам (TRAPS H1 / SKILL.md counter line / ORCHESTRATION §10); open-пробелы помечаются open без номера ловушки | PASS | ORCHESTRATION.md:315-318; все три точки реально существуют и синхронны |
| A4 | Исполнение: делегация Code-субагенту + review (blocking_status) + polish | PASS | ORCHESTRATION.md:319-321 |
| A5 | Границы: сам проход никогда не коммитит/пушит, YAML preamble не трогается, README не создаётся | PASS | ORCHESTRATION.md:322-324, с отсылкой на section 8 |
| B1 | Указатель в SKILL.md: русский, под заголовком Orchestration, ссылается на раздел ORCHESTRATION по имени, говорит «проход не коммитит сам» | PASS | SKILL.md:25-29, внутри блока `## Orchestration` (заголовок на стр. 18) |
| B2 | Preamble SKILL.md строки 1-9 байт-в-байт | PASS | git diff 6e673ab: первый ханк `@@ -22,6 +22,12 @@`, строки 1-21 не менялись; YAML 1-8 и пустая строка 9 нетронуты |
| C1 | Счётчики ловушек не тронуты свежей правкой и синхронны в 3 точках | PASS | TRAPS.md:1 = 50 (T01-T50, D01-D10); ORCHESTRATION.md:283 = T01-T50 + recipes D01-D10; SKILL.md:146 = 50 ловушек. Бамп 38->50 в SKILL.md - часть ПРЕДЫДУЩЕГО дельта-набора (волна T48-T50, уже прошедшая review-3), не свежей правки |
| C2 | Нумерация секций связная, без ренумерации; §10 заякорена; новая секция = §11 | PASS | diff ORCHESTRATION: ханк `@@ -281,3 +298,27 @@` - чистое дописывание в конец файла; §10 «Cross-references» на месте (стр. 279), секций 1-11, файл кончается на 324 |
| C3 | Нет дублирования с §8 commit-discipline / escalation - новая секция ссылается, а не повторяет | PASS | Граница §11 отсылает к «(section 8)»; правила коммитов/пуша в §11 не пересказываются; escalation-контент §9 не дублируется |
| C4 | Объёмы разумные | PASS | wc -l: ORCHESTRATION.md = 324 (<= 400), SKILL.md = 153 (< 400) |
| D1 | ORCHESTRATION английский, ASCII-стрелки, без em-dash | PASS | grep U+2014 / U+2013 / U+2192 по ORCHESTRATION.md и SKILL.md - ноль совпадений; стрелки `->` (например, §7 «observed 11 -> 5») |
| D2 | SKILL.md остаётся русским; императив; без воды | PASS | Указатель русский, стиль ASCII-дефиса совпадает с файлом; §11 императивен («MUST run», «delegate», «run a review pass», «apply cheap polish»), плотные буллеты без прозы |
| E1 | Ничего не закоммичено | PASS | git log по каталогу: единственный коммит `6e673ab Model implementation skill` |
| E2 | git status = ровно 6 изменённых файлов, нет untracked в каталоге скилла | PASS | 6 x ` M`: ORCHESTRATION.md, SKILL.md, TRAPS.md, tasks/task-04/06/07; `??` отсутствуют |
| F1 | Детерминизм: агент-исполнитель может пройти §11 дословно (триггер, цели, гейты, исполнение, границы; нет недостижимых шагов; термины определены) | PASS | Триггер: «END of every session that produced reusable knowledge» + закрытый перечень из 5 классов; цели конкретны (все файлы существуют: TRAPS.md, бриф task-06, §7/§8, SKILL.md counter line); гейты проверяемы; шаги достижимы. Два некритичных терминологических замечания см. ниже |

Итог по чек-листу: 16/16 PASS, блокеров нет.

## Блокирующие проблемы

Отсутствуют.

Некритичные замечания (wording nits, не блокируют):

1. **Термин `blocking_status` не определён в файле** (ORCHESTRATION.md:321 — единственное
   вхождение во всём скилле). Термин однозначно читается по сложившейся практике кампании
   (обзоры review-1..3 завершаются `blocking_status: pass|fail`), но перекрёстная ссылка
   на конвенцию review-артефактов сделала бы §11 самодостаточнее.
2. **Указатель разрывает подводку к списку брифов**: SKILL.md:23 заканчивается «брифы задач:»,
   затем вставлен абзац самообновления (25-29), и только на стр. 31 начинается список
   task-00..07. Косметика читабельности; разместить указатель после списка или до абзаца
   с «брифы задач:».
3. **Неточности в авторском отчёте** (на суть не влияют): автор указал счётчик SKILL.md на
   строке 145 — фактически строка 146 (145 = заголовок `## Матрица ловушек`); «первый ханк
   diff на строке 24» — заголовок ханка `@@ -22,6` (первая вставленная строка = 25).
4. Для родительского агента (устаревшая память, не правка скилла): описание памяти
   `ft-gguf-native-serving-skill-32d4c4478ea4.md` в индексе всё ещё говорит «T01-T47 + D01-D10»,
   тогда как актуальное состояние скилла — 50 ловушек (T01-T50, D01-D10).

## Доказательства

ORCHESTRATION.md:302-307 (триггер + классы знаний):

```
## 11. Session-end self-update

At the END of every session that produced reusable knowledge, the Orchestrator
MUST run a self-update pass before closing out. Reusable knowledge: root
causes with commit ids, measured expectation classes / numbers, new traps,
protocol or hygiene refinements, corrections to now-stale facts.
```

ORCHESTRATION.md:309-318 (цели + гейты):

```
- Where to integrate - update-in-place over duplication, cross-reference
  instead of repeating:
  - [TRAPS.md](../../../../.veai/skills/gguf-native-serving/TRAPS.md) - a new T-number continuing the sequence, or an
    in-place extension of an existing trap already covered;
  - task briefs - expected-result classes (task-06 for serving expectations);
  - this file - hygiene / protocol bullets (sections 7 and 8).
- Consistency gates: trap / recipe counters stay in sync across the TRAPS.md
  H1, the SKILL.md counter line and section 10; ASCII only - no em-dashes,
  ASCII arrows; no lab-diary prose; open gaps are labeled open, without a
  trap number.
```

ORCHESTRATION.md:319-324 (исполнение + границы):

```
- Execution: delegate the integration to a Code subagent with the
  session-fact payload; for non-trivial integrations run a review pass
  (blocking_status) and apply cheap polish for non-blocking notes.
- Boundaries: the pass NEVER commits or pushes by itself - commits remain an
  explicit user decision (section 8); the SKILL.md YAML preamble is never
  touched; no README creation.
```

SKILL.md:25-29 (русский указатель):

```
В конце сессии скилл самообновляется: новые знания сессии (причины с
коммит-идами, замеренные классы ожиданий, новые ловушки, правки протокола)
интегрируются в TRAPS.md / брифы / ORCHESTRATION.md (см. раздел
"Session-end self-update" в [ORCHESTRATION.md](../../../../.veai/skills/gguf-native-serving/ORCHESTRATION.md)); проход
никогда не коммитит сам - коммит остаётся решением пользователя.
```

Синхронизация счётчиков (3 точки, актуальное состояние):

```
TRAPS.md:1        # TRAPS.md - 50 ловушек для native GGUF serving (T01-T50, рецепты D01-D10)
ORCHESTRATION.md:283  - [TRAPS.md](../../../../.veai/skills/gguf-native-serving/TRAPS.md) - T01-T50 + recipes D01-D10; the pre-close checklist
SKILL.md:146      См. [TRAPS.md](../../../../.veai/skills/gguf-native-serving/TRAPS.md) - 50 ловушек, каждая из которых стоила реального
```

git-состояние (ничего не закоммичено, ровно 6 файлов, без untracked):

```
$ git log --oneline -5 -- .veai/skills/gguf-native-serving/
6e673ab Model implementation skill

$ git status --porcelain -- .veai/skills/gguf-native-serving/
 M .veai/skills/gguf-native-serving/ORCHESTRATION.md
 M .veai/skills/gguf-native-serving/SKILL.md
 M .veai/skills/gguf-native-serving/TRAPS.md
 M .veai/skills/gguf-native-serving/tasks/task-04-kernel-dispatch.md
 M .veai/skills/gguf-native-serving/tasks/task-06-e2e-ab.md
 M .veai/skills/gguf-native-serving/tasks/task-07-cpu-compute-tier.md
```

Целостность preamble и чистое дописывание §11 (git diff 6e673ab):

```
SKILL.md:        первый ханк @@ -22,6 +22,12 @@  -> строки 1-21 (вкл. YAML 1-8) не менялись
SKILL.md numstat: 7 insertions, 1 deletion (6 строк указателя + бамп 38->50 из прежней волны)
ORCHESTRATION.md: ханк §11 @@ -281,3 +298,27 @@ -> чистое дописывание в конец (324 строки)
```

Стиль (grep по ORCHESTRATION.md и SKILL.md): em-dash U+2014 - 0, en-dash U+2013 - 0,
unicode-стрелка U+2192 - 0. wc -l: ORCHESTRATION.md = 324, SKILL.md = 153.

## Итоговый blocking_status: pass
