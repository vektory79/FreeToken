---
title: "Регламент интеграции origin/main в приватную ветку: волны, прерванная волна, boot-smoke"
date: 2026-09-22
hardware: RTX 5090 32GB (boot-smoke)
branch-commits: "vektory79; слияния: раунд 1 - merge-коммит 380e9eb (родители 6e673ab + afd99cb, 8 конфликтных файлов / 16 ханков), раунд 2 - afd99cb -> cac247a (ноль конфликтов), раунд 3 - merge 5001504 (родители 1445fc1 + 6410c97, ноль текстовых конфликтов, продолжен после прерывания), раунд 4 - merge aa0e71c (родители 41d3581 + 581fdca = upstream/main, 3 конфликта по 1 ханку), раунд 5 - merge f0eed55 (родители b8c3794 + 11c1adb = upstream/main, один аменд после ревью), раунд 6 - merge 594934d -> amend 220b79e (родители 23c1f02 + 32056cf = upstream/main, 27 конфликтных файлов, один аменд после ревью F1-F5); rebase-подход отклонён в пользу merge"
status: validated
tags: [merge, main, upstream, remotes, waves, discovery, adaptation-log, test-results, boot-smoke, provenance, merge-tree, stash, duplicates]
---

# Интеграция origin/main в приватную ветку: регламент волн

**Вердикт (читайте это первым).** Слияние выполняется волнами: discovery ->
adaptation -> test -> boot-smoke, с артефактами discovery-report /
adaptation-log / test-results. Базовые правила: дубликат функциональности ->
main выигрывает; merge вместо rebase (rebase-эксперимент отклонён). Прерванную
волну НЕ переделывать вслепую: сначала побитовая верификация pending-состояния
через `git merge-tree --write-tree` - если дерево совпало, волна ПРОДОЛЖАЕТСЯ,
а история фиксируется в adaptation-log.

## Волны и артефакты

- **Discovery**: карта будущих конфликтов (кандидатные файлы, дивергенции
  сторон), delta ветки против main; отчёт - discovery-report.md раунда.
- **Adaptation**: разрешение конфликтов по правилам ниже + СЕМАНТИЧЕСКАЯ
  верификация каждого авто-юниона (ort может слить текст без конфликтов и
  получить бессмыслицу); журнал решений - adaptation-log.md.
- **Test**: сериализованные прогоны pytest под hard-timeout с цензурой
  процессов до/после каждой волны; сводка - test-results.md (пример: раунд 3,
  383 passed / 0 failed / 4 skipped).
- **Boot-smoke**: реальный бут ft serve на слияном дереве, обязателен когда
  merge трогает путь загрузки весов; ранер и вердиктная строка см. ниже.
- Примеры всех трёх артефактов: раунд 2 -
  [discovery-report.md](../cases/merge-main-work/merge-main-round2-work/discovery-report.md),
  [adaptation-log.md](../cases/merge-main-work/merge-main-round2-work/adaptation-log.md),
  [test-results.md](../cases/merge-main-work/merge-main-round2-work/test-results.md);
  раунд 3 - [discovery-report.md](../cases/merge-main-work/merge-main-round3-work/discovery-report.md),
  [adaptation-log.md](../cases/merge-main-work/merge-main-round3-work/adaptation-log.md),
  [test-results.md](../cases/merge-main-work/merge-main-round3-work/test-results.md).

## Резолв целевого ref: опрашивай ВСЕ remote

В репо ДВА remote: origin (приватный) и upstream (FlashML-org), и новые
изменения могут попадать в upstream/main при стоящем origin/main. Discovery
раунда 4 сначала вернул NOTHING_TO_MERGE по origin/main (его tip 6410c97 уже
был влит раундом 3), и только пользовательская подсказка вскрыла
upstream-дельту в 13 коммитов (581fdca). Правило: discovery обязан опрашивать
ВСЕ remote (`git remote -v` + fetch + merge-base/rev-list по каждому) ДО
вердикта nothing-to-merge; родство проверять через
`git merge-base --is-ancestor` (в раунде 4 upstream/main оказался той же
линией, строго впереди origin/main, так что сохранение раунда 3 было
верифицировано).

## Раунд 4 (2026-10-05): первый upstream/main-мердж

Целевым ref впервые был upstream/main (581fdca, 13 коммитов / 71 файл);
origin/main (6410c97) - строгий предок upstream/main, раунд 3 сохранился
полностью. Merge-коммит aa0e71c (родители 41d3581 + 581fdca), ровно 3 конфликта
по одному ханку (остальные 20 файлов пересечения - авто-юнион):

- [**ftw.py**](../../python/freetoken/checkpoint/ftw.py) (структурный):
  upstream #601 вынес классификацию bank-записей в хелпер
  `_split_bank_entries`; веточные inline-сплит и кросс-чек отпали как точные
  дубликаты (main-wins), а веточная gguf_types-валидация перенесена verbatim в
  `load_ftw_banks` (ftw.py:684-714) сразу после распаковки хелпера, с явным
  `reader.close()` перед каждым raise (хелпер сам бросает без закрытия, а
  блок стоит ПОСЛЕ его try/except). Валидация осталась ТОЛЬКО в
  `load_ftw_banks`, не в хелпере: иначе гейтился бы новый upstream
  `load_ftw_banks_to_device` (fused NVFP4/MXFP4 путь), которого у HEAD не
  было, - семантическая эквивалентность с HEAD сохранена.
- **engine.py** (тривиальный импорт): сохранены оба импорта
  (`physical_core_cpus` ветки + `attach_resident_banks` upstream #601);
  upstream-импорт `attach_offload_moe_cache` отпал сознательно: веточная
  маршрутизация signature-групп сводится к тому же single-cache attach для
  однородных файлов.
- **test_e4m3_compat.py**: обе стороны добавили разные тесты в одно место -
  оба сохранены рядом (веточный constexpr-тест + upstream ROCm-тест).

Дубликатов нет: upstream #601 fused banks и веточная gguf-цепочка
сосуществуют (gguf грузится через `load_ftw_banks`, fused - через
`load_ftw_banks_to_device`, роутинг по resident-флагу). CPU-тесты на слитом
дереве: 286 passed / 0 новых failed / 150 Environment-skips суммарно (269 +
17 top-up; все 3 failed - класс Environment без GPU, на pre-merge HEAD
падают так же). Boot-smoke и GPU-порции тестов исполнены в cloud-окне в тот же
день по VRAM-протоколу ([orchestration-notes.md](../methods/orchestration-notes.md)):
boot-smoke PASS - /ready 200 после 112s (envelope 94-132s), слот-гейт молчит,
decode 18.67 tok/s, SIGTERM rc=143 штатно, VRAM 1098 -> 29276 -> 1098 MiB ровно
baseline; GPU-порции 103 passed / 0 failed (test_ftw_gguf_banks 10,
test_gguf_expert_banks 49, test_qsa_backend 10, test_qsa_pool_fp8 9,
test_e4m3_compat 5+1 ROCm-skip, test_cpu_moe_gguf_iq 20) - закрыты и долги
CPU-окна (PinFailed-тест, gguf_types raise-path, 2 CUDA expert_banks,
11 qsa GPU-only); аменд не потребовался, HEAD остался aa0e71c. Артефакты окна -
[boot-smoke-r4.log](../../.tasks/merge-main-round4-work/boot-smoke-r4.log) и
[test-results.md](../../.tasks/merge-main-round4-work/test-results.md) (секция
Cloud window); артефакты CPU-части раунда (discovery / adaptation / test /
review) остались в рабочей папке волны и в kb не зеркалированы - суть
дистиллирована здесь.

Follow-up F1 (minor, не merge-регрессия): upstream
[load_ftw_banks_to_device](../../python/freetoken/checkpoint/ftw.py)
(ftw.py:851-915) не запускает gguf_types-валидацию - gguf-FTW, доехавший до
resident fused-пути, молча дал бы ExpertBanks с gguf_types=None. Сегодня
недостижимо: resident-ветка требует `method is not None`
([expert_banks.py](../../python/freetoken/moe/expert_banks.py):611), а
gguf-модели грузятся с method=None через gguf-провайдер
(expert_banks.py:617-618). Закрыто коммитом 9ac8830: guard отзеркален в
load_ftw_banks_to_device перед любыми чтениями (тот же re-convert контракт),
тест test_ftw_gguf_types_meta_rejected_to_device - fails-before/passes-after,
CPU-исполним.

## Раунд 5 (2026-10-17): re-platform GGUF-стека, minimal green merge

upstream/main (11c1adb) пере-платформировал весь GGUF-стек: e5d3bd3 (#690) удалил
собственное CUDA-дерево csrc/gguf и python-клей, вместо них вендоренные ggml-ядра
и квант-пакет layers/quantization/; плюс per-format слот-пулы offload-кэша
(#689/#700) и переписанный CPU-MoE executor (#670/#662/#704). Merge-коммит
f0eed55 (родители b8c3794 + 11c1adb), бэкап - backup/vektory79-pre-merge5-b8c3794,
один аменд после ревью (docs и комментарии). Карта порт-долга D1-D5 - что именно
пере-реализовывать на новых механиках - вынесена в статью
[round5-ggml-replatform-deferred-ports.md](round5-ggml-replatform-deferred-ports.md).
Процедурные уроки раунда:

- **Рамка re-platform раунда - явное решение пользователя ДО волны.** Развилка:
  "minimal green merge + отложенные порты" (один merge-коммит чинит только сборку
  и тесты на механиках main'а, всё глубокое - в порт-долг) либо "полная адаптация
  внутри merge-коммита". Выбирать до волны, а не по ходу конфликтов; раунд 5 шёл
  по minimal green merge, глубокая адаптация вынесена в D1-D5.
- **Skip-маркеры отложенной функциональности - легитимный способ удержать minimal
  green merge зелёным**, если reason называет отложенный порт (пример раунда 5:
  "D5: generic-triton fp8/nvfp4 KV kernel path removed by main's attention
  rewrite; re-land deferred"). Ревьюер обязан ПРОБНО запускать скипнутые тесты:
  скип, который проходит, - маскировка. Бухгалтерия обязана сходиться: счётчики
  скипов в логах == фактическим маркерам (раунд 5: engine - 18, test_cache_budget
  - 9 по D3 + 1 по D4).
- **Падение boot-smoke рецепта может быть арифметикой конфига, а не merge-багом.**
  В раунде 5 прогоны r1/r2 умерли на новом cache-budget assert main'а со старым
  рецептом 786k; прошёл r3 на GLM-5.3-Flash-NVFP4-FTW с `--moe-strategy offload
  --moe-cache-auto --disable-moe-prefill-overlap --kv-reserve-tokens 262144
  --kv-cache-dtype nvfp4 --memory-ratio 0.85 --max-prefill-length 4096
  --max-running-requests 1` (порт 18081, /ready 87 с, chat 200 @11.21 tok/s,
  SIGTERM rc=143). Когда старый рецепт опирается на стратегию, ушедшую в
  порт-долг, - валидируй дерево той стратегией, которая проверяет новые пути
  main'а, и задокументируй отклонение от прежнего рецепта.
- **Pathspec-stash и reconcile на посту - не слепой pop.** Если pop сталкивается
  с заново созданными файлами: для каждого пути стэша проверить, что версия в
  рабочем дереве равна или новее версии в стэше (каденс frontmatter/индекса);
  отсутствующее в дереве восстановить и отдать в рабочую папку задачи; только
  затем drop - поверх более новых файлов памяти не накатывать.
- **engine.py auto-юнион после re-platform проверять на висящие хвосты.** Авто-слив
  может оставить веточные хвосты со ссылками на неопределённые имена; резолв -
  "ханки main'а + вплетение тел main'а", затем grep по каждому выброшенному имени
  хелпера на живые ссылки по всему дереву.

## Раунд 6 (2026-10-24/25): второй re-platform upstream/main, ловушки main-wins

upstream/main 32056cf (38 коммитов, 202 файла, +11997/-1528) пере-платформировал
NVFP4-квант-механику (#732/#739 BREAKING, #737 w4a4 на sm_120), hybrid fetch
(#742/#778/#734) и принёс собственные mmproj-пути (#803/#810), дублирующие
веточные порты D2/D4 round 5. Merge 594934d (родители 23c1f02 + 32056cf),
27 конфликтных файлов; режим по директиве пользователя ДО волны - минимальный
зелёный merge + deferred-ports; после ревью один аменд (F1-F5) -> 220b79e.
Кейс волны и карта порт-долга round 6 -
[../cases/merge-round6-upstream-main/TASK.md](../cases/merge-round6-upstream-main/TASK.md).
Процедурные уроки:

- **Main-wins-резолв МОЛЧА вернул регрессию round 5.** Резолв "theirs" на
  register.py потерял веточный точечный фикс `_gguf_tensor_types` (48915b4:
  FTW -> индекс-мета gguf_tensor_types -> семейный контракт ftw_tensor_types ->
  громкий re-convert RuntimeError) - баг "gguf->FTW не бутится" вернулся,
  поймал ровно один тест (test_ftw_quant_config_resolves_schemes). Правило:
  при main-wins на файле, где ветка имеет точечные фиксы, grepить веточные
  фиксы по списку недавних коммитов ветки, а не полагаться на union/main-wins
  механику резолва.
- **Счётчики тестов в test-results.md обязаны быть замеренными, а не
  пересказанными.** Ревью нашло 2 расхождения волны: test_cache_budget заявлено
  53 passed, замер на 220b79e - 51 passed; сплит skip-маркеров заявлен
  D-PORT-D2 x30 / D-PORT-D4 x51, grep-факт - x44 / x37 (суммы 35 passed /
  91 skipped сходятся). Бухгалтерия по памяти ломается на многофазной волне -
  только замер на финальном дереве.
- **Осиротелый хелпер после main-wins rewrite.** `_gguf_auto_floor_gate`
  (engine.py) остался без единого прод-call-site: юнит-тесты звали def напрямую
  и были зелёными, маскируя мёртвую гвардию fail-fast unfundable slot floors,
  при этом wiring-тест был красный. Правило: grep call-sites каждого веточного
  хелпера, попавшего в main-wins файл; def без прод-call-site = либо вживить,
  если зависимости живы (здесь гейт зависел только от gguf_signature_groups /
  check_partition_floors / _resolve_auto_moe_cache_size), либо формально
  задеферить.
- **Skip-маркеры с именованным D-PORT - по round-5 дисциплине, но сплит причин
  замерять grep-ом.** Выведенные D2/D4-тесты без маркеров громко падали
  (test_glm5_next_gguf.py на сыром merge: 90 failed / 35 passed); только
  маркировка reason= именованный порт сделала merge честно зелёным (итог
  35 passed / 91 skipped).

Статус раунда: CPU-зелёный (0 failed на 220b79e: checkpoint 69 passed / 13
skipped, test_cache_budget 51/2, test_glm5_next_gguf 35/91, консолидированный
прогон 155/16); boot-smoke и GPU-порции отложены в cloud-окно.

## Раунд 7 (2026-10-26): split-round CPU-only, UNION-резолвы и триаж латентных композиций

upstream/main a2ce636 (12 коммитов, 84 файла) принёс rows_per_req graph
capture (#833/#837), radix_lookahead seam (#834) и волну flint/minimax/muse
(#816/#797/#829/#830); 7 конфликтных файлов, все резолвы UNION. Режим -
минимальный зелёный merge в варианте SPLIT-ROUND: CPU-часть локально при
резидентном LLM, boot-smoke + GPU-порции - cloud-окно. Merge d8803e0
(родители 67754c9 + a2ce636), CPU 970 passed / 43 skipped / 0 failed,
ревью 0 blockers, без амендов. Кейс волны с брифом cloud-окна -
[../cases/merge-round7-upstream-main/TASK.md](../cases/merge-round7-upstream-main/TASK.md).
Процедурные уроки:

- **UNION-резолв корректен при disjoint-функциональности, но после него
  сверять слитые call-sites и default-значения против main-поведения**:
  union может ввести семантику, которой нет ни у одной из сторон. Пример:
  status.py `_report_decode` получил main-параметр generated, но main сам
  не передаёт его из scheduler.py - слитый call-site остался без generated
  (идентично main-поведению); при этом веточный счётчик `+= len(batch.reqs)`
  заменён main-вариантом с generated, осмысленным при multi-row decode.
- **Триаж латентной merge-композиции - замером достижимости, а не
  фикс-в-мердж**: «код некорректен в недостижимой комбинации флагов»
  проверяется тремя grep-ами: единственный production-конструктор класса
  (grep по python/ вне самого файла; здесь CacheManager - scheduler.py:88-104),
  какие аргументы он передаёт, и call-sites включающего флага
  (radix_lookahead - 0 в проде). Итог: tier-restored insert с raw input_ids
  в cache.py - deferred-followup, не регресс на достижимом пути; фикс -
  отдельным коммитом по явной просьбе пользователя.
- **Расхождение доки-vs-кода - не автоматически merge-регрессия**: датировать
  обе строки (git log -S) и проверить, менял ли merge зону (git diff
  <pre-merge-parent>..HEAD по файлам). Здесь обе строки веточные (03fb043
  добавила строку дока, 2f554c9 - 'auto'-choice позже), в main файла-зоны
  нет вовсе - FP как merge-регрессия, deferred-followup.
- **Осиротелый import в конфликтной зоне import-блока**: union imports
  может притащить уже не используемый импорт (glob в expert_banks.py после
  main-замены glob-логики на iter_weight_files). Замерять по слитому
  дереву до фиксации: import без call-sites - отбрасывать (вариант урока
  раунда 6 про осиротелый хелпер, на более лёгкой форме).

Статус раунда: CPU-зелёный (970/43/0 на d8803e0); boot-smoke и GPU-порции
отложены в cloud-окно.

## Правила разрешения конфликтов

- **Дубликат функциональности -> main выигрывает.** Если main пере-реализовал
  то же, что и ветка, веточная копия отпадает (примеры раунда 1: сайдкар
  hf_quant_config.json, устаревшая строка docs/models.md).
- **Merge вместо rebase**: rebase-подход был опробован и отклонён (приватная
  ветка vektory79 хранит свои merge-коммиты; rebase перепроигрывает их).
  Лог раунда 1, оформленного как merge, -
  [adaptation-log.md](../cases/merge-main-work/rebase-onto-main-work/adaptation-log.md),
  триаж ревью - [findings-triage.md](../cases/merge-main-work/rebase-onto-main-work/findings-triage.md).
- **Ноль текстовых конфликтов != ноль семантических решений.** Каждый
  авто-юнион верифицируется в слитом дереве: обе стороны на месте, guard-ы
  объединены (пример раунда 1: авто-гибрид guard объединил три независимых
  условия в AND; пример раунда 3: main-обёртка `_weight_load_context`
  вокруг общего вызова загрузки банков при веточном floor-gate вне обёртки).
- Обёртки/исключения main'а расширяют веточный код ТОЛЬКО если у main есть
  собственный аналог; менять классификацию сбоев веточных путей под main без
  парного кода main'а нельзя (принятый гэп раунда 3 в
  [adaptation-log.md](../cases/merge-main-work/merge-main-round3-work/adaptation-log.md)).

## Прерванная волна: продолжение вместо redo

Субагент может упасть НА доставке результата ПОСЛЕ того, как работа сделана
(раунд 3: ошибка доступа пришла после старта merge и boot-smoke). Следующая
волна обязана проверить pending-состояние побитово, а не abort+redo:

1. Проверить, что staged-дерево идентично свежему результату
   `git merge-tree --write-tree HEAD <target>`: OID дерева + per-file blob
   OID каждого затронутого файла; worktree == index; grep по конфликтным
   маркерам пуст.
2. Если идентично - ПРОДОЛЖИТЬ: финализировать merge, задокументировать
   провенанс в adaptation-log. Побитовое равенство делает продолжение
   эквивалентным свежему прогону; redo лишь тратит ~2 минуты GPU-бута и
   дублирует работу.
3. Pathspec-stash прерванного прогона (стянутый dirty-набор рабочих файлов) -
   НЕ чужой stash: reconcile на посте - файлы, живущие только в stash,
   восстановить (`git restore --source=stash@{0}`), затем drop; файлы,
   обновлённые параллельными сессиями позже стэша, считать новее стэша.
4. Boot-smoke, выполненный прерванной инкарнацией, - приемлемое доказательство,
   если blob OID слитых файлов совпадают с коммит-деревом и файлы не менялись
   после (независимо перепроверяется через rev-parse/ls-tree).

## Boot-smoke: ранер и ловушка улик

- Ранер - [boot-smoke-r3.sh](../harness/repro/boot-smoke-r3.sh): trap-on-EXIT
  уборка, FAILPAT-дозорный каждые 3 с (AssertionError / OutOfMemoryError /
  backend worker / CUDA error / ValueError), опрос /ready до 250x3 с,
  контрольный POST /v1/chat/completions с полем model.
- **Ловушка улик: uvicorn access log НИКОГДА не печатает строки /ready** -
  без сохранённого stdout ранера доказательство готовности остаётся только
  косвенным (ранер вышел rc=1 без 200 + архивный POST 200). Обязательно tee
  stdout ранера в артефакт, чтобы строка вердикта «READY after Ns» была
  прямой уликой (урок раунда 3, находка B1).
- Вердиктная строка и критерии PASS: slot-gate молчит, суммарные слоты >=
  рабочего набора, время бута в полосе, чистое завершение по SIGTERM, чистая
  цензура процессов; точные VRAM-числа не сравниваются (зависят от стартовой
  свободной VRAM). Пример разбора -
  [test-results.md](../cases/merge-main-work/merge-main-round3-work/test-results.md),
  секция boot-smoke.

## Инварианты финализации

- Ровно один коммит над родителями: `git rev-list --count HEAD --not <parents>`
  == 1; diff первого родителя - ровно дельта main (для merge-коммита `git show
  --name-only` печатает меньше файлов, чем реально вошло - файлы, идентичные
  второму родителю, скрыты; проверять через ls-tree).
- Тесты ДО финализации merge-коммита (если git сам закоммитил - откатить и
  переделать как `--no-commit`; прецедент раунда 2).
- Бэкап-ветка pre-merge HEAD, в merge-диффе нет ничего кроме дельты main
  (service-грязь параллельных сессий в коммит не попадает), нет MERGE_HEAD,
  цензура процессов чистая, без push и без rebase.
- Ревью-проходы по слитому дереву (A корректность / B граничные случаи) с
  триажем TP/FP: раунд 1 - [findings-triage.md](../cases/merge-main-work/rebase-onto-main-work/findings-triage.md);
  примеры юнион-решений 16 ханков - в логе раунда 1 (выше).

Смежное: волновой протокол кампаний и STOP GATE -
[orchestration-notes.md](../methods/orchestration-notes.md); почему длинные
рецепты могут fail-fast на буте - [vram-budget.md](vram-budget.md).