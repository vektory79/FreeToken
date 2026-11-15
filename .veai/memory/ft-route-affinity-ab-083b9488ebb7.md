---
name: "ft-route-affinity-ab"
description: "Route affinity: per-request variant B Ф0-Ф2 done, VERDICT written; uncommitted, commits await owner"
type: project
lastUpdated: 2026-11-16T00:45
lastRecall: 2026-11-16T00:51
---

## CLI FLAG LANDED (2026-11-15, evening): env hook replaced

Владелец решил: CLI-флаг, дефолт 0.0. Реализовано: `--route-affinity` (float,
args.py) -> EngineConfig.route_affinity (+валидация >= 0 в __post_init__) ->
инъекция в ModelConfig в блоке `if is_moe:` engine.py (рядом с
moe_strategy/decode_target, boot-строка `route affinity ...` при > 0) ->
Glm5NextSparseBlock читает config.route_affinity; env FREETOKEN_ROUTE_AFFINITY
полностью убран из python/. GPU-верификация двумя смоуками: флаг 0.3 -
баннер в логе + hit 66.1% (smoke 64-tok) + 40.19 tok/s; дефолт - баннера нет,
hit 26.3%, 21.49 tok/s (без изменений). docs/cli.md - строка флага; харнесс
run_arm.py (live+kb) переведён на флаг; VERDICT.md дополнен разделом
«После гейта». Некоммичено - по правилу ждёт явной команды.

## PER-REQUEST ЗАДАЧА (2026-11-15, вечер)

Владелец утвердил вариант B per-request route affinity: поле `route_affinity`
в запросе (по образцу temperature) снапится к ближайшему значению лестницы
заранее захваченных CUDA-графов (bs, affinity); при разнобое значений в
батче - fallback 0.0 + warn; некаптурированных значений нет (снап всегда).
0.0 в лестнице - прозрачен (= сегодняшние дефолт-графы); чередование
значений между запросами - без рекапчура (граф выбирается на каждом
декод-шаге); bs>1 с разнобоем - только fallback (честный per-request
семантически невозможен: бонус на слой и батч, маска общая). Цена лестницы:
графы делят pool, оценка единицы-десятки МБ/граф, реальная цена - время бута
(|bs| x |лестница| захватов), замерить Ф2.

Реализация НЕ запускалась. Полный бриф со спекой правок по файлам:
kb/cases/route-affinity-request/TASK.md (фазы: Ф0 CPU + тесты, Ф1 приёмочные
пробы, Ф2 цена лестницы). Кейс проиндексирован в kb/cases/README.md вместе с
утерянной записью route-affinity.

## Ф0 РЕАЛИЗОВАНА (2026-11-15/16, код-волна + 2 ревью-прохода, НЕ коммичена)

Перерекон: поле запроса route_affinity (все три API-ручки, сырым значением) -> resolve_sampling -> Req -> DecodeManager.schedule_next_batch: batch_route_affinity() (единая точка резолва+снапа в engine/config.py: None->server_default, снап к ближайшему лестницы ties->меньшее, разнобой -> 0.0 + warning_once) -> Batch.route_affinity -> граф по ключу (padded_bs, affinity) или eager с хуком apply_affinity -> Glm5NextForCausalLM.set_route_affinity. Лестница --route-affinity-ladder: CLI args.py, резолв/дедуп/сорт в EngineConfig.__post_init__; захват графа по всем значениям с повторным сбросом offload-кэша на каждом значении; eager-путь печёт батчевое значение через apply_affinity. Единая точка эффективной лестницы: resolve_route_affinity_ladder(ladder, moe_strategy, has_setter) в engine/config.py — едят и GraphRunner, и DecodeManager (гейт: не-offload/hybrid или нет сеттера -> лестница () + boot-warning; иначе молчаливый eager перф-регресс для запросов с полем).

Блокер ревью ЗАКРЫТ: prefill-контаминация значением eager-декода (apply_affinity оставлял чужое значение в блоках навсегда; prefill-роутинг не по дефолту). Фикс: graph.py affinity_default; apply_affinity: phase=="prefill" -> печь дефолт, decode -> батчевое. Регресс-тест test_eager_decode_then_prefill_restores_the_server_default.

CPU-хвост: 563 passed (test_args, test_graph, 3 API-харнеса, tests/scheduler). Полный not-slow 3174 passed / 44 failed — все GPU-зоны (Triton/CUDA OOM от занятой VRAM), stash-бейзлайн подтверждён.

Осталось: Ф1 приёмочные GPU-пробы (0.3->hit/ток как aff03 гейт; чередование 0.0<->0.3; пара одновременных с разными -> fallback 0.0+warn), Ф2 цена лестницы (free GPU memory + время бута {default} vs {0.0,0.05,0.3}), VERDICT.md кейса. API-доки полей запросов в docs/ отсутствуют — ограничено docs/cli.md.

## Ф1+Ф2 ВЫПОЛНЕНЫ (2026-11-16, GPU-окно, все PASS; VERDICT.md написан)

Ф1 (бут ladder {0.0,0.3}, hybrid, decode 768 ignore_eos, fill 65.6k): F1a три одиночных 0.3 -> 43.4-44.6 tok/s steady, p50 22.1-22.8 ms, hit 78.96/80.03/79.62% (гейт 0.3-рука 45.29/80.9); F1b чередование 0.3<->0.0: 77.2-80.1% / 43.5-44.2 tok/s против 32.4-33.7% / 21.1-21.6 tok/s, разрыв ~44 п.п., captures_after_ready=0 — переключение без рекапчура; F1c два одновременных разных -> fallback 0.0, ровно 1 warn, оба по 768 ток, ~14.4 tok/s (bs=2), hit 32.3% = база. Ф2 цена лестницы (bs=[1], boot A дефолт vs B {0.0,0.05,0.3}): ready 70 vs 64 s (шум), дельта захвата 41 vs 51 MiB -> ~5 MiB/граф, teardown чистый; цена растёт |bs| x |лестница|; дефолтить широкую лестницу можно — решение владельца. Артефакты: .tasks/route-affinity-request-f1f2/ (RUNBOOK, EXPECTED, ARTEFACTS, logs/f1_result.json), kb/cases/route-affinity-request/VERDICT.md (status PHASES DONE), чеклист TASK.md отмечен до пункта «зеркала/коммиты».

ОСТАЛОСЬ: коммиты — по явной команде владельца (диффы в рабочем дереве); 4-й пункт чеклиста (зеркала + уведомление) не закрыт; поле запроса задокументировано только в docs/cli.md (API-доков полей запросов в docs/ нет). Памятка: capture-фаза F1c на --max-running-requests 4 (гейт 1) — только для смешанного батча.
