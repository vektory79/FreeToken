---
name: "ft-route-affinity-ab"
description: "Route affinity: gate GO-CLI (0.3 +112%) AND CLI flag --route-affinity landed (default 0.0, env removed); uncommitted"
type: project
lastUpdated: 2026-11-15T21:01
lastRecall: 2026-11-15T17:53
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
