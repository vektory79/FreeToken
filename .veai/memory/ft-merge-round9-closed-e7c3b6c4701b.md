---
name: "ft-merge-round9-closed"
description: "Round 9 merge b79457e CLOSED: boot-smoke PASS attempt 1 (READY 229s), GPU 969 passed/4 baseline; STOP GATE closed"
type: project
lastUpdated: 2026-11-09T20:57
lastRecall: 2026-11-09T20:33
---

# Merge round 9 (2026-11-08 wave): CLOSED - b79457e, STOP GATE closed

Merge upstream/main 5a73685 -> vektory79: **b79457e** (parents 7154bab + 5a73685), rev-list count 1, .veai=0, ровно один merge-коммит, amend не потребовался ни в CPU-, ни в cloud-окне. Дельта: 3 коммита (340db81 #877 graph-capture KV writes to dummy page - engine/graph.py; ef6b303 #878 cap MoE route sort shared memory at device limit - mmid.cu; 5a73685 release 0.2.2), 8 файлов +70/-86. Base 8215a94. 0 конфликтов (merge-tree предсказал верно), один main-wins: upstream удалил свой устаревший тест test_graph_capture_reuses_warm_offload_cache_before_capture (77 строк, покрытие superset в tests/engine/test_graph.py). Backup: backup/vektory79-pre-merge9-7154bab. origin/vektory79 отстаёт на 36; push НЕ делался.

## Верификация
- CPU-окно (split-round): 1553 passed / 51 skipped / 0 failed (engine 182, scheduler 391, server 708, kvcache 272/17sk, test_graph 1, test_ple_disk CPU 2, test_offload 23/30sk). test_ggml.py не гонялся (100% GPU-only).
- Review: 0 blocker / 0 major / 2 minor-Info, оба deferred-followup (GPU-тесты -> cloud; HIP/Pascal smpbo note - upstream behavior #878, риг не задет). TP=0.
- Cloud-окно boot-smoke PASS попытка 1: READY 229 s (round-8 ref 224 s), слот-гейт молчит (3 unified pools 26-984), chat-гейт 200 + usage 348/8, FAILPAT 0, shutdown rc 143, VRAM 1287 MiB (baseline 1347). Decode-медиана 20.84 tok/s vs round-8 22.97 - дельта 2.13 чуть за полосой шума ~1.5, зафиксирована как ИНФОРМАЦИОННАЯ без атрибуции (decode-only, инварианты держатся).
- Cloud GPU: 4 failed / 969 passed / 167 skipped (202 s, один прогон). Все 4 = известные baseline (pinned_tensor UVA, qsa_fp8 CUDA arg, 2x dflash bitwise main-baseline 08cee44). Новых падений нет; test_ggml.py целиком + 4 @requires_cuda test_ple_disk отработали зелёными (GPU-долг закрыт). Baseline-классы fa/glm_dsa/test_ple в этом прогоне не сработали (flake/param-зависимые).

## Уроки
- merge-tree --write-tree предчек в discovery точно предсказал 0 конфликтов и дизъюнктность регионов - валидный дефолт для маленьких дельт; review двух проходов в CPU-окне при отложенном boot-smoke легитимен (skip-критерии не выполнены из-за pending smoke, но стат-ревью дёшево).
- Сплит-раунд-протокол отработал штатно 4-й раз (rounds 6-9).

Артефакты: .tasks/merge-main-round9-work/ (discovery-report.md, adaptation-log.md, test-results.md, review-report.md, merge-commit.txt, pending-handoff.md, boot_smoke_r9.sh, boot-smoke-r9.log, boot-smoke-r9-server.log, boot-smoke-r9-vram.csv, gpu-suite-r9.log).

kb: kb/cases/merge-round9-upstream-main/TASK.md (distilled 2026-11-09); merge-tree --write-tree предчек закреплён дефолтом discovery в SKILL.md (Wave 1 п.4, 325->328). Незакоммичено: kb-файлы + SKILL.md - коммит за пользователем.
