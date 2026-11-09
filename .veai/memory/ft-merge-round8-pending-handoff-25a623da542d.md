---
name: "ft-merge-round8-pending-handoff"
description: "Merge round 8 fe975da CLOSED: CPU 1653 passed, boot-smoke 22.97 tok/s, GPU 954/4 baseline, review clean"
type: project
lastUpdated: 2026-11-09T11:29
lastRecall: 2026-11-09T12:33
---

# Merge round 8 (2026-10-27): CLOSED - HEAD fe975da

Merge upstream/main 8215a94 -> vektory79: fe975da (parents 5a170e8 + 8215a94), rev-list count 1, .veai=0, ровно один merge-коммит, amend НЕ потребовался. Base a2ce636, дельта 28 коммитов (DFlash-2 spec-decode, ricursive/ripple-65b, gpt-oss GGUF, ggml MMQ counting sort, llama.cpp pin вместо freetoken.patch 60059b4, SWA prefill #651, сервер usage/timings). Push не делался (origin/vektory79 отстаёт на 29). Backup: backup/vektory79-pre-merge8-5a170e8. Артефакты: .tasks/merge-main-round8-work/ (merge-brief.md, cloud-window-brief.md, discovery-report.md, adaptation-log.md, test-results.md, review-report.md, boot-smoke-r8.log, gpu-suite-r8.log, boot_smoke_r8.sh). kb: cases/merge-round8-upstream-main/TASK.md + раунд 8 в topics/merge-wave-procedure.md; SKILL.md фолд 316->325. kb/SKILL.md файлы незакоммичены - коммит за пользователем.

## Верификация
- CPU (split-round): 1653 passed / 67 skipped / 1 baseline-флейк (test_ple bitwise). 6 фейлов первого прогона исправлены в merge-коммите (flag_hints -> moe.nvfp4=w4a4 по main #851/#866; 2 abort-теста репоинтированы на tracker-pop).
- Review: 0 blockers / 0 majors / 3 minor-Info, все deferred-followup (фикс только по явной просьбе): (1) status.py:196-199 docstring RequestTimingTracker обещает llama.cpp-style timings block при живом main ReqStats.timings; (2) scheduler.py:451,467,712 tracker.pop() release-only; (3) adaptation-log item 7: output[:3] это ForwardOutput, не SpecForwardOutput.
- Cloud-окно boot-smoke PASS с попытки 1: READY 224 s (round-7 ref 229 s), chat-гейт 200 + usage 348/8/356, слот-гейт молчит (3 унифицированных пула 26-1012 слотов), radix priming подтверждён, decode-медиана 22.97 tok/s vs round-7 21.73 (+1.24, внутри шума ~1.5 - сигнала нет), shutdown rc 143, VRAM 1012 MiB (baseline 1015), FAILPAT 0 попаданий.
- Cloud-окно GPU: 954 passed / 4 failed / 167 skipped (8 min 01 s, один сериализованный прогон). Все 4 НЕ merge-регрессии: pinned_tensor UVA и qsa_fp8 CUDA arg - известные baseline; 2x dflash test_draft bitwise window-restore assert (line 216, [4/8-dtype0]) - НОВЫЙ main-baseline класс: файл введён main 08cee44 (DFlash-2 #857), на веточном родителе 5a170e8 отсутствует, A/B worktree на 8215a94 воспроизводит те же 2 фейла. bf16-параметр зелёный на обеих сторонах.

## Ключевые резолвы
16 конфликтов: UNION scheduler.py drain (3-tuple + verify-дрейн, aborted/stale_uids сохранены), status.py, triton/attention.py (constexpr тройка HAS_KV_SCALE/KV_NVFP4/KV_OFFSET_DTYPE согласована на 4 сигнатурах и 4 launch-сайтах), kvcache/__init__, experts.py (_bias+streamed), qwen4_exp/common.py, test_qsa_backend. Main-wins: timings-цепочка ветки снята как дубль main ReqStats.timings+timings_block (no-dual-logic), веточные дельты перенесены (stream-stall fc5fd4a на openai_api.py:274-438, RequestTimingTracker, own-window forward_ms/warmup); engine.py, message/*, tokenizer/server.py, generation.py. Веточные фиксы пережили: cache.py:664-667 2bb45b2, cli.md 'auto' df40fea. Осиротелых хелперов нет.

## Инциденты/уроки
- checkout --theirs на 8 файлах снёс бы веточные авто-слитые фичи (ловушка round 6 ВОСПРОИЗВЕЛАСЬ второй раз); поймано, восстановлено git checkout -m, резолв по ханкам. Урок: ханк-резолв дефолт с первого хода.
- .venv отсутствовал: uv venv --python 3.14 (дефолт uv тянет cp315 без wheel torchvision).
- Чужой ft serve FreeTokenProd pid 464693 в census - не тронут.
- GPU-бриф из CPU-хэндоффа тащил ведущий CUDA_VISIBLE_DEVICES= (копипаст CPU-окна) - снять при переносе в cloud-окно.
- Upstream-введённый тест, ни разу не гонявшийся на риге = отдельный baseline-класс; дискриминатор - A/B против MAIN-родителя, не веточного.
