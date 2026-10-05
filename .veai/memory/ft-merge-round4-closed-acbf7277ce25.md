---
name: "ft-merge-round4-closed"
description: "Round 4 aa0e71c fully CLOSED: boot-smoke PASS, GPU 103/0, F1 fixed by 9ac8830 (guard in load_ftw_banks_to_device)"
type: project
lastUpdated: 2026-10-05T21:54
lastRecall: 2026-10-05T22:08
---

# Round 4 merge aa0e71c - отложенное cloud-LLM окно (актуально до исполнения)

Merge aa0e71c (parents 41d3581 + 581fdca=upstream/main) влит 2026-10-05 при занятой VRAM
(работала локальная LLM). CPU-покрытие сделано (286 passed / 0 новых failed); в
cloud-LLM окне (пользователь переключился на cloud LLM, VRAM свободна) ОБЯЗАТЕЛЬНО:

1. Boot-smoke gguf-конфигурации (триггер сработал: задет engine/config.py #573
   hf-overrides + rotary table_positions, frozen-shim зона). Рецепт: port 18081,
   .venv/bin/ft serve, конфиг gguf-786k из памяти (serving-gguf-tuning),
   trap-on-EXIT runner + FAILPAT-часовой, hard timeout 900s, stdout runner-а tee в
   артефакт (без tee нет прямого доказательства /ready 200). Успех: POST
   /v1/chat/completions с полем "model", SIGTERM shutdown, VRAM к baseline.
2. GPU-порции тестов (падали/скипались как Environment в CPU-окне):
   test_ftw_without_gguf_meta_keeps_legacy_semantics (PinFailed cudaHostRegister),
   2x test_gguf_expert_banks CUDA-тесты, GPU-части qsa_fp8 / qsa_nvfp4 / e4m3 native /
   test_cpu_moe_gguf_iq, test_ftw_gguf_banks gguf_types raise-path (F4), qsa_backend /
   qsa_pool_fp8 (полностью GPU-only).
3. Follow-up F1 (по ЯВНОЙ просьбе пользователя, отдельный коммит - НЕ в merge):
   зеркалировать gguf_types-валидацию в load_ftw_banks_to_device (ftw.py:851-915,
   upstream #601); сегодня недостижим в проде (resident-ветка требует method!=None,
   expert_banks.py:611/:617-618), но граница беззащитна.

Если boot-smoke упадёт ПО ПРИЧИНЕ МЕРДЖА - amend в aa0e71c до push.

**Why:** скилловое окно верификации было сужено CPU-only по решению пользователя.
**How to apply:** любая сессия, где пользователь говорит "переключился на cloud LLM"
и контекст round 4 ещё актуален (проверь git log - если есть новые коммиты поверх,
сверь свежесть).

**ИСПОЛНЕНО 2026-10-05 (cloud-окно):** boot-smoke PASS (/ready 200 @112s, слот-гейт молчит, decode 18.67 tok/s, VRAM ровно к baseline, SIGTERM rc=143); GPU-порции 103 passed / 0 failed (test_ftw_gguf_banks 10, test_gguf_expert_banks 49, qsa_backend 10, qsa_pool_fp8 9, e4m3 5, cpu_moe_gguf_iq 20); PinFailed-тест и gguf_types raise-path PASSED; аменд не потребовался, HEAD остался aa0e71c. ЕДИНСТВЕННЫЙ незакрытый пункт: F1 follow-up - ждёт явной просьбы пользователя. Остальная часть этой памяти - историческая (рецепт boot-smoke остаётся образцом).

**F1 ЗАКРЫТ 2026-10-05, коммит 9ac8830** "fix(checkpoint): guard gguf_types in load_ftw_banks_to_device" поверх aa0e71c (не запушен). Решение: блок валидации из load_ftw_banks (ftw.py:677-703) отзеркален в load_ftw_banks_to_device сразу после _split_bank_entries, ДО device-операций; контракты ошибок те же (re-convert / does not match its config). Populate gguf_types в to_device - НЕТ (resident fused требует method!=None, потребители gguf_types все offload-сторона); решение задокументировано ASCII-комментарием в ftw.py. Тест test_ftw_gguf_types_meta_rejected_to_device (2 параметризованных кейса) в tests/checkpoint/test_ftw_gguf_banks.py: fails-before "2 failed DID NOT RAISE", passes-after файл 12 passed, CPU-исполним (guard до чтений). ROUND 4 ПОЛНОСТЬЮ ЗАКРЫТ: незакрытых пунктов нет, memory можно архивировать при следующей чистке.
