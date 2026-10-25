---
name: "ft-merge-round6-pending-handoff"
description: "Round 6 merge CLOSED: d66f6e3 green, boot-smoke 22.21 tok/s; port debt D2/D4/D5 kb/cases/round6-deferred-ports"
type: project
lastUpdated: 2026-10-25T01:37
lastRecall: 2026-10-25T01:38
---

# Round 6 merge (2026-10-24/25): CLOSED - CPU + cloud-window verified, HEAD d66f6e3

Merge round 6: upstream/main 32056cf -> vektory79, merge 594934d -> amend 220b79e -> amend d66f6e3 (parents 23c1f02+32056cf, rev-list count 1). Backup: backup/vektory79-pre-merge6-23c1f02. Push не делался. Артефакты: .tasks/merge-main-round6-work/ (discovery/adaptation-log/test-results/review-report, boot-smoke-r6*.log, boot_smoke_r6.sh); kb/cases/merge-round6-upstream-main/TASK.md.

## Исполнено (CPU-окно + cloud-окно 2026-10-25)
- CPU: 5 TP-findings ревью закрыты; GPU-порции: 411 passed / 2 failed (glm_dsa OutOfResources - baseline с 09-12) / 33 skipped.
- Boot-smoke PASS (попытка 2): матричный дефолт 0.82/4095 fp8, reserve 400k, hybrid, FTW GLM-5.3-Flash-UD-Q3_K_XL-FTW-vision-new; READY 51 s, decode медиана 22.21 tok/s (верх полосы матрицы 14.3-21.2 - регрессии нет), слот-гейт молчит, VRAM к baseline 976/31226. Попытка 1 упала: NameError has_nvfp4 (glm_dsa_sparse.py:459) - невалидный сплайс мерджа: сигнатура main (out/strides) + веточные call-sites (scale_pool/block_pool). Фикс: веточный квант-прелюд + main out/reach/strides; в dsa.py::_attend возвращены квант-аргументы -> amend d66f6e3.

## Ловушка-урок
Сплайс "сигнатура main + веточные call-sites" в CUDA-ядре ловится ТОЛЬКО boot-smoke (захват CUDA-графа decode): CPU-сюиты зелёные. При merge ядер - отдельно сверять сигнатуры вызываемых ядер между сторонами.

## Порт-долг round 6 (отдельная задача)
Задача: kb/cases/round6-deferred-ports/TASK.md. D2 (mmproj vision, main-wins #803/#810): 44 маркера D-PORT-D2 test_glm5_next_gguf + 4 D-PORT-1 test_ftw_gguf_banks (gguf->FTW vision packing) + RuntimeError weight.py:263 + MMProj smoke #810. D4 (loader contract): 37 маркеров D-PORT-D4 (ftw_tensor_types уже возвращён в F1). D5 (nvfp4-KV generic, ветка 660c814): + recalibrate 2 stale-pin test_quant_config (test_hybrid_explicit_kernel_error_names_the_usable_alternative, test_nvfp4_auto_selection_still_lands_on_triton vs #737); D5 нужен для возврата nvfp4-KV 786k стека (эталон 27.71 tok/s). D3: честный A/B только после D5 (сейчас fp8-KV 22.21 vs nvfp4 27.71 несравнимы). Не-долг: test_gguf_expert_banks module-skip (round-5), glm_dsa OOR baseline.

**How to apply:** порт-долг исполнять фазами по TASK.md; skip-маркеры D-PORT снимать только при исполнении соответствующей фазы.
