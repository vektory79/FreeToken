---
name: "ft-merge-round6-pending-handoff"
description: "Round-6 merge closed at d66f6e3, boot-smoke PASS; CUDA kernel-splice trap; port debt in round6-deferred-ports-execution"
type: project
lastUpdated: 2026-10-26T02:27
lastRecall: 2026-10-26T12:37
---

# Round 6 merge (2026-10-24/25): CLOSED - HEAD d66f6e3

Merge upstream/main 32056cf -> vektory79: merge 594934d -> amend 220b79e -> amend d66f6e3 (parents 23c1f02+32056cf, rev-list count 1). Backup: backup/vektory79-pre-merge6-23c1f02. Push не делался. Артефакты: .tasks/merge-main-round6-work/ (discovery/adaptation-log/test-results/review-report, boot-smoke-r6*.log, boot_smoke_r6.sh); kb/cases/merge-round6-upstream-main/TASK.md.

## Верификация
- CPU: 5 TP-findings ревью закрыты; GPU-порции: 411 passed / 2 failed (glm_dsa OutOfResources - baseline с 09-12) / 33 skipped.
- Boot-smoke PASS (попытка 2): матричный дефолт 0.82/4095 fp8, reserve 400k, hybrid, FTW GLM-5.3-Flash-UD-Q3_K_XL-FTW-vision-new; READY 51 s, decode медиана 22.21 tok/s (верх полосы матрицы 14.3-21.2 - регрессии нет), слот-гейт молчит, VRAM к baseline 976/31226. Попытка 1 упала: NameError has_nvfp4 (glm_dsa_sparse.py:459) - невалидный сплайс мерджа (сигнатура main out/strides + веточные call-sites scale_pool/block_pool). Фикс: веточный квант-прелюд + main out/reach/strides; в dsa.py::_attend возвращены квант-аргументы -> amend d66f6e3.

## Ловушка-урок
Сплайс "сигнатура main + веточные call-sites" в CUDA-ядре ловится ТОЛЬКО boot-smoke (захват CUDA-графа decode): CPU-сюиты зелёные. При merge ядер - отдельно сверять сигнатуры вызываемых ядер между сторонами.

## Порт-долг round 6 - исполнен отдельным кейсом
CPU-объём D2 (mmproj vision), D4 (loader contract) и D5 (nvfp4-KV; посылка опровергнута - read-path пережил merge, ампутирован только GPU-деквант) исполнен и закоммичен 2026-10-25. Коммиты, карты маркеров и уроки - в памяти round6-deferred-ports-execution, НЕ дублировать здесь. Остался только GPU-объём (cloud-окно): D3 честный A/B nvfp4-KV 786k vs 22.21 (fp8) vs 27.71 (nvfp4 round-5 эталон); D2 MMProj smoke #810 + verify 347/347; D4 boot-smoke raw GGUF + кросс-валидация 38/38; D5 GPU kernel-батареи (test_kv_nvfp4, test_triton_attention, test_kv_fp8). Скилл-файлы (TRAPS T92-T95) незакоммичены - user-gated.
