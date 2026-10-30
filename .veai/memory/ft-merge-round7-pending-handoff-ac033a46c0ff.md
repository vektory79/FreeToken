---
name: "ft-merge-round7-pending-handoff"
description: "Round 7 merge d8803e0 CLOSED: boot-smoke PASS 21.73 tok/s, GPU 205/5/9 all Environment/baseline, 0 merge regressions"
type: project
lastUpdated: 2026-10-29T19:38
lastRecall: 2026-10-29T19:32
---

# Round 7 merge (2026-10-26/27): CLOSED - HEAD d8803e0

Merge a2ce636 (origin/main == upstream/main, +12 коммитов: #837 LinearAttnBackend, #833 multi-row decode seams, #834 radix_lookahead, mmproj-vision GGUF модели, #797 flint) -> vektory79: merge d8803e0 (parents 67754c9 + a2ce636, rev-list count 1, .veai=0), amend НЕ потребовался. Push не делался (origin/vektory79 отстаёт на 13). Backup: backup/vektory79-pre-merge7-67754c9. Артефакты: .tasks/merge-main-round7-work/ (discovery/adaptation-log/test-results/review-report, boot-smoke-r7.log, gpu-logs/, boot_smoke_r7.sh, gpu_wave_r7.sh); kb/cases/merge-round7-upstream-main/TASK.md (итоги дописаны).

## Верификация
- CPU (split-round, CUDA_VISIBLE_DEVICES=""): 970 passed / 43 skipped / 0 failed (35 файлов). Skips = needs-CUDA (Environment) + module-skip test_gguf_expert_banks (старый API удалён).
- Review: 0 blockers, 4 findings -> triage: TP-фиксов нет. Deferred: (1) cache.py:654-656 tier-restored insert raw req.input_ids вместо self._keys() - латент (radix_lookahead: 0 прод-call-sites, прод-конструктор scheduler.py:88-104 флаг не передаёт); (2) docs/cli.md:78 --kv-cache-dtype без 'auto' - pre-existing веточный. Фикс только по явной просьбе пользователя.
- Cloud-окно W1 boot-smoke PASS (gguf-786k nvfp4, порт 18081): READY 229 s, слот-гейт молчит (moe_cache_size=1074, unified 3 pools), chat-гейт 200+usage, decode-медиана 21.73 tok/s (референс round 6 D3 same-config 23.60 -> -8%, внутри межбутного шума ~1.5 tok/s, сигнала нет; к 22.21 fp8 / 27.71 nvfp4 несравнимо). Сплайс-ловушка round 6 НЕ воспроизвелась. Fetch-фракции v6-профиля пережили мердж (RESOLVED ~26-30%). Shutdown rc 143, VRAM 1155 MiB (baseline 1195). FULL bench-bw rc=0, 8/8 строк.
- Cloud-окно W2 GPU: 205 passed / 5 failed / 9 skipped (17 файлов). Все 5 падений - НЕ merge-регрессии: capture_rows fa-1/fa-4 = Environment (guard fa.py:39 "no FA3/FA4 kernels for sm_120" от 3af9d90); test_ple snapshot = pre-existing branch baseline (A/B на 67754c9 идентично); test_glm_dsa 2x OutOfResources = baseline с 09-12. Приоритетные новые seam-тесты #833/#834 зелёные.

## Уроки
Сфолдены в SKILL.md (310->316, без коммита): (1) триаж латентной merge-композиции 3-grep-замером достижимости до fix-vs-defer; (2) decode-референс boot-smoke - только same-config/stack/method, шум ~1.5 tok/s. 4 урока в kb/topics/merge-wave-procedure.md (UNION-послесверка, латентный триаж, дока-vs-код git log -S, осиротелый import). kb-файлы (3) незакоммичены - commit за пользователем.

## W3 followups исполнены (2026-10-27, по явной просьбе пользователя)
- 2bb45b2 fix(scheduler): tier-restored insert в finish-ветке _cache_req_hybrid -> self._keys(req.input_ids, insert_len) (анкор фактический cache.py:664-667; _keys = метод (ids, n) на cache.py:938). Поведение в проде не меняется (radix_lookahead недостижим).
- df40fea docs: docs/cli.md:78 --kv-cache-dtype += 'auto' (правда - args.py:537 choices [auto, bf16, fp8, nvfp4]).
- Тесты: 245 passed / 0 failed (test_session_tier + test_hybrid_cache_manager, CPU). HEAD df40fea; origin/vektory79 отстаёт на 15. kb/TASK.md после пересборки коммитов - untracked (инцидент staged-захвата в первом коммите исправлен reset --soft + явные пути).
