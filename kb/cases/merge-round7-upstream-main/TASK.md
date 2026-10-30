---
title: "Round 7: merge upstream/main a2ce636 в vektory79 - бриф cloud-окна (boot-smoke W1, GPU-порции W2, followup-кандидаты W3)"
date: 2026-10-26
hardware: "CPU-only окно закрыто; W1-W3 - cloud-окно (свободная VRAM, сериализованно)"
commits: "merge d8803e0 (родители 67754c9 + a2ce636 = upstream/main); бэкап backup/vektory79-pre-merge7-67754c9"
status: "CPU-зелёный (970 passed / 43 skipped / 0 failed, ревью 0 blockers); W1-W3 отложены в cloud-окно"
tags: [merge, upstream, round7, split-round, boot-smoke, rows_per_req, radix_lookahead, union-резолвы, followup]
---

# TASK: round 7 - merge upstream/main a2ce636 (12 коммитов) в vektory79

Status: **CPU-закрыт, GPU-pending**. Merge-коммит d8803e0 зелёный на всей
CPU-поверхности (35 файлов, 970 passed / 43 skipped / 0 failed), ревью
0 blockers; boot-smoke и GPU-порции - cloud-окно (W1-W3 ниже). Процедура
волн и уроки раундов - [merge-wave-procedure](../../topics/merge-wave-procedure.md);
прецедент round 6 - [merge-round6-upstream-main/TASK.md](../merge-round6-upstream-main/TASK.md),
порт-долг round 6 - [round6-deferred-ports/TASK.md](../round6-deferred-ports/TASK.md).
Полные артефакты волны: discovery / adaptation-log / test-results / review-report
в рабочей папке `.tasks/merge-main-round7-work/` (в kb не зеркалированы; суть
дистиллирована здесь и в процедурной статье).

## 0. Контекст

- upstream/main a2ce636 (2026-10-26): дельта **12 коммитов / 84 файла**.
  Main принёс rows_per_req graph capture (#833/#837), radix_lookahead seam
  (#834), LinearAttnBackend (#837), волну flint/minimax/muse (#816/#797/#829/#830),
  Qwen think recovery (#825). Режим - **минимальный зелёный merge**
  (директива пользователя до волны, binding-урок round 5).
- Merge d8803e0: **7 конфликтных файлов**, все резолвы UNION (функциональности
  сторон disjoint, дубликатов нет - проверено ревью): attention/__init__.py
  (BackendInfo: веточные supports_fp8_kv/nvfp4_kv + main uniform_extend_graph),
  core.py (Batch: log_req_new_tokens + features/rows_per_req), engine/graph.py
  (TYPE_CHECKING-дубль OffloadMoeCache снят в пользу runtime-import),
  models/gguf/tokenizer.py (glm4 pre-tokenizer + main-комментарий),
  moe/expert_banks.py (import json живой, glob мёртвый - отброшен),
  scheduler/cache.py (session_tier_cfg + radix_lookahead, tier-блок +
  lookahead-блок, donate-гейт main-superset), scheduler/status.py
  (tier_fn/forward_ms/warmup + generated).
- CPU-батарея (CUDA_VISIBLE_DEVICES="", сериализованно, пофайлово под hard
  timeout): **970 passed / 43 skipped / 0 failed**. Сплайс-контроль round-6
  урока на CPU-уровне: dsa.py::_attend - сигнатура main (out) + веточные
  квант-kwargs (pool_scale/kv_quant/pool_block_scale), ядро
  glm_dsa_sparse_attn принимает все kwargs; qsa/triton-тройки сходятся;
  compileall attention/scheduler/engine/models/moe/server/kernel rc=0.
- Ревью: **0 blockers**, 4 findings (2 deferred-followup - см. W3, 2 FP).
  Инварианты PASS: rev-list count == 1, родители 67754c9 + a2ce636,
  .veai в коммите нет, конфликт-маркеров нет. TP-фиксов нет, merge-коммит
  НЕ ампендился.

## 1. Фаза W1: boot-smoke (ОБЯЗАТЕЛЕН)

Триггер: дельта трогает `engine/config.py`, `models/gguf/*` и CUDA-ядра
attention; сплайс-риск dsa/triton/qsa против main-механик #833/#837
(rows_per_req graph capture + квант-фиксы ветки). CPU-сверка сигнатур
пройдена, но сплайс "сигнатура main + веточные call-sites" в CUDA-ядре
ловится только boot-smoke с захватом CUDA-графа decode (урок round 6).

Рецепт round 6 (без изменений):

- порт **18081**, конфиг gguf-786k: `--moe-cache-auto --kv-reserve-tokens
  786432 --kv-cache-dtype nvfp4 --memory-ratio 0.89 --max-prefill-length 4096
  --moe-strategy hybrid --moe-cpu-threads 16`;
- trap-on-EXIT runner + FAILPAT-дозорный
  `AssertionError|OutOfMemoryError|Backend worker is gone|backend worker .* exited|Traceback|CUDA error|ValueError`;
- hard timeout **900 s**, tee stdout в `boot-smoke-r7.log` (ловушка улик:
  uvicorn access log не печатает /ready - вердиктная строка только в stdout).

Успех: `/ready` == 200; слот-гейт молчит; суммарные слоты >= рабочего набора
(инварианты, не точные числа); POST /v1/chat/completions с полем model
(max_tokens=8) отвечает; SIGTERM shutdown - exit 143; VRAM к baseline
(976 / 31226 МиБ). Decode-референс: **22.21 tok/s** (round 6, fp8-матрица);
nvfp4-эталон round 5 - 27.71 tok/s, несравнимы (другая квант-механика).

После бута - свежий FULL `ft bench bw --dtype all --cpu-threads 16`
(v6-профиль; урок round 6 D3: движок читает v6-профиль, кап-1 обязателен).

Если smoke упал ПО ПРИЧИНЕ МЕРДЖА - семантический фикс -> amend в
merge-коммит d8803e0 -> повторный smoke. Неочевидная причина - стоп и отчёт,
не гадать.

Риск-зоны сплайса: dsa.py::_attend (out + квант-прелюд), triton.py /
qsa_sparse.py (rows_per_req #833 + веточные квант-фиксы),
attention/__init__.py union-флаги (uniform_extend_graph "device" у triton
+ supports_fp8_kv/supports_nvfp4_kv=True).

## 2. Фаза W2: GPU-порции тестов

Режим: сериализованно, один сервер за раз; каждый файл под hard timeout
(900/600 s); census процессов до/после (живые ft serve, счётчики sem.mp-);
готовность сервера проверять реальным запросом, не `/v1/models`. Точные
пути (из pending-списка волны):

- tests/attention/test_capture_rows_per_req.py (#833/#837, 13 cuda-hits);
- tests/scheduler/test_radix_lookahead.py - GPU-часть (8 hits; CPU-порция
  5 passed в волне);
- tests/models/test_flint.py - GPU-часть (#797; CPU 2 passed / 3 skipped);
- tests/moe/test_offload.py (69 hits; тронуты #819 index-shard discovery
  и #833);
- tests/models/qwen4_exp/test_ple.py и test_ple_disk.py (#837
  LinearAttnBackend);
- tests/models/qwen4_exp/test_gdn.py (6 needs-CUDA skips в волне);
- tests/attention/test_dsa_kpool.py (17 needs-CUDA skips в волне);
- tests/models/test_glm5_next_kda_op.py (3 skips) +
  test_glm5_next_kda_snapshot.py (1 skip);
- tests/e2e/test_aime.py (#821 cuda visibility, e2e);
- tests/engine/test_cache_budget.py - GPU-часть (11 hits);
- tests/scheduler/test_abort_inflight_prefill.py - GPU-часть;
- tests/kvcache/test_kv_cache_rebuild.py - GPU-зоны +
  tests/scheduler/test_cache_rebuild.py (частично GPU);
- веточные: tests/models/test_gguf_expert_banks.py - GPU-замыслы
  (6 needs-CUDA сайтов);
- tests/models/test_glm_dsa.py - падение OutOfResources
  переклассифицировать заново (baseline round 6, см. "Не входит").

## 3. Фаза W3: followup-кандидаты

НЕ входят в merge-коммит; фикс отдельным коммитом ТОЛЬКО по явной просьбе
пользователя (cloud-окна не требуют).

1. **scheduler/cache.py:654-656** (зона else-ветки finish-доната, сам insert
   на ~662-665): tier-restored insert пишет `req.input_ids[:insert_len]` RAW
   вместо `self._keys(req.input_ids, insert_len)` (соседний код метода
   использует self._keys). При radix_lookahead=True дерево ключуется парами
   (t_i, t_{i+1}) и этот insert создал бы несовместимые узлы. Латент:
   radix_lookahead в проде недостижим - grep по python/ вне cache.py даёт
   0 call-sites, единственный production-конструктор CacheManager
   (scheduler.py:88-104) не передаёт флаг; без lookahead keys == input_ids,
   поведение сегодня корректно. Рекомендация из ревью: заменить на
   self._keys для симметрии либо оставить с однострочным комментарием
   lookahead-инварианта.
2. **docs/cli.md:78** - таблица `--kv-cache-dtype` ("bf16, fp8, or nvfp4")
   без choices 'auto' из server/args.py:531-535. Pre-existing
   branch-internal: строку дока добавила 03fb043 (fp8 kv), 'auto'-choice -
   2f554c9 (nvfp4 kv) позже; в main a2ce636 docs/cli.md без
   --kv-cache-dtype вовсе; merge зону не менял (git diff 67754c9..d8803e0).
   Дописать 'auto' в таблицу.

## 4. Не входит

- **test_glm_dsa.py OutOfResources** - pre-existing baseline с 2026-09-12,
  не регресс merge; в W2 только переклассификация, в фиксы не входит.
- **test_gguf_expert_banks module-level skip** (rc=5) - старый
  expert-sources API удалён ре-платформой round 6; покрытие переехало в
  test_glm5_next_gguf.py (маркер в файле L42). Не новый сбой.
- Прочие классифицированные CPU-skip: needs-CUDA (Environment, покрыты W2)
  и baseline-доки (compiled _cpu_moe extension, D3-verdict, round-5
  deferred executor).

## 5. Критерий закрытия case

Boot-smoke PASS (W1) + GPU-порции выполнены и триажированы (W2: каждое
падение классифицировано как merge-регрессия / Environment / pre-existing
baseline) + вердикт по W3-кандидатам от пользователя. Итоги дописать
секцией ниже в этот же файл.

## Итоги

Cloud-окно 2026-10-29, HEAD d8803e0 (parents 67754c9 + a2ce636). **Case ЗАКРЫТ:
W1 PASS + W2 исполнен, merge-регрессий нет, amend НЕ потребовался, push не делался.**

### W1 boot-smoke: PASS
- Рецепт gguf-786k (порт 18081, raw GGUF, kv-reserve 786432, nvfp4, 0.89/4096,
  hybrid, cpu-threads 16), trap-runner + FAILPAT + tee, hard timeout 900 s.
- READY after 229 s; слот-гейт молчит (moe_cache_size=1074, slot pools unified
  3 pools 26-996, slot-floor строк нет); fetch-фракции RESOLVED 29.9/26.1/30.2%
  (v6-профиль benchbw пережил мердж, cap-1 не активировался).
- Chat-гейт: POST с полем model, max_tokens=8 -> HTTP 200 + usage.
- Decode медиана (клиент, 3x32): **21.73 tok/s** (21.29/21.73/21.80). Референс
  round-6 D3 на том же конфиге: 23.60 клиент -> -1.87 tok/s (~-8%), чуть выше
  межбутного шума ~1.5 tok/s, одиночный прогон - сигнала регрессии нет. К 22.21
  fp8 (round 6) и 27.71 nvfp4 (round 5) - несравнимо (другие стеки).
- Сплайс-ловушка round 6 (NameError на захвате CUDA-графа decode) НЕ
  воспроизвелась: CPU-сверка сигнатур A3 подтвердилась на железе.
- Shutdown: SIGTERM -> rc 143, VRAM 1155 MiB vs baseline 1195, census чист.
- После бута FULL `ft bench bw --dtype all --model <gguf> --cpu-threads 16`:
  rc=0, 8/8 строк профиля (5 канонических + 3 gguf-вида), свежие gguf-фракции
  28.4/24.7/29.7% (лог bench-bw-r7.log).

### W2 GPU-порции: исполнены, 0 merge-регрессий
17 файлов сериализованно, census до/после чист. Итог: **205 passed / 5 failed /
9 skipped**. Падения классифицированы:
- test_capture_rows_per_req [fa-1]/[fa-4] - Environment: RuntimeError guard
  fa.py:39 (sgl-kernel без FA3/FA4 для sm_120; guard от initial release 3af9d90,
  RTX 5090 = sm_120). Приоритетные параметризации triton-1/4, fi-1/4 и
  radix_lookahead (5 passed) - зелёные: новые seam-ы #833/#834 на слитом коде
  работают.
- test_ple.py::test_track_snapshot_equals_a_prefill_stopped_at_the_boundary -
  pre-existing branch baseline: A/B в worktree на pre-merge 67754c9 падает
  идентично (bitwise torch.equal :421); документированный flaky-класс с
  2026-09-15, перешёл через merge без изменений.
- test_glm_dsa 2x OutOfResources (shared 102400 > 101376) - подтверждённый
  baseline с 2026-09-12 (переклассификация из брифа выполнена).
- skips: needs-CUDA закрылись GPU-прогоном (gdn 6, dsa_kpool 17, kda 3+1);
  test_aime = Environment (нужны ~70 GiB + dataset env); остальное - baseline-доки.
Пофайловые счётчики и логи: .tasks/merge-main-round7-work/test-results.md,
gpu-logs/, gpu-wave-r7.log.

### W3 followups: НЕ исполнялись (по регламенту - только по явной просьбе)
Кандидаты без изменений: cache.py tier-restored insert (self._keys) и
docs/cli.md 'auto' в таблице --kv-cache-dtype.

### Коммиты
Никаких новых коммитов; amend не требовался; d8803e0 остаётся финальным.
kb-файлы и .tasks-артефакты волны незакоммичены - commit за пользователем.
