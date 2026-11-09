---
title: "Round 8: merge upstream/main 8215a94 в vektory79 - split-round CPU-волна и итоги cloud-окна"
date: 2026-11-09
hardware: "CPU-волна - CPU-only окно (резидентная LLM); cloud-окно 2026-11-09 - свободная VRAM, сериализованно (RTX 5090 32GB)"
commits: "merge fe975da (родители 5a170e8 + 8215a94 = upstream/main); бэкап backup/vektory79-pre-merge8-5a170e8; amend не потребовался"
status: "ЗАКРЫТ: CPU 1653 passed / 67 skipped / 1 baseline-flake, ревью 0 blockers (3 deferred minor), boot-smoke PASS, GPU 954/4/167 - все падения baseline"
tags: [merge, upstream, round8, split-round, boot-smoke, checkout-theirs, timings-chain, constexpr-splice, dflash]
---

# TASK: round 8 - merge upstream/main 8215a94 (28 коммитов) в vektory79

Status: **ЗАКРЫТ** - все волны исполнены, merge-коммит fe975da остался без
амендов, merge-регрессий нет. Процедура волн и уроки раундов -
[merge-wave-procedure](../../topics/merge-wave-procedure.md); прецедент
round 7 - [../merge-round7-upstream-main/TASK.md](../merge-round7-upstream-main/TASK.md).
Полные артефакты волны (discovery / adaptation-log / test-results /
review-report, boot и GPU логи) остались в рабочей папке волны и в kb не
зеркалированы; суть дистиллирована здесь и в процедурной статье.

## 1. Контекст и режим

- upstream/main 8215a94 от базы round 7 (a2ce636): дельта **28 коммитов**.
  `GIT_EDITOR=true git merge upstream/main --no-commit` - **16 конфликтных
  файлов**. Режим - минимальный зелёный merge, SPLIT-ROUND: CPU-часть
  локально при резидентном LLM, boot-smoke и GPU-порции - cloud-окно.
- Инварианты PASS: ровно один merge-коммит над родителями, .veai в коммите
  нет, конфликт-маркеров нет.
- **Инцидент (восстановлен)**: 8 файлов на старте резолва получили
  `git checkout --theirs`, что молча снесло авто-слитую веточную сторону
  (kv-quant backend-валидация engine.py, mmproj plumbing tokenizer/server.py,
  flint-парсер generation.py). Поймано диффом
  `git diff upstream/main 5a170e8 -- <file>`, восстановлено
  `git checkout -m --` (воссоздаёт 3-way конфликт) с повторным ханк-резолвом.
  Второе воспроизведение ловушки после round 6 - урок зафиксирован как
  дефолт в [merge-wave-procedure](../../topics/merge-wave-procedure.md).
- **venv-инцидент**: .venv в чекауте отсутствовал; `uv run` молча создал
  окружение на дефолтном CPython (cp315, у torchvision нет wheel) и упал до
  тестов. Пересоздано явно: `uv venv --python 3.14` +
  `uv pip install -e ".[accel,dev]"`; интерпретатор проверен по
  `freetoken.__file__`.

## 2. Карта резолвов (16 файлов, суть)

Main-wins на уровне ханков; веточные авто-слитые фрагменты сохранены всюду.

- **Таймингс-цепочка (решение no-dual-logic)**: main-словарь per-request
  `timings` (ReqStats: queue/prompt/predicted спаны, opt-in
  --enable-timings-report, llama.cpp-style timings_block) замещает веточную
  wire-цепочку prefill_ms/decode_ms/prompt_warmup (то же назначение,
  подмножество спанов). Wire = только main. Веточные внутренности живы как
  потребители main-механики: RequestTimingTracker, own-window
  batch_forward_ms, warmup-маркировка префил-строки, _decode_window_seconds,
  stream-stall предупреждения fc5fd4a (пере-нанесены на main-wire). Веточный
  wire-хелпер _timings() удалён вместе с 3 emission-сайтами; два
  scheduler-теста перепоказаны на tracker pop.
- **UNION**: scheduler.py drain (веточный 3-tuple last_data + main
  `output[:3]` unpack и verify-ветка _drain_verify + веточные
  aborted_uids/stale_uids), status.py (_decode_window_seconds + main
  _spec_req_steps/_spec_tokens/_spec_counts), kvcache/__init__.py (kv_quant +
  num_speculative_tokens на create и create_kvcache_pool), experts.py (main
  _bias + веточный streamed=), qwen4_exp/common.py (оба параметра),
  docstring test_qsa_backend.
- **Constexpr-сплайс** triton/attention.py: HAS_KV_SCALE + KV_NVFP4 (ветка) +
  KV_OFFSET_DTYPE (main, #854 overflow fix) на обеих конфликтных ядрах - 4
  сигнатуры и 4 launch-сайта как единое целое; веточная 3-path структура
  load (nvfp4/fp8-scale/plain) сохранена, main `.to(KV_OFFSET_DTYPE)` на
  slots load. Сплайс-ловушка round 6 не воспроизвелась (подтверждено
  boot-smoke).
- Прочее: engine.py (main _disable_unsafe_cpu_moe_graphs + веточная
  kv-quant валидация, интегрированная в main-цикл
  _resolve_auto_attention_backend), qwen3_5_moe/gguf.py (main _iter_block/
  _untile + удалён осиротелый import untile), tokenizer/server.py (main
  timings forwarding + веточный mmproj_path/get_mm_processor).

## 3. CPU-батарея: 1653 passed / 67 skipped / 1 failed

CUDA_VISIBLE_DEVICES="", сериализованно, пофайлово под hard timeout.
Единственный failed - test_ple bitwise flake (известный baseline, снимок
torch.equal на float-пути). 6 падений первого прогона оттриажены и
исправлены внутри merge-коммита:

- test_deprecated_flag_hints: main #851/#866 добавил w4a4-кандидат раньше
  moe.nvfp4=triton в candidates - замеренная строка хинта обновлена на
  `moe.nvfp4=w4a4` (устаревшее веточное ожидание, не регресс);
- 2 abort-теста test_abort_inflight_prefill: веточные ассерты на удалённые
  поля ответа перепоказаны на RequestTimingTracker terminal pop (атрибуция
  и порядок release сохранены), после фикса 18/18 в файле.

## 4. Ревью: 0 blockers, 3 deferred minor findings

Все три - документация/формулировки, фикс в merge-коммит не требуется:

1. scheduler/status.py:196-199,257 - docstring RequestTimingTracker всё ещё
   описывает llama.cpp-style timings block и возвращаемые pop() поля; после
   таймингс-решения wire едет на main ReqStats.timings, все 3 pop call-сайта
   отбрасывают return. Требуется переформулировка ("entry release /
   internal attribution");
2. scheduler/scheduler.py:451,467,712 - tracker.pop() release-only,
   возвращаемое значение нигде не используется (опционально: return None
   либо отдельный release()-метод);
3. adaptation-log волны называет `output[:3]`-unpack "SpecForwardOutput
   NamedTuple"; нормальный путь возвращает ForwardOutput (оба NamedTuple,
   unpack корректен для обоих; формулировка может ввести в заблуждение
   следующий раунд).

Pass A подтвердил эквивалентность UNION-резолвов, сохранность веточных
фиксов (cache.py self._keys, docs/cli.md 'auto', stream-stall fc5fd4a),
согласованность сплайс-тройки и чистоту sweep осиротелых хелперов; Pass B -
пустые конфликт-маркеры, инварианты merge-коммита, отсутствие ссылок
тестов на удалённые wire-поля.

## 5. Cloud-окно (2026-11-09): boot-smoke PASS на первой попытке

- Рецепт round 7 без изменений (gguf-786k nvfp4: --moe-cache-auto
  --kv-reserve-tokens 786432 --kv-cache-dtype nvfp4 --memory-ratio 0.89
  --max-prefill-length 4096 --moe-strategy hybrid --moe-cpu-threads 16,
  порт 18081); runner trap-on-EXIT + FAILPAT-дозорный + tee stdout,
  hard timeout 900 s. Новых config-assert main не внёс - адаптация рецепта
  не потребовалась.
- READY 224 s (референс round 7: 229 s); слот-гейт молчит (3 unified пула
  по 26-1012 слотов); chat-гейт POST с полем model, max_tokens=8 -> HTTP 200
  + usage 348/8/356; radix priming подтверждён (#cached-token: 320).
- Decode медиана (клиент, 3x32): **22.97 tok/s** (21.02/22.97/23.02) против
  референса того же конфига round 7 (21.73 tok/s): дельта +1.24 tok/s
  внутри межбутного шума ~1.5 tok/s - сигнала регрессии нет.
- Shutdown SIGTERM -> rc 143 штатно; VRAM 1012 MiB после против baseline
  1015; census чист, FAILPAT без попаданий. Merge-коммит НЕ амендился,
  HEAD fe975da.

## 6. GPU-порции: 954 passed / 4 failed / 167 skipped

Один сериализованный прогон 481 s, census до/после чист. Сравнение по
FAILURE SET (collection drift), не по тоталам. Все 4 падения - не
merge-регрессии:

- tests/kernels/test_pinned_tensor.py (UVA identity) - известный baseline
  брифа волны;
- tests/kernels/test_qsa_fp8.py [1-1-64] (CUDA invalid argument) -
  известный baseline брифа;
- 2x tests/models/dflash/test_draft.py::test_block_forward_matches_reference
  ([4-dtype0-0.015625], [8-dtype0-0.015625]) - **новый baseline-класс**:
  финальный bitwise restore assert (torch.equal первого forward против
  window-rebuilt ring forward) при проходящем tolerance-сравнении с
  референсом (assert_close atol/rtol) - last-ulp float шум. Файл введён
  main 08cee44 (DFlash-2 #857) и на веточном родителе 5a170e8 отсутствует;
  A/B на detached worktree чистого main-родителя 8215a94 - те же 2
  параметра падают идентично. Вердикт: main baseline на этом риге, не
  damage от merge-композиции.
- Отклонение от брифа (документировано): ведущий `CUDA_VISIBLE_DEVICES=`
  из CPU-хэндоффа снят для GPU-прогона (пустое значение скрывает GPU и не
  дал бы проявиться baseline-классам); путь и флаги - как в брифе,
  добавлен --tb=short.

## 7. Уроки и закрытие

Процедурные уроки round 8 - ханк-резолв как дефолт вместо checkout
--theirs, восстановление .venv в CPU-окне (uv venv --python 3.14),
подтверждение правила constexpr-сплайса, класс "upstream-введённый тест,
ни разу не гонявшийся на риге" и снятие CPU-префикса с GPU-брифа -
[merge-wave-procedure](../../topics/merge-wave-procedure.md), раунд 8.
Открытые пункты: 3 deferred findings из ревью - фикс отдельным коммитом
только по явной просьбе пользователя; kb-файлы и артефакты волны
незакоммичены - коммит за пользователем.
