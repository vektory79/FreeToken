---
title: "Round 9: merge upstream/main 5a73685 в vektory79 - чистый автомердж трёх коммитов, merge-tree предчек, закрытие split-round"
date: 2026-11-09
hardware: "CPU-волна - CPU-only окно (резидентная LLM); cloud-окно 2026-11-09 - свободная VRAM, сериализованно (RTX 5090 32GB)"
commits: "merge b79457e (родители 7154bab + 5a73685 = upstream/main); бэкап backup/vektory79-pre-merge9-7154bab; amend не потребовался"
status: "ЗАКРЫТ: CPU 1553 passed / 51 skipped / 0 failed, ревью 0 blockers (2 deferred minor), boot-smoke PASS с попытки 1 (READY 229 s), GPU 969/4/167 - все падения baseline"
tags: [merge, upstream, round9, split-round, boot-smoke, merge-tree, main-wins, graph-capture, ggml]
---

# TASK: round 9 - merge upstream/main 5a73685 (3 коммита) в vektory79

Status: **ЗАКРЫТ** - все волны исполнены, merge-коммит b79457e остался без
амендов, merge-регрессий нет; STOP GATE закрыт в cloud-окне 2026-11-09.
Процедура волн и уроки раундов -
[merge-wave-procedure](../../topics/merge-wave-procedure.md); прецедент
round 8 - [../merge-round8-upstream-main/TASK.md](../merge-round8-upstream-main/TASK.md).
Полные артефакты волны (discovery / adaptation-log / test-results /
review-report, boot и GPU логи) остались в рабочей папке волны и в kb не
зеркалированы; суть дистиллирована здесь и в процедурной статье.

## 1. Контекст и режим

- upstream/main 5a73685 от базы round 8 (8215a94 - второй родитель merge
  fe975da): дельта **3 коммита**, 8 файлов +70/-86:
  - 340db81 fix(engine) **#877**: KV-записи graph-capture уходят на dummy
    page (engine/graph.py + тесты test_graph/test_ple_disk/test_offload и
    tests/README.md);
  - ef6b303 fix(ggml) **#878**: cap shared memory MoE route sort на device
    limit (kernel/csrc/ggml/ggml-cuda/mmid.cu + tests/kernels/test_ggml.py);
  - 5a73685 chore(release): 0.2.2 (version.py).
- Режим - SPLIT-ROUND (резидентная LLM): merge и CPU-тесты локально при
  CUDA_VISIBLE_DEVICES="", boot-smoke и GPU-порции - cloud-окно.
- Pre-merge HEAD 7154bab = e9bc7a0 + 2 пользовательских коммита, трогающих
  только .veai/* - приняты как первый родитель (протокол, не откатывать).
- Инварианты PASS: rev-list count == 1 над родителями; .veai в коммите нет;
  конфликт-маркеров нет; dirty до merge - только .veai/memory/*; stash пуст;
  staged diff == дельта upstream (8 файлов, +70/-86).

## 2. Discovery: merge-tree предчек совпал с фактом

- Конфликт-кандидаты (пересечение файлов дельты с веточной стороной от
  базы): только engine/graph.py и tests/moe/test_offload.py; остальные 6
  файлов дельты ветка поверх базы не трогала - конфликтов нет по определению.
- Пробный `git merge-tree --write-tree HEAD upstream/main`: rc=0, список
  конфликтов пуст. Автомердж подтвердил прогноз 1:1: rc=0, **0 конфликтов**,
  ханк-резолвы не понадобились, checkout --theirs не применялся. Предчек
  закреплён дефолтом discovery-шага в
  [merge-wave-procedure](../../topics/merge-wave-procedure.md), раунд 9.
- Зоны риска: mmid.cu (ggml CUDA-кернел) + graph.py (CUDA graph capture) ->
  boot-smoke обязателен; зоны (a)-(d) чек-листа не затронуты.

## 3. Семантическая проверка авто-юнионов (2 файла пересечения)

Ноль текстовых конфликтов не отменяет семантическую верификацию:

- **engine/graph.py**: веточная дельта f801a08 (маршрутизация gguf
  expert-банков через оффлоад) жива после слияния - list-поле
  moe_offload_caches (graph.py:132-164), _reset_moe_offload_cache (:179) и
  все call-sites (:220/:249/:255). Upstream #877 меняет дизъюнктные регионы
  (GraphCaptureBuffer.init +dummy_loc :37-43, call-site dummy page в
  _capture_graphs); обе механики зовут один и тот же reset-хук - композиция
  без fork; construction sites проверены (engine.py, spec/mtp/proposer.py).
- **tests/moe/test_offload.py**: upstream #877 удалил 77-строчный тест
  warm-capture; ветка этот регион не трогала (393b682 - адаптация hybrid moe
  к slot pools - добавляла свой slot-pool тест в другом месте файла) -
  чистый main-wins. Замещающее покрытие живо и сильнее:
  tests/engine/test_graph.py::
  test_capture_runs_on_the_dummy_page_with_a_warm_offload_cache проверяет
  обе половины старой семантики (warm offload cache через eager -> capture)
  плюс новый out_loc dummy page - строгое надмножество семантики
  (no-dual-logic). Ревью перезапустило: 1 passed + 23 passed / 30 skipped.
- **mmid.cu**: ветка не трогала ни разу на 8215a94..7154bab -> файл == версия
  upstream дословно; #878 меняет только host-прелюдию лаунчера (max_shared =
  smpbo - warp_sums, assert nbytes_shared <= max_shared,
  CUDA_SET_SHARED_MEMORY_LIMIT на все 4 кернела, mmid.cu:207-225); сигнатуры
  и call-site mmq.cu:327 не тронуты; сплайс-ловушка сигнатур N/A.

## 4. CPU-батарея: 1553 passed / 51 skipped / 0 failed

CUDA_VISIBLE_DEVICES="", строго `uv run --extra dev python -m pytest`,
сериализованно, каждый прогон под hard timeout; интерпретатор проверен по
freetoken.__file__. Ноль падений во всём окне - триаж не понадобился.

- Тесты дельты main: tests/engine/test_graph.py 1 passed; CPU-подмножество
  tests/models/qwen4_exp/test_ple_disk.py 2 passed (4 @requires_cuda -
  GPU-only, cloud-окно); tests/moe/test_offload.py 23 passed / 30 skipped
  (Environment-класс); tests/kernels/test_ggml.py - 100% GPU-only, не
  гонялся вовсе (отложен с явной классификацией).
- Пин-сеты ветки: tests/engine 182/4, tests/scheduler 391, tests/server 708,
  tests/kvcache 272/17 - все скип-классы Environment или известные deferred.
- Import smoke: версия 0.2.2 подъехала (бамп 5a73685).

## 5. Ревью: 0 blockers, 2 deferred minor

Static reading + git archaeology (VRAM-протокол: без GPU-прогонов); PASS A
подтвердил: upstream-файлы легли bit-for-bit (git diff HEAD 5a73685 пуст),
combined diff двух файлов пересечения == ручной union, дубликатов и двойной
логики нет. Deferred minor:

1. tests/kernels/test_ggml.py: новые строки параметризации n_experts=12300
   (тот самый случай >48 KiB shared memory, мотивировавший #878) GPU-only -
   до cloud-окна кернел-фикс не верифицирован на железе (закрыто в
   cloud-окне: зелёный);
2. mmid.cu:208-215 - на HIP макрос CUDA_SET_SHARED_MEMORY_LIMIT no-op, а на
   устройствах с smpbo 48 KiB (Pascal-класс) новый случай 12300 экспертов
   упал бы по GGML_ASSERT: замечание к поведению upstream, риг не затронут,
   не merge-регрессия.

## 6. Cloud-окно (2026-11-09): boot-smoke PASS с первой попытки

- Рецепт round 8 без изменений (gguf-786k nvfp4: --moe-cache-auto
  --kv-reserve-tokens 786432 --kv-cache-dtype nvfp4 --memory-ratio 0.89
  --max-prefill-length 4096 --moe-strategy hybrid --moe-cpu-threads 16,
  порт 18081); отклонений от рецепта нет, семантических фиксов нет.
- READY 229 s (референс round 8: 224 s); slot-gate молчит (3 unified пула
  по 26-984 слотов); warm gate POST с полем model -> HTTP 200 + usage 348/8;
  FAILPAT 0 попаданий; shutdown SIGTERM rc 143 штатно; VRAM 1287 MiB против
  baseline 1347; цензура процессов чистая.
- Decode медиана (клиент, 3x32): **20.84 tok/s** (18.56/22.48/20.84) против
  референса round 8 22.97 tok/s на том же конфиге - дельта 2.13 tok/s, чуть
  выше межбутного шума ~1.5 tok/s. Decode-only информационная метрика: все
  инварианты держатся, #877/#878 не на decode hot-пути; записано как
  noise-adjacent без атрибуции, сигнала регрессии нет. Merge-коммит НЕ
  амендился, HEAD b79457e.

## 7. GPU-порции: 969 passed / 4 failed / 167 skipped

Один сериализованный прогон 202 s; сравнение по FAILURE SET, не по тоталам.
Все 4 падения - известный baseline:

- tests/kernels/test_pinned_tensor.py (UVA identity) - baseline брифа;
- tests/kernels/test_qsa_fp8.py [1-1-64] (CUDA invalid argument) - baseline;
- 2x tests/models/dflash/test_draft.py::test_block_forward_matches_reference
  ([4/8-dtype0-0.015625], bitwise window-restore) - main-baseline с 08cee44
  (DFlash-2 #857; класс разобран A/B на main-родителе в round 8).

Flake/параметр-зависимые baseline-классы (fa capture_rows, glm_dsa
OutOfResources x2, test_ple bitwise) в этом прогоне не сработали - failure
set строгим подмножеством baseline, новых необъяснённых падений нет.
Deferred GPU-покрытие CPU-окна закрыто: test_ggml.py зелёный (включая новые
строки n_experts=12300 #878), все 4 @requires_cuda теста test_ple_disk.py
зелёные.

## 8. Уроки и закрытие

- **merge-tree --write-tree предчек - дефолт discovery**: точный прогноз
  числа конфликтов и дизъюнктности регионов калибрует объём Wave 2 до её
  старта (validated round 9) -
  [merge-wave-procedure](../../topics/merge-wave-procedure.md), раунд 9.
- **Малая дельта не отменяет ни boot-smoke, ни семантическую проверку**:
  3 коммита, но кернел + graph capture -> обязательный boot-smoke; оба файла
  пересечения проверены чтением; main-wins по удалённому upstream-ом тесту -
  только с проверкой, что замещающее покрытие живо и сильнее.
- STOP GATE закрыт: CPU зелёный, ревью оттриажировано, boot-smoke PASS,
  GPU failure set == baseline. Push не выполнялся (за пользователем);
  kb-файлы и правка SKILL.md незакоммичены - коммит за пользователем.
