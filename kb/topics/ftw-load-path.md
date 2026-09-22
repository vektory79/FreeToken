---
title: "FTW fast path для GGUF: конвертация и быстрая загрузка банков (задача S)"
date: 2026-09-22
hardware: RTX 5090 32GB
branch-commits: "vektory79; GGUF->FTW fast path (задача S): 8568807 (kind_kernel_for(\"gguf\") -> (None, None) в legacy-format), e9ad38d (gguf_types в мете FTW-индекса и восстановление на загрузке), ef14efb (floor-gate и capability-фоллбеки по FTW-индексу), 2e9fcf1 (имена банков в reject-сообщениях fused-copy-плана)"
status: validated
tags: [ftw, gguf, checkpoint, convert, load, boot, moe, hybrid, signature-groups, silent-none]
---

# FTW fast path для GGUF (задача S)

**Вердикт (читайте это первым).** GGUF-чекпоинт конвертируется в родной формат
FTW и грузится быстрым загрузчиком: бут 52.1 с против 94-132 с на bare-GGUF,
префилл и декод в паритете с bare, качество в паритете. Приземлено: 4 коммита
8568807 / e9ad38d / ef14efb / 2e9fcf1 на vektory79 (закрыты 3 гэпа + one-line
фикс reject-сообщений). Конвертация выполняется ОДИН раз:
`uv run ft checkpoint --model <gguf> --out <dir> --moe-backend offload` -
без `--quant-backend` (репак победил бы raw-стриминг).

## Механизм: что починено

FTW-ридер формат-агностичен: `load_ftw_banks` читает per-layer записи
`{bank}#L{id:05d}` (uint8 `[num_experts, rows, row_bytes]`, ALIGN-4096) в
HostBanks мультипоточно с O_DIRECT, born-pinned аллокацией и pin-as-completed.
Конвертер при этом уже умел стримить GGUF сырыми ggml-блоками без
деквантования. Ломали путь три дыры, закрываемые по отдельности:

- **Гэп 1 - KeyError("gguf") при загрузке.** `kind_kernel_for("gguf")` не имел
  записи. Решение: entry `(None, None)` добавлен в тэг-ключевую
  `_KIND_KERNEL` (`moe/legacy_format.py`); у GGUF-банков kind=None - ядерная
  привязка не нужна. **Ловушка:** `LEGACY_FORMAT` рядом - это ОБРАТНАЯ карта
  `(kind, kernel) -> тэг`; класть запись туда нельзя (первый прогон тестов
  доказал очевидное место неправильным). Регресс-тест извлечён отдельно и
  fails-before: `tests/moe/test_legacy_format.py`.
- **Гэп 2 - gguf_types терялись между конвертером и serve.** Пер-сигнатурные
  партиции и CPU-исполнители требуют список `[[gate, up, down], ...]` на MoE-слой
  (явный порядок ролей). Теперь `convert.py` пишет gguf_types в финализирующую
  мету FTW-индекса; `load_ftw_banks` валидирует структуру (список, длина ==
  число MoE-слоёв, строки из 3 int, ридер закрывается перед raise) и передаёт
  в `ExpertBanks(gguf_types=...)` (default None - nvfp4 не тронут). Движковая
  проводка не менялась: присвоение `cache.gguf_types` не зависит от источника.
- **Гэп 3 + capability - ранний floor-gate и гибридное отклонение не видели
  FTW.** `gguf_ftw_signature_groups` выводит группы сигнатур из шейпов записей
  FTW-индекса (floor-gate отсекает нефандимый план ДО загрузки банков), а
  `gguf_expert_bank_types` с FTW-фоллбеком чинит отклонение hybrid и
  жизнеспособность cpu-исполнителей на FTW-директориях - без этого запланированный
  гибридный бут падал на этапе конфигурации (нашёл оба ревью как Major).
- **Попутный фикс 2e9fcf1:** reject-сообщения `_build_fused_copy_plan` несут
  имя банка (раньше вместо RuntimeError поднимался NameError и маскировал
  причину).

## Контракт silent-None (НЕ считать raising-путём)

Сканы сигнатур в `moe/expert_banks.py` - `gguf_signature_groups` ->
`_gguf_bank_role_stacks`, `gguf_ftw_signature_groups` и `_ftw_gguf_index_meta` -
каждый ловит Exception и возвращает None. Нечитаемый FTW-индекс НЕ роняет
загрузку сырой ошибкой: floor-gate проваливается к поздней сплит-проверке.
Рассуждения о failure-режимах должны считать эти сканы best-effort; оборачивать
их в WeightLoadError нельзя - это противоречило бы семантике main'а (#518:
конфигурационные сбои сохраняют свой тип). Разбор того, как этот контракт
чуть не был «исправлен» вслепую на merge-раунде 3 -
[adaptation-log.md](../cases/merge-main-work/merge-main-round3-work/adaptation-log.md),
раздел о принятом гэпе.

## Числа с конфигурацией

Модель: GLM-5.3-Flash-UD-Q3_K_XL (GGUF, 147.5 GB) -> FTW 134.24 GiB за 268 с.
Флаги бута - «победные» из кампании тюнинга: `--moe-cache-auto
--kv-reserve-tokens 500000 --kv-cache-dtype fp8 --memory-ratio 0.85
--max-prefill-length 8191 --moe-strategy hybrid --moe-cpu-threads 16
--max-running-requests 1` (порт 18801; llama-swap остановить; ratio 0.89
может fail-fast на раннем floor-gate при съеденной рабочим столом VRAM).

| метрика | FTW fast path | bare-GGUF якорь |
|---|---|---|
| бут (старт -> /ready 200) | **52.1 с** | 94-132 с |
| загрузка банков | 125G @ **4.12 GB/s** (30 с внутри бута) | серийная копия ~134 GB |
| префилл, медиана @8128-чанках | **790.7 tok/s** | 792-808 tok/s |
| декод @65k (3 повтора) | **14.78 / 15.52 / 15.02 tok/s** | паритет |
| radix HIT | 65536 во всех повторах | так же (после fix-1) |
| батарея качества | 24/24, паритет на уровне формулировок | 4 референс-бута |

Слоты по партициям 388/288/288; free-after-init 4.06 GiB; fetch-доля 30.7%
(не cap-1); пулы CPU-исполнителей [15,1,1] @16T avx2, строки пулов несут
gguf-сигнатуры. Заметьте: цифра ~293 tok/s в ранних брифах была ДО-якором
(dense-q80/m-tile кампаний ещё не было) и устарела —
[prefill-throughput.md](prefill-throughput.md).

## Ловушки

- Флаг пути - `--model-path`; позиционный аргумент serve отвергается.
- FTW-директория от старого конвертера fail-fast в `load_ftw_banks` с
  сообщением о необходимости переконвертировать (мета без gguf_types).
- В FTW-пути НЕТ строки-гистограммы ggml-типов, которая есть у bare-пути:
  сигнатуры искать в строках пулов CPU-исполнителей.
- Счётчик tracker.note обязан остаться ровно один на слой (после трёх ролевых
  копий) - иначе пиннинг молча пропускается и первый decode-miss даёт IMA
  при партиционном gather.
- Гэп-тесты bisect-green: серия коммитов устроена так, что каждый зелёный
  в бисекте (test_ftw_gguf_banks.py ставился двумя зависимыми срезами).

## Как измерить / проверить

- Регресс-сьюита: `tests/checkpoint/test_ftw_gguf_banks.py` (roundtrip через
  реальный convert_checkpoint с CUDA-skipif, 3-сигнатурный несимметричный
  фиксчер, вариант down-first, 5 малформ-мето-отклонений, sentinel-тесты
  capability) + `tests/engine/test_cache_budget.py` + `tests/moe/test_legacy_format.py`.
  Батарея: 175 passed / 7 skipped.
- Конвертация и измерения: скрипты
  [run-convert-wave1.sh](../harness/ftw/run-convert-wave1.sh),
  [verify-convert-wave1.py](../harness/ftw/verify-convert-wave1.py),
  [run-wave2-measure.py](../harness/ftw/run-wave2-measure.py),
  [wave2-parity.py](../harness/ftw/wave2-parity.py); машиночитаемые результаты -
  [wave2_results.json](../baselines/ftw/wave2_results.json) и
  [wave2_parity.json](../baselines/ftw/wave2_parity.json).
- Полный бриф с Outcome-секцией -
  [TASK.md](../cases/ftw-gguf-fastpath/TASK.md).
- Смежное: модель стоимости гибридного MoE -
  [moe-hybrid-cost-model.md](moe-hybrid-cost-model.md); VRAM-бюджет бута -
  [vram-budget.md](vram-budget.md); переиспользование radix-кэша -
  [radix-cache-reuse.md](radix-cache-reuse.md); термины -
  [glossary.md](../glossary.md).