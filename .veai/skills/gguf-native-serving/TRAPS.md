# TRAPS.md - 38 ловушек для native GGUF serving

Каждая ловушка стоила реального отладочного времени в GLM-5.3-Flash-UD-Q3_K_XL
кампании. Проверяй КАЖДУЮ перед закрытием соответствующей фазы.

## Discovery (Task 00, [tasks/task-00-discovery.md](tasks/task-00-discovery.md))

- **T01** mmproj-BF16.gguf может лежать уровнем выше основного .gguf - проверяй оба пути.
- **T02** gguf-py DROPS empty arrays on write - отсутствие ключа в дампе = нет
  массива в файле, НЕ баг reader'а.
- **T03** llama.cpp arch key map FLAT - каждый ключ `<arch>.<name>`, нет per-arch case
  blocks; llama-arch.cpp строки с capability-флагами - НЕ именные маппинги.

## Config shim (Task 01, [tasks/task-01-config-shim.md](tasks/task-01-config-shim.md))

- **T04** parse_gguf_config должен давать Args field-identical к load_args(HF config)
  той же модели - "близко" недостаточно.
- **T05** shim vocab_size = token_embd.weight shape[-1] (ggml ne-order: ne0=hidden,
  ne1=vocab) - не путать с HF (vocab, hidden).
- **T06** indexer_types per-LAYER ("full",)*num_layers, НЕ per-DSA-count
  (config.py:111 + dsa.py:154 индексируют по layer id).
- **T07** tie_word_embeddings = (output.weight отсутствует); если present -> untied.

## Tensor translator (Task 02, [tasks/task-02-tensor-translator.md](tasks/task-02-tensor-translator.md))

- **T08** ssm_a: BARE name (без .weight), holds -exp(A_log); A_log = log(-ssm_a) fp32,
  guard x >= 0.
- **T09** ssm_dt приходит как .bias, маппится в plain param (не Linear.bias).
- **T10** attn_k_b/attn_v_b 3D (head-major); transpose пересекает Q8_0 pack axis ->
  fusion ОБЯЗАН dequant'ить эти два куска.
- **T11** Packed fusion валиден ТОЛЬКО если все куски имеют одинаковый ne0
  (= одинаковый row_bytes); guard перед cat'ом.
- **T12** Reference checkpoint может иметь SEPARATE q/k/v - валидируй против кусков,
  не против pre-fused reference тензора.
- **T12b** Incomplete fusion groups at end-of-stream -> ValueError (не bare assert -
  умирает под -O, silently partial param set).

## Tokenizer (Task 03, [tasks/task-03-tokenizer.md](tasks/task-03-tokenizer.md))

- **T13** Sibling tokenizer.json НЕ читается для .gguf (reader.py:31-33 - embedded
  vocab = THE convention). Не строй --tokenizer-path флаг.
- **T14** gguf-py drops empty arrays on write; BPE initializer требует каждый merge's
  result в vocab - fixture-ватч.
- **T15** Serve path re-encodes rendered template TEXT; если specials не зарегистрированы
  как AddedTokens - байт-сплиттятся в мусор.
- **T16** Reference pre_tokenizer digits = \p{N}{1,3} (1-3 chars), НЕ one-per-token -
  портируй ТОЧНЫЙ regex.

## Kernel dispatch (Task 04, [tasks/task-04-kernel-dispatch.md](tasks/task-04-kernel-dispatch.md))

- **T17** IQ форматы без MMQ (ggml_mul_mat_a8 не имеет case; moe_get_block_size = 0) -
  moe_vec покрывает; не добавляй в _MMQ без проверки csrc.
- **T18** BLOCK_SHAPE + _DEQUANT консистентность (upstream #358 = missing-entry bug).
- **T19** Tied variant -> GGUFTiedLMHead; untied -> обычный GGUFLinear head.
- **T20** Single-signature degenerate = byte-identical pre-partition behavior.

## Expert banks (Task 05, [tasks/task-05-expert-banks.md](tasks/task-05-expert-banks.md))

- **T21** Pin-count trap: expected_per_layer != NOTE count -> тишина -> IMA.
  Construction-time guard ОБЯЗАТЕЛЕН.
- **T22** Cross-group prefetch hop может быть no-op (все партиции degraded) -
  characterization test вместо hop'а.
- **T23** Per-role EXTREME sizing завышает aggregate (+24.7 GiB для 3 сигнатур) -
  header-only scan для точного значения.
- **T24** set_bank_sources uniform-shape assert заменяется per-signature grouping.

## E2E (Task 06, [tasks/task-06-e2e-ab.md](tasks/task-06-e2e-ab.md))

- **T25** argparse: --model-path (обязательный флаг), positional rejected.
- **T26** /v1/chat/completions "model" field REQUIRED (422 без него).
- **T27** TTFT на cached prefix = first-decode-step overhead, не re-prefill.
- **T28** KDA autotune ~256 MiB do_bench scratch - eager tail может OOM при
  малом free-after-init (capture-config boot может не OOM - другой VRAM профиль).
- **T29** max_tokens=8 с reasoning parser -> пустой content (токены в
  reasoning_content) - не error.

## CPU compute tier (Task 07, [tasks/task-07-cpu-compute-tier.md](tasks/task-07-cpu-compute-tier.md))

- **T30** ISA env-override (FREETOKEN_CPU_MOE_ISA) обязан КАПИТЬСЯ поддержкой CPU
  (CPUID pick_isa): некапнутый override на бокс без флагов = SIGILL в проде.
  Разрешён только cap-down + негативный тест с некорректным env (63b9bff;
  pick_isa в cpu_moe_ext.cpp).
- **T31** benchbw пишет ПОЛНЫЙ per-GPU профиль на каждом прогоне
  (benchbw.py:836-861, _atomic_write_json): single-dtype реран (--dtype gguf)
  ЗАТИРАЕТ записи других форматов -> их гибриды молча деградируют в cap-1
  (единственный симптом - ранг-0 warning). После ЛЮБОГО bench-изменения - FULL
  реран всех форматов + проверка profile-файла на все записи.
- **T32** Параллельные измерения коррумпируют друг друга и машину: первый раунд
  ggml-микробенча ДИСКАРДИРОВАН (concurrent measurement processes). Один
  измерительный процесс за раз, hard timeout, PID-reap + ps census между
  точками (ggml-microbench.md VOID notice; рецепт D07).
- **T33** Фиксированный "handshake floor", измеренный при высоком f, -
  мисатрибуция PCIe fetch-объёма как sync-стоимости: bisect Task 05 записал
  1.9-3.0 ms/layer "handshake" (verification/bisect-notes.md), NVFP4 A/B того
  же движка показала fixed sync ~0.6-1.0 ms/layer
  (verification/nvfp4-ab-measurements.md, разд. 5), а fetch-объём эндогенен
  скорости CPU-leg. Изолируй sync при f->0 (cap-0 прогон) до любого вывода
  NO-GO.
- **T34** Even thread-split между партициями голодает доминантную: 6/16 тредов
  -> 2.96 GB/s эффективных против 11.3 benched, 7.5/20 ядер занято
  (bisect-notes.md, п. 2). Взвесь пулы числом слоёв партиции ([15,1,1] после
  979e3fc), сохрани floor-at-one + disjointness, guard'ь wrap-уголок явного
  пути (63b9bff).
- **T35** Проекшн без якоря на рабочую reference - ложный NO-GO: бумажная модель
  (research/task06-rework-estimate.md) дала "реалистично 8-11 tok/s, NO-GO",
  железо с портированными ggml-ядрами дало 16.67 (verification/re-acceptance.md).
  Каждый проекшн заякоривай на A/B рабочей реализации (рецепт D08).
- **T36** У benchbw-профиля нет TTL и нет kernel-tier fingerprint: профиль,
  записанный ДО смены ядер (scalar f=78%), молча потребляется ПОСЛЕ порта.
  Репрофиль после любого изменения ядер; верифицируй boot-строку fraction
  против ожидаемой (re-acceptance.md STEP 1/5: 30.7%/24.8% восстановлены
  FULL-рераном). Fingerprint/schema bump - известный follow-up, НЕ реализован.
- **T37** CC/CXX, экспортированные gguf JIT-хелпером, утекают в последующие
  сборки (неверный компилятор для nvcc/host-целей). Скопи переменные строго
  на вызов сборки (1635ecd).
- **T38** Engine capability gates спрашивают capability set исполнителя
  (partition_executor_rejection, aad5d3a), а не хардкодят type id/имя формата:
  gate "excludes gguf by name" на engine.py:~1817 был баг-классом (PLAN.md,
  Task 04 DISCOVERY).

## Debugging

- **D01** CUDA IMA асинхронен: портивший launch ПРЕДШЕСТВУЕТ видимому кадру.
  CUDA_LAUNCH_BLOCKING=1 синхронизирует; compute-sanitizer memcheck точен.
- **D02** Bisect: eager (--cuda-graph-max-bs 0) vs capture -> capture-specific или нет.
- **D03** Per-layer hidden-state bisection: forward hooks на обеих моделях
  (SEQUENTIAL boots, никогда обе в RAM); первый cos << 1 names the module.
- **D04** Static fusion validation: mmap reference per-tensor (не полный load) -
  дешевле и быстрее, чем live bisect; делай ВСЕГДА первым.
- **D05** Green GPU test с .cuda()-тензорами НЕ тестирует pinning - pageable-vs-pinned
  только на реальном device. Construction-time guard обязателен.
- **D06** Standalone CPU-микробенч ядер reference: компилируй СОБСТВЕННЫЕ .c файлы
  reference standalone (include paths only) с ТЕМИ ЖЕ ISA-флагами, что и
  референс-сборка (build/CMakeCache.txt + per-variant flags.make; haswell-ярус =
  -O3 -mf16c -mfma -mavx -mavx2 -fopenmp); линк-шимы только для недостижимых
  символов ggml.c; блоки - синтетические byte-exact, f16 d ограничен finite
  normals; парити SIMD-vs-scalar на случайных строках (max rel err < 1e-3) на
  КАЖДОМ прогоне ДО таймингов; blended + per-format прогоны, свип тредов;
  OMP_PLACES/OMP_PROC_BIND выставлять ДО старта процесса (libgomp читает их в
  pre-main конструкторе; setenv из main молча сваливает все треды на ядро 0);
  warmup + автокалибровка >= 3 s на точку. Gate: sustained GB/s >= эффективной
  PCIe gather полосы. Эвиденс: .tasks/gguf-glm5next-hybrid/verification/
  ggml-microbench.md + ggml-microbench-harness.cpp.
- **D07** Serialized-run дисциплина: один измерительный процесс за раз; hard
  timeout; PID-reap + ps census между точками; pre-run чистка сирот
  (multiprocessing PIDs + /dev/shm/sem.*), post-run census (GPU idle, CPU idle).
  Параллельные прогоны и утекают, и портят числа друг друга (VOID notice в
  ggml-microbench.md).
- **D08** Reference-A/B декомпозиция: тот же движок, тот же клиент, тот же fill,
  единственная переменная --moe-strategy; collect-stats дампы (missing/fetched/cpu
  per layer) + benchbw-ноги; шаг = max(ноги) + fixed sync; гэп объясняется тем,
  КАКАЯ память несёт промахи (DDR5 vs PCIe). Метод-образец:
  verification/nvfp4-ab-measurements.md.
