# TRAPS.md - 50 ловушек для native GGUF serving (T01-T50, рецепты D01-D10)

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
  Тот же класс бьёт и в префилл: 8191@0.90 OOM в ПЕРВОМ префилл-чанке (triton
  do_bench, driver.py:761: 256.00 MiB при ~187 MiB free) - для 8191 нужен
  0.85+mr1 (free-after-init ~4.05 GiB; измерено 2026-09-18).
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

## MMQ-prefill кампания 2026-09-17/18 (kernel swap) T39-T47

Ловушки kernel-swap кампании MMQ-префилла (artifacts .tasks/mmq-prefill-kernel/,
приёмка 7f8c570).

- **T39** (Phase 4, design) Переупорядочивание (token,expert)-пар по expert id
  (L2-сортировка) - throughput no-op на RTX 5090: тысячи резидентных CTA исполняют
  соседние пары КОНКУРЕНТНО, порядок запуска не режет конкурентный трафик; BW-bound
  член двигает только сокращение полного трафика (чтение весов один раз).
  Измерено: 289.54->291.26 tok/s (+0.6% шум) при подтверждённой активности пути
  (.tasks/mmq-prefill-kernel/v0-ab.md). Не гнаться за cache-locality reorderings
  для стриминговых GEMV/MoE ядер.
- **T40** (Phase 4, vendored-code audit) Перед переиспользованием вендоренных ядер
  проверять молчаливые bound-допущения: moe.cuh exp_idx > 255 молча дропал
  экспертов 256+ при E=288 (bound должен браться из expert-размерности
  weight-тензора); moe_q ds-load читал token_offs[threadIdx.y] OOB (фикс -> [0];
  латентный баг апстрима vLLM). Класс аудита: индексные константы, размеры
  per-thread массивов, диапазоны expert-id vs геометрия модели.
- **T41** (Phase 4, sourcing) Tile glue для iq-форматов не вендорится дословно из
  llama.cpp на вендоренный pre-#8495 vLLM-интерфейс; рабочий источник - vLLM PR
  #36226 (raw block bytes в tile_x_ql, decode внутри vec_dot); согласованность:
  VDR=4 + need_sum=true (need_sum=false портит чтение half2 {d,sum}; VDR=2 даёт
  double-count). Attribution называть ДВОЙНЫМ: vLLM (Apache-2.0) + llama.cpp
  lineage (MIT) - MIT-only недостаточно.
- **T42** (Phase 4, infra reuse) Перед портированием vLLM-хелперов проверять
  наличие продюсера в freetoken: moe_align_block_size уже есть (moe/fused.py:47;
  sgl-бекенд + triton-фолбэк; контракт трио: sentinel sorted_token_ids = numel,
  expert_ids int32, num_tokens_post_pad (1,)); ggml_moe_a8/ggml_moe_get_block_size
  были bound-but-dead. ОДНО трио обслуживает gate/up (top_k=8) и down (top_k=1,
  tokens=M*top_k) - ядро делит flat pair index на свой аргумент top_k.
- **T43** (Phase 6, measurement) Строка "input throughput" ПОСЛЕДНЕГО полного
  чанка фиктивна (~1552-1602 tok/s при реальных ~290; tail-drain артефакт,
  воспроизводится между кампаниями): медианы префилла считать исключая chunk 1
  (warmup) И последний полный чанк.
- **T44** (Phase 6, A/B validity) BEFORE-стадия обязана воспроизвести
  историческую кампанийную базу (в пределах run-variance ~3%) - только после
  этого дельта AFTER доверийна; иначе результат неинтерпретируем.
- **T45** (Phase 6, A/B mechanics) Same-session A/B без stash: per-call env kill
  switch (os.environ.get в forward-пути) + два бута с env-флипом; JIT disk-cache
  компилирует один раз и обслуживает оба бута (python-only правки вообще не дают
  пересборки). После приёмки свитч удалять (прецедент
  FREETOKEN_GGUF_GROUPED_PREFILL: добавлен под A/B, удалён post-validation
  7f8c570).
- **T46** (Phase 6, liveness) Смена ядра требует kernel-level liveness proof:
  nsys kernel-name контракт + ТОЧНАЯ математика счётчиков запусков (пример:
  grouped moe ядра 42 слоя x 3 proj x 9 чанков = 1134 + moe_align 42x9 = 378;
  ноль запусков старого ядра в префилл-чанках; старое ядро присутствует в decode
  graph-replays). Bare stdlib logger'ы невидимы в boot-логах (init_logger не
  вешает хендлер; lastResort дропает INFO) - one-time INFO маркеры ловить
  PYTHONPATH sitecustomize-пробой.
- **T47** (process, tooling search) Прежде чем пропустить валидационный пункт
  из-за "нет тулинга", искать по ВСЕМ .tasks кампанийным папкам и /tmp (слишком
  узкий grep стоил двух пропусков quality battery; батарея выжила в
  gguf-glm5next-path-a + /tmp и была успешно адаптирована). Волатильные /tmp
  артефакты копировать в .tasks немедленно.

## Fix-1 radix кампания 2026-09-18 (scheduler/radix) T48-T50

Ловушки кампании fix-1 radix-reuse (артефакты .tasks/fix1-radix-track-seqlen/;
фикс 5b72aba "fix(scheduler): carry mamba_last_track_seqlen across prefill
chunk transitions").

- **T48** (scheduler, fixed 5b72aba) Дроп поля на переходе префилл-чанков:
  mamba_last_track_seqlen (L) не форвардился из pending_req.chunked_req в
  continuation Req (try_add_one, scheduler/prefill.py) -> промпт, чей ПОСЛЕДНИЙ
  префилл-чанк кончается ниже x64 track boundary, финиширует с L=None -> нет
  GDN snapshot donation -> МОЛЧИВЫЙ radix MISS на идентичном повторе.
  Диагностическая сигнатура: повторный HIT на крупных чанках (8128 -> 65536),
  MISS на мелких (4096 -> 0/49) при любом ratio/mr - конфиг-независимый баг
  (НЕ dtype/mr/ratio/timing), зависит только от chunk size. Пост-фикс, повтор
  65k: #cached-token 65472 @4096 / 65536 @8191 (8128-чанки, 0.85+mr1).
- **T49** (gate, Environment-классификация) NaN/Environment-фейл полного
  гейта, который проходит изолированно И с диффом, И без - это suite-ordering
  Environment, а не регрессия (прецедент
  test_gguf_expert_banks.py::test_grid_caps_raise_before_kernel_launch: NaN
  isfinite только под full-suite порядком; гейт 2135 passed / 6 baseline-class
  failed). Прежде чем винить изменение - изолированный A/B прогон.
- **T50** (boot, slot-floor) Ранний slot-floor gate фейл-фастит, когда
  десктопные приложения съедают VRAM (~492 MiB hoptodesk-налог): "minimum of
  1059 layer-0-width slots vs planned 1040" при --memory-ratio 0.89. Лестница
  ретраев: 0.89 -> 0.90 (free-after-init 2.59 GiB); для 8191 нужен 0.85+mr1
  (см. T28). Налог переменный день ото дня - VRAM-конкуренция, а не регрессия
  конфига.

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
- **D09** nsys live-serve профилирование: nsys launch отвергает -o; рабочая
  схема = launch + start-after-ready + --cuda-graph-trace=node; голый mid-run
  start даёт отчёт с API/graph-записями, но БЕЗ eager kernel activities; анализ
  через sqlite export; окно брэкетится по строкам "input throughput"; overhead
  ~nil (чанки внутри/вне окна идентичны).
- **D10** Step-0 split перед design kernel-swap: один nsys-проход + sqlite
  export фиксирует раскол GEMM/copies/rest с kernel-name учётом на чанк
  (пример: moe_vec_q 126/чанк = 42 слоя x 3 proj; fast_index_copy_multi
  42/чанк); GPU idle ~0.05% = нет CPU-starvation; замер ДО проектирования
  перекроил список бутылочных горлышек (dense q8_0 GEMM стал top "rest",
  а не fetch copies).
