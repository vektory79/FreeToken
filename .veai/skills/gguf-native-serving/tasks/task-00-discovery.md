# Task 00: Discovery - gguf inventory + anchor verification

Type: Ask (read-only) | Agent: Ask or Code (measurement) | Duration: ~30-45 min

## Goal

Собрать полный инвентарь gguf-файла и верифицировать якорные факты текущего дерева.
Без этой фазы остальные фазы работают вслепую.

## Instructions for the subagent

1. Dump the gguf metadata via gguf-py GGUFReader (mmap, no tensor data):
   - ALL tensor names + types + shapes -> `tensors.txt` (lexicographically sorted)
   - ALL metadata fields -> `metadata.txt` (scalars fully; arrays >64 as `<ARRAY len=N>`)
   - tokenizer.ggml.* keys, chat_template, special token ids
   - Output: `.tasks/<campaign>/tensors.txt`, `.tasks/<campaign>/metadata.txt`
2. Verify anchors in the FreeToken tree:
   - models/gguf/config.py: GGUF_ARCH_TO_REGISTRY whitelist + the ValueError line
   - server/args.py -> utils/hf.py: the detection chain
   - kernel/csrc/gguf/gguf_kernel.cu: WHICH quant formats have dequant + MMVQ + MMQ +
     moe_vec symbols (grep for each format the file uses)
   - layers/gguf.py: which formats the python dispatch already exposes
   - moe/cpu_executor.py _WFMT_IDS: which formats the CPU executor supports
   - The target family's HF/FTW code under python/freetoken/models/<family>/:
     args dataclass fields, weight.py tensor-name patterns
3. Upstream alignment or waive: check the target model family against the
   repo roadmap (FreeToken issue #79) and the upstream gguf umbrella issue
   (#34). If the campaign follows upstream expectations, record "upstream:
   aligned" in the discovery dump; if it deviates (e.g. private-use local
   serving the user does not plan to publish), obtain an EXPLICIT user waive
   and record it with the date ("upstream: waived by user <date>") in
   research/anchor-verification.md. Campaign precedent: glm5next ran
   private-use under an explicit waive of the upstream-issue gate
   (2026-09-13). No later phase starts before this resolves.
4. Working reference implementation - ASK THE USER EXPLICITLY for the path to
   llama.cpp (or other) sources where the target gguf format is ALREADY
   supported. If the format is absent upstream, ask for whatever reference
   exists - projections must be anchored on a working reference, never on
   paper math alone. Record into anchor-verification.md:
   - vec_dot kernels: ggml/src/ggml-cpu/arch/<isa>/quants.c (x86: arch/x86/quants.c)
   - activation quantizers: quantize_row_* in ggml/src/ggml-quants.c
   - block layouts: ggml/src/ggml-common.h (static_assert'd byte offsets)
   - the reference build's ISA flags: build/CMakeCache.txt (+ per-variant
     flags.make) - REQUIRED later for microbenchmark comparability (task-07)
   - the backend-scheduling pattern for CPU-resident weights:
     ggml_backend_sched splits + boundary copy nodes (--n-cpu-moe buffer
     overrides push ffn_*exps to CPU)
5. If a llama.cpp reference is available (kind map for phase 2):
   - src/models/<arch>.cpp: LLM_TENSOR kinds, hparams reads, load_tensors
   - src/llama-arch.cpp: the FLAT key map (every key is `<arch>.<name>`)
   - LLM_TENSOR_NAMES: resolved gguf name patterns
6. CPU compute-tier support matrix (feeds task-07): per format in the file -
   CUDA symbols (dequant + MMVQ + MMQ + moe_vec); existing CPU tier in
   moe/cpu_executor.py _WFMT_IDS + cpu_moe_ext.cpp (select_ggufdot tiers,
   select_nvi8dot); the PAIRED activation format (K-quants and IQ-quants ->
   q8_K, q4_0 -> q8_0, nvfp4 -> pg16/VNNI - read it from the reference, never
   guess); sustained GB/s if a bench already exists. Flag every format with
   NO CPU integer tier - that list is the phase-7 port scope.
7. Write all findings to `.tasks/<campaign>/research/anchor-verification.md`.

## Output

- tensors.txt, metadata.txt (complete, no gaps)
- anchor-verification.md: every fact with file:line evidence, including the
  reference implementation path + what was extracted from it (kernels, LUTs,
  layouts, ISA flags, scheduling pattern)
- A gap matrix: what the CUDA kernels support vs what the file needs
- The CPU compute-tier support matrix row per format (CUDA symbols / CPU tier /
  paired activation format / reference kernel file:line)
- The upstream-alignment-or-waive decision (aligned, or the explicit user
  waive with date)

## Traps (from [TRAPS.md](../TRAPS.md))

- T01: The mmproj file may be one directory UP from the main .gguf - check both.
- T02: gguf-py DROPS empty arrays on write - a missing key in a dump means the file
  has no such array, NOT a bug in the reader.
- T03: The llama.cpp arch key map is FLAT - do not look for per-arch case blocks.
- T35: A projection without a working-reference anchor produces false NO-GO -
  the reference-extraction outputs above are that anchor.

## Acceptance criteria

- [ ] I've created a git commit for this task
- Runs under ORCHESTRATION.md (quality loop + hygiene + commit gate).
- This is the discovery wave that asks the reference-sources question (the
  working reference where the target gguf format is already supported) - the
  campaign cannot start before the user answers it.
- Upstream alignment or an explicit user waive recorded in the discovery dump.
