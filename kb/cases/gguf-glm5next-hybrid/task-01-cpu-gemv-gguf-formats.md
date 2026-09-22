# Task 01: CPU GEMV for gguf bank formats (IQ3_XXS / IQ4_XS / Q6_K) - W1

Type: implementation (C++ extension + python binding + tests)
Suggested agent: Code (strong)
Spec coverage: R1, S1; executor side of R2's per-projection requirement.

## Goal

cpu_moe_ext.cpp computes GEMV for IQ3_XXS(gguf 18), IQ4_XS(23), Q6_K(14);
cpu_executor.py accepts these formats with PER-PROJECTION dispatch (the three
projections of one layer may have different types); parity tests vs the gguf-py
reference pass within the dfloat bound.

## Why This Task Exists

CpuMoeExecutor hard-fails unknown formats (_WFMT_IDS has no gguf types) and the
engine gate blocks gguf+hybrid. Without CPU GEMV for the real file's bank types
the hybrid path cannot run at all. This is the foundation for Tasks 02-05.

## Required Inputs (read first)

1. .tasks/gguf-glm5next-hybrid/research/w1-cpu-gemv-mapping.md - the verified
   code-site map; every file:line below is evidenced there.
2. .tasks/gguf-glm5next-hybrid/TASK.md - "Lessons" section (all 10 binding) and
   Constraints.
3. CONTRIBUTING.md (project root) - binding; conventional commits, one change
   per PR, bug fixes need failing-before tests.

## Files/Areas (verified line refs)

- kernel/csrc/cpu_moe/cpu_moe_ext.cpp (2160 lines, NOTE the cpu_moe subdir):
  pybind11 torch extension (PYBIND11_MODULE:2114); boundary crosses RAW
  POINTERS only (per-layer int64 tables via _make_table, task IO ptrs).
  Per-format = function pointer + ISA selector (select_q4dot:1216,
  select_nvi8dot:756); enum WFmt:1223; row dispatch gemm1_dot:1484 /
  gemm2_dot:1506. q4_0 is W4A8 (quant_q8_0:1463, input prequant in submit:1941,
  intermediate in prep_g_row:1826); nvfp4 W4A8 via use_vnni + quant_i8_pg16:1462.
  New format = enum id + dot fn + selector + ctor strides + gemm branches.
- kernel/csrc/gguf/dequantize.cuh + ggml-common.h (VENDORED - read-only source
  of truth): Q6_K 210 B block, d LAST (208:210), ql128/qh64/int8 scales16;
  IQ3_XXS 98 B, d FIRST, 8x uint32 aux32 inside qs bytes 64:96, iq3xxs_grid
  (ggml-common.h:563), ksigns_iq2xs (:888); IQ4_XS 136 B, d FIRST, scales_h +
  scales_l[4], kvalues_iq4nl (:927). dfloat = half (ggml-common.h:930). The
  LUTs are __device__ in vendored files -> copy them VERBATIM into cpu_moe
  host code; do NOT edit the vendored files.
- python/freetoken/moe/cpu_executor.py: _WFMT_IDS:72 = {bf16,nvfp4,
  mxfp4_triton,ds_fp4,q4_0}. Resolver precedent _resolve_q4_0_banks:407-432:
  receives cache.bank_sources (role -> list of per-layer pinned uint8
  [E, rows, row_bytes] tensors), returns C++ pointer tables + (H,I) - NOT a
  GgufTensor. No python GEMV fallback exists; everything dispatches to _cpu_moe.
  decode_submit:543 ships pinned x/ids/w; topk_ids may carry -1 (C++ skips).

## Design guidance

- Kernel shape: TASK.md says "scalar/LUT GEMV" - dequant-on-the-fly LUT + fp
  accumulation is the intended path (IQ grid indices are LUT fp values, they do
  not map to W4A8 int8 weights). Options: (a) dequant weights to a bf16 scratch
  buffer then reuse the existing bf16 dot path (scratch sizing: top_k=8 *
  rows(4096) * H(4096) * 2 B ~ 268 MB per layer transient - check host RAM
  headroom), or (b) fuse dequant into the dot loop (zero extra memory, slower).
  Pick one, justify in the report. Keep the ISA-selector structure (scalar tier
  at minimum; AVX2/AVX512 tier optional follow-up).
- Activations: existing W4A8 formats prequantize input (submit:1941). For the
  fp-dequant path keep the boundary contract (raw pointers, pinned IO, -1 id
  skipping) and feed the dot with the dtype the chosen kernel shape needs.
- GATE/UP SPLIT (main C++ design decision): gguf banks are SEPARATE gate and up
  tensors, but the C++ side assumes one gate_up table with row I+i. Extend the
  pointer-table layer to take per-role tables (or an adapter that presents two
  tables). Per-role geometry: down rows = H (4096), gate/up rows = I (2048) -
  never reuse one row count.
- Per-projection formats: executor currently takes ONE weight_format; extend to
  per-role fmt so gate=IQ3_XXS + up=IQ3_XXS + down=IQ4_XS works in one layer.
- Implement incrementally: Q6_K first (simplest - no grid LUT, d last), then
  IQ4_XS (kvalues_iq4nl LUT), then IQ3_XXS (grid + signs, most complex).

## Constraints / Non-Goals

- No edits to vendored kernel/csrc/gguf/* files.
- No python dequant in the hot path; gguf-py (venv quants.py) is the TEST
  reference only (proven cos 0.99994-0.99997 vs CUDA in the Phase 6 bisect).
- Do not port IQ3_XXS dequant into models/gguf/dequant.py (it has none; not
  needed - the venv gguf-py is the test reference).
- dfloat tolerance: per-element bound 8*2^-11*factor*|d| + 1e-3
  (tests/kernels/test_gguf_quant.py:156); bit-parity vs fp32 is impossible.
- After csrc changes: python setup.py build_ext --inplace.
- One change per commit; conventional commits; Assisted-by: Veai. No push.

## What to Do

1. Read the Required Inputs; then read the target files around the cited lines.
2. Implement the three C++ dot paths + enum/selector/stride/gemm wiring.
3. Verbatim host LUT copies (iq3xxs_grid, ksigns_iq2xs, kvalues_iq4nl).
4. cpu_executor.py: extend _WFMT_IDS, add a gguf-format resolver mirroring
   _resolve_q4_0_banks (per-role tables, per-role fmt), thread per-projection
   fmt through the boundary.
5. Tests:
   - Parity vs gguf-py reference for all 3 formats (template:
     tests/moe/test_cpu_moe_q4_0.py; fixture crafting helpers:
     _craft_uniform_gguf_banks in tests/models/test_gguf_expert_banks.py:204,
     _FP16_SCALE_FIELDS in tests/kernels/test_gguf_quant.py:53).
   - Per-projection test: gate IQ3_XXS + down IQ4_XS in one layer computes
     correctly.
   - Flip tests/models/test_gguf_expert_banks.py:127 (asserts gguf NOT in
     _WFMT_IDS - now it IS; update the assertion and its message).
   - Bug-fix style: failing-before/passing-after where applicable.
6. Build extension; run: new tests + tests/moe + tests/kernels +
   tests/models/test_gguf_expert_banks.py. Record exact outputs.
7. Commit (one change per commit, conventional, Assisted-by: Veai).

## Expected Output (report_result)

- Summary of the kernel shape chosen (scratch vs fused) and why.
- Exact list of changed/added files.
- Test commands + exact pass/fail output lines.
- Any blockers or unresolved assumptions.

## Acceptance Criteria

- [ ] C++ GEMV for IQ3_XXS/IQ4_XS/Q6_K matches gguf-py reference within the
      dfloat bound (new parity tests green)
- [ ] Per-projection types work in one layer (gate IQ3_XXS + down IQ4_XS)
- [ ] test_gguf_expert_banks.py:127 flip landed; tests/moe, tests/kernels,
      tests/models/test_gguf_expert_banks.py green
- [ ] I've created a git commit for this task
