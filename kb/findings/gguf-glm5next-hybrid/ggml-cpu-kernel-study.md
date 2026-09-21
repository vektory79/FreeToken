# ggml CPU integer dot kernels: study for the cpu_moe_ext.cpp port

Read-only research, 2026-09-15. Sources: FreeToken @ /media/ai/src/FreeToken (vektory79, eb7de4c,
clean) and the llama.cpp checkout at /media/ai/src/llama-cpp-glm5next (MIT, "Copyright (c)
2023-2026 The ggml authors" - LICENSE:1-3). Goal: decide whether transcribing ggml's CPU integer
dot kernels closes the ~4-14x CPU-leg gap vs llama.cpp, and what that does to CPU-only and hybrid
decode.

Model geometry (read from the real file header, GLM-5.3-Flash-UD-Q3_K_XL.gguf):
- H = embedding_length = 4096, I = expert_feed_forward_length = 2048, n_expert = 288,
  n_expert_used = 8, blocks = 46 with 3 leading dense (header says 43 MoE blocks; the campaign's
  accepted count is 42 - numbers below use 42, delta is +2.4%).
- Expert tensor formats on disk: gate/up = IQ3_XXS on 41 blocks (1 IQ4_XS, 1 Q3_K), down =
  IQ4_XS on 39 (3 Q6_K, 1 Q4_K). So the campaign's three formats IQ3_XXS / IQ4_XS / Q6_K cover
  everything that matters.
- Weight bytes per expert per typical layer: 2 x 4096x2048 x (98/256) + 4096x2048 x (136/256)
  = 10.88 MB. Per token: 8 experts x 10.88 MB x 42 layers = 3.66 GB streamed from RAM.
  10 tok/s => 36.6 GB/s sustained single-pass read (matches the task's ~42 GB/s estimate).

---

## 1. Kernel anatomy (the three vec_dot kernels, AVX2 tier)

File: ggml/src/ggml-cpu/arch/x86/quants.c (x86 AVX2/AVX/SSSE3 variants live per-arch under
ggml/src/ggml-cpu/arch/; scalar references named *_generic are in ggml/src/ggml-cpu/quants.c,
dispatched for non-x86 by ggml/src/ggml-cpu/arch-fallback.h:27-36).

- ggml_vec_dot_q6_K_q8_K: arch/x86/quants.c:2426; AVX2 body 2439-2508.
- ggml_vec_dot_iq3_xxs_q8_K: arch/x86/quants.c:3260; AVX2 body 3274-3350.
- ggml_vec_dot_iq4_xs_q8_K: arch/x86/quants.c:4004; AVX2 body 4022-4064.
- Shared helpers in the same file: mul_add_epi8 (sign-trick + _mm256_maddubs_epi16, :68-74),
  hsum_float_8, get_scale_shuffle (:60-66 and vec.h for SSE shuffles).

Instruction mix per 256-element super-block (QK_K=256):

IQ3_XXS (98 B block): per 64-element iteration (iq3xxs AVX2 body):
- LUT gather #1: `_mm256_set_epi32(iq3xxs_grid[q3[0..7]])` - 8 scalar loads from the 256-entry
  uint32 grid, compiled to ~14 blend/insert uops; x2 per iteration.
- Sign LUT: 8 scalar loads from keven_signs_q2xs[128] (uint64) + 2 `_mm256_set_epi64x`.
- Then per 32-element sub-block: `_mm256_sign_epi8(q8, signs)` -> `_mm256_maddubs_epi16(grid,
  q8s)` (u8*i8 -> i16) -> `_mm256_madd_epi16(dot, set1(2*ls+1))` (per-32 isum scale folded as an
  integer multiply, ls from bits 28-31 of the aux32 words) -> add_epi32 into 2 accumulators.
- One f32 FMA of d*(sumi1+sumi2) per super-block; final `*s = 0.25f * hsum_float_8(accumf)`
  (arch/x86/quants.c:3348-3350).
- Count: ~56 uops / 64 elems ~ 0.9 uops/elem ~ 0.22 cyc/elem at 4 uops/cyc.

Q6_K (210 B block): per 128-element j-iteration:
- Unpack two 6-bit streams from ql (lower 4) + qh (upper 2): 4x and/slli/srli/or chains ~ 12 uops.
- 4 loads q8 (32 B each), 4 `_mm256_maddubs_epi16`, then per-32 isum scale applied in int16
  domain: 4x `_mm_shuffle_epi8(scales,...)` + 4x `_mm256_cvtepi8_epi16` + 4x `_mm256_madd_epi16`.
- The -32 offset is NOT applied per-element: once per block, `q8sclsub = slli_epi32(madd_epi16(
  bsums(q8), scales_16), 5)` is subtracted (2449-2452, 2505) - this is why q8_K needs the
  bsums[16] fields.
- ~74 uops / 256 elems ~ 0.3 uops/elem ~ 0.10-0.15 cyc/elem.

IQ4_XS (136 B block): per 64-element iteration:
- Nibble LUT: 2x `_mm_shuffle_epi8(values128, nibble)` per sub-block (4 total, values128 =
  kvalues_iq4nl 16x int8), plus 2x and + 2x srli_epi16 nibble extraction.
- mul_add_epi8 per sub-block (3 uops each: sign,sign,maddubs), per-32 scale in int16 domain:
  `ls = ((scales_l nibble) | (2 bits from scales_h)) - 32`, applied by 2x `_mm256_madd_epi16(p16,
  set1(ls))`, 2x add_epi32; one f32 FMA per super-block (4058-4060).
- ~30 uops / 64 elems ~ 0.5 uops/elem ~ 0.11-0.13 cyc/elem.

Summary: all three are maddubs/madd integer chains with at most one scalar LUT gather stage;
0.1-0.25 cyc/elem compute - i.e. 10-25x lighter than FreeToken's current float-dequant scalar
kernels (~2.5-4 cyc/elem, PLAN.md Task 01). Every kernel is DRAM-bound on weight streaming well
below 20 GB/s/core, so the leg ceiling is memory bandwidth, not arithmetic.

Block layouts MATCH FreeToken's exactly (we vendored ggml-common.h as source of truth):
- block_iq3_xxs = 98 B: f16 d first + qs[3*QK_K/8] - ggml-common.h:405-411
  (static_assert :411); FreeToken gguf_block_bytes WF_IQ3_XXS -> 98 (cpu_moe_ext.cpp:1391).
- block_iq4_xs = 136 B: f16 d + u16 scales_h + scales_l[QK_K/64] + qs[QK_K/2] - ggml-common.h
  :455-460; FreeToken -> 136 (cpu_moe_ext.cpp:1393).
- block_q6_K = 210 B: ql[QK_K/2] + qh[QK_K/4] + scales int8[QK_K/16] + f16 d LAST -
  ggml-common.h:362-368; FreeToken -> 210 with d at offset 208 (cpu_moe_ext.cpp:1307-1311).
- LUTs already vendored verbatim: iq3xxs_grid, ksigns_iq2xs, kmask_iq2xs, kvalues_iq4nl at
  cpu_moe_ext.cpp:1250-1299 (values sourced from ggml-common.h per the comment at :1249-1252).

Verdict: the kernels are small, self-contained, and the data layout is byte-identical. A
transcription is mechanical; the only structural change on the FreeToken side is the ACTIVATION
format (bf16 today, q8_K in ggml) - see section 2.

## 2. Activation side: quantize_row_q8_K

ggml/src/ggml-quants.c:2768 quantize_row_q8_K_ref, per 256-element block:
1. amax pass over 256 floats (finds signed max, not abs-max: it keeps the sign of the max).
2. iscale = -127/max (NEGATIVE on purpose - comment :2783-2784 "We need this change for IQ2_XXS";
   the sign flip cancels in the dot, d = 1/iscale absorbs it).
3. quantize pass: qs[j] = MIN(127, nearest_int(iscale*x[j])) (note: only the upper side clamps).
4. bsums[j] = sum of each group of 16 quants (int16, QK_K/16 = 16 of them) - consumed only by
   q6_K's offset trick.
Block: block_q8_K = f32 d + int8 qs[256] + int16 bsums[16] = 292 B (ggml-common.h:371-376).

Cost per token per layer: ggml quantizes the activation row ONCE per mul_mat_id call (decode:
src1 is [H, 1]; the 8 expert slots share the same q8_K row; ggml-cpu.c:1627-1643, kernel reads
`(i11 + i12*ne11)*row_size` with ne11=1). For our geometry: 2 rows per layer - x (H=4096, feeds
gate+up) and the silu-mul output (I=2048, feeds down). Scalar cost ~4-5 cyc/elem over 6144 elems
~ 27-30k cycles ~ 6 us single-thread; x 42 layers ~ 0.25 ms/token ~ 0.3% of a 80 ms step. Can be
thread-split by block ranges exactly as ggml does (ggml-cpu.c:1634-1640).

Can FreeToken's pipeline produce q8_K? Yes, trivially:
- quant_q8_0 (cpu_moe_ext.cpp:1672-1691) already does per-32 amax + int8 round for bf16 rows.
  q8_K differs only in: block width 256 (one f32 scale per 256), the -127/max NEGATIVE iscale
  convention (d = 1/iscale), and the 16 bsums per block. No per-32 sub-block scales, no f16 - a
  single f32 d. It is SIMPLER than q8_0 per element.
- quant_i8_pg16 (:1693-1715) is the VNNI per-16 layout - orthogonal, untouched.
- Wiring: the gguf GEMV path currently feeds bf16 activations (the design note at
  cpu_moe_ext.cpp:1238-1248 chose W4A16 "no activation prequant" for the scalar tier). The port
  adds a W4A8-K form: quantize x -> q8_K once per token (gate/up) and the gate_out row once per
  (token,expert) (down), then dispatch the transcribed kernels. Scratch grows by
  top_k*I*292 B/token (~4.8 MB at I=2048, K=2048) + H*292 B/token - negligible vs the existing
  gi8/xi8 scratch pools sized at :1657-1661.

## 3. Batching: one token per vec_dot call

ggml decode path (batch 1): one MUL_MAT_ID node per projection. In
ggml_compute_forward_mul_mat_id_one_chunk (ggml-cpu.c:1471-1524) vec_dot is called per OUTPUT ROW
ir0 per (expert, token): `vec_dot(ne00, &tmp[ir0-iir0], 0, src0_cur + ir0*nb01, 0, src1_col, 0, 1)`
(:1516) - nrc=1, n = row length. No row-pair batching (vec_dot_num_rows=1 for all K-quant types;
the 2-row nrc=2 path only exists for ARM mmla). The iqp panel GEMM (iqp.cpp:17-19,
GGML_IQP_MIN_BATCH 8) only engages when >= 8 rows share an expert - never true at decode.

So per token: 42 layers x [gate: 8 experts x 2048 rows x n=4096 + up: same + down: 8 x 4096 rows x
n=2048] = 42 x 65,536 = 2.75M vec_dot calls, each streaming one 1.09-1.57 KB packed weight row
against one cached q8_K row. Thread work split: chunked by output rows with atomic chunk
stealing (ggml-cpu.c:1700-1740); the same shape FreeToken's executor already uses (worker_loop
over (expert,row) pairs, cpu_moe_ext.cpp:1079-1105). No change needed: FreeToken's per-step call
shape is IDENTICAL (rows = H or I, top_k=8, 3 projections, 42 layers).

Prefill note: at ubatch 1024 llama.cpp still uses the same vec_dot path (iqp only if it were
built to repack - it is not by default), so the port also helps prefill CPU legs, but our hybrid
prefill overlap degrades >= 2E anyway (offload_cache.py:176 invariant).

## 4. Scheduling without per-layer handshakes

How llama.cpp runs MoE-on-CPU + attention-on-GPU:
- Weight placement: --n-cpu-moe/-cmoe just pushes buffer-type overrides for the ffn_*exps tensors
  (common/arg.cpp:2787-2800, LLM_FFN_EXPS_REGEX -> CPU buffer). Expert weights are CPU-resident
  for the whole run. There is NO per-token expert fetch protocol - no misses exist.
- Node assignment: ggml_backend_sched assigns each graph node to a backend from its tensors'
  buffers (ggml-backend.cpp:888 ggml_backend_sched_backend_from_buffer, :921
  backend_id_from_cur) and cuts the graph into per-backend SPLITS (:1066
  ggml_backend_sched_split_graph), inserting explicit copy nodes at every backend boundary.
- Execution: ggml_backend_sched_compute_splits (:1654) walks splits in order: cross-backend sync
  via events, copies of the few-KB activations, then ONE ggml_backend_graph_compute_async per
  split. The CPU split (3x MUL_MAT_ID + silu-mul + weighted add for a layer) runs as a single
  graph_compute with the whole threadpool - one pool join per split, not per op. The host thread
  does participate per layer (CPU graph_compute blocks), but the per-layer cost is an 8-32 KB
  copy + enqueue, NOT a routing round trip: the router matmul + top-k run on the CPU backend
  itself (they are nodes of the CPU split), so no routing data crosses backends.
- Bonus: when weights would be copied, the sched has a used-experts-only copy optimization
  (ggml-backend.cpp:2025-2033) - llama.cpp's own "expert paging" hybrid. With --n-cpu-moe=999
  nothing is copied.

What FreeToken's engine would need for a CPU-resident-MoE mode:
- Today _decode_hybrid (layers/moe.py:299-341) per layer does: device-side routing split ->
  cache.ensure_experts_hybrid (host bookkeeping) -> executor.decode_submit (host node) -> PCIe
  copy_missing -> GPU GEMM -> executor.decode_sync (host node). Two host round trips + a routing
  D2H sync per layer x 42 = the measured 80-127 ms/step floor (PLAN.md:82,
  verification/bisect-notes.md).
- A CPU-resident mode removes the entire cache protocol (no ensure/copy_missing - banks are
  already the permanent owner, like n-cpu-moe). What remains is the per-layer activation
  handoff GPU->CPU->GPU. Minimum viable: keep the per-layer structure but replace
  decode_submit/decode_sync host nodes with (a) routing computed on the CPU pool (a 288x4096
  router GEMV is 2.4 MB - cheap) or ids passed via device-written pinned flags, and (b) the
  activation D2H as an async memcpy into a pinned slot polled by the pool. The cuMemOp64
  primitives for device-flag CPU waiting ALREADY EXIST in cpu_moe_ext.cpp (:596-664,
  cumemop_submit/cumemop_sync, cuStreamWaitValue64) - the Task 06 research found memop h=3.02 ms
  vs hostfunc h=1.90 ms with an irreducible floor ~0.1-0.3 ms/layer (research/
  task06-rework-estimate.md), so the realistic target is one submission per step (pool works
  through all 42 layers' MoE as one job), not literally zero round trips. The cross-layer
  dependency (block k+1 routing consumes block k MERGED output) is architecturally forced - a
  single async submission per decode step still means GPU-attn(k) <-> CPU-MoE(k) ping-pong per
  layer, just with cheap handoffs.

Bottom line: kernels alone do not fix hybrid (Task 06 verdict stands). Kernels + CPU-resident
mode + cheap per-layer handoff is the configuration that can match llama.cpp.

## 5. Port map into cpu_moe_ext.cpp

Vendoring discipline (CONTRIBUTING): the existing CUDA-side gguf files (kernel/csrc/gguf/*.cuh)
are verbatim copies from vLLM's csrc/quantization/gguf (vecdotq.cuh:1-3 states "copied from
vllm ... copied and adapted from llama.cpp b2899 ggml-cuda/vecdotq.cuh and mmq.cu") - i.e. our
vendored chain is llama.cpp-derived but two hops away and carries no license headers (a gap).
llama.cpp is MIT (LICENSE:1-3) - verbatim host transcription is permitted with an attribution
comment; per repo rules the user must own/explain every line, and local private work only (the
glm5next workstream is explicitly private-use, upstream gate waived). sgl-kernel: NOT installed
in the FreeToken venv (pip list: no sgl*), and sgl-kernel ships CUDA kernels, not CPU int dots -
no reusable kernel there; the CPU int dots exist only in ggml itself (and its forks/downstreams:
vLLM ships only the CUDA versions).

Transcribe (verbatim host copies, MIT attribution comment):
- mul_add_epi8 + hsum_float_8 + get_scale_shuffle helpers (~40 LOC).
- ggml_vec_dot_iq3_xxs_q8_K AVX2 (~75 LOC), ggml_vec_dot_iq4_xs_q8_K AVX2 (~70 LOC),
  ggml_vec_dot_q6_K_q8_K AVX2 (~85 LOC). Input signature changes from (row, bf16 x, K) to
  (row, const block_q8_K* rows, K) - a new fn type `ggufdot_i8_fn(const uint8_t* row, const
  int8_t* q8, const float* q8d, const int16_t* q8b, int K)` or a packed q8_K struct.
- Optionally a scalar q8_K reference kernel for tests (~60 LOC, modeled on the *_generic
  versions in ggml/src/ggml-cpu/quants.c:851, :1050, :1283) - the scalar bf16 kernels already
  serve as value-parity reference, but an int-activation scalar makes A/B of ISA tiers clean.

Changes:
- New q8_K quantizer quant_q8_k (~40 LOC, section 2), scratch sizing (+~5 MB/token
  at :1657-1661), and per-token/per-route quant calls mirroring quant_q8_0 at :2205-2224.
- ISA tier: reuse ISA_AVX2 (pick_isa at :681-707 already detects avx2+fma); extend select_ggufdot
  (:1404-1410) to return the i8 kernel per tier (the comment at :1398-1401 already reserves this
  slot). No new enum needed; AVX-512 variants can come later (ggml's AVX512 path for these
  kernels is the same maddubs code - AVX2 is the perf-relevant tier on Zen4/5 for int8 anyway).
- Executor: gemm1_dot/gemm2_dot gain a gguf-i8 branch (~40 LOC); decode/prefill prepare phases
  call quant_q8_k where they today quant_q8_0/leave bf16.
- Python boundary: NONE - formats, banks, and executor interfaces are unchanged; activation prep
  stays inside the C++ executor exactly like q4_0/nvfp4 today (the existing python-side
  construction guard remains sufficient).

Rough total: 350-450 LOC in cpu_moe_ext.cpp (+ tests). Risk: LOW for numerics (byte-identical
layouts + verbatim LUTs; parity testable against the existing scalar kernels, the gguf-py
reference dequant, and the analytic fixtures from tests/moe/test_cpu_moe_gguf_iq.py). MEDIUM for
the executor prepare-phase surgery (touches the decode fast path for all gguf formats; the
dfloat contract tests in tests/kernels/test_gguf_quant.py must stay green). No API/ABI risk.

## 6. Projected numbers (PROJECTIONS, not measurements)

Basis: benchbw Task 03 measured the CURRENT scalar gguf GEMV leg at 11.71 GB/s vs STREAM
75.84 GB/s on this rig (PLAN.md:130); scalar ~2.5-4 cyc/elem vs projected 0.1-0.25 cyc/elem for
the transcribed kernels. The kernels are memory-bound above ~20 GB/s/core aggregate, so the leg
should approach a large fraction of STREAM. Assume 45-60 GB/s effective single-pass expert
streaming (55-80% of STREAM; conservative for one read stream + tiny reused activation row).

- CPU-only decode (MoE on CPU pool, attention on GPU), kernels ported:
  MoE bytes 3.66 GB/token -> 61-81 ms -> 12.3-16.4 tok/s MoE-only. Adding attention + step
  overhead ~10-15 ms (offload control does ~70 ms total with 72 ms of PCIe fetch, so overhead is
  small): ~10-14 tok/s end-to-end. The user's llama.cpp 10 tok/s on the same file/box is the
  empirical anchor; 10-13 is the honest central estimate.
- CPU-only decode WITHOUT the scheduling fix (per-layer hostfunc round trip kept, h=1.9 ms x 42 =
  ~80 ms/step, task06-rework-estimate.md): 3.66 GB/45-60 GB/s + 80 ms = 141-176 ms -> ~6-7 tok/s.
  The round-trip floor, not the kernel, is the binding constraint.
- Hybrid (GPU 78% + CPU 22% overflow) with fast kernels but current architecture: handshake floor
  + PCIe fetch dominate; modeled ceiling 3.9-4.3 tok/s (lever B), 5.7-6.1 with cheap handoffs
  (B+A), 7.5-7.9 (B+A+C) - task06-rework-estimate.md; realistic landing 8-11 even with all levers,
  i.e. at best a tie with the 14.25 offload control. Offload remains the recommendation for this
  file; the port's real prize is a CPU-ONLY mode (frees the entire MoE VRAM envelope, ~134 GB) at
  ~2x-4.5x today's ~3-5 tok/s projection, and a much stronger CPU leg for any future expert-paging
  scheme (it would beat PCIe 40.1 GB/s, flipping Task 03's offload verdict IF the leg sustains
  >40 GB/s - which the port is exactly what could enable).

The ONE microbenchmark to run first (before any port commitment):
extend benchbw's gguf leg (added in 2169baa) with the transcribed AVX2 kernels and measure
sustained single-pass GB/s on the real block mix (98/136/210 B, top_k=8 working set, 16 vs 20
threads), A/B: scalar vs avx2 tier. Gate: >= 40 GB/s sustained (i.e. >= 3.4x the 11.71 GB/s
scalar leg). That single number (a) validates the 10-14 tok/s CPU-only projection, (b) decides
whether a CPU-resident mode can outrun PCIe 40.1 GB/s, and (c) costs 1-2 days with zero engine
changes (FREETOKEN_CPU_MOE_ISA=avx2 A/B already wired).

## Recommendation

Do the kernel port (350-450 LOC, low risk, mechanical, MIT-clean) gated on the benchbw A/B
microbench. Do NOT expect it to fix hybrid - pair it with the Task 06 re-scope work (cheap
per-layer handoffs; the cuMemOp64 polling primitive already exists) and aim it at a
CPU-resident-MoE mode whose value is VRAM freedom at ~10-13 tok/s, not hybrid throughput. Keep
offload as the default for this file.
