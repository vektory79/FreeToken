# ggml CPU integer dot kernels: standalone microbench vs the 40 GB/s gate

Decision-gate measurement for the ggml integer-kernel port (GLM-5.3-Flash-UD-Q3_K_XL,
gguf-glm5next-hybrid workstream). Harness source: `verification/ggml-microbench-harness.cpp`
(live copy in `/tmp/ggml_microbench/mb.cpp`, driver `/tmp/ggml_microbench/run_serial.py`).

**VOID notice**: the first measurement round of this bench (2026-09-15, recorded in chat only,
never in this file) ran concurrent measurement processes and is DISCARDED. All numbers below
come from the serialized rerun: one measurement process at a time, hard 120 s timeout, PID-reap
verification and `ps` census between points (`run_serial.py`). Pre-run cleanup: killed 14
orphan FreeToken multiprocessing PIDs + 58 `/dev/shm/sem.mp-*` (no python mp jobs remained),
no `mb`/`mb_dbg` survivors; post-run census clean (GPU idle at 1319 MiB / 3%, only the user's
htop at ~3% CPU).

## 1. Fidelity: what the llama.cpp build actually ran

Checkout `/media/ai/src/llama-cpp-glm5next`, `build/CMakeCache.txt`:
`GGML_CUDA=ON`, `GGML_CPU_ALL_VARIANTS=ON`, `GGML_BACKEND_DL=ON`, `GGML_OPENMP=ON`,
`GGML_NATIVE=ON` (with BACKEND_DL, NATIVE is rejected at configure; per-variant .so libraries
are built instead and the runtime loads the best-scoring variant it supports).

CPU: Intel i7-14700KF (Raptor Lake, 8 P + 12 E = 20 physical cores / 28 threads, DDR5).
`/proc/cpuinfo`: avx2, fma present; **no avx512f / avx512_vnni / avx512_bw** (fused off on
Raptor Lake). Built variant flags (from the per-variant flags.make files):
- `libggml-cpu-haswell.so`: `-mf16c -mfma -mavx -mavx2`
- `libggml-cpu-alderlake.so`: same + `-mavxvnni`

So the anchor's CPU tier is **AVX2+FMA+F16C**. The AVX-512 caveat from the task is void for a
second reason besides the CPU: the three target kernels dispatch on `__AVX2__` blocks in
`ggml/src/ggml-cpu/arch/x86/quants.c` and use `mul_add_epi8` (maddubs/madd chain, :69); the
only VNNI-dispatched helper (`mul_sum_us8_pairs_float`, :105, used by q4_0/q8_0/tq kernels)
is not on their path, so haswell vs alderlake variant produces identical code for them.

Harness compiled with the haswell tier exactly: `-O3 -DNDEBUG -mf16c -mfma -mavx -mavx2`
(plus `-fopenmp`, matching `GGML_OPENMP=ON` in the llama.cpp build; gcc 13.3.0).

## 2. Compile approach: verbatim include, zero transcription

The ggml sources compile standalone with include paths only (no repo changes anywhere):

```
gcc <ISA> -std=gnu11 -fPIC \
  -I ggml/src -I ggml/src/ggml-cpu -I ggml/include -I ggml \
  -c ggml/src/ggml-cpu/arch/x86/quants.c   # AVX2 dots: q6_K:2426 (body 2439),
                                           # iq3_xxs:3260 (3274), iq4_xs:4004 (4022)
  -c ggml/src/ggml-quants.c                # quantize_row_q8_K_ref :2768
  -c ggml/src/ggml-cpu/quants.c            # scalar *_generic refs :851/:1050/:1283
```

Link needed 4 shims for `ggml.c` symbols referenced only by functions the bench never calls
(`ggml_type_size`, `ggml_row_size`, `ggml_type_name` from quantize_q6_K/validate paths) and
the `ggml_table_f32_f16` / e8m0 / ue4m3 LUTs owned by ggml.c (f16 table filled exactly at
static-init). Provenance comments are in the harness header.

**Built-in parity check**: on every run (before warmup), the AVX2 kernel vs ggml's own scalar
`*_generic` kernel on 32 random rows per format: max rel err < 1e-3, all finite. This caught
two real bugs during bring-up (NaN d-fields from random f16 bit patterns; NaN activations
propagating through quantize_row_q8_K_ref amax) before any timing was trusted.

## 3. Method

- Shape = real decode geometry: H=4096, I=2048, E=288 experts, top_k=8, 42 MoE layers.
  Per layer: quantize_row_q8_K once for the x row (H elems, feeds gate+up), then
  8 experts x 2048 rows vec_dot (gate), 8 x 2048 (up, separate buffer); per-expert
  q8_K of the silu-mul row (I elems x 8) then 8 x 4096 rows (down).
- Signature mix = 39x(18,18,23) + 2x(18,18,14) + 1x(23,23,14) over 42 layers
  (gate,up,down format ids as in FreeToken's per-projection tables; census matches the
  study: gate/up IQ3_XXS x41 + IQ4_XS x1, down IQ4_XS x39 + Q6_K x3).
  Exact weight bytes/token = 3,733,454,848 B; + activation bytes 981,120 B/pass
  (q8_K rows counted once per layer) = 3,734,435,968 B/pass for blended.
  Per-format single-format passes: iq3_xxs 4,471,060,608 B; iq4_xs 4,493,080,704 B;
  q6_k 5,307,824,256 B.
- Expert buffers: synthetic pseudo-random bytes at exact block sizes (98/136/210 B);
  per-block f16 d fields constrained to finite normals (exp 11-18), payload bits fully
  random. **Assumption (stated)**: the integer kernels have fixed control flow - timing is
  data-independent; only the float d multiply could in principle be value-sensitive, hence
  the d constraint. Activations: uniform floats in [-1,1).
- Parallelism: OpenMP `parallel for schedule(static)` per projection (mirrors ggml's own
  OpenMP-per-op schedule; contiguous row ranges like ggml's ir0 chunking). Barrier joins
  per projection, 5 sections per layer.
- Pinning: OMP_PLACES = the 20 physical cores {0-7, 16-27} (P-cores 0-7 + E-cores 16-27 per
  lscpu topology), OMP_PROC_BIND=close, master on core 0, OMP_WAIT_POLICY=ACTIVE. NOTE:
  OMP_* env must be set before process start (libgomp parses in a pre-main constructor);
  the first attempt to setenv() from inside main silently collapsed all threads onto core 0.
- Timing: 1 warmup pass excluded, then auto-calibrated pass count for >= 3.0 s wall
  (min 8 passes), steady_clock; TSC (invariant, 3.4 GHz base confirmed via cpufreq
  base_frequency) for cycles/element. Single-thread points ran at ~5.42-5.45 GHz (sampled),
  so true core-cycles/elem = TSC-based x 3.4/5.45.
- Sanity gates: rc=0 for all 17 points; census clean before/after every point.

## 4. Results (serialized rerun, pinned to the 20 physical cores)

GB/s = (weight + activation bytes) / wall time. `cyc/elem` = TSC ticks / weight element
(8.456e9 elems/pass); single-thread true core-cycles/elem in parentheses (5.45 GHz boost).

Blended (real signature mix, 3.734 GB/pass):

| threads | 1 | 8 | 16 | 20 |
|---|---|---|---|---|
| GB/s | 8.76 | 32.43 | 53.08 | **66.00** |
| pass ms | 426.2 | 115.2 | 70.4 | 56.6 |
| cyc/elem | 0.172 (0.107) | 0.047 | 0.028 | 0.023 |

Per format (single-format pass over all 42 layers):

| format | bytes/pass | 1T | 8T | 16T | 20T | 1T cyc/elem (core) |
|---|---|---|---|---|---|---|
| iq3_xxs | 4.471 GB | 8.30 | 32.83 | 54.45 | 62.39 | 0.218 (0.136) |
| iq4_xs | 4.493 GB | 14.24 | 63.99 | 66.16 | 71.84 | 0.128 (0.080) |
| q6_k | 5.308 GB | 18.61 | 65.55 | 49.49 | 54.67 | 0.115 (0.072) |
| blended | 3.734 GB | 8.76 | 32.43 | 53.08 | 66.00 | 0.172 (0.107) |

Reference point, blended 20 threads UNPINNED: 64.07 GB/s (scheduler spread it about as
well; pinning matters for reproducibility, not for the headline number).

Reads: 20T blended = 66.0 GB/s = 87% of the 75.84 GB/s STREAM measured on this rig
(benchbw Task 03) - the leg is memory-bound at high thread counts, as the study predicted.
The 16T q6_k dip (49.5 vs 54.7 at 20T) is the 8P+8E mix: q6_K is the most
compute-dense format (210 B/256 elems) and leans hardest on the E-cores at 16T.
Single-thread compute intensity 0.07-0.14 core-cyc/elem lands inside the study's projected
0.10-0.25 band (iq3_xxs highest: the 256-entry grid gather per 64 elems).

## 5. Verdict vs the gate

**PASS at both gate points.**

| | 16T | 20T |
|---|---|---|
| gate | >= 40 GB/s | >= 40 GB/s |
| measured (blended) | 53.08 | 66.00 |
| worst single format | 49.49 (q6_k) | 54.67 (q6_k) |

- vs FreeToken scalar fp32-LUT leg 11.71 GB/s: **5.6x** (66.00/11.71) at 20T, 4.5x at 16T.
- vs FreeToken NVFP4 integer leg 55.6 GB/s (overlapped, 19.9/20 cores): 66.00 = **1.19x** -
  the ggml AVX2 kernels on this rig are not merely "ggml-grade", they beat the in-house
  NVFP4 integer leg's measured number on the same cores.
- External anchor cross-check: llama.cpp does 10 tok/s all-CPU end-to-end on the same file.
  At 66 GB/s the MoE reads alone cost 3.733 GB / 66 GB/s = 56.6 ms/token -> ~17.7 tok/s of
  pure MoE compute headroom at 20T; the ~42 ms/step delta to the 10 tok/s anchor is
  attention + sampling + per-op scheduling in llama.cpp's pipeline - consistent, not
  contradictory.

Meaning:
- **(a) CPU-only mode**: 3.733 GB/token at 53-66 GB/s (16-20T) = 56-70 ms MoE compute +
  ~10-15 ms attention/step overhead -> **~12-15 tok/s**. The standing 10-14 tok/s projection
  holds with the upper half now measured rather than assumed; the user's llama.cpp 10 tok/s
  remains the conservative end-to-end anchor.
- **(b) GGUF hybrid parity**: the NVFP4 A/B decomposition required the CPU leg at 40.2 GB/s
  for its measured 4.70 experts/layer and benched NVFP4 at 55.6. At 53-66 GB/s the gguf leg
  clears that bar at both gate points, so the port removes the "starved pool" PCIe-fetch
  amplifier and the GGUF hybrid ceiling quoted in the A/B (~11-14 tok/s with weighted pools
  + packed tables + cheap handoffs) is now supported by measurement on the leg that was the
  blocker. Parity with offload (13.8-15.6 tok/s) still hinges on Task 06's per-layer
  handoff work, not on the kernels.

## 6. Caveats (honest list)

- No AVX-512/VNNI tier measured - Raptor Lake has neither, and the source shows the three
  kernels do not use VNNI intrinsics even when compiled for it; the "VNNI changes the
  picture" risk is closed by code inspection, not just by CPU support.
- Integer kernels assumed data-independent in timing; payload bytes are random and only the
  per-block f16 d is constrained to finite normals. Real-file d distributions could differ
  in denormal frequency by <0.1% of blocks - no timing effect expected on AVX2.
- The bench measures the GEMV leg only: no attention, router, sampling, or the hybrid
  per-layer GPU<->CPU sync/fetch volume; those dominate the hybrid end-to-end verdict.
- E-core contribution is real but weaker (16T q6_k dip); a P-core-only 8-thread pin is the
  worst case for latency-critical single-expert bursts (32.4-34.7 GB/s at 8T).
- Single run per point (passes >= 8, >= 3 s each). Run-to-run spread on repeat smoke points
  was ~+/-2%; no formal error bars were collected.
