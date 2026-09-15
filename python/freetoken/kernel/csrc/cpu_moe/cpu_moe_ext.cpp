// CPU-compute MoE executor for the "cpu" offload backend.
//
// Decode ships activations to the CPU, computes the routed experts here (reading
// the pinned host expert banks at full RAM bandwidth), and ships the results
// back. To keep the whole decode path inside a single CUDA graph we expose
// submit/sync as host nodes via cudaLaunchHostFunc -- the callbacks only touch a
// CPU worker pool + pinned host buffers and never call any CUDA API.
//
// One task is in flight at a time (per MoE layer): submit() wakes the pool,
// sync() blocks the host-func thread until the pool drains. The heavy GEMV runs
// on the persistent worker threads, not the host-func thread.
//
// Weight formats: bf16, NVFP4, MXFP4, ds_fp4 and Q4_0 expert banks (see WFmt and
// the per-format bank schemas). Compute is FP32-accumulate; the intermediate is
// stored bf16 to match the GPU decode path. ISA is chosen once at construction
// (AVX-512-BF16 dpbf16 -> AVX-512F widening -> AVX2+FMA -> scalar).

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime_api.h>
#include <torch/extension.h>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#define CPU_MOE_HAS_AFFINITY 1
#else
#define CPU_MOE_HAS_AFFINITY 0
#endif

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define CPU_MOE_X86 1
#else
#define CPU_MOE_X86 0
#endif

namespace {

using bf16_t = uint16_t;

inline float bf16_to_f32(bf16_t v) {
  uint32_t u = static_cast<uint32_t>(v) << 16;
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

inline bf16_t f32_to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  // round-to-nearest-even
  const uint32_t lsb = (u >> 16) & 1u;
  u += 0x7fffu + lsb;
  return static_cast<bf16_t>(u >> 16);
}

// ACT_SWIGLUOAI is the clamped (up + 1) swiglu (gpt-oss "swigluoai" /
// MiniMax-M3): gate/up are combined jointly with the runtime alpha/limit
// scalars, so it is handled in the do_pass1 epilogue (act_apply never sees it;
// the mxfp4 kernel additionally fuses its own copy of the same math).
// ACT_SWIGLU_CLAMP (GLM-5.3 "swiglu_limit") is the same clamped form WITHOUT
// the (up + 1) bias: clamp(gate, max=lim) * sigmoid(alpha*gate) * clamp(up, +-lim).
enum ActKind {
  ACT_SILU = 0,
  ACT_GELU = 1,
  ACT_GELU_TANH = 2,
  ACT_SWIGLUOAI = 3,
  ACT_SWIGLU_CLAMP = 4,
};

inline float act_apply(int act, float x) {
  if (act == ACT_SILU) return x / (1.0f + std::exp(-x));
  if (act == ACT_GELU)
    return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752440f));
  // gelu_tanh
  const float k0 = 0.79788456080286535588f;  // sqrt(2/pi)
  const float inner = k0 * (x + 0.044715f * x * x * x);
  return 0.5f * x * (1.0f + std::tanh(inner));
}

// ------------------------------- dot products -------------------------------
// dot(weight[bf16], act[bf16], n) -> fp32. The selected impl is a function
// pointer chosen at runtime; the per-row call overhead is negligible vs n.

using dot_fn = float (*)(const bf16_t*, const bf16_t*, int);

float dot_scalar(const bf16_t* w, const bf16_t* x, int n) {
  float acc = 0.0f;
  for (int i = 0; i < n; ++i) acc += bf16_to_f32(w[i]) * bf16_to_f32(x[i]);
  return acc;
}

// Software prefetch distance (bytes) ahead of the current weight row stream. The
// weight stream is the bandwidth bottleneck (read once, never reused); nudging the
// HW prefetcher with a few cache lines of lookahead raises sustained throughput.
constexpr int PF_AHEAD = 512;

#if CPU_MOE_X86
__attribute__((target("avx512f")))
float dot_avx512f(const bf16_t* w, const bf16_t* x, int n) {
  // 4 independent accumulators -> more in-flight loads (memory-level parallelism),
  // which is what lifts a bandwidth-bound GEMV toward peak.
  __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
  __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
  int i = 0;
  for (; i + 64 <= n; i += 64) {
    _mm_prefetch(reinterpret_cast<const char*>(w + i) + PF_AHEAD, _MM_HINT_T0);
    for (int j = 0; j < 64; j += 16) {
      __m256i wi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i + j));
      __m256i xi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i + j));
      __m512 wf = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(wi), 16));
      __m512 xf = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(xi), 16));
      __m512& acc = (j == 0) ? a0 : (j == 16) ? a1 : (j == 32) ? a2 : a3;
      acc = _mm512_fmadd_ps(wf, xf, acc);
    }
  }
  for (; i + 16 <= n; i += 16) {
    __m256i wi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i));
    __m256i xi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i));
    __m512 wf = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(wi), 16));
    __m512 xf = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(xi), 16));
    a0 = _mm512_fmadd_ps(wf, xf, a0);
  }
  float s = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3)));
  for (; i < n; ++i) s += bf16_to_f32(w[i]) * bf16_to_f32(x[i]);
  return s;
}

#if (defined(__GNUC__) && __GNUC__ >= 10) || defined(__clang__)
#define CPU_MOE_HAS_AVX512BF16 1
__attribute__((target("avx512bf16,avx512f")))
static inline __m512bh load_bh(const bf16_t* p) {
  __m512i raw = _mm512_loadu_si512(reinterpret_cast<const void*>(p));
  __m512bh out;
  std::memcpy(&out, &raw, sizeof(out));
  return out;
}

__attribute__((target("avx512bf16,avx512f")))
float dot_avx512bf16(const bf16_t* w, const bf16_t* x, int n) {
  // 4 accumulators (128 bf16/iter) for memory-level parallelism + a prefetch nudge.
  __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
  __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
  int i = 0;
  for (; i + 128 <= n; i += 128) {
    _mm_prefetch(reinterpret_cast<const char*>(w + i) + PF_AHEAD, _MM_HINT_T0);
    a0 = _mm512_dpbf16_ps(a0, load_bh(w + i), load_bh(x + i));
    a1 = _mm512_dpbf16_ps(a1, load_bh(w + i + 32), load_bh(x + i + 32));
    a2 = _mm512_dpbf16_ps(a2, load_bh(w + i + 64), load_bh(x + i + 64));
    a3 = _mm512_dpbf16_ps(a3, load_bh(w + i + 96), load_bh(x + i + 96));
  }
  for (; i + 32 <= n; i += 32) {
    a0 = _mm512_dpbf16_ps(a0, load_bh(w + i), load_bh(x + i));
  }
  float s = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3)));
  for (; i < n; ++i) s += bf16_to_f32(w[i]) * bf16_to_f32(x[i]);
  return s;
}
#endif  // avx512bf16 available

// --------- AVX2 + FMA (256-bit fallback for CPUs without AVX-512) ----------
// Covers Intel 12-14th gen / Arrow Lake (AVX-512 fused off) and AMD Zen<4. The
// bf16->fp32 widen is a zero-extend + <<16; with 4 independent accumulators the
// GEMV is memory-bandwidth bound, same as the AVX-512 path (just half the width).
__attribute__((target("avx2,fma")))
inline float hsum256(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  lo = _mm_add_ps(lo, _mm256_extractf128_ps(v, 1));
  lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
  lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 0x55));
  return _mm_cvtss_f32(lo);
}

__attribute__((target("avx2,fma")))
float dot_avx2(const bf16_t* w, const bf16_t* x, int n) {
  __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
  __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
  int i = 0;
  for (; i + 32 <= n; i += 32) {
    _mm_prefetch(reinterpret_cast<const char*>(w + i) + PF_AHEAD, _MM_HINT_T0);
    for (int j = 0; j < 32; j += 8) {
      __m128i wi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i + j));
      __m128i xi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + i + j));
      __m256 wf = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(wi), 16));
      __m256 xf = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(xi), 16));
      __m256& acc = (j == 0) ? a0 : (j == 8) ? a1 : (j == 16) ? a2 : a3;
      acc = _mm256_fmadd_ps(wf, xf, acc);
    }
  }
  for (; i + 8 <= n; i += 8) {
    __m128i wi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i));
    __m128i xi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + i));
    __m256 wf = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(wi), 16));
    __m256 xf = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(xi), 16));
    a0 = _mm256_fmadd_ps(wf, xf, a0);
  }
  float s = hsum256(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
  for (; i < n; ++i) s += bf16_to_f32(w[i]) * bf16_to_f32(x[i]);
  return s;
}
#endif  // CPU_MOE_X86

// --------------------------- NVFP4 (W4A16) dequant ---------------------------
// Weights: e2m1 4-bit codes (2/byte, low nibble first), per-16 block scale in
// fp8-e4m3, per-output-row global scale in fp16. Dequant matches the GPU kernels
// (freetoken/kernel/triton/nvfp4_dequant.py): w = E2M1[code] * e4m3(scale) * global.
// Activations stay bf16 (W4A16); the GEMV dequantizes weights inside the K-loop.

const float kE2M1[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                         -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

// e2m1 * 2 as exact int8 (all e2m1 values are multiples of 0.5). Used by the AVX-VNNI
// W4A8 path: nibble -> int8 weight via PSHUFB LUT, then VPDPBUSD against int8 activations;
// the *2 is undone by a 0.5 folded into the final scale. Mirrors ggml's kvalues_mxfp4.
alignas(16) const int8_t kE2M1x2[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

inline float fp16_to_f32(uint16_t h) {
  const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t man = h & 0x3FFu;
  uint32_t f;
  if (exp == 0) {
    if (man == 0) {
      f = sign;
    } else {
      exp = 127 - 15 + 1;
      while ((man & 0x400u) == 0) {
        man <<= 1;
        --exp;
      }
      man &= 0x3FFu;
      f = sign | (exp << 23) | (man << 13);
    }
  } else if (exp == 0x1Fu) {
    f = sign | 0x7F800000u | (man << 13);
  } else {
    f = sign | ((exp + (127 - 15)) << 23) | (man << 13);
  }
  float out;
  std::memcpy(&out, &f, sizeof(out));
  return out;
}

// e4m3 (OCP "fn": finite, max-normal 448, exp bias 7). Decoded into a 256-entry LUT.
inline float e4m3_decode(uint8_t v) {
  const float sign = (v & 0x80u) ? -1.0f : 1.0f;
  const uint32_t exp = (v >> 3) & 0xFu;
  const uint32_t man = v & 0x7u;
  if (exp == 0) return sign * (man / 8.0f) * 0.015625f;  // 2^(1-7) = 2^-6
  return sign * (1.0f + man / 8.0f) * std::ldexp(1.0f, (int)exp - 7);
}

// Activations pre-deinterleaved to fp32 (xe[m]=x[2m], xo[m]=x[2m+1]); see the
// ds_fp4 dot below for why (drops the hot loop to ~1.5 shuffle ops / 16 weights).
using nvdot_fn = float (*)(const uint8_t*, const uint8_t*, float, const float*, const float*,
                           int, const float*, const float*);

float dot_nvfp4_scalar(const uint8_t* packed, const uint8_t* scale, float global,
                       const float* xe, const float* xo, int K, const float* e2m1,
                       const float* e4m3) {
  float acc = 0.0f;
  const int nb = K / 16;
  for (int b = 0; b < nb; ++b) {
    const float bs = e4m3[scale[b]];
    const uint8_t* pk = packed + (size_t)b * 8;
    const float* xeb = xe + (size_t)b * 8;  // 16 K -> 8 even + 8 odd
    const float* xob = xo + (size_t)b * 8;
    float bsum = 0.0f;
    for (int j = 0; j < 8; ++j) {
      const uint8_t byte = pk[j];
      bsum += e2m1[byte & 0xF] * xeb[j];
      bsum += e2m1[byte >> 4] * xob[j];
    }
    acc += bs * bsum;
  }
  return acc * global;
}

// ---- NVFP4 W4A8 (int8 activations) dot: nibble->int8 via LUT, per-16 act scale ----
// asi8: int8 activations laid out per-16 block as [even(8), odd(8)]; asb[b] = per-block
// activation scale (absmax/127). Result folds the e2m1*2 -> *0.5 into the scale.
using nvi8dot_fn = float (*)(const uint8_t*, const uint8_t*, float, const int8_t*, int,
                             const float*, const float*);

[[maybe_unused]] float dot_nvfp4_i8_scalar(const uint8_t* packed, const uint8_t* scale,
                          float global, const int8_t* asi8, int K, const float* e4m3,
                          const float* asb) {
  float acc = 0.0f;
  const int nb = K / 16;
  for (int b = 0; b < nb; ++b) {
    const uint8_t* pk = packed + (size_t)b * 8;
    const int8_t* ae = asi8 + (size_t)b * 16;       // even(8)
    const int8_t* ao = ae + 8;                      // odd(8)
    int isum = 0;
    for (int j = 0; j < 8; ++j) {
      isum += (int)kE2M1x2[pk[j] & 0xF] * (int)ae[j];
      isum += (int)kE2M1x2[pk[j] >> 4] * (int)ao[j];
    }
    acc += (e4m3[scale[b]] * asb[b]) * (float)isum;
  }
  return acc * (0.5f * global);
}

#if CPU_MOE_X86
// AVX2 e2m1 nibble decode: codes (0..15) in 8 int32 lanes -> fp32. AVX2 vpermps is
// only 8-wide, so instead of a 16-entry LUT we use the e2m1 sign/magnitude symmetry:
// value = (code&8 ? - : +) * mag8[code&7], mag8 = e2m1[0..7]. The sign is bit 3 of
// the code shifted into the fp32 sign bit (bit 31). Bit-identical to the e2m1 LUT.
__attribute__((target("avx2,fma")))
inline __m256 e2m1_decode8(__m256i codes, __m256 mag8) {
  __m256 mag = _mm256_permutevar8x32_ps(mag8, _mm256_and_si256(codes, _mm256_set1_epi32(7)));
  __m256i sgn = _mm256_slli_epi32(_mm256_and_si256(codes, _mm256_set1_epi32(8)), 28);
  return _mm256_xor_ps(mag, _mm256_castsi256_ps(sgn));
}

// Two 16-K blocks (16 packed bytes) per iter: lo nibbles -> even-K, hi -> odd-K,
// gathered via two vpermps. The per-16 e4m3 scale differs across the two blocks, so
// it is applied per lane (low 8 lanes = block b, high 8 = block b+1).
__attribute__((target("avx512f")))
inline __m512 nvfp4_blk2(const uint8_t* pk, const float* xeb, const float* xob, __m512 lut,
                         __m512i loma, float s0, float s1) {
  __m512i wi = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pk)));
  __m512 vlo = _mm512_permutexvar_ps(_mm512_and_si512(wi, loma), lut);
  __m512 vhi = _mm512_permutexvar_ps(_mm512_and_si512(_mm512_srli_epi32(wi, 4), loma), lut);
  __m512 prod = _mm512_fmadd_ps(vlo, _mm512_loadu_ps(xeb), _mm512_mul_ps(vhi, _mm512_loadu_ps(xob)));
  // lanes 0-7 (block b) -> s0, lanes 8-15 (block b+1) -> s1  (pure AVX512F mask move)
  __m512 scv = _mm512_mask_mov_ps(_mm512_set1_ps(s0), 0xFF00, _mm512_set1_ps(s1));
  return _mm512_mul_ps(prod, scv);
}

__attribute__((target("avx512f")))
float dot_nvfp4_avx512(const uint8_t* packed, const uint8_t* scale, float global,
                       const float* xe, const float* xo, int K, const float* e2m1,
                       const float* e4m3) {
  const __m512 lut = _mm512_loadu_ps(e2m1);  // 16 e2m1 values
  const __m512i loma = _mm512_set1_epi32(0xF);
  __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
  const int nb = K / 16;  // 8 packed bytes + one e4m3 scale per block
  int b = 0;
  for (; b + 4 <= nb; b += 4) {  // two blk2 calls -> 4 blocks, 2 accumulators
    acc0 = _mm512_add_ps(acc0, nvfp4_blk2(packed + (size_t)b * 8, xe + (size_t)b * 8,
                                          xo + (size_t)b * 8, lut, loma, e4m3[scale[b]], e4m3[scale[b + 1]]));
    acc1 = _mm512_add_ps(acc1, nvfp4_blk2(packed + (size_t)(b + 2) * 8, xe + (size_t)(b + 2) * 8,
                                          xo + (size_t)(b + 2) * 8, lut, loma, e4m3[scale[b + 2]], e4m3[scale[b + 3]]));
  }
  for (; b + 2 <= nb; b += 2)
    acc0 = _mm512_add_ps(acc0, nvfp4_blk2(packed + (size_t)b * 8, xe + (size_t)b * 8,
                                          xo + (size_t)b * 8, lut, loma, e4m3[scale[b]], e4m3[scale[b + 1]]));
  float s = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
  for (; b < nb; ++b) {  // odd final 16-K block
    const uint8_t* pk = packed + (size_t)b * 8;
    const float* xeb = xe + (size_t)b * 8;
    const float* xob = xo + (size_t)b * 8;
    float bsum = 0.0f;
    for (int j = 0; j < 8; ++j) {
      bsum += e2m1[pk[j] & 0xF] * xeb[j] + e2m1[pk[j] >> 4] * xob[j];
    }
    s += e4m3[scale[b]] * bsum;
  }
  return s * global;
}

// AVX2: one 16-K block (8 packed bytes) per call, 8 even + 8 odd lanes.
__attribute__((target("avx2,fma")))
inline __m256 nvfp4_blk_avx2(const uint8_t* pk, const float* xeb, const float* xob,
                             __m256 mag8, float sc) {
  __m256i wi = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(pk)));
  __m256 vlo = e2m1_decode8(_mm256_and_si256(wi, _mm256_set1_epi32(0xF)), mag8);
  __m256 vhi = e2m1_decode8(_mm256_srli_epi32(wi, 4), mag8);
  __m256 prod = _mm256_fmadd_ps(vlo, _mm256_loadu_ps(xeb), _mm256_mul_ps(vhi, _mm256_loadu_ps(xob)));
  return _mm256_mul_ps(prod, _mm256_set1_ps(sc));
}

__attribute__((target("avx2,fma")))
float dot_nvfp4_avx2(const uint8_t* packed, const uint8_t* scale, float global,
                     const float* xe, const float* xo, int K, const float* e2m1,
                     const float* e4m3) {
  const __m256 mag8 = _mm256_loadu_ps(e2m1);  // e2m1[0..7] magnitudes
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
  const int nb = K / 16;
  int b = 0;
  for (; b + 2 <= nb; b += 2) {
    acc0 = _mm256_add_ps(acc0, nvfp4_blk_avx2(packed + (size_t)b * 8, xe + (size_t)b * 8,
                                              xo + (size_t)b * 8, mag8, e4m3[scale[b]]));
    acc1 = _mm256_add_ps(acc1, nvfp4_blk_avx2(packed + (size_t)(b + 1) * 8, xe + (size_t)(b + 1) * 8,
                                              xo + (size_t)(b + 1) * 8, mag8, e4m3[scale[b + 1]]));
  }
  for (; b < nb; ++b)
    acc0 = _mm256_add_ps(acc0, nvfp4_blk_avx2(packed + (size_t)b * 8, xe + (size_t)b * 8,
                                              xo + (size_t)b * 8, mag8, e4m3[scale[b]]));
  return hsum256(_mm256_add_ps(acc0, acc1)) * global;
}

// AVX-VNNI W4A8: decode 8 packed bytes (16 nibbles) of one 16-block to int8 [lo(8),hi(8)]
// via PSHUFB against the e2m1*2 LUT (replaces the 2 vpermps fp32 expands -- ~4x less
// port-5 traffic). lo=even-K weights, hi=odd-K, matching the [even(8),odd(8)] act layout.
__attribute__((target("avx2,avxvnni,fma")))
inline __m128i nvfp4_decode_block_i8(const uint8_t* pk, __m128i lut) {
  __m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(pk));   // 8 bytes
  __m128i lo = _mm_and_si128(b, _mm_set1_epi8(0x0F));
  __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), _mm_set1_epi8(0x0F));
  return _mm_shuffle_epi8(lut, _mm_unpacklo_epi64(lo, hi));            // [lo(8),hi(8)] -> int8
}

// Two 16-blocks per VPDPBUSD (32 int8). Sign trick (ggml mul_add_epi8): |w|*(sign(w)*a)=w*a,
// so u8*s8 needs no offset/correction term. Per-block scale (e4m3 * act-scale) folded in fp32
// (lanes 0-3 -> block b, 4-7 -> block b+1). Bit-faithful weight; only the int8 activation quant
// (W4A8) differs from the bf16 reference.
__attribute__((target("avx2,avxvnni,fma")))
float dot_nvfp4_i8_vnni(const uint8_t* packed, const uint8_t* scale, float global,
                        const int8_t* asi8, int K, const float* e4m3, const float* asb) {
  const __m128i lut = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kE2M1x2));
  __m256 accF = _mm256_setzero_ps();
  const int nb = K / 16;
  int b = 0;
  for (; b + 2 <= nb; b += 2) {
    __m128i wb = nvfp4_decode_block_i8(packed + (size_t)b * 8, lut);
    __m128i wb1 = nvfp4_decode_block_i8(packed + (size_t)(b + 1) * 8, lut);
    __m256i w = _mm256_set_m128i(wb1, wb);                              // [blk b | blk b+1]
    __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(asi8 + (size_t)b * 16));
    __m256i aw = _mm256_sign_epi8(w, w);                               // |w| (u8 operand)
    __m256i sa = _mm256_sign_epi8(a, w);                               // sign(w)*a (s8 operand)
    __m256i di = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), aw, sa);
    __m256 scv = _mm256_blend_ps(_mm256_set1_ps(e4m3[scale[b]] * asb[b]),
                                 _mm256_set1_ps(e4m3[scale[b + 1]] * asb[b + 1]), 0xF0);
    accF = _mm256_fmadd_ps(_mm256_cvtepi32_ps(di), scv, accF);
  }
  float s = hsum256(accF);
  for (; b < nb; ++b) {  // tail (odd block count)
    const uint8_t* pk = packed + (size_t)b * 8;
    const int8_t* ae = asi8 + (size_t)b * 16; const int8_t* ao = ae + 8;
    int isum = 0;
    for (int j = 0; j < 8; ++j)
      isum += (int)kE2M1x2[pk[j] & 0xF] * (int)ae[j] + (int)kE2M1x2[pk[j] >> 4] * (int)ao[j];
    s += (e4m3[scale[b]] * asb[b]) * (float)isum;
  }
  return s * (0.5f * global);
}

#if (defined(__GNUC__) && __GNUC__ >= 10) || defined(__clang__)
#define CPU_MOE_HAS_AVX512VNNI 1

// Software-prefetch distance for the W4A8 weight stream, in 16-K blocks (8 packed
// bytes each). Returns -1 when FREETOKEN_CPU_MOE_PF_BLOCKS is unset: the kernel then
// uses the built-in default min(512 blocks = 4 KB, 2 rows) -- 4 KB is the empirical
// optimum on large-row machines (Emerald Rapids sweep), while the 2-row cap keeps a small-row
// model's overshoot bounded (the executor works in 32-row tiles, so a fixed byte
// distance otherwise prefetches another worker's tile: duplicated DRAM traffic that
// regresses at the bandwidth ceiling). An EXPLICIT env value is honored verbatim
// (no clamp; 0 disables): the per-machine optimum can sit past the safe default (+20%
// at 4 KB on a 24-thread Ice Lake with 256B rows), so the escape hatch must reach it.
// Prefetch never faults, so overshooting a row/bank tail is safe.
static int nvfp4_pf_blocks() {
  static const int v = [] {
    const char* s = getenv("FREETOKEN_CPU_MOE_PF_BLOCKS");
    return (s && s[0]) ? atoi(s) : -1;
  }();
  return v;
}

// AVX-512 VNNI W4A8: FOUR 16-K blocks per VPDPBUSD (64 int8) -- 2x the AVX-VNNI
// (256-bit) path. Decode 32 packed bytes -> 64 int8 with a single _mm512_shuffle_epi8
// (e2m1*2 LUT replicated to all 4 128-bit lanes). AVX-512 has no _mm512_sign_epi8, so
// the u8*s8 sign trick (|w| as u8, sign(w)*a as s8) uses abs_epi8 + a masked negate.
// Bit-faithful weight; only the int8 activation quant (W4A8) differs from bf16.
// One 4-block group (64 int8) -> scaled fp32 partial (16 lanes). Isolated as a helper so
// the caller can run several independent chains into separate accumulators (the decode ->
// dpbusd -> scale chain is long, so a single accumulator leaves the core latency-bound).
__attribute__((target("avx512f,avx512bw,avx512vnni,avx2")))
static inline __m512 nvfp4_i8_grp4(const uint8_t* packed, const uint8_t* scale,
                                   const int8_t* asi8, const float* e4m3, const float* asb,
                                   int b, __m512i lut, __m512i idx, __m512i mask0F,
                                   __m512i idxsc) {
  const __mmask64 hi_half = 0xFF00FF00FF00FF00ULL;  // bytes 8-15 of each 128b lane -> hi nibbles
  __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed + (size_t)b * 8));
  __m512i src = _mm512_permutexvar_epi64(idx, _mm512_castsi256_si512(raw));
  __m512i lo = _mm512_and_si512(src, mask0F);
  __m512i hi = _mm512_and_si512(_mm512_srli_epi16(src, 4), mask0F);
  __m512i comb = _mm512_mask_blend_epi8(hi_half, lo, hi);
  __m512i w = _mm512_shuffle_epi8(lut, comb);  // 64 int8 weights (e2m1*2)
  // u8*s8 sign trick without _mm512_sign_epi8: aw=|w|, sa = (w<0 ? -a : a) (a moot at w==0)
  __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(asi8 + (size_t)b * 16));
  __m512i aw = _mm512_abs_epi8(w);
  __mmask64 neg = _mm512_movepi8_mask(w);
  __m512i sa = _mm512_mask_sub_epi8(a, neg, _mm512_setzero_si512(), a);
  __m512i di = _mm512_dpbusd_epi32(_mm512_setzero_si512(), aw, sa);  // 16 int32 (groups of 4)
  // int32[0..3]->blk b, [4..7]->b+1, [8..11]->b+2, [12..15]->b+3. The 4 block scales
  // (e4m3 LUT x per-block act scale) are computed vectorized: a 4-byte load + epu8->epi32
  // widen + one 4-lane LUT gather + one mul replaces 8 scalar loads + 4 scalar muls +
  // a set_ps assembly, which otherwise dominates the per-group op count.
  int sc_raw;
  memcpy(&sc_raw, scale + b, 4);
  __m128i sc4 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(sc_raw));
  __m128 s4 = _mm_mul_ps(_mm_i32gather_ps(e4m3, sc4, 4), _mm_loadu_ps(asb + b));
  __m512 scv = _mm512_permutexvar_ps(idxsc, _mm512_castps128_ps512(s4));
  return _mm512_mul_ps(_mm512_cvtepi32_ps(di), scv);
}

__attribute__((target("avx512f,avx512bw,avx512vnni,avx2")))
float dot_nvfp4_i8_avx512vnni(const uint8_t* packed, const uint8_t* scale, float global,
                              const int8_t* asi8, int K, const float* e4m3, const float* asb) {
  const __m512i lut = _mm512_broadcast_i32x4(
      _mm_loadu_si128(reinterpret_cast<const __m128i*>(kE2M1x2)));
  const __m512i idx = _mm512_set_epi64(3, 3, 2, 2, 1, 1, 0, 0);  // block i -> 128b lane i
  const __m512i mask0F = _mm512_set1_epi8(0x0F);
  const __m512i idxsc = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
  __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
  __m512 acc2 = _mm512_setzero_ps(), acc3 = _mm512_setzero_ps();
  const int nb = K / 16;
  // Built-in default: min(4 KB, 2 rows) -- see nvfp4_pf_blocks(). An explicitly set
  // env value is used verbatim so a per-machine sweep can reach operating points past
  // the conservative default.
  const int pfb = nvfp4_pf_blocks();
  const int pf = (pfb < 0) ? std::min(512, 2 * nb) : pfb;
  int b = 0;
  // Four independent 4-block groups per iter -> four accumulator chains hide the ~10-op
  // decode->dpbusd->scale latency (1-2 chains leave the loop latency-bound: the per-core
  // rate sat at ~60% of the core's achievable DRAM stream rate).
  for (; b + 16 <= nb; b += 16) {
    // Prefetch the weight stream ahead: the interleaved decode lowers the L1-miss
    // concurrency the HW prefetcher sustains on its own (134 -> 167 GB/s on Emerald Rapids).
    if (pf > 0) {
      _mm_prefetch(reinterpret_cast<const char*>(packed + ((size_t)b + (size_t)pf) * 8),
                   _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char*>(packed + ((size_t)b + (size_t)pf) * 8 + 64),
                   _MM_HINT_T0);
    }
    acc0 = _mm512_add_ps(acc0, nvfp4_i8_grp4(packed, scale, asi8, e4m3, asb, b, lut, idx,
                                             mask0F, idxsc));
    acc1 = _mm512_add_ps(acc1, nvfp4_i8_grp4(packed, scale, asi8, e4m3, asb, b + 4, lut, idx,
                                             mask0F, idxsc));
    acc2 = _mm512_add_ps(acc2, nvfp4_i8_grp4(packed, scale, asi8, e4m3, asb, b + 8, lut, idx,
                                             mask0F, idxsc));
    acc3 = _mm512_add_ps(acc3, nvfp4_i8_grp4(packed, scale, asi8, e4m3, asb, b + 12, lut, idx,
                                             mask0F, idxsc));
  }
  for (; b + 4 <= nb; b += 4)
    acc0 = _mm512_add_ps(acc0, nvfp4_i8_grp4(packed, scale, asi8, e4m3, asb, b, lut, idx,
                                             mask0F, idxsc));
  float s = _mm512_reduce_add_ps(
      _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3)));
  for (; b < nb; ++b) {  // tail (<4 remaining 16-K blocks)
    const uint8_t* pk = packed + (size_t)b * 8;
    const int8_t* ae = asi8 + (size_t)b * 16; const int8_t* ao = ae + 8;
    int isum = 0;
    for (int j = 0; j < 8; ++j)
      isum += (int)kE2M1x2[pk[j] & 0xF] * (int)ae[j] + (int)kE2M1x2[pk[j] >> 4] * (int)ao[j];
    s += (e4m3[scale[b]] * asb[b]) * (float)isum;
  }
  return s * (0.5f * global);
}
#endif  // avx512vnni available
#endif

// =====================================================================================
// CUDA stream memory operations (driver API, resolved via dlopen -- no link-time or
// toolchain dependence). The GPU side of the flag handshake: submit = WRITE_VALUE
// (done[slot]=0 then ready[slot]=1), sync = WAIT_VALUE(done[slot] >= 1). The wait is
// executed by the GPU front-end (no SM-resident kernel), so GPU "utilization" stays
// truthful during CPU compute windows -- a resident spin kernel pinned it at 99%,
// which laptop CPU/GPU dynamic power schedulers answered by clamping the CPU's max
// frequency (GEMV workers -1.5x: the reported edge regression). Availability is
// probed functionally at startup (memops_probe); anything unsupported (Windows WDDM,
// vGPU, old drivers) falls back to the cudaLaunchHostFunc path.
#if defined(_WIN32)
#include <windows.h>
static void* cumemop_dlopen() { return (void*)::LoadLibraryA("nvcuda.dll"); }
static void* cumemop_dlsym(void* h, const char* n) {
  return (void*)::GetProcAddress((HMODULE)h, n);
}
#else
#include <dlfcn.h>
static void* cumemop_dlopen() {
  void* h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
  if (h == nullptr) h = dlopen("libcuda.so", RTLD_LAZY | RTLD_LOCAL);
  return h;
}
static void* cumemop_dlsym(void* h, const char* n) { return dlsym(h, n); }
#endif

using cuMemOp64_fn = int (*)(void* stream, unsigned long long addr, unsigned long long value,
                             unsigned int flags);
static cuMemOp64_fn g_cu_write64 = nullptr;
static cuMemOp64_fn g_cu_wait64 = nullptr;
static constexpr unsigned int kCuWaitValueGeq = 0x0;   // CU_STREAM_WAIT_VALUE_GEQ
static constexpr unsigned int kCuWriteDefault = 0x0;   // CU_STREAM_WRITE_VALUE_DEFAULT

static bool cumemop_resolve() {
  static bool resolved = [] {
    void* h = cumemop_dlopen();
    if (h == nullptr) return false;
    // 11.7+ made the v2 entry points the default; older drivers export only the v1
    // names with the same signature.
    g_cu_write64 = reinterpret_cast<cuMemOp64_fn>(cumemop_dlsym(h, "cuStreamWriteValue64_v2"));
    if (g_cu_write64 == nullptr)
      g_cu_write64 = reinterpret_cast<cuMemOp64_fn>(cumemop_dlsym(h, "cuStreamWriteValue64"));
    g_cu_wait64 = reinterpret_cast<cuMemOp64_fn>(cumemop_dlsym(h, "cuStreamWaitValue64_v2"));
    if (g_cu_wait64 == nullptr)
      g_cu_wait64 = reinterpret_cast<cuMemOp64_fn>(cumemop_dlsym(h, "cuStreamWaitValue64"));
    return g_cu_write64 != nullptr && g_cu_wait64 != nullptr;
  }();
  return resolved;
}

// Functional probe on a scratch pinned int64: enqueue WRITE(7) + WAIT(>=7) + sync.
// Returns true only if the whole memop path works on THIS stream/device/driver.
static bool cumemops_probe(uintptr_t stream, uintptr_t scratch_addr) {
  if (!cumemop_resolve()) return false;
  auto* s = reinterpret_cast<void*>(stream);
  if (g_cu_write64(s, (unsigned long long)scratch_addr, 7ULL, kCuWriteDefault) != 0) return false;
  if (g_cu_wait64(s, (unsigned long long)scratch_addr, 7ULL, kCuWaitValueGeq) != 0) return false;
  return cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)) == cudaSuccess;
}

// GPU side of the flag handshake (see the block comment above): enqueued on the
// caller's (possibly capturing) stream; the WAIT immediate is the constant 1,
// replay-safe under CUDA graphs.
// The startup probe validates EAGER memops; a driver could still reject them at graph
// capture time. Those enqueue errors would otherwise be swallowed here and surface only
// as a later EndCapture failure -- log the first CUresult so triage is one step.
static void cumemop_check(int rc, const char* what) {
  static std::atomic<bool> warned{false};
  if (rc != 0 && !warned.exchange(true)) {
    std::fprintf(stderr,
                 "[freetoken/cpu_moe] %s failed with CUresult=%d (first occurrence; "
                 "subsequent errors are not repeated). If this happened during CUDA "
                 "graph capture, the driver lacks capture support for stream memops -- "
                 "set FREETOKEN_CPU_MOE_FLAG_SYNC=0.\n",
                 what, rc);
  }
}

static void cumemop_submit(uintptr_t stream, uintptr_t done_addr, uintptr_t ready_addr,
                           int64_t slot) {
  auto* s = reinterpret_cast<void*>(stream);
  // Order matters and is preserved by the front end: reset done BEFORE raising ready,
  // so the coordinator's completion write for THIS step can never be wiped.
  cumemop_check(g_cu_write64(s, (unsigned long long)(done_addr + (size_t)slot * 8), 0ULL,
                             kCuWriteDefault),
                "cuStreamWriteValue64(done)");
  cumemop_check(g_cu_write64(s, (unsigned long long)(ready_addr + (size_t)slot * 8), 1ULL,
                             kCuWriteDefault),
                "cuStreamWriteValue64(ready)");
}

static void cumemop_sync(uintptr_t stream, uintptr_t done_addr, int64_t slot) {
  cumemop_check(g_cu_wait64(reinterpret_cast<void*>(stream),
                            (unsigned long long)(done_addr + (size_t)slot * 8), 1ULL,
                            kCuWaitValueGeq),
                "cuStreamWaitValue64(done)");
}

struct DotChoice {
  dot_fn fn;
  const char* name;
};

// SIMD tiers, ascending. Each format picks the highest tier <= the one chosen by
// pick_isa() that it implements (fp4 formats have no bf16-specific tier, so the
// avx512bf16 tier maps to their avx512 kernel).
enum IsaTier { ISA_SCALAR = 0, ISA_AVX2 = 1, ISA_AVX512 = 2, ISA_AVX512BF16 = 3 };

// Best tier the CPU+build supports, optionally capped DOWN by
// FREETOKEN_CPU_MOE_ISA={scalar,avx2,avx512,avx512bf16} (A/B testing on a machine
// that supports more). FREETOKEN_CPU_MOE_SCALAR=1 forces scalar (legacy alias).
inline IsaTier pick_isa() {
#if CPU_MOE_X86
  if (getenv("FREETOKEN_CPU_MOE_SCALAR")) return ISA_SCALAR;
  IsaTier best = ISA_SCALAR;
  if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) best = ISA_AVX2;
  if (best >= ISA_AVX2 && __builtin_cpu_supports("avx512f")) best = ISA_AVX512;
#ifdef CPU_MOE_HAS_AVX512BF16
  if (best >= ISA_AVX512 && __builtin_cpu_supports("avx512bf16")) best = ISA_AVX512BF16;
#endif
  if (const char* f = getenv("FREETOKEN_CPU_MOE_ISA")) {
    IsaTier want = best;
    if (!std::strcmp(f, "scalar")) want = ISA_SCALAR;
    else if (!std::strcmp(f, "avx2")) want = ISA_AVX2;
    else if (!std::strcmp(f, "avx512")) want = ISA_AVX512;
    else if (!std::strcmp(f, "avx512bf16")) want = ISA_AVX512BF16;
    if (want < best) best = want;  // cap downward; never force above hw/build support
  }
  return best;
#else
  return ISA_SCALAR;
#endif
}

DotChoice select_dot() {
  const IsaTier t = pick_isa();
#if CPU_MOE_X86
#ifdef CPU_MOE_HAS_AVX512BF16
  if (t >= ISA_AVX512BF16) return {dot_avx512bf16, "avx512bf16"};
#endif
  if (t >= ISA_AVX512) return {dot_avx512f, "avx512f"};
  if (t >= ISA_AVX2) return {dot_avx2, "avx2"};
#endif
  (void)t;
  return {dot_scalar, "scalar"};
}

nvdot_fn select_nvdot() {
  const IsaTier t = pick_isa();
#if CPU_MOE_X86
  if (t >= ISA_AVX512) return dot_nvfp4_avx512;
  if (t >= ISA_AVX2) return dot_nvfp4_avx2;
#endif
  (void)t;
  return dot_nvfp4_scalar;
}

// AVX-VNNI (VEX-256 VPDPBUSD) availability: Alder/Raptor Lake, Sapphire Rapids+, Zen5.
// Distinct from AVX-512 VNNI. Opt out with FREETOKEN_CPU_MOE_NO_VNNI=1 (A/B the W4A8 path).
inline bool cpu_has_avxvnni() {
#if CPU_MOE_X86
  const char* no = getenv("FREETOKEN_CPU_MOE_NO_VNNI");
  if (no && no[0] && no[0] != '0') return false;  // ignore unset/empty/"0"
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("avxvnni");
#else
  return false;
#endif
}

// AVX-512 VNNI (512-bit VPDPBUSD): Cascade Lake+, Ice Lake, Sapphire/Emerald Rapids,
// Zen4+. 2x the 256-bit AVX-VNNI width. FREETOKEN_CPU_MOE_NO_AVX512VNNI=1 forces the
// 256-bit path (A/B the two W4A8 kernels on the same box); FREETOKEN_CPU_MOE_NO_VNNI=1
// still disables the whole W4A8 family (back to the faithful fp32 nvdot), so it is
// honored here too.
inline bool cpu_has_avx512vnni() {
#if CPU_MOE_X86 && defined(CPU_MOE_HAS_AVX512VNNI)
  const char* no = getenv("FREETOKEN_CPU_MOE_NO_AVX512VNNI");
  if (no && no[0] && no[0] != '0') return false;
  const char* no_vnni = getenv("FREETOKEN_CPU_MOE_NO_VNNI");
  if (no_vnni && no_vnni[0] && no_vnni[0] != '0') return false;
  return __builtin_cpu_supports("avx512vnni");
#else
  return false;
#endif
}

// Best W4A8 (int8-activation) nvfp4 dot, or nullptr if no SIMD VNNI (caller keeps the
// faithful fp32 nvdot path). The scalar i8 dot exists only as a correctness reference.
nvi8dot_fn select_nvi8dot() {
#if CPU_MOE_X86
#if defined(CPU_MOE_HAS_AVX512VNNI)
  if (cpu_has_avx512vnni()) return dot_nvfp4_i8_avx512vnni;
#endif
  if (cpu_has_avxvnni()) return dot_nvfp4_i8_vnni;
#endif
  return nullptr;
}

// ----------------------- DeepSeek-V4 ds_fp4 (W4A8) ---------------------------
// Row-major e2m1 (2/byte, low nibble first) + e8m0 per-32 block scale, no global
// (w = E2M1[code] * 2^(e8m0-127)); activations are FP8-e4m3 round-tripped (per-128
// block, ue8m0 scale) before each GEMM. Matches kernel/triton/dsv4 (fused_moe +
// fp8_linear): silu(clamp(gate,max=lim)) * clamp(up,-lim,lim), router weight on the
// down output.

// Activations are pre-deinterleaved to fp32 (xe[m]=x[2m], xo[m]=x[2m+1]) once per
// token/route and reused across all output rows. This drops the hot dot to one
// vpmovzxbd + two vpermps per 32 weights (1.5 shuffle ops / 16 vs 4 for a bf16,
// dup-permute, per-element gather), so the row-major fp4 GEMV stops being port-5
// bound and approaches the bf16 memory-bandwidth ceiling.
using dsdot_fn = float (*)(const uint8_t*, const uint8_t*, const float*, const float*, int,
                           const float*, const float*);

float dot_dsfp4_scalar(const uint8_t* packed, const uint8_t* scale, const float* xe,
                       const float* xo, int K, const float* e2m1, const float* e8m0) {
  float acc = 0.0f;
  const int nb = K / 32;
  for (int b = 0; b < nb; ++b) {
    const float sc = e8m0[scale[b]];
    const uint8_t* pk = packed + (size_t)b * 16;  // 16 bytes = 32 codes
    const float* xeb = xe + (size_t)b * 16;
    const float* xob = xo + (size_t)b * 16;
    float bsum = 0.0f;
    for (int j = 0; j < 16; ++j) {
      const uint8_t byte = pk[j];
      bsum += e2m1[byte & 0xF] * xeb[j];   // low nibble  -> even-K activation
      bsum += e2m1[byte >> 4] * xob[j];    // high nibble -> odd-K activation
    }
    acc += sc * bsum;
  }
  return acc;
}

#if CPU_MOE_X86
// One 32-block: 16 bytes -> 16 low + 16 high nibble values via two vpermps, times
// the pre-split even/odd fp32 activations, folded by the per-32 e8m0 scale.
__attribute__((target("avx512f")))
inline __m512 dsfp4_blk(const uint8_t* pk, const float* xeb, const float* xob, __m512 lut,
                        __m512i loma, float sc) {
  __m512i wi = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pk)));
  __m512 vlo = _mm512_permutexvar_ps(_mm512_and_si512(wi, loma), lut);
  __m512 vhi = _mm512_permutexvar_ps(_mm512_and_si512(_mm512_srli_epi32(wi, 4), loma), lut);
  __m512 prod = _mm512_fmadd_ps(vlo, _mm512_loadu_ps(xeb), _mm512_mul_ps(vhi, _mm512_loadu_ps(xob)));
  return _mm512_mul_ps(prod, _mm512_set1_ps(sc));
}

__attribute__((target("avx512f")))
float dot_dsfp4_avx512(const uint8_t* packed, const uint8_t* scale, const float* xe,
                       const float* xo, int K, const float* e2m1, const float* e8m0) {
  const __m512 lut = _mm512_loadu_ps(e2m1);
  const __m512i loma = _mm512_set1_epi32(0xF);
  __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
  const int nb = K / 32;  // 16 packed bytes + one e8m0 scale per block
  int b = 0;
  for (; b + 2 <= nb; b += 2) {  // two independent accumulators hide FMA latency
    acc0 = _mm512_add_ps(acc0, dsfp4_blk(packed + (size_t)b * 16, xe + (size_t)b * 16,
                                         xo + (size_t)b * 16, lut, loma, e8m0[scale[b]]));
    acc1 = _mm512_add_ps(acc1, dsfp4_blk(packed + (size_t)(b + 1) * 16, xe + (size_t)(b + 1) * 16,
                                         xo + (size_t)(b + 1) * 16, lut, loma, e8m0[scale[b + 1]]));
  }
  for (; b < nb; ++b)
    acc0 = _mm512_add_ps(acc0, dsfp4_blk(packed + (size_t)b * 16, xe + (size_t)b * 16,
                                         xo + (size_t)b * 16, lut, loma, e8m0[scale[b]]));
  return _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
}

// AVX2: a 32-K block is 16 bytes -> two 8-lane halves (8 even + 8 odd each).
__attribute__((target("avx2,fma")))
inline __m256 dsfp4_half_avx2(const uint8_t* pk, const float* xeb, const float* xob, __m256 mag8) {
  __m256i wi = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(pk)));
  __m256 vlo = e2m1_decode8(_mm256_and_si256(wi, _mm256_set1_epi32(0xF)), mag8);
  __m256 vhi = e2m1_decode8(_mm256_srli_epi32(wi, 4), mag8);
  return _mm256_fmadd_ps(vlo, _mm256_loadu_ps(xeb), _mm256_mul_ps(vhi, _mm256_loadu_ps(xob)));
}

__attribute__((target("avx2,fma")))
float dot_dsfp4_avx2(const uint8_t* packed, const uint8_t* scale, const float* xe,
                     const float* xo, int K, const float* e2m1, const float* e8m0) {
  const __m256 mag8 = _mm256_loadu_ps(e2m1);
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
  const int nb = K / 32;
  for (int b = 0; b < nb; ++b) {
    const uint8_t* pk = packed + (size_t)b * 16;
    const float* xeb = xe + (size_t)b * 16;
    const float* xob = xo + (size_t)b * 16;
    const __m256 sc = _mm256_set1_ps(e8m0[scale[b]]);
    acc0 = _mm256_fmadd_ps(dsfp4_half_avx2(pk, xeb, xob, mag8), sc, acc0);
    acc1 = _mm256_fmadd_ps(dsfp4_half_avx2(pk + 8, xeb + 8, xob + 8, mag8), sc, acc1);
  }
  return hsum256(_mm256_add_ps(acc0, acc1));
}
#endif

dsdot_fn select_dsdot() {
  const IsaTier t = pick_isa();
#if CPU_MOE_X86
  if (t >= ISA_AVX512) return dot_dsfp4_avx512;
  if (t >= ISA_AVX2) return dot_dsfp4_avx2;
#endif
  (void)t;
  return dot_dsfp4_scalar;
}

// ------------------------- mxfp4 (gpt-oss) GEMV -----------------------------
// Transposed split-K layout: blk[Kpairs, N2] (N innermost), scl[Kpairs/16, N2]
// e8m0 per 32-K. Computes out[c] = sum_kb (E2M1[lo]*x[2kb] + E2M1[hi]*x[2kb+1])
// * 2^(e8m0-127) for a contiguous column tile (blk/scl already offset to col 0 of
// the tile). Vectorized over N (16 columns / __m512), K stays the outer (cache-
// sequential) loop. Used by both gate_up (K=H) and down (K=I).
using mxgemv_fn = void (*)(float*, const uint8_t*, const uint8_t*, const bf16_t*, int, int,
                           int, const float*, const float*);

void mxfp4_gemv_scalar(float* out, const uint8_t* blk, const uint8_t* scl, const bf16_t* x,
                       int Kpairs, int N2, int ncol, const float* e2m1, const float* e8m0) {
  for (int c = 0; c < ncol; ++c) out[c] = 0.0f;
  for (int kb = 0; kb < Kpairs; ++kb) {
    const uint8_t* w = blk + (size_t)kb * N2;
    const uint8_t* s = scl + (size_t)(kb >> 4) * N2;
    const float xl = bf16_to_f32(x[2 * kb]);
    const float xh = bf16_to_f32(x[2 * kb + 1]);
    for (int c = 0; c < ncol; ++c) {
      const uint8_t byte = w[c];
      out[c] += (e2m1[byte & 0xF] * xl + e2m1[byte >> 4] * xh) * e8m0[s[c]];
    }
  }
}

#if CPU_MOE_X86
__attribute__((target("avx512f")))
void mxfp4_gemv_avx512(float* out, const uint8_t* blk, const uint8_t* scl, const bf16_t* x,
                       int Kpairs, int N2, int ncol, const float* e2m1, const float* e8m0) {
  (void)e8m0;  // e8m0[c]=2^(c-127) computed via bit construction (no gather)
  const __m512 lut = _mm512_loadu_ps(e2m1);
  const __m512i loma = _mm512_set1_epi32(0xF);
  // K-outer / N-inner: each kb cache line is read once and all live column chunks
  // (up to 4 -> 64 cols) accumulate from registers, so DRAM/L2 stream the tile once.
  int c0 = 0;
  for (; c0 + 16 <= ncol; c0 += 64) {
    const int nchunk = std::min(4, (ncol - c0) / 16);
    __m512 acc[4];
    for (int ci = 0; ci < nchunk; ++ci) acc[ci] = _mm512_setzero_ps();
    for (int kblk = 0; kblk < Kpairs; kblk += 16) {  // 16 K-pairs = 32 K = one scale row
      __m512 sc[4];
      for (int ci = 0; ci < nchunk; ++ci) {
        __m128i sraw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(
            scl + (size_t)(kblk >> 4) * N2 + c0 + ci * 16));
        sc[ci] = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu8_epi32(sraw), 23));
      }
      __m512 blk_acc[4];
      for (int ci = 0; ci < nchunk; ++ci) blk_acc[ci] = _mm512_setzero_ps();
      for (int kk = 0; kk < 16; ++kk) {
        const int kb = kblk + kk;
        const uint8_t* wbase = blk + (size_t)kb * N2 + c0;
        // The transposed layout strides K by N2 bytes; prefetch ahead so the strided
        // reads are not exposed to DRAM latency (the HW streamer misses big strides).
        constexpr int PFD = 8;
        if (kb + PFD < Kpairs)
          _mm_prefetch(reinterpret_cast<const char*>(blk + (size_t)(kb + PFD) * N2 + c0),
                       _MM_HINT_T0);
        const __m512 xl = _mm512_set1_ps(bf16_to_f32(x[2 * kb]));
        const __m512 xh = _mm512_set1_ps(bf16_to_f32(x[2 * kb + 1]));
        for (int ci = 0; ci < nchunk; ++ci) {
          __m512i wi = _mm512_cvtepu8_epi32(
              _mm_loadu_si128(reinterpret_cast<const __m128i*>(wbase + ci * 16)));
          __m512 vlo = _mm512_permutexvar_ps(_mm512_and_si512(wi, loma), lut);
          __m512 vhi = _mm512_permutexvar_ps(_mm512_and_si512(_mm512_srli_epi32(wi, 4), loma), lut);
          blk_acc[ci] = _mm512_fmadd_ps(vlo, xl, blk_acc[ci]);
          blk_acc[ci] = _mm512_fmadd_ps(vhi, xh, blk_acc[ci]);
        }
      }
      for (int ci = 0; ci < nchunk; ++ci) acc[ci] = _mm512_fmadd_ps(blk_acc[ci], sc[ci], acc[ci]);
    }
    for (int ci = 0; ci < nchunk; ++ci) _mm512_storeu_ps(out + c0 + ci * 16, acc[ci]);
  }
  for (int c = c0; c < ncol; ++c) {  // tail columns (< 16)
    float o = 0.0f;
    for (int kb = 0; kb < Kpairs; ++kb) {
      const uint8_t byte = blk[(size_t)kb * N2 + c];
      uint32_t bits = (uint32_t)scl[(size_t)(kb >> 4) * N2 + c] << 23;
      float sc;
      std::memcpy(&sc, &bits, 4);
      o += (e2m1[byte & 0xF] * bf16_to_f32(x[2 * kb]) +
            e2m1[byte >> 4] * bf16_to_f32(x[2 * kb + 1])) * sc;
    }
    out[c] = o;
  }
}

__attribute__((target("avx2,fma")))
void mxfp4_gemv_avx2(float* out, const uint8_t* blk, const uint8_t* scl, const bf16_t* x,
                     int Kpairs, int N2, int ncol, const float* e2m1, const float* e8m0) {
  (void)e8m0;  // e8m0[s]=2^(s-127) built via s<<23 (no gather)
  const __m256 mag8 = _mm256_loadu_ps(e2m1);
  int c0 = 0;
  for (; c0 + 8 <= ncol; c0 += 32) {  // up to 4 chunks of 8 = 32 cols
    const int nchunk = std::min(4, (ncol - c0) / 8);
    __m256 acc[4];
    for (int ci = 0; ci < nchunk; ++ci) acc[ci] = _mm256_setzero_ps();
    for (int kblk = 0; kblk < Kpairs; kblk += 16) {  // 16 K-pairs = one scale row
      __m256 sc[4];
      for (int ci = 0; ci < nchunk; ++ci) {
        __m128i sraw = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(
            scl + (size_t)(kblk >> 4) * N2 + c0 + ci * 8));
        sc[ci] = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu8_epi32(sraw), 23));
      }
      __m256 blk_acc[4];
      for (int ci = 0; ci < nchunk; ++ci) blk_acc[ci] = _mm256_setzero_ps();
      for (int kk = 0; kk < 16; ++kk) {
        const int kb = kblk + kk;
        const uint8_t* wbase = blk + (size_t)kb * N2 + c0;
        constexpr int PFD = 8;
        if (kb + PFD < Kpairs)
          _mm_prefetch(reinterpret_cast<const char*>(blk + (size_t)(kb + PFD) * N2 + c0),
                       _MM_HINT_T0);
        const __m256 xl = _mm256_set1_ps(bf16_to_f32(x[2 * kb]));
        const __m256 xh = _mm256_set1_ps(bf16_to_f32(x[2 * kb + 1]));
        for (int ci = 0; ci < nchunk; ++ci) {
          __m256i wi = _mm256_cvtepu8_epi32(
              _mm_loadl_epi64(reinterpret_cast<const __m128i*>(wbase + ci * 8)));
          __m256 vlo = e2m1_decode8(_mm256_and_si256(wi, _mm256_set1_epi32(0xF)), mag8);
          __m256 vhi = e2m1_decode8(_mm256_srli_epi32(wi, 4), mag8);
          blk_acc[ci] = _mm256_fmadd_ps(vlo, xl, blk_acc[ci]);
          blk_acc[ci] = _mm256_fmadd_ps(vhi, xh, blk_acc[ci]);
        }
      }
      for (int ci = 0; ci < nchunk; ++ci) acc[ci] = _mm256_fmadd_ps(blk_acc[ci], sc[ci], acc[ci]);
    }
    for (int ci = 0; ci < nchunk; ++ci) _mm256_storeu_ps(out + c0 + ci * 8, acc[ci]);
  }
  for (int c = c0; c < ncol; ++c) {  // tail columns (< 8); none when ncol%8==0
    float o = 0.0f;
    for (int kb = 0; kb < Kpairs; ++kb) {
      const uint8_t byte = blk[(size_t)kb * N2 + c];
      uint32_t bits = (uint32_t)scl[(size_t)(kb >> 4) * N2 + c] << 23;
      float sc;
      std::memcpy(&sc, &bits, 4);
      o += (e2m1[byte & 0xF] * bf16_to_f32(x[2 * kb]) +
            e2m1[byte >> 4] * bf16_to_f32(x[2 * kb + 1])) * sc;
    }
    out[c] = o;
  }
}
#endif

mxgemv_fn select_mxgemv() {
  const IsaTier t = pick_isa();
#if CPU_MOE_X86
  if (t >= ISA_AVX512) return mxfp4_gemv_avx512;
  if (t >= ISA_AVX2) return mxfp4_gemv_avx2;
#endif
  (void)t;
  return mxfp4_gemv_scalar;
}

// Round a clamped |x|<=448 to nearest float8-e4m3 (RNE), back to fp32. Matches
// torch.float8_e4m3fn / triton .to(float8e4nv).
inline float e4m3_round(float x) {
  const float sign = x < 0.0f ? -1.0f : 1.0f;
  const float a = std::fabs(x);
  if (a == 0.0f) return 0.0f;
  if (a >= 448.0f) return sign * 448.0f;
  int e;
  std::frexp(a, &e);  // a in [2^(e-1), 2^e)
  float step = std::ldexp(1.0f, e - 4);
  const float min_step = std::ldexp(1.0f, -9);  // e4m3 subnormal step (2^-9)
  if (step < min_step) step = min_step;
  float r = std::nearbyint(a / step) * step;
  if (r > 448.0f) r = 448.0f;
  return sign * r;
}

// IEEE ceil(log2(v)) for v>0 (matches dsv4 _log2_ceil / fast_round_scale).
inline int ceil_log2_pos(float v) {
  uint32_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  const int exp = (int)((bits >> 23) & 0xFF);
  const int man = (int)(bits & 0x7FFFFF);
  return exp - 127 + (man != 0 ? 1 : 0);
}

// Split an interleaved bf16 row into fp32 even/odd halves (even[m]=src[2m]).
// bf16->fp32 is exact, so this only reorders -- done once per token/route and
// reused across every output row of the GEMV.
inline void deinterleave_bf16_f32(const bf16_t* src, float* even, float* odd, int K) {
  for (int m = 0; m < K / 2; ++m) {
    even[m] = bf16_to_f32(src[2 * m]);
    odd[m] = bf16_to_f32(src[2 * m + 1]);
  }
}

// DeepSeek-V4 activation FP8 round-trip (bf16 in/out): per 128-block,
// s = 2^ceil(log2(max(|x|,1e-4)/448)); y = round_e4m3(clamp(x/s,+-448)) * s.
void fp8_roundtrip_bf16(const bf16_t* src, bf16_t* dst, int K) {
  for (int b0 = 0; b0 < K; b0 += 128) {
    const int b1 = std::min(K, b0 + 128);
    float amax = 1e-4f;
    for (int i = b0; i < b1; ++i) amax = std::max(amax, std::fabs(bf16_to_f32(src[i])));
    const float s = std::ldexp(1.0f, ceil_log2_pos(amax * (1.0f / 448.0f)));
    const float inv_s = 1.0f / s;
    for (int i = b0; i < b1; ++i) {
      float q = bf16_to_f32(src[i]) * inv_s;
      q = std::min(448.0f, std::max(-448.0f, q));
      dst[i] = f32_to_bf16(e4m3_round(q) * s);
    }
  }
}

// --------------------------------- executor ---------------------------------

struct CpuMoeExecutor;

struct MoeTask {
  CpuMoeExecutor* exec;
  int layer_id;
  int num_tokens;
  const bf16_t* x;     // [num_tokens, H]
  const int32_t* ids;  // [num_tokens, top_k]  (raw expert ids; <0 = skip)
  const float* w;      // [num_tokens, top_k]
  bf16_t* y;           // [num_tokens, H]
};

// Output-row tiling. Small enough to give every worker independent work even at
// batch size 1; large enough to amortize the atomic work-grab.
//
// Bandwidth notes (Sapphire Rapids 8480+, 13 cores): the two passes already read
// every expert weight byte exactly once per token (each output row block is owned
// by one worker), and x stays hot in L1 across a (token,expert)'s rows -- so the
// kernel is single-read bandwidth-optimal at bs=1 (~205 GB/s vs ~55 GB/s PCIe).
// One worker per *physical* core, pinned, is the sweet spot; SMT oversubscription
// thrashes the spin-barrier. Deferred (not worth it here / for this workload):
//   - AMX-bf16: a GEMM tile engine; decode is M=1 GEMV so tiles sit idle. It would
//     only pay off in a grouped/batched (dedup) path.
//   - expert dedup for bs>1: read each distinct expert once and GEMM its tokens.
//     Helps locality+bytes when bs is large; decode batches here are tiny (<=4).
//   - NUMA: a single node is assumed. Multi-socket machines would split each
//     expert's K dimension per node (banks are already per-row contiguous).
constexpr int IBLK = 32;
constexpr int HBLK = 32;

// -------------------------------- Q4_0 (W4A8) --------------------------------
// Native GGUF Q4_0 experts (gemma4 GGUF): per-32 block = fp16 scale d + 16 packed
// bytes; byte j holds element j in its low nibble and j+16 in its high nibble, so a
// block's storage order is [lo0..lo15, hi0..hi15] and w = (nibble - 8) * d. Matches
// the reference dequant (models/gguf/dequant.py) and the packed banks the GPU offload
// path streams.
//
// llama.cpp ggml_vec_dot_q4_0_q8_0: W4A8. The activation is pre-quantized to Q8_0
// (per-32-block int8 ``aq`` + fp32 scale ``asb``); each block unpacks its 16 bytes to
// 32 int8 weights in [-8,7] (bytes_from_nibbles_32: low nibbles -> elems 0..15, high
// -> 16..31) and runs an integer block dot -- VPDPBUSD (AVX-VNNI) or VPMADDUBSW+VPMADDWD
// (AVX2) with the ggml sign trick |w|*(sign(w)*a)=w*a, or a scalar int loop -- then
// scales the block sum by wd*xd in fp32. No fp weight dequant / shuffle chain. The GPU
// offload path (ggml_moe_a8_vec / MMVQ) is also W4A8, so cpu and hybrid stay close.
using q4dot_fn = float (*)(const uint8_t*, const int8_t*, const float*, int);

float q4_0_dot_i8_scalar(const uint8_t* w, const int8_t* aq, const float* asb, int K) {
  float acc = 0.0f;
  const int nb = K / 32;
  for (int b = 0; b < nb; ++b) {
    const uint8_t* blk = w + (size_t)b * 18;
    uint16_t dh;
    std::memcpy(&dh, blk, sizeof(dh));
    const uint8_t* q = blk + 2;  // 16 nibble bytes
    const int8_t* a = aq + (size_t)b * 32;
    int isum = 0;
    for (int j = 0; j < 16; ++j) {
      isum += ((int)(q[j] & 0x0F) - 8) * (int)a[j];       // elem j
      isum += ((int)(q[j] >> 4) - 8) * (int)a[16 + j];    // elem 16+j
    }
    acc += fp16_to_f32(dh) * asb[b] * (float)isum;
  }
  return acc;
}

#if CPU_MOE_X86
// fp16 block scale -> fp32 via HW F16C (single value in lane 0).
__attribute__((target("f16c")))
static inline float q4_scale(uint16_t h) {
  return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)h)));
}

// Unpack one Q4_0 block's 16 bytes -> 32 int8 weights in [-8,7] (elems 0..15 = low
// nibbles, 16..31 = high nibbles). ``eight`` = _mm256_set1_epi8(8).
__attribute__((target("avx2")))
static inline __m256i q4_unpack32(const uint8_t* blk, __m128i mask, __m256i eight) {
  const __m128i qb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk + 2));
  const __m128i lo = _mm_and_si128(qb, mask);
  const __m128i hi = _mm_and_si128(_mm_srli_epi16(qb, 4), mask);
  return _mm256_sub_epi8(_mm256_set_m128i(hi, lo), eight);
}

// AVX2 W4A8 (llama.cpp non-VNNI mul_sum_i8_pairs): integer block dot via VPMADDUBSW +
// VPMADDWD (sign trick), scaled by wd*xd. |aw*sa| pair sums <= 8*127*2 < 32767 -> no
// int16 saturation. This is the fast path on AVX2 CPUs without AVX-VNNI (and the
// avx512-tier fallback, since the block dot is 256-bit either way).
__attribute__((target("avx2,fma,f16c")))
float q4_0_dot_i8_avx2(const uint8_t* w, const int8_t* aq, const float* asb, int K) {
  const __m128i mask = _mm_set1_epi8(0x0F);
  const __m256i eight = _mm256_set1_epi8(8);
  const __m256i ones16 = _mm256_set1_epi16(1);
  __m256 accF = _mm256_setzero_ps();
  const int nb = K / 32;
  for (int b = 0; b < nb; ++b) {
    const uint8_t* blk = w + (size_t)b * 18;
    _mm_prefetch(reinterpret_cast<const char*>(blk) + 512, _MM_HINT_T0);
    uint16_t dh;
    std::memcpy(&dh, blk, sizeof(dh));
    __m256i wq = q4_unpack32(blk, mask, eight);
    __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(aq + (size_t)b * 32));
    __m256i aw = _mm256_sign_epi8(wq, wq);              // |wq|          (unsigned operand)
    __m256i sa = _mm256_sign_epi8(a, wq);               // sign(wq) * a  (signed operand)
    __m256i d32 = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, sa), ones16);  // 8 int32
    accF = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d32), _mm256_set1_ps(q4_scale(dh) * asb[b]), accF);
  }
  return hsum256(accF);
}

// AVX-VNNI W4A8: one VPDPBUSD per block (the fast path on modern CPUs).
__attribute__((target("avx2,avxvnni,fma,f16c")))
float q4_0_dot_i8_vnni(const uint8_t* w, const int8_t* aq, const float* asb, int K) {
  const __m128i mask = _mm_set1_epi8(0x0F);
  const __m256i eight = _mm256_set1_epi8(8);
  __m256 accF = _mm256_setzero_ps();
  const int nb = K / 32;
  for (int b = 0; b < nb; ++b) {
    const uint8_t* blk = w + (size_t)b * 18;
    _mm_prefetch(reinterpret_cast<const char*>(blk) + 512, _MM_HINT_T0);
    uint16_t dh;
    std::memcpy(&dh, blk, sizeof(dh));
    __m256i wq = q4_unpack32(blk, mask, eight);
    __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(aq + (size_t)b * 32));
    __m256i aw = _mm256_sign_epi8(wq, wq);   // |wq|            (unsigned operand)
    __m256i sa = _mm256_sign_epi8(a, wq);    // sign(wq) * a    (signed operand)
    __m256i di = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), aw, sa);
    // All 32 elems of the block share wd*xd; distribute over di's 8 partial sums and
    // reduce at the end (equivalent to scale * block_total).
    accF = _mm256_fmadd_ps(_mm256_cvtepi32_ps(di), _mm256_set1_ps(q4_scale(dh) * asb[b]), accF);
  }
  return hsum256(accF);
}
#endif  // CPU_MOE_X86

// All tiers are W4A8 (int8 activations pre-quantized to Q8_0). AVX-VNNI is orthogonal to
// the ISA tier (gated by cpu_has_avxvnni() / FREETOKEN_CPU_MOE_NO_VNNI), so it wins when
// present; otherwise the 256-bit VPMADDUBSW kernel covers both the avx2 and avx512 tiers.
q4dot_fn select_q4dot() {
  const IsaTier t = pick_isa();
#if CPU_MOE_X86
  if (cpu_has_avxvnni()) return q4_0_dot_i8_vnni;
  if (t >= ISA_AVX2) return q4_0_dot_i8_avx2;
#endif
  (void)t;
  return q4_0_dot_i8_scalar;
}

enum WFmt {
  WF_BF16 = 0,
  WF_NVFP4 = 1,
  WF_MXFP4 = 2,
  WF_DSFP4 = 3,
  WF_Q4_0 = 4,
  WF_IQ3_XXS = 5,  // gguf type 18
  WF_IQ4_XS = 6,   // gguf type 23
  WF_Q6_K = 7,     // gguf type 14
};

// ---------------- GGUF K-quant family (IQ3_XXS / IQ4_XS / Q6_K) --------------
// Native gguf expert banks (glm5next): 256-wide blocks, row-major over K, read in
// place from the pinned banks the offload path streams. Unlike q4_0/nvfp4 these
// codes are grid/LUT indices, not nibble magnitudes, so there is no W4A8 form: the
// GEMV is W4A16, dequantizing each row's blocks on the fly into the fp32 dot over
// bf16 activations -- no activation prequant, no prepare phase, no dequant scratch
// (a bf16 dequant-then-reuse-dot scratch would transiently cost top_k*rows*H*2 B
// per layer, ~268 MB at the glm5next geometry, to save one LUT gather per weight).
// Value formulas match the vendored CUDA kernels (kernel/csrc/gguf/dequantize.cuh
// dequantize_block_{q6_K,iq3_xxs,iq4_xs}, the block-layout source of truth); the
// CPU chain stays in fp32 (the half d widens exactly), inside the dfloat parity
// contract (tests/kernels/test_gguf_quant.py).
//
// The scalar kernels are the FALLBACK tier. The hot path runs the verbatim ggml
// AVX2 integer kernels in the W4A8-K section below: activations q8_K-quantized
// once per row and reused across the expert dots. The no-prequant tradeoff above
// was about a bf16 DEQUANT scratch (one row per expert); a q8_K row is 292 B per
// 256 elems (~23 KB/token at the glm5next geometry), the same one-shot prepare
// the nvfp4/q4_0 W4A8 paths already pay.
//
// Verbatim host copies of the vendored device LUTs (ggml-common.h: iq3xxs_grid
// :563, ksigns_iq2xs :888, kmask_iq2xs + kvalues_iq4nl :927). Values are owned by
// the vendored files -- do not edit here or there.
static const uint32_t iq3xxs_grid[256] = {
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414, 0x04041c0c,
    0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14, 0x040c140c, 0x040c142c,
    0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404, 0x04140414, 0x04140424, 0x04140c0c,
    0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e, 0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c,
    0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c, 0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e,
    0x04243e1c, 0x04243e2c, 0x042c040c, 0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04,
    0x043e0c24, 0x043e0c34, 0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c,
    0x0c04141c, 0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04, 0x0c143e14,
    0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c, 0x0c24042c, 0x0c242c04,
    0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414, 0x0c3e2404, 0x14040404, 0x14040414,
    0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434, 0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c,
    0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c, 0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404,
    0x14140414, 0x14140c0c, 0x14140c3e, 0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c,
    0x141c0c04, 0x141c0c24, 0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c,
    0x142c3e24, 0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c, 0x1c0c2424,
    0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14, 0x1c1c0c0c, 0x1c1c1c1c,
    0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414, 0x1c2c2c2c, 0x1c340c24, 0x1c341c34,
    0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e, 0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e,
    0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404, 0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424,
    0x24242c0c, 0x24243424, 0x242c142c, 0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04,
    0x2c040c14, 0x2c04240c, 0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14,
    0x2c143e14, 0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c, 0x340c340c,
    0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14, 0x34341c1c, 0x343e041c,
    0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14, 0x3e042c14, 0x3e0c1434, 0x3e0c2404,
    0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c, 0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c,
    0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
};

static const uint8_t ksigns_iq2xs[128] = {
    0,   129, 130, 3,   132, 5,   6,   135, 136, 9,   10,  139, 12,  141, 142, 15,  144, 17,  18,  147, 20,  149,
    150, 23,  24,  153, 154, 27,  156, 29,  30,  159, 160, 33,  34,  163, 36,  165, 166, 39,  40,  169, 170, 43,
    172, 45,  46,  175, 48,  177, 178, 51,  180, 53,  54,  183, 184, 57,  58,  187, 60,  189, 190, 63,  192, 65,
    66,  195, 68,  197, 198, 71,  72,  201, 202, 75,  204, 77,  78,  207, 80,  209, 210, 83,  212, 85,  86,  215,
    216, 89,  90,  219, 92,  221, 222, 95,  96,  225, 226, 99,  228, 101, 102, 231, 232, 105, 106, 235, 108, 237,
    238, 111, 240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

static const uint8_t kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};
static const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

inline uint16_t load_u16le(const uint8_t* p) {
  uint16_t v;
  std::memcpy(&v, p, sizeof(v));
  return v;
}

// One packed row (K/256 blocks) dotted against bf16 activations, fp32 accumulate.
// Q6_K (210 B/block, d LAST at 208:210): ql lower 4 bits + qh upper 2 bits give a
// 6-bit code in [-32, 31]; per-32 sub-scales are int8; y = d * sc * (q - 32).
float q6_k_dot_scalar(const uint8_t* row, const bf16_t* x, int K) {
  float acc = 0.0f;
  for (int b = 0; b < K / 256; ++b) {
    const uint8_t* blk = row + (size_t)b * 210;
    const float d = fp16_to_f32(load_u16le(blk + 208));
    const uint8_t* ql = blk;                                    // 128
    const uint8_t* qhbuf = blk + 128;                           // 64
    const int8_t* sc = reinterpret_cast<const int8_t*>(blk + 192);  // 16
    for (int ip = 0; ip < 2; ++ip) {
      for (int il = 0; il < 32; ++il) {
        const int is = 8 * ip + il / 16;
        const uint8_t ql0 = ql[64 * ip + il];
        const uint8_t ql32 = ql[64 * ip + il + 32];
        const uint8_t qhb = qhbuf[32 * ip + il];
        const bf16_t* xb = x + (size_t)b * 256 + 128 * ip + il;
        acc += d * (float)sc[is + 0] *
               (float)((int)((ql0 & 0xF) | (((qhb >> 0) & 3) << 4)) - 32) * bf16_to_f32(xb[0]);
        acc += d * (float)sc[is + 2] *
               (float)((int)((ql32 & 0xF) | (((qhb >> 2) & 3) << 4)) - 32) * bf16_to_f32(xb[32]);
        acc += d * (float)sc[is + 4] *
               (float)((int)((ql0 >> 4) | (((qhb >> 4) & 3) << 4)) - 32) * bf16_to_f32(xb[64]);
        acc += d * (float)sc[is + 6] *
               (float)((int)((ql32 >> 4) | (((qhb >> 6) & 3) << 4)) - 32) * bf16_to_f32(xb[96]);
      }
    }
  }
  return acc;
}

// IQ4_XS (136 B/block, d FIRST): 4-bit codes into kvalues_iq4nl; the per-32 scale
// is a 6-bit field spread over scales_l nibbles + 2 scales_h bits, minus 32.
float iq4_xs_dot_scalar(const uint8_t* row, const bf16_t* x, int K) {
  float acc = 0.0f;
  for (int b = 0; b < K / 256; ++b) {
    const uint8_t* blk = row + (size_t)b * 136;
    const float d = fp16_to_f32(load_u16le(blk));
    const uint16_t scales_h = load_u16le(blk + 2);
    const uint8_t* scales_l = blk + 4;  // 4
    const uint8_t* qs = blk + 8;        // 128
    for (int ib = 0; ib < 8; ++ib) {
      const int s = ((((scales_l[ib / 2] >> 4 * (ib % 2)) & 0xF) | (((scales_h >> 2 * ib) & 3) << 4)) - 32);
      const float db = d * (float)s;
      const uint8_t* q4 = qs + 16 * ib;
      const bf16_t* xb = x + (size_t)b * 256 + 32 * ib;
      for (int j = 0; j < 16; ++j) {
        acc += db * (float)kvalues_iq4nl[q4[j] & 0xF] * bf16_to_f32(xb[j]);
        acc += db * (float)kvalues_iq4nl[q4[j] >> 4] * bf16_to_f32(xb[j + 16]);
      }
    }
  }
  return acc;
}

// IQ3_XXS (98 B/block, d FIRST): 64 grid-index bytes then 8 uint32 scale words
// (bits 28-31: sub-scale nibble, bits 0-27: four 7-bit sign indices); values come
// from the iq3xxs_grid bytes, signed by ksigns_iq2xs.
float iq3_xxs_dot_scalar(const uint8_t* row, const bf16_t* x, int K) {
  float acc = 0.0f;
  for (int b = 0; b < K / 256; ++b) {
    const uint8_t* blk = row + (size_t)b * 98;
    const float d0 = fp16_to_f32(load_u16le(blk));
    const uint8_t* qs = blk + 2;  // 96
    const uint8_t* gas = qs + 64;  // 8 uint32 scale words
    for (int ib = 0; ib < 8; ++ib) {
      const uint32_t aux32 = (uint32_t)load_u16le(gas + 4 * ib) |
                             ((uint32_t)load_u16le(gas + 4 * ib + 2) << 16);
      const float d = d0 * (0.5f + (float)(aux32 >> 28)) * 0.5f;
      const uint8_t* q3 = qs + 8 * ib;
      const bf16_t* xb = x + (size_t)b * 256 + 32 * ib;
      for (int il = 0; il < 4; ++il) {
        const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
        const uint8_t* g1 = reinterpret_cast<const uint8_t*>(iq3xxs_grid + q3[2 * il + 0]);
        const uint8_t* g2 = reinterpret_cast<const uint8_t*>(iq3xxs_grid + q3[2 * il + 1]);
        for (int j = 0; j < 4; ++j) {
          acc += d * (float)g1[j] * (signs & kmask_iq2xs[j] ? -1.0f : 1.0f) * bf16_to_f32(xb[8 * il + j]);
          acc += d * (float)g2[j] * (signs & kmask_iq2xs[j + 4] ? -1.0f : 1.0f) * bf16_to_f32(xb[8 * il + 4 + j]);
        }
      }
    }
  }
  return acc;
}

using ggufdot_fn = float (*)(const uint8_t* row, const bf16_t* x, int K);

inline bool is_gguf_fmt(int f) { return f == WF_IQ3_XXS || f == WF_IQ4_XS || f == WF_Q6_K; }

inline int gguf_block_bytes(int f) {
  if (f == WF_IQ3_XXS) return 98;
  if (f == WF_IQ4_XS) return 136;
  return 210;  // WF_Q6_K
}

// Packed-row stride for K contiguous input dims (K % 256 == 0).
inline int gguf_row_bytes(int f, int K) { return (K / 256) * gguf_block_bytes(f); }

// Per-fmt dot selection for the SCALAR tier (bf16 activations, fp32 LUT dequant):
// correctness reference and fallback. The AVX2 integer tier (select_ggufdot_i8 in
// the W4A8-K section below) dispatches at executor construction when the CPU has
// AVX2+FMA.
ggufdot_fn select_ggufdot(int f) {
  if (f == WF_IQ3_XXS) return iq3_xxs_dot_scalar;
  if (f == WF_IQ4_XS) return iq4_xs_dot_scalar;
  if (f == WF_Q6_K) return q6_k_dot_scalar;
  return nullptr;
}

// ---------------- GGUF integer tier: W4A8-K (q8_K activations) ---------------
// Verbatim AVX2 integer dot kernels for the three gguf formats, ported from
// llama.cpp / ggml (MIT, "Copyright (c) 2023-2026 The ggml authors"). Sources
// (llama.cpp checkout, haswell variant flags -mf16c -mfma -mavx -mavx2):
//   ggml_vec_dot_q6_K_q8_K    ggml/src/ggml-cpu/arch/x86/quants.c:2426 (AVX2 body :2439)
//   ggml_vec_dot_iq3_xxs_q8_K arch/x86/quants.c:3260  (AVX2 body :3274)
//   ggml_vec_dot_iq4_xs_q8_K  arch/x86/quants.c:4004  (AVX2 body :4022)
//   quantize_row_q8_K_ref     ggml/src/ggml-quants.c:2768
//   helpers                   arch/x86/quants.c:46 hsum_float_8, :68 mul_add_epi8,
//                             :540 get_scale_shuffle; MM256_SET_M128I (simd-mappings.h)
// Adaptations (everything else is upstream text, upstream 4-space indent kept for
// diff-fidelity against future re-syncs):
//   - ggml's per-variant `#if defined __AVX2__` dispatch is dropped: these ARE the
//     AVX2 branches, compiled via the target attribute like every other SIMD kernel
//     in this file. The three kernels run mul_add_epi8 / maddubs / madd chains, not
//     the VNNI-dispatched helper, so haswell == alderlake codegen for them.
//   - GGML_CPU_FP16_TO_FP32(x.d) -> fp16_to_f32(x.d): the local exact fp16 decoder
//     the scalar kernels above already use.
//   - ggml's block structs -> the layout-identical Block* structs below; MIN ->
//     std::min; the bsums/blocks math is untouched.
// The decision-gate microbench ran these exact bodies (verbatim include, zero
// transcription) at 53-66 GB/s sustained on the real glm5next decode shape vs
// 11.3 GB/s for the scalar tier, AVX2-vs-ggml-scalar parity < 1e-3 rel
// (.tasks/gguf-glm5next-hybrid/verification/ggml-microbench.md).

constexpr int QK_K = 256;  // ggml K-quant super-block width (ggml-common.h)

// ggml-common.h:371-376: block_q8_K = f32 d + int8 qs[QK_K] + int16 bsums[QK_K/16].
struct BlockQ8K {
  float d;
  int8_t qs[QK_K];
  int16_t bsums[QK_K / 16];
};
static_assert(sizeof(BlockQ8K) == 292, "block_q8_K layout drifted from ggml-common.h");
// ggml-common.h:362-368: block_q6_K = ql, qh, int8 scales, f16 d LAST (208:210).
struct BlockQ6K {
  uint8_t ql[QK_K / 2];
  uint8_t qh[QK_K / 4];
  int8_t scales[QK_K / 16];
  uint16_t d;
};
static_assert(sizeof(BlockQ6K) == 210, "block_q6_K layout drifted from ggml-common.h");
// ggml-common.h:405-411: block_iq3_xxs = f16 d FIRST + qs[3*QK_K/8].
struct BlockIq3Xxs {
  uint16_t d;
  uint8_t qs[3 * QK_K / 8];
};
static_assert(sizeof(BlockIq3Xxs) == 98, "block_iq3_xxs layout drifted from ggml-common.h");
// ggml-common.h:455-460: block_iq4_xs = f16 d FIRST + u16 scales_h + scales_l + qs.
struct BlockIq4Xs {
  uint16_t d;
  uint16_t scales_h;
  uint8_t scales_l[QK_K / 64];
  uint8_t qs[QK_K / 2];
};
static_assert(sizeof(BlockIq4Xs) == 136, "block_iq4_xs layout drifted from ggml-common.h");

// ggml-quants.c:621 (verbatim): round-to-nearest-even via the fp32 mantissa trick.
static inline int nearest_int(float fval) {
    assert(fabsf(fval) <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

// ggml-quants.c:2768 quantize_row_q8_K_ref (verbatim). Per 256-block: SIGNED amax
// (sign kept - see iscale), iscale = -127/max NEGATIVE on purpose (the sign cancels
// in the dot; d = 1/iscale absorbs it; required by the IQ kernels' maddubs operand
// order), upper-side clamp at 127 only, and per-16 bsums consumed by the q6_K
// offset trick.
void quantize_row_q8_K_ref(const float * x, BlockQ8K * y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        float max = 0;
        float amax = 0;
        for (int j = 0; j < QK_K; ++j) {
            float ax = fabsf(x[j]);
            if (ax > amax) {
                amax = ax; max = x[j];
            }
        }
        if (!amax) {
            y[i].d = 0;
            memset(y[i].qs, 0, QK_K);
            x += QK_K;
            continue;
        }
        // We need this change for IQ2_XXS, else the AVX implementation becomes very awkward
        const float iscale = -127.f/max;
        for (int j = 0; j < QK_K; ++j) {
            int v = nearest_int(iscale*x[j]);
            y[i].qs[j] = std::min(127, v);
        }
        for (int j = 0; j < QK_K/16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) {
                sum += y[i].qs[j*16 + ii];
            }
            y[i].bsums[j] = sum;
        }
        y[i].d = 1/iscale;
        x += QK_K;
    }
}

#if CPU_MOE_X86
// ggml arch/x86/quants.c:46-56 (verbatim): horizontally add 8 floats. (ggml
// compiles the whole file with -mavx2; here the ISA rides the per-function
// target attribute like every other SIMD helper in this file.)
__attribute__((target("avx2")))
static inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

// ggml arch/x86/quants.c:68-74 (verbatim).
__attribute__((target("avx2")))
static inline __m256i mul_add_epi8(const __m256i x, const __m256i y) {
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    return _mm256_maddubs_epi16(ax, sy);
}

// ggml arch/x86/quants.c:540-552 (verbatim).
static inline __m128i get_scale_shuffle(int i) {
    static const uint8_t k_shuffle[128] = {
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
         2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
         4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
         6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
         8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
        10,10,10,10,10,10,10,10, 11,11,11,11,11,11,11,11,
        12,12,12,12,12,12,12,12, 13,13,13,13,13,13,13,13,
        14,14,14,14,14,14,14,14, 15,15,15,15,15,15,15,15
    };
    return _mm_loadu_si128((const __m128i*)k_shuffle + i);
}

// ggml-cpu/arch/x86/quants.c:26 (verbatim definition).
#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

// ggml arch/x86/quants.c:2624 keven_signs_q2xs: 128 groups of 8 +-1 sign bytes (the
// AVX2 iq3_xxs kernels gather one 64-bit lane group at a time). ggml holds 1024
// literals; entry [8*i + k] is exactly (ksigns_iq2xs[i] & kmask_iq2xs[k]) ? -1 : 1
// over the vendored LUTs above (spot-checked against ggml's table), so it is
// derived here - single source of truth stays the vendored files.
inline const int8_t* keven_signs_q2xs() {
  static const std::array<int8_t, 1024> table = [] {
    std::array<int8_t, 1024> t{};
    for (int i = 0; i < 128; ++i)
      for (int k = 0; k < 8; ++k)
        t[8 * i + k] = (ksigns_iq2xs[i] & kmask_iq2xs[k]) ? int8_t{-1} : int8_t{1};
    return t;
  }();
  return table.data();
}

// ggml's UNUSED (ggml-impl.h) for the verbatim bodies below; undefined after.
#define UNUSED(x) (void)(x)

// ggml_vec_dot_q6_K_q8_K AVX2 branch (arch/x86/quants.c:2426-2508, verbatim).
__attribute__((target("avx2,fma")))
void ggml_vec_dot_q6_K_q8_K(int n, float * __restrict s, size_t bs, const void * __restrict vx, size_t bx, const void * __restrict vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const BlockQ6K * __restrict x = (const BlockQ6K *)vx;
    const BlockQ8K * __restrict y = (const BlockQ8K *)vy;

    const int nb = n / QK_K;

    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i m15 = _mm256_set1_epi8(15);

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * fp16_to_f32(x[i].d);

        const uint8_t * __restrict q4 = x[i].ql;
        const uint8_t * __restrict qh = x[i].qh;
        const int8_t  * __restrict q8 = y[i].qs;

        const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y[i].bsums);
        const __m128i scales = _mm_loadu_si128((const __m128i*)x[i].scales);
        const __m256i scales_16 = _mm256_cvtepi8_epi16(scales);
        const __m256i q8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales_16), 5);

        __m256i sumi = _mm256_setzero_si256();

        int is = 0;

        for (int j = 0; j < QK_K/128; ++j) {
            const __m256i q4bits1 = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
            const __m256i q4bits2 = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
            const __m256i q4bitsH = _mm256_loadu_si256((const __m256i*)qh); qh += 32;

            const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m3), 4);
            const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(12)), 2);
            const __m256i q4h_2 = _mm256_and_si256(q4bitsH, _mm256_set1_epi8(48));
            const __m256i q4h_3 = _mm256_srli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(-64)), 2);

            const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
            const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
            const __m256i q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h_2);
            const __m256i q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h_3);

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

            __m256i p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
            __m256i p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
            __m256i p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
            __m256i p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

            const __m128i scale_0 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 0));
            const __m128i scale_1 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 1));
            const __m128i scale_2 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 2));
            const __m128i scale_3 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 3));
            is += 4;

            p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
            p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
            p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2);
            p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));

        }

        sumi = _mm256_sub_epi32(sumi, q8sclsub);
        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }

    *s = hsum_float_8(acc);
}

// ggml_vec_dot_iq3_xxs_q8_K AVX2 branch (arch/x86/quants.c:3260-3350, verbatim).
__attribute__((target("avx2,fma")))
void ggml_vec_dot_iq3_xxs_q8_K(int n, float * __restrict s, size_t bs, const void * __restrict vx, size_t bx, const void * __restrict vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const BlockIq3Xxs * __restrict x = (const BlockIq3Xxs *)vx;
    const BlockQ8K    * __restrict y = (const BlockQ8K *)vy;

    const int nb = n / QK_K;

    const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs();

    uint32_t aux32[2];

    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = fp16_to_f32(x[i].d) * y[i].d;
        const uint8_t * __restrict q3 = x[i].qs;
        const uint8_t * __restrict gas = x[i].qs + QK_K/4;
        const int8_t  * __restrict q8 = y[i].qs;
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q2_1 = _mm256_set_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]],
                                                  iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
            q3 += 8;
            const __m256i q2_2 = _mm256_set_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]],
                                                  iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
            q3 += 8;
            memcpy(aux32, gas, 8); gas += 8;
            const __m256i s2_1 = _mm256_set_epi64x(signs64[(aux32[0] >> 21) & 127], signs64[(aux32[0] >> 14) & 127],
                                                   signs64[(aux32[0] >>  7) & 127], signs64[(aux32[0] >>  0) & 127]);
            const __m256i s2_2 = _mm256_set_epi64x(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127],
                                                   signs64[(aux32[1] >>  7) & 127], signs64[(aux32[1] >>  0) & 127]);
            const __m256i q8s_1 = _mm256_sign_epi8(q8_1, s2_1);
            const __m256i q8s_2 = _mm256_sign_epi8(q8_2, s2_2);
            const __m256i dot1  = _mm256_maddubs_epi16(q2_1, q8s_1);
            const __m256i dot2  = _mm256_maddubs_epi16(q2_2, q8s_2);
            const uint16_t ls1 = aux32[0] >> 28;
            const uint16_t ls2 = aux32[1] >> 28;
            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16(2*ls1+1));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16(2*ls2+1));
            sumi1 = _mm256_add_epi32(sumi1, p1);
            sumi2 = _mm256_add_epi32(sumi2, p2);
        }

        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);

    }

    *s = 0.25f * hsum_float_8(accumf);
}

// ggml_vec_dot_iq4_xs_q8_K AVX2 branch (arch/x86/quants.c:4004-4064, verbatim).
__attribute__((target("avx2,fma")))
void ggml_vec_dot_iq4_xs_q8_K(int n, float * __restrict s, size_t bs, const void * __restrict vx, size_t bx, const void * __restrict vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    assert(n % QK_K == 0);

    const BlockIq4Xs * __restrict x = (const BlockIq4Xs *)vx;
    const BlockQ8K   * __restrict y = (const BlockQ8K *)vy;

    const int nb = n / QK_K;

    const __m128i values128 = _mm_loadu_si128((const __m128i*)kvalues_iq4nl);
    const __m128i m4b  = _mm_set1_epi8(0x0f);

    __m256 accum = _mm256_setzero_ps();
    for (int ibl = 0; ibl < nb; ++ibl) {
        const uint8_t * qs = x[ibl].qs;
        const int8_t  * q8 = y[ibl].qs;
        uint16_t sh = x[ibl].scales_h;
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib = 0; ib < QK_K/32; ib += 2) {
            const __m128i q4bits_1 = _mm_loadu_si128((const __m128i*)qs);  qs += 16;
            const __m128i q4bits_2 = _mm_loadu_si128((const __m128i*)qs);  qs += 16;
            const __m256i q8b_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8b_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q4b_1 = MM256_SET_M128I(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b)),
                                                  _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b)));
            const __m256i q4b_2 = MM256_SET_M128I(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b)),
                                                  _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_2, m4b)));
            const __m256i p16_1 = mul_add_epi8(q4b_1, q8b_1);
            const __m256i p16_2 = mul_add_epi8(q4b_2, q8b_2);
            const int16_t ls1 = ((x[ibl].scales_l[ib/2] & 0xf) | ((sh << 4) & 0x30)) - 32;
            const int16_t ls2 = ((x[ibl].scales_l[ib/2] >>  4) | ((sh << 2) & 0x30)) - 32;
            sh >>= 4;
            const __m256i p_1 = _mm256_madd_epi16(p16_1, _mm256_set1_epi16(ls1));
            const __m256i p_2 = _mm256_madd_epi16(p16_2, _mm256_set1_epi16(ls2));
            sumi1 = _mm256_add_epi32(p_1, sumi1);
            sumi2 = _mm256_add_epi32(p_2, sumi2);
        }
        accum = _mm256_fmadd_ps(_mm256_set1_ps(fp16_to_f32(x[ibl].d)*y[ibl].d),
                _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accum);
    }

    *s = hsum_float_8(accum);
}

#undef MM256_SET_M128I
#undef UNUSED

// nrc=1 adapters onto the executor's GEMV call shape: one packed weight row vs one
// q8_K activation row. The activation row is quantized ONCE per task (input row in
// submit(), intermediate per (token,route) in the prepare phase) and reused across
// every expert dot that reads it - the same amortization the nvfp4/q4_0 W4A8 paths use.
using ggufdot_i8_fn = float (*)(const uint8_t* row, const void* q8, int K);

float q6_k_dot_i8(const uint8_t* row, const void* q8, int K) {
  float s;
  ggml_vec_dot_q6_K_q8_K(K, &s, sizeof(float), row, 0, q8, 0, 1);
  return s;
}

float iq3_xxs_dot_i8(const uint8_t* row, const void* q8, int K) {
  float s;
  ggml_vec_dot_iq3_xxs_q8_K(K, &s, sizeof(float), row, 0, q8, 0, 1);
  return s;
}

float iq4_xs_dot_i8(const uint8_t* row, const void* q8, int K) {
  float s;
  ggml_vec_dot_iq4_xs_q8_K(K, &s, sizeof(float), row, 0, q8, 0, 1);
  return s;
}

// Tier for the gguf dots. "scalar" = the fp32-LUT kernels above (correctness
// reference + fallback, bf16 activations); "avx2" = the verbatim ggml integer
// kernels (W4A8-K: q8_K activations). Default auto picks AVX2 when the CPU has
// AVX2+FMA - the same detection pattern as pick_isa - scalar otherwise.
// FREETOKEN_GGUF_DOT_TIER={scalar,avx2} overrides (A/B the tiers); an explicit
// tier is capped down at CPU support, never forced above it (pick_isa semantics).
enum GgufDotTier { GGUF_DOT_SCALAR = 0, GGUF_DOT_AVX2 = 1 };

inline GgufDotTier pick_gguf_dot_tier() {
  GgufDotTier best =
      (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) ? GGUF_DOT_AVX2
                                                                        : GGUF_DOT_SCALAR;
  if (const char* f = getenv("FREETOKEN_GGUF_DOT_TIER")) {
    GgufDotTier want = best;
    if (!std::strcmp(f, "scalar")) want = GGUF_DOT_SCALAR;
    else if (!std::strcmp(f, "avx2")) want = GGUF_DOT_AVX2;
    return want < best ? want : best;  // cap downward; never force above hw support
  }
  return best;
}

inline ggufdot_i8_fn select_ggufdot_i8(int f) {
  if (f == WF_IQ3_XXS) return iq3_xxs_dot_i8;
  if (f == WF_IQ4_XS) return iq4_xs_dot_i8;
  if (f == WF_Q6_K) return q6_k_dot_i8;
  return nullptr;
}
#endif  // CPU_MOE_X86

// Each ctor pointer arg is the address of a CPU int64 array of length
// num_layers (one base address per layer, built by cpu_executor.py's
// _make_table), not a single flat bank. tbl_at resolves
// tbl[layer_id] once per task/pass; a null table (bank unused by this fmt, ptr
// arg 0) resolves to nullptr without dereferencing.
inline const void* tbl_at(const uint64_t* tbl, int layer_id) {
  return tbl ? reinterpret_cast<const void*>(tbl[layer_id]) : nullptr;
}

struct CpuMoeExecutor {
  int num_threads;
  int num_layers, num_experts, top_k;
  int H, I;
  int act, apply_on_input;
  int fmt;                // WFmt
  bool needs_di = false;  // pre-deinterleave activations to fp32 (nvfp4/ds_fp4)
  // Per-layer pointer tables (one base address per layer, see tbl_at). gate_up_tbl
  // doubles as the bf16 gate_up table and the nvfp4/mxfp4/q4_0/ds_fp4 packed-gate_up
  // table (down_tbl likewise for down); which reinterpretation applies is picked by
  // fmt at each resolve site (see gemm1_dot/gemm2_dot/do_pass1_mxfp4/do_pass1_dsfp4).
  const uint64_t* gate_up_tbl;   // bf16: [E,2I,H] rows; else: packed e2m1/mxfp4-blocks
  const uint64_t* down_tbl;      // bf16: [E,H,I] rows; else: packed e2m1/mxfp4-blocks
  const uint64_t* gu_scale_tbl;  // nvfp4/mxfp4/ds_fp4: [E,2I,*] block scales
  const uint64_t* gu_global_tbl; // nvfp4: [E,2I] fp16 row globals
  const uint64_t* dn_scale_tbl;  // nvfp4/mxfp4/ds_fp4: [E,H,*] block scales
  const uint64_t* dn_global_tbl; // nvfp4: [E,H] fp16 row globals
  const uint64_t* gu_bias_tbl;   // mxfp4: [E,2I] bf16 biases
  const uint64_t* dn_bias_tbl;   // mxfp4: [E,H] bf16 biases
  float swiglu_alpha;
  float swiglu_limit;          // +inf == no clamp
  dot_fn dot;
  nvdot_fn nvdot;
  nvi8dot_fn nvi8dot = nullptr;  // AVX-VNNI W4A8 nvfp4 dot (nullptr -> use fp32 nvdot)
  bool use_vnni = false;         // nvfp4 + AVX-VNNI: decode via int8 VPDPBUSD (W4A8)
  bool use_q4a8 = false;       // q4_0: always W4A8 (llama.cpp Q4_0 x Q8_0); int8 pre-quant
  dsdot_fn dsdot;
  mxgemv_fn mxgemv;
  q4dot_fn q4dot;
  // GGUF K-quant family: SEPARATE per-role tables + per-role fmt (gate/up/down may
  // differ within one layer; the legacy formats share gate_up_tbl with the up row at
  // I+i, and keep gate_tbl/up_tbl null). Row strides derive from (role fmt, K).
  const uint64_t* gate_tbl = nullptr;  // gguf split: [E, I, row_bytes] per layer
  const uint64_t* up_tbl = nullptr;    // gguf split: [E, I, row_bytes] per layer
  int fmt_up = 0, fmt_down = 0;        // WFmt per role (== fmt for legacy formats)
  bool is_gguf = false;
  ggufdot_fn gdot = nullptr, udot = nullptr, ddot = nullptr;
  int g_row_bytes = 0, u_row_bytes = 0, d_row_bytes = 0;
  // W4A8-K integer tier (verbatim ggml AVX2 kernels): selected at construction
  // when the CPU has AVX2+FMA (FREETOKEN_GGUF_DOT_TIER overrides). Reads the SAME
  // packed rows as the scalar tier - only the activation format changes (q8_K,
  // quantized once per row and reused across the expert dots).
  bool gguf_i8 = false;
  ggufdot_i8_fn gdot_i8 = nullptr, udot_i8 = nullptr, ddot_i8 = nullptr;
  // ds_fp4: the caller already FP8-round-tripped the input activations on the GPU
  // (same reference grid), so submit() must not repeat it on the host-callback
  // thread. That scalar per-element pass is single-threaded ON THE DECODE CRITICAL
  // PATH (~0.3ms/layer at H=4096, every worker and the GPU waiting on it); moving
  // it to a captured GPU elementwise kernel removes it while keeping the official
  // W4A8 numerics bit-exact. Set via set_input_prequant (see cpu_executor.py).
  bool input_prequant = false;
  // Q4_0 packed-row byte strides (H/32*18 for gate_up over K=H, I/32*18 for down over K=I).
  int q4_gu_row_bytes = 0, q4_dn_row_bytes = 0;
  float e2m1_lut[16];
  float e4m3_lut[256];
  float e8m0_lut[256];         // mxfp4 block scale: 2^(s-127), s clamped to [0,254]
  const char* isa;

  std::vector<bf16_t> g_scratch;   // [max_tokens * top_k * I] intermediate
  std::vector<bf16_t> xq_scratch;  // [max_tokens * H] ds_fp4 fp8-roundtripped input
  // ds_fp4 activations pre-deinterleaved to fp32 (even/odd K) for the row-major dot.
  std::vector<float> xe_scratch, xo_scratch;  // [max_tokens * H/2]   (input)
  std::vector<float> ge_scratch, go_scratch;  // [max_tokens*top_k*I/2] (intermediate)
  // AVX-VNNI W4A8: per-16-block int8 activations [even(8),odd(8)] + per-block scale.
  std::vector<int8_t> xi8_scratch, gi8_scratch;  // [max_tokens*H], [max_tokens*top_k*I]
  std::vector<float> xas_scratch, gas_scratch;   // [max_tokens*H/16], [..*top_k*I/16]
  // GGUF W4A8-K: q8_K activation rows. The input row (H elems -> H/256 blocks) is
  // quantized once per task in submit(); each (token,route) intermediate (I elems
  // -> I/256 blocks) once in the prepare phase. 292 B per 256-elem block =
  // ~23 KB/token at the glm5next geometry (H=4096, I=2048, top_k=8) - pre-sized in
  // the ctor, grown at most once in submit(); the hot path only indexes into them
  // (no hot-path allocations).
  std::vector<BlockQ8K> xq8k_scratch;  // [max_tokens * H/256]
  std::vector<BlockQ8K> gq8k_scratch;  // [max_tokens * top_k * I/256]
  std::string isa_str;

  std::vector<std::thread> workers;
  std::mutex task_mtx;
  std::condition_variable task_cv;
  std::mutex sync_mtx;
  std::condition_variable sync_cv;

  bool stop = false;
  uint64_t cur_gen = 0;
  MoeTask* cur_task = nullptr;
  std::atomic<uint64_t> submitted{0};
  std::atomic<uint64_t> completed{0};

  std::atomic<int64_t> p1_next{0};
  std::atomic<int64_t> p2_next{0};
  std::atomic<int64_t> prt_next{0};  // ds_fp4 intermediate fp8 round-trip phase
  int64_t p1_total = 0, p2_total = 0, prt_total = 0;
  int n_iblk = 0, n_hblk = 0;
  std::atomic<int> done_count{0};
  std::atomic<int> bar_count{0};
  std::atomic<int> bar_sense{0};

  std::vector<MoeTask*> owned_tasks;  // persistent task descriptors (graph-stable)
  std::vector<int> core_ids;          // worker tid -> logical CPU to pin to (may be empty)

  // ---- Flag-based GPU<->CPU handshake (replaces the per-layer cudaLaunchHostFunc pair) ----
  // A tiny GPU kernel bumps ready_flags[slot] at submit; this coordinator thread busy-polls
  // it, runs the slot's task on the worker pool, and sets done_flags[slot], which a GPU
  // spin-wait kernel polls at sync. This removes the ~2x30-50us host-func dispatch round
  // trips per MoE layer per decode step that otherwise idle the GPU (~6 ms/step on a
  // 75-layer model). One slot per (layer, decode batch size) pair -- the Python side
  // allocates slots as tasks are created. Flags live in mapped-pinned host memory (UVA:
  // the same pointers are used by the GPU kernels and by this thread).
  std::thread coord_thread;
  std::atomic<bool> coord_stop{false};
  volatile int64_t* ready_flags = nullptr;  // GPU increments, this thread polls
  volatile int64_t* done_flags = nullptr;   // this thread sets, GPU spin-waits
  int coord_num_slots = 0;
  std::vector<MoeTask*> flag_task;           // slot -> task (registered lazily)
  std::vector<int64_t> flag_served;          // slot -> completed dispatch count (tests/debug)
  std::mutex flag_task_mtx;

  // Portable ordering for the flag handshake: "ready observed => the DMA'd inputs that
  // preceded the bump are visible" and "y stores are visible before done". Plain
  // volatile loads lean on x86 TSO; acquire/release makes it hold on aarch64 too
  // (GH200/Jetson) at zero x86 cost. (MSVC branch is x86-only today: compiler barrier
  // + TSO.)
  static int64_t flag_load_acquire(const volatile int64_t* p) {
#if defined(_MSC_VER)
    const int64_t v = *p;
    _ReadWriteBarrier();
    return v;
#else
    return __atomic_load_n(const_cast<const int64_t*>(p), __ATOMIC_ACQUIRE);
#endif
  }

  static void flag_store_release(volatile int64_t* p, int64_t v) {
#if defined(_MSC_VER)
    _ReadWriteBarrier();
    *p = v;
#else
    __atomic_store_n(const_cast<int64_t*>(p), v, __ATOMIC_RELEASE);
#endif
  }

  CpuMoeExecutor(int num_threads_, int num_layers_, int num_experts_, int top_k_,
                 int hidden_size, int inter_size, int max_tokens, int activation_id,
                 int apply_router_weight_on_input, int weight_format,
                 uintptr_t gate_up_ptr, uintptr_t down_ptr, uintptr_t gate_up_scale_ptr,
                 uintptr_t gate_up_global_ptr, uintptr_t down_scale_ptr,
                 uintptr_t down_global_ptr, uintptr_t gate_up_bias_ptr,
                 uintptr_t down_bias_ptr, double swiglu_alpha_, double swiglu_limit_,
                 std::vector<int> core_ids_, int up_fmt_ = -1, int down_fmt_ = -1,
                 uintptr_t gate_ptr_ = 0, uintptr_t up_ptr_ = 0)
      : num_threads(num_threads_ > 0 ? num_threads_ : 1),
        num_layers(num_layers_),
        num_experts(num_experts_),
        top_k(top_k_),
        H(hidden_size),
        I(inter_size),
        act(activation_id),
        apply_on_input(apply_router_weight_on_input),
        fmt(weight_format),
        gate_up_tbl(reinterpret_cast<const uint64_t*>(gate_up_ptr)),
        down_tbl(reinterpret_cast<const uint64_t*>(down_ptr)),
        gu_scale_tbl(reinterpret_cast<const uint64_t*>(gate_up_scale_ptr)),
        gu_global_tbl(reinterpret_cast<const uint64_t*>(gate_up_global_ptr)),
        dn_scale_tbl(reinterpret_cast<const uint64_t*>(down_scale_ptr)),
        dn_global_tbl(reinterpret_cast<const uint64_t*>(down_global_ptr)),
        gu_bias_tbl(reinterpret_cast<const uint64_t*>(gate_up_bias_ptr)),
        dn_bias_tbl(reinterpret_cast<const uint64_t*>(down_bias_ptr)),
        swiglu_alpha(static_cast<float>(swiglu_alpha_)),
        swiglu_limit(static_cast<float>(swiglu_limit_)),
        core_ids(std::move(core_ids_)) {
    switch (weight_format) {
      case WF_BF16:
      case WF_NVFP4:
      case WF_MXFP4:
      case WF_DSFP4:
      case WF_Q4_0:
      case WF_IQ3_XXS:
      case WF_IQ4_XS:
      case WF_Q6_K:
        break;
      default:
        throw std::runtime_error("unknown weight_format for the CPU MoE executor");
    }
    // Per-role formats: -1 inherits the primary (gate) fmt; the gguf resolver passes
    // all three explicitly (gate/up/down may differ within one layer).
    fmt_up = up_fmt_ < 0 ? fmt : up_fmt_;
    fmt_down = down_fmt_ < 0 ? fmt : down_fmt_;
    gate_tbl = reinterpret_cast<const uint64_t*>(gate_ptr_);
    up_tbl = reinterpret_cast<const uint64_t*>(up_ptr_);
    DotChoice c = select_dot();
    dot = c.fn;
    nvdot = select_nvdot();
    dsdot = select_dsdot();
    mxgemv = select_mxgemv();
    q4dot = select_q4dot();
    if (weight_format == WF_Q4_0) {
      if (H % 32 != 0 || I % 32 != 0)
        throw std::runtime_error("Q4_0 CPU MoE requires H and I to be multiples of 32");
      q4_gu_row_bytes = (H / 32) * 18;  // K = H (gate_up rows)
      q4_dn_row_bytes = (I / 32) * 18;  // K = I (down rows)
    }
    isa = c.name;
    // nvfp4 (AVX-VNNI only): W4A8 int8 decode when the CPU supports it. q4_0 is always
    // W4A8 (activations pre-quantized to Q8_0); select_q4dot picks VPDPBUSD / VPMADDUBSW
    // / scalar for the tier, so the tag reflects which of those q4dot resolved to.
    nvi8dot = select_nvi8dot();
    use_vnni = (weight_format == WF_NVFP4) && (nvi8dot != nullptr);
    use_q4a8 = (weight_format == WF_Q4_0);
    // gguf K-quant family: all three roles must be gguf formats with per-role tables
    // (the gguf file has no packed gate_up bank); K-quant blocks are 256 wide.
    is_gguf = is_gguf_fmt(fmt) || is_gguf_fmt(fmt_up) || is_gguf_fmt(fmt_down);
    if (is_gguf) {
      if (!(is_gguf_fmt(fmt) && is_gguf_fmt(fmt_up) && is_gguf_fmt(fmt_down)) ||
          gate_tbl == nullptr || up_tbl == nullptr || down_tbl == nullptr)
        throw std::runtime_error(
            "gguf IQ/QK CPU MoE requires per-role gate/up/down formats and tables");
      if (H % 256 != 0 || I % 256 != 0)
        throw std::runtime_error("gguf IQ/QK CPU MoE requires H and I to be multiples of 256");
      // Per-role geometry: gate/up rows = I over K = H, down rows = H over K = I.
      g_row_bytes = gguf_row_bytes(fmt, H);
      u_row_bytes = gguf_row_bytes(fmt_up, H);
      d_row_bytes = gguf_row_bytes(fmt_down, I);
      gdot = select_ggufdot(fmt);
      udot = select_ggufdot(fmt_up);
      ddot = select_ggufdot(fmt_down);
      // Integer tier: the verbatim ggml AVX2 kernels (W4A8-K) when the CPU has
      // AVX2+FMA; the scalar bf16 dots stay resolved as the fallback tier.
#if CPU_MOE_X86
      if (pick_gguf_dot_tier() == GGUF_DOT_AVX2) {
        gdot_i8 = select_ggufdot_i8(fmt);
        udot_i8 = select_ggufdot_i8(fmt_up);
        ddot_i8 = select_ggufdot_i8(fmt_down);
        gguf_i8 = gdot_i8 != nullptr && udot_i8 != nullptr && ddot_i8 != nullptr;
      }
#endif
    }
    const char* q4tag = use_q4a8 ? (cpu_has_avxvnni() ? "+vnni(q4_0-w4a8)" : "+q4_0-w4a8") : "";
    const char* vnni_tag =
        cpu_has_avx512vnni() ? "+avx512vnni(nvfp4-w4a8)" : "+vnni(nvfp4-w4a8)";
    isa_str = std::string(c.name) + (use_vnni ? vnni_tag : "") + q4tag +
              (is_gguf ? std::string("+gguf(iq/qk)") +
                               (gguf_i8 ? "+avx2-w4a8k" : "+w4a16")
                       : "");
    isa = isa_str.c_str();
    for (int i = 0; i < 16; ++i) e2m1_lut[i] = kE2M1[i];
    for (int i = 0; i < 256; ++i) e4m3_lut[i] = e4m3_decode((uint8_t)i);
    // e8m0 (mxfp4 block scale) = 2^(s-127); the GPU GEMV clamps s to [0,254].
    for (int i = 0; i < 256; ++i) e8m0_lut[i] = std::ldexp(1.0f, std::min(i, 254) - 127);
    g_scratch.assign(static_cast<size_t>(max_tokens) * top_k * I, 0);
    // Row-major fp4 (nvfp4/ds_fp4) pre-deinterleaves activations to fp32 even/odd.
    needs_di = (fmt == WF_NVFP4 || fmt == WF_DSFP4);
    if (needs_di) {
      if (fmt == WF_DSFP4) xq_scratch.assign(static_cast<size_t>(max_tokens) * H, 0);
      xe_scratch.assign(static_cast<size_t>(max_tokens) * (H / 2), 0);
      xo_scratch.assign(static_cast<size_t>(max_tokens) * (H / 2), 0);
      ge_scratch.assign(static_cast<size_t>(max_tokens) * top_k * (I / 2), 0);
      go_scratch.assign(static_cast<size_t>(max_tokens) * top_k * (I / 2), 0);
      if (use_vnni) {
        xi8_scratch.assign(static_cast<size_t>(max_tokens) * H, 0);
        xas_scratch.assign(static_cast<size_t>(max_tokens) * (H / 16), 0);
        gi8_scratch.assign(static_cast<size_t>(max_tokens) * top_k * I, 0);
        gas_scratch.assign(static_cast<size_t>(max_tokens) * top_k * (I / 16), 0);
      }
    }
    // q4_0 W4A8: per-32-block Q8_0 activations (int8 + fp32 scale) for input + intermediate.
    if (use_q4a8) {
      xi8_scratch.assign(static_cast<size_t>(max_tokens) * H, 0);
      xas_scratch.assign(static_cast<size_t>(max_tokens) * (H / 32), 0);
      gi8_scratch.assign(static_cast<size_t>(max_tokens) * top_k * I, 0);
      gas_scratch.assign(static_cast<size_t>(max_tokens) * top_k * (I / 32), 0);
    }
    // GGUF W4A8-K: q8_K activation scratch (see the member comment for sizing).
    if (gguf_i8) {
      xq8k_scratch.assign(static_cast<size_t>(max_tokens) * (H / QK_K), BlockQ8K{});
      gq8k_scratch.assign(static_cast<size_t>(max_tokens) * top_k * (I / QK_K), BlockQ8K{});
    }
    for (int t = 0; t < num_threads; ++t)
      workers.emplace_back([this, t] { worker_loop(t); });
  }

  // Quantize a bf16 activation row to Q8_0 (llama.cpp): per-32-block symmetric int8 in
  // natural order + one fp32 scale (amax/127) per block. Done once per token/route,
  // amortized over every output row that the q4_0 W4A8 GEMV reads. K % 32 == 0.
  void quant_q8_0(const bf16_t* x, int K, int8_t* aq, float* asb) {
    const int nb = K / 32;
    for (int b = 0; b < nb; ++b) {
      const bf16_t* xb = x + (size_t)b * 32;
      float xf[32], amax = 0.0f;
      for (int j = 0; j < 32; ++j) {
        xf[j] = bf16_to_f32(xb[j]);
        amax = std::max(amax, std::fabs(xf[j]));
      }
      const float d = amax > 0.0f ? amax / 127.0f : 1.0f;
      asb[b] = d;
      const float inv = amax > 0.0f ? 1.0f / d : 0.0f;
      int8_t* o = aq + (size_t)b * 32;
      for (int j = 0; j < 32; ++j)
        o[j] = (int8_t)std::max(-127, std::min(127, (int)std::lround(xf[j] * inv)));
    }
  }

  // Quantize the pre-deinterleaved fp32 even/odd activations to per-16-block int8 in the
  // [even(8),odd(8)] layout the VNNI dot expects. Done once per token/route (amortized over
  // every output row), so a scalar pass is fine relative to the GEMV.
  void quant_i8_pg16(const float* xe, const float* xo, int K, int8_t* asi8, float* asb) {
    const int nb = K / 16;
    for (int b = 0; b < nb; ++b) {
      const float* xeb = xe + (size_t)b * 8;
      const float* xob = xo + (size_t)b * 8;
      float amax = 0.0f;
      for (int j = 0; j < 8; ++j)
        amax = std::max(amax, std::max(std::fabs(xeb[j]), std::fabs(xob[j])));
      const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
      asb[b] = s;
      const float inv = 1.0f / s;
      int8_t* ae = asi8 + (size_t)b * 16;
      for (int j = 0; j < 8; ++j) {
        int qe = (int)std::lround(xeb[j] * inv), qo = (int)std::lround(xob[j] * inv);
        ae[j] = (int8_t)std::max(-127, std::min(127, qe));
        ae[8 + j] = (int8_t)std::max(-127, std::min(127, qo));
      }
    }
  }

  // q8_K for the gguf W4A8-K tier: quantize_row_q8_K_ref (verbatim ggml, above)
  // over bf16 rows widened to fp32 one 256-block at a time. Done once per row -
  // the input row per task (submit), each intermediate per (token,route) in the
  // prepare phase - and reused across every expert dot that reads it.
  void quant_q8_k(const bf16_t* x, int K, BlockQ8K* out) {
    float xb[QK_K];
    for (int b = 0; b < K / QK_K; ++b) {
      for (int j = 0; j < QK_K; ++j) xb[j] = bf16_to_f32(x[(size_t)b * QK_K + j]);
      quantize_row_q8_K_ref(xb, out + b, QK_K);
    }
  }

  // gate_up output row `row` (in [0, 2I)) dotted with activation over K = H. ``e`` is
  // the layer-local expert row (0..num_experts); the layer bases (already resolved
  // once per task/pass by the caller via tbl_at) pick the layer's own tensors.
  // bf16 uses the interleaved bf16 row; nvfp4 uses the pre-split fp32 even/odd halves
  // (or, with AVX-VNNI, the per-16-block int8 activations).
  inline float gemm1_dot(const bf16_t* gate_up_l, const uint8_t* gu_packed_l,
                         const uint8_t* gu_scale_l, const uint16_t* gu_global_l, int e, int row,
                         const bf16_t* x, const float* xe, const float* xo, const int8_t* xi8,
                         const float* xas) {
    if (fmt == WF_BF16) {
      const bf16_t* w = gate_up_l + ((size_t)e * (2 * I) + row) * H;
      return dot(w, x, H);
    }
    if (fmt == WF_Q4_0) {
      const uint8_t* w =
          gu_packed_l + ((size_t)e * (2 * I) + row) * (size_t)q4_gu_row_bytes;
      return q4dot(w, xi8, xas, H);  // W4A8: int8 activations (Q8_0), scale in xas
    }
    const size_t r = (size_t)e * (2 * I) + row;
    if (use_vnni)
      return nvi8dot(gu_packed_l + r * (size_t)(H / 2), gu_scale_l + r * (size_t)(H / 16),
                     fp16_to_f32(gu_global_l[r]), xi8, H, e4m3_lut, xas);
    return nvdot(gu_packed_l + r * (size_t)(H / 2), gu_scale_l + r * (size_t)(H / 16),
                 fp16_to_f32(gu_global_l[r]), xe, xo, H, e2m1_lut, e4m3_lut);
  }

  // down output row `row` (in [0, H)) dotted with the intermediate over K = I. Same
  // layer-local-base convention as gemm1_dot.
  inline float gemm2_dot(const bf16_t* down_l, const uint8_t* dn_packed_l,
                         const uint8_t* dn_scale_l, const uint16_t* dn_global_l, int e, int row,
                         const bf16_t* g, const float* ge, const float* go, const int8_t* gi8,
                         const float* gas, const BlockQ8K* gq8) {
    if (fmt == WF_BF16) {
      const bf16_t* w = down_l + ((size_t)e * H + row) * I;
      return dot(w, g, I);
    }
    if (fmt == WF_Q4_0) {
      const uint8_t* w = dn_packed_l + ((size_t)e * H + row) * (size_t)q4_dn_row_bytes;
      return q4dot(w, gi8, gas, I);  // W4A8: int8 activations (Q8_0), scale in gas
    }
    if (is_gguf) {
      const uint8_t* w = dn_packed_l + ((size_t)e * H + row) * (size_t)d_row_bytes;
      if (gguf_i8) return ddot_i8(w, gq8, I);  // W4A8-K: q8_K intermediate (prep_g_row)
      return ddot(w, g, I);  // scalar tier: fused dequant over the raw bf16 intermediate
    }
    const size_t r = (size_t)e * H + row;
    if (use_vnni)
      return nvi8dot(dn_packed_l + r * (size_t)(I / 2), dn_scale_l + r * (size_t)(I / 16),
                     fp16_to_f32(dn_global_l[r]), gi8, I, e4m3_lut, gas);
    return nvdot(dn_packed_l + r * (size_t)(I / 2), dn_scale_l + r * (size_t)(I / 16),
                 fp16_to_f32(dn_global_l[r]), ge, go, I, e2m1_lut, e4m3_lut);
  }

  void pin_self(int tid) {
#if CPU_MOE_HAS_AFFINITY
    if (core_ids.empty()) return;
    const int cpu = core_ids[tid % static_cast<int>(core_ids.size())];
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
    (void)tid;
#endif
  }

  ~CpuMoeExecutor() {
    coord_stop.store(true);
    if (coord_thread.joinable()) coord_thread.join();
    {
      std::lock_guard<std::mutex> lk(task_mtx);
      stop = true;
    }
    task_cv.notify_all();
    for (auto& th : workers)
      if (th.joinable()) th.join();
    for (MoeTask* t : owned_tasks) delete t;
  }

  uintptr_t create_task(int layer_id, int num_tokens, uintptr_t x_ptr,
                        uintptr_t ids_ptr, uintptr_t w_ptr, uintptr_t y_ptr) {
    MoeTask* t = new MoeTask{this,
                             layer_id,
                             num_tokens,
                             reinterpret_cast<const bf16_t*>(x_ptr),
                             reinterpret_cast<const int32_t*>(ids_ptr),
                             reinterpret_cast<const float*>(w_ptr),
                             reinterpret_cast<bf16_t*>(y_ptr)};
    owned_tasks.push_back(t);
    return reinterpret_cast<uintptr_t>(t);
  }

  const char* isa_name() const { return isa; }

  void barrier(int& local_sense) {
    local_sense ^= 1;
    if (bar_count.fetch_add(1) + 1 == num_threads) {
      bar_count.store(0);
      bar_sense.store(local_sense);
    } else {
      while (bar_sense.load() != local_sense) {
#if CPU_MOE_X86
        _mm_pause();
#endif
      }
    }
  }

  void do_pass1(const MoeTask* t, int64_t p) {
    if (fmt == WF_MXFP4) {
      do_pass1_mxfp4(t, p);
      return;
    }
    if (fmt == WF_DSFP4) {
      do_pass1_dsfp4(t, p);
      return;
    }
    if (is_gguf) {
      do_pass1_gguf(t, p);
      return;
    }
    const int64_t ib = p % n_iblk;
    const int64_t tk = p / n_iblk;
    const int k = static_cast<int>(tk % top_k);
    const int tok = static_cast<int>(tk / top_k);
    const int e = t->ids[static_cast<size_t>(tok) * top_k + k];
    if (e < 0 || e >= num_experts) return;
    const float w_in = apply_on_input ? t->w[static_cast<size_t>(tok) * top_k + k] : 1.0f;
    // Resolve this task's layer base once; row indexing below is layer-local (e).
    const bf16_t* gate_up_l = reinterpret_cast<const bf16_t*>(tbl_at(gate_up_tbl, t->layer_id));
    const uint8_t* gu_packed_l = reinterpret_cast<const uint8_t*>(tbl_at(gate_up_tbl, t->layer_id));
    const uint8_t* gu_scale_l = reinterpret_cast<const uint8_t*>(tbl_at(gu_scale_tbl, t->layer_id));
    const uint16_t* gu_global_l =
        reinterpret_cast<const uint16_t*>(tbl_at(gu_global_tbl, t->layer_id));
    const bf16_t* x_row = t->x + (size_t)tok * H;
    const float* xe = needs_di ? xe_scratch.data() + (size_t)tok * (H / 2) : nullptr;
    const float* xo = needs_di ? xo_scratch.data() + (size_t)tok * (H / 2) : nullptr;
    const int8_t* xi8 =
        (use_vnni || use_q4a8) ? xi8_scratch.data() + (size_t)tok * H : nullptr;
    const float* xas = use_vnni ? xas_scratch.data() + (size_t)tok * (H / 16)
                     : use_q4a8 ? xas_scratch.data() + (size_t)tok * (H / 32)
                                  : nullptr;
    bf16_t* g_row = g_scratch.data() + ((size_t)tok * top_k + k) * I;
    const int i0 = static_cast<int>(ib) * IBLK;
    const int i1 = std::min(I, i0 + IBLK);
    for (int i = i0; i < i1; ++i) {
      // gate = row i, up = row I+i
      float gate =
          gemm1_dot(gate_up_l, gu_packed_l, gu_scale_l, gu_global_l, e, i, x_row, xe, xo, xi8, xas) * w_in;
      float up = gemm1_dot(gate_up_l, gu_packed_l, gu_scale_l, gu_global_l, e, I + i, x_row,
                           xe, xo, xi8, xas) * w_in;
      g_row[i] = act_epilogue(gate, up);
    }
  }

  // Shared pass-1 epilogue: silu/gelu families, or the clamped swiglu forms --
  // swigluoai carries the (up + 1) up bias (gpt-oss/MiniMax); swiglu_clamp (GLM-5.3)
  // does not. lim == +inf: no clamp.
  inline bf16_t act_epilogue(float gate, float up) {
    if (act == ACT_SWIGLUOAI || act == ACT_SWIGLU_CLAMP) {
      if (gate > swiglu_limit) gate = swiglu_limit;
      if (up > swiglu_limit) up = swiglu_limit;
      else if (up < -swiglu_limit) up = -swiglu_limit;
      const float glu = gate / (1.0f + std::exp(-gate * swiglu_alpha));
      return f32_to_bf16(glu * (up + (act == ACT_SWIGLUOAI ? 1.0f : 0.0f)));
    }
    return f32_to_bf16(act_apply(act, gate) * up);
  }

  // GGUF K-quant pass 1: gate/up are SEPARATE per-role tables ([E, I, row_bytes]
  // each, rows = I over K = H) with per-role fmt -- the gguf file has no packed
  // gate_up bank. Scalar tier: activations stay bf16 (no prequant), so the down
  // pass reads the raw bf16 intermediate. W4A8-K tier: the input row was
  // q8_K-quantized once per token in submit(); every gate/up dot of every route
  // reuses it (the quantized row is also why the down leg quantizes the
  // intermediate per route in prep_g_row).
  void do_pass1_gguf(const MoeTask* t, int64_t p) {
    const int64_t ib = p % n_iblk;
    const int64_t tk = p / n_iblk;
    const int k = static_cast<int>(tk % top_k);
    const int tok = static_cast<int>(tk / top_k);
    const int e = t->ids[static_cast<size_t>(tok) * top_k + k];
    if (e < 0 || e >= num_experts) return;
    const float w_in = apply_on_input ? t->w[static_cast<size_t>(tok) * top_k + k] : 1.0f;
    const uint8_t* gate_l = static_cast<const uint8_t*>(tbl_at(gate_tbl, t->layer_id));
    const uint8_t* up_l = static_cast<const uint8_t*>(tbl_at(up_tbl, t->layer_id));
    bf16_t* g_row = g_scratch.data() + ((size_t)tok * top_k + k) * I;
    const int i0 = static_cast<int>(ib) * IBLK;
    const int i1 = std::min(I, i0 + IBLK);
    if (gguf_i8) {
      const BlockQ8K* xq8 = xq8k_scratch.data() + (size_t)tok * (H / QK_K);
      for (int i = i0; i < i1; ++i) {
        const float gate =
            gdot_i8(gate_l + ((size_t)e * I + i) * (size_t)g_row_bytes, xq8, H) * w_in;
        const float up =
            udot_i8(up_l + ((size_t)e * I + i) * (size_t)u_row_bytes, xq8, H) * w_in;
        g_row[i] = act_epilogue(gate, up);
      }
      return;
    }
    const bf16_t* x_row = t->x + (size_t)tok * H;
    for (int i = i0; i < i1; ++i) {
      const float gate = gdot(gate_l + ((size_t)e * I + i) * (size_t)g_row_bytes, x_row, H) * w_in;
      const float up = udot(up_l + ((size_t)e * I + i) * (size_t)u_row_bytes, x_row, H) * w_in;
      g_row[i] = act_epilogue(gate, up);
    }
  }

  void do_pass2(const MoeTask* t, int64_t p) {
    if (fmt == WF_MXFP4) {
      do_pass2_mxfp4(t, p);
      return;
    }
    if (fmt == WF_DSFP4) {
      do_pass2_dsfp4(t, p);
      return;
    }
    const int64_t hb = p % n_hblk;
    const int tok = static_cast<int>(p / n_hblk);
    const int h0 = static_cast<int>(hb) * HBLK;
    const int h1 = std::min(H, h0 + HBLK);
    // Resolve this task's layer base once; row indexing below is layer-local (e).
    const bf16_t* down_l = reinterpret_cast<const bf16_t*>(tbl_at(down_tbl, t->layer_id));
    const uint8_t* dn_packed_l = reinterpret_cast<const uint8_t*>(tbl_at(down_tbl, t->layer_id));
    const uint8_t* dn_scale_l = reinterpret_cast<const uint8_t*>(tbl_at(dn_scale_tbl, t->layer_id));
    const uint16_t* dn_global_l =
        reinterpret_cast<const uint16_t*>(tbl_at(dn_global_tbl, t->layer_id));
    bf16_t* y_row = t->y + (size_t)tok * H;
    for (int h = h0; h < h1; ++h) {
      float acc = 0.0f;
      for (int k = 0; k < top_k; ++k) {
        const int e = t->ids[static_cast<size_t>(tok) * top_k + k];
        if (e < 0 || e >= num_experts) continue;
        const float w_out = apply_on_input ? 1.0f : t->w[static_cast<size_t>(tok) * top_k + k];
        const size_t gr = (size_t)tok * top_k + k;
        const bf16_t* g_row = g_scratch.data() + gr * I;
        const float* ge = needs_di ? ge_scratch.data() + gr * (I / 2) : nullptr;
        const float* go = needs_di ? go_scratch.data() + gr * (I / 2) : nullptr;
        const int8_t* gi8 = (use_vnni || use_q4a8) ? gi8_scratch.data() + gr * I : nullptr;
        const float* gas = use_vnni ? gas_scratch.data() + gr * (I / 16)
                         : use_q4a8 ? gas_scratch.data() + gr * (I / 32)
                                      : nullptr;
        const BlockQ8K* gq8 = gguf_i8 ? gq8k_scratch.data() + gr * (I / QK_K) : nullptr;
        acc += gemm2_dot(down_l, dn_packed_l, dn_scale_l, dn_global_l, e, h, g_row, ge, go, gi8,
                         gas, gq8) * w_out;
      }
      y_row[h] = f32_to_bf16(acc);
    }
  }

  // ----------------------------- mxfp4 (gpt-oss) -----------------------------
  // Transposed split-K layout (N innermost), so the GEMV streams K and accumulates
  // a contiguous block of N output columns -> cache-line-efficient, no repack, no
  // extra host memory. Pass 1 fuses gate_up + clamped-swiglu(+bias); pass 2 fuses
  // down(+bias) * router-weight, summed over the token's routes.
  //
  // Dequant: w = E2M1[code] * 2^(e8m0_scale - 127); two codes per byte (low nibble
  // first), one e8m0 scale per 32 contiguous K. Matches kernel/triton/mxfp4_moe.py.

  void do_pass1_mxfp4(const MoeTask* t, int64_t p) {
    const int64_t ib = p % n_iblk;
    const int64_t tk = p / n_iblk;
    const int k = static_cast<int>(tk % top_k);
    const int tok = static_cast<int>(tk / top_k);
    const int e = t->ids[static_cast<size_t>(tok) * top_k + k];
    if (e < 0 || e >= num_experts) return;
    // Resolve this task's layer base once; row indexing below is layer-local (e).
    const uint8_t* gu_packed_l = reinterpret_cast<const uint8_t*>(tbl_at(gate_up_tbl, t->layer_id));
    const uint8_t* gu_scale_l = reinterpret_cast<const uint8_t*>(tbl_at(gu_scale_tbl, t->layer_id));
    const bf16_t* gu_bias_l = reinterpret_cast<const bf16_t*>(tbl_at(gu_bias_tbl, t->layer_id));
    const int N2 = 2 * I;            // gate_up output width (gate/up interleaved)
    const int Hh = H / 2;            // packed-K rows
    const int i0 = static_cast<int>(ib) * IBLK;
    const int i1 = std::min(I, i0 + IBLK);
    const int nunit = i1 - i0;       // intermediate units owned by this tile
    const int col0 = 2 * i0;         // first gate_up column
    const int ncol = 2 * nunit;      // gate_up columns owned by this tile
    const bf16_t* x_row = t->x + (size_t)tok * H;
    const uint8_t* blk_e = gu_packed_l + (size_t)e * Hh * N2;
    const uint8_t* scl_e = gu_scale_l + (size_t)e * (size_t)(H / 32) * N2;
    float gu[2 * IBLK];
    mxgemv(gu, blk_e + col0, scl_e + col0, x_row, Hh, N2, ncol, e2m1_lut, e8m0_lut);
    const bf16_t* bias_e = gu_bias_l + (size_t)e * N2 + col0;
    bf16_t* g_row = g_scratch.data() + ((size_t)tok * top_k + k) * I;
    const float lim = swiglu_limit, alpha = swiglu_alpha;
    for (int j = 0; j < nunit; ++j) {
      float gate = gu[2 * j] + bf16_to_f32(bias_e[2 * j]);
      float up = gu[2 * j + 1] + bf16_to_f32(bias_e[2 * j + 1]);
      if (gate > lim) gate = lim;
      if (up > lim) up = lim;
      else if (up < -lim) up = -lim;
      const float glu = gate / (1.0f + std::exp(-gate * alpha));  // gate * sigmoid(alpha*gate)
      g_row[i0 + j] = f32_to_bf16(glu * (up + 1.0f));
    }
  }

  void do_pass2_mxfp4(const MoeTask* t, int64_t p) {
    const int64_t hb = p % n_hblk;
    const int tok = static_cast<int>(p / n_hblk);
    const int Ih = I / 2;            // packed-K rows
    const int h0 = static_cast<int>(hb) * HBLK;
    const int h1 = std::min(H, h0 + HBLK);
    const int nh = h1 - h0;
    // Resolve this task's layer base once; row indexing below is layer-local (e).
    const uint8_t* dn_packed_l = reinterpret_cast<const uint8_t*>(tbl_at(down_tbl, t->layer_id));
    const uint8_t* dn_scale_l = reinterpret_cast<const uint8_t*>(tbl_at(dn_scale_tbl, t->layer_id));
    const bf16_t* dn_bias_l = reinterpret_cast<const bf16_t*>(tbl_at(dn_bias_tbl, t->layer_id));
    float acc[HBLK];
    for (int c = 0; c < nh; ++c) acc[c] = 0.0f;
    for (int k = 0; k < top_k; ++k) {
      const int e = t->ids[static_cast<size_t>(tok) * top_k + k];
      if (e < 0 || e >= num_experts) continue;
      const float wt = t->w[static_cast<size_t>(tok) * top_k + k];
      const bf16_t* g_row = g_scratch.data() + ((size_t)tok * top_k + k) * I;
      const uint8_t* blk_e = dn_packed_l + (size_t)e * Ih * H;
      const uint8_t* scl_e = dn_scale_l + (size_t)e * (size_t)(I / 32) * H;
      float part[HBLK];
      mxgemv(part, blk_e + h0, scl_e + h0, g_row, Ih, H, nh, e2m1_lut, e8m0_lut);
      const bf16_t* bias_e = dn_bias_l + (size_t)e * H + h0;
      for (int c = 0; c < nh; ++c) acc[c] += (part[c] + bf16_to_f32(bias_e[c])) * wt;
    }
    bf16_t* y_row = t->y + (size_t)tok * H;
    for (int c = 0; c < nh; ++c) y_row[h0 + c] = f32_to_bf16(acc[c]);
  }

  // ----------------------------- ds_fp4 (DSV4) -------------------------------
  // Row-major e2m1 + e8m0/32 (no global); silu-swiglu with clamp; FP8-roundtripped
  // activations (x once in submit -> xq_scratch; the intermediate g in a dedicated
  // round-trip phase between the two passes). Router weight applies on the down output.

  void do_pass1_dsfp4(const MoeTask* t, int64_t p) {
    const int64_t ib = p % n_iblk;
    const int64_t tk = p / n_iblk;
    const int k = static_cast<int>(tk % top_k);
    const int tok = static_cast<int>(tk / top_k);
    const int e = t->ids[static_cast<size_t>(tok) * top_k + k];
    if (e < 0 || e >= num_experts) return;
    // Resolve this task's layer base once; row indexing below is layer-local (e).
    const uint8_t* gu_packed_l = reinterpret_cast<const uint8_t*>(tbl_at(gate_up_tbl, t->layer_id));
    const uint8_t* gu_scale_l = reinterpret_cast<const uint8_t*>(tbl_at(gu_scale_tbl, t->layer_id));
    const int N2 = 2 * I, Hh = H / 2, Hs = H / 32;
    // fp8-roundtripped input, pre-deinterleaved to fp32 even/odd halves.
    const float* xe = xe_scratch.data() + (size_t)tok * (H / 2);
    const float* xo = xo_scratch.data() + (size_t)tok * (H / 2);
    const uint8_t* gp = gu_packed_l + (size_t)e * N2 * Hh;
    const uint8_t* gs = gu_scale_l + (size_t)e * N2 * Hs;
    bf16_t* g_row = g_scratch.data() + ((size_t)tok * top_k + k) * I;
    const int i0 = static_cast<int>(ib) * IBLK;
    const int i1 = std::min(I, i0 + IBLK);
    const float lim = swiglu_limit;
    for (int i = i0; i < i1; ++i) {
      // gate_up is stored bf16 by the reference GEMV before swiglu; round to match.
      float gate = bf16_to_f32(f32_to_bf16(
          dsdot(gp + (size_t)i * Hh, gs + (size_t)i * Hs, xe, xo, H, e2m1_lut, e8m0_lut)));
      float up = bf16_to_f32(f32_to_bf16(dsdot(
          gp + (size_t)(I + i) * Hh, gs + (size_t)(I + i) * Hs, xe, xo, H, e2m1_lut, e8m0_lut)));
      if (lim > 0.0f) {
        if (gate > lim) gate = lim;
        if (up > lim) up = lim;
        else if (up < -lim) up = -lim;
      }
      const float glu = gate / (1.0f + std::exp(-gate));  // silu(gate)
      g_row[i] = f32_to_bf16(glu * up);
    }
  }

  // Prepare one intermediate row (token,route) for the down GEMV: ds_fp4 first FP8
  // round-trips it (DSV4 act_quant), then both formats deinterleave to fp32 even/odd
  // (reused across every down output row).
  void prep_g_row(int64_t r) {
    if (gguf_i8) {  // W4A8-K: q8_K-quantize the intermediate row for the down GEMV.
      quant_q8_k(g_scratch.data() + (size_t)r * I, I,
                 gq8k_scratch.data() + (size_t)r * (I / QK_K));
      return;
    }
    bf16_t* g = g_scratch.data() + (size_t)r * I;
    if (use_q4a8) {  // q4_0 W4A8: Q8_0-quantize the intermediate row for the down GEMV.
      quant_q8_0(g, I, gi8_scratch.data() + (size_t)r * I,
                 gas_scratch.data() + (size_t)r * (I / 32));
      return;
    }
    if (fmt == WF_DSFP4) fp8_roundtrip_bf16(g, g, I);
    float* ge = ge_scratch.data() + (size_t)r * (I / 2);
    float* go = go_scratch.data() + (size_t)r * (I / 2);
    deinterleave_bf16_f32(g, ge, go, I);
    if (use_vnni)
      quant_i8_pg16(ge, go, I, gi8_scratch.data() + (size_t)r * I,
                    gas_scratch.data() + (size_t)r * (I / 16));
  }

  void do_pass2_dsfp4(const MoeTask* t, int64_t p) {
    const int64_t hb = p % n_hblk;
    const int tok = static_cast<int>(p / n_hblk);
    const int Ih = I / 2, Is = I / 32;
    const int h0 = static_cast<int>(hb) * HBLK;
    const int h1 = std::min(H, h0 + HBLK);
    // Resolve this task's layer base once; row indexing below is layer-local (e).
    const uint8_t* dn_packed_l = reinterpret_cast<const uint8_t*>(tbl_at(down_tbl, t->layer_id));
    const uint8_t* dn_scale_l = reinterpret_cast<const uint8_t*>(tbl_at(dn_scale_tbl, t->layer_id));
    bf16_t* y_row = t->y + (size_t)tok * H;
    for (int h = h0; h < h1; ++h) {
      float acc = 0.0f;
      for (int k = 0; k < top_k; ++k) {
        const int e = t->ids[static_cast<size_t>(tok) * top_k + k];
        if (e < 0 || e >= num_experts) continue;
        const float wt = t->w[static_cast<size_t>(tok) * top_k + k];
        const float* ge = ge_scratch.data() + ((size_t)tok * top_k + k) * (I / 2);
        const float* go = go_scratch.data() + ((size_t)tok * top_k + k) * (I / 2);
        const uint8_t* dp = dn_packed_l + (size_t)e * (size_t)H * Ih + (size_t)h * Ih;
        const uint8_t* ds = dn_scale_l + (size_t)e * (size_t)H * Is + (size_t)h * Is;
        // The reference rounds each route's weighted down output to bf16 before the
        // fp32 sum over routes (down [T, top_k, H] bf16 -> .sum(dim=1)).
        acc += bf16_to_f32(f32_to_bf16(dsdot(dp, ds, ge, go, I, e2m1_lut, e8m0_lut) * wt));
      }
      y_row[h] = f32_to_bf16(acc);
    }
  }

  void run_task_body(const MoeTask* t) {
    int local_sense = 0;
    for (;;) {
      int64_t p = p1_next.fetch_add(1, std::memory_order_relaxed);
      if (p >= p1_total) break;
      do_pass1(t, p);
    }
    barrier(local_sense);
    // Row-major fp4: prepare the intermediate rows (per token,route) before the down
    // GEMV -- ds_fp4 FP8 round-trips (DSV4 act_quant), both deinterleave to fp32;
    // q4_0 W4A8 Q8_0-quantizes; gguf W4A8-K q8_K-quantizes. Needs all of pass1 done
    // (a full row spans every iblk).
    if (needs_di || use_q4a8 || gguf_i8) {
      for (;;) {
        int64_t r = prt_next.fetch_add(1, std::memory_order_relaxed);
        if (r >= prt_total) break;
        prep_g_row(r);
      }
      barrier(local_sense);
    }
    for (;;) {
      int64_t p = p2_next.fetch_add(1, std::memory_order_relaxed);
      if (p >= p2_total) break;
      do_pass2(t, p);
    }
  }

  void worker_loop(int tid) {
    pin_self(tid);
    uint64_t my_gen = 0;
    for (;;) {
      MoeTask* t;
      {
        std::unique_lock<std::mutex> lk(task_mtx);
        task_cv.wait(lk, [&] { return stop || cur_gen != my_gen; });
        if (stop) return;
        my_gen = cur_gen;
        t = cur_task;
      }
      run_task_body(t);
      if (done_count.fetch_add(1) + 1 == num_threads) {
        completed.store(my_gen, std::memory_order_release);
        {
          std::lock_guard<std::mutex> lk(sync_mtx);
        }
        sync_cv.notify_all();
      }
    }
  }

  void submit(MoeTask* t) {
    n_iblk = (I + IBLK - 1) / IBLK;
    n_hblk = (H + HBLK - 1) / HBLK;
    // Grow the per-token intermediate scratch if a larger batch shows up than the
    // construction-time hint (CUDA-graph capture warms the largest bs first, so
    // this happens at most once, before any capture, while the pool is idle).
    const size_t need = static_cast<size_t>(t->num_tokens) * top_k * I;
    if (need > g_scratch.size()) g_scratch.resize(need);
    p1_total = static_cast<int64_t>(t->num_tokens) * top_k * n_iblk;
    p2_total = static_cast<int64_t>(t->num_tokens) * n_hblk;
    prt_total = (needs_di || use_q4a8 || gguf_i8) ? static_cast<int64_t>(t->num_tokens) * top_k : 0;
    p1_next.store(0, std::memory_order_relaxed);
    p2_next.store(0, std::memory_order_relaxed);
    prt_next.store(0, std::memory_order_relaxed);
    done_count.store(0, std::memory_order_relaxed);
    bar_count.store(0, std::memory_order_relaxed);
    bar_sense.store(0, std::memory_order_relaxed);
    // ds_fp4: FP8 round-trip the per-token input once, up front (single-threaded;
    // tiny for decode, and done before the workers are woken below).
    if (needs_di) {
      const size_t xn = static_cast<size_t>(t->num_tokens) * H;
      if (xn / 2 > xe_scratch.size()) {
        xe_scratch.resize(xn / 2);
        xo_scratch.resize(xn / 2);
        ge_scratch.resize(static_cast<size_t>(t->num_tokens) * top_k * (I / 2));
        go_scratch.resize(static_cast<size_t>(t->num_tokens) * top_k * (I / 2));
        if (use_vnni) {
          xi8_scratch.resize(xn);
          xas_scratch.resize(xn / 16);
          gi8_scratch.resize(static_cast<size_t>(t->num_tokens) * top_k * I);
          gas_scratch.resize(static_cast<size_t>(t->num_tokens) * top_k * (I / 16));
        }
      }
      const bool ds = (fmt == WF_DSFP4) && !input_prequant;
      if (ds && xn > xq_scratch.size()) xq_scratch.resize(xn);
      for (int tok = 0; tok < t->num_tokens; ++tok) {
        const bf16_t* src = t->x + (size_t)tok * H;
        if (ds) {  // DSV4 FP8-round-trips the input before the gate_up GEMV
          bf16_t* xq = xq_scratch.data() + (size_t)tok * H;
          fp8_roundtrip_bf16(src, xq, H);
          src = xq;
        }
        float* xe = xe_scratch.data() + (size_t)tok * (H / 2);
        float* xo = xo_scratch.data() + (size_t)tok * (H / 2);
        deinterleave_bf16_f32(src, xe, xo, H);
        if (use_vnni)
          quant_i8_pg16(xe, xo, H, xi8_scratch.data() + (size_t)tok * H,
                        xas_scratch.data() + (size_t)tok * (H / 16));
      }
    }
    // q4_0 W4A8: Q8_0-quantize the per-token input once (single-threaded, tiny for decode).
    if (use_q4a8) {
      const size_t xn = static_cast<size_t>(t->num_tokens) * H;
      if (xn > xi8_scratch.size()) {
        xi8_scratch.resize(xn);
        xas_scratch.resize(xn / 32);
        gi8_scratch.resize(static_cast<size_t>(t->num_tokens) * top_k * I);
        gas_scratch.resize(static_cast<size_t>(t->num_tokens) * top_k * (I / 32));
      }
      for (int tok = 0; tok < t->num_tokens; ++tok)
        quant_q8_0(t->x + (size_t)tok * H, H, xi8_scratch.data() + (size_t)tok * H,
                   xas_scratch.data() + (size_t)tok * (H / 32));
    }
    // GGUF W4A8-K: q8_K-quantize each token's input row once (single-threaded, tiny
    // for decode) -- every gate/up dot of this task reuses it. The intermediate rows
    // quantize in the threaded prepare phase (prep_g_row) after pass 1.
    if (gguf_i8) {
      const size_t nb_in = static_cast<size_t>(t->num_tokens) * (H / QK_K);
      const size_t nb_int = static_cast<size_t>(t->num_tokens) * top_k * (I / QK_K);
      if (nb_in > xq8k_scratch.size()) xq8k_scratch.resize(nb_in);
      if (nb_int > gq8k_scratch.size()) gq8k_scratch.resize(nb_int);
      for (int tok = 0; tok < t->num_tokens; ++tok)
        quant_q8_k(t->x + (size_t)tok * H, H, xq8k_scratch.data() + (size_t)tok * (H / QK_K));
    }
    {
      std::lock_guard<std::mutex> lk(task_mtx);
      cur_task = t;
      ++cur_gen;
      submitted.store(cur_gen, std::memory_order_release);
    }
    task_cv.notify_all();
  }

  void sync() {
    const uint64_t target = submitted.load(std::memory_order_acquire);
    std::unique_lock<std::mutex> lk(sync_mtx);
    sync_cv.wait(lk, [&] { return completed.load(std::memory_order_acquire) >= target; });
  }

  void submit_with_cuda_stream(uintptr_t stream, uintptr_t task) {
    cudaLaunchHostFunc(reinterpret_cast<cudaStream_t>(stream), &CpuMoeExecutor::submit_cb,
                       reinterpret_cast<void*>(task));
  }

  void sync_with_cuda_stream(uintptr_t stream, uintptr_t task) {
    cudaLaunchHostFunc(reinterpret_cast<cudaStream_t>(stream), &CpuMoeExecutor::sync_cb,
                       reinterpret_cast<void*>(task));
  }

  // Register a (layer, batch-size) slot's task so the coordinator can dispatch it on a
  // flag bump.
  void register_flag_task(int slot, uintptr_t task) {
    std::lock_guard<std::mutex> lk(flag_task_mtx);
    if (static_cast<int>(flag_task.size()) <= slot) flag_task.resize(slot + 1, nullptr);
    flag_task[slot] = reinterpret_cast<MoeTask*>(task);
  }

  int64_t flag_served_count(int slot) const {
    return (slot >= 0 && slot < static_cast<int>(flag_served.size())) ? flag_served[slot] : 0;
  }

  // Start the busy-poll coordinator over the mapped-pinned flag arrays. ``pin_core`` >= 0
  // pins the coordinator to that logical CPU (the worker auto-sizing reserves it), so its
  // polling never migrates onto / contends with a GEMV worker's core.
  void start_flag_coordinator(uintptr_t ready_ptr, uintptr_t done_ptr, int num_slots,
                              int pin_core) {
    ready_flags = reinterpret_cast<volatile int64_t*>(ready_ptr);
    done_flags = reinterpret_cast<volatile int64_t*>(done_ptr);
    coord_num_slots = num_slots;
    {
      std::lock_guard<std::mutex> lk(flag_task_mtx);
      if (static_cast<int>(flag_task.size()) < num_slots) flag_task.resize(num_slots, nullptr);
    }
    flag_served.assign(num_slots, 0);
    coord_stop.store(false);
    coord_thread = std::thread([this, pin_core] {
#if CPU_MOE_HAS_AFFINITY
      if (pin_core >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(pin_core, &set);
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
      }
#endif
      coordinator_loop();
    });
  }

  void coordinator_loop() {
    // Idle backoff (edge-device friendliness): spin hot only while decode traffic is
    // flowing, with a TIME-based hot window (an iteration count varies 2-6x with slot
    // count and pause cost, so some configs dozed every token). 50 ms since the last
    // flag comfortably covers intra- and inter-token gaps -- including host-func-only
    // stretches (prefill bursts, --moe-cpu-layers subsets) -- so steady decode never
    // sleeps and keeps the ~us-level wakeup. Past it, i.e. the engine is actually idle,
    // escalate to timed sleeps capped at 2 ms: a dozing coordinator costs <0.1% of a
    // core instead of 100%, and the only price is a <=2 ms discovery delay on the FIRST
    // MoE layer after an idle period (irrelevant next to prefill). The clock is sampled
    // every 1024 empty polls (~0.1-1 ms) to keep the hot loop cheap.
    using coord_clock = std::chrono::steady_clock;
    constexpr auto kHotWindow = std::chrono::milliseconds(50);
    constexpr int64_t kSleepCapUs = 2000;
    auto last_active = coord_clock::now();
    unsigned empty_polls = 0;  // unsigned: the hot-phase ++ must not overflow into UB
    int64_t sleep_us = 100;
    bool dozing = false;
    while (!coord_stop.load(std::memory_order_relaxed)) {
      bool any = false;
      for (int L = 0; L < coord_num_slots; ++L) {
        // Binary handshake (memop-compatible: the GPU-side WAIT compares against an
        // immediate baked at graph capture, so the protocol resets per step instead of
        // counting). Acquire: everything the GPU made visible before setting ready --
        // the D2H input copies -- is visible to the worker pool after this read.
        if (flag_load_acquire(&ready_flags[L]) != 0) {
          flag_store_release(&ready_flags[L], 0);  // consume this step's doorbell
          MoeTask* t;
          {
            std::lock_guard<std::mutex> lk(flag_task_mtx);
            t = (L < static_cast<int>(flag_task.size())) ? flag_task[L] : nullptr;
          }
          if (t != nullptr) {
            submit(t);
            sync();
          }
          // Release: the workers' y stores are visible before the GPU sees done.
          flag_store_release(&done_flags[L], 1);
          if (L < static_cast<int>(flag_served.size())) ++flag_served[L];
          any = true;
        }
      }
      if (any) {
        last_active = coord_clock::now();
        empty_polls = 0;
        sleep_us = 100;
        dozing = false;
        continue;
      }
      // Hot phase: pause-spin, consulting the clock only every 1024 empty polls (the
      // amortization must gate the CLOCK, not the sleep -- gating the sleep decision
      // let 1023/1024 idle iterations hot-scan all slots, ~20% of a core at 992 slots).
      // Doze phase: sleep on EVERY iteration until traffic returns; one wake-up scan
      // (~0.5 us) per 2 ms sleep is ~0.03% duty.
      if (!dozing) {
        if ((++empty_polls & 1023u) != 0 ||
            coord_clock::now() - last_active < kHotWindow) {
#if CPU_MOE_X86
          _mm_pause();
#endif
          continue;
        }
        dozing = true;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
      sleep_us = std::min<int64_t>(sleep_us * 2, kSleepCapUs);
    }
    // Teardown: release any in-flight (or future) spin-wait immediately so a replay
    // caught mid-shutdown exits its sync kernel now instead of owning the watchdog
    // stall. Runs before the destructor's join() returns, while the flag arrays are
    // still alive on the Python side.
    for (int L = 0; L < coord_num_slots; ++L) {
      flag_store_release(&done_flags[L], INT64_MAX);
    }
  }

  // Eager (non-graph) path: run one task to completion on the pool.
  void run_task(uintptr_t task) {
    MoeTask* t = reinterpret_cast<MoeTask*>(task);
    submit(t);
    sync();
  }

  static void CUDART_CB submit_cb(void* ud) {
    MoeTask* t = reinterpret_cast<MoeTask*>(ud);
    t->exec->submit(t);
  }
  static void CUDART_CB sync_cb(void* ud) {
    MoeTask* t = reinterpret_cast<MoeTask*>(ud);
    t->exec->sync();
  }
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  namespace py = pybind11;
  py::class_<CpuMoeExecutor>(m, "CpuMoeExecutor")
      .def(py::init<int, int, int, int, int, int, int, int, int, int, uintptr_t, uintptr_t,
                    uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t,
                    double, double, std::vector<int>, int, int, uintptr_t, uintptr_t>(),
           py::arg("num_threads"), py::arg("num_layers"), py::arg("num_experts"),
           py::arg("top_k"), py::arg("hidden_size"), py::arg("inter_size"),
           py::arg("max_tokens"), py::arg("activation_id"),
           py::arg("apply_router_weight_on_input"), py::arg("weight_format"),
           py::arg("gate_up_ptr"), py::arg("down_ptr"), py::arg("gate_up_scale_ptr"),
           py::arg("gate_up_global_ptr"), py::arg("down_scale_ptr"),
           py::arg("down_global_ptr"), py::arg("gate_up_bias_ptr"),
           py::arg("down_bias_ptr"), py::arg("swiglu_alpha"), py::arg("swiglu_limit"),
           py::arg("core_ids"), py::arg("up_fmt") = -1, py::arg("down_fmt") = -1,
           py::arg("gate_ptr") = 0, py::arg("up_ptr") = 0)
      .def("create_task", &CpuMoeExecutor::create_task, py::arg("layer_id"),
           py::arg("num_tokens"), py::arg("x_ptr"), py::arg("ids_ptr"), py::arg("w_ptr"),
           py::arg("y_ptr"))
      .def("submit_with_cuda_stream", &CpuMoeExecutor::submit_with_cuda_stream,
           py::arg("stream"), py::arg("task"), py::call_guard<py::gil_scoped_release>())
      .def("sync_with_cuda_stream", &CpuMoeExecutor::sync_with_cuda_stream,
           py::arg("stream"), py::arg("task"), py::call_guard<py::gil_scoped_release>())
      .def("run_task", &CpuMoeExecutor::run_task, py::arg("task"),
           py::call_guard<py::gil_scoped_release>())
      .def("register_flag_task", &CpuMoeExecutor::register_flag_task,
           py::arg("slot"), py::arg("task"))
      .def("flag_served_count", &CpuMoeExecutor::flag_served_count, py::arg("slot"))
      .def("start_flag_coordinator", &CpuMoeExecutor::start_flag_coordinator,
           py::arg("ready_ptr"), py::arg("done_ptr"), py::arg("num_slots"),
           py::arg("pin_core"))
      .def("set_input_prequant",
           [](CpuMoeExecutor& e, bool v) { e.input_prequant = v; },
           py::arg("value"))
      .def("isa_name", &CpuMoeExecutor::isa_name);
  m.def("memops_probe", &cumemops_probe, py::arg("stream"), py::arg("scratch_addr"));
  m.def("memop_submit", &cumemop_submit, py::arg("stream"), py::arg("done_addr"),
        py::arg("ready_addr"), py::arg("slot"));
  m.def("memop_sync", &cumemop_sync, py::arg("stream"), py::arg("done_addr"),
        py::arg("slot"));
  // ABI capability marker: the highest ActKind this build implements in the
  // GENERIC epilogue. CpuMoeExecutor.__init__ probes it before requesting an act
  // id the epilogue must handle -- a prebuilt .so from before ACT_SWIGLUOAI
  // accepts id 3 without error and silently computes the wrong activation
  // (act_apply falls through to gelu_tanh); the probe turns a stale extension
  // into a loud rebuild instruction instead of wrong model outputs.
  m.def("max_generic_act_id", []() { return static_cast<int>(ACT_SWIGLU_CLAMP); });
}
