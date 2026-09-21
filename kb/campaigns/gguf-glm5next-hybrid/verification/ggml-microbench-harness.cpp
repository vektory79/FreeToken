// Standalone microbenchmark of ggml's CPU integer dot kernels (AVX2 tier).
//
// Part of the FreeToken gguf-glm5next-hybrid decision gate: measure sustained
// GB/s of the exact decode GEMV shape and compare against the >= 40 GB/s gate.
//
// Kernels are compiled VERBATIM (zero transcription) from the llama.cpp
// checkout /media/ai/src/llama-cpp-glm5next (MIT, "Copyright (c) 2023-2026 The
// ggml authors"):
//   AVX2 dots:  ggml/src/ggml-cpu/arch/x86/quants.c
//               ggml_vec_dot_q6_K_q8_K    :2426 (AVX2 body :2439)
//               ggml_vec_dot_iq3_xxs_q8_K :3260 (AVX2 body :3274)
//               ggml_vec_dot_iq4_xs_q8_K  :4004 (AVX2 body :4022)
//   quantizer:  ggml/src/ggml-quants.c:2768 quantize_row_q8_K_ref
//   scalar ref: ggml/src/ggml-cpu/quants.c:851/1050/1283 *_generic (parity check)
//
// ISA tier: compiled with -mf16c -mfma -mavx -mavx2 (the "haswell" variant
// flags). The user's llama.cpp build uses GGML_CPU_ALL_VARIANTS=ON +
// GGML_BACKEND_DL=ON, shipping libggml-cpu-haswell.so (-mf16c -mfma -mavx
// -mavx2) and libggml-cpu-alderlake.so (same + -mavxvnni); the runtime picks
// the best supported variant. On this i7-14700KF (Raptor Lake: AVX2, no
// AVX-512) both AVX2 variants run identical code for these three kernels:
// they use mul_add_epi8 (maddubs/madd chain, quants.c:69), not the
// VNNI-dispatched mul_sum_us8_pairs_float (quants.c:105).
//
// Shape (from the real GLM-5.3-Flash-UD-Q3_K_XL.gguf geometry):
//   H=4096, I=2048, E=288 experts, top_k=8, 42 MoE layers.
//   Per layer: gate (I rows x K=H) + up (I rows x K=H) + down (H rows x K=I),
//   quantize_row_q8_K once for gate+up (x, H elems), once per expert for down
//   (silu-mul output, I elems x 8 experts).
//   Signature mix = 39x(IQ3_XXS,IQ3_XXS,IQ4_XS) + 2x(IQ3_XXS,IQ3_XXS,Q6_K)
//                 + 1x(IQ4_XS,IQ4_XS,Q6_K)   [gguf type ids 18/23/14 in the
//                 FreeToken per-projection tables; census matches the study]
// Weight bytes/token = 3,733,454,848 B (8 experts x 42 layers); matches
// llama.cpp all-CPU decode streaming.

#include <cstdio>
#include <string>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <omp.h>
#include <x86intrin.h>
#include <pthread.h>
#include <sched.h>

#define GGML_COMMON_DECL_CPP 1
#include "ggml-common.h"   // block_q8_K / block_iq3_xxs / block_iq4_xs / block_q6_K + size static_asserts

typedef void (*vecdot_fn_t)(int n, float* s, size_t bs, const void* vx, size_t bx,
                            const void* vy, size_t by, int nrc);

extern "C" {
void ggml_vec_dot_q6_K_q8_K   (int, float*, size_t, const void*, size_t, const void*, size_t, int);
void ggml_vec_dot_iq3_xxs_q8_K(int, float*, size_t, const void*, size_t, const void*, size_t, int);
void ggml_vec_dot_iq4_xs_q8_K (int, float*, size_t, const void*, size_t, const void*, size_t, int);
void ggml_vec_dot_q6_K_q8_K_generic   (int, float*, size_t, const void*, size_t, const void*, size_t, int);
void ggml_vec_dot_iq3_xxs_q8_K_generic(int, float*, size_t, const void*, size_t, const void*, size_t, int);
void ggml_vec_dot_iq4_xs_q8_K_generic (int, float*, size_t, const void*, size_t, const void*, size_t, int);
void quantize_row_q8_K_ref(const float* x, block_q8_K* y, int64_t k);
// stubs for ggml.c helpers referenced ONLY by functions this bench never calls
// (quantize_q6_K, ggml_validate_row_data); added so the verbatim ggml sources
// link without dragging in all of ggml.c
float ggml_table_f32_f16[65536];  // ggml.c owns this normally; exact f16->f32 init
// tables owned by ggml.c; referenced only by mxfp4/nvfp4 kernels never called here
float ggml_table_f32_e8m0_half[256];
float ggml_table_f32_ue4m3[16];
struct TableInit { TableInit() { for (int i = 0; i < 65536; ++i) ggml_table_f32_f16[i] = _cvtsh_ss((unsigned short)i); } };
static TableInit _table_init;
size_t ggml_type_size(int) { return 0; }
size_t ggml_row_size(int, int64_t) { return 0; }
const char* ggml_type_name(int) { return "stub"; }
// normally defined in ggml.c; GGML_ASSERT never fires in this bench
void ggml_abort(const char* file, int line, const char* fmt, ...) {
    (void)file; (void)line; (void)fmt; abort();
}
}

// ---- geometry ----
static constexpr int H = 4096, I = 2048, E = 288, TOPK = 8, NL = 42;
enum Fmt { F_IQ3XXS = 0, F_IQ4XS = 1, F_Q6K = 2 };
struct LayerType { Fmt gate, up, down; };
// signature mix: 39x(18,18,23) + 2x(18,18,14) + 1x(23,23,14)
static LayerType MIX[NL];
static void init_mix() {
    for (int l = 0; l < NL; ++l) {
        if (l < 39)      MIX[l] = {F_IQ3XXS, F_IQ3XXS, F_IQ4XS};
        else if (l < 41) MIX[l] = {F_IQ3XXS, F_IQ3XXS, F_Q6K};
        else             MIX[l] = {F_IQ4XS,  F_IQ4XS,  F_Q6K};
    }
}
static inline vecdot_fn_t fn_of(Fmt f) {
    switch (f) {
        case F_IQ3XXS: return ggml_vec_dot_iq3_xxs_q8_K;
        case F_IQ4XS:  return ggml_vec_dot_iq4_xs_q8_K;
        default:       return ggml_vec_dot_q6_K_q8_K;
    }
}
static inline vecdot_fn_t gen_of(Fmt f) {
    switch (f) {
        case F_IQ3XXS: return ggml_vec_dot_iq3_xxs_q8_K_generic;
        case F_IQ4XS:  return ggml_vec_dot_iq4_xs_q8_K_generic;
        default:       return ggml_vec_dot_q6_K_q8_K_generic;
    }
}
// row bytes: gate/up row = K=H -> 16 super-blocks; down row = K=I -> 8 super-blocks
static inline size_t rb_gate(Fmt f) { return f == F_IQ3XXS ? (size_t)16*98 : (size_t)16*136; }
static inline size_t rb_down(Fmt f) { return f == F_IQ4XS  ? (size_t) 8*136 : (size_t) 8*210; }

// ---- parallel sections (OpenMP; ggml's own CPU backend is OpenMP-parallel
// with a join per op, so a parallel-for per projection mirrors the real
// schedule shape; schedule(static) splits contiguous row ranges like ggml's
// ir0 chunking) ----
static void parallel_run(void (*fn)(int64_t, void*), void* ctx, int64_t n) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; ++i) fn(i, ctx);
}

// ---- job bodies ----
static thread_local float t_sink;

struct DotCtx {
    const uint8_t*  buf;
    size_t          rb;
    int64_t         rows;      // rows per expert
    int             K;
    const uint8_t*  q8;        // q8_K rows
    size_t          q8_stride; // bytes between per-expert q8 rows (0 = shared)
    vecdot_fn_t     fn;
};
static void dot_job(int64_t item, void* p) {
    DotCtx* c = (DotCtx*)p;
    int64_t e = item / c->rows, r = item % c->rows;
    const uint8_t* row = c->buf + (size_t)(e * c->rows + r) * c->rb;
    const void* q8 = c->q8 + (size_t)e * c->q8_stride;
    c->fn(c->K, &t_sink, sizeof(float), row, c->rb, q8, sizeof(block_q8_K), 1);
}

struct QuantCtx { const float* x; block_q8_K* y; };
static void quant_job(int64_t item, void* p) {
    QuantCtx* c = (QuantCtx*)p;
    quantize_row_q8_K_ref(c->x + item * 256, c->y + item, 256);
}

// ---- expert buffers ----
static uint8_t* buf_gate_iq3;  // E x I rows, rb = 16*98
static uint8_t* buf_up_iq3;
static uint8_t* buf_gate_iq4;  // E x I rows, rb = 16*136 (signature-mix layer 41)
static uint8_t* buf_up_iq4;
static uint8_t* buf_down_iq4;  // E x H rows, rb = 8*136
static uint8_t* buf_down_q6;   // E x H rows, rb = 8*210

static std::vector<float> xrow;       // H floats  (feeds gate+up)
static std::vector<float> downact;    // TOPK*I floats (silu-mul outputs)
static std::vector<block_q8_K> q8x;   // 16 blocks
static std::vector<block_q8_K> q8d;   // TOPK*8 blocks

static uint64_t sm_state;
static inline uint64_t splitmix64() {
    sm_state += 0x9E3779B97F4A7C15ull;
    uint64_t z = sm_state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static void fill_buf(uint8_t* p, size_t n, uint64_t seed) {
    sm_state = seed;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) { uint64_t v = splitmix64(); memcpy(p + i, &v, 8); }
    if (i < n) { uint64_t v = splitmix64(); memcpy(p + i, &v, n - i); }
}
static void fill_mt(uint8_t* p, size_t n, uint64_t seed) {
    const int T = 16;
    size_t chunk = (n + T - 1) / T;
    std::vector<std::thread> th;
    for (int t = 0; t < T; ++t) {
        size_t off = (size_t)t * chunk;
        if (off >= n) break;
        size_t len = off + chunk > n ? n - off : chunk;
        th.emplace_back([=] { fill_buf(p + off, len, seed ^ (0x1234abcdull + 0x9E3779B9ull * (uint64_t)t)); });
    }
    for (auto& x : th) x.join();
}

// ---- one decode pass = one token through all 42 MoE layers ----
// mode: -1 = blended (real signature mix), 0/1/2 = single format everywhere
// (0 iq3_xxs, 1 iq4_xs, 2 q6_K) for the per-format saturation points
static int g_mode = -1;
static LayerType mix_layer(int l) {
    if (g_mode < 0) return MIX[l];
    return LayerType{(Fmt)g_mode, (Fmt)g_mode, (Fmt)g_mode};
}
static void one_pass() {
    for (int l = 0; l < NL; ++l) {
        LayerType lt = mix_layer(l);
        int64_t eslot0 = (int64_t)l * TOPK;   // expert rotation covers all 288 experts
        (void)eslot0;
        QuantCtx qc{xrow.data(), q8x.data()};
        parallel_run(quant_job, &qc, H / 256);
        DotCtx dg{lt.gate == F_IQ3XXS ? buf_gate_iq3 : buf_gate_iq4, rb_gate(lt.gate),
                  I, H, (const uint8_t*)q8x.data(), 0, fn_of(lt.gate)};
        parallel_run(dot_job, &dg, (int64_t)TOPK * I);
        DotCtx du{lt.up == F_IQ3XXS ? buf_up_iq3 : buf_up_iq4, rb_gate(lt.up),
                  I, H, (const uint8_t*)q8x.data(), 0, fn_of(lt.up)};
        parallel_run(dot_job, &du, (int64_t)TOPK * I);
        QuantCtx qd{downact.data(), q8d.data()};
        parallel_run(quant_job, &qd, (int64_t)TOPK * (I / 256));
        DotCtx dd{lt.down == F_IQ4XS ? buf_down_iq4 : buf_down_q6, rb_down(lt.down),
                  H, I, (const uint8_t*)q8d.data(), (size_t)(I / 256) * sizeof(block_q8_K), fn_of(lt.down)};
        parallel_run(dot_job, &dd, (int64_t)TOPK * H);
    }
}

// ---- pinning ----
// P-cores = cpus 0-7 (+ SMT siblings 8-15), E-cores = cpus 16-27.
// Physical-core order: 8 P first siblings, then 12 E cores = 20 physical.
static const int kPhys[20] = {0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23,24,25,26,27};
static void pin_to(int cpu) {
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s);
    pthread_setaffinity_np(pthread_self(), sizeof(s), &s);
}

// ---- sanity: AVX2 kernel vs ggml scalar *_generic on random rows ----
static bool sanity(bool verbose) {
    quantize_row_q8_K_ref(xrow.data(), q8x.data(), H);
    quantize_row_q8_K_ref(downact.data(), q8d.data(), (int64_t)TOPK * I);
    struct Case { Fmt f; const uint8_t* row; int K; size_t rb; const block_q8_K* q8; };
    Case cases[3] = {
        {F_IQ3XXS, buf_gate_iq3, H, rb_gate(F_IQ3XXS), q8x.data()},
        {F_IQ4XS,  buf_down_iq4, I, rb_down(F_IQ4XS),  q8d.data()},
        {F_Q6K,    buf_down_q6,  I, rb_down(F_Q6K),    q8d.data()},
    };
    bool ok = true;
    for (const Case& c : cases) {
        const int nq = 32;
        for (int q = 0; q < nq; ++q) {
            float a = 0.f, b = 0.f;
            const uint8_t* row = c.row + (size_t)q * c.rb;
            fn_of(c.f) (c.K, &a, sizeof(float), row, c.rb, c.q8, sizeof(block_q8_K), 1);
            gen_of(c.f)(c.K, &b, sizeof(float), row, c.rb, c.q8, sizeof(block_q8_K), 1);
            if (!std::isfinite(a) || !std::isfinite(b)) { ok = false; printf("sanity fmt=%d row=%d non-finite: avx=%g gen=%g\n", c.f, q, a, b); break; }
            double rel = std::fabs((double)a - (double)b) / std::max(1e-30, std::fabs((double)b));
            if (rel > 1e-3) { ok = false; printf("sanity fmt=%d row=%d rel_err=%.3e (avx=%.9g gen=%.9g)\n", c.f, q, rel, a, b); break; }
        }
        if (verbose && ok) {
            float a = 0.f;
            fn_of(c.f)(c.K, &a, sizeof(float), c.row, c.rb, c.q8, sizeof(block_q8_K), 1);
            printf("sanity fmt=%d: 32 rows AVX2-vs-scalar max rel err < 1e-3 OK (row0=%.6g)\n", c.f, a);
        }
        if (!ok) break;
    }
    return ok;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int nthreads = argc > 1 ? atoi(argv[1]) : 8;
    bool pin = false;
    int passes = -1;
    int fmt_mode = -1;
    bool verbose = true;
    for (int i = 2; i < argc; ++i) {
        if (!strcmp(argv[i], "--pin")) pin = true;
        else if (!strcmp(argv[i], "--no-sanity")) verbose = false;
        else if (!strcmp(argv[i], "--quick")) passes = 2;
        else if (!strcmp(argv[i], "--fmt=blended")) fmt_mode = -1;
        else if (!strcmp(argv[i], "--fmt=iq3_xxs")) fmt_mode = 0;
        else if (!strcmp(argv[i], "--fmt=iq4_xs")) fmt_mode = 1;
        else if (!strcmp(argv[i], "--fmt=q6_k")) fmt_mode = 2;
        else if (!strncmp(argv[i], "--passes=", 9)) passes = atoi(argv[i] + 9);
    }
    g_mode = fmt_mode;
    init_mix();

    size_t sz_gi3 = (size_t)E * I * rb_gate(F_IQ3XXS);
    size_t sz_gi4 = (size_t)E * I * rb_gate(F_IQ4XS);
    size_t sz_di4 = (size_t)E * H * rb_down(F_IQ4XS);
    size_t sz_dq6 = (size_t)E * H * rb_down(F_Q6K);
    buf_gate_iq3 = (uint8_t*)malloc(sz_gi3);
    buf_up_iq3   = (uint8_t*)malloc(sz_gi3);
    buf_gate_iq4 = (uint8_t*)malloc(sz_gi4);
    buf_up_iq4   = (uint8_t*)malloc(sz_gi4);
    buf_down_iq4 = (uint8_t*)malloc(sz_di4);
    buf_down_q6  = (uint8_t*)malloc(sz_dq6);
    if (!buf_gate_iq3 || !buf_up_iq3 || !buf_gate_iq4 || !buf_up_iq4 || !buf_down_iq4 || !buf_down_q6) {
        printf("alloc failed\n"); return 1;
    }
    fill_mt(buf_gate_iq3, sz_gi3, 0x1000);
    fill_mt(buf_up_iq3,   sz_gi3, 0x2000);
    fill_mt(buf_gate_iq4, sz_gi4, 0x3000);
    fill_mt(buf_up_iq4,   sz_gi4, 0x4000);
    fill_mt(buf_down_iq4, sz_di4, 0x5000);
    fill_mt(buf_down_q6,  sz_dq6, 0x6000);

    // Constrain the per-block f16 d fields to finite normals: pure random
    // bytes make d NaN/Inf ~5% of the time and the dot outputs go non-finite.
    // qs/qh/scales payload bits stay fully random; the integer kernels are
    // data-independent in timing, this only keeps the float dot finite.
    {
        uint64_t s = 0xfeed;
        auto rnd_f16 = [&s]() -> uint16_t {
            s = s * 6364136223846793005ull + 1442695040888963407ull;
            uint16_t m = (uint16_t)(s >> 33);
            int exp = 11 + (int)((s >> 40) & 7);            // f16 exp 11..18 -> d ~ [2^-4, 2^4)
            return (uint16_t)((m & 0x83FF) | ((uint16_t)exp << 10));
        };
        for (size_t off = 0; off < sz_gi3; off += 98)  *(uint16_t*)(buf_gate_iq3 + off) = rnd_f16();
        for (size_t off = 0; off < sz_gi3; off += 98)  *(uint16_t*)(buf_up_iq3 + off)   = rnd_f16();
        for (size_t off = 0; off < sz_gi4; off += 136) *(uint16_t*)(buf_gate_iq4 + off) = rnd_f16();
        for (size_t off = 0; off < sz_gi4; off += 136) *(uint16_t*)(buf_up_iq4 + off)   = rnd_f16();
        for (size_t off = 0; off < sz_di4; off += 136) *(uint16_t*)(buf_down_iq4 + off) = rnd_f16();
        for (size_t off = 208; off < sz_dq6; off += 210) *(uint16_t*)(buf_down_q6 + off) = rnd_f16();
    }

    // activations: proper uniform floats in [-1, 1) (never NaN/Inf, like real
    // hidden states; quantizer amax stays typical)
    xrow.resize(H); downact.resize((size_t)TOPK * I);
    sm_state = 0x7000;
    for (float& v : xrow)    v = (float)((double)(int64_t)(splitmix64() & 0xFFFFFF) / 8388608.0 - 1.0);
    for (float& v : downact) v = (float)((double)(int64_t)(splitmix64() & 0xFFFFFF) / 8388608.0 - 1.0);
    q8x.resize(H / 256); q8d.resize((size_t)TOPK * (I / 256));

    omp_set_dynamic(0);
    omp_set_num_threads(nthreads);
    if (pin) {
        // NOTE: OMP_PLACES / OMP_PROC_BIND / OMP_WAIT_POLICY must be set in the
        // environment BEFORE the process starts (libgomp parses them in a
        // pre-main constructor); the harness only binds the master to the
        // first physical core so close-binding walks the place list from there
        pin_to(kPhys[0]);
    }

    bool ok = sanity(verbose);
    if (!ok) { printf("SANITY FAILED\n"); return 2; }

    // warmup (first touch + branch predictors), excluded from timing
    one_pass();

    double t1;
    {
        auto t0 = std::chrono::steady_clock::now();
        one_pass();
        t1 = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
    int P;
    if (passes > 0) P = passes;
    else P = (int)std::ceil(3.0 / std::max(1e-9, t1));
    if (P < 8) P = 8;
    if (P > 600) P = 600;

    uint64_t c0 = __rdtsc();
    auto t0 = std::chrono::steady_clock::now();
    for (int p = 0; p < P; ++p) one_pass();
    auto t1c = std::chrono::steady_clock::now();
    uint64_t c1 = __rdtsc();
    double wall = std::chrono::duration<double>(t1c - t0).count();

    // exact per-pass byte accounting from the mix
    uint64_t wbytes = 0;
    for (int l = 0; l < NL; ++l) {
        LayerType lt = mix_layer(l);
        wbytes += (uint64_t)TOPK * (I * rb_gate(lt.gate) + I * rb_gate(lt.up) + H * rb_down(lt.down));
    }
    uint64_t abytes = (uint64_t)NL * (H / 256 * sizeof(block_q8_K) + (uint64_t)TOPK * (I / 256) * sizeof(block_q8_K));
    double bytes_pass = (double)wbytes + (double)abytes;
    double elems_pass = (double)NL * (double)TOPK * 3.0 * I * H;  // weight elements

    const char* mode = g_mode < 0 ? "blended" : (g_mode == 0 ? "iq3_xxs" : (g_mode == 1 ? "iq4_xs" : "q6_k"));
    printf("threads=%d pin=%d fmt=%s passes=%d wall=%.3fs pass_ms=%.2f bytes_pass=%.0f GBps=%.2f GBps_wonly=%.2f elems_pass=%.4ge9 tsc_cyc_per_elem=%.4f\n",
           nthreads, (int)pin, mode, P, wall, wall / P * 1e3, bytes_pass,
           bytes_pass * P / wall / 1e9,
           (double)wbytes * P / wall / 1e9,
           elems_pass / 1e9,
           (double)(c1 - c0) / (P * elems_pass));
    return 0;
}
