# v3a implementation-surface map (research wave R2, 2026-09-18)

READ-ONLY research. Line anchors verified by R2 against the v2 changeset
(commit 7f8c570); re-verify before editing (they drift).

CRITICAL NAMING NOTE (fixes a wrong premise in the v3a brief): in moe.cuh,
**x = weights** (`vx` = `w`, indexed by `exp_idx*exp_stride`, moe.cuh:62),
**y = activations** (q8_1 tokens). The x-tile (`tile_x_*`) scales with
mmq_y = MOE_Y (32, unchanged); only the **y-tile scales with mmq_x = MOE_X**.
The brief's "tile_x/tile_y scales with the m-tile" is half wrong - see §2.

## 1. MOE_X definition and flow - all touchpoints

Shared body `moe_q` (template, moe.cuh:8-158): mmq_x baked in at
`token_offs[mmq_x/nwarps]` (:48), y-tile `tile_y_qs[mmq_x*32]` /
`tile_y_ds[mmq_x*4]` (:72-73, QI8_1=8, ggml-common.h:63),
`sum[mmq_y/32][mmq_x/nwarps]` (:75), i-loops `for(i=0;i<mmq_x;i+=nwarps)`
(:49, :96), boundary guard `blockIdx.y*mmq_x` (:60).

(a) moe.cuh per type (uniform pattern; CSV-verified anchors):
`#define MOE_X_<T>` ROCm/CUDA pair - Q4_0 :164/:168, Q4_1 :292/:296,
Q5_0 :420/:424, Q5_1 :548/:552, Q8_0 :676/:680, Q2_K :804/:808, Q3_K :932/:936,
Q4_K :1060/:1064, Q5_K :1188/:1192, Q6_K :1316/:1320, IQ3_XXS :1450/:1453,
IQ4_XS :1585/:1588 (+ MOE_Y, NWARPS beside each). Kernel
`moe_<type><scalar_t,need_check>` reads `const int mmq_x = MOE_X_<T>`
(:193 Q4_0 ... :1486 IQ3_XXS, :1620 IQ4_XS) and passes it to moe_q with
`allocate_tiles_<t><mmq_y>`, `load_tiles_<t><mmq_y,nwarps,need_check>`, VDR,
vec_dot. Launcher `ggml_moe_<t>_q8_1_cuda` re-reads mmq_x (:245 ... :1538,
:1672) for grid: `block_num_y = tokens_post_padded / mmq_x`,
`block_dims(32, nwarps, 1)`; need_check = `nrows_x % mmq_y != 0`.
NO `__launch_bounds__` on CUDA (ROCM-only, :173-175). No unroll bound depends
on mmq_x except the j-loop unroll in moe_q (:117-123) - factor = mmq_x/nwarps.

(b) gguf_kernel.cu `ggml_moe_a8` (:360): plain per-type switch (cases
2,3,6,7,8,10,11,12,13,14,18,23), each calling `ggml_moe_<t>_q8_1_cuda`;
`tokens_post_padded` = `sorted_token_ids.sizes()[0]` (buffer size, not npp).
No block-size logic here - adding m-tile variants means a variant-selection
inside/next to this switch.

(c) `ggml_moe_get_block_size` :885-912 returns `MOE_X_<T>` per type -> the
single block_size source feeding python.

(d) python: kernel/gguf.py:113-145 wrappers (no hardcoded 4); layers/moe.py:489
dispatch predicate (`all(get_block_size(t)>0)`), :536-548 `trio(block_size)`
per-size cache + grid cap (:542), :550-551/:569 trio selection - all
auto-adapt via get_block_size. moe/fused.py:47-123 `moe_align_block_size`
(sgl path: buffer = pairs+(E+1)*(bs-1), :131-134). No python hardcode of 4.

(e) No dispatch.h involvement (AT_DISPATCH float only, dispatch.h:24-29).

HIDDEN TOUCHPOINT (the one that corrupts at m>4): moe_q's y-ds fill (:106-116)
has NO i-loop - it fills only ds rows `[0, nwarps)` via
`tile_y_ds[threadIdx.y*4 + kby]` and reads `token_offs[0]` (comment :108-110
states the `mmq_x/nwarps==1` assumption). All four served vec_dots read
`y_ds[j*4 + ...]` with j = threadIdx.y + loop offset (vecdotq.cuh iq3_xxs
:2289-2290, iq4_xs :2145-2147, q6_K :1751-1753, q8_0 :1042-1046) -> at
m=8/16/32, ds rows >= nwarps are uninitialized garbage. Must port the dense
`mul_mat_q` full-coverage ds loop (mmq.cuh:77-88), adapted to
`token_offs[i/nwarps]`/`sorted_token_ids[col_dst_0+ids]`. The qs fill
(:96-103) already generalizes.

## 2. SMEM + occupancy (all static __shared__; no cudaFuncSetAttribute anywhere in tree; 48 KB static limit never approached)

x-tile (mmq_y=32 only, m-invariant): q8_0 4752 B (vecdotq.cuh:986-993),
q6_K 8980 B (:1651-1658; QI6_K=32), IQ3_XXS 4356 B (:2184-2190), IQ4_XS
4884 B (:2058-2065). y-tile = 144*m B (moe.cuh:72-73).
Totals per block: q8_0 5328/5904/7056/9360 B; q6_K 9556/10132/11284/13588 B;
IQ3_XXS 4932/5508/6660/8964 B; IQ4_XS 5460/6036/7188/9492 B at m=4/8/16/32.
All feasible to m=32.
Occupancy at 128 threads (4 warps), sm_120 ~100 KB smem/SM, 1536 threads/SM
-> 12 blocks/SM thread cap: q8_0 12/12/12/10; q6_K 10/10/9/7 (binding
format); IQ3_XXS 12/12/12/11; IQ4_XS 12/12/12/10. q6_K loses ~30% blocks at
m=32 - each block does 8x more work, so net fine; real cap likely registers
(see risks).

## 3. Variant mechanism

Recommend compile-time `MMQ_X` template param on `moe_<type>` +
`ggml_moe_<t>_q8_1_cuda`, explicit instantiations {4,8,16,32}, selected in the
`ggml_moe_a8` switch; **keep `ggml_moe_get_block_size` and the kernel
selection derived from the same C++ constant/env override** - if trio
block_size != kernel mmq_x, bins get consumed cross-expert / tail pairs
dropped silently. No in-repo precedent for m-tile variants (MMQ_X_Q* are
single defines, mmq.cuh:141-...), so mirror the ROCm/CUDA #if + switch
pattern. Grid: m-blocks are blockIdx.y (not z); shape
`(ceil(nrows_x/mmq_y), tokens_post_padded/mmq_x, 1)` needs no change. Cap:
moe.py:542 `sorted_ids.numel()//block_size > _MOE_VEC_MAX_GRID(=65535, :617)`
is block_size-generic (auto-relaxes); no re-derivation needed. JIT cache:
`torch.utils.cpp_extension.load(name="freetoken_gguf_kernels", gguf_kernel.cu)`
(kernel/gguf.py:74-80) keys on source hash - header edit -> one rebuild;
**freetoken-kernel-cache wheel has zero gguf coverage** (TVM-FFI only) -> no
stale-prebuilt hazard; kernel symbol names are internal, python API names
unchanged.

## 4. Tail padding cost

moe_align pads each n_e to a multiple of bs; waste w_e = (bs - n_e mod bs)
mod bs. Model: E[w_e] ~ (bs-1)/2 over the ~1..600 spread -> total wasted rows
~ 288*(bs-1)/2 of 65,024 live pairs: bs=4 ~ 432 (0.7%), 8 ~ 1008 (1.6%),
16 ~ 2304 (3.5%), 32 ~ 4608 (7.1%); n_e=1-2 experts waste bs-n_e (up to
31/32 dead lanes in one CTA at bs=32). Shows as extra grid.y CTAs doing
partial work (pad slots skip y-loads, vec_dot computes on stale tile rows,
discarded at write guard moe.cuh:143-146; sentinel bins return at :57) -
compute waste, zero correctness risk. Down projection reuses the same trio ->
same padding thrice per layer.

## 5. Test surface (extend, don't create)

tests/kernels/test_gguf_quant.py: `test_grouped_moe_parity_vs_reference` :228
(param IQ3_XXS/IQ4_XS; add m-tile dimension),
`test_grouped_moe_expert_bound_288` :265,
`test_grouped_moe_matches_moe_vec` :304 (Q8_0/Q6_K/IQ3_XXS/IQ4_XS, rtol 1e-3),
`test_moe_align_trio_contract` :335 (**bs=4 hardcoded at :343**),
`test_moe_align_tail_contract` :363 (**bs=4 at :367**, triton producer).
tests/models/test_gguf_expert_banks.py:
`test_expert_gemm_gguf_grouped_prefill_matches_decode` :305
(`calls["align"]==1` at :358 - holds while the three roles share one block
size; breaks if per-role sizes diverge),
`test_grouped_prefill_noncontig_activations` :383,
`test_grid_caps_raise_before_kernel_launch` :428 (**re-pin: the 32768x8
ValueError case assumes bs=4 arithmetic; at bs=8 needs >= ~522k pairs, e.g.
65536x8**), `test_moe_vec_chunk_cap` :113 (moe_vec-only, unaffected),
`test_expert_gemm_gguf_dispatch` :242 (decode-side, unaffected).

## 6. Decode isolation - confirmed

Decode/prefill-fallback branch (moe.py:441-468) calls only `ggml_moe_a8_vec`
-> `moe_vec_q` (moe_vec.cuh:10-56): grid `(rows, 1, tokens*top_k)`, reads
`topk_ids[blockIdx.z]` directly; zero references to MOE_X, get_block_size, or
the trio. `_gguf_grouped_supported`/trio/get_block_size are reached only
under `is_prefill` (moe.py:438). Decode kernels recompile once with the
module (same source unit) but are textually unchanged; CUDA-graph capture
happens post-build at runtime - no capture hazard.

## v3a implementation checklist (ordered)

1. moe.cuh: port full-coverage ds fill into `moe_q` (:106-116) generalizing
   token_offs - required for ANY m>4.
2. moe.cuh: template `moe_<type>`/`ggml_moe_<t>_q8_1_cuda` on MMQ_X for types
   8, 14, 18, 23 (+ keep 4 default); sweep-override honored identically by
   `ggml_moe_get_block_size`.
3. gguf_kernel.cu: switch selection in `ggml_moe_a8` (:360-612) +
   `ggml_moe_get_block_size` (:885-912) - one shared source of truth.
4. moe.py: no functional change (trio/cap auto-adapt); update :509 docstring
   "every MOE_X block size is 4".
5. Tests per section 5 (trio bs sweep, cap re-pin, per-format parity at each
   tile, moe_vec regression).
6. A/B sweep 4/8/16/32 per format; exclude last-full-chunk artifact; nsys
   kernel-count probe re-derived (bins shrink with m).

## Top 5 risks

1. ds-fill row garbage at m>4 if the section-1 port is missed - silent
   corruption, passes shape checks.
2. block_size/kernel-tile divergence (two sources of truth) - cross-expert
   pair corruption; enforce single-source.
3. Register pressure at m=32: 8 inlined vec_dot bodies + sum[1][8]; no CUDA
   `__launch_bounds__` -> possible spills; consider adding bounds if
   occupancy drops below the smem estimate.
4. q6_K occupancy 10->7 blocks/SM at m=32 - smallest headroom; sweep may
   saturate there first.
5. Stale test arithmetic: cap test (:428) and trio bs pins
   (test_gguf_quant.py:343/:367) fail confusingly rather than guard; re-pin
   before the sweep so failures are meaningful.
