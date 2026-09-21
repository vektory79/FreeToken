# Replan wave 2 - Candidate A: dense q8_0 m-tile template + FREETOKEN_GGUF_DENSE_MTILE

Branch vektory79 @ 4505572 (= 9f09596 N-split default-flip + a .veai-only Memory
commit; python tree matches the campaign state). Payoff is UNMEASURED: this wave
makes the tile sweep measurable (compile-time variants + knob); the A/B sweep is
a later wave. v3a blueprint in-tree: moe.cuh launchers templated on mmq_x,
ft_gguf_moe_mtile() single source of truth in gguf_kernel.cu.

## T40/T51 pre-audit (written BEFORE the code change)

Scope: shared core `mul_mat_q` (mmq.cuh:19-113) is ALREADY templated on
mmq_x/mmq_y/nwarps - the dense wrapper `mul_mat_q8_0` (mmq.cuh:440) and its
launcher (mmq.cuh:478) just pin them via macros (CUDA 4x32/4warps, ROCm
64x128/8warps). Widening = new template instantiations only. Constants:
WARP_SIZE_GGUF=32, QK8_0=32, QR8_0=1, QI8_0=8, QI8_1=8, VDR_Q8_0_Q8_1_MMQ=8,
mmq_y=32 (unchanged), nwarps=4 CUDA / 8 ROCm.

1. sum[mmq_y/32][mmq_x/nwarps] per-thread accumulator (mmq.cuh:49)
   CUDA -> [1][mmq_x/4] = 1 / 2 / 4 / 8 / 16 floats at tiles 4/8/16/32/64.
   Zero-length hazard needs mmq_x < nwarps - impossible on CUDA (nwarps=4,
   min tile 4). ROCm nwarps=8 WOULD produce [1][0] for tiles < 8 -> ROCm keeps
   its single 64-wide tile (64/8=8) and the templated kernel gains
   static_asserts (mmq_x >= nwarps && mmq_x % nwarps == 0). Tile 64 CUDA:
   16-float accumulator + vec_dot working set (v[8]/u[8] int + 2 float) ->
   register check via ptxas; exclude if local-memory spills appear.
   SUPPORTED: 8/16/32; 64 pending ptxas.

2. y-tile SMEM (mmq.cuh:50-51): tile_y_qs[mmq_x*32] int + tile_y_ds[mmq_x*4]
   half2 -> 0.6/1.1/2.3/4.6/9.2 KB at 4/8/16/32/64. x-tile (q8_0,
   vecdotq.cuh:986): tile_x_qs[32*33] int (4224 B) + tile_x_d[132] float
   (528 B) = 4.75 KB, mmq_x-independent. Block totals 5.3/5.9/7.1/9.3/13.8 KB
   - ALL static, all << 48 KB default static limit; cudaFuncSetAttribute not
   needed. SUPPORTED incl. 64.

3. tile_y_qs fill (mmq.cuh:69-75): i steps nwarps=4, ty 0..3 ->
   (ty+i) enumerates 0..mmq_x-1 exactly once for any mmq_x ≡ 0 (mod 4);
   kqs = ir*32+tx covers 0..31 per ir. Beyond-M columns clamped via
   min(col_y_0 + ..., ncols_y - 1) -> in-bounds loads feeding discarded
   outputs (same mechanism as today's M%4 tail). SUPPORTED 8/16/32/64.

4. ds-fill coverage (mmq.cuh:77-89; the loop v3a ported to moe): stride
   nwarps*QI8_1 = 32; ids = (ids0 + 8*ty + tx/4) % mmq_x, kby = tx % 4.
   t8: 8*ty + tx/4 spans 0..31 -> all residues mod 8 (multi-writers are
   benign: dsi_src depends only on (ids, kby, ir, ib0), so same-value same-
   address writes). t16/t32: same window covers [0,mmq_x) via the modulo.
   t64: two rounds (ids0 in {0,32}) cover 0..63. Full coverage at every tile.
   need_sum=false for q8_0: ds.x converted to float in place (4-byte over
   half2) - footprint unchanged. SUPPORTED (v3a finding re-verified in tree).

5. need_check boundary (launcher mmq.cuh:496-505): keyed on
   nrows_x % mmq_y == 0 = N % 32 (weight rows) - INDEPENDENT of mmq_x,
   unchanged by the knob. The y/M side has no need_check path: clamped loads
   (item 3) + epilogue write guard `col_dst >= ncols_dst -> return`
   (mmq.cuh:97-102, after all __syncthreads -> no divergence hazard). The
   boundary math is untouched by design. SUPPORTED.

6. vec_dot_q8_0_q8_1_mul_mat (vecdotq.cuh:1037): reads
   y_qs[j*32+k] / y_df[j*4 + k/8] with j < mmq_x - y indices scale with the
   template param and stay in-bounds; x-side indices (i < 32, k < 32,
   x_ql[i*33+k], x_dmf[i*4 + i/8 + k/8]) are mmq_y-only. SUPPORTED unchanged.

7. Grid math (launcher): block_num_x = ceil(N/32) unchanged;
   block_num_y = ceil(M/mmq_x) = 2032/1016/508/254/127 at M=8128 - dim3-y
   limit 65535 fine; block dims (32, 4) unchanged. M=8128 = 2^6 * 127 divides
   evenly by 4/8/16/32/64 (no tail at the campaign anchor). SUPPORTED.

8. Dispatch topology: no moe_align trio on the dense path - the knob has no
   cross-site consumer to keep in sync (unlike v3a's block_size trio); the
   only new reader is the test probe ggml_dense_get_mtile(). quant_X staging
   [batch, padded/32*9] is tile-independent. MMVQ (mmvq.cuh), the other dense
   formats' single-tile wrappers, and the grouped MoE path are untouched;
   NSPLIT composes for free - each sub-N half is itself a case-8 dispatch.

Exclusions: none at 8/16/32 on indexing/SMEM grounds. Tile 64: SMEM fine
(13.8 KB), decision deferred to the ptxas register/spill check below. ROCm:
excluded by design (knob returns MMQ_X_Q8_0 = 64 unchanged, pinned by
static_assert).

## ptxas verdict (sm_120, nvcc 13.3, -O3 --expt-relaxed-constexpr, bf16 == half)

Mini-TU probe forcing all 10 dense q8_0 instantiations (both need_check
branches x 5 tiles); log /tmp/ptxas_dense.log:

| tile | need_check | regs | smem (B) | spill st/ld |
|------|-----------|------|----------|-------------|
| 4    | 0/1       | 56/64  | 5328  | 0/0 |
| 8    | 0/1       | 40/68  | 5904  | 0/0 |
| 16   | 0/1       | 60/72  | 7056  | 0/0 |
| 32   | 0/1       | 64/106 | 9360  | 0/0 |
| 64   | 0/1       | 78/87  | 13968 | 0/0 |

- SMEM matches the audit byte math exactly (5328 = 4752 x-tile + 512 + 64;
  13968 = 4752 + 8192 + 1024). All < 48 KB static limit.
- ZERO spill stores/loads at every tile including 64; max 106 regs (tile 32
  need_check=1) << 255. TILE 64 INCLUDED in the knob.
- Note: tile 8/need_check=0 compiles to FEWER regs (40) than tile 4 (56) -
  ptxas scheduling variance, not a linear trend; the A/B decides payoff.

## Implementation (this wave)

- mmq.cuh: mul_mat_q8_0 templated <scalar_t, need_check, mmq_x> (+2
  static_asserts pinning mmq_x >= nwarps and mmq_x % nwarps == 0) and launcher
  ggml_mul_mat_q8_0_q8_1_cuda templated <scalar_t, mmq_x> (grid from the
  template; need_check branches unchanged, keyed on N % 32).
- gguf_kernel.cu: ft_gguf_dense_mtile() (single source of truth, per-call
  getenv, unset/empty/"4"->4, 8/16/32/64, else TORCH_CHECK) placed before
  ggml_mul_mat_a8; FT_DENSE_MMQ_ARGS + FT_DENSE_TILE_SWITCH() macros (ROCm
  branch calls <scalar_t, MMQ_X_Q8_0> directly, static_assert pinning
  MMQ_X_Q8_0 == 64); case 8 now expands the switch; probe ggml_dense_get_mtile
  exposed for the knob unit test (mirrors ggml_moe_get_block_size).
- kernel/gguf.py: ggml_dense_get_mtile wrapper + __all__.
- layers/moe.py: grouped-prefill docstring documents BOTH knobs (separate
  meanings, never affect each other's paths).

## Test evidence (targeted, RTX 5090, serialized)

- New tests in tests/kernels/test_gguf_quant.py (mirror rule, no new files):
  test_dense_mtile_env_knob (default/empty->4, per-call flip 8/16/32/64,
  cross-knob isolation both directions, stragglers "12"/"0"/"-4"/"abc"/
  " 8"/"04"/"+8"/"08" fail fast), test_dense_mtile_scope_mmvq_and_moe_untouched
  (DENSE_MTILE=32 + MOE_MTILE=32: batch 4 still MMVQ vec, batch 7 MMQ),
  test_dense_q80_mtile_bitwise_invariance (torch.equal across 8/16/32/64 vs
  tile-4 baseline on (128,96,27) noncontig-geom + (4096,24896,256) production
  + (128,100,13) need_check=1/M-tail), test_dense_mtile_nsplit_composition
  (NSPLIT=2 + tile 32 == NSPLIT=2 + tile 4 bitwise, launches stay [12448,
  12448]).
- First targeted run: 4 passed in 49.49s (includes the ONE JIT rebuild).
  Immediate full-file rerun: 80 passed in 4.61s -> NO repeated rebuild; the
  source hash is stable.

## Full gate + Environment classification

`uv run --extra dev pytest tests/ -m "not slow" --ignore=tests/models/
test_glm5_next_kda_snapshot.py -q`: **16 failed / 2179 passed / 206 skipped**
in 178s (wave-1 baseline: 16/2173/206; passed delta = the 4 new tests + 2
parametrized ids elsewhere in the collection drift band).

The 16 failures isolated + classified per protocol:
1. Current tree (with wave changes): 16 failed, error classes =
   ModuleNotFoundError "No module named 'tests'" x11 (dsa_kpool x7, dsa_pool x2,
   muse_glimmer x1, glm_dsa ragged[nvfp4] x1 - import failure at
   test_glm_dsa.py:316), triton OutOfResources 102400 > 101376 (glm_dsa x2),
   cudaHostGetDevicePointer pinned+mapped (pinned_tensor), CUDA invalid argument
   (qsa_fp8 [1-1-64]), ple boundary (qwen4_exp).
2. Stash-repro (5 wave paths stashed, HEAD 4505572 tree): the SAME 16
   nodeids failed identically in 2.93s.
-> ALL 16 Environment (pre-existing baseline), zero wave-caused failures.
No gguf test among them; my 5 changed paths are disjoint from every failing
file. Census before/after: no ft serve / nsys / pytest strays; GPU 1.5 GB
desktop baseline before, 1.5 GB after (all test memory freed).

## Deviations / notes

- Task brief said HEAD 9f09596; actual HEAD 4505572 (= 9f09596 + a .veai-only
  "Memory" commit). Python tree identical to the expected state; no action.
- Probe ggml_dense_get_mtile added to the extension + kernel/gguf.py: the
  dense knob has no Python-visible getter otherwise; mirrors the v3a
  ggml_moe_get_block_size single-source-of-truth regression hook.
- The stash-repro was run on the 16 failing nodeids only (not the full gate)
  to avoid a second full JIT rebuild; the failing files are disjoint from the
  gguf path, so no rebuild was triggered at all (2.93s run).
- A/B sweep of tiles 8/16/32/64 on hardware is the NEXT wave (this wave only
makes it measurable). MoE knob default flip remains a separate user-gated
micro-commit.

## ptxas + census artifacts (A/B sweep pre-step, 2026-09-19T22:20+03:00)

Review-B F3/F4 + Review-A F2 closure. Census at the sweep preflight:
`pgrep -a 'ft serve|nsys|pytest'` -> no matches (rc=1); GPU 1569/32607 MiB
desktop baseline; sem.mp- count 5 (IDE baseline, attributed per the
between-wave rule). No foreign VRAM holders; nothing killed.

`/tmp/ptxas_dense.log` re-read verbatim (2026-09-19T22:20+03:00) - 20 entries
(bf16 x 10 + half x 10), all `sm_120`, all `0 bytes spill stores, 0 bytes spill
loads`. Two `#20012-D` BFloat16-defaulted-ctor warnings head the log (probe-TU
noise, not kernel warnings). regs/SMEM table matches the wave table above
EXACTLY (mangled names confirm the `<scalar_t, need_check, mmq_x>` signature):

| tile | need_check | regs (nc1/nc0) | smem (B) | spill st/ld |
|------|-----------|----------------|----------|-------------|
| 4    | 1/0       | 64/56          | 5328     | 0/0 |
| 8    | 1/0       | 68/40          | 5904     | 0/0 |
| 16   | 1/0       | 72/60          | 7056     | 0/0 |
| 32   | 1/0       | 106/64         | 9360     | 0/0 |
| 64   | 1/0       | 87/78          | 13968    | 0/0 |

The 8267-byte log was /tmp-TTL at risk; this table + the gate log are the
durable copies. Full gate rerun for this wave: gate-candidate-a.log (invocation
header + per-test nodeids; counts recorded in the A/B wave report and
ab-mtile.md).

Gate result (2026-09-19T22:19 invocation, HEAD 4505572, changeset tree incl.
the :505 de-indent fix): **16 failed / 2179 passed / 206 skipped in 223.57s** -
counts identical to the wave-2 baseline. F4 resolution from the nodeid list:
import class = dsa_kpool x7 + dsa_pool x2 + muse_glimmer x1 + glm_dsa
ragged[nvfp4] x1 (test_backend_ragged_prefill_identity_and_selection[nvfp4] -
ModuleNotFoundError at test_glm_dsa.py:316) = 11; glm_dsa Out-of-Resources = 2
(test_sparse_kernel_equals_dense_when_all_selected,
test_splitk_matches_single_program); plus pinned_tensor UVA, qsa_fp8 [1-1-64],
qwen4_exp ple boundary. CORRECTION (review-b F6): an earlier version of this
note read "glm_dsa Out-of-Resources = 3" and inferred a wave-1 -> wave-2 class
shift - that was a misread of gate-candidate-a.log: the [nvfp4] item is an
import-class failure at :316, the same site wave-1 F4 cited. There was NO class
shift: 11 import / 2 OOR exactly as wave 1; the failing FILE set is identical
(test_glm_dsa.py); all 16 in documented
baseline Environment families, all failing files disjoint from the 5 changed
paths, zero gguf-path failures.

