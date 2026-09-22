# Review-A - Replan wave 2, Candidate A: dense q8_0 m-tile templating + FREETOKEN_GGUF_DENSE_MTILE

Focus: kernel-correctness of the templating + knob (protocol Review-A: correctness /
behavior / architecture). Read-only review; grounded in current source, not the wave
report's words.

Reviewed tree: branch vektory79, HEAD 4505572 (4505572f0070316b1ef595ed88bfabad4ede2b81).
Changeset (uncommitted, verified via `git diff --stat`): mmq.cuh (+20/-13),
gguf_kernel.cu (+94), kernel/gguf.py (+6), layers/moe.py (+7/-3), tests/kernels/
test_gguf_quant.py (+108). Method: full read of mul_mat_q core (mmq.cuh:18-116),
mul_mat_q8_0 + launcher (mmq.cuh:441-513), vecdotq.cuh:1037-1055, gguf_kernel.cu knob
+ switch + probe + case-8 site, layers/gguf.py dispatch + kda_in_proj_forward,
the 4 new tests, ptxas log /tmp/ptxas_dense.log, git history 9f09596..4505572.

## A1. Templating purity - CONFIRMED

mul_mat_q is already fully templated on mmq_x; the wrapper change swaps a macro
constant for a template param (mmq.cuh:441-474). Per-element reduction order is
tile-invariant, verified structurally in the core:

- Each output element (weight row, token) is computed by exactly ONE thread:
  sum[i/32][j/nwarps] accumulates only over k (mmq.cuh:87-96). No cross-token
  reduction exists at any tile; widening mmq_x only adds j-strips per thread.
- The k iteration (`k = ir*32/qr; k < (ir+1)*32/qr; k += vdr` inside
  `ir < qr` inside `ib0 += blocks_per_warp`) depends only on qr/vdr/qi
  (q8_0 constants), never on mmq_x. Per element the vec_dot call sequence
  and operand order are identical at every tile -> bitwise invariance for
  ALL M/N, not only the tested shapes.
- tile_y_qs/tile_y_ds content per token row r is y[col_y_0 + r] at every tile
  (fill loops mmq.cuh:69-75, 77-89 scale with mmq_x but keep the same layout);
  vec_dot reads y_qs[j*32+k] / y_df[j*4+k/8] with j < mmq_x always in-bounds
  (vecdotq.cuh:1037-1055). x-side (weights) indices are mmq_y-only and the
  x-tile allocators load_tiles_q8_0<mmq_y,...> / allocate_tiles_q8_0<mmq_y>
  are not templated on mmq_x.
- ds/scale handling stays per-32-row K-blocks (QK8_1=32) at every tile; the
  need_sum=false in-place half2->float ds write is unchanged.
- Epilogue guards (mmq.cuh:97-116): `col_dst >= ncols_dst -> return` (M-tail;
  safe because col_dst grows monotonically with the loop j and the return sits
  after all __syncthreads - no divergence hazard) and `row_dst >= nrows_dst ->
  continue` (N-tail) are mmq_x-independent and cover tails identically at tile 64.

BLOCKER check: no tile can reorder the reduction. The torch.equal tests pin three
shapes; the argument above holds for all M/N.

## A2. Pre-audit verification against source - CONFIRMED (a-d)

(a) sum[mmq_y/32][mmq_x/nwarps] = [1][mmq_x/4] on CUDA (mmq.cuh:53): minimum
    tile 4 gives [1][1]; zero-length impossible with nwarps=4. The ROCm nwarps=8
    hazard is double-pinned: the knob's ROCm branch returns MMQ_X_Q8_0=64 only,
    and the two static_asserts (mmq_x >= nwarps, mmq_x % nwarps == 0) exist in
    the kernel body (mmq.cuh:462-465) and bind at every instantiation (verified:
    they are plain C++ constant expressions over the template param and a macro
    literal; compile under hipcc semantics unguarded).
(b) ds-fill coverage at tile 64 (mmq.cuh:77-89): stride nwarps*QI8_1 = 32 gives
    two ids0 rounds {0, 32}; each round enumerates ids = ids0 + ty*8 + tx/4 over
    exactly 32 consecutive values, kby = tx%4 covers all 4 sub-slots per ids ->
    full coverage of [0,64) x [0,4), no unwritten slot, multi-writers (tiles
    4/8/16) are same-value benign (dsi_src depends only on (ids,kby,ir,ib0)).
    Read side matches: y_df[j*4 + k/8] over k/8 in [0,4).
(c) Beyond-M clamps: qs-fill min(col_y_0 + ty + i, ncols_y-1) (mmq.cuh:71) and
    ds-fill min(col_y_0 + ids, ncols_y-1) (mmq.cuh:81) clamp to the SAME last
    token; the discarded rows are killed by the epilogue guard (A1). N-side
    rows are clamped in load_tiles via i = min(i, i_max) under need_check.
(d) need_check keyed on nrows_x % mmq_y == 0 = N % 32 (launcher mmq.cuh:498),
    untouched by the knob; the M side has no need_check path by design and is
    covered by (c). Verified in the launcher diff: only `bool` -> `constexpr`
    and the added template arg changed.

## A3. ft_gguf_dense_mtile single source of truth - CONFIRMED

- Exactly ONE getenv("FREETOKEN_GGUF_DENSE_MTILE") site in the repo
  (gguf_kernel.cu:216; project-wide search: all other hits are comments/docstrings/
  tests). Consumers: the FT_DENSE_TILE_SWITCH macro (one expansion site, case 8)
  and the test probe ggml_dense_get_mtile. Per-call read, no static cache.
- unset / empty / "4" -> 4 (gguf_kernel.cu:218-221); strict strcmp whitelist
  8/16/32/64 (no trim/numeric parsing, so " 8"/"04"/"+8"/"08" all reject);
  TORCH_CHECK message names the knob (gguf_kernel.cu:232).
- Scope: FT_DENSE_TILE_SWITCH() expands only in case 8 of ggml_mul_mat_a8;
  every other case (other dense formats) and ggml_moe_a8 / ggml_moe_a8_vec
  are untouched. MMVQ cannot reach it: fused_mul_mat_gguf routes
  batch <= _MMVQ_SAFE(6) to ggml_mul_mat_vec_a8 (layers/gguf.py:71-73) which
  has its own switch with no knob; the knob consumer is only the batch > 6
  MMQ route (layers/gguf.py:74-75). ggml_mul_mat_q8_0_q8_1_cuda call sites:
  the macro's 6 expansions only (searched).

## A4. ROCm path - CONFIRMED

- ROCm FT_DENSE_TILE_SWITCH calls <scalar_t, MMQ_X_Q8_0> directly (single
  instantiation shape 64) with static_assert(MMQ_X_Q8_0 == 64) pinning the
  shape (gguf_kernel.cu:214). Kernel static_asserts pass at 64/8. No ROCm
  compilation path can instantiate a zero-length strip tile: the only dense
  q8_0 instantiation site is the macro; the kernel asserts bind under plain
  C++/hipcc (not CUDA-guarded, which is correct here).

## A5. Composition with NSPLIT - CONFIRMED

- kda_in_proj_forward (layers/gguf.py:126-175) splits at a 32-row edge:
  half = (n_out//2)//32*32; for any N with N%32==0 both halves are %32==0
  (N=32*(2m) -> exact halves; N=32*(2m+1) -> m*32 and (m+1)*32), so each half
  keeps need_check=false. Each half is a fused_mul_mat_gguf -> ggml_mul_mat_a8
  case 8 dispatch -> per-call knob read -> BOTH halves get the widened tile;
  launch structure [12448, 12448] preserved, gridY = ceil(12448/tile) shrinks
  per tile (3112 -> 195 at tile 64). Gates (n_tok>6, type in _MMQ, N>=64,
  N%32==0) unaffected by the tile knob; batch <= 6 halves fall back to MMVQ.
- Test test_dense_mtile_nsplit_composition pins bitwise equality + launch
  structure [12448, 12448] at tile 32.

## A6. Probe + pybind - CONFIRMED (one accepted deferral)

- ggml_dense_get_mtile() returns ft_gguf_dense_mtile() - the SAME single
  source of truth, no duplicate getenv (gguf_kernel.cu:964-967); pybind
  m.def mirrors ggml_moe_get_block_size; kernel/gguf.py wrapper + __all__
  added. The knob unit test uses the probe as the single-source regression
  hook (no numbers pinned twice).
- Residual gap (non-blocking): the probe proves knob RESOLUTION, not launch
  WIRING; the bitwise tests exercise the real path but cannot distinguish
  which tile ran. Wiring is code-verified (single macro expansion is the only
  consumer) and the T46 liveness check (grid shrink per launch) is already
  deferred to the A/B wave per the wave report. Accepted deferral; the A/B
  wave must actually run it.

## A7. Test adequacy - CONFIRMED with named gaps (non-blocking)

Covered: bitwise across 8/16/32/64 vs tile-4 baseline on (128,96,27) single
partial y-block at every tile, (4096,24896,256) production, (128,100,13)
need_check=1 + M-tail; knob unit (default/empty, per-call flips, whitelist
stragglers incl. " 8"/"04"/"+8"/"08", cross-knob isolation BOTH directions);
MMVQ cutoff (batch 4 vec / batch 7 MMQ under both knobs); NSPLIT composition
(bitwise + launch structure). Mirrors the v3a battery design.

Missing cases (could hide a real bug; none blocks this changeset):
1. Multi-y-block M at wide tiles: no shape with M spanning >1 full y-block at
   tile 64 plus a partial second block (e.g. M=65..127, suggest (128,96,100)).
   M=27/13 exercise a single partial block; M=256 divides evenly. Analytically
   covered by A1, but untested on hardware.
2. K%128 != 0 (q8_0 outer ib0 tail): tested K values (128, 4096) give whole
   blocks_per_warp=4 iterations. Note: load_tiles_q8_0 reads blocks
   [ib0, ib0+4) with no kbxd guard, so K%128!=0 is structurally unsupported by
   the q8_0 MMQ - PRE-EXISTING upstream behavior, tile-invariant, unchanged by
   this changeset; worth a follow-up issue, not a Candidate A blocker.
3. fp16 scalar path: new tests run x as bf16 only (production dtype); the
   ptxas probe covered both scalar instantiations.

## A8. PTXAS claims - CONFIRMED

/tmp/ptxas_dense.log re-read: all 20 entries (half + bf16 x 10) match the wave
table verbatim (regs 56/64, 40/68, 60/72, 64/106, 78/87; SMEM 5328/5904/7056/
9360/13968; 0 spill stores/loads; sm_120; mangled names confirm the
<scalar_t, need_check, mmq_x> signature). SMEM byte math exact: x-tile 4752 B
(tile_x_qs[32*33] int 4224 + tile_x_d[132] float 528, mmq_x-independent) +
y-tile 144*mmq_x B (tile_y_qs[mmq_x*32] int 128*mmq_x + tile_y_ds[mmq_x*4]
half2 16*mmq_x) -> 5328..13968 B, all << the 48 KB static limit. Regs: max 106
(tile 32 need_check=1) << 255; 78 regs at tile 64 with a 16-float accumulator
is plausible (78 x 128 thr = 9984 regs/block vs 64K/SM regfile); the t8-nc0=40
< t4-nc0=56 inversion is ptxas scheduling variance, not impossible. The tests'
in-tree JIT rebuild at all tiles additionally proves in-tree compilability.

## Findings

- BLOCKER: none.
- MAJOR F1 (out-of-changeset, tree integrity): the ".veai-only Memory commit"
  claim for 4505572 is FALSE. `git log 9f09596..4505572` = exactly one commit
  ("Memory", 4505572) and its stat includes pyproject.toml +1 line:
  `ipykernel>=7.3.0` added to core dependencies - the same stray line wave 1
  explicitly left out of its commit. The NARROW claim IS true: no
  python/freetoken file changed in that range, so the reviewed changeset sits
  on the intended python tree and is unaffected. Action: user-gated - revert
  the pyproject.toml line in a follow-up commit (it is a parallel-session
  artifact, origin unknown) and correct the wave report's HEAD note.
- NIT F2 (in-changeset, cosmetic): layers/moe.py:505 - the added dense-knob
  sentence de-indented the existing docstring line "One moe_align trio is
  shared..." to column 0 inside the docstring. Docstring-only, no behavior;
  fix the indent as cheap polish at commit time.
- NIT F3: A6 wiring gap - closed operationally by running the T46
  grid-shrink liveness check in the already-planned A/B wave; do not skip it.

## Overall verdict

SHIP (Candidate A changeset as reviewed). No kernel-correctness blocker: the
templating is pure (per-element k-order tile-invariant), the knob has a single
per-call source of truth with a strict whitelist and case-8-only scope, ROCm
is pinned, NSPLIT composes, and the ptxas evidence is verified verbatim.
Route F1 to the user (revert + report correction), apply F2 at commit time,
and bind the T46 liveness check into the A/B wave (F3).
