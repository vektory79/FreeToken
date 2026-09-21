# TASK: v3 - weight-stationary MoE schedule for grouped GGUF MMQ (close the read-once gap)

Status: v3a COMPLETE through hardware A/B (2026-09-18); winner tile 32
(+67.2% @8128 = 556.43 tok/s); v3b NOT triggered. COMMITTED 1706aae
(2026-09-19, 5 files +318/-147, not pushed).
Composed
2026-09-18 from the v2 outcome
(`.tasks/mmq-prefill-kernel/TASK.md` v2 verdict block + `v2-ab.md`). Read
CONTRIBUTING.md first - it is binding. Line anchors were verified 2026-09-18
against the v2 changeset; re-verify before editing (they drift). Private-use
scope for GGUF glm5next: no push, no upstream PR.

## Goal

Close the gap between the v2 grouped MMQ prefill (measured +16.4% @8128:
289.92 -> 337.61 tok/s; MoE term 18.13 -> ~9.5-10.5 s per chunk) and the
read-once roofline (MoE 1.5-3 s -> implied ~758 tok/s on the 8128 chunk).
After v2 the MoE kernel still re-reads expert weights per 4-token m-block
(HYPOTHESIS from v2-ab.md, UNVERIFIED: ~57x per expert at ~226 pairs/expert,
i.e. traffic cut ~4x vs moe_vec instead of to-once).

## Verified background (do not re-derive; re-anchor before editing)

- Step-0 measured split of the 28.77 s 8128 chunk
  (.tasks/mmq-prefill-kernel/step0-profile.md): MoE GEMM 18.13 s (63%),
  fetch copies 3.15 s, rest 7.49 s - dense q8_0 GEMM 6.04 s is the top
  "rest" item and the next bottleneck after this task.
- v2 state (committed 2026-09-18): grouped entry `ggml_moe_a8` with cases
  18 (IQ3_XXS) / 23 (IQ4_XS) wired via the `moe_align_block_size` trio
  (freetoken.moe.fused), block_size = MOE_X = 4 (one trio serves gate/up
  and down), moe_vec prefill survives only as the unsupported-type
  fallback (block size 0), kill switch removed, decode untouched
  (moe_vec, CUDA-graph capture). Validation: 126 tests, kernel-level nsys
  liveness, perf A/B (+14.4/+15.6/+16.4% @4096/6144/8128), 24-prompt
  quality battery PASS.
- Geometry: H=4096, I=2048, E=288, top_k=8, 42 routed MoE layers,
  pairs <= 65024 per 8128 chunk (~226 pairs/expert). Expert types:
  39x (IQ3_XXS gate, IQ3_XXS up, IQ4_XS down), 2x (IQ3_XXS x2, Q6_K),
  1x (IQ4_XS x2, Q6_K).

## Step 0 - verify the re-read hypothesis (first, before any design)

v2-ab.md labels the per-m-block re-read model UNVERIFIED. Pin it:
1. Source read: the vendored `kernel/csrc/gguf/moe.cuh` moe_q schedule
   (vLLM PR #36226 lineage): how many weight-tile bytes each m-block
   (MOE_X = 4 tokens) loads per k-step; derive the per-expert weight
   re-read factor analytically (pairs/expert / m-tile x tile bytes).
2. Reconcile with measurement: grouped MoE ~9.5-10.5 s vs moe_vec 18.13 s
   (~1.8x time cut) is consistent with a ~4x traffic cut at constant BW -
   confirm the 4x factor from source, not just the ratio.
3. Optional cross-check: ncu single-layer microbench (one expert, N pairs):
   weight bytes moved vs N x W bytes.

RESULT (2026-09-18, source read R1; full report + file:line anchors in
`step0-source-read.md`):
- "~57x per-expert re-read at MOE_X=4" VERIFIED: factor = ceil(n_e/MOE_X)
  per projection; per-m-block weight bytes = the full projection matrix S
  (moe.cuh:43-88 k-loop; load_tiles per k-step = full 32-row k-span).
- Traffic model VERIFIED: moe_vec 29.53 TB/chunk (back-solves to the measured
  18.13 s within 0.6%); grouped 7.43 TB/chunk = 3.97x cut = 4.54 s BW floor.
- Time reconciliation PARTIAL: measured ~10 s = 2.2x above the floor -> the
  kernel left the BW-bound regime; dominant terms are the per-MAC decode
  ALU/LDS/LUT work (invariant across kernels) + k-step serialization (64
  syncs/block, no double-buffering) + unaligned 98-B byte-assembly loads (4x
  LSU amplification). L2 (~96 MB) serves co-resident same-expert m-blocks and
  pushes time DOWN; padding ~1-2% negligible; SMEM 4.9 KB/block not limiting
  (no __launch_bounds__ on CUDA, moe.cuh:1469-1472 is ROCm-only).
- Factor formula for the sweep: F(MOE_X) = ceil(n_e/MOE_X) -> 57 / 28.5 /
  14.1 / 7.1 at m-tile 4/8/16/32; chunk traffic cut ~7.9x/15.8x/30.3x incl.
  padding.
- v3a code caveats: the ds-load token_offs[0] fix (moe.cuh:113-118) holds
  only while mmq_x/nwarps == 1; sum registers grow as (mmq_y/32)x(mmq_x/nwarps).
- Decisive ncu set if the sweep saturates unexplained: dram__bytes.sum +
  warp-stall breakdown (short_scoreboard/long_scoreboard/barrier/
  mio_throttle/not_selected) + l1tex LDS wavefronts; mio_throttle/
  short_scoreboard dominant = ALU/LDS-bound (sweep futile), barrier/
  long_scoreboard dominant = staging-bound (sweep pays).

The verified factor is recorded; v3a/v3b design may start.

## v3a - larger m-tile (config-level experiment, ~1 day)

MOE_X_<type> = 4 tokens per block; each m-block reloads the expert's
weight tiles for its k-slices. Raising the m-tile multiplies tokens served
per weight-tile load (4 -> 16/32/64 = 4-16x further traffic cut toward
to-once) without a new kernel:
1. Add larger m-tile variants (or a runtime m-tile selection) for types
   18/23 (+ q8_0/q6_K for uniformity) in `moe.cuh`, the `ggml_moe_a8`
   switch in `gguf_kernel.cu`, and `ggml_moe_get_block_size` - block_size
   feeds `moe_align_block_size`, so the trio adapts automatically; the
   trio-contract and cap tests must be re-pinned for the new size.
2. Watch: the x-tile (weights) is m-INVARIANT (mmq_y=32); only the y-tile
   (activations) scales with the m-tile (144 x m bytes) - the brief's
   "tile_x/tile_y scales with the m-tile" was half wrong, corrected by the
   R2 surface map (v3a-surface-map.md). Real gates: the ds-fill port (see
   design decisions), register pressure of load_tiles/vec_dot unrolls
   (sum[mmq_y/32][mmq_x/nwarps] grows), and tail-expert padding waste
   (segments pad to the m-tile; n_e = 1-2 experts waste (m-tile-1) rows -
   measured model: 0.7/1.6/3.5/7.1% of pair slots at m=4/8/16/32; compute
   waste only, zero correctness risk).

   v3a design decisions (from step0-source-read.md + v3a-surface-map.md,
   both in this directory):
   - PRECONDITION for any m>4: port the full-coverage ds fill into moe_q
     (moe.cuh:106-116 fills only ds rows [0, nwarps) and assumes
     mmq_x/nwarps==1 via token_offs[0]; rows >= nwarps are garbage at m>4).
     Adapt the dense mul_mat_q ds loop (mmq.cuh:77-88) to sorted_token_ids.
   - Variant mechanism: compile-time MMQ_X template on moe_<type> +
     ggml_moe_<t>_q8_1_cuda, explicit instantiations {4,8,16,32} for types
     8 (q8_0) / 14 (q6_K) / 18 (IQ3_XXS) / 23 (IQ4_XS); runtime env knob
     selects the variant; ggml_moe_a8 switch AND ggml_moe_get_block_size
     must derive from the SAME selection (divergence = cross-expert pair
     corruption - single source of truth).
   - SMEM feasible to m=32 for all four formats (q6_K binding: 10/10/9/7
     blocks/SM at 4/8/16/32); no __launch_bounds__ on CUDA today; static
     __shared__ only, 48 KB limit never approached.
   - Decode (moe_vec + CUDA-graph capture) reads none of it - confirmed
     isolated (moe.py:441-468 vs is_prefill-only grouped path).
   - JIT: source-hash keyed rebuild, kernel-cache wheel has zero gguf
     coverage - no stale-prebuilt hazard; clang++ host required.
3. A/B on hardware (protocol below): sweep m-tile 4/8/16/32 per format;
   expect the MoE term to scale roughly with the traffic cut until SMEM/
   occupancy binds; report where it saturates.

## v3b - expert-stationary persistent kernel (new kernel, bigger effort)

One CTA(-cluster) per (expert, layer): stream the expert's packed weights
once while iterating all its pairs' token tiles in SMEM (weights in the
outer loop = weight-stationary; to-once traffic). Only start v3b if the
v3a sweep saturates below the ceiling (SMEM/occupancy bound) or tail
padding dominates:
- SMEM cannot hold a 3.4-10.9 MB expert - weights stream from L2/VRAM
  exactly once; activations tile in SMEM.
- Pair counts per expert vary ~1..600 -> persistent CTAs with work
  stealing or a fixed expert partition; grid no longer tied to
  sorted_token_ids.numel()/mmq_x (issue-#186 cap logic changes - keep an
  equivalent guard).
- Parity must stay within the grouped-entry tolerances (q8_0/q6_K
  grouped-vs-moe_vec 1e-3; iq vs gguf-py reference thresholds).

## Hardware validation protocol

Same as `.tasks/mmq-prefill-kernel/TASK.md` (winner flags, port 18801,
medians of full chunks excluding c1 warmup AND the last-full-chunk report
artifact, decode @65k radix-hit, hygiene: census / FAILPAT watchdog
AssertionError|OutOfMemoryError|Backend worker is gone / serial reaped
runs / SIGTERM->SIGKILL 30 s). Notes for v3:
- Quality battery tooling EXISTS: `.tasks/mmq-prefill-kernel/quality/`
  (runner + divergence analyzer, 24 prompts, PASS bar documented in
  quality-ab.md). The env-flip A/B is gone (kill switch removed) - A/B =
  pre-change boot vs candidate boot, or compare against the committed v2
  numbers (289.92 / 325.28 / 337.61 tok/s) and the battery PASS record.
- Liveness: nsys kernel-name probe (grouped kernel launch counts and
  moe_align buffer sizes change with the m-tile - re-run the trio
  contract tests and the kernel-count math 42 x 3 x chunks).

## Tests

Extend `tests/kernels/test_gguf_quant.py` +
`tests/models/test_gguf_expert_banks.py` first: parity per format at the
new m-tile, trio contract with the new block_size, grid-cap tests, decode
regression (moe_vec + capture untouched), grouped-vs-moe_vec consistency.

## Commits

Conventional Commits, lowercase, imperative. Suggested:
`perf(kernels): widen grouped moe m-tile for prefill reuse` (v3a).
Commit only when the user asks. Never push.

## Wave log (v3a, 2026-09-18)

- Implementation (5 files): moe.cuh full-coverage ds-fill port + moe_q8_0/
  q6_K/iq3_xxs/iq4_xs + launchers templated on mmq_x; gguf_kernel.cu
  ft_gguf_moe_mtile() SINGLE SOURCE OF TRUTH (env FREETOKEN_GGUF_MOE_MTILE,
  global knob read per call, TORCH_CHECK on invalid), FT_MOE_ARGS/FT_MOE_TILE_SWITCH
  compiles tiles {4,8,16,32} x need_check for cases 8/14/18/23, get_block_size
  returns the knob; moe.py docstring; tests re-pinned + analytic ds uniform
  regression. Deviations from the brief: implicit instantiation via dispatch
  (same compiled set); ROCm knob returns the ROCm tile (8) and keeps the
  single-tile dispatch.
- Tests: touched files 119/119 green (post-fix); full gate 2383 passed /
  5 pre-existing baseline (pinned_tensor UVA, qsa_fp8 [1-1-64],
  qwen4_exp/test_ple :421, 2x glm_dsa OoOR). The old cap-test NaN flake is
  FIXED by this changeset (torch.empty -> torch.zeros x).
- Review x2: SHIP / SHIP, no blockers/majors. Triage: 6 TP applied (ROCm
  static_assert pinning of the four MOE_X equality; 16-arg comment;
  docstring per-call + do-not-flip-after-boot; other-types-unaffected assert
  for types 3,6,7,10-13; invalid-env stragglers " 8"/"04"/"+8"; iq
  grouped-vs-moe_vec tolerance tightened to atol 4e-3 / rtol 2e-3 with
  measured 3x+ headroom; analytic tail-expert test at tiles 16/32).
  Deferred, non-gating: Q6_K need_sum=true analytic fixture (optional
  follow-up); ptxas -v spill capture (sweep item); cap tiles 16/32 accepted
  (size-generic check).
- STOP GATE: pass except the commit gate - USER-GATED by repo rules; the
  commit lands after the A/B numbers exist. m=4 is byte-identical to v2
  (review-verified: tile-invariant reduction order) -> the committed v2
  numbers (289.92/325.28/337.61) and battery PASS remain valid baseline
  anchors (T44).
- Next: hardware sweep tiles 4/8/16/32 (winner flags, port 18801, T43
  last-chunk exclusion, T44 baseline reproduction, T45 knob mechanics, T46
  nsys liveness + exact launch-count math), per-format saturation read,
  battery on the winner tile, v3b trigger assessment.

## Hardware sweep RESULT (2026-09-18/19; full report v3a-ab.md + v3a-ab-results.json in this directory)

- Validity gate (T44) PASS: baseline tile-4 boot 332.72 tok/s = -1.45% vs
  the committed v2 class 337.61 (inside +-3%).
- A/B @8128 (medians c2..c7): tile4 332.72 | tile8 436.11 (+31.1%) |
  tile16 512.48 (+54.0%) | tile32 556.43 (+67.2%; +64.8% vs committed
  337.61). Decode flat 14.07-14.74 (prefill-only knob); peak VRAM flat
  ~32.1 GB; all radix re-sends HIT 65536; no OOM/watchdog events.
- Saturation: marginal gain halves per doubling; MoE-term ratio vs pure
  1/tile = 0.500/0.260/0.152; nsys grouped MoE 12.62 -> 5.01 s/chunk at
  tile 16 = 2.52x for a 15.8x traffic cut -> ALU/LDS/serialization-bound,
  NOT SMEM/occupancy-bound (Step-0 regime confirmed); padding does not
  dominate. Practical saturation point: tile 32 (tile 64 would buy ~3-4%
  at best with ~14% padding waste and dropping q6_K occupancy).
- v3b DOES NOT FIRE: traffic reduction beyond tile ~32 cannot pay under
  the ALU/LDS floor. The chunk is now ~80% "rest" - the next levers are
  dense q8_0 GEMM (6.04 s) and fetch copies (3.15 s), both out of scope
  here (see below).
- Liveness (T46, nsys tile16, D09): 126 grouped + 42 moe_align per 8128
  chunk, ZERO moe_vec_q in the prefill span, 7938 moe_vec in decode
  graphs (42x63x3), CTAs/launch shrink 3.8x vs tile-4 -> knob live in the
  serving path.
- Quality battery: 24/24 outputs BITWISE IDENTICAL between tile 4 and
  tile 32 (per-element reduction order is tile-invariant), 4/4 digit
  prompts, 0 flips - PASS at the strongest possible bar.
- Campaign verdict: v3a closes the task. v3b not built (premise dead per
  measurement, not paper). Deferred follow-ups: Q6_K need_sum=true
  analytic fixture; ptxas -v spill capture (skipped, non-gating).

## Out of scope (separate tasks)

- Dense q8_0 GEMM (6.04 s rest item) - becomes the top bottleneck once the
  MoE term is fixed.
- Fetch copies 3.15 s (PCIe-class copy kernel, `fast_index_copy_multi`).
- Prefill overlap enablement (576 slots/partition) - capacity redesign.
