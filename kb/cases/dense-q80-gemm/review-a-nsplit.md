# Review-A - N-split of kda_in_proj (Replan wave 1, dense-q80-gemm)

Scope: implementation CORRECTNESS vs spec (TASK.md "Replan wave 1", review-a-step0.md C7). Diff vs HEAD 1706aae, uncommitted on vektory79: python/freetoken/layers/gguf.py, python/freetoken/models/glm5_next/kda.py, tests/kernels/test_gguf_quant.py, tests/models/test_glm5_next_kda_op.py. Method: full read of the changed sources plus the consumers (fused_mul_mat_gguf, GGUFLinear.forward, the csrc wrapper gguf_kernel.cu, the mmq.cuh q8_0 kernel/launcher, the loader swap in glm5_next/gguf.py, engine/graph.py, engine.py) and the x-quant kernel. Read-only pass; no code touched.

## 1. Per-item verdicts

### A1 - Exactness argument vs implementation: CONFIRMED (no BLOCKER)

- x-quant is internal, per-launch and deterministic in BOTH paths. ggml_mul_mat_a8
  quantizes X on every invocation (gguf_kernel.cu:209-212; quant_X is sized only by
  col/batch - identical for both launches). The helper passes the literal same x
  tensor to both sub-launches (gguf.py:155-156). quantize_q8_1 is a warp-butterfly
  fmaxf/sum reduction plus per-element roundf with exactly one writer per (block,
  lane) slot - no atomics, no cross-block state (gguf_kernel.cu:26-53) -> identical
  quant_X bytes in launch 1 and launch 2. K-padding to 512 is identical (same K).
- Weight slices stay block-aligned. qweight is [out, row_bytes] with each output row
  packed independently over K (gguf.py:100; module doc gguf.py:7-13); qw.narrow(0,...)
  is a pure byte view, so chunk-2 row j is read at the same absolute byte offset as in
  the single launch. Production row_bytes = 4352 = 256 x 17 -> the second base pointer
  (half x 4352 bytes) is 256-byte aligned. B = 12448 = 389 x 32.
- Boundary math is exact for the whole gated domain. half = (n_out//2)//32*32
  (gguf.py:151): for ANY n_out % 32 == 0, both chunks are multiples of 32, chunk 2 >=
  chunk 1 >= 32, and the n_out >= 64 gate (gguf.py:149) rules out zero/under-length
  chunks. Verified by hand for n_out = 64, 96, 160, 24896.
- k-only reduction untouched. mul_mat_q accumulates per element
  (sum[i][j] += vec_dot(...) in a fixed k order) and writes dst[col_dst * nrows_dst +
  row_dst] guarded per element (mul_mat_q body, mmq.cuh ~400-470); the launcher splits
  the grid only along block_num_x = ceil(nrows_x / mmq_y) (mmq.cuh:478 launcher body)
  - no cross-row accumulation anywhere, so each output element's value is independent
  of which launch computes it.
- need_check is keyed on nrows_x % mmq_y (the N dimension), NOT on batch
  (mmq.cuh launcher body, def at :478; constants MMQ_Y_Q8_0 = 32 CUDA / 128 ROCm at
  mmq.cuh:434-441). The n_out % 32 == 0 gate therefore keeps need_check=false in both
  sub-launches - the same template as the unsplit launch. The M tail is handled
  identically in both launches because batch is the same: y-loads clamp
  (min(col_y_0 + ..., ncols_y - 1), mul_mat_q body) and dst writes guard
  (col_dst >= ncols_dst -> return; row_dst >= nrows_dst -> continue).
- Empirical pins at two geometries: production (K=4096, N=24896, M=256) bitwise
  (tests/kernels/test_gguf_quant.py:659-682) and small (K=512, N=64/96, M=64) bitwise
  (:685-716). No step in the split path can reorder reduction or re-quantize
  differently -> no BLOCKER.

### A2 - Scope gates completeness: CONFIRMED

- (a) Layout: every GGUFLinear carries the per-output-row [out, row_bytes] packing by
  construction (gguf.py:100), including fused projections (qkv / gate_up are one
  qweight, module doc gguf.py:7-13), so a dim-0 narrow is layout-safe for ALL _MMQ
  formats (q6_K/IQ3_XXS/IQ4_XS/Q3_K/Q4_K/Q4_0 included). Splitting on non-Q8_0 MMQ
  formats is intended scope (docstring "the dense MMQ GEMM"; gate qt in _MMQ,
  gguf.py:149) and exact by the same format-agnostic argument - but unreachable in
  production glm5next: the loader swaps map-branch linears to GGML_Q8_0 only
  (glm5_next/gguf.py:515 comment, :693); q6_K appears only on GGUFLMHead
  (glm5_next/gguf.py:706), which is never passed to the helper. No transposed or
  otherwise differently-packed GGUFLinear exists in the tree.
- (b) batch gate == dispatch predicate. Helper: n_tok <= _MMVQ_SAFE or qt not in _MMQ
  -> fallback (gguf.py:149). fused_mul_mat_gguf routes batch <= _MMVQ_SAFE and qt in
  _MMVQ -> vec kernel, else qt in _MMQ -> ggml_mul_mat_a8 (gguf.py:76-79). Since
  _MMQ == _MMVQ today (gguf.py:42-43, both = the same 7 formats) and _IQ_ONLY is empty
  (:48), the helper's gate is exactly the MMQ-dispatch condition; no divergence. If
  the sets ever diverge, the helper falls back to layer.forward = today's behavior
  (safe by construction).
- (c) The gate uses the GEMM's M, not a leading sequence dim. x = hidden_states is 2D
  [total_tokens, H] at the call site (kda.py:100 total = hidden_states.shape[0];
  downstream reshape(total, -1) at kda.py:189); the GEMM batch = X.sizes()[0]
  (gguf_kernel.cu:203). Decode passes [bsz, H] -> M = bsz. ctx.batch (request count)
  is never used as the gate. Decode bs > 6 legitimately splits (spec-intended).

### A3 - Output assembly: CONFIRMED

- (a) dtype = x.dtype == the plain path (Y is allocated with X.dtype,
  gguf_kernel.cu:205-206); both returns are contiguous [M, N]. Device: helper allocates
  on x.device, plain path on W.device (gguf_kernel.cu:205) - identical in serving
  (single GPU; a real device mismatch would fail the old path at the kernel call too).
- (b) Bias exactly once, after assembly (gguf.py:157-158), mirroring
  GGUFLinear.forward's single out + bias (gguf.py:108). The sub-launches call
  fused_mul_mat_gguf directly, which never applies bias - no double-apply. Production
  in_proj has_bias=False (kda.py:67).
- (c) Inference-only: the engine's serving entry points run under
  @torch.inference_mode() (engine/engine.py:582/589/1229/1407); torch.empty + copy_ is
  grad-safe and the extension kernels were never autograd-capable -> no new hazard.
  Transient peak = out (405 MB) + one sub-result (202 MB at production shape), each
  temporary freed after its copy_ statement.
- (d) Caller: proj is consumed by torch.split(proj, [conv_dim, h, d, d], dim=-1)
  (kda.py:117-119) - plain views of the fresh buffer; _write_track_snapshot copies the
  conv window out (kda.py:91-93). The helper's out aliases neither x nor qweight and is
  returned once, never mutated afterwards by the helper. Semantically equivalent to the
  old extension-allocated Y.

### A4 - Knob semantics vs the v3a pattern: CONFIRMED

- Per-call os.environ.get (gguf.py:116), default 1 (unset / "" / "1",
  gguf.py:117-118), "2" -> split (:119-120), strict whitelist raise (:121). The
  rejection set is exactly the required one: " 2" / "02" / "+2" / "12" / "0" / "-2" /
  "abc" all raise (no trim, no numeric parsing) and are pinned by
  test_kda_nsplit_env_knob (tests/kernels/test_gguf_quant.py:652-655).
- Single source of truth: the env is read only inside _kda_nsplit_env;
  kda.py:116 -> kda_in_proj_forward is the only production entry; GGUFLinear.forward
  contains no knob if. Project-wide search for kda_in_proj_forward and
  FREETOKEN_GGUF_KDA_NSPLIT hits only gguf.py, kda.py and the two test files. No
  bypass path and no double-apply (the helper calls fused_mul_mat_gguf, not itself).
- Boot baking: docstring gguf.py:134-137 and the call-site comment kda.py:111-113 both
  say do-not-flip-after-boot and name engine/graph.py GraphRunner._capture_graphs -
  which exists (engine/graph.py:149). Accurate: capture runs the model forward through
  this helper, so the per-call env read bakes the launch structure into the decode
  graphs; replay never re-reads the env.

### A5 - Decode/encode invariants: CONFIRMED

- batch <= 6 falls back to layer.forward = the exact pre-change code path; pinned by
  test_kda_nsplit_decode_batch_stays_mmvq (vec launch [24896] + torch.equal vs
  fused_mul_mat_gguf, tests/kernels/test_gguf_quant.py:719-734).
- Non-GGUF passthrough byte-identical: isinstance fallback (gguf.py:142-143), pinned
  at unit level (SimpleNamespace forward, :715) and op level
  (test_nsplit_knob_non_gguf_passthrough, tests/models/test_glm5_next_kda_op.py:261-296,
  torch.equal vs a re-instantiated op on the HF quant-method layer).
- No other consumer can reach the helper: production callers = kda.py:116 only.
  The other in_proj owners - qwen4_exp/gdn.py:149 and qwen3_5_moe/gdn.py:147 - call
  their own in_proj.forward directly (LinearColParallelMerged, layers/linear.py:80, not
  a GGUFLinear and never routed through the helper). No shared-expert or attention
  projection path touches it.
- Naming vs gating: the gate is generic and provably safe for ANY dense GGUFLinear
  (the bitwise argument is format- and consumer-agnostic), while name + docstring scope
  it to the kda in_proj ("the ONLY call site that may consume" - true today). The
  naming under-promises relative to the gate; behavior is safe. No finding - if a
  second dense layer ever wants the split, the docstring's "only call site" line is
  what would need updating, not the logic.

### A6 - Test adequacy: CONFIRMED, gaps named (all cheap, none blocking)

Covered: production-shape bitwise parity + knob=1 == lin.forward (:659-682); launch
structure [24896] -> [12448, 12448] via the lazy-module kernel spy (:599-627); boundary
math N=64 -> [32,32] and N=96 -> [32,64] bitwise, ragged N=48/32 fallback
byte-identical (:685-716); decode M=4 stays on vec kernels byte-identical (:719-734);
knob matrix incl. per-call flip and all 7 invalid values (:629-656); non-GGUF
passthrough at unit and op level.

Missing cases (structural argument verified for each above; listed so the gap is
visible):
1. Bitwise parity at batch > 6 that is NOT a multiple of mmq_x = 4 (e.g. M=7/13) - the
   gate admits it; M-tail handling is identical in both launches (write guards), so the
   argument holds, but no test pins it. Cheapest addition, recommended.
2. A non-Q8_0 _MMQ format (e.g. q6_K) through the split - argument format-agnostic,
   unreachable in production glm5next.
3. Non-contiguous x through the split path - the internal .contiguous()
   (gguf.py:60-63) is deterministic and shared by both launches; v2 added such a test
   for the grouped path, none here.
4. "" env value -> 1 (docstring claims it, gguf.py:117; unpinned).
5. Op-level GGUF integration through Glm5NextKDA with the knob=2 (needs model
   fixtures; the seam is covered by unit tests plus the source-verified loader swap
   glm5_next/gguf.py:686-701).

### A7 - Copy overhead estimate: CONFIRMED

Re-derivation at production shape (M=8128, N=24896, B=12448, bf16): the two copy_
writes move 2 x (M x B read + M x B write) = M x N x 2 B x 2 = 810.4 MB per call (the
brief's formula, checks out); plus the second launch re-reads x (M x K bf16 =
66.6 MB; re-quant is compute-only). ~877 MB/call x 34 calls/chunk = ~27.5-29.8
GB/chunk. At step0's 1638 GB/s effective DRAM bandwidth that is 16.8-18.2 ms/chunk;
the quoted ~15 ms assumes ~1.9-2.0 TB/s effective (pure-copy class) - same order,
fine. That is 0.10-0.12% of the 14.87 s chunk vs the conservative 0.24-0.28 s
(1.6-1.9%) recovery target: net-positive by > 14x. Correctly framed as nettable in
the A/B wave brief.

## 2. Findings (protocol format)

- **F1 NIT (comment precision, gguf.py:146-148)**: the gate comment says 32-alignment
  "keeps the unsplit launch's need_check=false template and its per-element reduction
  order". Two imprecisions: (a) need_check is keyed on nrows_x % mmq_y - the N
  dimension - not on batch, and it only adds bounds guards, so bitwise exactness holds
  for ANY split point, aligned or not; (b) "per-element reduction order" is unaffected
  by need_check. The gate's real purpose is performance (avoid the slow guarded
  template, zero/ragged chunks). Reword so a reader does not infer that exactness
  depends on the alignment. Non-blocking; the docstring's exactness paragraph
  (gguf.py:128-133) is accurate as written.
- **F2 NIT (test gaps)**: the A6 list - M=7/13 bitwise parity, a q6_K split case,
  non-contiguous x, "" env value, op-level GGUF integration. Optional cheap hardening
  for the A/B wave; none hides a plausible bug given the structural argument.
- **F3 NIT (ROCm portability note, no action)**: on a ROCm build mmq_y = 128
  (mmq.cuh:434-436), so n_out % 32 == 0 would not guarantee mmq_y-aligned chunks -
  sub-launches would run need_check=true (correct, slower) and the ">= one full tile
  per chunk" rationale is CUDA-specific. CUDA-only campaign; recording for awareness.

No BLOCKER, no MAJOR. No FP needed.

## 3. Overall verdict

**SHIP.** The implementation matches the Review-A C7-verified spec on every audited
axis: the split is bitwise-exact by construction and by source verification for the
whole gated domain (not only the pinned production shape); the gates mirror the
actual MMQ dispatch predicate; the assembly (preallocated out + two narrow copy_
writes + single bias) is semantically identical to the plain path; the knob is the
v3a pattern with the exact rejection set and a truthful do-not-flip-after-boot
rationale; decode and non-GGUF paths are byte-identical and pinned; the copy
overhead re-derivation confirms it is negligible against the 0.24-0.28 s target.
Findings are three NITs (F1 comment precision - the only one worth a one-line
reword before commit; F2 optional test additions; F3 informational). Nothing blocks
the A/B wave (two boots, env unset vs =2, MTILE=32 anchor, T43/T44/T46 discipline).
