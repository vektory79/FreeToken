# Adaptation log: merge main -> vektory79

Repo: /media/ai/src/FreeToken
Branch: vektory79 (pre-merge HEAD 6e673ab), merged main = afd99cb (== origin/main), merge-base 0ffd5c8.
Merge commit: c8d514054eac6f397182c521dc790e2aae7219c1 "Merge branch 'main' into vektory79"
(parents: 6e673ab + afd99cb; the branch's 8 true merge commits stay intact by construction).
Backup: branch `backup/vektory79-pre-merge-6e673ab` at 6e673ab.

Discovery predicted 26 conflict-candidate files; git auto-merged 18 of them cleanly and
stopped on 8 files with 16 conflict hunks. Resolution rule (user directive): if main
duplicates branch functionality, main wins; otherwise preserve branch intent and integrate
main (standard merge semantics).

## Conflicted files and how they were resolved

### 1. docs/models.md (1 hunk)
- Divergence: merge-base ended the Notes section with "- Multimodal checkpoints are
  served text-only."; branch kept it and appended an fp8 KV-cache bullet
  (`--kv-cache-dtype fp8 ...`, from 03fb043); main DELETED the text-only bullet
  (image-input wave 84d236c/3faef36/63d6471/68a81ff/08d728d/8ff0cce now serves image
  input, so the claim was obsolete on main).
- Resolution: drop "Multimodal checkpoints are served text-only." (main's newer state),
  keep the branch's fp8 bullet (unique branch content, absent in main).
- Classification: main wins for the duplicated/obsolete line; branch content kept
  elsewhere.

### 2. python/freetoken/models/glm5_next/__init__.py (1 hunk)
- Divergence: branch (4162f7c/4443d7e/2637714/f801a08) imports parse_config + 4 gguf
  helpers + Glm5NextForCausalLM; main (vision wave) imports VisionConfig,
  parse_vision_config, Glm5NextForConditionalGeneration, Glm5NextVisionModel.
- Resolution: union of both import blocks (no duplicate functionality - gguf dispatch is
  branch-only, vision is main-only). The trailing `__all__` merged automatically and
  already contained every name from both sides.

### 3. python/freetoken/engine/graph.py (2 hunks)
- Divergence: branch (f801a08) widened `moe_offload_cache` to
  `OffloadMoeCache | list[OffloadMoeCache] | None` and normalizes it into
  `self.moe_offload_caches` (per-signature partitions); main added the `mrope: bool`
  parameter plus `self.mrope` (used at capture time for mrope_positions buffers).
- Resolution: keep branch's widened signature AND main's `mrope` param. In the body keep
  branch's list normalization; drop main's `self.moe_offload_cache = ...` attribute -
  on main it existed only to feed `_reset_moe_offload_cache`, which the branch's
  list-based `_reset_moe_offload_cache` supersedes (generalization of the same logic).
- Classification: branch generalizes main's single-cache attribute (branch intent wins
  for that attribute); main's mrope added alongside.

### 4. python/freetoken/kvcache/__init__.py (1 hunk)
- Divergence: branch (03fb043) threads `kv_quant=kv_quant` (+ explanatory comment) into
  the QSA pool factory; main threads `mrope=model_config.model_is_mrope`.
- Resolution: union - both kwargs kept, comment kept.

### 5. python/freetoken/kvcache/qsa_pool.py (1 hunk)
- Divergence: branch added `kv_quant: str = "none"` ctor param (fp8/nvfp4 validation);
  main added `mrope: bool = False` (3-axis rope positions slab, `self._mrope`).
- Resolution: union - both ctor params. Bodies merged automatically (both sides' logic
  coexists: kv_quant validation + _mrope slab sizing).

### 6. python/freetoken/attention/triton.py (1 hunk)
- Divergence: branch's backend forwards fp8/nvfp4 scale kwargs to the extend kernel
  (`k_scale`, `v_scale`, `kv_quant`, `k_block_scale`, `v_block_scale`); main forwards
  `block_ends` (multimodal bidirectional blocks).
- Resolution: union - all six kwargs forwarded.

### 7. python/freetoken/kernel/triton/attention.py (6 hunks)
- Divergence: branch (03fb043/2f554c9/73ca76c/05861fb) added fp8/nvfp4 KV decode to the
  extend kernels (HAS_KV_SCALE/KV_NVFP4 constexprs, scale/block-scale pointer args,
  per-slot dequant); main (mm wave) added block_ends/HAS_BLOCKS (bidirectional span
  keys for image tokens). Both touched the same kernel signatures and launch sites.
- Resolution: union everywhere -
  * both extend kernels' constexpr lists get HAS_KV_SCALE, KV_NVFP4, HAS_BLOCKS;
  * `extend_paged_attention` signature gets branch's 5 scale params AND main's
    `block_ends`; docstring merged (fp8 sentence + block_ends sentence);
  * body keeps branch's scale/scale-shape prep AND main's `block_ends_arg` fallback;
  * both kernel launches get all three constexpr kwargs.
  The kernel BODIES (dequant guards + block-mask logic) merged automatically and use
  both flag sets, which confirmed the union resolution was the intended shape.
- Post-merge fix (working tree, amend): the first edit pass dropped `k_scale`,
  `v_scale`, `kv_quant` from the `extend_paged_attention` signature (edit-tool line
  matching artifact) while the body still referenced them -> NameError in 15 extend
  tests. Re-inserted the three params after `v_extend`. No semantic decision here,
  mechanical repair.
- Test adaptation (see Duplicates/Other adaptations): main's block_ends backend test
  stub gained the fp8-era `k_scale`/`v_scale` None answers.

### 8. python/freetoken/engine/engine.py (3 hunks)
- Divergence hunk A/B (both GraphRunner construction sites, startup + resize):
  branch passes `moe_offload_cache=self.moe_offload_caches or self.moe_offload_cache`
  (list-or-single, gguf wave); main passes `moe_offload_cache=self.moe_offload_cache`
  plus `mrope=config.model_config.model_is_mrope`.
  Resolution: union - branch's list-or-single expression + main's mrope kwarg.
  (engine.py defines both attributes at line 408/409; the list is the branch's
  per-partition generalization, the single attribute stays as the dominant-cache
  pointer used by legacy readers.)
- Divergence hunk C (moe-strategy auto -> hybrid upgrade guard):
  branch (gguf wave, 78115df..63b9bff) guards `bench_fmt != "gguf"` - gguf gets no
  automatic hybrid upgrade (explicit --moe-strategy hybrid only); main (bea8d06, #445)
  guards `default_backend == "offload" and not unified_memory` - unified-memory GPUs
  (GB10) resolve to fused instead and must not be upgraded to hybrid.
  Resolution: union guard - all three conditions AND-ed:
  `default_backend == "offload" and not unified_memory and bench_fmt != "gguf" and
  load_backend_recommendation(...) == "hybrid"`.
  Both protections are independent (gguf gating vs unified-memory gating); nothing
  duplicated. Verified against moe/bench_profile.py:31-34 which documents that
  engine.py excludes gguf from the auto upgrade by design.

## Duplicates resolved in favor of main

Two cases (post-review correction: this log originally claimed exactly one):

1. docs/models.md - "Multimodal checkpoints are served text-only." (branch-side
   continuation of a merge-base line) dropped in favor of main's removal: main's
   image-input feature (commits 08d728d, 84d236c, 3faef36, 63d6471, 68a81ff,
   8ff0cce, db64879) superseded the text-only statement. Keeping it would have made
   the merged docs factually wrong for every model main now serves with image input.
2. models/register.py sidecar fetch (post-review triage A4): the branch's
   checkpoint_quant_config fetched hf_quant_config.json and folded it into the
   checkpoint's quantization config; main re-implemented the same functionality as
   utils/hf.py:_merge_sidecar_quantization_config (read inside _load_hf_config;
   config.json wins when it already has quantization_config). git auto-merged this
   file and the branch-side fetch fell away. Resolution correct per the duplicate
   rule (main wins); the surviving code needs no change - only this log entry.

No branch code feature was discarded as a duplicate of main. The gguf/glm5next wave,
the fp8/nvfp4 KV wave and the per-partition CPU MoE executor wave are all absent from
main and were kept in full; main's multimodal image-input pipeline and mrope support
are absent from the branch and were taken in full.

## Other adaptations (merge-semantics integration, not duplicates)

1. tests/kernels/test_triton_attention.py - main's
   `test_triton_backend_applies_the_batch_block_ends_only_when_the_spec_asks` FakeKVCache
   stub had no `k_scale`/`v_scale`; the merged backend (branch fp8 contract) reads them
   on every forward. Added the branch-stub pattern (two methods returning None, with the
   branch's own comment) to main's stub. Main's test intent (block_ends behavior)
   unchanged.
2. tests/kvcache/test_qsa_pool_fp8.py - branch test stubs (`_config()` and
   `test_factory_threads_kv_quant_into_the_qsa_pool`) built SimpleNamespace model
   configs without `model_is_mrope`; main's QSA cost model (qsa_pool.py:195) and QSA
   factory (kvcache/__init__.py:239) now read it. Added `model_is_mrope=False` to both
   stubs. Branch test intent (fp8 pool behavior) unchanged.

## NEEDS-USER-REVIEW

None. Every hunk resolved deterministically by the union/duplicate rules above; no
design decision had to be guessed beyond the documented rules.

## Post-merge integration fixes (amended into the merge commit)

1. kernel/triton/attention.py - first edit pass had dropped k_scale/v_scale/kv_quant
   from the extend_paged_attention signature (tool line-matching artifact); 15 extend
   tests failed with NameError. Re-inserted the three params after v_extend. Detected
   by targeted run 1, fixed before finalization.
2. graph.py moe_offload_cache normalization - the extended suite surfaced
   tests/moe/test_offload.py::test_graph_capture_reuses_warm_offload_cache_before_capture
   failing with TypeError ('FakeOffloadCache' object is not iterable). Triage: the
   failure reproduces byte-identically on pre-merge HEAD 6e673ab (verified in a scratch
   worktree at /tmp/ft-premerge-6e673ab, since removed) - the branch's own gguf wave
   (f801a08) introduced the isinstance-gated normalization after the 09-12 baseline and
   broke this merge-base-era test; it was already broken on the branch before the merge.
   Fix (inside the merge commit): single non-list value is wrapped duck-typed
   (`[moe_offload_cache]`), None -> [], list -> copy. Production semantics unchanged
   (engine passes real caches or lists); test doubles work again.
   Post-fix: tests/moe/test_offload.py 24 passed, tests/engine/test_cache_budget.py +
   tests/moe/test_hybrid_fetch.py 53 passed. Amended via `git commit --amend`
   (merge parents preserved). Final merge commit: 380e9eb9ba1c58697471980f94a7874c770a9893.

## Post-review fixes (second amend, same merge commit)

Reviews A (A1-A4) and B (B1-B4 + gaps G1-G6) were triaged against the tree at
380e9eb; full verdicts with code-level grounding in findings-triage.md. Confirmed
items fixed here:

1. docs/models.md (A1) - removed the dangling fp8 bullet: the merge auto-resolution
   kept only the first line of the branch's 7-line bullet, ending mid-sentence on a
   comma. All unique branch facts live in docs/cli.md #fp8-kv-cache (canonical
   location, main's structure); the branch's model-name enumeration is not copied
   over - it would duplicate models.md's per-architecture content.
2. python/freetoken/server/args.py + docs/cli.md (A2) - help and the FP8 bullet
   claimed fp8 is "not MLA/DSA ... fails at startup", but the merged code accepts
   fp8 on MLA/DSA: engine.py:1919-1941 validates kv_quant against the type set
   {FULL, SWA, QSA, MLA, DSA}; kvcache/__init__.py threads kv_quant into the
   MLA/DSA pools; dsa_pool.py implements the fp8 path; the dsa attention backend
   declares supports_fp8_kv. Only the DSV4 tiered pool and block-sparse (BSA,
   MiniMax-M3) models reject a quantized KV cache. The contradiction was inherited
   from the branch (6e673ab carried both the accepting code and the stale help);
   the merge did not introduce it. Help + cli.md updated to the factual
   pool/backend matrix, including the nvfp4 sentence (the code accepts MLA/DSA
   for nvfp4 too, head_dim divisible by 16).
3. python/freetoken/engine/graph.py (A3/B2) - the duck-typing fix (kept: single
   non-list cache wraps as [x], None -> [], list -> copy) changed tuple/generator
   behavior vs the branch (those used to materialize via list(...)). Tuple added to
   the iterable branch: isinstance(moe_offload_cache, (list, tuple)) -> list(...).
   List/None/single semantics preserved; tests/moe/test_offload.py stays green
   (24 passed, it passes a single duck-typed fake). No caller passes generators
   (GraphRunner is constructed only at engine.py:523/:1344, list-or-single-or-None).
4. A4 documented, no code change (duplicates section, case 2). B3/B4 recorded as
   INFO and G1-G6 as follow-up test gaps in findings-triage.md; intentionally not
   fixed in this merge.

Merge commit re-amended in place (parents 6e673ab + afd99cb preserved; still
exactly one commit on top of the parents: `git rev-list --count HEAD --not
6e673ab afd99cb` == 1). The SHA quoted earlier in this file (c8d5140 / 380e9eb)
are earlier amend states; the finalized SHA is the branch HEAD after this amend.

## Post-merge integration checks

- No conflict markers left: grep over python/, tests/, docs/ -> 0 hits.
- py_compile passes on all 7 touched python files.
- All GraphRunner construction sites updated (only engine.py, 2 sites).
- Only production caller of extend_paged_attention is attention/triton.py:205, which
  forwards the full union kwarg set.
- bench_profile.py comment (lines 31-34) cross-checked against the moe-guard union:
  gguf exclusion from auto-hybrid is the branch's deliberate current behavior.
