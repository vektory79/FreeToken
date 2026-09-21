# Plan: Path A - native GGUF serving for glm5next (GLM-5.3-Flash-UD-Q3_K_XL)

Status: Phases 1-4 committed (4162f7c, 4443d7e, 2637714); Phase 5 DONE - offload
acceptance PASSED on hardware 2026-09-14 (UNCOMMITTED); Phase 0 WAIVED by the user
(private-use work). Phase 6 (full A/B) next.

## Goal

`ft serve --model-path /media/ai/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL.gguf`
boots and serves with native GGUF quants (no bf16 materialization of experts - RAM-bound:
Q3/IQ experts expand to ~300+ GiB vs 188 GB). Reference implementation:
`/media/ai/src/llama-cpp-glm5next/` (llama-arch.cpp, src/models/glm5next.cpp, llama-kv-cache-dsa.cpp, llama-memory-recurrent.cpp).

## Verified facts to build on

- Boot today dies in seconds: `ValueError: GGUF architecture 'glm5next' is not supported
  (known: ['gemma4'])` at `python/freetoken/models/gguf/config.py:65` (detection path
  `server/args.py:821 -> utils/hf.py:218 build_gguf_shim`). The GGUF shim layer exists;
  the arch whitelist is the blocker.
- Files: main gguf arch `glm5next`, 1412 tensors, gguf v3; mmproj-BF16.gguf (348
  tensors, arch `clip`) sits at /media/ai/models/unsloth/GLM-5.3-Flash-GGUF/, one level
  above the UD subdir - out of scope, text-only serving. Quant inventory: experts IQ4_XS x41
  (ffn_down_exps, 1.2 GiB each), IQ3_XXS x82 (gate/up exps), Q6_K x4 + Q4_K x1 +
  Q3_K x2 (down exps of select layers), Q8_0 x644 (dense), F32 x638 (norms/biases/
  ssm scalars). DSA indexer tensors present: `blk.N.indexer.{attn_k,attn_q_b,k_norm.*,
  proj}`, `indexer_compressor_{ape,gate}`. Experts are stacked per layer
  (`ffn_down_exps.weight` etc.), NOT one tensor per expert.
- llama.cpp reference: tensor graph 814 lines / exactly 56 LLM_TENSOR kinds in
  `src/models/glm5next.cpp`; KV + tensor-name maps are FLAT (no per-arch case blocks,
  every key is `glm5next.<name>`; `llama-arch.cpp:1084-1149` are capability switches);
  DSA in `src/llama-kv-cache-dsa.cpp`; recurrent KDA state in
  `src/llama-memory-recurrent.cpp`. HF-name cross-reference = the RedHatAI
  GLM-5.3-Flash-NVFP4 checkpoint already on disk (same model, tokenizer identical).
- FreeToken in-tree pieces: GGUF shim `models/gguf/config.py` (96 lines);
  gemma4 adapter pattern `models/gemma4/gguf.py` (474 lines: parse_gguf_config,
  iter_gguf_weights, load_q4_0_expert_sources); vendored CUDA gguf kernels
  `kernel/csrc/gguf/gguf_kernel.cu` (port of sgl-kernel, JIT via `kernel/gguf.py`)
  with Q2_K/Q3_K/Q4_0/Q4_K/Q5_0/Q5_K/Q6_K/Q8_0 + iq2_xxs/iq2_xs/iq2_s/iq3_xxs/iq3_s
  symbols; glm5_next family code under `models/glm5_next/` (HF/FTW only today).
- Known gaps (re-verified 2026-09-13): IQ4_XS IS in `gguf_kernel.cu` now - full dequant
  (dequantize.cuh:431/:534/:577), dense MMVQ (gguf_kernel.cu case 23 :179-181,
  mmvq.cuh:323), grouped moe_vec MMVQ (:781-782, moe_vec.cuh:378); iq formats lack ONLY
  MMQ paths (ggml_mul_mat_a8 ends case 14, MoE-MMQ :518, ggml_moe_get_block_size returns
  0 for iq). The real gap is python-side wiring: layers/gguf.py:33-40 exposes only
  Q4_0/Q8_0/Q6_K; dequant.py BLOCK_SHAPE/GGML_NAME lack iq/Q3_K/Q4_K entries. CPU GEMV
  (moe/cpu_executor.py _WFMT_IDS :72) = bf16/nvfp4/mxfp4_triton/ds_fp4/q4_0; fp8_block is
  GPU-only. Upstream gemma4 GGUF branch still raw (#358, #186 grid.z at tokens*top_k >
  65535 - GLM top_k=8 hits it at 8192-token chunks, #194 heterogeneous banks - this file
  IS heterogeneous per layer).

## Phases

### Phase 0 - upstream alignment [WAIVED by the user 2026-09-14]
- The user decided this work is private/local-only (no upstream publication planned),
  so the CONTRIBUTING issue-first rule does not apply here. The draft comment stays at
  phase0-issue34-comment-draft.md in case that ever changes. See memory
  ft-gguf-glm5next-private-scope.

### Phase 1 - config shim for glm5next [DONE 2026-09-13, UNCOMMITTED]
- Implemented: GGUF_ARCH_TO_REGISTRY gets "glm5next": "Glm5NextGGUFForCausalLM"
  (models/gguf/config.py); spec "Glm5NextGGUFForCausalLM" in models/register.py; AOT
  arch_aliases claim at kernel/aot_models.py:287 (required by the registry invariant
  test); parse_gguf_config in NEW models/glm5_next/gguf.py - translates glm5next.*
  keys into a text-config namespace and delegates to glm5_next.config.parse_config
  (bit-identical to the HF path); iter_gguf_weights = Phase-2 stub
  (NotImplementedError).
- Verified: probe on the real gguf 20/20; resolved Glm5NextArgs field-identical to
  RedHatAI NVFP4 config.json load_args except inert indexer_rope_interleave (shim
  False vs HF True - no glm5_next consumer today; reconcile at Phase 2 indexer wiring).
- Tests: tests/models/test_glm5_next_gguf.py 33 passed (synthetic gguf-py fixture,
  real 147 GB file never opened); invariant tests/models/qwen4_exp/test_weight.py
  30 passed after the AOT claim; neighborhood 15 passed 1 skipped.
- Quality loop: Review-A + Review-B (10 findings, all TP, fixed) then delta review
  (6 findings, all TP, fixed: AOT claim, negative nextn gate, scalar-cause message,
  topk/absent-key tests, shexp comment). Cycles used: 2 of 3.
- Key-table deviation (justified): indexer_types = ("full",)*num_layers not
  per-DSA-count, because glm5_next/config.py:111 + attention/dsa.py:154 index by LAYER
  id; HF checkpoint also carries per-layer entries.
- Guard philosophy: fail loudly on variant quants that would silently mis-serve
  (gating != sigmoid-2, rope != 0, eps != 1e-6 hard-fail (recommended by delta review:
  no code path to honor another eps), kpool/topk <= 0, dense bounds, nextn range).

### Phase 2 - tensor-name translator + iter_gguf_weights [DONE 2026-09-13, UNCOMMITTED]
- Implemented in models/glm5_next/gguf.py (~600 lines): module maps (_COMMON/_KDA/
  _DSA/_DENSE_FFN/_MOE, casts qw/bf16/fp32 per weight.py dtypes), fusion slot tables
  (_IN_PROJ_SLOTS q|k|v|b|f_a|g_a, _CONV_SLOTS q|k|v, _KV_B_SLOTS), iter_gguf_weights
  (1008 yields from 1383 gguf rows; unknown tensor -> ValueError; blk.45 29 MTP tensors
  skipped; dtype/type per tensor via logger.debug) and iter_gguf_expert_sources (lazy
  (ckpt_layer, {gate,up,down: GgufTensor}), ggml_type intact; yields CHECKPOINT layer
  ids 3..44 - Phase 5 must offset via bank_layer_of, moe/expert_pieces.py:27).
  dequant.py: reference dequant_q8_0 added + 4 log-only GGML_NAME entries (IQ3_XXS/
  IQ4_XS/Q3_K/Q4_K; NO dequant for them - banks pass through packed).
- Fusions consumer-verified: in_proj 24896 rows [p,p,p,h,d,d] (the map's 24832 was
  wrong - it dropped ssm_beta's 64 rows; map corrected); conv1d (24576,1,4); kv_b =
  attn_k_b last-two-axes transpose + attn_v_b -> (32768,512), proven vs
  attention.py:141-152 twice; A_log = log(-ssm_a) fp32 round-trip verified; kv_b
  fusion necessarily dequantizes its two Q8_0 pieces (pack-axis crossing); expert
  banks never dequantize in the iterator.
- Verified: probe on the real file 17/17 (yielded multiset == derived, zero unmapped
  over 1412); tests/models/test_glm5_next_gguf.py 51 passed; neighborhood 45 passed
  1 skipped; config probe 20/20. Quality: Review-A + Review-B -> 8 TP fixed; delta
  review verified + 3 small TPs fixed; 2 of 3 cycles used.
- Phase 4 entry criteria (review notes): glm5_next has NO is_gguf_model /
  convert_*_to_gguf analog (gemma4/model.py:120-123 precedent) - .qweight yields
  assume the Phase 4 module swap; weight.py:245 hook load_q4_0_expert_sources absent
  (Phase 5); this file is UNTIED (output.weight Q6_K present - no GGUFTiedLMHead);

### Phase 3 - tokenizer [DONE 2026-09-13, UNCOMMITTED - convention CORRECTED]
- VERIFIED: the sibling-file option written here earlier is DEAD in reality - for .gguf
  paths the loader reads the EMBEDDED vocab only (utils/hf.py:32 load_tokenizer ->
  models/gguf/tokenizer.py:19-44 convert_gguf_tokenizer; sibling tokenizer.json is NOT
  read; reader.py:31-33 states embedded-vocab is THE convention). --tokenizer-path
  REJECTED (3 call sites).
- Implemented: _TOKENIZER_ARCH["glm5next"] = "gpt2" (tokenizer.py:15-19; file declares
  tokenizer.ggml.model=gpt2, pre=glm4; GGUFGPTConverter plain BPE; qwen2 rejected -
  hardcoded qwen AddedTokens). Embedded vocab complete (154880 tokens, merges 321649,
  chat_template 10648 chars) - no user action needed.
- Delta review caught TWO serve-blocking bugs (the first cut's "serving unaffected"
  claim was empirically false - serving re-encodes rendered template TEXT,
  tokenize.py:110-117): (1) specials byte-split -> FIXED arch-agnostically: after
  convert, register token_type CONTROL(3)/USER_DEFINED(4) tokens as atomic AddedTokens
  (tokenizer.py:36-49; gemma4 gains atomicity too, ids unchanged); (2) eos set was
  {154820} only but the model ends turns with <|user|> (reference eos [154820, 154827,
  154829]) -> FIXED: glm5next branch unions eom/eot keys (tokenizer.py:82-88). Both have
  failing-before/passing-after assertions in the env-gated test
  (FREETOKEN_GLM5NEXT_TOKENIZER_REF).
- Round-trip verdict (real run vs RedHatAI tokenizer.json): prose EN/RU/CJK identical
  (strict); code DIVERGES (gpt2 converter ignores pre=glm4 regex-split - pinned with
  concrete sequences); boundary `<|user|>` atomic after the registration fix; decode
  lossless everywhere.
- Tests: env-set 58 passed / env unset 57 passed 1 skipped; kernels+engine-gate 48
  passed; neighborhood 45 passed 1 skipped. Quality: delta review (2 Errors + 2 test
  issues, all fixed with before/after proofs); 2 of 3 cycles.

### Phase 4 - kernels (rescoped: python wiring) [DONE 2026-09-13, UNCOMMITTED]
- Implemented: layers/gguf.py dispatch extended (_MMVQ += Q3_K/Q4_K/IQ3_XXS/IQ4_XS;
  _MMQ/_DEQUANT += Q3_K/Q4_K only - csrc mul_mat_a8/moe_a8 exclude iq; _IQ_ONLY -> loud
  NotImplementedError for dense iq above the MMVQ cutoff; gemma4 paths bit-identical,
  additive diff); dequant.py BLOCK_SHAPE Q3_K(256,110)/Q4_K(256,144)/IQ3_XXS(256,98)/
  IQ4_XS(256,136), geometry-only (== gguf-py GGML_QUANT_SIZES); glm5_next convert path
  (moe_weight_format="gguf" marker, is_gguf_model, GGUFLMHead untied with ParallelLMHead
  prefill slicing, convert_glm5_next_to_gguf swapping exactly the .qweight set - kv_b/
  conv1d/hc/indexer/router/bias untouched) + model.py hook (gemma4/model.py:120-123
  precedent); per-target packed-type guards (token_embd Q8_0, output Q6_K, linears Q8_0
  - variant files fail at ITERATION time, never as late loader shape asserts); engine
  gate: gguf moe_weight_format + cpu/hybrid/fused strategy (or moe_cpu_layers) ->
  ValueError at config time; offload is the only wired expert path until Phase 5.
- Kernel parity MET on GPU (user installed clang++; JIT build succeeds):
  tests/kernels/test_gguf_quant.py 21 passed, zero warnings. ACCEPTANCE ADJUSTED:
  bit-parity vs fp32 gguf-py is mathematically impossible for the dfloat kernel
  (ggml-common.h:930 half chain; compiler contracts __hmul/__hsub) - the implemented
  contract is a per-element term-scaled bound 8*2^-11*factor*|d| + 1e-3 with per-format
  _DEQUANT_FACTOR (decode bugs shift O(factor*d), 256x above the bound -> still loud).
  Q4_K kernel root-caused CORRECT (element-for-element vs llama.cpp scalar AND gguf-py;
  the fp32 oracles were the wrong precision contract). Test scales bounded by
  construction (_FP16_SCALE_FIELDS, scales [0.25,2]).
- Toolchain facts: kernel/gguf.py:25-40 requires clang++ host (env FREETOKEN_GGUF_HOST_CXX
  override); gcc-13/15 trip the ATen List_inl.h conformance error with nvcc;
  -allow-unsupported-compiler does NOT help; nvcc 13.3 (user's /usr/local/cuda-13.3)
  resolves via /usr/local/cuda and is compatible with torch cu130; clang 18 default OK.
- Quality: Review-A/B (9 TP fixed) + delta review (3 TP fixed: per-target guards,
  engine gate test incl. 'fused', C4 type-hint FP with evidence - moe_cpu_layers IS a
  string spec). 2 of 3 cycles. CPU GEMV skipped by the offload-first decision.
- Phase 5 entry notes: _expert_gemm (layers/moe.py:409-430) asserts q4_0-only - extend
  for the bank formats; wire iter_gguf_expert_sources (ckpt layer ids 3..44, offset via
  bank_layer_of, moe/expert_pieces.py:27) -> registry -> _expert_gemm; benchbw "gguf"
  profile + exclude it from the engine.py:1562 auto->hybrid upgrade; glm5_next analog of
  the weight.py:245 expert-sources hook; offload-cache tag.

### Phase 5 - expert banks + offload integration [DONE 2026-09-14, UNCOMMITTED]
- Implemented: load_gguf_expert_sources hook chain (weight.py resolver, gemma4
  pattern); banks materialized into pinned HostBanks + PinPipeline (THE pin-count
  trap: LayerCompletionTracker expected_per_layer must equal the NOTE count -
  glm5next 1/layer, gemma4 2/write - mismatch silently skips ALL pinning -> CUDA IMA
  at the first decode copy; root-caused on hardware); _expert_gemm gguf branch
  (per-projection dispatch via cache.gguf_types, gated_act_and_mul with the layer's
  activation/alpha/limit, tokens/top_k wiring); engine-level per-signature
  OffloadMoeCache partitions (grouping by role-ordered (shape,dtype) signature,
  byte-proportional budget split with E floors + auto byte-envelope, local layer-id
  remap proven by id()-identity, graph.py plural resets); per-partition
  prefill_overlap DEGRADE (>= 2E only; minorities synchronous - NOT 2E flooring);
  #186 clamp 8192->8191; benchbw "gguf" profile + auto->hybrid exclusion; aggregated
  readouts; load-time MMVQ-set validation; _build_fused_copy_plan construction
  guards (pageable banks / widths / geometry).
- Acceptance PASSED on hardware (run 5, real 147 GB file): boot 2m40s -> /ready 200;
  histogram {(18,18,14):2, (18,18,23):39, (23,23,14):1}; minorities [8]/[9,41] 288
  slots overlap OFF; 3 CUDA graphs captured; prefill 4213 tok / 65s; decode 14.3
  tok/s cached (TTFT 4.13s) / 15.6 fresh. Suites: 25/102/21/146 (+tokenizer 57-58).
- Boot-smoke chronology (4 failing runs before PASS, each evidence-backed): 2E-overlap
  assert -> per-partition degrade; graph.py NameError -> import; CUDA IMA in
  fast_index_copy_multi_jit -> pin-count trap (the stale-tree hypothesis was falsified
  by run 4 reproducing byte-for-byte); KDA autotuner OOM = eager-tail artifact (the
  capture config fits). Full logs: verification/boot-smoke-server-log-2026-09-14*.txt.
- Deferred resolution (2026-09-14):
  * exact per-layer sizing - DONE (header-only scan via _gguf_bank_bytes, wired into
    bank_bytes_estimate with model_path at both production call sites; real-distribution
    pin 134,404,374,528 B vs conservative 160,922,861,568; no-path fallback kept;
    benchbw untouched). UNCOMMITTED.
  * cross-group prefetch hop - HONEST SKIP with hardware evidence: in the tuned plan
    (1231 slots -> 460/288/288) ALL partitions are overlap-degraded (< 2E), so NO
    boundary choreography runs at all - the hop would be dead code; minorities are
    floored at 288 and would need ~10-20x the byte envelope to reach 2E (impossible on
    32 GiB); an overlap->overlap boundary is cleanly implementable (~1 layer-copy per
    boundary) but unreachable on this card. Contract-pinning test added
    (test_prefill_choreography_group_boundary_no_hop_contract). Hardware probes
    252-256 tok/s vs 264.1 baseline = noise (no code changed). UNCOMMITTED (test only).
  * cache_status minority under-report - was ALREADY fixed in f801a08 (aggregated
    readouts verified); this wave fixed 2 silently-broken test fixtures from that
    commit + added 3 partition tests (19 passed, was 16 with 2 failing); dominant-only
    fallback extracted to _moe_offload_caches helper. UNCOMMITTED.
  * clang requirement docs - added to docs/install.md ("GGUF model kernels" section:
    clang++ on PATH, FREETOKEN_GGUF_HOST_CXX override, gcc trips ATen List_inl.h under
    nvcc, -allow-unsupported-compiler does not help, first JIT build takes minutes).
    UNCOMMITTED.
  * kernel-level assert for non-contiguous activations - STAYS DEFERRED by vendor
    discipline (csrc files are verbatim sgl-kernel ports; the python-side guard in
    fused_mul_mat_gguf covers the entry point).
- Phase 6 notes: quality A/B vs NVFP4 must compare DECODED TEXT (code-tokenization
  divergence); auto-KV ~9k tokens - budget tuning needed for the 64k-fill prefill
  probe; 8k prefill + 768 decode = 8959 tokens fits the 9024-token KV barely.

### Phase 6 - E2E bring-up + numbers [DONE 2026-09-14, measurement only]
- Tuned GGUF command (64k-capable): base + `--num-tokens 70016 --memory-ratio 0.85
  --max-prefill-length 4096`. Tuning trail: auto gives ~8-9k KV only; --num-tokens must
  be a page-64 multiple; explicit --moe-cache-size 864 FAIL-FASTS (per-signature floors
  demand >= 8510 slots - explicit sizing cannot shrink below auto; the working lever is
  auto slots + capped KV + ratio 0.85).
- A/B (same prompts, 4096 chunks, both configs measured 2026-09-14; artifacts in
  verification/phase6/): boot GGUF 115s vs NVFP4 56s. Prefill 64k fill: GGUF 263.9-268.1
  vs NVFP4 510.3-515.2 tok/s = 0.52x. Decode cached @64k: GGUF 14.67 tok/s (TTFT 3.29s)
  vs NVFP4 14.08 (TTFT 5.12s) - parity-or-better. 768-token run: both stopped early on
  natural EOS (GGUF 363 @ 14.28; NVFP4 232 @ 14.16). Plans: GGUF 1227 slots + 70,016 KV
  (0.78 GiB), overlap OFF on ALL partitions (464/288/288 < 576 -> synchronous prefill
  everywhere); NVFP4 322 slots + 526,784 KV. VRAM 28.8 vs 29.4 GiB; GGUF RAM 137/188
  GiB (pinned banks).
- BASELINE DEVIATION: the canonical NVFP4 1M-reserve command FAIL-FASTS on today's VRAM
  (min plan 8.19 GiB > budget 6.63; booted fine on 09-12) - baseline measured with
  --kv-reserve-tokens 524288 (322 slots < 336 working set -> NVFP4 decode 14.1 vs
  historical 15.13; same-session A/B unaffected).
- BLOCKING FINDING - chat quality: 24-prompt greedy battery, DECODED-TEXT comparison:
  0/24 identical, divergence at char 0 in 19/24, ON-TOPIC GGUF 3/24 vs NVFP4 24/24.
  GGUF answers a DIFFERENT task (GUI-agent JSON, image math, bounding boxes) - the user
  prompt effectively never reaches the model; deterministic across boots.
- RESOLUTION (same day, live per-layer bisect + root-cause): the defect was NOT the
  rendering path (tokenizer/vocab/template/embeddings/layer-order/output-head ALL
  re-exonerated; MoE exonerated three ways incl. gguf-py reference IQ dequant). ROOT
  CAUSE: fused_mul_mat_gguf (layers/gguf.py) fed NON-CONTIGUOUS activations - f_a/g_a
  from the in_proj split (glm5_next/kda.py:94-96) are views with stride (24896,1), and
  ggml_mul_mat_a8/vec_a8 assume row-major -> garbage gates -> KDA state decay corrupted
  in all 34 KDA layers from L0 -> context loss. FIX: +4 lines (normalize non-contiguous
  x in the dispatcher) + regression test test_dense_gguf_noncontig_activations[4|27]
  (FAILING-before/PASSING-after). VERIFIED: live battery 5/5 on-topic vs 0/5 pre-fix
  (incl. the exact BANANA regression). BONUS: residual-norm explosion 8.9->40k is
  architecture-normal mHC growth (both models).
- CLOSURE (commit 028f2d9): post-fix 24-prompt battery ON-TOPIC 20/24 (pre-fix 3/24,
  NVFP4 24/24); divergence moved to chars 49-339 = quantization-level wording drift;
  perf parity (prefill -0.5%, decode noise; the guard costs nothing); boot 91s.
- DIGIT-SPLIT RESOLVED (same day): root = GGUFGPTConverter's GPT-2 ByteLevel regex glued
  digits to the preceding space/token ("in 100 days" -> "in [blank] days"). The reference
  pre_tokenizer is Sequence(Split Regex "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\
  \p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+" Isolated) +
  ByteLevel(add_prefix_space=false) - digits chunk 1-3 chars. Ported natively in
  models/gguf/tokenizer.py (_GLM4_SPLIT_REGEX + _glm4_pre_tokenizer, glm5next-gated;
  gemma4 untouched). LIVE BATTERY 6/6 PASS with correct arithmetic (80 km/h, 100 mod 7
  -> Friday, 2^10=1024, 2+2=4, prose on-topic; zero placeholder signatures). ALL battery
  defects closed - full NVFP4-parity chat quality on the digit/code axis too.

## Estimation

- Phase 0-3: 2-4 days (mechanical, reference-backed).
- Phase 4: 2-5 days (down from 1-2 weeks: IQ4_XS CUDA already in-tree; python dispatch
  wiring + kernel parity tests are the work; MMQ-for-IQ optional).
- Phase 5: ~1 week (heterogeneous banks + benchbw + graphs).
- Phase 6: 1-2 days.
Total: ~3-6 weeks part-time. The pragmatic intermediate milestone is
"offload-only E2E" after Phases 0-4.1 (no CPU GEMV, no hybrid) - that alone makes the
file servable.

## First-session starter checklist

1. [DONE 2026-09-13] Re-verify anchors -> research/anchor-verification.md (IQ4_XS drift
   found and folded back into this plan).
2. [DONE] Dump tensors.txt / tensors-mmproj.txt / metadata.txt (1412 + 348 tensors).
3. [DONE] 3-way name diff -> research/phase2-tensor-map.md (1383 mapped + 29 skip-listed
   MTP blk.45, 0 unmatched both ways; constructed targets: in_proj 6-way concat
   q|k|v|b|f_a|g_a, conv1d 3-way concat, kv_b_proj = attn_k_b transpose + attn_v_b,
   A_log = log(-ssm_a), expert banks sliced dim-0 from ffn_*_exps stacks).
4. [DONE] Phase 1 key table -> research/phase1-config-keys.md; tensor kinds ->
   research/llamacpp-tensor-kinds.md.
5. [DECIDED] Offload-first for the first E2E milestone (decode_target=offload, no CPU
   GEMV; hybrid/CPU-GEMV is a follow-up). ALL PHASES COMMITTED: 4162f7c, 4443d7e,
   2637714, f801a08, 028f2d9, 78115df (post-campaign polish). CAMPAIGN COMPLETE:
   production-usable for private chat (full NVFP4-parity quality, prefill 264 tok/s,
   decode 13.8-15.6 tok/s). Only remaining deferred: kernel-level assert (vendor
   discipline).

## Research artifacts (2026-09-13/14)

- research/anchor-verification.md - in-tree anchor re-verification, file:line evidence
- tensors.txt / tensors-mmproj.txt / metadata.txt - full gguf inventory (1412 + 348 tensors)
- research/phase1-config-keys.md - config key table for the Phase 1 shim
- research/llamacpp-tensor-kinds.md - 56 llama.cpp LLM_TENSOR kinds, resolved names
- research/phase2-tensor-map.md - 3-way name mapping (Phase 2 core artifact)
- verification/boot-smoke-server-log-2026-09-14*.txt + boot-smoke-report - Phase 5
  acceptance runs 1-5 (4 documented failures then PASS on hardware)
- phase0-issue34-comment-draft.md - draft comment for upstream issue #34 (Phase 0
  WAIVED by the user; kept in case that changes)

## Hard rules from CONTRIBUTING (apply to every PR of this work)

- One change per PR, linked issue, hardware + checkpoint ID + exact command in the PR.
- Bug fixes come with a test that fails before and passes after; perf changes come with
  A/B numbers vs main.
- No pushes/PRs by agents; the user owns and can explain every line.