---
name: "ft-pr-relevance-glm-hybrid"
description: "PR verdicts GLM-5.3 hybrid: #300 kv-ladder no-gain, #339 no-op w/ sgl_kernel, #414/#439/#399 not applicable"
type: project
lastUpdated: 2026-09-13T02:18
lastRecall: 2026-09-15T01:49
---

# Upstream PR applicability for GLM-5.3-Flash-NVFP4 hybrid on RTX 5090 (researched 2026-09-12, branch vektory79 = main + #408 port)

Check here BEFORE porting any upstream perf PR; re-verify sgl_kernel presence before trusting the #339 verdict.

## PR #300 (OPEN) - Automatic KV/MoE Laddering
- What: `--enable-kv-ladder` (opt-in) + `--ladder-step-size` (default 32768). Startup KV >= 2x step; grows rungs by trading GPU-resident MoE slots for KV pages inside the measured budget; a request that could exceed the current capacity is held until the scheduler is idle, cache rebuilt (rollback-safe, ~0.7-1.0 s), then admitted. Restricted to TP=1, --max-running-requests 1, --moe-cache-auto (default-on), offload-family. Files: scheduler/kv_ladder.py (new), scheduler/scheduler.py, server/args.py, docs/cli.md.
- Claimed: up to +33% long decode (Qwen3.8-Flash-Next-NVFP4, RTX 5090, ratio 0.93: 65k KV/6425 slots -> 83.8 tok/s vs 262k KV/4627 slots -> 63.0 at 4096-token decode).
- Verdict for GLM-5.3-Flash-NVFP4: NO decode gain. Hybrid decode is slot-saturated here: working set = 42 MoE layers x top_k 8 = 336 pairs < 373 slots even at 1M nvfp4 KV (3.83 GiB). The +33% applies to models whose long-KV plan starves slots BELOW their working set. For this model the ladder is flexibility (boot small KV, grow on demand) at the cost of ~1 s rebuild stalls at rung crossings. Not ported.

## PR #339 (OPEN) - drop the D2H sync from the varlen GDN/KDA prefill conv
- The sync = int(seq_lens.max().item()) sizing the triton launch grid lives ONLY in the triton fallback of kernel/causal_conv1d.py. This venv HAS sgl_kernel -> fused sgl path runs -> sync never executes -> expected NO prefill gain on this box (patch also grants CUDA-graph capture legality, unused in serving).
- Shape: FLAMetadata gains max_seq_len=max(lens) (attention/linear.py build_fla_metadata); optional kwarg on causal_conv1d_varlen; passed from glm5_next/kda.py, qwen3_5_moe/gdn.py, qwen4_exp/gdn.py. `gh pr diff 339 --repo FlashML-org/FreeToken` applies clean on vektory79 (git apply --check OK).

## PR #414 (OPEN) - batch-memcpy probe stream-order fix
- Fixes the torch.zeros-then-copy probe race in kernel/batch_memcpy.py:39: on a busy current stream the probe reads zeros -> load_batch_memcpy raises "probe copied wrong bytes" -> --moe-prefill-hit-d2d silently inert in warm servers (matches the 3 baseline-failing tests/moe/test_prefill_hit_d2d.py).
- Useless for this config regardless: --moe-prefill-hit-d2d requires --moe-cache-size > 2*num_experts (576 slots for 288 experts) while large-KV auto plans leave 347-480 slots; and prefill misses are ~the whole layer, so little is skipped.

## PR #439 (OPEN) - stale prefill chunk cap
- CacheManager.__init__ snapshots swa_pool.prefill_chunk_budget, so /v1/cache/rebuild never refreshes it. Only affects runtime resize; static serving configs (this user) unaffected.

## PR #399 (OPEN, builds on #36) - fp8 CPU MoE
- fp8_block scale-stride fix + avx512f tier. Not applicable: GLM experts are nvfp4 (CPU isa avx2+vnni nvfp4-w4a8) and the i7-14700KF has no AVX-512.

## Already-known N/A
#354 fp8 KV (superseded by #408, merged in port); #398/#309 KV quant (no DSA support); #337 NVMe expert tier (166 GiB banks fit 188 GiB RAM); #340/#198 correctness-only; #411 (new in main 2026-09-12) server output default, not perf.

Why: each verdict is grounded in this box's measured facts (slot saturation at working set 336, sgl_kernel installed, nvfp4 CPU executor, static config); blind porting wastes A/B time or adds regressions.
How to apply: any "will MR X help this setup" question - start here, then A/B only what the verdict leaves open.

## Post-research verification (2026-09-12, measured - closes the hedges above)
- #339 A/B'd on hardware (A3 in glm53-post-iommu-baseline): NEUTRAL on this box - server prefill ~492-505 vs ~510 clean, decode flat; sgl_kernel confirmed installed; patch applied then reverted, tree clean. #339's real value = CUDA-graph capture legality, not speed here.
- #300 slot-saturation verdict confirmed by post-iommu B1: 373 -> 517 slots = only +2.5% @64k / +6% short decode. Ladder stays unported.
