---
name: "glm53-post-iommu-baseline"
description: "GLM-5.3 NVFP4 post-iommu baseline: 1M/512k budget recipes, chunk-size prefill lever; 09-14 fail-fast transient"
type: project
lastUpdated: 2026-09-19T03:12
lastRecall: 2026-09-19T03:10
---

# GLM-5.3-Flash-NVFP4 hybrid on RTX 5090: post-iommu=pt re-baseline (2026-09-12)

Supersedes the 2026-09-11 tok/s numbers in glm53-flash-nvfp4-cache-budget (measured with IOMMU Translated, PCIe ~25/24 GB/s; now H2D 45.8 / D2H 57 GB/s, nvbandwidth re-verified 45.82).

## A1 = the user's exact serve command (threads 16, ratio 0.89, 1M reserve)
hybrid, --moe-cpu-threads 16, --kv-reserve-tokens 1048576, --kv-cache-dtype nvfp4, --disable-moe-prefill-overlap, --memory-ratio 0.89, --max-prefill-length 4096, FTW /media/ai/models/RedHatAI/GLM-5.3-Flash-NVFP4-FTW/.
- Resolved: moe_cache_size=373, num_pages=16388 (1,048,832 KV tokens, 3.83 GiB), ready 50s (warm triton cache), fetch fraction 25.0% (v4 benchbw profile).
- Prefill: 64,524 tokens / 140.5 s = 459 tok/s client-side, ~514 tok/s server-side steady. 16 chunks x 4096 at ~8.0 s/chunk (first chunk 145 tok/s = JIT/autotune warmup - exclude from comparisons).
- Decode @64k ctx: 15.13 tok/s (65-70 ms/token). Short-ctx (2k): 15.5.
- VRAM: free 29.98 GiB before load -> 2.86 GiB after init -> 2.79 GiB after CUDA graphs.

## A2 = A1 with --moe-cpu-threads 20: decode 15.29 @64k / 15.6 short, prefill identical (~508-512 server)
Threads 16 == 20 post-iommu (noise). The 09-11 "20 threads best" advice is superseded; the 20t-benched benchbw profile is NOT mistuned at 16t. Decode vs pre-iommu 14.24-14.50 = +5-6% from iommu=pt alone (fresh 25% fetch fraction + 2x PCIe).

## B1 = --kv-reserve-tokens 524288 (ratio 0.89): moe_cache_size=517, num_pages=8230 (526,720 tokens, 1.92 GiB)
- Greedy fill converts freed 1.9 GiB of KV into slots: free after init STAYS 2.86 GiB. Shrinking the KV reserve NEVER creates transient-VRAM headroom; only lowering --memory-ratio (or explicit sizing) does.
- Decode 15.51 @64k (+2.5% vs A1) / short 16.5 (+6%): slots above the nominal 336 working set still help slightly - "saturated = zero gain" is overstated, but the gain is small.

## B2 = 512k + --max-prefill-length 8192
- ratio 0.89: CUDA OOM on the FIRST 8192 prefill inside triton do_bench ("Tried to allocate 256.00 MiB", 220 MiB free) - 8192-chunk transients + 256 MiB autotune do not fit the greedy-fill ~2.86 GiB headroom. Issue #401 class; hits even at 512k nvfp4. Backend worker dies and the server hangs in uvicorn "Waiting for connections to close" (see ft-serve-test-and-e2e-gotchas).
- ratio 0.85: boots clean (moe_cache_size=426, free after init 4.07 GiB, autotune passed). Server-side prefill ~693-695 tok/s vs ~510 at 4096 = +36%; decode @64k unchanged 15.5. Mechanism: each prefill chunk does a full-layer PCIe copy round (42 MoE layers); doubling the chunk halves the round count per token - THE prefill lever for this config.

## A4 series = 1M reserve with bigger chunks (measured 2026-09-12 evening)
- 1M + ratio 0.85 + 8192 does not BOOT: fail-fast in engine/cache_budget.py::plan_cache_budget - minimum plan (moe=288 slots, kv=16384 pages) 8.19 GiB > budget 8.10 GiB. Minimum ratio for a 1M nvfp4 reserve on this card is 0.89 (0.88 -> budget 8.18 GiB, still short).
- 8192-chunk transient peak measured ~3.4-3.5 GiB at a 64k fill, NOT the old ~2.4 GiB: OOMed twice with free-after-init 3.29 GiB (kernel/fla/kda_chunk_delta_h.py:369 v_new=torch.empty_like(u), 128 MiB alloc failed), passed only with 4.07 GiB free (B2).
- A4c TRAP - explicit --moe-cache-size 288 does NOT free VRAM: the planner poured the freed slot budget into MORE KV (1,237,760 tokens = 4.51 GiB instead of 3.83 GiB; explicit sizing participates in the auto fill, PR #198 behavior); free stayed 3.29 GiB -> same OOM. Converse of B1: slot caps alone never create transient headroom; must ALSO cap KV with --num-tokens / --num-pages (sibling of --num-tokens is --num-pages, dest=num_page_override).
- A4b = 1M + --moe-cache-size 340 + chunk 6144: the SAFE prefill win at 1M. Free 3.29 GiB, server prefill ~618 tok/s (+20%), decode 14.8 @64k / 15.2 short (borderline noise vs A1; 340 slots still >= 336 working set).
- 8192 at 1M with capped KV + 288 slots (~3.9 GiB free) was NOT measured - fragile candidate, expect OOM risk at deep fills where the DSA indexer workspace (~0.62 KB/token) eats the same headroom.

## Working set correction (invalidates the old 43x8=344 claim)
CPU executor boot log: "experts=288 layers=42 top_k=8" -> serving MoE layers = 42, working set = 336 pairs. Slots: 373 (1M/0.89), 426 (512k/0.85), 517 (512k/0.89) all >= 336; decode gain 373->517 slots only +2.5% @64k / +6% short. Removes the decode motive for PR #300 kv-ladder on this model (ft-pr-relevance memory).

## Measurement method
Same prompt reused across configs; radix prefix reuse gives clean decode (log "#cached-token: 64512"); TTFT on a cached prefix ~6 s is first-decode-step overhead, not re-prefill. Authoritative prefill metric = server log "input throughput (token/s)" per chunk. See ft-serve-test-and-e2e-gotchas for the 422/model-field, log-metric and backend-death-hang gotchas.

## A3 = PR #339 patch A/B (2026-09-12 late) - NEUTRAL on this install
Applied the full #339 diff (clean on vektory79), reran A1: server prefill ~492-505 vs ~510-514 clean (noise), decode 14.85/15.06 vs 15.13/15.5 (noise). Root cause: sgl_kernel IS installed here; the .item() D2H sync #339 removes lives only in the triton fallback of kernel/causal_conv1d.py - the sgl path never pays it. #339's real value = CUDA-graph capture legality, not speed here. Patch reverted (tree clean).

## A4d = the 1M max-prefill recipe, WORKS at 64k fill (2026-09-12 23:45)
`--moe-cache-size 288 --num-tokens 1048576 --max-prefill-length 8192` at ratio 0.89: boots, KV exactly 1,048,576 tokens (3.82 GiB), free after init 4.04 GiB, 8192 autotune + transients pass. Server prefill ~691-693 tok/s = +35% vs ~510 at 4096 (same as B2). Cost: decode 14.42 @64k / 14.81 short = -4.5% vs A1 (288 slots < 336 working set). CAVEAT: only 64k fill verified; deep fills (300k+) risky - DSA indexer workspace (~0.62 KB/context-token) grows into the same 4.0 GiB headroom, and 288 is the minimum expert floor.
- CLI flag: the KV cap flag is `--num-tokens` (args.py:413-417, dest=num_token_override); `--num-token-override` is NOT a CLI flag (argparse rejects).

## Final recommendation menu (post-iommu, RTX 5090, GLM-5.3-Flash-NVFP4-FTW, threads 16)
- 1M reserve, safe: add `--moe-cache-size 340 --max-prefill-length 6144` -> prefill ~618 tok/s (+20%), decode ~neutral (14.8-15.2 vs 15.13-15.5).
- 1M reserve, max prefill: add `--moe-cache-size 288 --num-tokens 1048576 --max-prefill-length 8192` -> prefill ~691 (+35%), decode -4.5%, deep-fill fragile.
- 524288 reserve: `--memory-ratio 0.85 --max-prefill-length 8192` -> prefill ~693 (+36%), decode unchanged 15.5 @64k (426 slots); ratio 0.89 + explicit --num-tokens cap keeps ~470+ slots but was NOT measured.
- Decode levers exhausted: threads 16==20 (noise), fresh benchbw profile already banked the iommu=pt +5-6%; remaining decode gap is per-layer sync overhead, not flag-tunable.
- Verify flags resolved: "Allocating ... tokens for KV cache" (must equal the reserve, not grow), "Free memory after initialization" (>= ~3.5 GiB for 8192, ~2.9 GiB for 6144).

## 1M boot status history
- 2026-09-14: the canonical 1M nvfp4 command (ratio 0.89) FAIL-FASTED (min plan 8.19 GiB > budget 6.63; booted 09-12) after an unexplained budget shift; working reserve was --kv-reserve-tokens 524288 -> 322 slots (< 336 WS), decode ~14.1 @64k.
- 2026-09-15: that failure was TRANSIENT - the canonical 1M command boots again (341 slots, short decode 15.18, nvfp4 f=24.8% profile restored). A1 is the reference config again; the 322-slot baseline above is historical. See pr408-kv-nvfp4-1m-port (final section) and gguf-hybrid-reacceptance-final.

