---
name: "glm53-flash-nvfp4-cache-budget"
description: "GLM-5.3-Flash-NVFP4 RTX 5090: ft serve cache budget + hybrid decode A/B (threads 20, ratio 0.89); PRs #198/#340/#319"
type: project
lastUpdated: 2026-09-12T00:26
lastRecall: 2026-09-12T15:29
---

# GLM-5.3-Flash-NVFP4 on RTX 5090: --moe-cache-auto budget floor

From the 2026-09-11 OOM investigation + HARDWARE-VERIFIED boot (assertion in `engine/cache_budget.py::plan_cache_budget`).

## The arithmetic
- Budget = memory_ratio * baseline_free - weights - fixed_cache. RTX 5090 32GB, baseline_free ~29.6 GiB, dense bf16 weights ~17 GiB (checkpoint: 16.97 GiB dense, 166.22 GiB NVFP4 experts, 1.05 GiB visual not loaded), GDN/DSA fixed pool ~1.2 GiB.
- MoE slot = one (layer, expert) pair, ~14.4 MB (288 experts, 43 MoE layers incl. MTP layer 45). KV page (dsa backend, page_size=64) ~0.733 MB -> ~11 KB/token (MLA latent + DSA index slab).
- With prefill overlap the floor is 2*num_experts = 576 slots = 8.3 GiB; --kv-reserve-tokens 262144 -> 4096 pages = 3 GiB. Min plan 11.3 GiB > ~9 GiB budget -> intentional fail-fast assert.
- The greedy fill ALWAYS leaves zero pool headroom: any fitting plan = budget exactly.

## VERIFIED working command (boot + 8k prefill + decode, no OOM, 2026-09-11)
```
ft serve --moe-strategy hybrid --moe-cpu-threads 8 --kv-reserve-tokens 262144 \
  --expert-load serial --max-running-requests 1 \
  --disable-moe-prefill-overlap --memory-ratio 0.85 --max-prefill-length 4096
```
Resolved: moe_cache_size=326, num_pages=4113 (263,232 KV tokens, 2.93 GiB), free after init 4.17 GiB, ready in ~3 min.

## Gotchas discovered on hardware
- ratio 0.9 + only --disable-moe-prefill-overlap (plan moe=427/pages=4102, free 2.65 GiB) DIES on the first real prefill: KDA/GDN triton autotune (kernel/fla/solve_tril.py) needs triton do_bench 256 MiB + ~2.4 GiB chunk transients for an 8192 chunk. Issue #401 class. Lowering memory_ratio to 0.85 AND capping the chunk at 4096 fixes it.
- The prefill-length CLI flag is `--max-prefill-length` (dest max_extend_tokens); `--max-extend-tokens` is NOT a valid flag (argparse dies in 1s) despite issue #401 calling it --max-extend-tokens.
- With max_tokens=8 the glm reasoning parser returns content='' (tokens burned in reasoning_content) - not an error.

## Alternative (keep overlap): cap --kv-reserve-tokens at <= ~46k tokens (576 slots need budget - 0.54 GiB). Default reserve 8192 boots but leaves only ~8k KV tokens (issue #111).

## Related upstream (FlashML-org/FreeToken, open as of 2026-09-11)
- PR #340: dummy-page off-by-one in the auto KV floor (kv_reserve_pages+1); makes floor slightly stricter, not a fix.
- PR #198: explicit --num-pages/--num-tokens now participates in the auto reserve (fixes issue #383 class).
- KV ladder (grow KV on demand from expert slots, port of #300, commit 057ce31 referenced in #340 thread) - best future fit, not yet a standalone open PR.
- Issue #401: no room for prefill after auto boot; its workaround: explicit --moe-cache-size below the auto number. PR #337: NVMe disk tier for expert banks (RAM overflow).

## How to apply
Any boot of this model class on a 32GB card: compute the min plan first; prefer --disable-moe-prefill-overlap + memory_ratio ~0.85 + --max-prefill-length 4096 over raising memory_ratio; expect the first prefill to autotune triton kernels (one-time 256 MiB bench cache).

## Decode tuning A/B (2026-09-11 hardware, same prompt, 768 tokens, batch 1)
- Hybrid decode per (layer, step): LRU ensure + capped PCIe fetch OVERLAPPED with CPU GEMV on misses; layer completes at max(cpu, pcie); slots pool is one global LRU (layers/moe.py _decode_hybrid). Auto fetch fraction from ~/.cache/freetoken/benchbw/<uuid>.json overlapped benches.
- Measured tok/s decode / GPU% in decode window: baseline (8t, 326 slots, auto 19.7% fetch) 12.84/59; --moe-cpu-threads 20 -> 13.66/73; + --memory-ratio 0.89 (417 slots, KV 4112 pages = 262k kept) 14.24/76 = BEST; + --moe-hybrid-max-fetch 3 -> 10.15/99% (GPU 99% but -29% speed!); re-profiling benchbw at 8 threads is a no-op (55.8 vs 57.7 GB/s - 8 threads nearly saturate DRAM, profile is thread-count-insensitive).
- Lesson: past the balanced fetch point GPU utilization rises and throughput FALLS; slots >= per-step working set (43 layers x top-8 = 344 pairs) add ~+4%; remaining ~2x gap to the ~27 tok/s channel bound is fixed per-layer sync overhead, not flag-tunable.
- Best config: --moe-cpu-threads 20 + --memory-ratio 0.89, auto max-fetch, same KV floor. Do NOT chase 99% GPU util.
- Branch facts: perf/default-vendored-topk-router already merged into main (e05cff8, PR #319); fix/prefill-jit-warmup merges with a trivial engine/config.py comment-field conflict, adds startup prefill warmup + --skip-prefill-warmup; ple-disk/fp8-scaled-mm/multi-gpu-select/split-residency/decode-token-checkpoint - no decode gain for GLM on this card.

- v5 (warmup branch + topk-noop, best flags): 13.75 tok/s / GPU 69% - within run-to-run noise of v2 (14.24); branch combo is decode-neutral, merges with one trivial config.py comment conflict (resolved manually during the test).

## Boot load time (2026-09-11 A/B)
- Baseline (--expert-load serial) ready in 245s: ~14s dense+config, ~220s "expert banks: slow path (serial build)" (mmap, single-stream, 185 GiB experts at ~0.86 GB/s), ~11s CUDA graphs. Disk is NVMe (nvme0n1, xfs) - not the bottleneck.
- --expert-load parallel: ready in 131s (177 GiB experts at ~1.7-2.4 GB/s), warmup 200. Host RAM dips to ~8 GiB available at peak (banks 166 GiB + whole-shard anonymous buffers, non-reclaimable) - tight but OK on 188 GB box; auto mode would fall back to serial on this box (its rule: banks+one shard must fit free RAM, logs "low free RAM -> serial build... override with --expert-load parallel").
- Third lever (untested): one-time `ft checkpoint --model <hf> --out <ftw> --moe-backend offload --quant-backend moe.nvfp4=b12x` -> server auto-detects FTW and boots via "expert banks: FTW fast path" (checkpoint/__main__, expert_banks.py ~line 300).
- Page-cache warmth between restarts gives no speedup empirically (all same-day boots ~225-245s).

## FTW conversion (2026-09-11, measured)
- One-time: `ft checkpoint --model <hf_dir> --out <ftw_dir> --moe-backend offload --quant-backend moe.nvfp4=triton --shard-gib 8` -> 177 GiB in 439s (GPU repack, ~300-630 MB/s disk). quant_format nvfp4, 252 per-layer experts_bank entries.
- FTW boot (threads 20, ratio 0.89): ready in 72s (vs 245s serial / 131s parallel-HF); "Loading expert banks (FTW)" 160G with bursts to 15 GB/s and pin waves 2-3 GB/s; host RAM avail dips to ~15 GiB. FTW path ignores --expert-load (expert_banks.py checks is_ftw_checkpoint first).
- FTW decode sanity: 14.50 tok/s / GPU 77% (no regression vs v2 14.24); resolved moe_cache_size=427, num_pages=4113 (262k KV kept), free after init 2.84 GiB (tighter than ratio 0.85's 4.14).
- Load-time ladder for this checkpoint on RTX 5090: serial 245s -> parallel 131s -> FTW 72s.

## 512k KV experiment (2026-09-12, FTW checkpoint)
- Model config max_position_embeddings=1048576, so 512k is VRAM-bound only. KV: 0.733 MB/page -> +100k tokens ~= +0.73 GiB ~= -52 expert slots or +0.025 memory-ratio.
- ratio 0.93 + kv-reserve-tokens 524288 FAILS fail-fast assert (min plan 288 slots + 8192 pages = 10.36 GiB > budget 10.24 GiB, short by 115 MB). ratio 0.93 + kv-reserve 512000 (8000 pages) PASSES: resolved moe=294 slots, num_pages=8006 (512384 tokens, 5.71 GiB), free after init 1.81 GiB, boot 68s.
- Decode at 512k reserve: 14.19 tok/s / GPU 82% vs 262k reserve 14.50/77% - <=2% loss (slots 294 vs 427; within noise).
- Filling ~500k tokens in ONE prefill OOMs: kernel/fla/kda_chunk_delta_h.py:364 h=k.new_empty(B,NT,H,V,K) - GDN state history scales with the WHOLE prefill length (NT), independent of --max-prefill-length; needed 128 MiB with 169 MiB free because the idle wrapper on :18080 holds ~1.27 GiB VRAM. Workarounds: free the wrapper's VRAM, or fill context incrementally (HybridRadixCache reuses GDN-state prefixes across requests, so each pass is short). KV ladder (#300 port, discussed in #340) would grow KV from expert slots on demand - exists in NO branch (git log --all verified).
