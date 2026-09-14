---
name: "glm53-flash-nvfp4-cache-budget"
description: "GLM-5.3-Flash-NVFP4 boot: cache_budget min-plan math, recipes, FTW load ladder; decode numbers superseded post-iommu"
type: project
lastUpdated: 2026-09-13T23:44
lastRecall: 2026-09-14T11:45
---

# GLM-5.3-Flash-NVFP4 on RTX 5090: --moe-cache-auto budget floor

From the 2026-09-11 OOM investigation + HARDWARE-VERIFIED boot (assertion in `engine/cache_budget.py::plan_cache_budget`). All decode tok/s numbers here were measured PRE-iommu=pt and are superseded by glm53-post-iommu-baseline (2026-09-12: 15.13-15.5 tok/s, threads 16 == 20 flat, "20 threads best" advice RETRACTED - see glm53-post-iommu-baseline and ft-serve-moe-flags-semantics; lessons below still stand).

## The arithmetic
- Budget = memory_ratio * baseline_free - weights - fixed_cache. RTX 5090 32GB, baseline_free ~29.6 GiB, dense bf16 weights ~17 GiB (checkpoint: 16.97 GiB dense, 166.22 GiB NVFP4 experts, 1.05 GiB visual not loaded), GDN/DSA fixed pool ~1.2 GiB.
- MoE slot = one (layer, expert) pair, ~14.4 MB (288 experts). Serving working set = 42 MoE layers x top_k 8 = 336 pairs - the old 43x8=344 claim is WRONG (MTP layer 45 is not a serving MoE layer; verified via CPU executor boot log, see ft-serve-moe-flags-semantics).
- KV page (dsa backend, page_size=64) ~0.733 MB -> ~11 KB/token bf16 (MLA latent + DSA index slab).
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

## Related upstream (FlashML-org/FreeToken)
- PR #300 (OPEN) KV ladder - grow KV on demand from expert slots. Verdict for THIS model: no decode gain (slots already >= working set 336 on every plan; see ft-pr-relevance-glm-hybrid) - it buys flexibility, not speed, here.
- PR #340: dummy-page off-by-one in the auto KV floor (kv_reserve_pages+1); makes floor slightly stricter, not a fix.
- PR #198: explicit --num-pages/--num-tokens now participates in the auto reserve (fixes issue #383 class).
- Issue #401: no room for prefill after auto boot; its workaround: explicit --moe-cache-size below the auto number. PR #337: NVMe disk tier for expert banks (RAM overflow).

## How to apply
Any boot of this model class on a 32GB card: compute the min plan first; prefer --disable-moe-prefill-overlap + memory_ratio ~0.85 + --max-prefill-length 4096 over raising memory_ratio; expect the first prefill to autotune triton kernels (one-time 256 MiB bench cache).

## Hybrid decode lessons (principles; A/B numbers superseded by glm53-post-iommu-baseline)
- Standing lessons: hybrid decode = LRU ensure + capped PCIe fetch OVERLAPPED with CPU GEMV on misses, one global LRU slot pool (layers/moe.py _decode_hybrid); auto fetch fraction from ~/.cache/freetoken/benchbw/<uuid>.json; past the balanced fetch point GPU util rises and throughput FALLS - do NOT chase 99% GPU util; slots above the 336-pair working set add only ~+2.5-6%; remaining ~2x gap to the ~27 tok/s channel bound is per-layer sync overhead, not flag-tunable; benchbw re-profile at 8 threads = no-op (thread-insensitive profile).
- Branch A/Bs 2026-09-11 (pre-iommu): ple-disk / fp8-scaled-mm / multi-gpu-select / split-residency / decode-token-checkpoint - no decode gain for GLM on this card; v5 (warmup branch + topk-noop, PR #319 merged e05cff8) decode-neutral.

## Boot load time (2026-09-11 A/B)
- Baseline (--expert-load serial) ready in 245s: ~14s dense+config, ~220s "expert banks: slow path (serial build)" (mmap, single-stream, 185 GiB experts at ~0.86 GB/s), ~11s CUDA graphs. Disk is NVMe (nvme0n1, xfs) - not the bottleneck.
- --expert-load parallel: ready in 131s (177 GiB experts at ~1.7-2.4 GB/s), warmup 200. Host RAM dips to ~8 GiB available at peak (banks 166 GiB + whole-shard anonymous buffers, non-reclaimable) - tight but OK on 188 GB box; auto mode would fall back to serial on this box (its rule: banks+one shard must fit free RAM, logs "low free RAM -> serial build... override with --expert-load parallel").
- Third lever (untested): one-time `ft checkpoint --model <hf> --out <ftw> --moe-backend offload --quant-backend moe.nvfp4=b12x` -> server auto-detects FTW and boots via "expert banks: FTW fast path" (checkpoint/__main__, expert_banks.py ~line 300).
- Page-cache warmth between restarts gives no speedup empirically (all same-day boots ~225-245s).

## FTW conversion (2026-09-11, measured)
- One-time: `ft checkpoint --model <hf_dir> --out <ftw_dir> --moe-backend offload --quant-backend moe.nvfp4=triton --shard-gib 8` -> 177 GiB in 439s (GPU repack, ~300-630 MB/s disk). quant_format nvfp4, 252 per-layer experts_bank entries.
- FTW boot (threads 20, ratio 0.89): ready in 72s (vs 245s serial / 131s parallel-HF); "Loading expert banks (FTW)" 160G with bursts to 15 GB/s and pin waves 2-3 GB/s; host RAM avail dips to ~15 GiB. FTW path ignores --expert-load (expert_banks.py checks is_ftw_checkpoint first).
- Load-time ladder for this checkpoint on RTX 5090: serial 245s -> parallel 131s -> FTW 72s.

## 512k KV experiment (2026-09-12, BF16 KV, FTW checkpoint)
- Model config max_position_embeddings=1048576, so 512k is VRAM-bound only. KV: 0.733 MB/page -> +100k tokens ~= +0.73 GiB ~= -52 expert slots or +0.025 memory-ratio.
- ratio 0.93 + kv-reserve-tokens 524288 FAILS fail-fast assert (min plan 288 slots + 8192 pages = 10.36 GiB > budget 10.24 GiB, short by 115 MB). ratio 0.93 + kv-reserve 512000 (8000 pages) PASSES: resolved moe=294 slots, num_pages=8006 (512384 tokens, 5.71 GiB), free after init 1.81 GiB, boot 68s.
- Decode at 512k reserve: <=2% loss vs 262k reserve (slots 294 vs 427; within noise).
- Filling ~500k tokens in ONE prefill OOMs: kernel/fla/kda_chunk_delta_h.py:364 h=k.new_empty(B,NT,H,V,K) - GDN state history scales with the WHOLE prefill length (NT), independent of --max-prefill-length; needed 128 MiB with 169 MiB free because the idle wrapper on :18080 holds ~1.27 GiB VRAM. Workarounds: free the wrapper's VRAM, or fill context incrementally (HybridRadixCache reuses GDN-state prefixes across requests, so each pass is short). KV ladder = PR #300, see verdict above.

## OOM cause depends on KV mode
- bf16 KV -> GDN history buffer (kda_chunk_delta_h.py:364). nvfp4 1M reserve -> DSA indexer workspace ~0.62 KB/token (dsa_indexer_kpool.py:276), NOT GDN; 578k succeeds - see pr408-kv-nvfp4-1m-port.
