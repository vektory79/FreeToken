# Task 05 decode numbers @64k fill (REAL file, 2026-09-15)

Method (per plan): re-send the SAME 64512-token prompt to reuse the radix prefix
(server log confirms `#cached-token: 64512, #new-token: 12`), stream, count deltas
carrying content OR reasoning_content, exclude TTFT (fully-cached-prefix TTFT ~3.3 s
is first-step overhead). max_tokens=768, temp0/topk1.

| run | config | decode tok/s | step time | TTFT s | burst mean/min tok/s |
|---|---|---|---|---|---|
| offload_control | --moe-strategy offload (same tuned plan) | **14.25** | 70.2 ms | 3.26 | 14.36 / 14.21 |
| hybrid_main | auto f=0.78, overlap ON, flag-sync, threads 6/5/5 | **2.99** | 334.4 ms | 3.42 | 3.07 / 2.95 |
| hybrid_cap6 | --moe-hybrid-max-fetch 6 (CPU leg ~0), rest identical | **5.08** | 196.9 ms | 3.46 | 4.20 / 4.02 |
| hybrid_ov0 | FREETOKEN_HYBRID_OVERLAP=0 (serial CPU->fetch) | **2.57** | 389.1 ms | 3.37 | 2.45 / 2.39 |
| hybrid_hostfunc | FREETOKEN_CPU_MOE_FLAG_SYNC=0 (hostfunc handshake) | see ab-table.md | | | |

Gate: >= 14.3 tok/s (offload-only reference). **hybrid_main FAILS at 2.99 (4.8x below);
the same-session offload control reproduces the campaign anchor at 14.25**, so the gate
number is honest and the box state is healthy.

- burst probe (6x cold 2k prompts, 128 tokens each): hybrid never beats offload on any
  per-step comparison (3.07 vs 14.36 mean) -> "not slower than offload" check FAILS.
- overlap A/B: ov0 2.57 vs ov1 2.99 -> overlap recovers ~+14-16%; not the problem.
- prefill is UNAFFECTED by hybrid: steady-state 263-264 tok/s on the 64k fill
  (first 4096-chunk 47.9 = JIT warmup), identical to the offload-only campaign's 264-268.

## VRAM delta (f)

| plan | slots | KV pages | free after init | free after graphs |
|---|---|---|---|---|
| hybrid_main | 1232 | 133 | 3.37 GiB | 3.30 GiB |
| offload_control | 1231 | 137 | 3.36 GiB | 3.29 GiB |

Delta: +1 slot, -4 KV pages, +0.01 GiB free = NOISE. The hybrid path does NOT free
slot/KV budget on this plan: the full LRU slot pool is still required (the cap only
limits per-step fetches, not residency), so the "hybrid frees the GPU slot budget"
expectation is NOT observed. nvidia-smi before/after each run: 1348 -> ~1357 MiB.

## CPU utilization during decode misses (e)

/proc/stat sampler (1 Hz, aggregate busy cores of 20 physical):
- hybrid 64k decode window: avg 7.5 cores busy (pool 1/3 pinned to 6 workers + coord
  + GPU-side python) -> 12+ cores idle while the starved pool grinds: the CPU leg is
  thread-starved, the machine is not saturated.
- offload control decode window: avg 3.9 cores busy (no CPU GEMV at all).

## Boot time (a)

hybrid_main ready in 136s (prior offload campaign 91-115s; +15-20 s for the three
per-partition executor initializations + graph capture at bs 1/2/4).
