# Task 05 fetch fraction + per-layer overflow stats (c) + (d)

## Fraction assert (c): AUTO ACTIVE, not the cap-1 fallback

Boot log (3 partitions, one line each):

    --moe-hybrid-max-fetch auto: fetching 78.0% of each decode step's expert misses
    over PCIe (benched PCIe/CPU bandwidth ratio), the rest on the CPU

- No `no usable 'ft bench bw' profile ... fixed fetch cap of 1` warning -> the cap-1
  fallback (the ~2.5x-worse full-burst regression path, engine.py _resolve_hybrid_fetch)
  did NOT trigger.
- Fraction source: dtype_kernels["gguf"] overlapped pair (CPU 11.3 + PCIe 40.1 GB/s)
  -> 40.1/51.4 = 0.780 (benchbw refresh reproduces Task 03's 0.781).
- Realized split (overflow stats, pool 1/3, the 39-layer dominant partition):
  fetch_rate = 0.848 (fetched 4.58 of missing 5.40 per layer-step; the kernel rounds
  fraction x misses per step, so the realized rate sits slightly above 0.78).

## Per-layer overflow stats (d)

Driver (measurement-only, /tmp/ft_serve_driver.py) forces EngineConfig
moe_collect_stats=True (no CLI surface exists) and dumps every partition's
decode_miss_stats + decode_miss_stats_per_layer() at scheduler exit ->
/tmp/hybrid_decode_stats.json (also mirrored below). Session = 5319 decode steps
(767 + 6x128 burst + 24 battery prompts).

pool 1/3 - gguf (18,18,23) x39 layers (dominant):

    layer_calls 207441 | active/layer 8.0006 | missing/layer 5.4044
    miss_rate 0.6755 | fetched/layer 4.5811 | cpu/layer 0.8233 | fetch_rate 0.8477
    per-layer missing/step: min 4.29 / max 7.38 / mean 5.40
    per-layer fetched/step: min 3.65 / max 6.39 / mean 4.58

pool 2/3 - gguf (23,23,14) x1 layer: missing/layer 0.0045 (fully cached; layer 45
class routing is nearly stationary) - fetch_rate 0.875 on a negligible base.
pool 3/3 - gguf (18,18,14) x2 layers: missing/layer 1.0284, fetched 0.9677,
cpu 0.0607, fetch_rate 0.941.

Interpretation: the split machinery works exactly as designed (miss rate 0.68 at 64k
fill, 78% of misses fetched over PCIe, the rest on the CPU). The failure is NOT in the
split - it is the cost of each leg (see bisect-notes.md): the CPU leg runs on 6 of 16
threads (even 3-pool split) at ~2.96 GB/s effective, and the per-layer GPU<->CPU
handshake adds ~3.0 ms x 42 layers even with an idle CPU leg (cap-6 probe).

Method note: collect_stats adds 8 tiny in-graph accumulation ops per layer per step
(<2% step time); the offload control ran without the driver (campaign-identical
conditions), which if anything biases the comparison in hybrid's disfavor.
