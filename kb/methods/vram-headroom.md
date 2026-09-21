# VRAM-headroom verification: memory-ratio / kv-reserve / max-prefill matrix

Hardware wave 2026-09-20, branch vektory79 @ 315b278 (code identical to 8e2e4c7;
"Memory" commit only). RTX 5090 32607 MiB, plain boots, port 18801,
GLM-5.3-Flash-UD-Q3_K_XL.gguf, fp8 KV, hybrid, moe-cpu-threads 16,
max-running-requests 1, committed kernel-knob defaults (nothing set).
Question: which knobs buy >= 1.5-2 GB FREE VRAM at peak while serving big
contexts, and at what throughput cost. peak-free = 32607 - peak used (2 s
phase-tagged sampler, includes the ~1.5 GiB desktop/IDE baseline).

## Result table (peak-free at fill peak, prefill median c2..cN, decode steady)

| boot | ratio | reserve | chunk | fill peak-free | decode peak-free | prefill median | decode | verdict |
|------|-------|---------|-------|----------------|------------------|----------------|--------|---------|
| a    | 0.85  | 500k    | 8191  | 508 MiB (0.50 GiB) | 504 MiB | 805.3 tok/s | 13.85 tok/s | ok, control; anchor ~0.5 GB reproduced |
| b    | 0.80  | 500k    | 8191  | -  | -  | -  | -  | FAIL-FAST: slot floors unfundable |
| b2   | 0.80  | 400k    | 8191  | 2008 MiB (1.96 GiB) | 2004 MiB | 804.85 tok/s | 14.68 tok/s | ok (adjustment boot) |
| c    | 0.75  | 400k    | 8191  | -  | -  | -  | -  | FAIL-FAST: slot floors unfundable |
| d    | 0.85  | 350k    | 8191  | 508 MiB (0.50 GiB) | 504 MiB | 801.62 tok/s | 15.18 tok/s | ok; reserve cut = ZERO peak VRAM |
| e    | 0.85  | 500k    | 4095  | 2526 MiB (2.47 GiB) | 2524 MiB | 597.06 tok/s | 14.00 tok/s | ok, chunk lever |

Prefill medians exclude c1 warmup + the bogus last-full-chunk line + the tail
(T43). A median 805.3 vs the 808 anchor (-0.3%, noise). Decode band 13.2-15.0
holds; no lever shows a decode cost signal (bs=1 MMVQ graph untouched).

## Recommendation

Primary: **b2 = `--memory-ratio 0.80 --kv-reserve-tokens 400000
--max-prefill-length 8191`**. 1.96 GiB free at fill peak with a measured
prefill cost of -0.1% (804.85 vs 805.3) and decode +6% (14.68 vs 13.85, within
the boot noise band). This is the only measured config that meets the 1.5-2 GB
target without prefill loss.

Alternative: **e = chunk 4095** (`--max-prefill-length 4095`, ratio/reserve
unchanged). 2.47 GiB free at peak, but prefill -25.9% (597.06 vs 805.3);
observed chunk size 4032 tokens (server rounding below the 4095 cap), 16 full
chunks + 1073-token tail. Choose it only if 4096-class chunks are acceptable.

Not usable:
- **kv-reserve-tokens is NOT a headroom lever**: d (500k -> 350k) changed the
  fill peak by 0 MiB. The planner refills the freed KV budget into +88 MoE
  slots (moe_cache_size 1148 -> 1236); the peak tracks the ratio envelope.
  Its +9.6% decode (15.18) is within/near the noise band and buys no VRAM.
- **0.75 is infeasible** on this model/quant: slot-floor fail-fast at 400k
  reserve (see below). The measured ratio floor is 0.80 (and only with the
  reserve cut to 400k).

## Measured factors (replaces paper estimates)

- **VRAM per 0.05 ratio: 1500 MiB (1.46 GiB)**. a peak 32103 -> b2 peak 30603;
  reserve proven a no-op, so the full delta is the ratio step. Paper estimate
  1.3-1.4 GB per 0.05: order matched, measurement ~10% above the band.
- **VRAM per 100k reserve: 0 MiB**. a vs d peaks identical (32103). The paper
  estimate (linear KV bytes) is refuted for peak-VRAM purposes.
- **Chunk 8191 -> 4095-class: +2020 MiB free at peak** (32103 -> 30083) at an
  identical plan (1148 slots / 7830 pages) - pure prefill-transient reduction.

## Boot-log plan lines (ratio/reserve -> pool)

- a: `moe_cache_size=1148 num_pages=7830`, KV 501120 tokens = 2.98 GiB,
  fetch 30.7%, free-after-init 4.08 GiB class
- b2: `moe_cache_size=1062 num_pages=6266`, KV 401024 tokens = 2.38 GiB
- d: `moe_cache_size=1236 num_pages=5485`, KV 351040 tokens = 2.09 GiB
- e: identical to a (`1148/7830`, 2.98 GiB) - e's headroom is all transients

## Boot failures (exact lines, not retried blindly)

b (0.80/500000) and c (0.75/400000), both at cache_budget.py:77
check_partition_floors, fail-fast before serving:

```
ERROR Backend supervisor: ValueError: --moe-cache-auto budget cannot fund the
per-signature slot floors: 3 signature groups (39/1/2 layers; groups
[[0, 1, ... 40], [8], [9, 41]]) each need 288 slots (one whole ex...
```

Rule applied: the single adjustment boot was spent on b2 (0.80/400000) per the
"cut reserve to match" example; c had no adjustment left (cap: one extra boot).
The all-miss fallback (0.70/300000) was not run - unnecessary, b2 met the
target, and the cap was spent.

## Radix capacity check

All four measured boots re-sent the 65,585-token filler and got
`#cached-token: 65536` (HIT at the 8128-chunk boundary expectation) - no hit
downgrade at 401k or 351k-token KV pools.

## Quality sanity (informational, NOT a gate)

6/6 prompts ok in every measured boot. Identical-sha counts across the 4
measured boots: p0 4/4, p5 4/4, p1 3/4, p2 2/4, p3 2/4, p4 2/4 - the known
bimodal boot-mode nondeterminism (T54), config-independent as expected.

## Deviations

- b and c failed fast (slot-floor); both are deliverable failure-matrix rows.
- E's observed chunk size is 4032 tokens, not 4096 (server rounding below the
  4095 cap); median over 14 steady chunks.
- Census: sem.mp- 12 (preflight) -> 18 (final): 6 semaphores leaked by the two
  fail-fast scheduler kills; processes and VRAM identical pre/post (1551 MiB).
- a's fill wall was 137 s vs 99 s on b2/d (c1 warmup variance); medians match.

## Artifacts

- results: `.tasks/dense-q80-gemm/vram-headroom-results.json`
- per-boot run jsons: `.tasks/dense-q80-gemm/vram_{a,b,b2,c,d,e}_boot.json`
- VRAM sampler csv (ts,epoch,phase,used_mib): `.tasks/dense-q80-gemm/vram_{a,b,b2,c,d,e}.csv`
- boot logs: `.tasks/ft-gguf-serve-tuning/logs/vram_{a,b,b2,c,d,e}.log`(+`.pid`)
- run transcript: `.tasks/dense-q80-gemm/vram-headroom-run.out`
- harness: `.tasks/dense-q80-gemm/vram_headroom_run.py`
- census: `.tasks/dense-q80-gemm/vram-headroom/census.log`
