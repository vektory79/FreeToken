---
name: "ft-decode-research-campaign"
description: "GGUF decode 5-arm A/B + additive step model + T1 block decomposition; harness .tasks/decode-research"
type: project
lastUpdated: 2026-09-29T23:00
lastRecall: 2026-10-08T01:01
---

# decode-research campaign: 5-arm hybrid decode A/B on the user config (2026-09-26)

Harness: `.tasks/decode-research/` (run_arm.py / campaign.py / analyze.py; per-CPU /proc/stat+MHz sampler). Base = user's exact serve command (FTW dir, 0.82/400k, fp8, tier 10/50, 8191, threads 16). 65536-token radix-reused decode, 768 max_tokens, temp0.

## Results (steady tok/s, ms/step)
- base **16.29** / 61.4 ms = re-acceptance anchor reproduced (16.67 same-day noise)
- fetch3 (`--moe-hybrid-max-fetch 3`, f 30.7%->~59%): 16.14 (-0.9%) - split volume is NOT a lever
- ov0 (`FREETOKEN_HYBRID_OVERLAP=0`): 12.46 (-23.5%) - overlap hides ~19 ms/step
- flagsync0 (`FREETOKEN_CPU_MOE_FLAG_SYNC=0`, hostfunc): 12.77 (-21.6%, +17 ms) - REVERSES the task06 microbench ranking (h=3.02 memop vs 1.90 hostfunc); memop doorbell is production-right
- fetch0 (`--moe-hybrid-max-fetch 0`, all misses CPU): 9.87 (-39.4%, ~101 ms)

## Additive model (fits all five arms)
step = (GPU_chain + sync ~ 44 ms = 1.06 ms/layer) + CPU_exposed (17 ms base / 36 ov0 / 57 fetch0) + hostfunc penalty (17 ms if flag-sync off). GPU_chain composition unknown -> next campaign.

## Why: the "26.9 GB/s effective CPU leg" framing is idle-inflated
Pool busy only ~59% of the decode window (P workers ~62% @4.9 GHz, E workers ~55% @3.85 GHz; burst rate ~45 GB/s ~= bench 50). Minority pools nearly idle (7-16%) on E-cores 23/24. Fixed ~17 ms CPU_exposed does not shrink with volume (fetch3 flat) = per-layer fixed costs (doorbell dispatch, pool wake, barrier tails), not compute volume.

## Rig facts (i7-14700KF)
8 P-cores = cpu0-15 (SMT), 12 E-cores = cpu16-27; AVX-512 fused off (AVX2+VNNI ceiling); dominant pool 8P+7E pinned "cores 0..22", minorities 1 E-core each.

## Instrumentation gap (blocks the next step)
`moe_collect_stats` has NO CLI flag; `decode_routing_stats` (oracle_hit/entropy/working set) and `decode_miss_stats_per_layer` exist in offload_cache.py but have no caller; the campaign's /tmp collect-stats driver is lost. In-process driver = prerequisite for decomposing the 44 ms block and the routing-skew oracle.

## Real optimization candidates (post-instrumentation)
T1 decode step-0 instrumentation (in-process stats + per-layer trace of the 44 ms block); T2 attack the fixed per-layer CPU_exposed costs; T3 freq-aware cache policy (shrinks both legs; oracle via routing histogram; interleave-wash precedent). NOT levers: split recalibration, flag-sync swap, AVX-512, GPU math (not bound), FlashAttention (N/A for GDN+DSA arch).

Gotchas: temp0 e2e is boot-to-boot nondeterministic again (completion 763/552/546/545/553 across arms - lengths differ, rates don't bias); fill restores 64k from session-tier L2 in ~1 s (fix-3 works); tier machinery passive during decode; all arms WATCH=0, VRAM back to 1866-1874 MiB.

## Post-script (2026-09-26, T1 phase)
- Instrumented hybrid run measured 15.82 tok/s @467 completion tokens (clean rate outside the unopened profiler window; the window never fired - threshold trap documented in ft-decode-stats-instruments-unwired). Another temp0 boot-to-boot length datapoint: 467.
- The T1 driver now exists: .tasks/decode-research/driver/sitecustomize.py + analyze_trace.py (env-activated, plain boots unaffected); the "Instrumentation gap" above is closed at harness level.
- New lever discovered for the offload strategy: production gather ~29 GB/s vs benched 43.6 -> see ft-offload-decode-gather-gap.

## Outcome status (2026-09-27)
All three candidates resolved: T1 done (driver v3 - ft-decode-stats-instruments-unwired); T3 REFUTED by oracle (oracle_hit 25.7% vs LRU 25.1% - cache policy dead, only more VRAM slots cuts misses); offload gather lever measured and DEPRIORITIZED (closing only ties hybrid - ft-offload-decode-gather-gap). T2 wave-1 landed (t2d +7.9/+10.4% vs control 15.62, battery 24/24, NO COMMIT pending user - ft-t2-host-delivery-wave1). Live next candidate: lever #2, overlap remaining fetch into the 15.44 ms inter-call gaps.

UPDATE 2026-09-27 (later same day): the T2 "NO COMMIT pending user" line above is superseded - the pool-widening default flip (e541423) and the env knob (227e7a3) are COMMITTED; see ft-t2-host-delivery-wave1.

## T1 decomposition (2026-09-26 evening, driver v3) - single copy of the T1 findings
Hybrid stream span 59.1 ms of ~62 wall (offload 73.6/72.9) - step fully serialized; the 44 ms block = stream-wait-for-CPU-pool (~40 ms) + fetch copies (~6-8 ms) + kernels. Combined host delivery 2.57 GiB/step at ~41.5 GB/s vs benchbw overlapped-pair 72 GB/s concurrent-capable -> the gap is per-layer ping-pong micro-gaps (pool ramp/tails, doorbell dispatch) = the live T2 lever. Eager decode 15.84 ~= graphed 15.82 (CUDA graphs ~neutral at bs=1); miss stats identical across modes. Driver armor: any driver error disables the driver, never the engine (two boots were lost to a KeyError on _e1 before this fix).
