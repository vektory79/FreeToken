---
name: "ft-t2-host-delivery-wave1"
description: "T2 wave-1 + default flip 17.36 tok/s; FREETOKEN_T2_POOL_THREADS knob; adjacent-SMT trap; commits 227e7a3..d751511"
type: project
lastUpdated: 2026-09-27T15:47
lastRecall: 2026-09-27T17:48
---

# T2 host-delivery wave 1: minor-pool starvation CONFIRMED, t2d adopted-pending-commit (2026-09-26 night)

Campaign: kb/cases/decode-t2-host-delivery/ (TASK.md, step0-phase-table.md, wave1-ab-summary.md - kb self-sufficient). Step 0 refuted the brief's wake/dispatch hypothesis: wake+dispatch = 0.303 ms/step (0.76% of the ~40 ms block), dominant pool already at ~48.7 GB/s in-window. Real lever found by decomposition: pool2 (2 layers) = 8.28 ms/step (13% of 63.9 ms wall) on ONE E-core at ~11.7 GB/s.

## Adopted lever (t2d)
- Delivery: env-knob FREETOKEN_T2_POOL_THREADS + FREETOKEN_T2_POOL_CPUS, python/freetoken/engine/engine.py `_t2_pool_override` (~:65) applied in `_init_cpu_moe_executors` (~:1270-1274). Flat-token greedy CPUS parsing; both vars required; fail-fast validation; INFO boot line; SMT-doubling guard (WARN when physical reps < threads, via `physical_core_cpus()` cpu_executor.py:201). Python only, gate-off = byte-identical.
- Spec: `FREETOKEN_T2_POOL_THREADS=15,1,4 FREETOKEN_T2_POOL_CPUS=0,2,4,6,8,10,12,14,16-22,23,24-27` (pool2 -> 4 E-cores 24-27).
- A/B (same window, radix-reuse 64k, temp0): control base_r3 15.62 tok/s / p50 62.9 ms / pool0 busy 39.78 ms/step / pool2 4.018 ms/call; t2d 16.86 (+7.9%); t2d_r2 17.25 (+10.4%); pool2 -> 1.47-1.64 ms/call @ 4 workers; pool0 stayed in clean band. DRAM-concurrency fear from earlier void arms NOT confirmed.
- Gates: +5% on 2 arms MET; absolute anchor-gate 17.1 tok/s met on 1 of 2 (today's control class 15.4-15.6 = -4..-5% day noise vs 16.29 anchor). Prefill: knob is decode-only (executors never called in prefill); 49-token tail lines -0.65/-1.09% = noise. Battery 24/24 PASS (kb/harness/quality/run_battery.py env1, override verified in its serve.log).
- NO COMMIT (user decides). Default-flip would be a separate micro-commit. Lever #2 next if more needed: overlap the remaining fetch into the 15.44 ms inter-call gaps.

## RIG FACT (cost 3 void arms - never spec P cores as "0-7" on this box)
i7-14700KF on this kernel enumerates P SMT siblings as ADJACENT pairs (0,1)(2,3)..(14,15), E = 16-27. `physical_core_cpus()` = [0,2,4,6,8,10,12,14,16..27]. Default dominant MoE pool = [0,2,4,6,8,10,12,14,16..22]; the boot print "pinned to cores 0..22" is min..max of a NON-contiguous list. Spec "0-7" = only 4 physical P cores + 4 SMT siblings -> resolve_threads_and_affinity pads with siblings -> dominant pool crippled (+10 ms/step busy, -6..-17% decode) that mimics a concurrency penalty and is invariant to the variable you think you are testing.
**Why:** three treated arms (t2a/t2b/t2c) were voided by this; discrimination cost a full night wave.
**How to apply:** any core-list spec or pin documentation on this rig: enumerate via physical_core_cpus() or the explicit even/odd list; treat "a..b" boot prints as min..max, never as ranges; the _t2_pool_override guard now warns on reps < threads.

Artifacts: .tasks/decode-research/arm_{base_r2,base_r3,t2n,t2n2,t2a,t2b,t2c,t2d,t2d_r2}.json, instr_t2/t2_phase_trace_*.csv, cpu_*.csv samplers, analyze_t2_phases.py / analyze_t2_wave1.py / cpu_audit_w1p3.py.

## 2026-09-27 follow-up: default flip LANDED + committed; weight-order correction
- CORRECTION to the phrasing above: the real partition order is weights [39,1,2] - the 2-layer starved pool prints as pool 3/3 (cores 24..27), the 1-layer pool as 2/3 (cpu23). "Pool2 = 2 layers" in this file refers to the trace pool column, not boot print order.
- Default widening (commit e541423, cpu_executor.py e_like_cpus/_widen_starved_minority, cap 4, costliest-starved-first; env knob committed separately as 227e7a3 so the new engine-level test is coherent): flip_check arm with NO env = 17.36 tok/s steady, pools [15,1,4], pool 3/3 = [24..27]. A literal lowest-weight-first rule widened the WRONG pool (the 1-layer one) and measured 14.78 tok/s on hardware - disproven; costliest-starved-first is the shipped rule.
- Other commits: e10b424 (cpu_moe_ext.cpp env-gated phase trace), d751511 (kb t2 case). The 2026-09-26 pending kb-distill commit had already landed as c3b2a96 (parallel session) - the pending-commit memory was deleted.
- Tests: tests/moe/test_hybrid_fetch.py 24 passed (incl. the CUDA-marked ones, tiny tensors).
- Lever #2 (gap overlap) resolved the same night: fetch was ALREADY overlapped (28.7 ms/step hidden under pool windows); only live micro-lever = coordinator doze-fix (+1.6-1.8%). See kb/cases/decode-t2-host-delivery/lever2-gap-anatomy.md and ft-decode-lever-research.
