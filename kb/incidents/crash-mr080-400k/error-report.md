# ft serve --memory-ratio 0.80 --kv-reserve-tokens 400000 fp8 crash: reproduce + root cause

- Date: 2026-09-20 (evening), tree vektory79 @ 5001504 (merge round 3 of origin/main)
- Verdict: **REPRODUCED** - pre-serving fail-fast at boot, never reaches /ready
- Classification: **Environment** (desktop VRAM tax ~357 MiB higher than the 09-20 wave)
  on top of a **config recipe that sits exactly on the ratio floor** (09-20 margin was
  +3 slots). **NOT a merge regression.**

## 1. Reproduction

Exact user command on port 18081 (runner: `repro-runner.sh`, stdout: `runner.out`,
server log: `serve.log`). Preflight census: no foreign procs, sem.mp- = 18 (baseline),
nvidia-smi 1908 MiB used / 30241 MiB free. Post-run census identical (sem.mp- = 18,
1870 used / 30279 free). Failure at ~9 s (before bank load).

```
BOOT-VERDICT: FAIL (ready not reached)
RUNNER-RC=1
```

## 2. Verbatim error

```
ERROR    Backend supervisor: ValueError: --moe-cache-auto budget cannot fund the
per-signature slot floors: 3 signature groups (39/1/2 layers; groups
[[0..7, 10..40], [8], [9, 41]]) each need 288 slots (one whole expert layer) at
10878976/15794176/13303808 bytes/slot, a byte-weighted minimum of 1059
layer-0-width slots vs the planned 1035; lower --kv-reserve-tokens or raise
--memory-ratio
Traceback (most recent call last):
  File ".../engine/engine.py", line 447, in __init__
    self._init_offload_moe_cache(config)
  File ".../engine/engine.py", line 919, in _init_offload_moe_cache
    self._gguf_auto_floor_gate(config, method, self._resolve_auto_moe_cache_size)
  File ".../engine/engine.py", line 860, in _gguf_auto_floor_gate
    check_partition_floors(groups, per_group_slots, config.model_config.num_experts, envelope)
  File ".../engine/cache_budget.py", line 77, in check_partition_floors
    raise ValueError(
```

Boot numbers today: `Free memory before loading model: 28.97 GiB`; need 1059, planned 1035.

## 3. Root cause

The gate math (`engine/cache_budget.py:68-83`, called from `engine.py:860`) computes
planned slots = (0.80 x baseline_free - weights - fixed_cache - kv_reserve_bytes) //
slot0_bytes and rejects if below the byte-weighted floor of 1059. Solving the
identical formula against the archived 09-20 wave (tree 315b278):

| boot | baseline_free | planned | verdict |
|------|---------------|---------|---------|
| 09-20 b2, winner 0.80/400k | 29.32 GiB | 1062 | PASS (+3 over floor, razor thin) |
| 09-20 b, 0.80/500k | 29.32 GiB | 1003 | FAIL (archived fail-fast) |
| 09-20 c, 0.75/400k | 29.32 GiB | 917 | FAIL (archived fail-fast) |
| today, 0.80/400k | 28.97 GiB | 1033..1035 (observed 1035) | FAIL |

The closed-form model reproduces all four plans (also boot a 0.85/500k = 1148 exact)
with weights+fixed as the single fitted constant and only baseline_free varying.
Delta explanation: today's desktop stack used 1908 MiB pre-boot vs 1551 MiB on
09-20 (from `.tasks/dense-q80-gemm/vram-headroom.md` census) = **+357 MiB**;
0.8 x 357 MiB = 286 MiB budget = 27.5 slots; 1062 - 27 = 1035 - the observed plan
exactly. Pass threshold at 0.80/400k is baseline_free >= ~29.28 GiB; the 09-20
winner had 29.32 GiB (+3 slots = ~39 MiB of headroom), so any ~40+ MiB rise in
desktop VRAM flips this recipe to fail-fast - consistent with the known
hoptodesk-flip behavior from 2026-09-17.

Not a regression:
- `git log 315b278..5001504 -- python/freetoken/engine/cache_budget.py` = empty;
  the gate arithmetic is untouched by merge round 3.
- engine.py merge hunks on this path only (a) add the FTW-index fallback scan
  `gguf_ftw_signature_groups` to `_gguf_auto_floor_gate` (fires the gate for FTW
  paths that previously skipped it), (b) wrap `load_expert_banks` in
  `_weight_load_context()`. Neither changes the envelope or floor math. The
  traceback does not pass through `_weight_load_context` or the tvm-ffi arch pin.
- Even without the early gate, the late safety net `check_partition_floors` at
  `engine.py:774` rejects the same 1035-slot plan post-load - pre-merge the command
  would fail identically, only after the multi-minute bank load. Round 3 moved the
  failure ~9 s earlier, it did not create it.
- fp8 angle checked: the crash path has no KV-dtype branch (KV dtype only enters
  as cache_per_page arithmetic, identical config both days). No fp8-specific bug.
- Merge-regression stop condition (step 4 of the task) not triggered.

## 4. Options (from the code's own math; NOT applied, not a "fix")

Minimal bootable changes at today's 28.97 GiB (planned ranges from the fitted model):

1. `--memory-ratio 0.81` (rest unchanged): planned 1062..1064 - boots, but +3..+5
   slots, same razor-thin class as 09-20; a further ~50 MiB desktop rise can flip it.
2. `--kv-reserve-tokens 350000` at 0.80: planned 1063..1065 - boots; 360000 is
   borderline (1057..1059). Measured costs (09-20 wave): reserve cut changes peak
   VRAM by ~0 (planner refills KV into slots; peak tracks the ratio envelope) and a
   351k pool still radix-HITs 65536-token contexts (09-20 boot d).
3. Robust margin: `0.82` (or `0.81` + 350k): planned 1091..1093 (+32), tolerates a
   ~400-500 MiB desktop swing. Cost ~300 MiB peak-free per 0.01 ratio (measured
   1500 MiB per 0.05) -> ~1400-1700 MiB peak-free, eating into the 1.5-2 GiB target.

Reminder: the recipe itself is config-infeasible-by-margin on this desktop; raising
ratio trades the measured 1.96 GiB peak-free for boot robustness. `--num-tokens`
explicit KV cap + `--moe-cache-size 288` is the other known direction but costs the
336 working set (decode -4.5%, deep-fill fragile).

## 5. Artifacts

- `.tasks/ft-serve-crash-mr080-400k/serve.log` - full server log incl. traceback
- `.tasks/ft-serve-crash-mr080-400k/runner.out` - runner stdout (FAILPAT hit + verdict)
- `.tasks/ft-serve-crash-mr080-400k/repro-runner.sh` - the runner
- 09-20 references: `.tasks/dense-q80-gemm/vram-headroom.md`,
  `.tasks/ft-gguf-serve-tuning/logs/vram_{a,b,b2,c}.log`

## 6. Census

Before: no pytest/ft serve/benchbw/ft_run procs; sem.mp- 18; 1908 MiB used (hoptodesk
2x302 MiB compute + desktop baseline). After: procs none; sem.mp- 18; 1870 MiB used /
30279 MiB free - clean, no leaks.
