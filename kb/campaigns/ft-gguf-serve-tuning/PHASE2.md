# Phase 2: radix-reuse isolation (fp8 KV) + clean decode

DEVIATION: overnight hoptodesk (~492 MiB) shrank baseline VRAM; BASE@0.89 fails the slot-floor gate (needs 1059 slot-0-eq, plan gives 1056). B_A/CAL/base_decode2 run with --memory-ratio 0.90; B_B/B_C unchanged (boot fine today).

Prompt alignment: {"A": {"try1_prompt": 1134, "try1_err": null, "try2_prompt": 1152, "final_prompt_tokens": 1152, "aligned": true}, "B": {"try1_prompt": 9622, "try1_err": null, "try2_prompt": 9664, "final_prompt_tokens": 9664, "aligned": true}, "A_chars": 6108, "B_chars": 50427}

## Reuse probes (max cached tokens of the REPEAT request)

| boot | config | A2 | B2 | A3 (60s) |
|---|---|---|---|---|
| B_A | --memory-ratio 0.90 | 1088 | 9600 | 1088 |
| B_B | --max-running-requests 1 | 1088 | 9600 | - |
| B_C | --max-running-requests 1 --max-prefill-length 8191 --memory-ratio 0.85 | 1088 | 9600 | - |

## Decode (patched metric)

| run | steady | p50 | cached/new |
|---|---|---|---|
| base_decode2 | 13.75 | 13.76 | 0/49 |
| base_mr1_decode2 | 14.12 | 14.43 | 0/49 |

Known reference: win_p_8191_r085 decode steady=13.58 p50=13.43 cached=65536 (mr1+8191+0.85)

## 65k-reuse disambiguation (2026-09-17, daytime; hoptodesk active, baseline free 29.38 GiB)

| config | ready_s | prefill median (full chunks) | decode steady/p50 | decode cached/new | verdict |
|---|---|---|---|---|---|
| mr1 + 8191-chunks + ratio 0.89 (dis_chunk089) | OOM | - | - | - | INFEASIBLE: CUDA OOM in first-prefill autotune (256 MiB vs 220.5 MiB free; free-after-init 2.88 GiB) |
| mr1 + 4096-chunks + ratio 0.85 (dis_ratio085) | 108.4 | 263.8 (4096) | 15.04 / 14.72 | 0/49 | MISS |

VERDICT: the 65k radix-reuse switch on fp8 KV is PREFILL CHUNK SIZE (8128 vs 4096), not memory-ratio, not max-running-requests, not inter-request timing, not KV dtype:
- 4096-chunks: MISS at ratio 0.89, 0.90 and 0.85 -> chunk size alone reproduces the miss across ratios.
- 8128-chunks: HIT at 0.85 (win + ladder); at 0.89 the config cannot survive the first prefill (OOM) -> ratio's only role is funding 8128 transients.
Practical consequence: {mr1 + max-prefill-length 8191 + memory-ratio 0.85} is the ONLY measured configuration with both +prefill (+21%) and 65k radix reuse on fp8.
Decode note: 15.04 (4096/0.85) vs 13.58 (8128/0.85) = ~-10% for the winner; caveat - single 63-token sample per config, daytime CPU load (CPU-leg-bound decode), treat as indicative.
Mechanistic hint (for code work): at 4096-chunking the 65536-token boundary falls exactly ON the 16th chunk end, yet its snapshot is lost; at 8128-chunking the boundary falls mid-chunk (x64-tracked) and survives. Suspects: scheduler/cache.py:383-414 (per-chunk donation) and glm5_next/kda.py:166-178 (snapshot write).
Hygiene: serial runs, census-clean between boots; dis_chunk089 measure.py SIGTERMed after backend death; final census clean (no ft serve, GPU back to baseline).

## Back-to-back decode A/B (dab, finished by dab_finish.py after dab.py died mid-run)

BASE r090 note: 0.89 fail-fasted on the early floor gate again (VRAM deficit from hoptodesk); the dab.py retry booted with duplicated --memory-ratio flags (effective 0.90).

| boot | sample | steady tok/s | p50 tok/s | ttft s | completion | wall s |
|---|---|---|---|---|---|---|
| base_r090 | base_r090_0 | 13.68 | 13.73 | 0.0 | 63 | 256.1 |
| base_r090 | base_r090_1 | 13.33 | 13.61 | 0.0 | 63 | 257.1 |
| win | win_0 | 14.91 | 15.02 | 1.1 | 63 | 8.6 |
| win | win_1 | 14.41 | 15.61 | 0.0 | 63 | 7.7 |
| win | win_2 | 14.84 | 15.66 | 0.0 | 63 | 7.6 |

mean steady: base=13.50 win=14.72 delta=9.0%
