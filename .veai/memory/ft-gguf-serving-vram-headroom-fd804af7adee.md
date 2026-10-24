---
name: "ft-gguf-serving-vram-headroom"
description: "GGUF glm5next VRAM: post-round5 matrix in USED MiB (5 configs + ladders), 0.80 floor, kb report used-based"
type: project
lastUpdated: 2026-10-24T20:32
lastRecall: 2026-10-24T19:08
---

# GGUF glm5next serving VRAM headroom: measured config matrices (RTX 5090, 32607 MiB)

## Post-round-5 wave (2026-10-24, HEAD ddefb5e ggml re-platform stack; artifacts .tasks/vram-regression-post-round5/)

USER SYMPTOM: the 09-20 winner profile (0.82/400k/8191, FTW-vision-new carrier) fills VRAM and OOMs at generation. Boot numbers look normal (plan 1127 slots/6256 pages, free-after-graphs 4.97 GiB); the OOM is peak transients exceeding the (1-ratio) remainder.

CAUSES (static diff tag 2026.10.17=b8c3794 vs HEAD + measured): plan machinery byte-identical (cache_budget.py:38-45); NEW off-plan consumers: (1) Triton split-KV decode scratch _MAX_KV_SPLITS 8->64 SM-scaled (kernel/triton/attention.py:43-76, #729) - fp32 logits in graphs; (2) ggml-cuda MMQ fp32-dequant x32 scratch peaking at big-batch prefill (csrc/ggml); (3) torch 2.14.1/triton 3.8/flashlib baseline shift; (4) new carrier slot widths. NOT causes: vision (host-streamed default, ~60 MiB resident), CUDA graphs (bs=1 only, ~30 MiB), --enable-cache-report (reporting-only), session tier (RAM/SSD).

MEASURED MATRIX (unique-nonce prompts - session-tier/radix restore otherwise fakes cache hits and hides peaks; desktop 796-1000 MiB across boots). METRIC = PEAK USED VRAM (nvidia-smi used MiB; user directive 2026-10-24: reports carry used, not free - free = 32607 - used - desktop, reader computes):
- 0.82/400k/8191: peak used 31966 MiB (user's OOM-prone config)
- 0.80/400k/8191: 31368 MiB, prefill 2129.4 tok/s - boot razor-thin (1069 slots)
- 0.82/400k/4095: 30510 MiB, prefill 997.5 tok/s (-53.2%), decode 21.2 tok/s - ROBUST RECIPE
- 0.81/400k/4095: 30241 MiB, prefill 1094.5 tok/s, 1083 slots
- 0.80/400k/4095: 29941 MiB, prefill 1081.7 tok/s, 1053 slots @ desktop 1000 MiB
- Factors (script-computed): ratio step -0.01 = -299 MiB used @8191 / -284.5 @4095; chunk 8191->4095 = -1456 MiB @0.82 / -1427 @0.80; slot deltas: 0.82->0.81 = -44 slots, 0.81->0.80 = -30, +204 MiB desktop = -16 slots (1069->1053); depth erosion +339..637 MiB used by 378660 tokens.
- Deep-fill ladders to 378660 tokens (94.57% of the 400384 pool), all HTTP 200: 0.82+4095 max used 31427 MiB; 0.80+4095 max 30693 MiB. Ratio floor 0.80 at reserve 400k (0.79 infeasible). kv-reserve peak-invariant by planner arithmetic (no A/B on current stack).

RECIPES: robust default = 0.82 + --max-prefill-length 4095. Balance = 0.81/4095. Min used = 0.80/4095 (light desktop for boot). Full speed = 0.80/8191 (light desktop). Avoid 0.82/8191. Side finds: benchbw profile stale for the new engine (version 4 vs 5, ignored -> hybrid fetch cap 1; rerun `ft bench bw`); session tier L2 dir was 49.94/50 GiB near cap; curl bodies >128 KiB must go via stdin (-d @-), not argv.

## Old-stack matrix (2026-09-20, pre-re-platform @ 315b278, artifacts .tasks/dense-q80-gemm/vram-headroom.md)

Winner was 0.80/400k/8191 -> 2008 MiB peak-free, prefill 804.85 tok/s, decode 14.68. Ratio floor 0.80 with reserve cut to 400k. kv-reserve 500k->350k = 0 MiB peak change. Chunk 8191->4095 = +2020 MiB free at -25.9% prefill. Flip condition: +357 MiB desktop swing flipped 0.80/400k boot.

## kb pointer (2026-10-24, kb-distill)

Canonical self-sufficient report (USED-VRAM metric, script-computed numbers only): kb/topics/gguf-serving-perf-matrix.md - 5 configs x prefill/decode/used, both depth ladders, slot-floor bounds, recipes. Also kb/topics/gguf-vram-headroom-post-round5.md (causes, L2-cap semantics), kb/methods/vram-peak-measurement.md (2s sampler, nonce'd prompts, growing-prefix ladder, stdin payloads, exe-checked kill).
