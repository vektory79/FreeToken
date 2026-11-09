---
name: "ft-kv-quant-ab-campaign"
description: "kv-quant-ab campaign complete: speed flat, NIAH 60/60 all modes, walls 653k/1.04M/1.03M; fp8 sweet spot"
type: project
lastUpdated: 2026-10-31T00:22
lastRecall: 2026-10-30T21:01
---

# kv-quant-ab campaign: bf16/fp8/nvfp4 KV-quant comparison on GLM-5.3-Flash GGUF (2026-10-30, ALL WAVES DONE)

User goal: compare all `--kv-cache-dtype` modes (bf16/none, fp8, nvfp4) for GLM-5.3-Flash GGUF
by quality AND decode speed on long sessions, on the user's exact serve config
(0.81 / 400k / 4095 / hybrid / threads 16 / session tier 10+50 / FTW-vision-new, RTX 5090).
Plan: `.tasks/kv-quant-ab/PLAN.md`; harness + logs + summarize_w2.py + niah.py in `.tasks/kv-quant-ab/`.

## FINAL VERDICT (all measured 2026-10-30)
- SPEED: flat everywhere - decode 21.0-23.6 tok/s across modes, slot counts (599-1164),
  and depths (71k -> 1.04M). Prefill 1230-1260 tok/s all modes. KV quant is a CAPACITY lever only.
- CAPACITY (auto plans at 0.81): bf16 870 slots / wall ~653k tokens; fp8 1071 slots /
  wall ~1.037M; nvfp4 1144 slots / wall ~1.025M. Ceiling ladder (boot+64k fill):
  bf16 655360 PASS / 786432 FAIL; fp8, nvfp4 1048576 PASS.
- QUALITY: NIAH 4 lengths x 5 depths x 3 legs = 60/60 HIT in all modes (deepest session
  352162 real tokens, needles at 10/30/50/70/90%). No retrieval degradation from fp8/nvfp4.
- RECOMMENDATION: fp8 = sweet spot for long sessions (2x bf16 capacity, same speed);
  nvfp4 for >800k sessions (+7% slots over fp8, only mode with zero depth sag);
  bf16 only if a future quality probe demands it. User keeps fp8 config.

## Matrix (0.81/400k/4095, fill ~71-74k nonce, decode @71k)
- bf16: 870 slots, KV 4.46 GiB (11.14 KB/tok), peak used 30234 MiB, prefill 1229.5, decode 22.48
- fp8: 1071 slots, KV 2.38 GiB (6.22 KB/tok), peak 30415 MiB, prefill 1241.0, decode 22.77
- nvfp4: 1144 slots, KV 1.46 GiB (3.81 KB/tok), peak 30169 MiB, prefill 1257.8, decode 22.83
Free after init ~5.2 GiB all legs; peak-invariance of kv-reserve confirmed (bf16 peak == fp8's).

## W3 decode-at-capacity (growing-prefix session, decode from last good depth)
- bf16 @653k: 21.02 steady (-6.5% vs own @71k); fp8 @1.037M: 21.84 (-4.1%); nvfp4 @1.025M: 23.24 (+1.8%, flat).
- Capacity wall = HTTP 400 on the next 22.6k fill step at the reserve edge (expected).

## W4 quality (niah.py: 5 needles/doc at depths 10-90%, turn-based Q&A session, kb battery)
- NIAH: bf16/fp8/nvfp4 ALL 20/20 (64k, 150k, 250k, 376k x 5 depths; 352k real tokens deepest).
- Battery 24-prompt (v3 baseline set): all legs failed=0; cross-leg verbatim identity
  fp8 1/24, nvfp4 0/24 - but comparison is cross-BOOT, and boot nondeterminism of temp0 on
  this stack dominates; dtype effect not separable without arbitration boots (W5 candidate).
  Precedent: PR #408 same-day battery was byte-identical bf16-vs-nvfp4 on 12 tasks.

## Track A note
Equalized --moe-cache-size NOT RUN - measured speed flatness makes it moot for speed;
would matter only if a future quality probe shows slot-count-dependent generation quality.

## Harness (`.tasks/kv-quant-ab/`)
measure.py (isolated DIR copy), gen_filler.py nonce filler, run_w1/w2/w3/w4 scripts,
deepfill.py (deep fill + decode-at-depth), niah.py (NIAH + battery), summarize_w2.py.
Lessons: measure.py always rc=0 (grep '"ready": true'); --log omits logs/ prefix;
--extra overrides BASE_ARGS by last-wins; driver crashes must teardown in except
(NameError orphan at 26.9 GiB once); readlink -f resolves .venv python to linuxbrew base
(kb kill-pattern mismatch - kill by verified PID); TCPStore EADDRINUSE between boots ->
sleep 20; doc generator BYTES_PER_TOKEN=4.62 for niah shape (5.25 was the old filler;
8% overshoot from uncounted line prefix); battery max_tokens from baseline artifacts
(glm reasoning parser burns short budgets).
