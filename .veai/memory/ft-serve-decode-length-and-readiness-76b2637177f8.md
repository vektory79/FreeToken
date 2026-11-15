---
name: "ft-serve-decode-length-and-readiness"
description: "ft serve: ignore_eos pins greedy decode length; session-tier replay blocks real readiness for minutes"
type: project
lastUpdated: 2026-11-15T14:22
---

# Pinning decode length (ignore_eos) and the tier-replay readiness lag (2026-11-15)

- Greedy (temperature 0, top_k 1) decode of the SAME 64k filler prompt hits EOS at a boot-to-boot VARYING token count: 545 observed (offline-coverage-curve attempt-1), 467..763 historically (decode-research). Any experiment needing an exact decode length (per-step traces, fixed step counts) must pin it: the chat completions API supports `"ignore_eos": true` (req.ignore_eos, openai_api.py / api_models.py) - with it, max_tokens=768 yields exactly 768 streamed tokens every boot.
- Real readiness can lag "API server is ready to serve" by MINUTES after a heavy session: session-tier replay (280+ records over a full 49.9/50 GiB L2) plus idle-loop compact block request processing while probes keep failing. Observed: API-ready at +55 s, first successful real request at +336 s; the failed readiness probes still get PROCESSED by the engine (13-token "ping" prefills + 1 decode step each, one "Prefill batch" log line every ~4 s).
- Consequences: (a) verify readiness with a REAL chat request and a generous timeout (the decode-research harness pattern), never a cheap endpoint; (b) probe decode steps land in any per-step trace and cumulative telemetry - consistent on both sides, so replay-style validation stays correct, but expect a few extra steps beyond the main decode.
- Fill of the 64k prompt: ~84 s cold (~780 tok/s server prefill at chunk 4095) vs 4.5-4.8 s when the session tier restores the same prompt from L1 (radix hit).

**Why:** cost one wasted boot each (EOS-truncated trace; stats window confusion during the readiness lag).
**How to apply:** any serve experiment with fixed step counts sets ignore_eos; any readiness wait on this rig budgets minutes for tier replay after heavy sessions.
