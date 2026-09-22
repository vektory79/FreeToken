# Phase 6 - full A/B: GGUF offload vs NVFP4-FTW baseline (RTX 5090, 2026-09-14)

All runs on tree f801a08 (Phase 5 committed). Same 64k prompt (64,212 tokens on BOTH
tokenizers), same method, same 4096 chunk size, same box, same day. Watchdog runner
(FAILPAT every 5s + trap cleanup) for every session; both servers shut down cleanly
(SIGTERM -> exit 143 by design).

## Final tuned GGUF command (Config A, verbatim)

    ft serve --model-path /media/ai/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL.gguf --moe-strategy offload --moe-cache-auto --num-tokens 70016 --memory-ratio 0.85 --max-prefill-length 4096 --port 18081

Tuning trail (Phase A): auto plan gives only ~8-9k KV tokens (accepted baseline
evidence: auto resolved moe_cache_size=1368, KV 8256 tokens, 2.83 GiB free). Explicit
`--moe-cache-size 864 --num-tokens 70000` -> fail-fast: "--num-tokens 70000 is not a
multiple of the resolved page size 64" (use 70016). Then explicit 864 -> fail-fast:
"--moe-cache-size 864 cannot fund the per-signature slot floors ... raise it to at
least 8510 or let --moe-cache-auto size it" - explicit slot sizing CANNOT shrink below
the auto fill for this 3-signature file (per-partition byte floors). Working lever:
auto slots + capped KV + lower ratio. `--memory-ratio 0.85` gives free-after-init
3.34-3.36 GiB (>= the ~3.5 GiB target within noise; 4096-chunk transients passed with
margin, 16 chunks, no OOM; 8192 chunks would need ~3.5+ and were NOT attempted).

Baseline deviation (Config B'): the canonical 1M-reserve command FAILS on today's VRAM
(fail-fast: minimum plan 8.19 GiB > budget 6.63 GiB; the same command booted 2026-09-12
- environment VRAM drift). Measured baseline = same flags with --kv-reserve-tokens
524288 (the previously-verified B1 config): moe_cache_size=322, KV 526,784 tokens.

## Phase B - A/B table (same session per config, same prompts)

| metric | GGUF offload (tuned) | NVFP4 hybrid (B') | ratio |
|---|---|---|---|
| boot to /ready 200 (warm triton) | 115 s (bank build ~93 s inside) | 56 s | 2.05x slower |
| prefill 64k fill, server steady | 263.9-268.1 tok/s (chunk1 warmup 93.1 excluded) | 510.3-515.2 (warmup 60.0 excluded) | GGUF = 0.52x |
| prefill client wall | 64,212 tok / 255.9 s = 251 tok/s | 64,212 tok / 177.5 s = 362 tok/s | 0.69x |
| decode cached @64k (256 req) | 14.67 tok/s, TTFT 3.29 s | 14.08 tok/s, TTFT 5.12 s | GGUF +4.2% |
| decode cached (768 req, natural EOS stop) | 14.28 tok/s @ 363 tok | 14.16 tok/s @ 232 tok | parity |
| decode fresh short (256 req) | 13.00 tok/s, TTFT 3.21 s | 14.18 tok/s, TTFT 4.23 s | 0.92x |
| plan | 1227 slots, KV 70,016 tok (0.78 GiB), 3 partitions, overlap OFF on ALL | 322 slots, KV 526,784 tok (1.92 GiB) | - |
| VRAM during serve | 28,817 / 32,607 MiB | 29,371 / 32,607 MiB | - |
| free after init | 3.34 GiB | 2.81 GiB | - |
| RAM during serve | 137 GiB used (pinned banks, of 188) | (not sampled, host banks smaller) | - |
| battery avg wall/prompt | 14.7 s | 15.2 s | parity |

Prefix-cache hit verified on both ("#new-token: 20, #cached-token: 64192"). Shutdown
both: exit 143, teardown 13-16 s, VRAM back to desktop baseline (1313-1410 MiB), no
compute processes.

## Phase C - quality battery (24 greedy prompts, temp0+top_k1 accepted on both)

- identical texts: 0/24; first divergence at char 0 in 19/24 (rest at 15-21 chars).
- ON-TOPIC answers (keyword-in-answer heuristic): GGUF 3/24 vs NVFP4 24/24 (heuristic
  flags 8 RU prompts OFF on NVFP4 only because the reasoning paraphrases in English;
  their content is in Russian and correct - effectively 24/24).
- GGUF failure mode: the model's reasoning conditions on a DIFFERENT task entirely -
  "solve a math problem based on the provided image", "find the area bounded by
  y=sin(x)...", GUI-agent JSON ({"action_type": "click", "point": [640, 360]}),
  bounding-box detection, random package/script generation - i.e. the user prompt
  effectively never reaches the model. Deterministic across two independent boots
  (session A and A2 battery runs agree).
- NVFP4 side: every prompt answered on-topic, correct reasoning (RU answers in Russian).

VERDICT: the GGUF chat path is BROKEN at the conversation-rendering layer (tokenizer /
chat-template / role rendering into the gguf vocab), NOT in the offload machinery -
performance is production-acceptable, quality is not. This is a Phase-3-adjacent
defect (embedded-vocab gpt2 tokenizer + template rendering), to be root-caused via the
orchestrator; the offload/kernels/serving stack itself did not corrupt anything.

## Honest caveats

1. Quality: GGUF answers are unusable for chat as of this tree (3/24 on-topic). The
   performance numbers below are still valid as path measurements.
2. All partitions run prefill overlap OFF at the tuned plan (464+288+288 slots < 576
   threshold): synchronous materialized prefill everywhere. The dominant-group overlap
   ON would require >= 576 dominant slots (e.g. --moe-cache-size >= ~1430 with the
   byte floors) at the cost of KV/headroom; not measured here.
3. Baseline B' uses 512k reserve instead of the canonical 1M (today's VRAM budget);
   322 slots < 336 working set -> NVFP4 decode ~14.1 vs the 15.13 historical A1.
   Same-session A/B is unaffected (both configs measured today, same method).
4. Tokenizer divergence on code prompts (Phase 3 known): ids are NOT comparable; this
   battery compared decoded text only. Both tokenizers agreed on the 64k EN prompt
   (64,212 tokens each).
5. The 768-token decode runs stopped early on BOTH sides (EOS at 363/232 tokens) -
   natural stop, reported as-is.
6. GGUF TTFT is LOWER than NVFP4 (3.3 vs 5.1 s cached) - first-decode-step overhead,
   not re-prefill (prefix hit logged on both).

## Artifacts

- gguf-server.log (session A: battery + 49k-class prefill), gguf-server-probe.log
  (session A2: 64k prefill + decodes + battery rerun), nvfp4-server.log (session B'),
  ab_gguf.json, ab_gguf2.json, ab_nvfp4.json, battery_gguf/p00-23.json,
  battery_nvfp4/p00-23.json, compare.md (pairwise table + divergence examples).

## Anomalies during the campaign

- Harness bug (fixed in harness only): first client dropped the required "model" field
  in stream calls -> 422 on the three decode probes of session A; fixed, re-run as A2.
- First tuned attempt died on the page-size multiple check (70000 -> 70016) and the
  explicit slot floor (864 -> "at least 8510"); both fail-fast, no load, by design.
- NVFP4 canonical command fail-fast (above) -> single documented deviation (reserve
  1048576 -> 524288).
- No FAILPAT hits in any measurement session; no OOMs; teardown clean every time.
