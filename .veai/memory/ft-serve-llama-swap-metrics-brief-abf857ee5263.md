---
name: "ft-serve-llama-swap-metrics-brief"
description: "llama-swap stats accepted on HW; endpoint /api/metrics/activity; D4 tail-chunk inflation; page_size=64 quirk"
type: project
lastUpdated: 2026-09-26T19:12
lastRecall: 2026-09-26T19:01
---

# llama-swap stats for ft serve: W1+W5 ACCEPTED on hardware (2026-09-27)

Brief: [kb/cases/serve-llama-swap-metrics/TASK.md], report:
[kb/cases/serve-llama-swap-metrics/REPORT.md]. Code W2+W3+W4 done (uncommitted
unless user committed), review-verified (wire format vs llama-swap v259
metrics.go: timings overrides usage; omitted rate keys parse as 0.0; cache_n
feeds Cached). Hardware wave accepted same day. Full gate green vs baseline:
2325 passed, 16 baseline-class fails (import-mode + hardware, set byte-identical
to 2026-09-19).

Key facts (verified on hardware, do not re-derive):
- Live config /media/ai/soft/llama-swap/config.yaml CHANGED since 09-26: only
  2 ft serve sections remain - GLM-5.3-Flash-gguf (4 think-variants) and
  Qwen3.8-Flash-Next; Qwen3.6-35B-A3B, Qwen3.8-27B, gemma-4-E4B are now
  llama.cpp-backed (stats natively). W1 applied to live (8 stream_options
  blocks + 2x --enable-cache-report; backup config.yaml.bak-w1-20260926).
  PROD venv ft serve supports --enable-cache-report but LACKS W2 code - live
  instance shows only Prompt/Generated/Cached until dev code is deployed.
- llama-swap v259 stats endpoint: GET /api/metrics/activity (webui Requests
  table). Verified all 8 columns on test instance (dev venv), Duration within
  17-25 ms of client wall clock.
- Measured: GLM chunked probe 24013 tok - prompt_per_second 806.63 (validated
  ~800-805 band), timings sum = wall - 75 ms; TTFT correctly excluded from
  predicted_ms; cache repeat cache_n=832 -> rate over NEW tokens 0.27.
- KNOWN ISSUE D4: tail (non-full) chunk of multi-chunk prefill still logs
  inflated tok/s (16851 measured; overlap drain-lag collapses own-window).
  Request-level timings honest; per-chunk log line not. Open fix decision.
- Quirk: radix page_size=64 -> repeats shorter than 64 tokens honestly give
  cache_n=0.
- v259 quirk: Duration = max(wall, prompt_ms+predicted_ms); boot rows = wall.
- Harness/artifacts: .tasks/ls-metrics-test/ (make_work_config.py, client.py,
  verify_activity.py, start/stop_instance.sh, failpat_watchdog.sh + artifacts).

How to apply: deploy decision (dev -> PROD venv) is the user's; live webui
gains Prefill/Decode only after that. D4 fix is an open follow-up. Anthropic
message_delta rates = desirable follow-up (GenDone already carries values).
