---
name: "long-context-eval-api-benchmarks"
description: "Long-context eval via OpenAI API: RULER/NIAH/BABILong usable; PPL and lm-eval loglikelihood blocked (no logprobs)"
type: reference
lastUpdated: 2026-10-30T18:02
---

# Long-context quality eval usable through an OpenAI-compatible HTTP API (logprobs-free)

Landscape surveyed 2026-11 (searxng) for the kv-quant-ab campaign (GLM-5.3-Flash, FreeToken
server rejects logprobs - openai_api.py), so ONLY generation-based tools apply.

## Usable via HTTP generation
- NVIDIA RULER (github.com/NVIDIA/RULER, paper arxiv 2404.06654) - 13 tasks, 4 categories,
  configurable length/difficulty; connects to any OpenAI-compatible endpoint; smoke variant
  = 6 tasks, 4k-128k, ~90 min (aievals.co cookbook). THE main long-context metric.
- gkamradt/needle-in-a-haystack - v2 sweep (length x depth) via YAML, OpenAI SDK. Cheap
  detector for KV-quant damage.
- BABILong (arxiv 2406.10149) - QA to 1M+ tokens, generation-based.
- THUDM/LongBench v1+v2; OpenBMB/InfiniteBench (100k+, API-based eval);
  LCM-Lab/LOOM-Eval (github.com/LCM-Lab/LOOM-Scope) - explicit "API protocol: openai" long-
  context aggregator.
- ctxbench (vllm-dspark-nvfp4 deepwiki) - needle-passcode, vLLM OpenAI API.

## NOT usable (and why)
- Perplexity via API: needs token logprobs; FreeToken API rejects logprobs (openai_api.py).
- lm-evaluation-harness: local-completions tasks are loglikelihood (logprobs) - only
  generation tasks work; few long-context ones anyway.
- HELMET (arxiv 2410.02694): best aggregator but in-process (HF/vLLM weights required).
- llama.cpp llama-perplexity: own inference stack on weights - measures llama.cpp, not ft.
- KIVI: engine-level integration, not API.
- MTBench101: no public OpenAI-API runner found.

## Key external lesson for KV-quant A/B
vLLM blog "State of FP8 KV-Cache" (vllm.ai 2026-04-22): NIAH @128k caught fp8 KV accuracy
collapsing 91% -> 13% from imprecise FP32 accumulation in Tensor Cores; two-level
accumulation restored ~89%. FreeToken quant is STORAGE-only (dequant before attention math),
so this specific failure mode should not apply - but NIAH is exactly the detector that would
catch its equivalent. Tensormesh: FP4 ~3.56x tokens vs BF16, 1.78x vs FP8.
GLM-Flash/MoE: no public long-context benchmarks exist (vendor blogs only) - must measure
locally.
