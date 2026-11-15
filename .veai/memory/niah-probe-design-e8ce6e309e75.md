---
name: "niah-probe-design"
description: "NIAH probe recipe GLM GGUF: own-line needle, 160-token budget, 5.53 ch/tok depth calibration, deepest-first radix reuse"
type: project
lastUpdated: 2026-11-15T18:21
---

# NIAH probe design on the GLM GGUF rig (validated 2026-11-15)

Working recipe from the route-affinity gate harness (.tasks/route-affinity/run_arm.py, niah_items(); smoke smk001: 2/2 pass, finish=stop). kb/cases/kv-quant-ab does NOT exist on disk - this memory is the durable recipe (kv-quant-ab's own NIAH methodology was never mirrored to kb).

- Needle ON ITS OWN LINE ("\nService note: the access code for key ARC-<nonce> equals <V>.\n" inserted into the repeated-record filler). Mid-sentence insertion made the model treat it as a text anomaly; with a tight budget it then missed the 40k needle (1/2 in smk000).
- Budget 160 tokens, not 96: the reasoning scan of a 40-60k-token context needs room; 96 truncated the scan mid-search -> false miss. finish_reason=stop confirms sufficiency. Auto-check = expected value substring in content+reasoning (glm parser keeps answers in reasoning_content on short budgets).
- Depth -> char offset calibration: offset = depth_tokens * len(filler) / fill_prompt_tokens. filler.txt (362,472 B) = 65,585 prompt tokens = 5.53 chars/token on GLM-5.3-Flash-UD-Q3_K_XL.
- Run needles DEEPEST-FIRST: radix cache shares the 64k fill prefix up to each needle offset, so each query prefills only ~ (64k - depth) tokens.
- Unique nonce+value per query (radix must not serve a previous answer); keep keys/values FIXED across A/B arms for comparability.
- Gate threshold is RELATIVE to the base arm ("not below base - 2pp"), not absolute: occasional deep-mid misses are model-capability noise, not harness error.

Why: two ~8-min smoke runs separated harness bugs (budget, insertion style, statelessness) from genuine retrieval misses; without them every arm would carry the same false-miss noise.
How to apply: any long-context retrieval/quality probe on this rig; reuse niah_items() from run_arm.py.
