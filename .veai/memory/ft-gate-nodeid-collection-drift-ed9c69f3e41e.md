---
name: "ft-gate-nodeid-collection-drift"
description: "Full-gate nodeid counts drift across runs: test_quant_config.py collection-time glob parametrization; key on failure set"
type: project
lastUpdated: 2026-09-19T18:12
lastRecall: 2026-09-26T19:11
---

# Full-gate collected-nodeid drift: test_quant_config.py collection-time glob parametrization

Observed 2026-09-19 (dense-q80-gemm N-split wave, review-b-nsplit.md B2).

- tests/models/test_quant_config.py parametrizes at COLLECTION time over globbed model directories (HF cache + /mnt/nvme/models). The collected-nodeid total therefore drifts between full-gate runs on the SAME tree: observed 2473 vs 2410 (= 63 nodes) with IDENTICAL failure sets.
- Why it matters: a "my change added/removed tests" false alarm, or a mis-sized baseline comparison, when the only thing that changed is the model-directory glob contents on disk.
- How to apply: full-gate comparisons must key on the per-test FAILURE SET, not on collected totals. When a count anomaly appears: re-run `--collect-only`, confirm the reproduced count, and treat that as authoritative (the implementer's run 2 + collect-only was justified this way; run 1's log was absent). Also note the standard invocation: `uv run --extra dev pytest tests/ -m "not slow" --ignore=tests/models/test_glm5_next_kda_snapshot.py`; 2026-09-19 run: 16 failed (all baseline: 11x "No module named 'tests'" import class, pinned_tensor UVA, qsa_fp8 [1-1-64], qwen4_exp/test_ple :421, 2x glm_dsa OutOfResources) / 2173 passed / 206 skipped.
