---
name: "ft-gate-baseline-nan-flake"
description: "Full-gate baseline flake: test_gguf_expert_banks NaN isfinite only under suite ordering (Environment); gate 2135 passed"
type: project
lastUpdated: 2026-09-18T21:45
lastRecall: 2026-09-19T00:42
---

# Full-gate baseline addition (2026-09-18, fix1-radix-track-seqlen wave)

- tests/models/test_gguf_expert_banks.py::test_grid_caps_raise_before_kernel_launch: NaN isfinite assert fires under FULL-SUITE ordering only. Passes in isolation both with the wave's prefill.py fix reverted to HEAD and with the fix in place -> Environment class (suite-ordering/GPU-state), NOT a regression.
- Scale of that gate run: 2135 passed / 6 failed / 206 skipped (`.venv/bin/python -m pytest tests/ -m "not slow" --ignore=tests/models/test_glm5_next_kda_snapshot.py` on the main checkout).
- muse_glimmer tests-import and the "No module named 'tests'" import-mode items did NOT appear under `python -m pytest` (import-mode fix works).
- How to apply: classify this test as pre-existing flake in future full-gate triage; do not chase; verify with isolated run before blaming a diff.
