---
name: "ft-gate-baseline-nan-flake"
description: "Cap-test NaN flake root-caused to torch.empty x and FIXED in v3a (zeros); gate scale 2135 tests; classification method"
type: project
lastUpdated: 2026-09-19T02:56
lastRecall: 2026-09-19T18:15
---

# Full-gate baseline addition (2026-09-18, fix1-radix-track-seqlen wave)

- tests/models/test_gguf_expert_banks.py::test_grid_caps_raise_before_kernel_launch: NaN isfinite assert fires under FULL-SUITE ordering only. Passes in isolation both with the wave's prefill.py fix reverted to HEAD and with the fix in place -> Environment class (suite-ordering/GPU-state), NOT a regression.
- Scale of that gate run: 2135 passed / 6 failed / 206 skipped (`.venv/bin/python -m pytest tests/ -m "not slow" --ignore=tests/models/test_glm5_next_kda_snapshot.py` on the main checkout).
- muse_glimmer tests-import and the "No module named 'tests'" import-mode items did NOT appear under `python -m pytest` (import-mode fix works).
- How to apply: classify this test as pre-existing flake in future full-gate triage; do not chase; verify with isolated run before blaming a diff.

## SUPERSEDED (2026-09-18, v3a wave): flake root-caused and gone
The NaN isfinite flake in test_grid_caps_raise_before_kernel_launch is FIXED by the v3a changeset (uncommitted on vektory79 at measurement time): the test's x tensor was torch.empty (uninitialized bf16 garbage ~1e38 overflows the q8_1 fp16 scale -> inf/NaN path); switching to torch.zeros removes the non-finite input directly. Review-B traced the causality; the flake did not fire in the v3a full gate (2383 passed / 5 pre-existing baseline). How to apply: this test no longer needs the Environment-class waiver; if a NaN fires there again, suspect a NEW cause (or a concurrent-VRAM wave per the cross-wave OOM rule), not this historical one. Gate-scale numbers below remain valid history.
