---
name: "ft-serve-test-and-e2e-gotchas"
description: "FreeToken test/e2e gotchas: --extra dev pytest, tests-import blocker, ft serve SIG_IGN worker gotcha, tqdm mp-lock"
type: project
lastUpdated: 2026-09-05T19:27
lastRecall: 2026-09-11T22:15
---

# FreeToken test/e2e environment gotchas

Durable operational facts discovered 2026-09-04 while fixing `ft serve` /ready + graceful shutdown (plan package: .tasks/health-and-graceful-shutdown-tasks/PLAN.md, e2e evidence: .tasks/health-and-graceful-shutdown-work/verification/e2e-report.md).

## Test invocation
- CONTRIBUTING's `uv run pytest tests/ -m "not slow"` fails two independent ways:
  1. The venv is split: `.venv/bin/python` is 3.14 (fastapi+freetoken, NO pytest) while `.venv/bin/python3` is 3.12 (pytest, no fastapi). Working invocation: `uv run --extra dev pytest ...`.
  2. Collection aborts: `tests/models/test_glm5_next_kda_snapshot.py:21` and `tests/server/test_muse_glimmer_parsers.py:872` do `from tests.` imports and there is no `tests/__init__.py` -> "ModuleNotFoundError: No module named 'tests'". Add `--ignore=tests/models/test_glm5_next_kda_snapshot.py`; the muse_glimmer one then fails as a test (same root cause).
- Baseline failing set on this box (2026-09-04, unchanged across all runs, NOT regressions): pinned_tensor UVA driver error, swiglu_clamp CUDA fallout, qwen4_exp/test_ple flaky bitwise GPU assert, muse_glimmer tests-import. Everything else green (~1560 passed).

## ft serve e2e recipe
- Launch on port 18081 (user's wrapper occupies 18080), as a background job; RTX 5090: load ~42-49s; `GET /ready` gates readiness (503 during load -> 200 when serving; /health always 200 with body status); SIGTERM -> graceful reap, exit 143 BY DESIGN (uvicorn re-raises the captured signal); VRAM back to baseline in ~3.6s; semaphore census: `ls /dev/shm | grep -c "^sem\.mp-"`.

## SIG_IGN inheritance gotcha (hardware-debugged lesson)
- Spawn workers inherit SIGINT=SIG_IGN when the parent was launched as a background shell job (POSIX rule survives fork+exec); CPython does NOT install default_int_handler over an inherited SIG_IGN, so the kernel silently DROPS relayed SIGINTs -> no KeyboardInterrupt -> grace expiry -> SIGTERM escalation -> leaks.
- Stub-based unit tests CANNOT catch this (test stubs install their own signal handling) - only hardware e2e revealed it. Discriminators: strace the relay, `getsignal(SIGINT)` in the worker, /proc/PID/SigIgn.
- Fix pattern (in tree): `_arm_sigint_handler()` in launch.py re-arms `signal.default_int_handler` as the FIRST statement of every worker entry (`_run_scheduler`, `_run_tokenize_worker`). ANY new worker entry must call it.
- pyzmq 27.2.0 + CPython 3.14.7 deliver KeyboardInterrupt from blocking ZMQ recv fine once a Python handler exists; teardown of the full 28GB context measured 0.114s (empty_cache ~0 with expandable_segments).

## tqdm gotcha
- tqdm's TqdmDefaultWriteLock creates an mp RLock registered with the resource_tracker (call site models/qwen4_exp/weight.py:166 during expert weight load). Any worker KILLED BY SIGNAL leaks it -> "resource_tracker: There appear to be N leaked semaphore objects" warning. Clean exit runs SemLock Finalize and unlinks it. tqdm 4.70.0 has no lock_args/threading-only option.
