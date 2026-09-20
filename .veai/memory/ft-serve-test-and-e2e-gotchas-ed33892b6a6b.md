---
name: "ft-serve-test-and-e2e-gotchas"
description: "FreeToken e2e gotchas: pytest --extra dev, chat 422 no model, backend-death hang, log-tail watchdog harness"
type: project
lastUpdated: 2026-09-19T18:18
lastRecall: 2026-09-20T17:40
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

## Full-gate baseline update (2026-09-12, agent-run, uv run --extra dev pytest tests/ -m "not slow" --ignore=tests/models/test_glm5_next_kda_snapshot.py -> 19 failed / 1801 passed / 170 skipped)
- Still baseline-classified: pinned_tensor UVA (cudaHostGetDevicePointer, :128), qwen4_exp/test_ple flaky bitwise (:421), muse_glimmer tests-import (:872).
- NEW known-failing class since commit 04d4621 (2026-09-08, nvfp4 latent dsa kv): 10x "ModuleNotFoundError: No module named 'tests'" (7x tests/attention/test_dsa_kpool.py[nvfp4]:175, 2x tests/kvcache/test_dsa_pool.py:220, 1x tests/models/test_glm_dsa.py:316) - same no-tests/__init__.py root cause.
- 2x tests/models/test_glm_dsa.py: triton OutOfResources shared memory 102400 > 101376 (glm_dsa_sparse.py:487/:517).
- 3x tests/moe/test_prefill_hit_d2d.py: "cudaMemcpyBatchAsync probe copied wrong bytes" (kernel/batch_memcpy.py:39).
- 1x tests/kernels/test_qsa_fp8.py::test_fp8_codes_match_the_bf16_cache_bit_for_bit[1-1-64]: CUDA invalid argument (:48).
- None of these touch layers/quantization/method.py or server/args.py.
- 2026-09-12 kernel-UX change context: hybrid decode requires kernels with a CPU executor format; nvfp4 triton is the only cpu_ok one (marlin/b12x are gpu-only); deprecated shim maps --nvfp4-backend flashinfer -> b12x (args.py _nvfp4_entry), so hybrid+flashinfer fails by design; select_kernel now appends "(usable here: <kernels>)" and args.py warns at parse time. GLM FTW auto path fills triton from checkpoint metadata (engine.py _adjust_ftw_quant_backend) - never pass a conflicting explicit --quant-backend.

## A/B serving benchmark gotchas (2026-09-12, agent harness on :18081)
- /v1/chat/completions rejects requests without a "model" field: HTTP 422 Unprocessable Content. Always send "model": <served_model_name> (e.g. "GLM-5.3-Flash-NVFP4-FTW").
- Prefill pace: client-side wall clock UNDERSTATES server throughput (459 vs ~514 tok/s on a 64k fill; tokenize/HTTP overhead). Authoritative metric = server log "Prefill batch ... input throughput (token/s)" lines, one per 4096-token chunk; the first chunk is low (JIT/autotune warmup) - compare steady-state chunks only.
- Decode: re-send the SAME prompt to reuse the HybridRadixCache prefix (log shows "#new-token: 12, #cached-token: 64512") and stream for a clean decode rate (exclude TTFT). TTFT on a fully cached prefix is ~6 s - that is first-decode-step overhead (CUDA graph/sync), not re-prefill.
- glm reasoning parser streams reasoning_content: count any delta carrying content OR reasoning_content as a generated token for tok/s.
- max_tokens=1 non-stream works as a prefill-only probe (usage.prompt_tokens present; content empty because reasoning parser burns the budget).

## Backend-death hang gotcha (2026-09-12, OOM during B2 8192-chunk benchmark)
- A CUDA OOM in the first prefill kills the backend worker ("Backend worker freetoken-TP0-scheduler exited"; API logs "Backend worker is gone and cannot be restarted"), then the API server hangs indefinitely in uvicorn "INFO: Waiting for connections to close." - /ready never returns 200 or dies; a /ready-polling benchmark runner blocks until its timeout while the ft process stays alive in shutdown-wait.
- Recovery: kill -9 the ft serve process tree AND the runner script (graceful SIGTERM never completes once the worker is gone; the runner's kill -0 liveness check passes because the dying process is still alive). Then verify GPU is empty via nvidia-smi before the next config.
- Design fix for future harnesses: poll /ready AND the process's backend-log for "exited"/"OutOfMemoryError", with a hard timeout that force-kills both PIDs.

## Benchmark harness must self-detect backend death (2026-09-12, user feedback)
- User rule: the runner must detect a backend crash itself and print the reason - the user should never have to relay "it crashed" or "it is just waiting" (said after B2/A4c OOM hangs).
- Implemented pattern (scripts are ephemeral /tmp/ft_run_config.sh + /tmp/ft_ab_bench.py; recreate from this if lost):
  * FAILPAT='OutOfMemoryError|AssertionError|Backend worker is gone|backend worker .* exited'; grep it in the server log every 5 s during the boot-wait AND in a background watchdog loop (2 s) while the benchmark runs; on hit: print matched lines + the "Tried to allocate" line, pkill the client, kill -9 the ft process, exit non-zero.
  * trap on runner EXIT kills watchdog + client + server, so a failed run never leaks processes.
  * client: per-phase try/except that captures the HTTPError BODY (first ~400 chars, e.g. 422 validation detail), prints phase names as progress, exits non-zero if any phase failed - the runner then reports rc and the failure lines.
- Why: a dead backend leaves uvicorn alive in "Waiting for connections to close"; /ready stays non-200 while the process is still alive, so a /ready-only poller hangs until its timeout.

## Baseline pointer
The 2026-09-12 full-gate baseline above is historical; current authority (09-15/19 classes + scale) lives in ft-pytest-worktree-baseline-gotchas (drift cause: ft-gate-nodeid-collection-drift). A quiet baseline class (test_prefill_hit_d2d) = drift, not regression.
