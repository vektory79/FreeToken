---
name: "process-hygiene-protocol"
description: "Test-wave hygiene: census, trap+watchdog runners, SIGTERM->SIGKILL, serialized runs; cross-wave GPU OOM = Environment"
type: feedback
lastUpdated: 2026-09-16T16:42
lastRecall: 2026-09-20T09:19
---

# Process-hygiene protocol for agent test waves

Rule (user directive 2026-09-15): every subagent test wave that spawns processes (serve, pytest, benchbw, harnesses) must run a preflight and postflight process census, use runner scripts with trap-on-EXIT cleanup + FAILPAT watchdog + hard timeouts, and escalate SIGTERM -> SIGKILL within 30 s. Between waves the orchestrator runs a REPORT-ONLY sweep; agents never kill an active wave's processes (active patterns are named in the prompt). A force-stopped agent is assumed dirty - the next wave starts with a sweep of its patterns.

**Why:** CPU-eating leftovers accumulated across waves (a prior-campaign `ft serve` hung 14+ hours on port 18085; graceful uvicorn shutdown after backend death never completes; tqdm/mp semaphores accumulate in /dev/shm).

**How to apply:** the concrete runner patterns live in ft-serve-test-and-e2e-gotchas - the FAILPAT list, trap-on-EXIT runner and backend-death watchdog in its "Benchmark harness must self-detect backend death" section, and the launch /ready / SIGTERM / semaphore-census commands in its "ft serve e2e recipe" section (census: `ls /dev/shm | grep -c "^sem\.mp-"`). Follow them for every wave; do not improvise cleanup ad hoc.

**Serialization within a wave:** measurement invocations must be SERIALIZED - one run at a time, reaped (wait + verify the PID is gone) before the next starts, each under a hard timeout. Concurrent measurement runs are doubly harmful: leaked processes eat CPU AND simultaneous runs corrupt each other's timing (core contention invalidates the numbers). After each run, re-census the binary's name before launching the next. Real incident (2026-09-15, ggml microbench): the agent's own loop left `./mb 16 --pin` (1332% CPU) and `./mb_dbg 8` (400%) running concurrently - agent stopped, results voided, harness relaunched with serialized+reaped runs.

## Cross-wave GPU contention (2026-09-15 incident)
A CPU-path pytest wave ran while a concurrent hardware wave (ft serve deep-fill verification, same 786k config, port 18081) held 29.8 GiB VRAM on the single RTX 5090: 2 pre-existing CUDA tests in tests/models/test_gguf_expert_banks.py failed with torch.OutOfMemoryError - NOT regressions (they were green before that wave started). The wave agent correctly did NOT touch the active wave (kill rule above) and documented it instead.
- Why: one GPU per box; any CUDA-touching test inherits whatever VRAM the concurrent wave leaves free, so overlapping waves produce false OOM failures.
- How to apply: before flagging CUDA test failures, check for a live hardware wave (pgrep 'ft serve' + nvidia-smi) and classify OOMs under concurrent VRAM pressure as Environment - re-run just those tests after the wave ends. Conversely, test-only or CPU-path fixes do not invalidate a concurrent hardware verification run (no restart needed); changes touching the hardware-executed path do.
