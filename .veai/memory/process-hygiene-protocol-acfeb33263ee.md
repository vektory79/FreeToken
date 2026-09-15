---
name: "process-hygiene-protocol"
description: "User rule for test waves: process census pre/postflight, trap+watchdog runners, SIGTERM->SIGKILL, serialized runs"
type: project
lastUpdated: 2026-09-15T14:55
lastRecall: 2026-09-15T17:10
---

# Process-hygiene protocol for agent test waves

Rule (user directive 2026-09-15): every subagent test wave that spawns processes (serve, pytest, benchbw, harnesses) must run a preflight and postflight process census, use runner scripts with trap-on-EXIT cleanup + FAILPAT watchdog + hard timeouts, and escalate SIGTERM -> SIGKILL within 30 s. Between waves the orchestrator runs a REPORT-ONLY sweep; agents never kill an active wave's processes (active patterns are named in the prompt). A force-stopped agent is assumed dirty - the next wave starts with a sweep of its patterns.

**Why:** CPU-eating leftovers accumulated across waves (a prior-campaign `ft serve` hung 14+ hours on port 18085; graceful uvicorn shutdown after backend death never completes; tqdm/mp semaphores accumulate in /dev/shm).

**How to apply:** the concrete runner patterns live in ft-serve-test-and-e2e-gotchas - the FAILPAT list, trap-on-EXIT runner and backend-death watchdog in its "Benchmark harness must self-detect backend death" section, and the launch /ready / SIGTERM / semaphore-census commands in its "ft serve e2e recipe" section (census: `ls /dev/shm | grep -c "^sem\.mp-"`). Follow them for every wave; do not improvise cleanup ad hoc.

**Serialization within a wave:** measurement invocations must be SERIALIZED - one run at a time, reaped (wait + verify the PID is gone) before the next starts, each under a hard timeout. Concurrent measurement runs are doubly harmful: leaked processes eat CPU AND simultaneous runs corrupt each other's timing (core contention invalidates the numbers). After each run, re-census the binary's name before launching the next. Real incident (2026-09-15, ggml microbench): the agent's own loop left `./mb 16 --pin` (1332% CPU) and `./mb_dbg 8` (400%) running concurrently - agent stopped, results voided, harness relaunched with serialized+reaped runs.
