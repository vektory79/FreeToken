---
name: "ft-bare-logger-liveness-trap"
description: "Bare stdlib getLogger modules are boot-log-invisible; sitecustomize PYTHONPATH probe for liveness (layers/moe.py case)"
type: project
lastUpdated: 2026-09-17T19:24
lastRecall: 2026-09-18T19:47
---

# Bare-logger liveness trap in freetoken boot logs

Production modules that log via a bare stdlib `logging.getLogger(__name__)` (verified: layers/moe.py, 2026-09-17 v0 A/B campaign) NEVER emit into `ft serve` boot logs: freetoken's `init_logger` does not wire a handler for them, the root logger stays unconfigured, and `logging.lastResort` drops INFO-level records.

Consequence: a boot-log liveness check ("if the marker line is absent, the code path never ran") is unpassable BY DESIGN for such modules - absence of the line does NOT mean the path is dead.

Probe pattern that works (harness-side, zero production code changes): PYTHONPATH-injected `sitecustomize.py` that adds a root logging handler before freetoken imports; the module's INFO marker then fires exactly once and liveness is confirmed. Used in .tasks/mmq-prefill-kernel/v0ab_probe.py (probe boot) to confirm the expert-sorted prefill path was active despite the silent boot log.

Why it cost a boot cycle: the A/B protocol mandated a liveness grep that could never succeed; a naive reading would have (wrongly) concluded the sorted path was inactive and the A/B measured nothing.

How to apply: before concluding "the instrumented path never ran" from an absent boot-log line, check whether the module's logger is init_logger-wired; if not, use the sitecustomize probe (or rewire the module logger through init_logger if a permanent fix is wanted). Any new "one-time INFO" liveness markers added to production code should go through init_logger-wired loggers or they are boot-log-invisible.
