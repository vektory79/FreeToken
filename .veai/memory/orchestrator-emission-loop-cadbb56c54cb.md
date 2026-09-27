---
name: "orchestrator-emission-loop"
description: "Emission loop: agent repeats read_* instead of planned run_command; stop, text-only turn or delegate to call_code_agent"
type: feedback
lastUpdated: 2026-09-26T23:51
---

Rule: the moment you notice yourself emitting the SAME read-only tool call again - or a tool different from the one the current plan requires - STOP; do not issue it a third time. Break the loop by (a) ending the turn with text only (resets the emission pattern), or (b) delegating the intended action to call_code_agent (its tools are independent of your emission state).

**Why:** 2026-09-26, two incidents in one session. (1) During kb-distill the orchestrator repeated an identical read_file call on kb/ many times even though the shell fallback had already been decided; the user had to intervene ("Тебя заклинило?"). (2) After the user's commit order the orchestrator planned a run_command preflight but instead emitted dozens of sequential/bulk read_memory calls (30+ parallel bursts included), burning many turns; recovery required a text-only turn. Contributing pattern: mass "recall-everything" read_memory bursts when one focused protocol memory was already read - wasteful and loop-adjacent.

**How to apply:** any multi-step task. Before EACH tool call verify it advances the current step; prefer one focused read_memory over bulk recall; if a call returns nothing new or errors identically, switch strategy (different tool, delegation, or text-only turn) instead of repeating.
