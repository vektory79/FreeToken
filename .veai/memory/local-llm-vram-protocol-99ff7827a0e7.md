---
name: "local-llm-vram-protocol"
description: "Local LLM occupies rig VRAM: work serial/sync; for VRAM-heavy work ask user to switch to cloud model"
type: feedback
lastUpdated: 2026-09-21T20:49
lastRecall: 2026-09-25T16:52
---

# Local-LLM VRAM constraint on the FreeToken rig (feedback, set 2026-09-20)

Rule given by the user: when working on this rig, assume a local neural
network is running and occupies nearly all VRAM.

- Process only one request at a time: no async or parallel processes.
- Subagents allowed, but strictly one at a time, synchronously.
- Do NOT run anything VRAM-heavy. If VRAM-heavy work is unavoidable, collect it
  into a single group, ask the user to switch to the cloud model for that
  batch, run it, then ask to switch back.

Why: GPU benchmarks and profiling on the RTX 5090 collide with the local LLM's
VRAM residency; parallel execution risks OOM and noise (consistent with the
desktop-VRAM-swing effects recorded in ft-gguf-serving-vram-headroom).
How to apply: any GPU-touching or massively parallel work on this box; CPU-only
work (doc analysis, sqlite distillation, git) is fine. The instruction was
framed as session-scoped, but the rig condition recurs whenever the local LLM
is resident - confirm with the user per session.
