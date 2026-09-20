---
name: "measured-advice-over-paper"
description: "User rejects paper extrapolations as config advice; demands measured numbers (2026-09-20 VRAM-headroom case)"
type: feedback
lastUpdated: 2026-09-20T14:07
lastRecall: 2026-09-20T14:24
---

# Measured advice over paper extrapolations (user expectation)

Rule: any user-facing configuration or performance recommendation (serving flags, VRAM budgets, throughput expectations) on this rig must be backed by measurements - or explicitly labeled UNMEASURED with a verification wave launched in the same turn.

**Why:** 2026-09-20, GGUF serving VRAM question: the assistant answered with paper extrapolations ("~1.3-1.4 GB free per -0.05 memory-ratio", linear reserve scaling) without measurement; the user immediately demanded verification ("Нужны проверенные решения и информация как эти решения сказываются на скорость"). Same failure class as the skill's T35 (projection without a measured anchor) but in user-advice form - and the same campaign had already refuted two paper models (dense floor-model predicted +1.9..+3.6% e2e, measured +40.46% at tile 64).

**How to apply:** when answering "how do I run/configure X" questions: (1) state only measured anchors and cite the campaign/artifact they come from; (2) label any extrapolation UNMEASURED; (3) if the user needs a decision, launch the verification wave immediately (precedent: the VRAM-headroom matrix -> .tasks/dense-q80-gemm/vram-headroom.md). Paper numbers may shape the experiment design, never the recommendation itself.
