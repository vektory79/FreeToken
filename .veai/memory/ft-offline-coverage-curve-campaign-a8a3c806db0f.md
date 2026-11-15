---
name: "ft-offline-coverage-curve-campaign"
description: "offline-coverage-curve CLOSED NO-GO: S0=820, validation 0.13pp, +300 slots=+3.85pp; ceiling 82%; route affinity next"
type: project
lastUpdated: 2026-11-15T14:26
lastRecall: 2026-11-15T14:22
---

# offline-coverage-curve: CLOSED NO-GO (2026-11-15)

Case kb/cases/offline-coverage-curve/ (gate for KV-streaming / capacity levers:
offline C(S) "slots -> coverage"; gate: +300 slots < +5 p.p. = NO-GO, >= +8 p.p.
= GO-prototype, between = owner).

VERDICT: **NO-GO** - +300 slots (820->1120) = **+3.85 p.p.** C(S) (< +5).
Curve: S0+600 = +7.0; 2*S0 (1640) = +9.2; ceiling (all 12096 pairs in VRAM) =
82.2% agg / 85.4% med-per-step - ~15-18% of pairs are one-shot, uncachable by
ANY slot fund (the hard limit of ALL capacity levers). cpu_gib/step: 1.768 @S0
-> 1.634 @+300 (-7.6% CPU-leg bytes) -> 0.623 @ceiling. Marginal pp per 100
slots decay (1.41 -> 1.24 -> 1.20 -> 1.16 -> 1.03 -> 0.95...).

Validation at S0: C tel 0.4913 vs C sim 0.4927 = **0.13 p.p.** (threshold 2.0)
PASS; max per-layer miss_rate delta 0.31 p.p. - the numpy replica of
_ensure_experts_hybrid_kernel is near-exact (timestamp-LRU, BY_RECENCY score
rec*E+(E-1-e), min-wait rounding, victims argmin(usage) ties lowest slot, cap =
min(missing, frac*misses, E, pool-hits), overflow->CPU, recency bumped after
selection). The old "coverage ~0.50" anchor reproduced on this config too
(0.49 at 820 slots) - coincidence of values, not an anchor transfer.

Boot anchors, config 0.81/800k fp8 tier 10/50 chunk 4095 eager
(--cuda-graph-max-bs 0), GGUF GLM-5.3-Flash-UD-Q3_K_XL, checkpoint
GLM-5.3-Flash-UD-Q3_K_XL-FTW-vision-new (the old -FTW/ path is dead):
- S0 = 820 slots (stable across 5 boots), 3 pools [20, 39, 761] = GGUF format
  groups (2 layers IQ3_XXS/Q6_K, 1 IQ4_XS/Q6_K, 39 IQ3_XXS/IQ4_XS), unified
  scope, equal split.
- benchbw v6 fractions 28.4%/24.7%/29.7%; slot bytes 10.375/12.688/15.062 MiB
  (engine._moe_layer_slot_bytes); L=42 E=288 top_k 8; locked/unpinned empty.
- eager decode 13.5-15.3 tok/s (prod graphs 21-23); greedy EOS cut decode at
  545 without ignore_eos - always pin decode length.

Gotchas (each cost a boot on 2026-11-15): greedy EOS; driver INERT via
atexit NameError swallowed by module armor (lambda for forward refs); Engine
keeps moe_offload_cache SINGULAR (plural lives on GraphRunner) - naive plural
getattr silently yields no caches AND silently skips collect_stats flip;
helper processes' exit _dump clobbers stats (guard: skip when _ENGINES empty);
_copy_feat_bytes may be None - read eng._moe_layer_slot_bytes. T1 driver fixed
and mirrored (kb/harness/decode-research/driver/, FREETOKEN_DR_ROUTES mode).

Deliverables: kb/cases/offline-coverage-curve/{VERDICT.md,PHASE0.md,
results_trace5.md,.csv}, TASK.md checklist all [x]; harness mirrored to
kb/harness/coverage-curve/ (run_trace.py, sim_coverage.py, test_sim_synth.py,
README). Artifacts in .tasks/offline-coverage-curve/ (trace5 final).

Why: the gate killed KV-streaming for this config without any kernel work -
the lesson is the offline gate protocol itself (eager trace -> exact policy
replica -> S0 validation -> sweep). Route affinity (Maya) is now the main live
candidate: separate HW arm + quality battery; NOT covered by an offline curve
(it shifts the routing the trace was taken from).
How to apply: any future capacity lever for this config starts from the
ceiling fact (82%) and this curve; do not re-derive S0 per boot (stable 820).
