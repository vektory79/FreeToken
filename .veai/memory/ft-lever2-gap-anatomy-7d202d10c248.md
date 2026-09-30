---
name: "ft-lever2-gap-anatomy"
description: "Lever2 anatomy: fetch already overlapped 28.7 ms; gaps = chain kernels 11.5 ms; C1 doze-fix +1.6-1.8% only lever"
type: project
lastUpdated: 2026-09-27T15:47
lastRecall: 2026-09-30T22:35
---

# Lever #2 (gap overlap) RESOLVED: fetch was already overlapped - gaps are the GPU kernel chain (2026-09-27)

Campaign: kb/cases/decode-t2-host-delivery/lever2-gap-anatomy.md (self-sufficient). Research-only wave, no code changes.

## What the nsys probe showed (verified collection; 77.5 MB report; 468 steps segmented via ensure_experts 42/step)
- The T2 brief's premise was WRONG: PCIe fetch is NOT sitting in the gaps. 28.66 ms/step of copies run HIDDEN under the pool windows at ~24 GB/s (= bench rate). Issuing fetch right after the doorbell is already production behavior; the ov0 arm's -23.5% was pricing exactly this existing overlap.
- Gap budget (t2d layout, envelope 57.5 ms/step): gaps 13.93 ms = serialized chain kernels 11.52 (dense_gemv 6.67, gdn/mhc 1.91, add/copy 1.36, topk 0.76, rest small) + minor-pool coordinator doze ~1.0 + launch/memop/inter-step ~1.5. GPU busy 77%. Another ~10-12 ms is WAIT(done) slack INSIDE pool windows - the CPU pool sits at its ~48.7 GB/s ceiling, irreducible without a faster CPU leg.
- Blocking dependency is a TRUE data chain, not graph-baked order: WAIT(done) cpu_executor.py:991 -> router/attention L+1 -> decode_submit moe.py:330 -> pool L+1. Cross-layer lookahead is impossible (router N+1 needs flag N); the speculative variant dies with the T3 oracle; fused step-start copy infeasible (all memcpy = 0.13 ms/step total).

## Only live lever: C1 doze-fix (+1.6-1.8%, BELOW the +5% adoption gate)
Minor-pool coordinators: their done-flags arrive every ~57 ms while the hot window is 50 ms (cpu_moe_ext.cpp:2934) -> the coordinator dozes every step; the p0->p1 edge costs 1.31 ms vs 0.27-0.43 for siblings. Fix shape: raise hot window / shared last-active across executors (knob FREETOKEN_CPU_MOE_COORD_HOT_MS); timing-only, no recapture. Discriminator: edge p0->p1 1.31 -> <=0.45 ms in the trace.

## Re-ranking for the user
The serialized GPU kernel chain (dense_gemv 6.7 ms/step) is now the biggest non-pool block. The old verdict "GPU math is not a lever" was made before nsys decomposition; reviving it is a USER decision.

**Why:** three waves of the campaign assumed unfetched overlap headroom; the probe killed that class of ideas and re-pointed the critical path.
**How to apply:** any further hybrid-decode optimization talk starts from this budget; do not propose fetch-side streaming, lookahead, or fused copies (refuted); nsys probe recipes and traps in memory nsys-silent-no-collection-trap.
