---
name: "ft-lever3-chain-cl1"
description: "mhc NS=64 db0f6b3 +3.4% paired; micro-fix bundle failed+reverted; lessons distilled+committed 61397a2/d76c3ce/a8fb231"
type: project
lastUpdated: 2026-09-27T22:09
lastRecall: 2026-09-27T22:09
---

# Lever 3 / C-L1: mhc stage1 NS-split ACCEPTED on hardware (2026-09-27) - uncommitted

Chain anatomy verdict (kb/cases/decode-t2-host-delivery/lever3-chain-anatomy.md): the GPU chain has NO single +5% lever - the big GEMVs are at the DRAM wall (in_proj 1531 GB/s = 85% of 1792 spec, lm_head 91%, aggregate 8.35 GB/step at 1320 GB/s = 74%; perfect-kernel floor 5.22 ms vs measured 6.32 -> max ~1.1 ms recoverable from the whole 6.67 ms class). Launch overhead immaterial (3556 kernels/step, 96% of inter-kernel gaps <0.5 us under graph replay). MLA-absorb bmm at 65% spec = no-lever.

## C-L1 accepted (the one real defect)
Triton _mhc_stage1_kernel ran grid (1,8)x128 = 8 CTAs on 170 SMs, 67 GB/s, 13.35 us x 90 launches = 1.20 ms/step.
- Knob: FREETOKEN_MHC_STAGE1_NS (read once at import, graph-capture-safe). Unset/"1" = bitwise-identical legacy; 2..64 = split count along the stage1 hidden dim (grid axis 1, 64-elem BLOCK_H floor, over-split clamps to legacy grid on tiny h); 0/negative/non-int/>64 = fail-fast ValueError. Recommended value: 64.
- Mechanism (nsys, NS=64): grid (1,64)x128; stage1 p50 13.35 -> 4.26 us (-68%); per step stage1 1.20 -> 0.391 ms, stage2 0.25 -> 0.275 (combine over 64 partials), mhc-class total 1.52 -> 0.736 ms (-0.78, inside predicted -0.75..-0.93).
- e2e (serial same-window A/B, run_arm): ctl_r2 17.26 tok/s / 56.4 ms; mhc64 18.28 (+5.9%) / 53.4; mhc32 17.83 (+3.3%); mhc64_r2 18.05 (+4.6%). Battery 24/24 with the knob exported. Prefill tails/fill wall within noise.
- HONEST CAVEAT: e2e p50 gain (-2.1..-3.0 ms) EXCEEDS the -0.78 ms nsys mechanism delta; direction consistent across both NS=64 boots, magnitude unexplained (secondary occupancy/chain effect or step-time variance).
- Files (uncommitted at save time): python/freetoken/kernel/triton/mhc.py (+44/-12), tests/layers/test_mhc.py (+211, 42 passed incl. bitwise legacy parity). kb verdict: lever3-chain-anatomy.md "Вердикт C-L1".

## Remaining chain bundle (each its own micro-commit if pursued)
C-L2 small bf16 GEMV fusion (+0.2-0.25%), C-L3 fused router topk (+0.5-0.7%, MEDIUM risk - tie semantics affect routing), C-L4 quantize_q8_1 into MMVQ (+0.35-0.5%), C-L5 f_b/g_b row-concat at load (+0.25-0.35%). Explicit no-ops: big MMVQs, MLA bmm, fast_index_copy, moe_vec, dsa/gdn mice.

**Why:** the "GPU math is not a lever" verdict from the pre-nsys era was half-wrong: no headroom in the GEMVs, but one occupancy defect worth ~+5% was hiding in the chain.
**How to apply:** hybrid-decode GPU work starts from the lever3 per-class table (bound analysis vs 1792 GB/s spec); never assume a Triton kernel's default grid fills the machine - check CTAs vs 170 SMs. Artifacts: .tasks/decode-research/arm_{ctl_r2,mhc64,mhc32,mhc64_r2}.*, instr_t2/nsys_gap/nsys_mhc64.sqlite.

## 2026-09-28 night: micro-fix bundle FAILED on hardware (all four knobs default-OFF, uncommitted)
- Paired mhc verdict (same window): NS=64 default 18.65 tok/s / 53.0 ms vs legacy(NS=1) 18.04 / 54.4 = +3.4% -> commit db0f6b3 earns its keep; the ambiguous 17.14 confirm was day noise.
- C-L4 FREETOKEN_FUSED_ACTQ: DEFECT - in-kernel actquant collapses MMVQ occupancy: mul_mat_vec_q 27.1 -> 236.9 us (5.5-8.7x), class 6.31 -> 45.37 ms/step, e2e -43% (10.56 tok/s). Launch counts worked (quantize 442->126), perf did not. Needs occupancy-aware redesign; original two-step costs only 0.32 ms/step.
- C-L3 FREETOKEN_TOPK_FUSED: DEFECT - tie-exact (ATen port verified) but slow: ft_topk_canonical p50 226.8 us (~20x ATen), topk class 0.64 -> 3.43 ms/step, e2e +3.8 ms slower. Single-block-per-slice design; needs multi-block redesign or drop.
- C-L5/C-L2: nsys mechanisms PASS (gemv trio 33->11, f/g site 68->0 + new 16384-row site 34/step) BUT (a) boot-crash on FTW checkpoints - fused keys emitted only by the raw-GGUF reader, FTW replays conversion-time names (needs FTW checkpoint rebuild to use); (b) raw-GGUF e2e: c5 +0.07 tok/s (noise), c2 -0.63 (unexplained, needs clean rerun). No measurable gain.
- Combo: -43%, entirely the C-L4 defect; synergy unmeasurable; cumulative-effect hypothesis NOT confirmed - the only clean pairing (mhc default) is additive +3.4%.
- Production safe: knobs default OFF; committed state (db0f6b3) unaffected. Experimental code sits uncommitted in the working tree (glm5_next kda/attention/weight/gguf, csrc mmvq/topk, layers/gguf.py, kernel/topk.py, tests).
- Artifacts: kb/cases/decode-t2-host-delivery/lever3-microfix-ab.md (linter OK); .tasks/decode-research/arm_mf_*.{json,log}, instr_t2/nsys_mf/; raw-path control showed 3.4-4.0 s single-step hitches (documented noise section).

## 2026-09-28: bundle surgically reverted; lessons distilled
- Revert: 14 tracked files checkout to HEAD + 3 untracked deleted (agent caught 2 more bundle members beyond the brief: moe.py, attention/dsv4_indexer.py - both imported freetoken.kernel.topk). After-inventory = exactly the preserve set (.veai/**, kb/** incl. lever2/lever3/microfix docs + README.md, .tasks/**). Stray root arm_ctl_r2.json was a byte-identical duplicate of the .tasks copy - deleted. Rebuild rc=0 (plain gcc), import ok, 139 pytest pass, FTW production-path confirm arm 17.89 tok/s / p50 54.6 ms (in band), census clean. HEAD stays db0f6b3.
- Distillation: kb/topics/decode-chain-microfusion.md (atomic article, 6 lessons: launch-count != speedup / occupancy collapse; semantics port != performance-model port; FTW reader-path key emission rule; sub-1% effects unmeasurable e2e; paired same-window A/B as house standard; occupancy audit pays both ways on 170 SMs) + .veai/skills/gguf-native-serving/TRAPS.md T64-T66 + kb/README.md + kb/topics/README.md + kb/cases/README.md status + glossary CTA/occupancy. Linter OK, links verified.
- Uncommitted in tree now: only kb/ docs (this distillation + lever2/lever3/microfix docs + README) and .veai/** churn - user decides the docs commit.

## 2026-09-28: kb docs COMMITTED (user order)
- 61397a2 docs(kb): t2 lever2/lever3 anatomy + microfix ab results (case folder 3 docs + kb/cases/README.md)
- d76c3ce docs(kb): decode chain microfusion lessons distill (topics article + kb/README.md + kb/topics/README.md + glossary CTA/occupancy)
- a8fb231 docs(skill): TRAPS T64-T66 (.veai/skills/gguf-native-serving/TRAPS.md only)
- .veai/memory churn deliberately left uncommitted (agent memory, not project kb). The "uncommitted kb docs" status in the sections above is superseded.
