---
name: "ft-gguf-v2-grouped-mmq-measured"
description: "v2 grouped MMQ: battery PASS, +16.4% @8128; kill switch removed; committed 7f8c570; v3 hypothesis now verified"
type: project
lastUpdated: 2026-09-19T02:54
lastRecall: 2026-09-19T18:13
---

# v2 grouped MMQ prefill: measured outcome and the remaining gap

Campaign: `.tasks/mmq-prefill-kernel/TASK.md` (v2 section + verdict block), full report `v2-ab.md`, `v2-ab-results.json`, liveness `v2ab_probe_liveness.json`, trace `report3.nsys-rep` (2026-09-17/18, RTX 5090, winner flags, port 18801, HEAD 1a444e4, changeset uncommitted).

## What was implemented (11 files, +1300/-170 class)
- Vendored iq tile glue into kernel/csrc/gguf (vecdotq.cuh / mmq.cuh / moe.cuh) following **vLLM PR #36226** as the blueprint (NOT the llama.cpp-shim design from research-r2): RAW block bytes in tile_x_ql, decode inside vec_dot, VDR=4 with need_sum=true (self-consistency verified by review: need_sum=false would corrupt the half2 read; VDR=2 belongs to the abandoned post-#8495 layout and double-counts). Attribution: vLLM Apache-2.0 + llama.cpp 4696d5674 MIT lineage, headers correct.
- Cases 18/23 in ggml_mul_mat_a8 + ggml_moe_a8; ggml_moe_get_block_size(18/23)=4 so ONE moe_align trio serves gate/up (top_k=8) and down (top_k=1, tokens=M*top_k).
- moe_align trio from the EXISTING `freetoken.moe.fused.moe_align_block_size` (moe/fused.py:47, sgl backend on this rig) with per-size cache.
- Fixes made along the way: expert bound `exp_idx >= num_experts` replaced the `>255` guard (silent drop of experts 256-287 at E=288 + latent OOB); moe_q ds-load OOB `token_offs[threadIdx.y]` -> `[0]` (upstream vLLM latent bug); triton moe_align_small sorted/expert_ids tail sentinel-pinning + moe_q boundary `>` -> `>=` belt; kill switch env `FREETOKEN_GGUF_GROUPED_PREFILL` (default ON, per-call read, moe_vec fallback).

## MEASURED (same-session env-flip A/B, BEFORE reproduces campaign baseline -> valid)
- Prefill: 263.82 -> 301.88 tok/s (+14.4%) @4096; 281.38 -> 325.28 (+15.6%) @6144; **289.92 -> 337.61 (+16.4%) @8128**.
- Decode @65k radix-hit: 13.94 -> 14.60 (noise; code byte-identical on that path).
- Liveness kernel-confirmed: grouped moe_iq3_xxs/iq4_xs/q6_K = 1134 launches + moe_align 378 per 9 chunks, ZERO moe_vec_q in prefill chunks; decode 7938 moe_vec_q inside CUDA-graph replays; marker "gguf moe grouped mmq prefill active" fired once (sitecustomize probe).

## Why the ~758 tok/s ceiling was not met (hypothesis, UNVERIFIED)
Step-0 pinned MoE GEMM term = 18.13 s/28.77 s chunk; read-once roofline promised MoE 1.5-3 s -> ~758 tok/s. Measured grouped MoE = ~9.5-10.5 s/chunk (~100 ms per projection call) = only ~1.8x on the term. Working hypothesis: the vendored vLLM moe.cuh schedule re-reads expert weights per 4-token m-block (~57x per expert at 226 pairs/expert), cutting traffic ~4x instead of to-once. Reaching the full ceiling needs a weight-stationary grouped schedule (weights resident in registers/smem across the whole expert segment) - beyond the TASK.md v2 scope; would be a v3 investigation. After a true read-once kernel, the next bottlenecks per Step-0: dense q8_0 GEMM 6.04 s, fetch copies 3.15 s.

## Lessons for future waves
- env-flip kill switch (per-call read) beats pathspec-stash for same-session A/B; JIT compiles once and the disk cache serves both boots.
- nsys 2026.1.3 recipe for this rig: `launch` rejects -o; working = launch + start-after-ready + `--cuda-graph-trace=node`; mid-run `start` yields zero eager kernel activities.
- The tail-report artifact (last full 8128 chunk reports hardware-impossible tok/s) reproduces across campaigns - always exclude the last full chunk from medians.
- Parallel writer sessions touched vendored headers during the fix wave (duplicate license lines, condensed comment) - re-verify file state before git operations that assume a known tree.

## QUALITY BATTERY (2026-09-18, PASS - closes the validation protocol)
Historical tooling found (v0/v2 "no tooling" skips were a too-narrow grep): 24 prompts .tasks/gguf-glm5next-path-a/verification/phase6/battery-post-fix/p00-p23.json, runner /tmp/ft_phase6_battery.py (now copied to .tasks/mmq-prefill-kernel/quality/), methodology in compare.md / re-acceptance.md. Executed: two boots, env-flip FREETOKEN_GGUF_GROUPED_PREFILL=0/1, winner flags, identical prompt order. RESULT: 24/24 on-topic BOTH stages; 4 digit prompts correct both; 0 task-level flips (one div@4 suspect verified same-task); wording drift div min/median/max 4/193/572 chars, 18/24 in the historical 49-339 band, 5 over-band but read as same answers; 0/24 bitwise-identical (expected, greedy not bitwise). Liveness cross-check: marker absent in env0, fired once in env1; resolved boot plans byte-identical. Count correction: v2 changeset = 12 files (6 csrc/gguf, moe_align.py, gguf.py, moe.py, 3 test files), not 11. TASK.md v2 verdict: FINAL VALIDATION STATUS protocol COMPLETE (perf + liveness + quality). Commit still pending user.

## KILL SWITCH REMOVED (2026-09-18, post-validation, user request)
FREETOKEN_GGUF_GROUPED_PREFILL deleted: grouped MMQ prefill is now unconditional for supported types (dispatch predicate = all declared ggml block sizes > 0, kernel moe switch types 2,3,6,7,8,10,11,12,13,14,18,23); unsupported types (block size 0, e.g. IQ2_XS/IQ4_NL) still prefill via the moe_vec fallback with _assert_moe_vec_chunk. Verified: 126/126 tests (JIT cache hit, python-only change), zero env references left in python/tests/docs/benchmarks (one stale .pyc also purged), two boot smokes pass (marker "gguf moe grouped mmq prefill active" exactly once, 3 battery prompts on-topic with historical-class reasoning lengths, decode + radix cache re-send ok; 4333-token cold prefill 138 tok/s is single-request smoke noise, NOT a perf datum). Grid-cap test re-pinned via the IQ2_XS-fallback path (moe_vec cap reachable post-removal). Net diff: layers/moe.py -10/+6 + tests/models/test_gguf_expert_banks.py cap-test adaptation. Note: .tasks harnesses (quality/run_battery.py env1 stage) still set the env var - now simply unread, grouped stays on. A/B and battery conclusions remain valid (env0 == the fallback path that still exists).

## COMMITTED (2026-09-18)
7f8c570 "feat(kernels): grouped mmq prefill for iq3_xxs/iq4_xs gguf experts" on vektory79 - exactly the 12 files, +1467/-66, staged by explicit path (stale-index defense), 126 tests re-verified green pre-commit, .veai/memory churn excluded, NOT pushed. v3 (close the read-once gap: weight-stationary schedule) recorded as .tasks/mmq-v3-stationary-moe/TASK.md - Step 0 = verify the ~4x/57x per-expert m-block re-read hypothesis, v3a = larger m-tile sweep (4/8/16/32, SMEM/occupancy + tail-padding watch), v3b = expert-stationary persistent kernel only if v3a saturates; quality tooling reusable; after MoE term the top bottlenecks are dense q8_0 GEMM 6.04 s and fetch copies 3.15 s.

## SUPERSEDED (2026-09-18): the m-block re-read hypothesis above is now VERIFIED, and the reconciliation changed the picture
Step-0 source read (campaign .tasks/mmq-v3-stationary-moe/, memory ft-gguf-v3-mtile-campaign) verified ceil(n_e/MOE_X) ~57x and the 3.97x traffic cut, BUT measured grouped MoE ~10 s is 2.2x ABOVE its 4.5 s DRAM floor: the grouped kernel left the BW-bound regime (per-MAC ALU/LDS + k-step serialization dominate). The ~758 tok/s ceiling is therefore NOT reachable by traffic reduction alone. v3a (m-tile 8/16/32 via FREETOKEN_GGUF_MOE_MTILE) is implemented, Review SHIP, hardware sweep pending - state in ft-gguf-v3-mtile-campaign.
