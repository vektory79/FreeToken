---
name: "ft-ftw-gguf-fastpath-outcome"
description: "Task S FTW gguf fastpath: Gaps 1-3 + capability fallback done and committed (8568807..2e9fcf1); boot 52.1s vs 94-132s"
type: project
lastUpdated: 2026-09-20T18:24
lastRecall: 2026-09-22T12:00
---

# FTW GGUF fast path (task S): implemented + hardware-validated, COMMITTED

Task brief: .tasks/ftw-gguf-fastpath/TASK.md (status + Outcome section updated in place).
Summary memory ft-serve-gguf-tuning-campaign-2026-09 holds the bare-GGUF tuning context
(winner flags, radix root cause); this memory holds the task S outcome only.

## What landed (commits 8568807/e9ad38d/ef14efb/2e9fcf1 on vektory79, +477/-16, 8 files)
- Gap 1: kind_kernel_for("gguf") -> (None, None) via tag-keyed _KIND_KERNEL in
  moe/legacy_format.py. TRAP: LEGACY_FORMAT is the INVERSE (kind,kernel)->tag map; the
  entry must NOT go there (first test run proved the obvious placement wrong).
- Gap 2: convert.py persists gguf_types (nested [[gate,up,down]] per MoE layer, explicit
  role order) in the finalize meta; load_ftw_banks (checkpoint/ftw.py) reads it, validates
  structurally (list, len==num_layers, 3-int rows, reader closed before raise), passes into
  ExpertBanks(gguf_types=..., default None - nvfp4 untouched); engine wiring needed no
  change (cache.gguf_types assignment is source-agnostic).
- Gap 3 + capability: gguf_ftw_signature_groups (floor gate from FTW index shapes) and
  gguf_expert_bank_types FTW fallback (_ftw_gguf_index_meta/_ftw_gguf_types_by_layer) fix
  hybrid rejection + cpu-executor viability on FTW dirs - the Major both reviewers caught:
  without it the planned hybrid boot failed at config time.
- offload_cache._build_fused_copy_plan: loop zips bank_schema names, all 3 rejections now
  carry the bank name (was NameError masking the not-pinned RuntimeError).

## Quality loop
Review x2 (PASS-with-nits / FAIL on the Major) -> triage: 9 findings, all fixed, none
rejected. Tests: NEW tests/checkpoint/test_ftw_gguf_banks.py (roundtrip via real
convert_checkpoint with CUDA skipif, 3-signature non-square fixture, down-first variant,
5 malformed-meta rejections, capability sentinel tests) + tests/engine/test_cache_budget.py
(+59) + tests/moe/test_legacy_format.py (extracted regression, fails-before proven).
Battery: 175 passed / 7 skipped.

## Hardware (RTX 5090, winner flags --moe-strategy hybrid etc., port 18801)
- Conversion: 147.5 GB bare gguf -> 134.24 GiB FTW in 268 s
  (`uv run ft checkpoint --model <gguf> --out <dir> --moe-backend offload`; NO
  --quant-backend for gguf - repack would defeat raw streaming). Output dir:
  /media/ai/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL-FTW
- Boot 52.1 s start-to-ready vs 94-132 s bare-GGUF; banks 125G @ 4.12 GB/s (30 s in-boot).
- Slots 388/288/288; free-after-init 4.06 GiB; fetch fraction 30.7% (not cap-1); pools
  [15,1,1] @16T avx2; CPU executor pool lines carry the gguf signatures.
- Prefill median 790.7 tok/s (8128 chunks) vs current bare anchor 792-808 - the ~293 tok/s
  figure in the tuning memory/brief was the PRE-dense-q80/m-tile anchor (stale). Decode @65k
  14.78/15.52/15.02 tok/s, radix HIT 65536 on all repeats (fix-1 effective). Battery 24/24
  wording-level parity vs 4 bare-GGUF reference boots.
- Gotchas: serve boot flag is --model-path (positional rejected, T25); FTW path omits the
  bare-path ggml-types histogram line (signatures live in executor pool lines); FTW gguf
  dirs from older builds now fail fast in load_ftw_banks with a re-convert message.

## Status
Committed (bisect-green series; no push): 8568807 fix(checkpoint) gap1; e9ad38d
feat(checkpoint) gap2; ef14efb feat(moe) gap3+capability; 2e9fcf1 fix(moe) fused-copy-plan
bank names (no test - paths need fused-capable CUDA construction). Skill self-update
T57/T58 + T44 extension landed in .veai/skills (uncommitted, user decides).
Artifacts: .tasks/ftw-gguf-fastpath/ (run-convert-wave1.sh, verify-convert-wave1.py,
run-wave2-measure.py, wave2-parity.py, run-wave2/, /tmp/ftw-convert-glm53-q3k-wave1.log).
