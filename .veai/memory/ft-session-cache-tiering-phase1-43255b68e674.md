---
name: "ft-session-cache-tiering-phase1"
description: "Tiering: L1 fixed at 10 GiB (no ulimit raise); Qwen3.8 second model, tier must stay OFF until QSA guard lands"
type: project
lastUpdated: 2026-09-23T20:15
lastRecall: 2026-09-25T01:35
---

# Session-cache-tiering phase 1 (SessionTierStore) - CPU phase done, uncommitted

2026-09-22, branch vektory79, brief kb/cases/session-cache-tiering/TASK.md (PLANNED).
Orchestrated via .tasks/session-cache-tiering/PLAN.md; process model = gguf-native-serving ORCHESTRATION.md wave loop.

## Status
- Working tree UNCOMMITTED (user decides): python/freetoken/scheduler/session_tier.py (~600 lines), python/freetoken/kvcache/utils.py (chain_page_key, layering fix), additive EvictResult fields victim_path_keys/victim_paths (+VictimPath NamedTuple) in hybrid_radix_cache.py with opt-in derive_paths kwarg, cache.py tier hooks (SessionTierCfg, _tier_offer before free, adaptive keep set, try_restore/abandon_restore, _KVPageBytes codec, shutdown_tier = flush_live+compact), scheduler.py cfg build + shutdown flush, engine/config.py + server/args.py flags --session-tier-ram-gib/--session-tier-dir/--session-tier-ssd-gib.
- Gate: tests/kvcache/radix+tests/scheduler -m "not slow" = 291 passed/1 skipped; wide CPU suite only environmental failures (2 CUDA-required, 1 known flaky test_ple.py:421). Wash pins d82274c green unchanged.

## Key design facts (non-obvious)
- Page addressing = blake2b chain keys over token-id tuples (caller-computed via chain_page_key), NOT KV-byte hashing (no D2H on hot path). path_key opaque; page keys at offer/probe seams.
- Units: offer/boundary_len/probe depth in PAGES; _tier_snap_bound PAGES; admission bookkeeping (_tier_admission st=[tip,div,spare,under]) in TOKENS - ps>1 mismatch was a real blocker.
- L2: single blob.bin + journal.log, O_DIRECT; _blob_eof (append pos, never decreased) separate from _ssd_used (capacity); failed append resyncs via lseek(SEEK_END); crc32 per journal record lets replay converge after partial compaction (SIGKILL between blob and journal renames drops dead records automatically).
- Adaptive set: tip + ONE deepest under-divergence boundary + newest non-tip; session grouping by FIRST-page chain key (documented approximation).
- restore consumes pages+slot at match_req; abandon_restore wired into ALL 4 PrefillAdder._try_allocate_one refusal paths; lock(handle) clears _tier_pending_restore on success.
- --session-tier-dir alone enables L2-only (unlimited cap); ram_gib=0 + no dir = off (bit-identical).

## Hardware arms pending (RTX 5090, .tasks/interleave-cache-repro/, serial, needs VRAM free)
Arm 3 persistence (shutdown->boot first turn = HIT), Arm 4 two-session 150-200k scale; byte-identity greedy check. Checklist in PLAN.md: tier restore slot freed after async pool.copy_from (stream ordering); _KVPageBytes MHAKV layout via Arm-3 byte check.

## Process lesson
Review-1 caught 4 blockers the implementer missed (dup init block w/ NameError landmine, tokens-vs-pages, L2 EOF desync, refusal-path leak) - fresh-eyes review after each wave earned its cost; 2 rounds of review->fix converged the diff.

## UPDATE 2026-09-22 (same session, later)
- COMMITTED: d77d15e "feat(scheduler): tier session cache demotion to ram/ssd buffers" on vektory79 (11 files, 1892 insertions; excludes .veai/** churn, re-staged fresh per stale-index protocol). No push, no PR.
- Full gate WITH VRAM free (user switched local LLM to cloud): not-slow suite 5 failed/2254 passed/206 skipped - all 5 are the documented baseline classes {pinned_tensor UVA, qsa_fp8 :48, test_ple :421, 2x glm_dsa smem OOR}, none attributable; the 2 formerly CUDA-required test_gguf_expert_banks tests now PASS. Slow subset 11 passed/4 skipped (4 skipped = env-gated model checkpoints).
- Arms 3/4 (persistence + 150-200k scale) remain the open phase-1 acceptance items; arms package prep was cancelled by user - if ordered later, start with .tasks/interleave-cache-repro/ as the harness pattern.

## HARDWARE ARMS DONE (2026-09-22, resume 1b6dc0bb) - HW fixes UNCOMMITTED
- 5 hardware bugs found+fixed (CPU gate 296 passed/1 skipped, all fails-before tested): (1) cold-boot _tier_snap_bound rehydration (journal segments never rehydrated -> restore always re-prefilled); (2) _KVPageBytes scale layout is DSA/MLA 2-D [L,pages*ps] on GLM-5.3-Flash, not MHAKV 4-D - locate token dim by size; (3) restored handle finishing nonaligned skipped insert -> page leak (KV-only node + adopt aligned span); (4) partial-depth restore dup-span leak (253 pages): try_restore now refuses when tree match >= store depth via new owned_prefix(); (5) shutdown_tier offers LIVE tree snapshot boundaries before flush + probe prefers boundary-EXACT segment (deeper snapshot can't serve shallower resume).
- Arm 3 acceptance 5 PASS: reboot-A-turn1 cached=57344 wall=4.19s; reboot-B-turn1 44096/4.32s; 14 journal records replayed; phase-A wash ELIMINATED (a_full_miss_count=0 vs 11/11 arm-1 baseline). Acceptance 7 FAIL as written: greedy decode non-deterministic across prefill chunkings AND boots (intra-boot char 36; boot A vs C char 94) - NOT corruption (would flip token 1); CPU KV+scale roundtrip byte-exact. Acceptance-7 criterion needs amendment.
- Arm 4 PASS: 2x150k (2x200k rejected = hard KV edge), 8/8 non-cold HIT, median 37.5s vs 186-248s; tier dir 5.68 GB/15 records, no torn tail. L1 22 GiB never overflowed - L2 populates only at graceful shutdown.
- MEMLOCK hard limit 23.56 GiB: --session-tier-ram-gib 30 CANNOT mlock (RuntimeError by design); production 30 GiB needs root-raised ulimit -l. Arms run 22 GiB.
- Package .tasks/session-cache-tiering/: run_arm3.sh, run_arm4.sh, replay_tier4.py, tier_client.py, tier_state.py; logs arm3_console.log/arm4_console.log etc.

## CAMPAIGN COMPLETE (2026-09-22/23)
- HW fixes COMMITTED as 4b6aacb "fix(scheduler): harden session tier restore paths found on hardware" (4 files: hybrid_radix_cache.py, cache.py, session_tier.py, test_hybrid_cache_manager.py; +314/-30; gate 296 passed/1 skipped). Docs COMMITTED as f90b081 "docs(kb): amend tiering acceptance 7 and distill hardware results" (TASK.md amendment + topics article results section; kb status flipped proposal -> implemented-verified).
- Acceptance 7 AMENDED (user decision 2026-09-22) to behavioral fidelity: (a) HIT at restore latency, (b) CPU byte-exact KV+snapshot roundtrip incl. scale buffers, (c) no integrity violations - because greedy decode is non-deterministic across prefill chunkings AND boots (arm-3: intra-boot char 36, boot A vs C char 94).
- Arm scripts retuned to --session-tier-ram-gib 10 per user (RAM headroom; .tasks gitignored so no commit); at 10 GiB arm-4 worst case ~13 GiB exercises L1->L2 demotion by design.
- Phase 1 CLOSED. Phase 2 (out of scope): async prefetch, mr>1 concurrency hardening, scheduled compaction, metrics, non-hybrid models. Production 30 GiB L1 still needs root-raised ulimit -l (hard 23.56 GiB).

## OPERATOR DECISIONS (2026-09-23, user)
- ulimit -l hard limit stays 23.56 GiB (NO root raise). Production L1 = --session-tier-ram-gib 10; worst case ~13 GiB -> mid-run L1->L2 demotion accepted by design.
- Second model in rotation: Qwen3.8-Flash-Next-NVFP4-FTW (serve: moe-cpu-threads 16, moe-backend hybrid, kv-reserve 262144, mr=1, moe.nvfp4=triton; NO tier flags today -> tier OFF for it). Boot sanity check pending (needs VRAM window). Tier fit findings + phase-2 scoping: see memory ft-qwen38-flashnext-tier-fit (HAZARD: tier would activate for Qwen3.8 with incomplete QSA codec -> do NOT pass tier flags until guard/codec lands).
