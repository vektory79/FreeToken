---
name: "ft-tier-phase2-swave"
description: "Tier campaign committed d77d15e..9503cb4; HW verdicts 3 models; next brief offer-supersede ready"
type: project
lastUpdated: 2026-09-25T10:37
lastRecall: 2026-09-25T01:35
---

# Session-tier phase-2 S-wave (metrics, scheduled compaction, safety guards)

2026-09-23, vektory79. Full commit chain: d77d15e (phase-1 feat) -> 4b6aacb (HW hardening) -> f90b081 (kb docs) -> 2348da5 (S-wave) -> 88102dd (heavy briefs) -> f9ee745 (AGENTS.md cloud-resume rule) -> 9503cb4 (kb distill). All committed, no push/PR.

## What landed (summary)
- Phase 1 + heavy briefs: SessionTierStore (L1 mlock 10 GiB operator cap - MEMLOCK hard 23.56, no ulimit raise; L2 blob+journal O_DIRECT, crc replay), additive EvictResult (victim_path_keys/victim_paths), QSA codec (slab rows [page*ps/ratio) + mrope rope rows; isinstance branch - size locator can't find slab), KpoolDSA codec (page-owned latent+2-D scales+1/ratio SHADOW index; scratch/rings per-forward) - restored GLM coverage, generic KV-only tier (plain radix, tip-only set, RestoredRadixHandle), async restore prefetch (RestoreTicket cap _MAX_RESTORE_TICKETS=2, 0=off; reservations guard >=half pages + >=1 slot; adopt == sync gate order), metrics (Counter -> batch-log fragment + shutdown line; /v1/stats deferred), scheduled compaction (dead>25% AND >256 MiB; offer/evict_store/run_when_idle), per-handle pending restore, identity-keyed compact journal survival, _blob_eof resync in compact, BUG-1 restored-finish free_upto advance, rebuild codec refresh, FREETOKEN_WORKER_SIGINT_GRACE_S env (default 10; big flushes 120).

## HW-verified verdicts
- Qwen3.8-Flash-Next-NVFP4-FTW (kv262144, moe.nvfp4=triton, mr=1): reboot HIT 48256/38016 @ 3.3-4.0s; flush 16 seg/11.94 GB; page-crossing resumes ZERO integrity fails.
- GLM-5.3-Flash-UD-Q3_K_XL GGUF (0.82/400k): tier active (KpoolDSA), wash eliminated (0/11 full misses), prefetch A/B delta -0.1s/+0.02s = noise (honest negative; engaged once).
- Qwen3.8-27B-NVFP4-FTW (DENSE-MLP but HYBRID 48 GDN+16 full; KV 64 KiB/tok; ft checkpoint convert 5.6s pass-through; boot 21s): fill 8/8 HIT median 4.3s, flush 21 seg -> L2 23.8 GiB, scheduled compaction FIRED on boot (->14.75 GiB), BUG-1-shape resumes all HIT zero integrity fails. Scripts .tasks/session-cache-tiering/arm27b_*.sh; report REPORT-hw-arms-qsa-prefetch.md.
- Refuted: "radix wiped after refused restore" - token-usage metric excludes evictable pages (0.00 != empty); CPU repro clean.
- Open: plain DSAKVCache index slab not demoted (pre-existing committed behavior; needs codec branch or refusal); prefetch default-on with ~zero measured delta; KV-only HW arm deferred (no plain-radix model on the rig - all three models are hybrid).

## kb distilled (9503cb4)
kb/baselines/session-cache-tiering-arms.md (per-model anchors with configs); kb/topics/session-cache-tiering.md phase-2 section; glossary +7 terms; TASK.md EXECUTED; checks table_lint 0 / links 68-0.

## NEXT TASK (brief ready, fresh session)
.tasks/session-cache-tiering/briefs/offer-supersede.md: offer-time dedup of strictly-contained same-path L2 segments (23.8 GiB -> ~5-7 GiB expected; snapshot-bearing deeper offers must persist; identity-keyed compaction preserved); S-M + one HW arm. User accepted candidates as-is (topic 274 lines kept).
