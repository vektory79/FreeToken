---
name: "ft-qwen38-flashnext-tier-fit"
description: "Qwen3.8 qwen4_exp QSAKVCache tier-fit anchors; QSA corruption hazard RESOLVED by phase-2 codec (ft-tier-phase2-swave)"
type: project
lastUpdated: 2026-09-26T17:24
lastRecall: 2026-09-26T17:21
---

# Qwen3.8-Flash-Next (qwen4_exp) tier fit + phase-2 scoping

**STATUS 2026-09-26: the QSA hazard below is RESOLVED** - the phase-2 S-wave landed the QSA codec (slab rows + mrope rope rows, explicit isinstance branch) and Qwen3.8-Flash-Next-NVFP4-FTW tiering is HW-verified (reboot HIT 48256/38016 @ 3.3-4.0s, zero integrity fails); all five phase-2 items (mr>1 leak fix, async prefetch, scheduled compaction, metrics, non-hybrid tiers) EXECUTED. Details and commit chain: ft-tier-phase2-swave. The body below is kept for the code anchors; scoping line refs predate the S-wave and have drifted.

## Qwen3.8-Flash-Next-NVFP4-FTW
- Checkpoint exists: /media/ai/models/RadixArk/Qwen3.8-Flash-Next-NVFP4-FTW/ (FTW shards + config.json). Family = qwen4_exp (python/freetoken/models/qwen4_exp/): HYBRID - GDN linear-attention layers + 12-of-48 QSA full-attention (attention.py:1) + hyper-connections (hc.py) + PLE slot states in LinearStatePool (qwen4_exp/config.py:78-79,260).
- Runtime KV pool = QSAKVCache(MHAKVCache) (kvcache/qsa_pool.py:38): GQA 4-D _kv_buffer + compressed index-key slab _cmp_k_buffer + pending ring; AttnType.QSA when index_ratio>1 (models/config.py:222-226); cache_type -> hybrid_radix (engine.py:1984-1990).
- (HISTORICAL hazard, fixed) _KVPageBytes originally stored only _kv_buffer+scales; _cmp_k_buffer slab and _rope_positions were NOT stored -> restore yielded correct K/V with stale slab rows -> wrong block selection on the 12 QSA layers = silent corruption. The S-wave codec now persists slab rows [page*ps/ratio) + rope rows via the isinstance branch.
- GDN/PLE snapshot machinery IS covered for it (slot unpack incl. slot_states, cache.py:60-66,891). Family last commits: cade1a9 #486, 08d728d #454.

## Phase-2 item scoping (S<200 / M 200-500 / L >500-or-design) - ALL EXECUTED in S-wave (2348da5)
1. Async prefetch: M; restore was sync in match_req (cache.py:226-233, try_restore :863-920) called from _try_allocate_one (prefill.py:73); landed as RestoreTicket (cap _MAX_RESTORE_TICKETS=2, 0=off) + adopt-at-admission sync gate order.
2. mr>1 concurrency: the real bug was the single-slot _tier_pending_restore (cache.py:166/:917) -> per-handle pending restore landed in S-wave.
3. Scheduled compaction: landed (dead>25% AND >256 MiB; offer/evict_store/run_when_idle).
4. Metrics: landed (Counter -> batch-log fragment + shutdown line; /v1/stats still deferred).
5. Non-hybrid: QSA codec (done); generic KV-only tier (tip-only set, RestoredRadixHandle) landed.
