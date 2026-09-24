---
name: "ft-qwen38-flashnext-tier-fit"
description: "Qwen3.8 qwen4_exp hybrid: QSAKVCache passes tier gate, codec incomplete -> corruption risk; phase-2 scoping S/M/L"
type: project
lastUpdated: 2026-09-23T20:15
lastRecall: 2026-09-23T20:48
---

# Qwen3.8-Flash-Next (qwen4_exp) tier fit + phase-2 scoping

2026-09-23 research (agent 16cc58b0) after tier phase-1 closure (d77d15e -> 4b6aacb -> f90b081, gate 296/1).

## Qwen3.8-Flash-Next-NVFP4-FTW
- Checkpoint exists: /media/ai/models/RadixArk/Qwen3.8-Flash-Next-NVFP4-FTW/ (FTW shards + config.json). Family = qwen4_exp (python/freetoken/models/qwen4_exp/): HYBRID - GDN linear-attention layers + 12-of-48 QSA full-attention (attention.py:1) + hyper-connections (hc.py) + PLE slot states in LinearStatePool (qwen4_exp/config.py:78-79,260).
- Runtime KV pool = QSAKVCache(MHAKVCache) (kvcache/qsa_pool.py:38): GQA 4-D _kv_buffer + compressed index-key slab _cmp_k_buffer + pending ring; AttnType.QSA when index_ratio>1 (models/config.py:222-226); cache_type -> hybrid_radix (engine.py:1984-1990).
- HAZARD: QSAKVCache PASSES the _kv_buffer gate, so the session tier ACTIVATES for it if tier flags are passed - but _KVPageBytes stores only _kv_buffer+scales; _cmp_k_buffer slab and _rope_positions are NOT stored -> restore yields correct K/V with stale slab rows -> wrong block selection on the 12 QSA layers = SILENT corruption. User's current serve command has NO tier flags -> tier fully OFF (scheduler.py:97-101) -> nothing broken today.
- Fix path: interim guard (refuse tier activation when pool is QSAKVCache, cache.py:176; S, CPU); full fix = extend _KVPageBytes with per-page slab rows [page*ps/ratio,(page+1)*ps/ratio) (page_size % index_ratio == 0, qsa_pool.py:133) + rope rows; the size-based dim locator (cache.py:92-97) will NOT find the slab (token dim ntok/ratio+req_slots) -> explicit QSA branch, ~40-80 lines + CPU roundtrip tests, then HW acceptance with this checkpoint.
- GDN/PLE snapshot machinery IS covered for it (slot unpack incl. slot_states, cache.py:60-66,891). Family last commits: cade1a9 #486, 08d728d #454 - no tiering regression risk.

## Phase-2 item scoping (S<200 lines / M 200-500 / L >500-or-design)
1. Async prefetch: M; restore is sync in match_req (cache.py:226-233, try_restore :863-920) called from _try_allocate_one (prefill.py:73); needs probe+begin/pinned staging/adopt-at-admission + page+slot reservation + abandon-on-death; store pread-safe (session_tier.py:529-553); HW-gated end-to-end.
2. mr>1 concurrency: REAL BUG (S, CPU now) - _tier_pending_restore is a single slot (cache.py:166/:917, cleared in lock() :368-372 and abandon_restore :920-934); a second restore overwrites the pending record -> abandon_restore of the first no-ops -> pages + GDN slot leak. Dormant at mr=1; live with mr>1 or async prefetch. Fix: per-handle pending dict + tests. Store locks (session_tier.py:157 global, :75 per-segment) become load-bearing once threads appear.
3. Scheduled compaction: S-M, CPU; compact() (:657-696, crc-replay crash-safe) only in shutdown_tier (cache.py:783-785); needs dead-byte accounting, trigger policy (watermark / every-N-offers), idle safe-point hook (precedent _pending_rebuild, scheduler.py:120 area), and lock coverage (restore blob reads vs rewrite race).
4. Metrics: S, CPU; no prometheus; export paths = /v1/stats StatsTracker (server/stats.py, fed by FrontendManager, api_server.py:248) + daemon GET /engine/metrics (daemon/app.py:277); hook counters in offer/probe/restore/note_match; byte counters already exist (_l1_used, _ssd_used, _blob_eof).
5. Non-hybrid: (a) QSA codec/guard - see above; (b) generic KV-only tier for plain radix models: M - move _tier_admission out of the is_hybrid branch of match_req (cache.py:223-228), KV-only offer (supported: snapshot_slot=None, cache.py:849-856), KV-only restore bypassing the snaps+snap_bound gate (:894-898), shutdown offer gate (:781); CPU logic + HW acceptance.

## Recommendation
One CPU-only session can land: 2 (leak fix) + QSA guard + 3 + 4. Own briefs each: 1 (async prefetch), QSA codec (HW acceptance on the RadixArk checkpoint), 5b (generic non-hybrid).
