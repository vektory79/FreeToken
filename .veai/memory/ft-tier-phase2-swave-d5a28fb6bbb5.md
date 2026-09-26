---
name: "ft-tier-phase2-swave"
description: "Tier phase-2; offer-supersede e5dd1f2 no-op on hybrid flush; kb amended, l2-snapshot-dedup brief ready (extents)"
type: project
lastUpdated: 2026-09-25T18:29
lastRecall: 2026-09-25T23:37
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

## OFFER-SUPERSEDE EXECUTED (2026-09-24) - HW verdict: NO-OP on hybrid flush
Brief briefs/offer-supersede.md executed. CPU: uncommitted diff (session_tier.py +43/-6; offers_dedup skip + supersede via _release_l1+_discard; containment proven by page_keys prefix - KEY FACT: path_key is the VictimPath TIP chain key (hybrid_radix_cache.py:308), so successive boundaries of one session have DIFFERENT path_keys and same-path matching MUST use the page_keys prefix relation; gate 352 passed/1 skipped; review x2 PASS, accepted failure window documented at the supersede loop). HW arm 27B (same protocol as 23.8 GiB baseline): blob 23.81 GiB bit-identical to baseline, dedup=0, offers=21/0 - NO-OP; reboot HIT + 0 integrity fails + compaction identical (no regression; end state byte-identical to pre-diff baseline dir). Calibration: 21/21 segments SNAPSHOT-BEARING (147 MiB GDN fp32 snapshot each; kv ~20.6 GiB + snaps 3.07 GiB) - the 2-4x bloat is KV copies of contained boundaries inside snapshot-bearing segments, unreachable by offer-seam dedup by design (snapshots at different boundaries persist). Brief premise refuted for the hybrid flush path. Next step is a DESIGN DECISION (blob-level page dedup / reference-sharing for snapshot-bearing segs / flush-only-deepest), not a bug fix. Logs: .tasks/session-cache-tiering/arm27b_dedup*; classifier tier_journal_classify.py; gotcha: blob.bin is append-only with holes after compaction - stat size is NOT the metric, use the "l2 used" log value.

UPDATE 2026-09-25: offer-supersede COMMITTED on vektory79 as e5dd1f2 "feat(scheduler): supersede contained same-chain tier segments at offer" (3 files, +278/-2; honest body: measured no-op on hybrid flush, helps snapshot-free plain-radix offers). No push, no PR. kb topic NOT yet amended with the arm verdict - open follow-up, together with the L2-dedup design decision.

## FOLLOW-UPS DONE (2026-09-25, uncommitted)
- kb/topics/session-cache-tiering.md amended (274 -> 346 lines): new dated section "Supersede вложенных сегментов на offer (2026-09-25)" - what landed (e5dd1f2, page_keys-prefix correction), HW arm verdict (honest no-op on hybrid flush, verbatim metrics line), calibration (21/21 snapshot-bearing, 147 MiB GDN snaps; bloat unreachable at offer seam), open follow-up pointer; frontmatter branch-commits + status updated; kb/topics/README.md hook extended. Checks: table_lint 0 issues; links 614 checked / 0 broken. NOT committed.
- Design brief READY: .tasks/session-cache-tiering/briefs/l2-snapshot-dedup.md (333 L). Recommendation: direction (a) content/page-key-addressed KV extents in the blob, written at the _demote/_offer payload seam, restricted to prefix-proven same-chain sharing; (d) merge as compact()'s legacy-migration/defrag lever. Expected L2 ~7 GiB (one full KV ~4 GiB + 21 snaps ~3.07 GiB) vs 23.81. Direction (c) flush-only-deepest REFUTED - dropping contained snapshots breaks _restore_tail exact-depth gate for measured multi-depth reboot HITs. New anchor facts: session_tier.py is 1152 L at e5dd1f2 (old line refs drifted: _journal_append ~L943, compact ~L1042 keep-set ~L1061, _discard ~L622); compact() today copies live spans at ORIGINAL offsets and rewrites journal VERBATIM (record identity stable -> (a) can keep the (path_key, l2_off, l2_n) keep-set unchanged). Verdict: Size L (additive journal field + dual-read legacy path, extent refcount lifecycle, ~8-10 new tests, one HW arm reusing arm27b_dedup.sh + tier_journal_classify.py; acceptance asserts "l2 used" log value, not blob.bin stat).
- kb agent flagged (out of scope, not done): kb/baselines/session-cache-tiering-arms.md lacks the dedup-arm anchors; top-level kb/README.md topics list omits session-cache-tiering.md (pre-existing gap).
