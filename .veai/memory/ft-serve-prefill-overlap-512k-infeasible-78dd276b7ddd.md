---
name: "ft-serve-prefill-overlap-512k-infeasible"
description: "prefill overlap 2E floor: infeasible at KV=524288 (assert/OOM); also bites gguf per-signature cache partitions"
type: project
lastUpdated: 2026-09-16T16:42
lastRecall: 2026-09-16T16:40
---

# ft serve --moe-prefill-overlap: slot-floor incompatibility with big KV (GLM-5.3, RTX 5090)

Question (2026-09-12): can dropping `--disable-moe-prefill-overlap` at KV=524288 speed up prefill? Answer: NO - two measured failure modes on GLM-5.3-Flash-NVFP4-FTW / hybrid / nvfp4 KV:

- ratio 0.89 + overlap ON: BOOT FAILS with the cache-budget assert (engine/cache_budget.py -> resolve_moe_cache_auto): "minimum plan (moe=576 slots, kv=8192 pages) needs 10,216,275,968 B > budget 9,324,940,922 B". Root cause: with prefill overlap the slot floor is 2*num_experts = 576 (double-buffer holds two whole layers, ~14.4 MB/slot -> 8.09 GiB) + 524288-token nvfp4 KV (~2.0 GiB) > the 0.89 budget. Note: the cache_budget.py:60-80 docstring suggests overlap silently auto-disables when hi < 2E, but that relaxation applies only with an explicit max_slots cap - the greedy auto plan asserts instead.
- ratio 0.95 + overlap ON: BOOTS (resolved moe_cache_size=650 slots, prefill_overlap=True) but free after init = 0.99 GiB -> the FIRST prefill OOMs ("Tried to allocate 128.00 MiB" with 48.5 MiB free): greedy fill spends exactly the VRAM the prefill transients need (~2.4+ GiB at 4096 chunk).
- Budget math: overlap min plan ~10.2 GB at 512k; every bootable ratio leaves < ~1.5 GiB free, so first-prefill OOM is structural, not a tuning issue. Overlap would only become feasible around KV <= ~250-300k tokens on this card (estimate, unmeasured).
- Practical consequence: prefill levers at 512k are exactly the chunk size - `--memory-ratio 0.85 --max-prefill-length 8192` = +36% (see glm53-post-iommu-baseline). --disable-moe-prefill-overlap stays load-bearing at any big KV reserve.
- Both failures were caught and self-reported by the /tmp/ft_run_config.sh watchdog harness (FAILPAT grep: AssertionError|OutOfMemoryError|Backend worker is gone).

## Cross-context instance (2026-09-13/14): the same 2E invariant bites multi-partition offload caches (gguf glm5next)
- The gguf per-signature OffloadMoeCache partitions (GLM-5.3 GGUF, 3 width signatures) set per-partition floors of 1x num_experts; prefill_overlap=True needs 2xE per partition (offload_cache.py:176 __post_init__) -> the real-file boot died at 131s: 1326 auto slots < 3x576 = 1728 minimum.
- RESOLVED in Phase 5 of the gguf campaign (see ft-serve-gguf-glm5next-campaign): per-partition overlap DEGRADE via Engine._partition_prefill_overlap - overlap only when the group's slots >= 2*num_experts; minority signatures run synchronous materialized prefill. Deliberately NOT a per-partition 2E budget floor: flooring wide minority signatures would cost ~16 GiB. The Phase 6 tuned plan runs overlap OFF on all partitions (464/288/288 < 576).
- Lesson: ANY budget split that turns one MoE cache into partitions must resolve the 2E overlap invariant PER PARTITION (here: per-group degrade; a blanket 2E floor in the split would over-reserve, and the greedy auto plan only knows the pre-split total and asserts downstream).

UPDATE 2026-09-15: the per-signature FLOOR variant of this lesson now has an honest EARLY gate before banks load (Engine._gguf_auto_floor_gate + cache_budget.partition_floor_slots / check_partition_floors; real-file reject ~+17s vs +85s) - invariants in ft-gguf-nvfp4-1m-capacity-frozen-shim. The whole-cache 2E-overlap infeasibility above is unchanged.
