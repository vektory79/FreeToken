from __future__ import annotations

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, List, Tuple

import torch
from freetoken.core import Req
from freetoken.kvcache import BaseCacheHandle, MatchResult, create_prefix_cache
from freetoken.utils import align_down, div_ceil
from freetoken.utils.logger import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    from .utils import PendingReq

# Proactive out-of-window free_swa runs every `interval` forwards (== sglang SWA_EVICTION_INTERVAL).
def _swa_eviction_interval() -> int:
    raw = os.environ.get("FREETOKEN_SWA_EVICTION_INTERVAL", "128")
    try:
        return max(1, int(raw))
    except ValueError:
        raise ValueError(f"FREETOKEN_SWA_EVICTION_INTERVAL must be an integer, got {raw!r}")


_SWA_EVICTION_INTERVAL = _swa_eviction_interval()

# Finish-time retention keeps [P - window - gap, P) swa-live for the next turn's cut near the
# prompt end. The gap covers templates whose generation prompt injects tokens that vanish when
# the client drops reasoning (Qwen's "\ key\n": the re-render diverges 2 tokens BEFORE P).
_SWA_RETAIN_GAP = 16


class SessionTierCfg:
    """Raw sizes for the session-tier store; GiB flags are converted by the caller.
    ram_bytes == 0 and dir None == the store is inert (SessionTierStore.enabled False)."""

    __slots__ = ("ram_bytes", "dir", "ssd_bytes")

    def __init__(self, ram_bytes: int = 0, dir=None, ssd_bytes: int = 0):
        self.ram_bytes = ram_bytes
        self.dir = dir
        self.ssd_bytes = ssd_bytes


class _SlotSnapshot:
    """SnapshotSource view of one LinearStatePool slot: conv + recurrent + slot_states packed
    in pool-attribute order. The store materializes bytes inside offer(); the slot must still
    be LIVE at that moment (offer before free)."""

    def __init__(self, pool, slot: int):
        self._pool, self._slot = pool, slot

    def payload(self) -> bytes:
        return b"".join(
            p.contiguous().cpu().view(torch.uint8).numpy().tobytes()
            for p in _slot_parts(self._pool, self._slot))


def _slot_parts(pool, slot: int):
    parts = [pool.conv_states[:, slot], pool.recurrent_states[:, slot]]
    parts.extend(t[:, slot] for t in pool.slot_states.values())
    return parts


def _unpack_slot(pool, slot: int, data: bytes) -> None:
    pos = 0
    for p in _slot_parts(pool, slot):
        n = p.numel() * p.element_size()
        flat = torch.frombuffer(bytearray(data[pos:pos + n]), dtype=torch.uint8)
        p.copy_(flat.view(p.dtype).reshape(p.shape).to(p.device))
        pos += n
    assert pos == len(data), "snapshot byte length mismatch"


class _KVPageBytes:
    """Byte codec over the paged KV pool (MHAKVCache-family ``_kv_buffer`` layout), scale
    buffers included so a restored page is byte-identical. W2 scope: hybrid-linear models
    route through this pool family; others keep the tier inert (never offered, never probed)."""

    def __init__(self, pool):
        self._pool = pool
        buf = pool._kv_buffer
        self._ps = int(buf.shape[3])
        self._ntok = int(buf.shape[2]) * self._ps
        # Scale buffers are token-indexed (num_pages * page_size) but sit at a
        # family-specific dim: MHAKV [kv, L, tokens, heads(, blocks)] dim 2, DSA/MLA
        # [L, tokens(, blocks)] dim 1. Locate it by size; the assert keeps a wrong
        # layout from silently storing garbage instead of raising at boot.
        self._scales = []
        for name in ("_scale_buffer", "_block_scale_buffer"):
            s = getattr(pool, name, None)
            if s is None:
                continue
            dim = next((d for d, n in enumerate(s.shape) if n == self._ntok), None)
            assert dim is not None, (
                f"{name} shape {tuple(s.shape)} does not span the pool's page grid")
            self._scales.append((s, dim))

    def _scale_page(self, s, dim, page):
        idx = [slice(None)] * s.dim()
        idx[dim] = slice(page * self._ps, (page + 1) * self._ps)
        return s[tuple(idx)]

    def read_page(self, page: int) -> bytes:
        buf = self._pool._kv_buffer
        parts = [buf[:, :, page]]
        parts += [self._scale_page(s, dim, page) for s, dim in self._scales]
        return b"".join(
            t.contiguous().cpu().view(torch.uint8).numpy().tobytes() for t in parts)

    def write_page(self, page: int, data: bytes) -> None:
        buf = self._pool._kv_buffer
        parts = [buf[:, :, page]]
        parts += [self._scale_page(s, dim, page) for s, dim in self._scales]
        pos = 0
        for t in parts:
            n = t.numel() * t.element_size()
            flat = torch.frombuffer(bytearray(data[pos:pos + n]), dtype=torch.uint8)
            t.copy_(flat.view(t.dtype).reshape(t.shape).to(t.device))
            pos += n
        assert pos == len(data), "KV page byte length mismatch"


class CacheManager:
    def __init__(self, num_pages: int, page_size: int, page_table: torch.Tensor, type: str,
                 linear_state_pool=None, swa_pool=None, sliding_window_size=None,
                 session_tier_cfg=None):
        # The `_free_slots` follows a page-aligned manner. For example, if page_size = 2,
        # the `_free_slots` may look like [0, 2, 4, 6, ...], and each slot represents a page.
        device = page_table.device
        self.free_slots = torch.arange(num_pages, dtype=torch.int32, device=device) * page_size
        # Hybrid GDN models drive a second currency (GDN state snapshots in LinearStatePool)
        # through a HybridRadixCache; SWA models drive a second currency (swa-pool KV slots in
        # the HybridSWAKVCache global-paged mode) through a SWARadixCache; non-hybrid models keep
        # the plain naive/radix path.
        self.linear_state_pool = linear_state_pool
        self.swa_pool = swa_pool
        self.sliding_window_size = sliding_window_size
        self.is_hybrid = type == "hybrid_radix"
        self.is_swa = type == "swa_radix"
        # swa_paged: this SWA model drives the global-paged swa pool -- true for BOTH the naive
        # (NaivePrefixCache, no reuse) and radix (SWARadixCache) paths. Gates the swa slot
        # lifecycle (alloc_swa / out-of-window free / free-on-finish). is_swa gates only the extra
        # SWARadixCache reuse machinery (tree match/insert/evict_swa/swa_uuid lock).
        self.swa_paged = swa_pool is not None and getattr(swa_pool, "swa_paged", False)
        # Owned-pool capability pickup: a plugged-in swa pool may cap the prefill chunk (DSV4:
        # ~half the window working set). Instance attrs shadow the class defaults; absent
        # attributes leave the defaults untouched (Gemma4).
        if swa_pool is not None:
            self.prefill_chunk_budget = getattr(swa_pool, "prefill_chunk_budget", None)
        self.prefix_cache = self._make_prefix_cache(device, page_size, type)
        self.device = device
        self.num_pages = num_pages
        self.page_table = page_table
        self.page_size = page_size
        self.cache_type = type
        # Session tiering (W2): inert unless a hybrid manager gets an enabled cfg AND a byte
        # codec for its KV pool; off == bit-identical behavior (every hook early-returns).
        self.tier_store = None
        self._tier_on = False
        self._tier_sessions: dict = {}
        self._tier_snap_bound: dict = {}
        self._tier_restore_slots: set = set()
        self._tier_pending_restore = None
        self._page_bytes = None
        if self.is_hybrid and self.linear_state_pool is not None and session_tier_cfg is not None:
            from .session_tier import SessionTierStore

            store = SessionTierStore(session_tier_cfg)
            if store.enabled and getattr(swa_pool, "_kv_buffer", None) is not None:
                store.replay_journal()   # boot hook: rebuild the L2 index from the journal
                self.tier_store = store
                self._page_bytes = _KVPageBytes(swa_pool)
                self._tier_on = True
                # _tier_snap_bound is process-local (offer-time); journal-replayed segments
                # are invisible to try_restore's snapshot-at-boundary gate without this,
                # so the first post-boot turn would fall back to a full re-prefill.
                for seg in store._segments.values():
                    if seg.snap_lens and seg.snap_lens[0]:
                        self._tier_snap_bound[seg.path_key] = seg.boundary_len
            elif store.enabled:
                logger.warning("session tier: no paged-KV byte codec for this pool family; "
                               "tier stays inert")

    # ----- capability hooks (defaults; plugged-in pools may narrow them) -----
    supports_runtime_rebuild = True
    prefill_chunk_budget = None  # generic shared page pool: no per-model prefill chunk cap

    @property
    def prefill_chunk_align(self) -> int:
        """Granularity a non-final prefill chunk should end on. A hybrid snapshot is donated only
        at a page-aligned boundary, so at page_size>1 one unaligned chunk end costs every reuse
        point for the rest of the prompt. 1 (no-op) everywhere else."""
        return self.page_size if self.is_hybrid else 1

    def page_usage(self) -> tuple[int, int]:
        """(used_pages, total_pages): allocated, non-evictable pages over the pool total
        (active requests + protected prefix; evictable prefix-cache pages are excluded)."""
        total = self.num_pages
        evictable = (self.prefix_cache.full_evictable_size if (self.is_hybrid or self.is_swa)
                     else self.prefix_cache.size_info.evictable_size)
        return total - len(self.free_slots) - evictable // self.page_size, total

    def _make_prefix_cache(self, device, page_size, type):
        if type == "hybrid_radix":
            from freetoken.kvcache.hybrid_radix_cache import HybridRadixCache
            return HybridRadixCache(device, page_size)
        if type == "swa_radix":
            from freetoken.kvcache.swa_radix_cache import SWARadixCache
            return SWARadixCache(device, page_size, self.sliding_window_size)
        return create_prefix_cache(device=device, type=type, page_size=page_size)

    def match_req(self, req: PendingReq) -> MatchResult:
        input_len = req.input_len
        assert input_len > 0, "Input length must be greater than 0."
        ids = req.input_ids[: input_len - 1]
        if self.is_swa:
            from freetoken.kvcache.swa_radix_cache import SWACacheHandle
            m = self.prefix_cache.match_prefix(ids)
            return MatchResult(SWACacheHandle(m.cached_len, m.node, m.kv_indices))
        if self.is_hybrid:
            from freetoken.kvcache.hybrid_radix_cache import HybridCacheHandle
            m = self.prefix_cache.match_prefix(ids)
            self._tier_admission(ids, m.cached_len)
            restored = self.try_restore(ids, m.cached_len)
            if restored is not None:
                return restored
            return MatchResult(
                HybridCacheHandle(m.cached_len, m.node, m.kv_indices), mamba_value=m.mamba_value)
        return self.prefix_cache.match_prefix(ids)

    @property
    def available_size(self) -> int:
        evictable = (self.prefix_cache.full_evictable_size if (self.is_hybrid or self.is_swa)
                     else self.prefix_cache.size_info.evictable_size)
        return evictable + len(self.free_slots) * self.page_size

    @property
    def mamba_available_size(self) -> int:
        """Hybrid only: free GDN state slots + evictable (unlocked) tree snapshots."""
        return self.linear_state_pool.num_free_slots + self.prefix_cache.mamba_evictable_size

    @property
    def swa_available_size(self) -> int:
        """SWA only: free swa-pool slots + (radix) evictable unlocked live tree swa tokens.
        Naive has no tree, so only the free-list counts."""
        tree = self.prefix_cache.swa_evictable_size if self.is_swa else 0
        return self.swa_pool.swa_available_size() + tree

    def ensure_swa_slots(self, n: int) -> None:
        """Free swa-pool slots until >= ``n`` are available by tombstoning LRU tree swa nodes
        (evict_swa, internal -> tombstone in place / leaf -> free both pools), returning their swa
        slots to the pool and any deleted-leaf full KV to free_slots."""
        while self.swa_pool.swa_available_size() < n:
            ev = self.prefix_cache.evict_swa(n - self.swa_pool.swa_available_size())
            if ev.swa_indices.numel() == 0:
                break
            self.swa_pool.free_swa(ev.swa_indices)
            if ev.kv_indices.numel():
                self._free(ev.kv_indices)

    def ensure_mamba_slots(self, n: int) -> None:
        """Free GDN state slots until >= ``n`` are available by tombstoning LRU tree snapshots
        (evict_mamba), returning their slots + any freed KV to the pools."""
        while self.linear_state_pool.num_free_slots < n:
            er = self.prefix_cache.evict_mamba(
                n - self.linear_state_pool.num_free_slots, derive_paths=self._tier_on)
            if not er.mamba_slots:
                break
            if self._tier_on:
                self._tier_offer(er.victim_paths)   # offer BEFORE the free: slots stay live
            self.linear_state_pool.free(er.mamba_slots)
            self._free(er.kv_indices)

    def snapshot_toolcall_anchor(self, reqs: List[Req]) -> None:
        """Freeze each decoding request's GDN state at its tool-call anchor, into the ping-pong
        slot that is idle during decode (the kernel-side ×CHUNK track only runs on prefill
        extends). Must run on the engine stream before the current step's kernels: cached_len
        equals the anchor exactly when every enqueued step up to the anchor-consuming one has
        been issued and the next (current) one has not, so the copy lands between them in
        stream order. Reuses ``mamba_last_track_seqlen`` as the pending-donate mark -- the
        prefill track's own pending freeze was consumed by the prefill-commit ``cache_req``
        before any decode drain could set an anchor."""
        if not self.is_hybrid:
            return
        pool = self.linear_state_pool
        for r in reqs:
            a = r.toolcall_anchor_len
            if (
                a is None
                or r.mamba_ping_pong is None
                or r.mamba_last_track_seqlen is not None
                or r.cached_len != a
                or align_down(a, self.page_size) != a
            ):
                continue
            dst = r.mamba_ping_pong[r.mamba_next_track_idx]
            pool.copy_from(r.linear_slot_idx, dst)
            r.mamba_last_track_seqlen = a
            r.mamba_next_track_idx = 1 - r.mamba_next_track_idx

    def maybe_free_swa_out_of_window(self, reqs: List[Req], *, forward_iter: int) -> None:
        """Proactively free each decoding request's now-out-of-window SWA slots, bounding its swa
        footprint to ~one window so a smaller-than-full swa pool (swa_full_tokens_ratio<1) stays
        viable. Mirrors sglang ``ScheduleBatch.maybe_evict_swa`` / ``free_swa_out_of_window_slots``:
        evict every ``interval`` forwards; skip a request's first decode step (its extend forward
        may still be in-flight under overlap); floor the frontier at the request's protected
        (reused) prefix so it only frees its OWN slots, never the tree-shared prefix's swa; and keep
        a ``window + page_size`` margin so the freed slots are out-of-window for every in-flight
        forward."""
        if not self.swa_paged or forward_iter % _SWA_EVICTION_INTERVAL != 0:
            return
        window = self.sliding_window_size
        for req in reqs:
            if req.decode_batch_idx < 1:
                continue                       # overlap guard: extend forward may still be running
            floor = req.cache_handle.cached_len   # reused prefix -> its swa is tree-owned, not ours
            threshold = (req.device_len - 1) - window - self.page_size
            if req.toolcall_anchor_len is not None:
                # Keep the window ending at the anchor resumable: a client-side rewrite of the
                # echoed tool call forks after the anchor, and a resume there needs
                # [anchor - window, anchor) live. The finish-insert then adopts (rather than
                # tombstones) these never-evicted slots; they stay unlocked, so real pool
                # pressure can still reclaim them (same soft retention as the prompt-end pin).
                cap = req.toolcall_anchor_len - window - _SWA_RETAIN_GAP
                if threshold - cap > window + _SWA_RETAIN_GAP:
                    # The decode ran on far past the anchor (a tool call is normally within
                    # tens of tokens of the end). Holding the cap would grow this request's
                    # live swa without bound ("SWA pool exhausted" is unhandled) -- drop the
                    # anchor and let normal eviction resume. This bound is what the
                    # anchor-retention term in _swa_per_req_swa_floor sizes the pool for.
                    req.toolcall_anchor_len = None
                else:
                    threshold = min(threshold, cap)
            new_evicted = align_down(threshold, self.page_size)
            start = max(req.swa_evicted_seqlen, floor)
            if new_evicted > start:
                self._free_swa(self.page_table[req.table_idx, start:new_evicted])
                req.swa_evicted_seqlen = new_evicted

    def free_swa_out_of_window_extend(self, reqs: List[Req]) -> None:
        """Prefill sibling of ``maybe_free_swa_out_of_window``: before allocating a chunk, return
        each request's now-out-of-window SWA slots so a chunked prompt's live swa stays ~one window
        regardless of prompt length (else a prompt longer than the swa pool exhausts alloc_swa).
        Runs on EVERY prefill batch -- no eviction-interval cadence, since a long prompt would
        overflow the pool before a cadence fires. The frontier is based on ``cached_len`` (the
        pre-chunk, already-forwarded length; the chunk ``[cached_len, device_len)`` is allocated by
        the following ``allocate_paged``, not here), so only positions prior chunks consumed are
        freed, floored at the tree-owned reused prefix. Overlap-safe by the same scheduler stream
        gate + ``window + page_size`` margin the decode driver relies on; ``free_swa`` is idempotent
        over the sentinel, so re-freeing an earlier chunk's range is a no-op. The pool is always
        sized > one window (see the swa-pool floor), so a chunk can always make forward progress."""
        if not self.swa_paged:
            return
        window = self.sliding_window_size
        for req in reqs:
            floor = req.cache_handle.cached_len   # reused prefix -> its swa is tree-owned, not ours
            new_evicted = align_down(req.cached_len - window - self.page_size, self.page_size)
            start = max(req.swa_evicted_seqlen, floor)
            if new_evicted > start:
                self._free_swa(self.page_table[req.table_idx, start:new_evicted])
                req.swa_evicted_seqlen = new_evicted

    def lock(self, handle: BaseCacheHandle) -> None:
        # A locked handle means admission consumed the restored match: clear the pending
        # restore deterministically here (abandon_restore only handles the refusal paths).
        pending = self._tier_pending_restore
        if pending is not None and pending[0] is handle:
            self._tier_pending_restore = None
        if self.is_swa:
            # records the window boundary on the (frozen) handle for unlock/dec_lock.
            object.__setattr__(handle, "swa_uuid", self.prefix_cache.inc_lock(handle.node))
        elif self.is_hybrid:
            self.prefix_cache.inc_lock(handle.node)
        else:
            self.prefix_cache.lock_handle(handle, unlock=False)

    def unlock(self, handle: BaseCacheHandle) -> None:
        if self.is_swa:
            self.prefix_cache.dec_lock(handle.node, handle.swa_uuid)
        elif self.is_hybrid:
            self.prefix_cache.dec_lock(handle.node)
        else:
            self.prefix_cache.lock_handle(handle, unlock=True)

    def _free_swa(self, indices: torch.Tensor) -> None:
        """Free the swa-pool slots backing ``indices`` (full-pool slots). Idempotent over the
        0 sentinel, so safe to call on any slots being returned to free_slots."""
        if self.swa_pool is not None and len(indices) > 0:
            self.swa_pool.free_swa(indices)

    def allocate_paged(self, reqs: List[Req]) -> None:
        needed_pages = 0
        allocation_info: List[Tuple[int, int, int]] = []
        for req in reqs:
            first_page = div_ceil(req.cached_len, self.page_size)
            last_page = div_ceil(req.device_len, self.page_size)
            if last_page > first_page:
                needed_pages += last_page - first_page
                allocation_info.append((req.table_idx, first_page, last_page))
        if needed_pages > 0:
            allocated = self._page_to_token(self._allocate(needed_pages))
            if self.swa_paged:
                # Each newly-allocated full token needs a swa-pool slot (where its SWA-layer KV is
                # written; read back via the full->swa mapping). radix reuses the existing prefix's
                # (live, mapped) swa slots and evicts tree swa if the pool is short; naive has no
                # tree (the pool is sized concurrency x window so it always fits).
                if self.is_swa:
                    self.ensure_swa_slots(len(allocated))
                self.swa_pool.alloc_swa(allocated)
            _write_page_table(self.page_table, allocated, allocation_info, self.page_size)

    def cache_req(self, req: Req, *, finished: bool) -> None:
        if self.is_swa:
            return self._cache_req_swa(req, finished=finished)
        if self.is_hybrid:
            return self._cache_req_hybrid(req, finished=finished)
        # ==================================== valid cache region ====================================
        # [0, req.cached_len)                       This part is valid for attention kernel read/write.
        # [0, old_handle.cached_len)                This part is in the prefix cache before prefill.
        # [old_handle.cached_len, req.cached_len)   This part is allocated by cache manager for this request.
        # ================================== allocated cache region ==================================
        # [old_handle.cached_len, cached_len)       This part was not in the prefix cache when prefill,
        #                                           but later cached by other requests.
        #                                           We must free them to avoid memory leak.
        # [cached_len, new_handle.cached_len)       This part is newly inserted into the prefix cache.
        # [new_handle.cached_len, req.cached_len)   This part is tailing part that can not inserted into the prefix cache.
        #                                           We should free it if the request has finished.
        page_indices = self.page_table[req.table_idx, : req.cached_len]
        old_handle = req.cache_handle
        insert_ids = req.input_ids[: req.cached_len]
        cached_len, new_handle = self.prefix_cache.insert_prefix(insert_ids, page_indices)
        # unlock until all operations on handle is done
        self.unlock(old_handle)
        # this part is already in the prefix cache, free it. A naive-SWA request (swa_paged, no
        # reuse) also returns the swa slots backing every full slot it frees; the out-of-window
        # ones were already freed by the decode driver (free_swa is idempotent over the sentinel).
        if self.swa_paged:
            self._free_swa(page_indices[old_handle.cached_len : cached_len])
        self._free(page_indices[old_handle.cached_len : cached_len])
        if finished:  # this tail part should be freed
            tail = self._padded_tail(req, new_handle.cached_len)
            if self.swa_paged:
                self._free_swa(tail)
            self._free(tail)
        else:  # keep the tail part, update the handle
            # Re-point the deduped span at the tree's canonical pages: the request's own pages
            # for [old_handle.cached_len, cached_len) went back on the free list above, but the
            # attention backends read this row every step and the next allocation hands those
            # pages to someone else. [0, old_handle.cached_len) needs no rewrite -- it has been
            # locked since admission, so the row already equals canonical there.
            if cached_len > old_handle.cached_len:
                canonical = new_handle.get_matched_indices()
                self.page_table[req.table_idx, old_handle.cached_len : cached_len].copy_(
                    canonical[old_handle.cached_len : cached_len])
            req.cache_handle = new_handle
            self.lock(new_handle)

    def _cache_req_hybrid(self, req: Req, *, finished: bool) -> None:
        """Hybrid (GDN) cache_req: commit KV like radix AND manage the GDN state snapshot.
        Prefill chunk commit: DONATE the frozen ping-pong slot (the snapshot the forward wrote
        at the tracked ×64 boundary mamba_last_track_seqlen) into the tree; replace it with a
        fresh slot if the tree took it (dedup keeps it for reuse). Finish: donate the live slot
        (final full-sequence state, zero-copy since the req is done) and free the req's slots."""
        from freetoken.kvcache.hybrid_radix_cache import HybridCacheHandle

        pool = self.linear_state_pool
        old_handle = req.cache_handle
        page_indices = self.page_table[req.table_idx, : req.cached_len]

        if finished:
            # A pending freeze (the tool-call anchor, or a prefill ×64 track the request
            # finished too early to chunk-commit) is a strictly shorter prefix than the live
            # donate below: insert it first and advance the dedup-free floor to its boundary
            # -- [prefix_len, L) is now tree-owned by the donated node, so only [old, prefix_len)
            # is this request's dup to free. The frozen slot is consumed either way (taken by
            # the tree or freed here) and both ping-pong refs are dropped before
            # _free_req_slots so nothing double-frees.
            free_upto = old_handle.cached_len
            L = req.mamba_last_track_seqlen
            # Invariant: unaligned L cannot occur in production (see the commit-site note); the gate is defensive, never a donate.
            if (
                L is not None
                and 0 < L <= req.cached_len
                and align_down(L, self.page_size) == L
                and req.mamba_ping_pong is not None
            ):
                frozen_idx = 1 - req.mamba_next_track_idx
                frozen = req.mamba_ping_pong[frozen_idx]
                prefix_len, mamba_exist = self.prefix_cache.insert(
                    req.input_ids[:L], page_indices[:L], frozen)
                pool.free([s for s in req.mamba_ping_pong if mamba_exist or s != frozen])
                req.mamba_ping_pong = None
                self._free(page_indices[free_upto : max(free_upto, prefix_len)])
                free_upto = max(free_upto, L)
            # Donate the live slot (final full-sequence state). The live state is at cached_len;
            # only attach it when cached_len is itself the page-aligned node boundary (always for
            # page_size==1). For page_size>1 a non-aligned cached_len would attach an over-advanced
            # state to a shorter prefix node -> skip the finish-donate (the ×64 prefill snapshots
            # remain as reuse points).
            insert_len = align_down(req.cached_len, self.page_size)
            keep_live = False
            if insert_len == req.cached_len and insert_len > 0:
                prefix_len, mamba_exist = self.prefix_cache.insert(
                    req.input_ids[:insert_len], page_indices[:insert_len], req.linear_slot_idx)
                self.unlock(old_handle)
                self._free(page_indices[free_upto : max(free_upto, prefix_len)])
                keep_live = not mamba_exist           # tree now owns linear_slot_idx
            else:
                self.unlock(old_handle)
                # A session-tier restore hands the admission its OWN fresh pages for the
                # suffix [owned, cached_len) (owned..cached_len may span the whole prefix
                # when the tree was empty): no prefill-chunk commit ran to adopt them when
                # the restored request's prefill covered only the page tail. The non-aligned
                # finish must still insert the page-aligned span so the tree owns those
                # pages, donating nothing (the live slot is over-advanced past insert_len;
                # the boundary snapshot lives in the tier store). Skipping the insert leaked
                # the restored pages (hardware: 896-page leak -> integrity check kill).
                # tier-restored handles only; with tier off this branch is unreachable and
                # behavior is unchanged.
                if (self._tier_on and old_handle.tier_restored and insert_len > 0
                        and old_handle.cached_len <= insert_len):
                    prefix_len, _mamba_exist = self.prefix_cache.insert(
                        req.input_ids[:insert_len], page_indices[:insert_len], None)
                    self._free(page_indices[free_upto:max(free_upto, prefix_len)])
                self._free(page_indices[free_upto :])
            self._free_req_slots(req, keep_live=keep_live)
            return

        # Prefill chunk commit: donate the frozen snapshot at the tracked ×64 boundary.
        L = req.mamba_last_track_seqlen
        if L is None:
            return  # no ×64 boundary crossed this chunk; req keeps its pages (committed later)
        if align_down(L, self.page_size) != L:
            # page_size>1 only: insert would align the key down, attaching a state that encodes
            # L tokens to a SHORTER node -- a future hit would COW-restore an over-advanced
            # state. Skip; the next aligned boundary (or the finish-donate) commits instead.
            # Invariant: unreachable in production (page-aligned chunk starts, CHUNK_SIZE % page_size == 0); do not turn the skip into a donation.
            req.mamba_last_track_seqlen = None
            return
        frozen_idx = 1 - req.mamba_next_track_idx          # the slot the forward just wrote
        frozen = req.mamba_ping_pong[frozen_idx]
        prefix_len, mamba_exist = self.prefix_cache.insert(
            req.input_ids[:L], page_indices[:L], frozen)
        self.unlock(old_handle)
        self._free(page_indices[old_handle.cached_len : prefix_len])
        # Lock the committed snapshot node FIRST: the replacement-slot alloc below can trigger
        # evict_mamba (via ensure_mamba_slots), which would otherwise reclaim this still-unlocked
        # just-donated node -- freeing its KV pages under the still-decoding request.
        m = self.prefix_cache.match_prefix(req.input_ids[:L])
        # Same re-point as the generic path: the dedup free above returned this request's own
        # pages for [old_handle.cached_len, prefix_len) while its row still named them.
        if prefix_len > old_handle.cached_len:
            self.page_table[req.table_idx, old_handle.cached_len : prefix_len].copy_(
                m.kv_indices[old_handle.cached_len : prefix_len])
        req.cache_handle = HybridCacheHandle(m.cached_len, m.node, m.kv_indices)
        self.lock(req.cache_handle)
        if not mamba_exist:                                # tree took `frozen`; replace it
            self.ensure_mamba_slots(1)
            pp = list(req.mamba_ping_pong)
            pp[frozen_idx] = pool.alloc(1)[0]
            req.mamba_ping_pong = tuple(pp)
        req.mamba_last_track_seqlen = None

    def _cache_req_swa(self, req: Req, *, finished: bool) -> None:
        """SWA cache_req: commit the request's full KV prefix into the SWARadixCache (node.value =
        the canonical full-pool page indices; the swa KV rides along via the full->swa mapping).
        Tokens < req.swa_evicted_seqlen are marked tombstone on insert. No donate/COW (the swa KV
        is in the pool already). On any dup/tail free, free both pools (full slot + its swa slot)."""
        from freetoken.kvcache.swa_radix_cache import SWACacheHandle

        old_handle = req.cache_handle
        page_indices = self.page_table[req.table_idx, : req.cached_len]

        insert_len = align_down(req.cached_len, self.page_size)
        freed = page_indices[:0]
        if insert_len > 0:
            # insert reconciles tombstones (revives the in-window ones by ADOPTING the request's
            # live-swa slots into node.value) and returns every full slot to reclaim: the displaced
            # old tree slots + the request's non-adopted dups. swa_evicted_seqlen is the request's
            # own extend/decode free frontier: insert must tombstone [.., swa_evicted_seqlen) rather
            # than adopt those (now-sentinel) swa slots. This holds for BOTH finished and unfinished
            # commits -- the extend driver frees out-of-window swa during chunked prefill too, so an
            # unfinished chunk's frontier is already > 0 and must be honored (else insert adopts
            # sentinel slots -> the request's later SWA gathers read slot 0 -> corruption).
            _, freed = self.prefix_cache.insert(
                req.input_ids[:insert_len], page_indices[:insert_len],
                swa_evicted_seqlen=req.swa_evicted_seqlen,
                update_kv_after_len=old_handle.cached_len)
        self.unlock(old_handle)
        self._free_swa(freed)   # idempotent: revived/out-of-window slots are already sentinel -> no-op
        self._free(freed)
        if finished:
            # Page-unaligned tail (page_size>1) not inserted. The padded slice reaches to the
            # page-ceil bound: allocate_paged charged a swa slot for EVERY token of the last
            # partial page (whole-page alloc_swa), so the padding slots must return with it or
            # they leak (-cached_len mod page_size slots per request, permanently).
            tail = self._padded_tail(req, insert_len)
            self._free_swa(tail)
            self._free(tail)
            # Soft-pin the prompt-end window: decode never re-stamps the prompt path, so after
            # the unlock above it is the stalest LRU entry and the first evict_swa victim. A
            # follow-up turn diverges at the prompt end when the client drops reasoning; a cut
            # there only needs the trailing window live, so eagerly reclaim the head's swa
            # (full KV stays for the full-attn layers) and re-stamp the retained tail -- still
            # unlocked, so it remains reclaimable under real pressure.
            prompt_len = align_down(req.max_device_len - req.output_len, self.page_size)
            if prompt_len > 0:
                keep_from = align_down(
                    max(prompt_len - self.sliding_window_size - _SWA_RETAIN_GAP, 0),
                    self.page_size,
                )
                if keep_from > 0:
                    self._free_swa(
                        self.prefix_cache.trim_head_swa(req.input_ids[:prompt_len], keep_from))
                self.prefix_cache.match_prefix(req.input_ids[:prompt_len])
        else:
            # inc_lock is node-granular, and the suffix insert just made this chunk's whole
            # extend one node: locking it would pin the entire chunk's swa for all of decode,
            # though the request reads only its trailing window from here on. Force a node
            # boundary a window back (match_prefix splits) so the lock lands on that window
            # alone. The head stays live and unlocked -- still reusable while the pool is
            # roomy, evictable the moment it is not.
            keep_from = align_down(
                max(insert_len - self.sliding_window_size - _SWA_RETAIN_GAP, 0), self.page_size)
            if keep_from > 0:
                self.prefix_cache.match_prefix(req.input_ids[:keep_from])
            m = self.prefix_cache.match_prefix(req.input_ids[:insert_len])
            # Re-point the page table to the tree's live slots for the committed region. Any dup
            # slots insert reclaimed had their full->swa mapping reset to the 0 sentinel; unlike the
            # full pool (KV survives in place until realloc), a stale swa mapping would make the
            # request's subsequent SWA gathers read the sentinel -> corruption. The reconcile revived
            # the in-window tombstones, so the re-matched slots are live.
            if m.cached_len > 0:
                self.page_table[req.table_idx, : m.cached_len].copy_(m.kv_indices)
            req.cache_handle = SWACacheHandle(m.cached_len, m.node, m.kv_indices)
            self.lock(req.cache_handle)

    def _padded_tail(self, req: Req, start: int) -> torch.Tensor:
        """The request's OWN slice [start, page_ceil(cached_len)) of the page table. A finish
        frees through the page-CEIL bound, not cached_len: allocate_paged allocates (and, when
        swa_paged, charges swa for) whole pages, so the padding [cached_len, page_ceil) belongs
        to the finishing request. ``start`` is page-aligned (a match/insert boundary), so the
        full-pool page bases derived via ``[::page_size]`` are identical to the unpadded slice."""
        end = div_ceil(req.cached_len, self.page_size) * self.page_size
        return self.page_table[req.table_idx, start:end]

    def _free_req_slots(self, req: Req, keep_live: bool = False) -> None:
        """Return a finished request's GDN pool slots: both ping-pong slots, plus the live slot
        unless it was donated to the tree. Idempotent -- clears the refs so a re-entry frees
        nothing (defense-in-depth against the abort/finish double-free, see _free_req_resources)."""
        slots = list(req.mamba_ping_pong) if req.mamba_ping_pong is not None else []
        if not keep_live and req.linear_slot_idx is not None:
            slots.append(req.linear_slot_idx)
        if slots:
            self.linear_state_pool.free(slots)
        req.mamba_ping_pong = None
        req.linear_slot_idx = None

    def check_integrity(self) -> None:
        if self.is_hybrid:
            pc = self.prefix_cache
            pc.check_integrity()  # structural: every snapshot node owns a slot, refs >= 0
            cache_pages = (pc.full_evictable + pc.full_protected) // self.page_size
            # GDN-slot conservation upper bound: free slots + tree-held snapshots can never
            # exceed the (non-padding) pool capacity; the remainder is held by running requests.
            pool = self.linear_state_pool
            tree_slots = pc.mamba_evictable_size + pc.mamba_protected
            assert pool.num_free_slots + tree_slots <= pool.num_slots - 1, (
                f"GDN-slot leak: free({pool.num_free_slots}) + tree({tree_slots}) > "
                f"capacity({pool.num_slots - 1})"
            )
        elif self.is_swa:
            pc = self.prefix_cache
            pc.check_integrity()  # full>=swa refs, tombstone => no swa lock
            cache_pages = (pc.full_evictable + pc.full_protected) // self.page_size
            # swa-slot conservation upper bound: free swa slots + tree-held live swa tokens can
            # never exceed the (non-sentinel) swa-pool capacity; the rest is held by running reqs.
            tree_swa = pc.swa_evictable + pc.swa_protected
            cap = self.swa_pool.swa_num_tokens - 1  # slot 0 is the reserved sentinel
            # check_integrity is idle-only (like the exact full-pool check below), so no request
            # holds a swa slot: free + tree must equal cap exactly. `==` (not `<=`) so a LEAK
            # (free + tree < cap) is caught, not just a double-free (> cap).
            assert self.swa_pool.swa_available_size() + tree_swa == cap, (
                f"SWA-slot leak/double-free: free({self.swa_pool.swa_available_size()}) + "
                f"tree({tree_swa}) != capacity({cap})"
            )
        else:
            self.prefix_cache.check_integrity()
            cache_pages = self.prefix_cache.size_info.total_size // self.page_size
        if len(self.free_slots) + cache_pages != self.num_pages:
            raise RuntimeError(
                "CacheManager integrity check failed:"
                f" free_pages({len(self.free_slots)}) +"
                f" cache_pages({cache_pages}) != num_pages({self.num_pages})"
            )
        if self.page_size > 1:
            assert torch.all(self.free_slots % self.page_size == 0)

    def rebuild(self, num_pages: int, page_table: torch.Tensor) -> None:
        """Re-point the page table and reset page accounting + prefix cache IN PLACE.

        Idle-only: assumes no request holds a live handle. Builds a brand-new prefix
        cache (RadixPrefixCache.reset() is an unimplemented stub) rather than mutating
        the old one.
        """
        device = page_table.device
        self.device = device
        self.num_pages = num_pages
        self.page_table = page_table
        self.free_slots = torch.arange(num_pages, dtype=torch.int32, device=device) * self.page_size
        self.prefix_cache = self._make_prefix_cache(device, self.page_size, self.cache_type)
        # The discarded hybrid tree owned donated GDN-snapshot slots; rebuild is idle-only, so
        # reclaim the whole LinearStatePool free-list (else those slots leak -> admission hangs).
        if self.is_hybrid:
            self.linear_state_pool.reclaim_all_slots()

    @contextmanager
    def lazy_free_region(self):
        def lazy_free(indices: torch.Tensor) -> None:
            # clone: callers pass page-table VIEWS, and the deferred concat below only reads them
            # when the region exits. A commit that re-points the row in between (the dedup
            # re-point, the SWA one) would otherwise rewrite the pending free list underneath us
            # and return the tree's canonical pages instead of the request's duplicates.
            lazy_free_list.append(indices[:: self.page_size].clone())

        lazy_free_list: List[torch.Tensor] = []
        try:
            self._free = lazy_free
            yield
        finally:
            del self._free
            self.free_slots = torch.cat([self.free_slots] + lazy_free_list)

    def _allocate(self, needed_pages: int) -> torch.Tensor:
        if needed_pages > (free_pages := len(self.free_slots)):
            need = (needed_pages - free_pages) * self.page_size
            if self.is_swa:
                # Evicting KV leaf nodes drops their swa slots too -> return both pools.
                ev = self.prefix_cache.evict_full(need)
                evicted = ev.kv_indices
                self._free_swa(ev.swa_indices)
            elif self.is_hybrid:
                # Evicting KV leaf nodes drops their GDN snapshots too -> return both pools.
                er = self.prefix_cache.evict_full(need, derive_paths=self._tier_on)
                evicted = er.kv_indices
                if self._tier_on:
                    self._tier_offer(er.victim_paths)   # offer BEFORE the free: slots stay live
                if er.mamba_slots:
                    self.linear_state_pool.free(er.mamba_slots)
            else:
                evicted = self.prefix_cache.evict(need)
            self.free_slots = torch.cat([self.free_slots, evicted[:: self.page_size]])
            assert len(self.free_slots) >= needed_pages, "Eviction did not free enough space."
        allocated = self.free_slots[:needed_pages]
        self.free_slots = self.free_slots[needed_pages:]
        return allocated

    def _free(self, indices: torch.Tensor) -> None:
        if len(indices) > 0:
            self.free_slots = torch.cat([self.free_slots, indices[:: self.page_size]])

    def _page_to_token(self, pages: torch.Tensor) -> torch.Tensor:
        if self.page_size == 1:
            return pages
        # [X * page_size] -> [X * page_size, ..., X * page_size + page_size - 1]
        offsets = torch.arange(self.page_size, device=self.device, dtype=torch.int32)
        return (pages.unsqueeze(1) + offsets).flatten()

    # ------------------------------------------------- session tiering (W2, additive)

    def shutdown_tier(self) -> int:
        """Graceful-shutdown hook: offer the LIVE tree's snapshot boundaries to the store
        (they die with the process otherwise; the adaptive keep-set filters), then flush
        live tier segments to L2 and compact the journal (dead records dropped; crash-safe
        convergence via replay's payload-crc check)."""
        if self.tier_store is None:
            return 0
        if self._tier_on and self.is_hybrid:
            self._tier_offer(self.prefix_cache.snapshot_victim_paths())
        flushed = self.tier_store.flush_live()
        self.tier_store.compact()
        return flushed

    def tier_owns_restore_slot(self, slot: int) -> bool:
        return slot in self._tier_restore_slots

    def tier_release_restore_slot(self, slot: int) -> None:
        self._tier_restore_slots.discard(slot)

    def _chain_keys(self, ids) -> list:
        """The request's per-page chain keys, the same incremental hash the store indexes."""
        from freetoken.kvcache.utils import chain_page_key

        ps = self.page_size
        prev, chain = None, []
        for off in range(0, len(ids) - len(ids) % ps, ps):
            prev = chain_page_key(prev, tuple(ids[off:off + ps].tolist()))
            chain.append(prev)
        return chain

    def _tier_admission(self, ids, cached_len: int) -> None:
        """note_match wiring: a shallow/cold match vs the session's stored tip is a divergence
        -- refresh the store's LRU currency at the matched path key and record the
        under-divergence boundary for the adaptive keep set. All boundary bookkeeping here is
        in TOKENS (cached_len and the victim boundary_len are token counts); page units are
        derived only at the offer/restore seams. Sessions group by their FIRST-page
        chain key (a common system prefix groups coarser: documented phase-1 approximation).
        A full match (the whole stripped prompt cached) still promotes the tip."""
        if not self._tier_on:
            return
        chain = self._chain_keys(ids)
        if not chain:          # prompt shorter than page_size: nothing to index or probe
            return
        st = self._tier_sessions.setdefault(chain[0], [0, 0, 0, None])
        if cached_len < st[0]:
            st[1] = cached_len                      # last known divergence depth (tokens)
        elif cached_len > st[0]:
            st[2], st[0] = st[0], cached_len        # the old tip becomes the newest non-tip
        if cached_len > 0:                          # a cold match only tracks the session
            self.tier_store.note_match(chain[cached_len // self.page_size - 1], cached_len)

    def _tier_offer(self, victims) -> None:
        """Keep/drop each evicted victim against the adaptive set {tip, the DEEPEST boundary
        <= divergence depth, newest non-tip} of its tracked session; kept victims are offered
        to the store (bytes materialized via the page codec / SnapshotSource closure) BEFORE
        the caller frees the originals -- demotion never changes what the core frees.
        Session bookkeeping (st = [tip, div, spare, under]) and victim boundary_len are in
        TOKENS; offer() takes PAGES (len(vp.chain_keys)) and _tier_snap_bound stores PAGES
        (what try_restore's probe depth is counted in)."""
        ps = self.page_size
        for vp in victims:
            st = self._tier_sessions.get(vp.chain_keys[0])
            if st is None:
                continue
            if vp.boundary_len > st[0]:
                st[2], st[0] = st[0], vp.boundary_len
            elif st[0] > vp.boundary_len > st[2]:
                st[2] = vp.boundary_len
            if vp.boundary_len <= st[1] and (st[3] is None or vp.boundary_len >= st[3]):
                st[3] = vp.boundary_len                     # deepest under-divergence slot
            elif vp.boundary_len == st[0] or vp.boundary_len == st[2]:
                pass                                        # tip / newest non-tip slot
            else:
                continue                                    # stale grid point: not kept
            kv_pages = [(key, self._page_bytes.read_page(int(vp.kv_indices[i * ps]) // ps))
                        for i, key in enumerate(vp.chain_keys)]
            snaps = (_SlotSnapshot(self.linear_state_pool, vp.mamba_slot)
                     if vp.mamba_slot is not None else None)
            if self.tier_store.offer(vp.path_key, len(vp.chain_keys), kv_pages, snaps) \
                    and snaps is not None:
                self._tier_snap_bound[vp.path_key] = len(vp.chain_keys)   # pages
            elif vp.mamba_slot is not None:
                self._tier_snap_bound.pop(vp.path_key, None)   # dropped offer: stale bound

    def try_restore(self, ids, cached_len: int) -> MatchResult | None:
        """Synchronous session-tier restore on a shallow/cold admission match, before the
        re-prefill fall-through: probe the store with the request's page-hash chain; on a hit
        memcpy the page bytes into free KV pages and the snapshot bytes into a fresh
        LinearStatePool slot and hand the normal insert()/cache_req path a prefetched match
        (mamba_value rides the existing COW machinery). Budget rule: the restore consumes the
        free pages + one slot the replaced re-prefill would (no extra over-admit; the padding
        sink is never touched -- slots come from the pool's own free list). Anything missing
        (free pages, slot, snapshot at the matched depth) falls back to the normal path."""
        if not self._tier_on or cached_len >= len(ids):
            return None
        chain = self._chain_keys(ids)
        hit = self.tier_store.probe(chain)
        if hit is None:
            return None
        depth, handle = hit
        if cached_len >= depth:
            # The tree match already reaches the store's boundary: the normal path serves at
            # least as well, and a restore here would only duplicate tree-owned pages and
            # replace a deeper match with a shallower one (HW attempt-3: 253-page leak ->
            # integrity kill on exactly this shape).
            return None
        ps = self.page_size
        # Never duplicate tree-owned KV: restore only the missing suffix [owned, depth). The
        # tree's own pages - live snapshots AND KV-only tombstones (incl. a prior restore's
        # finish adoption) - stay in place, pinned via the walked node; the store's boundary
        # snapshot always rides along (with an adopted KV-only span it is the only live
        # snapshot at depth, so a snapshot-only restore revives the boundary).
        owned_node, owned, owned_pages = self.prefix_cache.owned_prefix(ids[:depth * ps])
        fresh = depth - owned
        if (fresh > len(self.free_slots) or self.linear_state_pool.num_free_slots < 1
                or depth == 0):
            return None
        res = self.tier_store.restore(handle)
        if res is None:
            return None
        pages, snaps = res
        # Hybrid reuse needs the GDN state AT the matched boundary: a KV-only restore would
        # hand the continuation a prefix it cannot resume from (no checkpointed boundary).
        if (self._tier_snap_bound.get(handle.path_key) != depth or not snaps
                or len(pages) != depth):
            return None
        allocated, self.free_slots = self.free_slots[:fresh], self.free_slots[fresh:]
        slot = self.linear_state_pool.alloc(1)[0]
        try:
            for i, data in enumerate(pages[owned:]):
                self._page_bytes.write_page(int(allocated[i]) // ps, data)
            _unpack_slot(self.linear_state_pool, slot, snaps[0])
        except Exception:
            self.linear_state_pool.free(slot)
            self.free_slots = torch.cat([allocated, self.free_slots])
            return None
        from freetoken.kvcache.hybrid_radix_cache import HybridCacheHandle

        handle_h = HybridCacheHandle(
            depth * ps, owned_node,
            torch.cat([owned_pages, self._page_to_token(allocated)]), tier_restored=True)
        slot_owned = MatchResult(handle_h, mamba_value=slot)
        self._tier_restore_slots.add(slot)          # no tree owner: scheduler frees post-COW
        self._tier_pending_restore = (handle_h, allocated, slot)   # for abandon_restore
        return slot_owned

    def abandon_restore(self, handle) -> None:
        """Return a restored-but-refused admission's pages and slot to their pools. Idempotent:
        the pending entry is cleared, and a slot already released (admitted -> COW-consumed)
        is not freed twice. No-op for any handle that was not a restored match."""
        pending = self._tier_pending_restore
        if pending is None or pending[0] is not handle:
            return
        self._tier_pending_restore = None
        _, allocated, slot = pending
        if slot in self._tier_restore_slots:
            self.tier_release_restore_slot(slot)
            self.linear_state_pool.free(slot)
        self.free_slots = torch.cat([self.free_slots, allocated])


def _write_page_table(
    page_table: torch.Tensor,
    allocated: torch.Tensor,
    allocation_info: List[Tuple[int, int, int]],
    page_size: int,
) -> None:
    needed_tokens = len(allocated)
    # Pinned only when there is a device to copy to asynchronously; CPU-only runs (unit tests,
    # a CPU CI runner) would otherwise raise instead of just doing a plain host allocation.
    pin = torch.cuda.is_available()
    table_idx_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=pin)
    positions_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=pin)
    offset = 0
    for table_idx, first_page, last_page in allocation_info:
        first_pos, last_pos = first_page * page_size, last_page * page_size
        length = last_pos - first_pos
        table_idx_host[offset : offset + length].fill_(table_idx)
        torch.arange(first_pos, last_pos, out=positions_host[offset : offset + length])
        offset += length
    assert offset == needed_tokens, "Mismatch in allocated tokens and filled tokens."
    table_idxs = table_idx_host.to(page_table.device, non_blocking=True)
    offsets = positions_host.to(page_table.device, non_blocking=True)
    assert allocated.dtype == page_table.dtype, (
        f"allocated dtype {allocated.dtype} != page_table dtype {page_table.dtype}"
    )
    page_table[table_idxs, offsets] = allocated
