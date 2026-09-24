"""Hybrid (full-attn KV + GDN linear-state) radix cache.

A SEPARATE class from ``RadixPrefixCache`` (Option C) that REUSES the shared ``RadixTreeNode``
and walk/split logic, so the production KV radix is untouched (zero risk to non-hybrid models).
It adds a second "currency": an optional GDN state snapshot (``node.mamba_value`` = a
LinearStatePool slot id) attached at chunk/page-aligned boundary nodes, with its own LRU
eviction. Mirrors sglang ``MambaRadixCache`` (donate-not-copy, dual eviction, internal-node
tombstone, ``full_ref >= mamba_ref``) on FreeToken's tree.

Currency seam: the secondary value + its eviction is the slot a future SWA component plugs
into. This class is pool-agnostic -- it stores/returns slot ids and KV page indices; the
caller (CacheManager / scheduler) does the actual LinearStatePool / KV-pool free.
"""
from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Tuple

import torch

from freetoken.utils import align_down

from .base import BaseCacheHandle
from .radix_cache import RadixTreeNode, _get_key_fn


@dataclass(frozen=True)
class HybridCacheHandle(BaseCacheHandle):
    """Lock handle for a matched hybrid prefix: the matched node (lock target) + the reusable
    KV page indices. ``cached_len`` is already truncated to the deepest live-snapshot boundary.
    Plugs into PrefillAdder (reads ``.cached_len`` / ``.get_matched_indices()``) like the plain
    RadixCacheHandle; the restore slot rides on ``MatchResult.mamba_value``. ``tier_restored``
    marks a session-tier restore handle: its row pages below ``cached_len`` include fresh
    (not yet tree-owned) pages the finish commit must adopt."""

    node: RadixTreeNode
    kv_indices: torch.Tensor
    tier_restored: bool = False

    def get_matched_indices(self) -> torch.Tensor:
        return self.kv_indices


class HybridMatch(NamedTuple):
    kv_indices: torch.Tensor      # reused KV page indices for [0:cached_len)
    cached_len: int               # truncated to the deepest LIVE-snapshot boundary
    mamba_value: Optional[int]    # GDN snapshot slot to restore from (None = cold start)
    node: RadixTreeNode           # the matched node (lock target)


class VictimPath(NamedTuple):
    """Per-victim path detail for the session-tier interception (W2 additive): the
    incremental chain-key path key, the end boundary length, the per-page chain keys,
    the FULL-path KV page indices [0:boundary_len) (ancestors included -- they stay in
    the tree, the caller only reads bytes through them) and the snapshot slot freed at
    this boundary (None for a KV-only victim). Derived on eviction only, and only when
    the caller asks (derive_paths=True); nothing is cached on the node."""

    path_key: bytes
    boundary_len: int
    chain_keys: Tuple[bytes, ...]
    kv_indices: torch.Tensor
    mamba_slot: Optional[int]


class EvictResult(NamedTuple):
    kv_indices: torch.Tensor      # KV page indices to free
    mamba_slots: List[int]        # GDN state slots to free
    victim_path_keys: tuple = ()  # per-victim path keys; empty unless derive_paths was requested
    victim_paths: tuple = ()      # per-victim VictimPath detail, aligned with victim_path_keys


class HybridRadixCache:
    def __init__(self, device: torch.device, page_size: int) -> None:
        from freetoken.kernel.fla.chunk import CHUNK_SIZE
        # Snapshots land on ×CHUNK_SIZE boundaries; require them to be page-aligned so the KV
        # node boundary and the GDN-state boundary coincide (page_size in {1,2,4,8,16,32,64}).
        assert CHUNK_SIZE % page_size == 0, (
            f"hybrid_radix needs CHUNK_SIZE({CHUNK_SIZE}) % page_size({page_size}) == 0"
        )
        self.device = device
        self.page_size = page_size
        self.key_fn = _get_key_fn(page_size)
        self.empty = torch.empty(0, dtype=torch.int32, device=device)
        self.root = RadixTreeNode(self.key_fn)
        self.root.set_key_value(self.empty, self.empty)
        self.root.ref_count = 1  # root is always protected
        self.full_evictable = 0
        self.full_protected = 0
        self.mamba_evictable = 0     # number of live, unlocked snapshots
        self.mamba_protected = 0

    # ---------------------------------------------------------------- match / insert
    def match_prefix(self, input_ids: torch.Tensor) -> HybridMatch:
        """Match the token prefix, then truncate the reusable length to the deepest node on
        the path that still owns a LIVE snapshot (a continuation can only resume the GDN
        recurrence from a checkpointed boundary)."""
        node, _ = self._walk(input_ids)
        # walk up to the deepest node whose END boundary has a live snapshot
        cur, end_len = node, self._path_len(node)
        while not cur.is_root():
            if cur.mamba_value is not None:
                # A use must out-age the walk's one shared tic: both the KV-LRU key
                # (timestamp) and the evict_mamba key (snapshot_lru) re-stamp here.
                cur.timestamp = time.monotonic_ns()
                cur.snapshot_lru = time.monotonic_ns()
                return HybridMatch(self._collect_kv(cur), end_len, cur.mamba_value, cur)
            end_len -= cur.length
            cur = cur.parent
        return HybridMatch(self.empty, 0, None, self.root)

    def owned_prefix(self, input_ids: torch.Tensor, *,
                     stamp: bool = True) -> Tuple[RadixTreeNode, int, torch.Tensor]:
        """(node, prefix_len, pages): the frontier of tree-owned KV along input_ids - live
        snapshots and KV-only tombstones ALIKE keep their pages - plus the node ending at
        that frontier and its collected page indices. The session-tier restore restores only
        the missing suffix; the returned node pins the tree-owned prefix for the request.
        stamp=False is the idle prefetch walk: it must not refresh node LRU currency
        (speculative interest must not outrank real admissions)."""
        node, prefix_len = self._walk(input_ids, stamp=stamp)
        return node, prefix_len, self._collect_kv(node)

    def snapshot_victim_paths(self) -> List[VictimPath]:
        """VictimPath for every LIVE snapshot-bearing node: the graceful-shutdown seam - the
        tier store re-offers them before the process dies, else a session whose tip stayed
        live in the tree loses it entirely and its post-reboot first turn falls back to a
        long re-prefill (hardware: B restored a stale 16k boundary at 37 s instead of its
        44k tip at ~4 s)."""
        return [self._derive_victim_path(n, n.mamba_value) for n in self._snapshot_nodes()]

    def insert(self, input_ids: torch.Tensor, kv_indices: torch.Tensor,
               mamba_value: Optional[int]) -> Tuple[int, bool]:
        """Insert the committed KV prefix and DONATE ``mamba_value`` at the (page-aligned) end
        boundary node. mamba_value=None inserts a KV-only node (no snapshot - the session-tier
        restore's finish adoption uses this). Returns (matched_prefix_len, mamba_exist). If the
        boundary node already owns a live snapshot, returns mamba_exist=True and does not attach
        (caller frees the donated slot -- dedup)."""
        insert_len = align_down(len(input_ids), self.page_size)
        input_ids, kv_indices = input_ids[:insert_len], kv_indices[:insert_len]
        node, prefix_len = self._walk(input_ids)
        if prefix_len != insert_len:
            new_node = RadixTreeNode(self.key_fn)
            new_node.set_key_value(input_ids[prefix_len:], kv_indices[prefix_len:].clone())
            new_node.set_parent(node)
            self.full_evictable += new_node.length
            node = new_node
        if node.is_root():
            return prefix_len, True   # root can't hold a snapshot; report exist so caller frees it
        if node.mamba_value is not None:
            # A dedup hit is the only proof the boundary is still a live reuse point: both
            # the KV-LRU key (timestamp) and the evict_mamba key (snapshot_lru) re-stamp here.
            node.timestamp = time.monotonic_ns()
            node.snapshot_lru = time.monotonic_ns()
            return prefix_len, True                 # dedup: caller frees its donated slot
        node.mamba_value = mamba_value              # fills a fresh node or a tombstone
        if mamba_value is not None:
            node.snapshot_lru = time.monotonic_ns()  # acquisition stamps the snapshot currency
            if node.mamba_ref_count == 0:
                self.mamba_evictable += 1
        return prefix_len, False

    # ---------------------------------------------------------------- locking (dual)
    def inc_lock(self, node: RadixTreeNode) -> None:
        """Protect a matched node's snapshot (mamba ref on the node) and its KV path
        (full ref node..root). Enforces full_ref >= mamba_ref: using a snapshot at N pins the
        whole root..N KV chain."""
        if node.mamba_value is not None:
            if node.mamba_ref_count == 0:
                self.mamba_evictable -= 1
                self.mamba_protected += 1
            node.mamba_ref_count += 1
        cur = node
        while not cur.is_root():
            if cur.ref_count == 0:
                self.full_evictable -= cur.length
                self.full_protected += cur.length
            cur.ref_count += 1
            cur = cur.parent

    def dec_lock(self, node: RadixTreeNode) -> None:
        if node.mamba_value is not None and node.mamba_ref_count > 0:
            node.mamba_ref_count -= 1
            if node.mamba_ref_count == 0:
                self.mamba_evictable += 1
                self.mamba_protected -= 1
        cur = node
        while not cur.is_root():
            cur.ref_count -= 1
            assert cur.ref_count >= 0
            if cur.ref_count == 0:
                self.full_evictable += cur.length
                self.full_protected -= cur.length
            cur = cur.parent

    # ---------------------------------------------------------------- eviction (dual)
    def evict_full(self, num_tokens: int, derive_paths: bool = False) -> EvictResult:
        """Evict KV tokens by LRU over UNLOCKED LEAF nodes (an internal node's KV is a prefix
        dependency for all descendants). Frees each evicted node's snapshot too.
        derive_paths=True additionally derives each popped leaf victim's chain-key path
        (session-tier demotion seam); cascaded KV-only tombstones are not victims of this
        policy and emit nothing."""
        leaves = [n for n in self._leaves() if n.ref_count == 0]
        heapq.heapify(leaves)
        kv, mamba, freed = [], [], 0
        victims: List[VictimPath] = []
        while freed < num_tokens and leaves:
            node = heapq.heappop(leaves)
            if node.ref_count != 0 or not node.is_leaf() or node.is_root():
                continue
            freed += node.length
            kv.append(node.value)
            n_slots = len(mamba)
            self.full_evictable -= node.length
            self._free_node_mamba(node, mamba)
            if derive_paths:
                victims.append(self._derive_victim_path(
                    node, mamba[n_slots] if len(mamba) > n_slots else None))
            parent, casc = self._cascade_tombstone_leaves(self._unlink(node), kv)
            freed += casc
            if parent.is_leaf() and parent.ref_count == 0 and not parent.is_root():
                heapq.heappush(leaves, parent)
        return EvictResult(torch.cat(kv) if kv else self.empty, mamba,
                           tuple(v.path_key for v in victims), tuple(victims))

    def evict_mamba(self, num: int, derive_paths: bool = False) -> EvictResult:
        """Evict GDN snapshots by LRU over UNLOCKED snapshot-bearing nodes -- internal nodes
        too. Internal node -> TOMBSTONE (free the slot, keep KV + children). Leaf node -> free
        both KV and slot and unlink, then cascade-delete any KV-only tombstone leaves it exposes
        upward (so a leaf always carries a live snapshot -- mirrors sglang). derive_paths=True
        additionally derives each victim's chain-key path (both the leaf and the internal
        tombstone victim emit theirs -- the tombstone victim's boundary is a live reuse point
        for the session-tier demotion seam)."""
        # Victim heap keys on (snapshot_lru, timestamp): FIFO by last validation with the
        # walk tic as tiebreak. __lt__ (timestamp) stays the KV-LRU key for evict_full.
        cands = [((n.snapshot_lru, n.timestamp), n) for n in self._snapshot_nodes()
                 if n.mamba_ref_count == 0]
        heapq.heapify(cands)
        kv, mamba, freed = [], [], 0
        victims: List[VictimPath] = []
        while freed < num and cands:
            node = heapq.heappop(cands)[1]
            if node.mamba_value is None or node.mamba_ref_count != 0 or node.is_root():
                continue
            if node.is_leaf() and node.ref_count == 0:
                kv.append(node.value)
                self.full_evictable -= node.length
                self._free_node_mamba(node, mamba)
                freed += 1
                if derive_paths:
                    victims.append(self._derive_victim_path(node, mamba[-1]))
                self._cascade_tombstone_leaves(self._unlink(node), kv)
            else:
                slot = node.mamba_value
                self._free_node_mamba(node, mamba)  # tombstone internal (or locked-KV) node
                freed += 1
                if derive_paths:
                    victims.append(self._derive_victim_path(node, slot))
        return EvictResult(torch.cat(kv) if kv else self.empty, mamba,
                           tuple(v.path_key for v in victims), tuple(victims))

    @property
    def full_evictable_size(self) -> int:
        return self.full_evictable

    @property
    def mamba_evictable_size(self) -> int:
        return self.mamba_evictable

    @property
    def size_info(self):
        """KV-page currency, for code that reads a BasePrefixCache size_info (metrics/usage).
        The GDN-snapshot currency is reported via mamba_evictable_size."""
        from .base import SizeInfo
        return SizeInfo(evictable_size=self.full_evictable, protected_size=self.full_protected)

    def check_integrity(self) -> None:
        # Structural: every snapshot-bearing node holds a real slot id; ref counts non-negative.
        # (KV/page conservation is checked by CacheManager.check_integrity.)
        for n in self._snapshot_nodes():
            assert n.mamba_value is not None and n.mamba_ref_count >= 0 and n.ref_count >= 0
            # a forgotten attach stamp would make the node the always-evicted-first victim
            assert n.snapshot_lru >= 0

    # ---------------------------------------------------------------- helpers
    def _derive_victim_path(self, node: RadixTreeNode, mamba_slot: Optional[int]) -> VictimPath:
        """One chain_page_key call per page walking root -> node; the node's FIRST-page key
        (the tree's dict key) is not enough -- the store needs the full ancestor chain. Not
        cached: computed on eviction only, only when a consumer asked."""
        from freetoken.kvcache.utils import chain_page_key

        keys, pages = [], []
        n = node
        while not n.is_root():
            keys.append(n._key)
            pages.append(n.value)
            n = n.parent
        keys.reverse()
        pages.reverse()
        ps = self.page_size
        prev: Optional[bytes] = None
        chain: List[bytes] = []
        for k in keys:
            for off in range(0, len(k), ps):
                prev = chain_page_key(prev, tuple(k[off:off + ps].tolist()))
                chain.append(prev)
        return VictimPath(prev, sum(len(k) for k in keys), tuple(chain), torch.cat(pages), mamba_slot)

    def _free_node_mamba(self, node: RadixTreeNode, out: List[int]) -> None:
        if node.mamba_value is not None:
            out.append(node.mamba_value)
            node.mamba_value = None
            if node.mamba_ref_count == 0:
                self.mamba_evictable -= 1

    def _unlink(self, node: RadixTreeNode) -> RadixTreeNode:
        parent = node.parent
        del parent.children[self.key_fn(node._key)]
        return parent

    def _cascade_tombstone_leaves(self, parent: RadixTreeNode, kv_out: List[torch.Tensor]):
        """After a leaf is unlinked, eagerly reclaim the KV-only tombstone leaves it exposes
        upward (mamba_value None, no children, unlocked): free their KV and unlink, walking up.
        Keeps the 'a leaf always carries a live snapshot' invariant (sglang
        _iteratively_delete_tombstone_leaf). Returns (highest surviving ancestor, freed_tokens)."""
        freed = 0
        while (parent.mamba_value is None and parent.is_leaf()
               and parent.ref_count == 0 and not parent.is_root()):
            kv_out.append(parent.value)
            self.full_evictable -= parent.length
            freed += parent.length
            parent = self._unlink(parent)
        return parent, freed

    def _path_len(self, node: RadixTreeNode) -> int:
        n, total = node, 0
        while not n.is_root():
            total += n.length
            n = n.parent
        return total

    def _collect_kv(self, node: RadixTreeNode) -> torch.Tensor:
        vals: List[torch.Tensor] = []
        n = node
        while not n.is_root():
            vals.append(n.value)
            n = n.parent
        vals.reverse()
        return torch.cat(vals) if vals else self.empty

    def _leaves(self) -> List[RadixTreeNode]:
        out, stack = [], [self.root]
        while stack:
            n = stack.pop()
            if n.is_leaf():
                if not n.is_root():
                    out.append(n)
            else:
                stack.extend(n.children.values())
        return out

    def _snapshot_nodes(self) -> List[RadixTreeNode]:
        out, stack = [], [self.root]
        while stack:
            n = stack.pop()
            if n.mamba_value is not None and not n.is_root():
                out.append(n)
            stack.extend(n.children.values())
        return out

    def _walk(self, input_ids: torch.Tensor, stamp: bool = True) -> Tuple[RadixTreeNode, int]:
        prefix_len, total = 0, len(input_ids)
        node = self.root
        tic = time.monotonic_ns() if stamp else 0
        while prefix_len < total:
            child = node.children.get(self.key_fn(input_ids[prefix_len:]))
            if child is None:
                return node, prefix_len
            node = child
            match_len = align_down(node.get_match_len(input_ids[prefix_len:]), self.page_size)
            prefix_len += match_len
            if match_len != node.length:
                node = node.split_at(match_len)
                if stamp:
                    node.timestamp = tic
                return node, prefix_len
            if stamp:
                node.timestamp = tic
        return node, prefix_len


__all__ = ["HybridRadixCache", "HybridMatch", "EvictResult", "VictimPath", "HybridCacheHandle"]
