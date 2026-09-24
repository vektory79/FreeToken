from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple, TypeAlias

import torch
from freetoken.core import get_global_ctx
from freetoken.utils import align_down

from .base import BaseCacheHandle, BasePrefixCache, InsertResult, MatchResult, SizeInfo

if TYPE_CHECKING:
    from .hybrid_radix_cache import VictimPath

KEY_FN: TypeAlias = Callable[[torch.Tensor], Any]


class RadixTreeNode:
    counter: int = 0

    def __init__(self, key_fn: KEY_FN, tic: int | None = None) -> None:
        self.key_fn = key_fn
        self.children: Dict[Any, RadixTreeNode] = {}
        self._parent: RadixTreeNode | None = None
        self.ref_count: int = 0
        self.uuid = RadixTreeNode.counter
        RadixTreeNode.counter += 1
        self.timestamp = tic or time.monotonic_ns()

        # Secondary "currency" for hybrid models (HybridRadixCache): an optional GDN state
        # snapshot slot attached at this node's END boundary, with its own ref count. None
        # for plain KV radix. ``split_at`` leaves the new prefix node's ``mamba_value`` None
        # ("a snapshot cannot be split") -- the snapshot stays on the original (suffix) node,
        # whose end boundary is unchanged. Forward-compat seam for SWA.
        self.mamba_value: int | None = None
        self.mamba_ref_count: int = 0
        # Hybrid snapshot currency's own LRU stamp: evict_mamba orders by it and walks never
        # touch it; -1 = no snapshot anchored yet (candidates always carry a live stamp).
        self.snapshot_lru: int = -1

        # SWA second currency (SWARadixCache). Unlike the GDN snapshot above, SWA stores NO
        # separate slot: ``value`` (full-pool page indices) is canonical and the swa KV is
        # reached via the pool's full->swa mapping. ``swa_tombstone`` marks that the swa KV for
        # these tokens was freed (slid out of window) while the full KV survives; ``swa_ref_count``
        # is the trailing-window lock; ``swa_uuid`` is the window-boundary handle (lock release).
        # All default off -> the plain KV radix and the GDN hybrid radix are unaffected.
        self.swa_tombstone: bool = False
        self.swa_ref_count: int = 0
        self.swa_uuid: int | None = None

        # these fields should be updated later
        self._key: torch.Tensor
        self._value: torch.Tensor
        self._length: int

    def set_key_value(self, key: torch.Tensor, value: torch.Tensor) -> None:
        assert len(key) == len(value)
        self._key = key
        self._value = value
        self._length = len(key)

    def set_parent(self, parent: RadixTreeNode) -> None:
        self._parent = parent
        parent.children[self.key_fn(self._key)] = self

    @property
    def length(self) -> int:
        return self._length

    @property
    def parent(self) -> RadixTreeNode:
        assert self._parent is not None
        return self._parent

    @property
    def value(self) -> torch.Tensor:
        return self._value

    def is_root(self) -> bool:
        return self._parent is None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_match_len(self, input_ids: torch.Tensor) -> int:
        from freetoken.kernel import fast_compare_key

        # compare key and input_ids, find the first diff
        return fast_compare_key(self._key, input_ids)

    def split_at(self, pos: int) -> RadixTreeNode:
        assert 0 < pos < self.length
        parent = self.parent

        new_node = RadixTreeNode(self.key_fn, self.timestamp)
        new_node.set_key_value(self._key[:pos], self._value[:pos])
        new_node.set_parent(parent)
        new_node.ref_count = self.ref_count
        # SWA: a tombstone covers all the node's tokens, so both halves inherit it; both halves
        # stay within a locked window, so the swa lock copies; the window-boundary uuid migrates
        # to the root-side (prefix) node and is cleared on the suffix (matches sglang _split_node).
        new_node.swa_ref_count = self.swa_ref_count
        new_node.swa_tombstone = self.swa_tombstone
        new_node.swa_uuid = self.swa_uuid
        self.swa_uuid = None
        # defensive: snapshot_lru belongs to the snapshot, not the node -- the snapshot stays
        # on this same suffix object, so this copy only guards a future split that moves it.
        new_node.snapshot_lru = self.snapshot_lru

        self.set_key_value(self._key[pos:], self._value[pos:])
        self.set_parent(new_node)

        return new_node

    def __lt__(self, other: RadixTreeNode) -> bool:
        return self.timestamp < other.timestamp


@dataclass(frozen=True)
class RadixCacheHandle(BaseCacheHandle):
    node: RadixTreeNode

    def get_matched_indices(self) -> torch.Tensor:
        node = self.node
        value_list: List[torch.Tensor] = []
        while not node.is_root():
            value_list.append(node.value)
            node = node.parent
        value_list.reverse()
        return torch.cat(value_list)


@dataclass(frozen=True)
class RestoredRadixHandle(RadixCacheHandle):
    """Session-tier restore handle (KV-only managers): the restored prefix's page indices
    are EXPLICIT because the fresh suffix pages are not tree-owned until the commit's
    insert_prefix adopts them - get_matched_indices cannot walk the tree yet.
    ``tier_restored`` mirrors HybridCacheHandle; lock_handle accepts the subclass as-is."""

    kv_indices: torch.Tensor
    tier_restored: bool = False

    def get_matched_indices(self) -> torch.Tensor:
        return self.kv_indices


class RadixPrefixCache(BasePrefixCache):
    def __init__(self, device: torch.device, page_size: int | None = None):
        super().__init__()
        self.device = device
        # Explicit page_size beats the global ctx: a manager constructed with page_size!=ctx
        # (tests, future multi-granularity) must not silently split the two (the tree would
        # align by one granularity while the manager frees by the other -> orphaned pages).
        self.page_size = get_global_ctx().page_size if page_size is None else int(page_size)
        self.key_fn = _get_key_fn(self.page_size)
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        self.evictable_size = 0
        self.protected_size = 0
        self.root_node = RadixTreeNode(self.key_fn)
        self.root_node.ref_count = 1  # root is always protected

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        assert isinstance(handle, RadixCacheHandle)
        node = handle.node
        if unlock:
            while not node.is_root():
                node.ref_count -= 1
                assert node.ref_count >= 0
                if node.ref_count == 0:
                    self.evictable_size += node.length
                    self.protected_size -= node.length
                node = node.parent
        else:
            while not node.is_root():
                if node.ref_count == 0:
                    self.evictable_size -= node.length
                    self.protected_size += node.length
                node.ref_count += 1
                node = node.parent

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        node, prefix_len = self._tree_walk(input_ids)
        return MatchResult(RadixCacheHandle(prefix_len, node))

    def owned_prefix(self, input_ids: torch.Tensor, *,
                     stamp: bool = True) -> Tuple[RadixTreeNode, int, torch.Tensor]:
        """(node, prefix_len, pages): the frontier of tree-owned KV along input_ids plus the
        node ending at that frontier and its collected page indices. The session-tier
        restore (KV-only managers) restores only the missing suffix; the returned node pins
        the tree-owned prefix for the request. stamp=False is the idle prefetch walk: it
        must not refresh node LRU currency (speculative interest must not outrank real
        admissions); callers other than the prefetch path keep the default stamping.
        Mirror of HybridRadixCache.owned_prefix."""
        node, prefix_len = self._tree_walk(input_ids, stamp=stamp)
        pages = (RadixCacheHandle(prefix_len, node).get_matched_indices()
                 if not node.is_root() else self.empty_tensor)
        return node, prefix_len, pages

    def live_victim_paths(self) -> List["VictimPath"]:
        """VictimPath for every live boundary (a plain radix node carries no secondary
        currency, so every non-root node is one). The graceful-shutdown seam: the tier
        store re-offers them before the process dies; the adaptive keep-set filters."""
        return [self._derive_victim_path(n) for n in self._nodes()]

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        insert_len = align_down(len(input_ids), self.page_size)
        input_ids, indices = input_ids[:insert_len], indices[:insert_len]
        node, prefix_len = self._tree_walk(input_ids)
        if prefix_len != insert_len:  # NOTE: prefix_len < insert_len
            new_node = RadixTreeNode(self.key_fn)
            new_node.set_key_value(input_ids[prefix_len:], indices[prefix_len:].clone())
            new_node.set_parent(node)
            self.evictable_size += new_node.length
            node = new_node
        return InsertResult(prefix_len, RadixCacheHandle(insert_len, node))

    def evict(self, size: int) -> torch.Tensor:
        return self._evict(size, derive_paths=False)[0]

    def evict_paths(self, size: int) -> Tuple[torch.Tensor, List["VictimPath"]]:
        """Session-tier eviction seam (KV-only managers; mirrors
        HybridRadixCache.evict_full derive_paths): evict by LRU over unlocked leaves and
        derive each popped victim's chain-key path. evict() stays the no-detail fast path."""
        return self._evict(size, derive_paths=True)

    def _evict(self, size: int, derive_paths: bool) -> Tuple[torch.Tensor, List["VictimPath"]]:
        if size == 0:
            return self.empty_tensor, []
        assert (
            size <= self.evictable_size
        ), f"Cannot evict {size}, only {self.evictable_size} is evictable"

        leave_nodes = self._collect_leave_nodes_for_evict()
        heapq.heapify(leave_nodes)
        evicted_indices: List[torch.Tensor] = []
        evicted_size = 0
        victims: List["VictimPath"] = []

        while evicted_size < size:
            assert (
                leave_nodes
            ), f"Cannot evict enough cache, need {size}, only {evicted_size} evicted"
            node = heapq.heappop(leave_nodes)
            assert node.ref_count == 0 and node.is_leaf() and not node.is_root()
            evicted_size += node.length
            evicted_indices.append(node.value)
            self.evictable_size -= node.length
            if derive_paths:
                # victim bytes are still pool-live here; the caller offers BEFORE freeing
                victims.append(self._derive_victim_path(node))
            parent = node.parent
            del parent.children[self.key_fn(node._key)]
            # NOTE: root is always protected, so won't be evicted
            if parent.is_leaf() and parent.ref_count == 0:
                heapq.heappush(leave_nodes, parent)

        return torch.cat(evicted_indices), victims

    def _derive_victim_path(self, node: RadixTreeNode) -> "VictimPath":
        """One chain_page_key call per page walking root -> node; the node's FIRST-page key
        (the tree's dict key) is not enough - the store needs the full ancestor chain.
        Computed on eviction/shutdown only. mamba_slot is None: a plain radix node carries
        no snapshot currency."""
        from freetoken.kvcache.utils import chain_page_key
        from .hybrid_radix_cache import VictimPath

        keys, pages = [], []
        n = node
        while not n.is_root():
            keys.append(n._key)
            pages.append(n.value)
            n = n.parent
        keys.reverse()
        pages.reverse()
        ps = self.page_size
        prev = None
        chain: List[bytes] = []
        for k in keys:
            for off in range(0, len(k), ps):
                prev = chain_page_key(prev, tuple(k[off:off + ps].tolist()))
                chain.append(prev)
        return VictimPath(prev, sum(len(k) for k in keys), tuple(chain), torch.cat(pages), None)

    def _nodes(self) -> List[RadixTreeNode]:
        out, stack = [], [self.root_node]
        while stack:
            n = stack.pop()
            if not n.is_root():
                out.append(n)
            stack.extend(n.children.values())
        return out

    def reset(self) -> None:
        raise NotImplementedError("RadixManager.reset is not implemented")

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(
            evictable_size=self.evictable_size,
            protected_size=self.protected_size,
        )

    def check_integrity(self) -> None:
        pass

    def _collect_leave_nodes_for_evict(self) -> List[RadixTreeNode]:
        nodes: List[RadixTreeNode] = [self.root_node]
        leave_nodes: List[RadixTreeNode] = []

        while len(nodes) > 0:
            node = nodes.pop()
            if node.is_leaf():
                if node.ref_count == 0:
                    leave_nodes.append(node)
            else:
                for child in node.children.values():
                    nodes.append(child)

        return leave_nodes

    def _tree_walk(self, input_ids: torch.Tensor,
                   stamp: bool = True) -> Tuple[RadixTreeNode, int]:
        prefix_len = 0
        indice_len = len(input_ids)
        node = self.root_node
        tic = time.monotonic_ns() if stamp else 0

        while prefix_len < indice_len:
            child_node = node.children.get(self.key_fn(input_ids[prefix_len:]))
            if child_node is None:
                return node, prefix_len
            node = child_node  # walk to child node

            # NOTE: at least 1 page is matched, so match_len >= page_size
            match_len = node.get_match_len(input_ids[prefix_len:])
            match_len = align_down(match_len, self.page_size)
            prefix_len += match_len

            # need to split the node if not fully matched
            if match_len != node.length:
                node = node.split_at(match_len)
                if stamp:
                    node.timestamp = tic
                return node, prefix_len

            # update timestamp for accessed node
            if stamp:
                node.timestamp = tic

        return node, prefix_len


def _get_key_fn(page_size: int) -> KEY_FN:
    if page_size == 1:
        return lambda x: x[0].item()
    return lambda x: tuple(x[:page_size].tolist())
