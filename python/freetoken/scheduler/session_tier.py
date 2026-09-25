"""SessionTierStore: tiered cache of evicted session segments (RAM pool -> SSD blobs).

A segment is one session path [0:N] of KV pages plus up to 3 GDN snapshot byte blobs.
Pages are content-addressed by chain page keys so shared subagent prefixes dedup and
appends are incremental. L1 is one mlock'ed contiguous allocation; L2 is append-only
blob files guarded by a fsync-ordered journal (blob + fsync BEFORE journal record), so
a SIGKILL mid-append can only truncate the tail, never corrupt accepted records.
"""
from __future__ import annotations

import ctypes
import json
import mmap
import os
import resource
import struct
import threading
import time
import zlib
from collections import Counter
from dataclasses import dataclass, field
from queue import SimpleQueue
from typing import Protocol, Sequence

from freetoken.kvcache.utils import chain_page_key  # noqa: F401  (re-exported)
from freetoken.utils.logger import init_logger

logger = init_logger(__name__)

_BLK = 4096
_MAX_SNAPSHOTS = 3
_DIGEST = 16
_JOURNAL_NAME = "journal.log"
_BLOB_NAME = "blob.bin"
# Scheduled compaction (runtime, no CLI flag): rewrite the blob once dead bytes exceed
# this share of the blob AND the blob is above the floor (no churn on small installs).
_COMPACT_DEAD_FRACTION = 0.25
_COMPACT_FLOOR_BYTES = 256 << 20
# Speculative restore prefetch (phase 2, no CLI flag): at most this many concurrent
# staging tickets, each holding one mlocked host buffer with the segment bytes until
# adopt/abandon. The scheduler's idle hook (via the cache manager) is the only caller.
_MAX_RESTORE_TICKETS = 2
# Guards the cumulative mlock accounting (_Pool._locked_total): it is mutated from the
# scheduler thread (pool construction, staging alloc, free_block) AND the reader thread
# (staging close at ticket finalize).
_LOCKED_TOTAL_LOCK = threading.Lock()


class SnapshotSource(Protocol):
    """Snapshot handle abstraction: W2 wires LinearStatePool slot -> bytes behind this."""

    def payload(self) -> bytes: ...


@dataclass(frozen=True)
class TierHandle:
    """Opaque restore token handed out by probe()."""

    path_key: bytes
    depth: int
    _seg_id: int


@dataclass
class _Page:
    key: bytes
    nbytes: int
    l1_off: int
    refs: int = 1  # number of L1 segments referencing this deduped page


@dataclass
class _Segment:
    seg_id: int
    path_key: bytes
    boundary_len: int
    page_keys: list[bytes]          # chain keys; page_keys[i] covers pages [0:i+1]
    page_lens: list[int]
    snap_lens: list[int]
    # L1 residency: per-page / per-snapshot pool offsets (-1 = empty).
    l1_page_offs: list[int] = field(default_factory=list)
    l1_snap_offs: list[int] = field(default_factory=list)
    # L2 residency: blob record range (unpadded length).
    blob: str = ""
    l2_off: int = -1
    l2_n: int = 0
    refs: int = 0
    last_validation: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def in_l1(self) -> bool:
        return bool(self.l1_page_offs) or bool(self.l1_snap_offs)

    @property
    def in_l2(self) -> bool:
        return self.l2_off >= 0


class _Pool:
    """First-fit free-list allocator over one anonymous mmap region, mlock'ed at start
    (LOCKED-style: grows the RLIMIT_MEMLOCK soft limit first; not CUDA born-pinned)."""

    _locked_total = 0  # bytes locked so far; the OS lock ceiling is a per-process quota

    def __init__(self, nbytes: int):
        self.size = nbytes
        self.mm = mmap.mmap(-1, nbytes)
        addr = ctypes.addressof(ctypes.c_char.from_buffer(self.mm))
        self._os_lock(addr, nbytes)
        self.free: list[list[int]] = [[0, nbytes]]  # [offset, size], sorted by offset

    @staticmethod
    def _os_lock(addr: int, nbytes: int) -> None:
        with _LOCKED_TOTAL_LOCK:
            want = _Pool._locked_total + nbytes + (256 << 20)
            soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
            if soft != resource.RLIM_INFINITY and soft < want:
                new_soft = want if hard == resource.RLIM_INFINITY else min(want, hard)
                if new_soft > soft:
                    try:
                        resource.setrlimit(resource.RLIMIT_MEMLOCK, (new_soft, hard))
                    except (OSError, ValueError):
                        pass  # keep the old limit; mlock below reports the real ceiling
            libc = ctypes.CDLL(None, use_errno=True)
            if libc.mlock(ctypes.c_void_p(addr), ctypes.c_size_t(nbytes)):
                err = ctypes.get_errno()
                raise OSError(err, f"mlock({nbytes / 2**30:.1f} GiB): {os.strerror(err)}")
            _Pool._locked_total += nbytes

    def alloc(self, nbytes: int) -> int:
        for i, (off, size) in enumerate(self.free):
            if size >= nbytes:
                self.free[i] = [off + nbytes, size - nbytes]
                if self.free[i][1] == 0:
                    del self.free[i]
                return off
        raise MemoryError(f"L1 pool exhausted: no free run for {nbytes} bytes")

    def free_block(self, off: int, nbytes: int) -> None:
        # The region stays mlock'ed for the pool's lifetime; this only unwinds the
        # cumulative locked-growth accounting future _Pool constructions quota against.
        with _LOCKED_TOTAL_LOCK:
            _Pool._locked_total -= nbytes
        spans = sorted(self.free + [[off, nbytes]])
        out: list[list[int]] = []
        for off2, size2 in spans:
            if out and out[-1][0] + out[-1][1] == off2:
                out[-1][1] += size2
            else:
                out.append([off2, size2])
        self.free = out

    def write(self, off: int, data: bytes) -> None:
        self.mm[off:off + len(data)] = data

    def read(self, off: int, nbytes: int) -> bytes:
        return bytes(self.mm[off:off + nbytes])


class _LockedBuffer:
    """One mlocked anonymous staging region (LOCKED-style: the L1 pool's _os_lock quota
    bookkeeping, grown the same way). Construct and close OFF the reader thread - the
    quota accounting in _os_lock is not thread-safe (host_banks precedent)."""

    def __init__(self, nbytes: int):
        self.size = max(1, nbytes)
        self.mm = mmap.mmap(-1, self.size)
        try:
            _Pool._os_lock(ctypes.addressof(ctypes.c_char.from_buffer(self.mm)), self.size)
        except BaseException:
            self.mm.close()
            raise
        self.mv = memoryview(self.mm)

    def close(self) -> None:
        try:
            self.mv.release()
        except BufferError:
            pass
        try:
            ctypes.CDLL(None, use_errno=True).munlock(
                ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(self.mm))),
                ctypes.c_size_t(self.size))
        except OSError:
            pass                     # the munmap below releases the OS lock regardless
        with _LOCKED_TOTAL_LOCK:
            _Pool._locked_total -= self.size
        self.mm.close()


@dataclass
class RestoreTicket:
    """One staged segment read (prefetch): the reader thread fills an mlocked host buffer
    with the segment's pages + snapshots; adopt consumes byte-identical slices, abandon
    frees the staging untouched. Every state transition happens under the store's global
    lock (pending -> ready | failed | abandoned); ready is the consumer's handoff signal."""

    handle: TierHandle
    seg_id: int
    boundary: int                    # staged page count: any adopt depth <= boundary fits
    ready: threading.Event
    state: str = "pending"
    error: str | None = None
    _finalized: bool = False
    _buf: _LockedBuffer | None = None
    _mv: memoryview | None = None
    _page_spans: list = field(default_factory=list)   # [(staging off, nbytes)] per page
    _snap_spans: list = field(default_factory=list)


class SessionTierStore:
    """Off (inert) when the ram pool is disabled and dir is None; never allocates."""

    def __init__(self, cfg):
        self.ram_bytes = int(getattr(cfg, "ram_bytes", 0) or 0)
        self.dir = getattr(cfg, "dir", None)
        self.ssd_bytes = int(getattr(cfg, "ssd_bytes", 0) or 0)
        # Off (inert) when ram is disabled AND dir is unset; a dir alone enables L2-only
        # mode (ssd_bytes 0 = unlimited cap - the GiB flag is a cap, not a gate).
        self.enabled = self.ram_bytes > 0 or self.dir is not None
        # Counter + dead-byte ledger exist even when inert: the offer() wrapper stamps
        # rejection counters on the disabled store before any early return. Pre-seeded so
        # snapshot()/stats_line() always see every key.
        self._dead_bytes = 0
        self._counters: Counter = Counter(offers_ok=0, offers_rej=0, offers_dedup=0,
                                          probe_hit=0,
                                          probe_miss=0, note_match=0, restore_l1=0,
                                          restore_l2=0, evictions=0, demotions=0,
                                          discards=0, prefetch_begin=0, prefetch_adopt=0,
                                          prefetch_abandon=0, prefetch_fail=0,
                                          prefetch_rej_cap=0, prefetch_probe_hit=0,
                                          prefetch_probe_miss=0)
        # Lock exists even when inert: the offer() wrapper takes it around the rejection
        # counter before any early return.
        self._lock = threading.RLock()
        # Prefetch staging (phase 2): present even when inert (every hook early-returns).
        self._tickets: dict[int, RestoreTicket] = {}
        self._work: SimpleQueue = SimpleQueue()
        self._worker: threading.Thread | None = None
        self._inflight = 0
        if not self.enabled:
            return
        if self.ram_bytes > 0:
            try:
                self._pool = _Pool(self.ram_bytes)
            except OSError as e:
                raise RuntimeError(
                    f"session tier: cannot mlock the {self.ram_bytes / 2**30:.2f} GiB L1 pool "
                    f"({e}); reduce --session-tier-ram-gib") from e
        else:
            self._pool = None
        self._segments: dict[int, _Segment] = {}
        self._index: dict[bytes, list[tuple[int, int]]] = {}   # page key -> [(seg, depth)]
        self._pages: dict[bytes, _Page] = {}                   # L1-deduped pages
        self._next_seg = 1
        self._l1_used = 0
        self._ssd_used = 0
        # Append position in the blob (O_APPEND writes at real EOF); never decreased.
        # _ssd_used is capacity accounting only and shrinks on L2 discard.
        self._blob_eof = 0
        self._blob_path = ""
        self._journal_path = ""
        self._blob_fd = -1
        self._journal_fd = -1
        # Blob bytes freed by L2 discards (regions read as zero holes) until compaction
        # reclaims them; boot replay recomputes this from the recovered record set.
        if self.dir is not None:
            os.makedirs(self.dir, exist_ok=True)
            self._blob_path = os.path.join(self.dir, _BLOB_NAME)
            self._journal_path = os.path.join(self.dir, _JOURNAL_NAME)
            self._blob_fd = os.open(self._blob_path,
                                    os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            self._journal_fd = os.open(self._journal_path,
                                       os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            self._ssd_used = os.path.getsize(self._blob_path)
            self._blob_eof = self._ssd_used
            logger.info("session tier on: L1=%.2f GiB, L2 dir=%s cap=%.2f GiB used=%.2f GiB",
                        self.ram_bytes / 2**30, self.dir, self.ssd_bytes / 2**30,
                        self._ssd_used / 2**30)

    # ------------------------------------------------------------------ probe

    def probe(self, page_hashes: Sequence[bytes], *,
              stamp: bool = True) -> tuple[int, TierHandle] | None:
        """Deepest match: walk the prompt's chain keys back to front; an equal key at
        depth d implies an equal prefix [0:d]. Ties go to the most recently validated.
        stamp=False is the idle prefetch probe: no LRU refresh (speculative interest must
        not outrank real matches) and its own counters."""
        if not self.enabled:
            return None
        now = time.monotonic_ns()
        hit_key = "probe_hit" if stamp else "prefetch_probe_hit"
        miss_key = "probe_miss" if stamp else "prefetch_probe_miss"
        with self._lock:
            best = None
            for d in range(len(page_hashes), 0, -1):
                for seg_id, key_depth in self._index.get(page_hashes[d - 1], []):
                    if key_depth != d:
                        continue
                    seg = self._segments.get(seg_id)
                    if seg is None or seg.boundary_len < d:
                        continue
                    if best is None or (seg.boundary_len == d, seg.last_validation) > \
                            (best[0].boundary_len == d, best[0].last_validation):
                        best = (seg, d)
                if best is not None:
                    seg, d = best
                    if stamp:
                        seg.last_validation = now
                    self._counters[hit_key] += 1
                    return d, TierHandle(seg.path_key, d, seg.seg_id)
            self._counters[miss_key] += 1
            return None

    def note_match(self, path_key: bytes, depth: int) -> None:
        """Stamp the segment owning path_key with a fresh validation time (same currency
        as the VRAM snapshot_lru refresh)."""
        if not self.enabled:
            return
        now = time.monotonic_ns()
        with self._lock:
            seg = self._by_path(path_key)
            if seg is not None:
                seg.last_validation = now
                self._counters["note_match"] += 1
                _ = depth

    # ------------------------------------------------------------------ offer

    def offer(self, path_key: bytes, boundary_len: int,
              kv_pages: Sequence[tuple[bytes, bytes]],
              snapshot_slot: bytes | SnapshotSource | Sequence[bytes | SnapshotSource] | None = None
              ) -> bool:
        # RLock is reentrant: _offer re-acquires it; the counter increment stays inside
        # the locked region (Counter += is not atomic and offer may run concurrently).
        with self._lock:
            ok = self._offer(path_key, boundary_len, kv_pages, snapshot_slot)
            self._counters["offers_ok" if ok else "offers_rej"] += 1
        if ok:
            self.maybe_compact()   # watermark gate: two int compares when below
        return ok

    def _offer(self, path_key: bytes, boundary_len: int,
               kv_pages: Sequence[tuple[bytes, bytes]],
               snapshot_slot: bytes | SnapshotSource | Sequence[bytes | SnapshotSource] | None = None
               ) -> bool:
        """Store a segment. kv_pages are (chain_key, raw_bytes) pairs; snapshot_slot is
        raw snapshot bytes, a SnapshotSource handle, or a list of up to 3 of those.
        Returns False if the segment did not fit in either tier."""
        if not self.enabled:
            return False
        if boundary_len != len(kv_pages):
            raise ValueError(f"boundary_len {boundary_len} != {len(kv_pages)} pages")
        snaps = self._resolve_snaps(snapshot_slot)
        now = time.monotonic_ns()
        page_keys = [k for k, _ in kv_pages]
        with self._lock:
            existing = self._by_path(path_key)
            if existing is not None and existing.boundary_len == boundary_len:
                try:
                    self._attach_snapshots(existing, snaps, now)
                    return True
                except MemoryError:
                    if self._pool is None or not self._demote_one_lru():
                        logger.warning("session tier: L1 full; offer dropped")
                        return False
                    try:
                        self._attach_snapshots(existing, snaps, now)
                        return True
                    except MemoryError:
                        logger.warning("session tier: L1 full; offer dropped")
                        return False
            # Same-chain dedup via the page_keys prefix relation (path_key is the TIP key,
            # so boundaries of one session never share it): a live deeper seg serves every
            # restore a snapshot-free offer could (restore depth-truncates; the hybrid
            # _restore_tail gate needs a snapshot at the exact bound) -> skip the store.
            relatives = self._same_path_segs(page_keys)
            if not snaps and any(self._is_prefix(page_keys, s.page_keys) for s in relatives):
                self._counters["offers_dedup"] += 1
                return True
            # Supersede contained shallower snapshot-free segs (refs guard, snapshots are
            # never discarded) BEFORE staging so their freed L1/L2 room counts toward
            # this offer; if staging then fails the chain drops capability until a
            # re-offer (pre-diff code kept the shallower seg) - accepted regression.
            for seg in relatives:
                if (seg.boundary_len < boundary_len and seg.refs == 0
                        and not any(seg.snap_lens)
                        and self._is_prefix(seg.page_keys, page_keys)):
                    self._supersede(seg)
            need = sum(len(data) for _, data in kv_pages) + sum(len(s) for s in snaps)
            if self._pool is not None:
                while self.ram_bytes - self._l1_used < need:
                    if not self._demote_one_lru():
                        logger.warning("session tier: L1 full, nothing evictable; offer dropped")
                        return False
                try:
                    seg = self._store_l1(path_key, boundary_len, kv_pages, snaps, now)
                except MemoryError:
                    # Byte check passed but first-fit contiguity can still fail: one LRU
                    # demote frees a whole segment, then retry once.
                    if not self._demote_one_lru():
                        logger.warning("session tier: L1 fragmented/full; offer dropped")
                        return False
                    try:
                        seg = self._store_l1(path_key, boundary_len, kv_pages, snaps, now)
                    except MemoryError:
                        logger.warning("session tier: L1 fragmented/full; offer dropped")
                        return False
            else:
                seg = _Segment(self._next_seg, path_key, boundary_len,
                               [k for k, _ in kv_pages],
                               [len(d) for _, d in kv_pages], [len(s) for s in snaps])
                self._next_seg += 1
                payload = b"".join(d for _, d in kv_pages) + b"".join(snaps)
                if not self._write_blob(seg, payload):
                    logger.warning("session tier: L2 cap reached; offer dropped")
                    return False
                self._segments[seg.seg_id] = seg
                self._touch_index(seg)
                seg.last_validation = now
            return True

    def _resolve_snaps(self, snapshot_slot) -> list[bytes]:
        if snapshot_slot is None:
            return []
        items = list(snapshot_slot) if isinstance(snapshot_slot, (list, tuple)) else [snapshot_slot]
        if len(items) > _MAX_SNAPSHOTS:
            raise ValueError(f"at most {_MAX_SNAPSHOTS} snapshots per segment, got {len(items)}")
        return [it if isinstance(it, bytes) else it.payload() for it in items]

    def _by_path(self, path_key: bytes) -> _Segment | None:
        for seg in self._segments.values():
            if seg.path_key == path_key:
                return seg
        return None

    @staticmethod
    def _is_prefix(prefix: Sequence[bytes], chain: Sequence[bytes]) -> bool:
        # Equal chain key at depth d implies an equal page prefix (probe's store
        # invariant): the prefix relation proves containment, boundary_len alone cannot.
        return len(prefix) <= len(chain) and list(prefix) == list(chain[:len(prefix)])

    def _same_path_segs(self, page_keys: list[bytes]) -> list[_Segment]:
        """Live segments on the offer's session path: chain-key prefix relation in either
        direction (path_key is the boundary tip key, so successive boundaries of one
        session never share it - only the prefix proves same-path containment)."""
        return [s for s in self._segments.values()
                if (s.in_l1 or s.in_l2)
                and (self._is_prefix(page_keys, s.page_keys)
                     or self._is_prefix(s.page_keys, page_keys))]

    def _store_l1(self, path_key, boundary_len, kv_pages, snaps, now) -> _Segment:
        seg = _Segment(self._next_seg, path_key, boundary_len, [k for k, _ in kv_pages],
                       [len(d) for _, d in kv_pages], [len(s) for s in snaps])
        self._next_seg += 1
        try:
            for key, data in kv_pages:
                seg.l1_page_offs.append(self._page_slot(key, data))
            for snap in snaps:
                if snap:
                    off = self._pool.alloc(len(snap))
                    self._pool.write(off, snap)
                    self._l1_used += len(snap)
                    seg.l1_snap_offs.append(off)
                else:
                    seg.l1_snap_offs.append(-1)
        except MemoryError:
            self._release_l1(seg)   # unwind partial allocations; seg was never registered
            raise
        self._segments[seg.seg_id] = seg
        self._touch_index(seg)
        seg.last_validation = now
        return seg

    def _page_slot(self, key: bytes, data: bytes) -> int:
        """Dedup: an L1 page is allocated once and refcounted by referencing segments."""
        page = self._pages.get(key)
        if page is not None:
            page.refs += 1
            return page.l1_off
        off = self._pool.alloc(len(data))
        self._pool.write(off, data)
        self._l1_used += len(data)
        self._pages[key] = _Page(key, len(data), off)
        return off

    def _attach_snapshots(self, seg: _Segment, snaps: list[bytes], now: int) -> None:
        seg.last_validation = now
        if not snaps or not seg.in_l1:
            return
        # Allocate every replacement FIRST: a mid-way MemoryError must not leave the
        # segment half-attached (its old offsets already freed).
        new: list[tuple[int, int]] = []
        try:
            for snap in snaps:
                if snap:
                    off = self._pool.alloc(len(snap))
                    self._pool.write(off, snap)
                    self._l1_used += len(snap)
                    new.append((off, len(snap)))
                else:
                    new.append((-1, 0))
        except MemoryError:
            for off, ln in new:
                if off >= 0:
                    self._pool.free_block(off, ln)
                    self._l1_used -= ln
            raise
        for off, ln in zip(seg.l1_snap_offs, seg.snap_lens):
            if off >= 0:
                self._pool.free_block(off, ln)
                self._l1_used -= ln
        seg.snap_lens = [ln for _, ln in new]
        seg.l1_snap_offs = [off for off, _ in new]

    def _touch_index(self, seg: _Segment) -> None:
        for i, key in enumerate(seg.page_keys):
            self._index.setdefault(key, []).append((seg.seg_id, i + 1))

    # --------------------------------------------------------------- eviction

    def _demote_one_lru(self) -> bool:
        """LRU (by last validation) L1 segment with refs==0 goes to L2."""
        victims = [s for s in self._segments.values() if s.in_l1 and s.refs == 0]
        if not victims:
            return False
        return self._demote(min(victims, key=lambda s: (s.last_validation, s.seg_id)))

    def _demote(self, seg: _Segment) -> bool:
        if self._blob_fd < 0:
            return False
        payload = bytearray()
        for i, ln in enumerate(seg.page_lens):
            payload += self._pool.read(seg.l1_page_offs[i], ln)
        for i, ln in enumerate(seg.snap_lens):
            if ln:
                payload += self._pool.read(seg.l1_snap_offs[i], ln)
        if not self._write_blob(seg, bytes(payload)):
            logger.warning("session tier: L2 cap reached, cannot demote seg %d", seg.seg_id)
            return False
        self._release_l1(seg)
        self._counters["demotions"] += 1
        return True

    def _write_blob(self, seg: _Segment, payload: bytes | None) -> bool:
        """Append one blob record (padded to _BLK for the O_DIRECT reader), fsync, then
        the journal record - the fsync order is what makes replay crash-safe. Every
        failure path returns False: no journal entry, no capacity accounting, and the
        real O_APPEND EOF is re-synced so the next record's offset stays true."""
        need = sum(seg.page_lens) + sum(seg.snap_lens)
        padded = (need + _BLK - 1) // _BLK * _BLK
        if self.ssd_bytes and self._ssd_used + padded > self.ssd_bytes:
            self._evict_oldest_l2(need)
        if self.ssd_bytes and self._ssd_used + padded > self.ssd_bytes:
            return False
        # Full-write loop: os.write may short-write (and silently truncate >= ~2 GiB
        # payloads). A failed/short append STILL advanced the fd's real EOF: lseek back
        # to it and re-sync _blob_eof, or the NEXT record's journal offset would point
        # below its data (corrupt restores).
        view = memoryview(payload + b"\0" * (padded - len(payload)))
        try:
            while view:
                n = os.write(self._blob_fd, view)
                if n <= 0:
                    raise OSError("short blob append")
                view = view[n:]
            os.fsync(self._blob_fd)
        except OSError as e:
            logger.warning("session tier: blob append failed (%s); record aborted", e)
            self._blob_eof = os.lseek(self._blob_fd, 0, os.SEEK_END)
            return False
        off = self._blob_eof
        self._blob_eof += padded
        seg.blob, seg.l2_off, seg.l2_n = _BLOB_NAME, off, need
        if not self._journal_append(seg, off, need, zlib.crc32(payload)):
            return False   # bytes are durable but unrecoverable across a boot: abort cleanly
        self._ssd_used += padded
        return True

    def _release_l1(self, seg: _Segment) -> None:
        for i, key in enumerate(seg.page_keys):
            page = self._pages.get(key)
            if page is not None and page.l1_off == seg.l1_page_offs[i]:
                page.refs -= 1
                if page.refs == 0:
                    self._pool.free_block(page.l1_off, page.nbytes)
                    self._l1_used -= page.nbytes
                    del self._pages[key]
        for off, ln in zip(seg.l1_snap_offs, seg.snap_lens):
            if off >= 0:
                self._pool.free_block(off, ln)
                self._l1_used -= ln
        seg.l1_page_offs, seg.l1_snap_offs = [], []

    def evict_store(self, n: int) -> int:
        """Evict up to n LRU segments: L1 victims demote to L2, L2 victims are truly
        discarded (the only path back to a re-prefill). Returns how many were evicted."""
        if not self.enabled:
            return 0
        count = 0
        with self._lock:
            for _ in range(n):
                if self._demote_one_lru():
                    count += 1
                    continue
                l2 = [s for s in self._segments.values() if s.in_l2 and s.refs == 0]
                if not l2:
                    break
                self._discard(min(l2, key=lambda s: (s.last_validation, s.seg_id)))
                count += 1
            self._counters["evictions"] += count   # under the lock: Counter += is not atomic
        self.maybe_compact()   # discard pressure is the other watermark check point
        return count

    def _evict_oldest_l2(self, keep: int) -> None:
        l2 = [s for s in self._segments.values() if s.in_l2 and s.refs == 0]
        while self.ssd_bytes and self._ssd_used + keep > self.ssd_bytes and l2:
            victim = min(l2, key=lambda s: (s.last_validation, s.seg_id))
            self._discard(victim)
            l2.remove(victim)

    def _discard(self, seg: _Segment) -> None:
        # Append-only blob: the discarded padded span becomes a zero hole, reclaimable
        # only by compaction once the dead-byte watermark trips (maybe_compact).
        padded = (seg.l2_n + _BLK - 1) // _BLK * _BLK
        self._ssd_used -= padded
        self._dead_bytes += padded
        self._counters["discards"] += 1
        for i, key in enumerate(seg.page_keys):
            entries = self._index.get(key, [])
            self._index[key] = [e for e in entries if e[0] != seg.seg_id]
            if not self._index[key]:
                del self._index[key]
        del self._segments[seg.seg_id]
        # A staged ticket for a discarded segment can never be probed again: drop it with
        # the segment (a pending ticket is impossible here - the reader holds refs and a
        # discard requires refs==0).
        for t in [t for t in self._tickets.values() if t.seg_id == seg.seg_id]:
            self._tickets.pop(id(t), None)
            self._close_staging(t)
            t.state = "abandoned"
            self._counters["prefetch_abandon"] += 1

    def _supersede(self, seg: _Segment) -> None:
        # L1 resources first: _discard only accounts the L2 padded span (a no-op when the
        # segment never reached L2) plus index removal and staged-ticket drop.
        if seg.in_l1:
            self._release_l1(seg)
        self._discard(seg)

    # ----------------------------------------------------------------- restore

    def restore(self, handle: TierHandle) -> tuple[list[bytes], list[bytes]] | None:
        """Byte-identical pages [0:depth] plus the segment's snapshots; counts as a
        validation. Per-segment lock serializes concurrent restores, refcount guards
        the segment against eviction mid-copy."""
        if not self.enabled or handle is None:
            return None
        with self._lock:
            seg = self._segments.get(handle._seg_id)
            if seg is None:
                return None
            seg.refs += 1
        try:
            with seg.lock:
                depth = min(handle.depth, seg.boundary_len)
                if seg.in_l1:
                    pages = [self._pool.read(off, ln) for off, ln in
                             zip(seg.l1_page_offs[:depth], seg.page_lens[:depth])]
                    snaps = [self._pool.read(off, ln) for off, ln in
                             zip(seg.l1_snap_offs, seg.snap_lens)]
                elif seg.in_l2:
                    raw = self._blob_read(seg.blob, seg.l2_off, seg.l2_n)
                    pages, snaps = [], []
                    pos = 0
                    for i, ln in enumerate(seg.page_lens):
                        if i < depth:
                            pages.append(raw[pos:pos + ln])
                        pos += ln
                    for ln in seg.snap_lens:
                        snaps.append(raw[pos:pos + ln])
                        pos += ln
                else:
                    return None
                with self._lock:
                    seg.last_validation = time.monotonic_ns()
                    # counts ATTEMPTS: try_restore may still discard the result at its
                    # boundary/snapshot gate (cache.py) - the bytes were read regardless.
                    self._counters["restore_l1" if seg.in_l1 else "restore_l2"] += 1
                return pages, snaps
        finally:
            with self._lock:
                seg.refs -= 1

    def _blob_read(self, blob: str, off: int, nbytes: int) -> bytes:
        """O_DIRECT blob read (buffered IO caps ~1.5 GB/s) over a block-aligned window;
        falls back to buffered pread only where the filesystem refuses O_DIRECT."""
        path = os.path.join(self.dir, blob)
        pad_off = off // _BLK * _BLK
        end = (off + nbytes + _BLK - 1) // _BLK * _BLK
        window = end - pad_off
        buf = mmap.mmap(-1, window)
        try:
            try:
                fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
            except OSError:
                fd = os.open(path, os.O_RDONLY)
            try:
                done = 0
                while done < window:
                    got = os.preadv(fd, [memoryview(buf)[done:]], pad_off + done)
                    if got <= 0:
                        raise OSError(f"short blob read at {pad_off + done}")
                    done += got
            finally:
                os.close(fd)
            return bytes(buf[off - pad_off:off - pad_off + nbytes])
        finally:
            buf.close()

    # -------------------------------------------------------- prefetch staging (phase 2)

    @staticmethod
    def _close_staging(t: RestoreTicket) -> None:
        if t._buf is not None:
            t._buf.close()
        t._buf = None
        t._mv = None

    def begin_restore(self, handle: TierHandle) -> RestoreTicket | None:
        """Stage the segment's bytes on the background reader thread into an mlocked host
        buffer. The caller owns the reservation discipline (KV pages / snapshot slot);
        this only stages bytes. Returns None (staging nothing) when the ticket cap is
        reached, the segment is gone, or the staging allocation fails; never raises.
        Locks: refs (taken here, released at the read's finalize) keeps the segment
        un-evictable mid-read; the read itself runs under seg.lock without the global
        lock, exactly like the sync restore's copy phase. Staging is allocated on the
        CALLER thread (mlock quota bookkeeping is not thread-safe)."""
        if not self.enabled or handle is None:
            return None
        with self._lock:
            if len(self._tickets) >= _MAX_RESTORE_TICKETS:
                self._counters["prefetch_rej_cap"] += 1
                return None
            seg = self._segments.get(handle._seg_id)
            if seg is None:
                return None
            depth = min(handle.depth, seg.boundary_len)
            if depth == 0 or not (seg.in_l1 or seg.in_l2):
                return None
            seg.refs += 1
            page_lens, snap_lens = list(seg.page_lens), list(seg.snap_lens)
        ticket = RestoreTicket(handle=TierHandle(handle.path_key, depth, handle._seg_id),
                               seg_id=handle._seg_id, boundary=len(page_lens),
                               ready=threading.Event())
        try:
            ticket._buf = _LockedBuffer(sum(page_lens) + sum(snap_lens))
            ticket._mv = ticket._buf.mv
        except OSError as e:
            with self._lock:
                seg.refs -= 1
            logger.warning("session tier: prefetch staging alloc failed (%s); skipped", e)
            return None
        off = 0
        for ln in page_lens:
            ticket._page_spans.append((off, ln))
            off += ln
        for ln in snap_lens:
            ticket._snap_spans.append((off, ln))
            off += ln
        with self._lock:
            self._tickets[id(ticket)] = ticket
            self._counters["prefetch_begin"] += 1
            self._ensure_worker()
            self._inflight += 1
            self._work.put(ticket)
        return ticket

    def _ensure_worker(self) -> None:
        if self._worker is None:
            self._worker = threading.Thread(target=self._worker_loop,
                                            name="session-tier-prefetch", daemon=True)
            self._worker.start()

    def _worker_loop(self) -> None:
        while True:
            ticket = self._work.get()
            try:
                self._stage_ticket(ticket)
            except Exception as e:                   # belt and braces: the reader never dies
                logger.warning("session tier: prefetch worker error (%s)", e)
                self._finalize_ticket(ticket, False, str(e))
            finally:
                with self._lock:
                    self._inflight -= 1

    def _stage_ticket(self, t: RestoreTicket) -> None:
        """Reader-thread body: copy the segment bytes into the staging buffer, then
        finalize under the global lock. Byte sources are exactly the sync restore's (L1
        pool offsets / _blob_read), so a consumed staging is byte-identical by
        construction. A same-boundary re-offer may swap the L1 snapshot offsets mid-read
        (attach is not refs-guarded) - the content at an unchanged boundary is identical,
        the same benign exposure the sync restore already has."""
        ok, err = True, None
        try:
            seg = self._segments.get(t.seg_id)       # refs keeps the entry alive
            if seg is None:
                raise RuntimeError("segment vanished before the staging read")
            with seg.lock:
                if t.state == "pending":
                    if seg.in_l1:
                        for (off, ln), l1_off in zip(t._page_spans, seg.l1_page_offs):
                            if ln:
                                t._mv[off:off + ln] = self._pool.mm[l1_off:l1_off + ln]
                        for (off, ln), l1_off in zip(t._snap_spans, seg.l1_snap_offs):
                            if ln:
                                t._mv[off:off + ln] = self._pool.mm[l1_off:l1_off + ln]
                    elif seg.in_l2:
                        raw = self._blob_read(seg.blob, seg.l2_off, seg.l2_n)
                        pos = 0
                        for off, ln in t._page_spans + t._snap_spans:
                            if ln:
                                t._mv[off:off + ln] = raw[pos:pos + ln]
                            pos += ln
                    else:
                        raise RuntimeError("segment has no bytes in either tier")
        except Exception as e:                       # noqa: BLE001 (finalized as failed)
            ok, err = False, str(e)
        self._finalize_ticket(t, ok, err)

    def _finalize_ticket(self, t: RestoreTicket, ok: bool, err: str | None) -> None:
        """Once-only ticket finalize: refs release + state + staging disposition, then the
        ready handoff. Both the reader's normal path and the worker's belt-and-braces
        except-path land here, so a double finalize (and a double refs decrement) is
        structurally impossible."""
        with self._lock:
            if t._finalized:
                return
            t._finalized = True
            seg = self._segments.get(t.seg_id)
            if seg is not None:
                seg.refs -= 1
            still = self._tickets.get(id(t)) is t
            if not ok:
                if still:
                    self._tickets.pop(id(t), None)
                self._close_staging(t)
                t.state, t.error = "failed", err
                self._counters["prefetch_fail"] += 1
            elif t.state == "abandoned":
                if still:                            # dropped while reading
                    self._tickets.pop(id(t), None)
                self._close_staging(t)
            else:
                t.state = "ready"
        t.ready.set()

    def consume_ticket(self, ticket: RestoreTicket) -> tuple[list[bytes], list[bytes]] | None:
        """Adopt a staged ticket: wait for the read, then hand out the staged bytes in the
        same (pages, snaps) shape restore() returns and release the staging. None when the
        ticket is gone or the staging failed - the caller falls back to the sync restore.
        The last_validation stamp rides the consume (same currency as restore()); the
        pages cover the FULL staged boundary - the caller depth-truncates like it does
        for restore()."""
        ticket.ready.wait()                          # the reader always finishes (or fails)
        with self._lock:
            if self._tickets.pop(id(ticket), None) is not ticket:
                return None
            if ticket.state != "ready" or ticket._mv is None:
                return None
            pages = [bytes(ticket._mv[o:o + n]) for o, n in ticket._page_spans]
            snaps = [bytes(ticket._mv[o:o + n]) for o, n in ticket._snap_spans]
            self._close_staging(ticket)
            seg = self._segments.get(ticket.seg_id)
            if seg is not None:
                seg.last_validation = time.monotonic_ns()
            self._counters["prefetch_adopt"] += 1
            return pages, snaps

    def abandon_ticket(self, ticket: RestoreTicket) -> None:
        """Drop a staged ticket WITHOUT consuming it: staging freed, nothing handed out,
        nothing stamped. Idempotent. A still-reading ticket hands its staging to the
        reader's finalize - the scheduler never waits on abandon."""
        with self._lock:
            if self._tickets.pop(id(ticket), None) is not ticket:
                return
            if ticket.state == "pending":
                ticket.state = "abandoned"           # the finalize owns the staging now
            else:
                self._close_staging(ticket)
                ticket.state = "abandoned"
            self._counters["prefetch_abandon"] += 1

    def drop_tickets(self) -> int:
        """Abandon every live staging ticket (shutdown / replay safety net): staging freed,
        nothing consumed. Returns how many were dropped."""
        with self._lock:
            dropped = list(self._tickets.values())
            self._tickets.clear()
            for t in dropped:
                if t.state == "pending":
                    t.state = "abandoned"            # the finalize owns the staging now
                else:
                    self._close_staging(t)
                    t.state = "abandoned"
                self._counters["prefetch_abandon"] += 1
            return len(dropped)

    def _tickets_settled(self) -> bool:
        """Test hook: every queued staging item has been finalized."""
        return self._work.empty() and self._inflight == 0

    # --------------------------------------------------------------- lifecycle

    def flush_live(self) -> int:
        """Graceful shutdown: demote every live L1 segment into L2. Returns the count."""
        if not self.enabled:
            return 0
        count = 0
        with self._lock:
            for seg in list(self._segments.values()):
                if seg.in_l1 and seg.refs == 0 and self._demote(seg):
                    count += 1
        if count:
            logger.info("session tier: flushed %d live segments to L2", count)
        return count

    def _journal_append(self, seg: _Segment, off: int, nbytes: int, crc: int) -> bool:
        rec = {
            "path_key": seg.path_key.hex(), "blen": seg.boundary_len,
            "page_keys": [k.hex() for k in seg.page_keys],
            "page_lens": seg.page_lens, "snap_lens": seg.snap_lens,
            "blob": seg.blob, "off": off, "n": nbytes, "ts": seg.last_validation,
            # payload checksum: replay drops records whose blob region no longer matches
            # (holes after compaction, torn/short appends) - the convergence mechanism.
            "crc": crc,
        }
        payload = json.dumps(rec, sort_keys=True).encode()
        try:
            os.write(self._journal_fd,
                     struct.pack("<II", len(payload), zlib.crc32(payload)) + payload)
            os.fsync(self._journal_fd)
        except OSError as e:
            logger.warning("session tier: journal append failed (%s); not durable", e)
            return False
        return True

    def replay_journal(self) -> int:
        """Boot path: rebuild the L2 index from the journal. A truncated or torn tail
        (SIGKILL mid-append) stops replay at the last complete record. Idempotent: the
        L2 index is rebuilt from scratch each call."""
        if not self.enabled or not self._journal_path or not os.path.exists(self._journal_path):
            return 0
        with open(self._journal_path, "rb") as f:
            journal = f.read()
        pos, records = 0, []
        while pos + 8 <= len(journal):
            plen, crc = struct.unpack_from("<II", journal, pos)
            if pos + 8 + plen > len(journal):
                break
            payload = journal[pos + 8:pos + 8 + plen]
            if zlib.crc32(payload) != crc:
                break
            records.append(json.loads(payload))
            pos += 8 + plen
        if pos != len(journal):
            logger.warning("session tier: journal tail truncated/torn, "
                           "%d of %d bytes recovered", pos, len(journal))
        # Payload-crc validation: a record whose blob region no longer matches (holes left
        # by compaction, a torn append that advanced EOF) is dead - drop, never resurrect.
        blob_fd = -1
        if self._blob_path and os.path.exists(self._blob_path):
            try:
                blob_fd = os.open(self._blob_path, os.O_RDONLY)
            except OSError:
                blob_fd = -1
        if blob_fd >= 0:
            kept, dropped = [], 0
            for rec in records:
                ok = True
                if "crc" in rec:
                    try:
                        ok = zlib.crc32(os.pread(blob_fd, rec["n"], rec["off"])) == rec["crc"]
                    except OSError:
                        ok = False
                (kept.append(rec) if ok else None)
                dropped += 0 if ok else 1
            os.close(blob_fd)
            records = kept
            if dropped:
                logger.info("session tier: replay dropped %d dead/torn records", dropped)
        with self._lock:
            for t in list(self._tickets.values()):   # boot replay: no ticket survives it
                self._tickets.pop(id(t), None)
                self._close_staging(t)
                t.state = "abandoned"
            keep_l1 = {sid: seg for sid, seg in self._segments.items() if seg.in_l1}
            self._segments = dict(keep_l1)
            self._index = {}
            self._next_seg = 1
            for seg in self._segments.values():
                self._touch_index(seg)
                self._next_seg = max(self._next_seg, seg.seg_id + 1)
            for rec in records:
                seg = _Segment(self._next_seg, bytes.fromhex(rec["path_key"]), rec["blen"],
                               [bytes.fromhex(k) for k in rec["page_keys"]],
                               rec["page_lens"], rec["snap_lens"])
                self._next_seg += 1
                seg.blob, seg.l2_off, seg.l2_n = rec["blob"], rec["off"], rec["n"]
                seg.last_validation = rec["ts"]
                self._segments[seg.seg_id] = seg
                self._touch_index(seg)
            # Crash recovery can leave dead blob regions (holes from an interrupted
            # compaction, discarded-on-other-boot records): recompute from the live set
            # so the runtime watermark starts from the true count.
            self._dead_bytes = max(0, self._ssd_used - sum(
                (s.l2_n + _BLK - 1) // _BLK * _BLK
                for s in self._segments.values() if s.in_l2))
        if records:
            logger.info("session tier: replayed %d journal records", len(records))
        return len(records)

    # ------------------------------------------------------------- log compaction

    def maybe_compact(self) -> int:
        """Watermark-gated runtime compaction: fires only when dead blob bytes exceed
        the constant share of the blob AND the blob is above the floor; two int compares
        (and no locking) on every below-watermark check point. Returns dead records
        dropped (0 = no-op)."""
        if not self.enabled or self._blob_fd < 0:
            return 0
        if self._blob_eof < _COMPACT_FLOOR_BYTES or \
                self._dead_bytes <= self._blob_eof * _COMPACT_DEAD_FRACTION:
            return 0
        dropped = self.compact()
        if dropped:
            logger.info("session tier: compaction dropped %d dead records; "
                        "l2 used %.2f GiB (blob eof %.2f GiB)",
                        dropped, self._ssd_used / 2**30, self._blob_eof / 2**30)
        return dropped

    def compact(self) -> int:
        """Shutdown-checkpoint log compaction (TASK.md Component 3): rewrite the blob with
        only the live segments - at their ORIGINAL offsets, dead regions left as holes -
        then rewrite the journal to the surviving records. Crash safety: the compacted blob
        is fsync'ed and atomically renamed BEFORE the journal rewrite, so a SIGKILL between
        the two leaves the OLD journal against the NEW blob; replay's payload-crc check
        drops every dead record (its region reads as a zero hole), converging to exactly
        the live segments. With zero dead records this is a no-op (returns 0).
        Lock coverage: runs entirely under the store's global RLock, serializing against
        offer/evict/discard. A restore's blob read happens outside that lock, but reads a
        LIVE record, which keeps its ORIGINAL offset across the rewrite (only dead regions
        become holes), and a mid-restore segment is refs-guarded against discard - a
        reader can never see different bytes for its segment."""
        if not self.enabled or self._blob_fd < 0:
            return 0
        with self._lock:
            records = self._parse_journal()
            # Survival keyed by EXACT record identity (path, offset, length), NOT last-wins
            # per path: a crashed discard + replay can leave TWO live segments on one
            # path_key, and last-wins survival would hole the older live segment's blob
            # region while it stays indexed (restore then returns zeros = corruption).
            live = {(seg.path_key.hex(), seg.l2_off, seg.l2_n)
                    for seg in self._segments.values() if seg.in_l2}
            keep = [rec for rec in records
                    if (rec["path_key"], rec["off"], rec["n"]) in live]
            if len(keep) == len(records):
                return 0                      # nothing dead: no-op (also covers empty journal)
            tmp = self._blob_path + ".compact"
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            ro = os.open(self._blob_path, os.O_RDONLY)   # _blob_fd itself is write-only
            try:
                for rec in keep:
                    # copy the PADDED record span: the O_DIRECT reader reads block-aligned
                    # windows, so a truncated tail would break restores of the last segment
                    span = (rec["n"] + _BLK - 1) // _BLK * _BLK
                    os.pwrite(fd, os.pread(ro, span, rec["off"]), rec["off"])
                os.fsync(fd)
            finally:
                os.close(ro)
                os.close(fd)
            os.rename(tmp, self._blob_path)   # atomic: old blob intact until here
            os.close(self._blob_fd)
            self._blob_fd = os.open(self._blob_path,
                                    os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            self._rewrite_journal(keep)
            self._ssd_used = sum((r["n"] + _BLK - 1) // _BLK * _BLK
                                 for r in keep)
            # Runtime compaction is new (phase 1 ran it shutdown-only): re-sync _blob_eof
            # to the truncated file, or the next append's journal offset would point above
            # its data. Dead accounting is fully reclaimed by the rewrite.
            self._blob_eof = os.lseek(self._blob_fd, 0, os.SEEK_END)
            self._dead_bytes = 0
            return len(records) - len(keep)

    def _parse_journal(self) -> list[dict]:
        """Ordered, crc-checked journal records (same format replay rebuilds from)."""
        if not self._journal_path or not os.path.exists(self._journal_path):
            return []
        with open(self._journal_path, "rb") as f:
            journal = f.read()
        pos, records = 0, []
        while pos + 8 <= len(journal):
            plen, crc = struct.unpack_from("<II", journal, pos)
            if pos + 8 + plen > len(journal):
                break
            payload = journal[pos + 8:pos + 8 + plen]
            if zlib.crc32(payload) != crc:
                break
            records.append(json.loads(payload))
            pos += 8 + plen
        return records

    def _rewrite_journal(self, records: list[dict]) -> None:
        tmp = self._journal_path + ".compact"
        with open(tmp, "wb") as f:
            for rec in records:
                payload = json.dumps(rec, sort_keys=True).encode()
                f.write(struct.pack("<II", len(payload), zlib.crc32(payload)) + payload)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp, self._journal_path)

    # ----------------------------------------------------------------- metrics

    def snapshot(self) -> dict:
        """Counter + byte-gauge snapshot for the metrics export; {} when off."""
        if not self.enabled:
            return {}
        with self._lock:
            snap = dict(self._counters)
            snap.update(l1_used=self._l1_used, ssd_used=self._ssd_used,
                        blob_eof=self._blob_eof, dead_bytes=self._dead_bytes,
                        segments=len(self._segments), tickets=len(self._tickets))
            return snap

    def stats_line(self) -> str:
        """Compact one-liner for the scheduler's periodic batch log; "" when off."""
        snap = self.snapshot()
        if not snap:
            return ""
        return (f"session-tier: l1={snap['l1_used'] / 2**20:.1f}MiB, "
                f"l2={snap['ssd_used'] / 2**20:.1f}MiB, "
                f"offers={snap['offers_ok']}/{snap['offers_rej']}, "
                f"dedup={snap['offers_dedup']}, "
                f"probes={snap['probe_hit']}/{snap['probe_miss']}, "
                f"restore={snap['restore_l1']}(l1)/{snap['restore_l2']}(l2), "
                        f"evict={snap['evictions']}, demote={snap['demotions']}, "
                        f"discard={snap['discards']}, dead={snap['dead_bytes'] / 2**20:.1f}MiB, "
                        f"pf={snap['prefetch_adopt']}/{snap['prefetch_begin']}/"
                        f"{snap['prefetch_abandon']}")
