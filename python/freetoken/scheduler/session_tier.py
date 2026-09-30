"""SessionTierStore: tiered cache of evicted session segments (RAM pool -> SSD blobs).

A segment is one session path [0:N] of KV pages plus up to 3 GDN snapshot byte blobs.
Pages are content-addressed by chain page keys so shared subagent prefixes dedup and
appends are incremental. L1 is one mlock'ed contiguous allocation; L2 is append-only
blob files guarded by a fsync-ordered journal (blob bytes durable BEFORE journal
record), so a SIGKILL mid-append can only truncate the tail, never corrupt accepted
records. The shutdown flush writes whole groups in parallel pwrite bursts with ONE
group fdatasync before the group's journal records: the crash loss window widens
from one segment to one group (replay drops the unjournaled blob tail), corruption
is still impossible. The flush is a producer/disk pipeline: records go out as
zero-copy pwritev bursts from the L1 pool mmap and group N+1 is built while group N
is written. Journal payload checksums are versioned per record (crc_alg):
new records carry hardware CRC32C, old catalogs' zlib-crc32 records replay unchanged.
"""
from __future__ import annotations

import crc32c
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
from concurrent.futures import Future, ThreadPoolExecutor
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
# Clean-shutdown sidecar (P2): written only after a completed shutdown_tier; lets boot
# replay skip the per-record payload crc re-read. Absent/stale/torn -> full verification.
_MARKER_NAME = "shutdown.marker"
# Scheduled compaction (runtime, no CLI flag): rewrite the blob once dead bytes exceed
# this share of the blob AND the blob is above the floor (no churn on small installs).
_COMPACT_DEAD_FRACTION = 0.25
_COMPACT_FLOOR_BYTES = 256 << 20
# Speculative restore prefetch (phase 2, no CLI flag): at most this many concurrent
# staging tickets, each holding one mlocked host buffer with the segment bytes until
# adopt/abandon. The scheduler's idle hook (via the cache manager) is the only caller.
_MAX_RESTORE_TICKETS = 2
# P3/P4 flush write path (flush_live only): segments are batched into volume groups -
# within a group the records go out as parallel pwritev bursts straight from the L1
# pool mmap (zero-copy, no payload assembly; the blob fd has no O_APPEND) and ONE
# fdatasync makes the whole group durable before its journal records. Groups are
# pipelined: group N+1 is built (span reserves + iovecs + payload crc) while group N
# is being written. Runtime demotions (offer/evict pressure) stay single-segment in
# _write_blob.
_FLUSH_GROUP_BYTES = 1 << 30
_FLUSH_WRITERS = 8
# pwritev iovec cap per call (Linux IOV_MAX is 1024) and the shared zero padding tail
# source for the reserved-but-unwritten span remainder.
_PWRITEV_MAX_IOV = 512
_ZERO_BLK = bytes(_BLK)
# Guards the cumulative mlock accounting (_Pool._locked_total): it is mutated from the
# scheduler thread (pool construction, staging alloc, free_block) AND the reader thread
# (staging close at ticket finalize).
_LOCKED_TOTAL_LOCK = threading.Lock()

# Journal payload-crc algorithm versioning: a record with "crc_alg":"crc32c" carries a
# Castagnoli crc (hardware SSE4.2 via the crc32c package); a record without the field is
# legacy zlib-crc32. The journal frame itself (<II len, crc32>) stays zlib: it is
# KB-sized (vs GiB payloads) and its algorithm cannot be known before the record is
# parsed, so there is no cheap way to version it.
_CRC32C_ALG = "crc32c"


def _calc_payload_crc(data, alg: str) -> int:
    """Payload checksum for a NEW journal record (alg is always _CRC32C_ALG today)."""
    if alg == _CRC32C_ALG:
        return crc32c.crc32c(data)
    return zlib.crc32(data)


def _match_payload_crc(data, alg: str | None, crc: int) -> bool:
    """Verify a record's payload checksum by its crc_alg field. An absent field is the
    legacy zlib format; an UNKNOWN algorithm can never verify, so the record is dead
    (replay drops it - never resurrect). Downgrade contract: an older zlib-only binary
    replaying a crc32c journal drops those records as dead - the journal stays
    well-formed (frames are still zlib), the price is those segments' data, no corruption."""
    if alg == _CRC32C_ALG:
        return crc32c.crc32c(data) == crc
    if alg is None or alg == "zlib":
        return zlib.crc32(data) == crc
    return False


def _calc_payload_crc_iovecs(bufs, need: int, alg: str) -> int:
    """Chained checksum over a record's iovec list, covering exactly `need` payload
    bytes - the zero padding tail is excluded, matching the assembled-path crc."""
    if alg != _CRC32C_ALG:
        data = bytearray()            # legacy algorithms are never produced today
        left = need
        for b in bufs:
            if left <= 0:
                break
            take = min(left, len(b))
            data += b[:take]
            left -= take
        return zlib.crc32(bytes(data))
    crc = 0
    left = need
    for b in bufs:
        if left <= 0:
            break
        take = min(left, len(b))
        crc = crc32c.crc32c(b[:take], crc)   # chained; GIL released above 32 KiB
        left -= take
    return crc


class _PendingVolume:
    """Reserved-but-unjournaled blob volume shared by the flush pipeline threads: the
    builder adds each reserved span before the cap check of the next one, the writer
    releases it once the group settles (journaled into _ssd_used, or aborted). A stale
    high read is conservative for the cap check, so plain condition-protected
    arithmetic suffices - the release order does not need to match the add order."""

    __slots__ = ("_cond", "_value")

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._value = 0

    def add(self, n: int) -> None:
        with self._cond:
            self._value += n

    def value(self) -> int:
        with self._cond:
            return self._value

    def release(self, n: int) -> None:
        with self._cond:
            self._value -= n
            self._cond.notify_all()

    def wait_drained(self, timeout: float) -> bool:
        """Block until every in-flight group has settled. Only safe on the builder
        thread BEFORE it has made any reservation of its own (otherwise its own pending
        volume could never drain - deadlock). On a timeout the caller falls back to a
        cap check with pending=0: that may transiently over-commit the ssd cap by up to
        one group volume. Acceptable because the flush pipeline runs only on the
        shutdown path (a bounded, one-shot drain), and the caller logs the timeout."""
        with self._cond:
            return self._cond.wait_for(lambda: self._value == 0, timeout)


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


@dataclass
class _GroupEntry:
    """One built flush-group record: reserved span plus its zero-copy iovec list and
    payload crc (both lifted off the L1 pool by the builder thread)."""

    seg: _Segment
    off: int
    need: int
    padded: int
    bufs: list = field(default_factory=list)
    crc: int = 0
    released: bool = False          # pipeline-claims guard, see _release_stage


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
        # Guards the _ssd_used / _dead_bytes arithmetic against the flush pipeline: the
        # builder thread evicts L2 under cap pressure while the writer thread journals
        # the previous group (both mutate _ssd_used; int += is not atomic). A leaf lock:
        # never acquired around another lock.
        self._account_lock = threading.Lock()
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
        # Blob tail reservation watermark (pwrite appends at reserved offsets); never
        # decreased. _ssd_used is capacity accounting only and shrinks on L2 discard.
        self._blob_eof = 0
        self._blob_path = ""
        self._journal_path = ""
        self._blob_fd = -1
        self._journal_fd = -1
        # Generation carried by the clean-shutdown marker: monotonic across shutdowns of
        # this directory (seeded from the marker at a fast-path boot), guarding against a
        # compacted journal shrinking back to a stale marker's recorded size.
        self._marker_generation = 0
        # Whether the marker file is believed to be on disk right now: gates the durable
        # unlink in invalidate_shutdown_marker so the discard hot path pays neither the
        # dir fsync nor the ENOENT syscall when no marker exists.
        self._marker_present = False
        # Blob bytes freed by L2 discards (regions read as zero holes) until compaction
        # reclaims them; boot replay recomputes this from the recovered record set.
        if self.dir is not None:
            os.makedirs(self.dir, exist_ok=True)
            self._blob_path = os.path.join(self.dir, _BLOB_NAME)
            self._journal_path = os.path.join(self.dir, _JOURNAL_NAME)
            self._marker_present = os.path.exists(os.path.join(self.dir, _MARKER_NAME))
            # No O_APPEND: appends are pwrite at reserved offsets (pwrite on an O_APPEND
            # fd would ignore the offset); _blob_eof is the reservation watermark.
            self._blob_fd = os.open(self._blob_path, os.O_WRONLY | os.O_CREAT, 0o644)
            self._journal_fd = os.open(self._journal_path,
                                       os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            self._ssd_used = os.path.getsize(self._blob_path)
            self._blob_eof = self._ssd_used
            # Provisional watermark seed only: replay_journal() reseeds _ssd_used from
            # the live record set and owns the "session tier on" line, which must
            # report the live figure on both boot paths.

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

    def _reserve_blob_span(self, need: int, pending: int = 0) -> tuple[int, int] | None:
        """Cap-check (evicting oldest L2 first), then reserve the record's padded span at
        the blob tail. Returns (off, padded), or None when the cap cannot be met.
        pending is the volume reserved earlier in the same flush group but not yet
        journaled (_ssd_used lags until the group's records land in _journal_and_account):
        counting it keeps the HEAD strict cap at reserve granularity, so a group can
        never reserve past the ssd cap by up to a group volume."""
        padded = (need + _BLK - 1) // _BLK * _BLK
        # No _account_lock on this snapshot: it can only be stale HIGH. A concurrent
        # journal moves a span from pending into _ssd_used while pending still counts
        # it (the sum over-counts until the group settles), and a concurrent eviction
        # only lowers _ssd_used - so the check stays conservative unsynchronized. The
        # lock also must NOT span the evictor below: _discard takes it (non-reentrant).
        if self.ssd_bytes and self._ssd_used + pending + padded > self.ssd_bytes:
            self._evict_oldest_l2(need, pending)
            if self.ssd_bytes and self._ssd_used + pending + padded > self.ssd_bytes:
                return None
        off = self._blob_eof
        self._blob_eof += padded
        return off, padded

    def _pwrite_span(self, off: int, view: memoryview) -> None:
        """Full-write loop at an explicit offset (pwrite may short-write). A failed or
        short write leaves the reserved span as a hole: the reservation watermark is
        already past it and no journal record will ever point into it, so the old
        O_APPEND lseek resync is unnecessary."""
        pos = 0
        while pos < len(view):
            n = os.pwrite(self._blob_fd, view[pos:], off + pos)
            if n <= 0:
                raise OSError("short blob write")
            pos += n

    def _journal_and_account(self, seg: _Segment, off: int, need: int, padded: int,
                             crc: int, crc_alg: str = _CRC32C_ALG) -> bool:
        """Journal one durable record and take the capacity accounting. False = the blob
        bytes are durable but unrecoverable across a boot: abort cleanly (the segment
        keeps its L1 residency; no journal entry, no accounting)."""
        seg.blob, seg.l2_off, seg.l2_n = _BLOB_NAME, off, need
        if not self._journal_append(seg, off, need, crc, crc_alg):
            return False
        with self._account_lock:      # racing the builder thread's cap evictions
            self._ssd_used += padded
        return True

    def _write_blob(self, seg: _Segment, payload: bytes | None) -> bool:
        """Append one blob record (padded to _BLK for the O_DIRECT reader): pwrite at the
        reserved offset, ONE fdatasync, then the journal record - the durable order is
        what makes replay crash-safe. Every failure path returns False: no journal entry,
        no capacity accounting. Single-segment path (runtime offer/evict pressure); the
        shutdown flush batches segments through _write_group instead."""
        need = sum(seg.page_lens) + sum(seg.snap_lens)
        r = self._reserve_blob_span(need)
        if r is None:
            return False
        off, padded = r
        view = memoryview((payload if payload is not None else b"")
                          + b"\0" * (padded - need))
        try:
            self._pwrite_span(off, view)
            os.fdatasync(self._blob_fd)
        except OSError as e:
            logger.warning("session tier: blob append failed (%s); record aborted", e)
            return False
        return self._journal_and_account(seg, off, need, padded,
                                         _calc_payload_crc(view[:need], _CRC32C_ALG))

    def _seg_iovecs(self, seg: _Segment, padded: int) -> list:
        """Zero-copy record layout: payload iovecs straight from the L1 pool mmap (pages,
        then snapshots) plus the zero padding tail. The pool regions stay stable until
        the write completes because flush_live holds the store lock - no offer/evict
        mutation can touch them mid-flight."""
        base = memoryview(self._pool.mm)
        bufs = [base[off:off + ln]
                for off, ln in zip(seg.l1_page_offs, seg.page_lens)]
        bufs.extend(base[off:off + ln]
                    for off, ln in zip(seg.l1_snap_offs, seg.snap_lens) if ln)
        pad = padded - sum(seg.page_lens) - sum(seg.snap_lens)
        if pad > 0:
            bufs.append(memoryview(_ZERO_BLK)[:pad])
        return bufs

    def _pwritev_span(self, off: int, bufs: list, total: int) -> None:
        """pwritev a buffer list at an explicit offset - the pwritev analogue of
        _pwrite_span: loops over short writes and chunks the iovec list to
        _PWRITEV_MAX_IOV. A failed or short write leaves the reserved span as a hole:
        the reservation watermark is already past it and no journal record will ever
        point into it."""
        pos = 0
        i, skip = 0, 0                     # cursor: bufs[i], `skip` bytes consumed in it
        while pos < total:
            iov, room = [], _PWRITEV_MAX_IOV
            j, jskip = i, skip
            while j < len(bufs) and room:
                n = len(bufs[j]) - jskip
                if n > 0:
                    iov.append(bufs[j][jskip:] if jskip else bufs[j])
                    room -= 1
                j += 1
                jskip = 0
            if not iov:
                raise OSError("short blob write")
            n = os.pwritev(self._blob_fd, iov, off + pos)
            if n <= 0:
                raise OSError("short blob write")
            pos += n
            while n and i < len(bufs):
                avail = len(bufs[i]) - skip
                if avail <= n:
                    n -= avail
                    i += 1
                    skip = 0
                else:
                    skip += n
                    n = 0

    def _release_stage(self, stage: list[_GroupEntry],
                       pending: _PendingVolume) -> None:
        """Release a built stage's pipeline claims exactly once per entry: the pending
        volume returns to the cap budget, the refs pin comes off (a segment between
        reserve and journal looks in_l2 to the evictor). Idempotent via the entry's
        released flag: the writer's finally and a BaseException sweep in _flush_groups
        can both reach the same stage."""
        for en in stage:
            if en.released:
                continue
            en.released = True
            pending.release(en.padded)
            en.seg.refs -= 1

    def _build_group_stage(self, group: list[_Segment], pending: _PendingVolume,
                           index: int = 0) -> list[_GroupEntry] | None:
        """Runs on the flush builder thread while the PREVIOUS group is being written:
        reserve every segment's span (cap check against the pipeline-wide pending
        volume, evicting oldest L2 under pressure), pin each admitted segment with a
        refs claim (a reserved-but-unjournaled span makes the segment look in_l2 to
        the evictor), then lift each record's zero-copy iovecs and payload crc off the
        L1 pool. No payload byte is ever copied. Returns the stage list, or None when
        nothing was admitted or the build failed - the segments keep their L1
        residency and the group is aborted with no journal side effect."""
        stage: list[_GroupEntry] = []
        cur_seg, cur_need = group[0], 0
        try:
            for seg in group:
                cur_seg, cur_need = seg, sum(seg.page_lens) + sum(seg.snap_lens)
                r = self._reserve_blob_span(cur_need, pending.value())
                if r is None and not stage and pending.value():
                    # P3 sacrifice semantics across the pipeline: the refusal may be
                    # caused by the previous group's still-unjournaled volume alone.
                    # Once it settles, the just-journaled segments become evictable and
                    # the reserve can succeed by sacrificing them - exactly what the
                    # serialized pre-P4 flush did (a later segment sacrificed an earlier
                    # journaled one). Only retried before this group's first admission,
                    # so the pending being waited on never includes this build's own
                    # reservations (they could never drain -> deadlock).
                    if not pending.wait_drained(60.0):
                        logger.warning(
                            "session tier: flush group #%d waited 60 s for the previous "
                            "group to settle, %d bytes still unjournaled; retrying the "
                            "cap check with pending=0", index, pending.value())
                    r = self._reserve_blob_span(cur_need, 0)
                if r is None:
                    logger.warning("session tier: L2 cap reached, cannot demote seg %d",
                                   seg.seg_id)
                    continue
                pending.add(r[1])
                seg.refs += 1                     # evictor guard, released in _write_group
                stage.append(_GroupEntry(seg, r[0], cur_need, r[1]))
            if not stage:
                return None
            for en in stage:
                cur_seg, cur_need = en.seg, en.need
                en.bufs = self._seg_iovecs(en.seg, en.padded)
                en.crc = _calc_payload_crc_iovecs(en.bufs, en.need, _CRC32C_ALG)
        except Exception as e:                    # build failure: abort the whole group
            logger.warning(
                "session tier: flush group #%d build failed at seg %d (%d bytes, %d of "
                "%d admitted): %s; group aborted",
                index, cur_seg.seg_id, cur_need, len(stage), len(group), e)
            self._release_stage(stage, pending)
            return None
        except BaseException:                     # KI/SystemExit: unpin, then propagate
            self._release_stage(stage, pending)
            raise
        return stage

    def _pwrite_group(self, stage: list[_GroupEntry]) -> set[int]:
        """Parallel pwritev bursts at the reserved offsets. Returns the indices whose
        write failed (their spans stay holes behind the watermark)."""
        failed: set[int] = set()

        def _write_one(i: int) -> None:
            en = stage[i]
            try:
                self._pwritev_span(en.off, en.bufs, en.padded)
            except OSError as e:
                logger.warning("session tier: blob write failed for seg %d (%s); "
                               "record aborted", en.seg.seg_id, e)
                failed.add(i)

        if len(stage) == 1:
            _write_one(0)
        else:
            with ThreadPoolExecutor(max_workers=min(_FLUSH_WRITERS, len(stage))) as ex:
                list(ex.map(_write_one, range(len(stage))))
        return failed

    def _write_group(self, stage: list[_GroupEntry], pending: _PendingVolume) -> int:
        """Write one built group: parallel pwritev bursts at the reserved offsets, ONE
        fdatasync for the whole group, then the group's journal records. Durable-order
        invariant unchanged - a journal record still appears only after the blob bytes
        of its group are durable - but the crash loss window widens from one segment to
        the whole group: a SIGKILL between the fdatasync and the journal appends leaves
        the group's bytes durable and unjournaled, and replay drops that blob tail
        (segments stay in L1 in a process that survives). The pipeline-wide pending
        volume stays counted until this group settles, so a concurrent builder always
        sees a conservative cap budget. Returns the demoted count."""
        if not stage:
            return 0
        count = 0
        try:
            failed = self._pwrite_group(stage)
            # Group fdatasync: every byte written above is durable BEFORE any journal
            # record of the group appears.
            try:
                os.fdatasync(self._blob_fd)
            except OSError as e:
                logger.warning("session tier: blob fdatasync failed (%s); group aborted", e)
                return 0
            for i, en in enumerate(stage):
                if i in failed:
                    continue
                if self._journal_and_account(en.seg, en.off, en.need, en.padded, en.crc):
                    self._release_l1(en.seg)
                    self._counters["demotions"] += 1
                    count += 1
        finally:
            # Release the pipeline's claims whatever happened above (journaled, aborted
            # or raised): the pending volume returns to the cap budget, the refs pin
            # protects nothing once the journal attempt is over.
            self._release_stage(stage, pending)
        return count

    def _flush_groups(self, groups: list[list[_Segment]]) -> int:
        """Group pipeline (double buffering): group N+1 is built - spans reserved,
        iovecs and payload crcs lifted off the L1 pool - while group N is being
        pwritten, fdatasync'ed and journaled. Cross-group reserve serialization goes
        through the shared pending volume, so the cap accounting stays exact no matter
        how many groups are in flight. A BaseException (KI/SystemExit) mid-pipeline
        releases every built-but-unsettled stage's claims and propagates unchanged."""
        if self._blob_fd < 0:
            return 0
        pending = _PendingVolume()
        if len(groups) == 1:
            stage = self._build_group_stage(groups[0], pending, 0)
            return self._write_group(stage, pending) if stage else 0
        count = 0
        built: list[_GroupEntry] | None = None
        futs: list[Future] = []
        try:
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="flush-build") as ex:
                fut = ex.submit(self._build_group_stage, groups[0], pending, 0)
                futs.append(fut)
                for gi, group in enumerate(groups[1:], start=1):
                    nxt = ex.submit(self._build_group_stage, group, pending, gi)
                    futs.append(nxt)
                    built = fut.result()
                    if built:
                        count += self._write_group(built, pending)
                    fut = nxt
                built = fut.result()
                if built:
                    count += self._write_group(built, pending)
        except BaseException:
            # KI/SystemExit mid-pipeline: the stage the writer was working on is released
            # by _write_group's own finally - or HERE if the write never got to run (a
            # stubbed/failed write releases nothing). Every built-but-unconsumed future's
            # stage must be swept too, or its refs pins and pending volume wedge the next
            # flush_live (segments silently lost on retry). _release_stage is idempotent
            # per entry, so sweeping everything reachable is safe. KI propagates as KI.
            if built:
                self._release_stage(built, pending)
            for f in futs:
                try:
                    r = f.result()
                except BaseException:
                    r = None          # the builder released its own partial stage
                if r:
                    self._release_stage(r, pending)
            raise
        return count

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

    def _evict_oldest_l2(self, keep: int, pending: int = 0) -> None:
        l2 = [s for s in self._segments.values() if s.in_l2 and s.refs == 0]
        while self.ssd_bytes and self._ssd_used + pending + keep > self.ssd_bytes and l2:
            victim = min(l2, key=lambda s: (s.last_validation, s.seg_id))
            self._discard(victim)
            l2.remove(victim)

    def _discard(self, seg: _Segment) -> None:
        # Append-only blob: the discarded padded span becomes a zero hole, reclaimable
        # only by compaction once the dead-byte watermark trips (maybe_compact).
        padded = (seg.l2_n + _BLK - 1) // _BLK * _BLK
        with self._account_lock:      # racing the writer thread's journal accounting
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
        # Discard mutates no on-disk bytes (journal and blob eof untouched), so the
        # marker's size checks cannot see it: without this explicit drop the next boot
        # could fast-path-resurrect the evicted segment.
        self.invalidate_shutdown_marker()

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
        """Graceful shutdown: demote every live L1 segment into L2. Returns the count.
        Segments are batched into ~_FLUSH_GROUP_BYTES volume groups; the groups are
        pipelined (build of group N+1 overlaps the write/fdatasync/journal of group N,
        see _flush_groups) and each group is made durable with ONE fdatasync before
        its journal records (see _write_group for the widened crash loss window)."""
        if not self.enabled:
            return 0
        count = 0
        with self._lock:
            groups: list[list[_Segment]] = []
            group: list[_Segment] = []
            vol = 0
            # list() snapshot: the builder thread is submitted only AFTER this loop, so
            # nothing mutates _segments while we iterate today; the copy is protection
            # against FUTURE mutations here (the pipeline builder's cap evictions run
            # right below), not a fix for an existing race.
            for seg in list(self._segments.values()):
                if not seg.in_l1 or seg.refs != 0:
                    continue
                padded = (sum(seg.page_lens) + sum(seg.snap_lens)
                          + _BLK - 1) // _BLK * _BLK
                if group and vol + padded > _FLUSH_GROUP_BYTES:
                    groups.append(group)
                    group, vol = [], 0
                group.append(seg)
                vol += padded
            if group:
                groups.append(group)
            if groups:
                count = self._flush_groups(groups)
        if count:
            logger.info("session tier: flushed %d live segments to L2", count)
        return count

    # ---------------------------------------------------- clean-shutdown marker (P2)

    def invalidate_shutdown_marker(self) -> None:
        """Drop the clean-shutdown marker. Called at the START of a graceful shutdown
        (before any flush/compact mutation) and after every runtime mutation the marker's
        size checks cannot self-invalidate (a discard rewrites neither the journal nor the
        blob). An interrupted shutdown must never leave a marker behind: the next boot
        would fast-path a state that was never fully reached. The unlink is made durable
        with a directory fsync - without it a power loss can resurrect the deleted name,
        and if its sizes still match, a fast-path boot resurrects an evicted segment. The
        dir fsync is paid only when a marker actually exists (_marker_present)."""
        if not self.enabled or not self._journal_path or not self._marker_present:
            return
        try:
            os.unlink(os.path.join(self.dir, _MARKER_NAME))
            dfd = os.open(self.dir, os.O_RDONLY)   # make the unlink itself durable
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except FileNotFoundError:
            pass                      # absent already: the common case
        except OSError as e:
            logger.warning("session tier: shutdown marker unlink failed (%s); "
                           "next boot re-verifies the full replay", e)
            return                    # on-disk state unknown: keep the flag up
        self._marker_present = False

    def write_shutdown_marker(self) -> None:
        """End-of-shutdown barrier: journal size, blob eof and a monotonic generation,
        fsync'ed and atomically renamed into place. Meaningful only when every journal/
        blob mutation since boot followed the durable blob-before-journal order and the
        shutdown finished (flush + compact + final fsyncs). The generation guards against
        a size coincidence: a compaction can legitimately shrink the journal back to a
        stale marker's recorded size."""
        if not self.enabled or not self._journal_path:
            return
        try:
            rec = {"journal_bytes": os.path.getsize(self._journal_path),
                   "blob_eof": os.path.getsize(self._blob_path),
                   "generation": self._marker_generation + 1}
            payload = json.dumps(rec, sort_keys=True).encode()
            tmp = os.path.join(self.dir, _MARKER_NAME + ".new")
            with open(tmp, "wb") as f:
                f.write(struct.pack("<II", len(payload), zlib.crc32(payload)) + payload)
                f.flush()
                os.fsync(f.fileno())
            os.rename(tmp, os.path.join(self.dir, _MARKER_NAME))
            self._marker_present = True
            self._marker_generation = rec["generation"]
            dfd = os.open(self.dir, os.O_RDONLY)   # make the rename itself durable
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except OSError as e:
            logger.warning("session tier: shutdown marker write failed (%s); "
                           "next boot re-verifies the full replay", e)

    def _read_shutdown_marker(self, journal_bytes: int) -> tuple[bool, dict | None]:
        """Boot-time validity check: the marker parses, its crc holds, and BOTH recorded
        sizes match the files on disk. Returns (valid, parsed) - parsed is the marker's
        JSON dict when it json-parsed to a dict, else None, so the caller can report a
        present-but-rejected marker with its recorded sizes. Missing/garbled/mismatched
        -> (False, ...): full verification path, behavior unchanged. The in-memory
        generation is seeded from ANY crc-valid marker (stale ones too) so the next
        write stays strictly monotonic; the on-disk presence refreshes _marker_present."""
        mpath = os.path.join(self.dir, _MARKER_NAME)
        self._marker_present = os.path.exists(mpath)
        if not self._marker_present:
            return False, None
        try:
            with open(mpath, "rb") as f:
                framed = f.read()
            plen, crc = struct.unpack_from("<II", framed, 0)
            if 8 + plen != len(framed) or zlib.crc32(framed[8:]) != crc:
                return False, None
            mk = json.loads(framed[8:])
            # A crc-valid payload is still arbitrary JSON: a non-dict (list/number)
            # would AttributeError on .get below and kill the whole boot replay, so
            # guard explicitly instead of widening the except (which would swallow
            # genuine bugs).
            if not isinstance(mk, dict):
                return False, None
            gen = mk.get("generation")
            if isinstance(gen, int) and gen > 0:
                # Seed BEFORE the size match: a stale marker must still raise the
                # in-memory floor, or the next write could emit a generation <= the
                # rejected one.
                self._marker_generation = max(self._marker_generation, gen)
            ok = (mk.get("journal_bytes") == journal_bytes
                  and mk.get("blob_eof") == os.path.getsize(self._blob_path)
                  and isinstance(gen, int) and gen > 0)
            return ok, mk
        except (OSError, ValueError, struct.error):
            return False, None

    def _journal_append(self, seg: _Segment, off: int, nbytes: int, crc: int,
                        crc_alg: str = _CRC32C_ALG) -> bool:
        rec = {
            "path_key": seg.path_key.hex(), "blen": seg.boundary_len,
            "page_keys": [k.hex() for k in seg.page_keys],
            "page_lens": seg.page_lens, "snap_lens": seg.snap_lens,
            "blob": seg.blob, "off": off, "n": nbytes, "ts": seg.last_validation,
            # payload checksum: replay drops records whose blob region no longer matches
            # (holes after compaction, torn/short appends) - the convergence mechanism.
            # crc_alg versions the algorithm: new records carry hardware CRC32C; records
            # without the field are legacy zlib-crc32 (old catalogs replay unchanged).
            # The journal frame crc below stays zlib: see the versioning note at the top.
            "crc": crc,
        }
        if crc_alg:
            rec["crc_alg"] = crc_alg
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
        torn = pos != len(journal)
        if torn:
            logger.warning("session tier: journal tail truncated/torn, "
                           "%d of %d bytes recovered", pos, len(journal))
        # Clean-shutdown fast path: after a graceful shutdown every record followed the
        # durable blob-before-journal order, so the payload crc re-read below is pure
        # redundancy (~22 s on a 41 GiB L2). Marker sizes matching the on-disk files plus
        # an intact tail prove the state is exactly the shutdown checkpoint. Missing or
        # stale marker, torn tail: fall through to the full verification, which owns
        # crash convergence - behavior unchanged.
        fast, marker = False, None
        if not torn:
            fast, marker = self._read_shutdown_marker(len(journal))
        if not fast and self._marker_present and not torn:
            # An existing-but-rejected marker is worth one info line; an ABSENT marker
            # (the common first-boot / crash case) stays silent as before.
            recorded = (f"journal_bytes={marker.get('journal_bytes')} "
                        f"blob_eof={marker.get('blob_eof')} "
                        f"generation={marker.get('generation')}"
                        if marker else "unreadable")
            blob = (os.path.getsize(self._blob_path)
                    if os.path.exists(self._blob_path) else -1)
            logger.info("session tier: clean-shutdown marker present but rejected "
                        "(%s; actual journal_bytes=%d blob_eof=%d); "
                        "full payload verification", recorded, len(journal), blob)
        if not fast:
            # Payload-crc validation: a record whose blob region no longer matches (holes
            # left by compaction, a torn append that advanced EOF) is dead - drop, never
            # resurrect.
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
                            ok = _match_payload_crc(os.pread(blob_fd, rec["n"],
                                                             rec["off"]),
                                                    rec.get("crc_alg"), rec["crc"])
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
            # Cap accounting must count only the live record set, never the blob-size
            # watermark: the blob carries holes (discards, previous generations), and a
            # watermark-seeded _ssd_used over-evicts or refuses the entire next flush
            # (P3 iron Finding 1). Both boot paths seed identically here; _blob_eof
            # stays the never-decreased watermark - hole-safe record reservation.
            live = sum((s.l2_n + _BLK - 1) // _BLK * _BLK
                       for s in self._segments.values() if s.in_l2)
            # _blob_eof is the never-decreased watermark: exact dead even on a re-replay.
            self._dead_bytes = max(0, self._blob_eof - live)
            self._ssd_used = live
        # Moved here from __init__: before replay only the blob-size watermark exists,
        # after the reseed above _ssd_used is the live figure both paths must report.
        logger.info("session tier on: L1=%.2f GiB, L2 dir=%s cap=%.2f GiB used=%.2f GiB",
                    self.ram_bytes / 2**30, self.dir, self.ssd_bytes / 2**30,
                    self._ssd_used / 2**30)
        if records:
            if fast:
                logger.info("session tier: replay fast-path (clean shutdown marker), "
                            "%d records", len(records))
            else:
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
            # The rewrite moves both files: a marker a previous shutdown left behind no
            # longer describes this state the moment the rewrite starts.
            self.invalidate_shutdown_marker()
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
            self._blob_fd = os.open(self._blob_path, os.O_WRONLY | os.O_CREAT, 0o644)
            self._rewrite_journal(keep)
            # The rename above orphaned the open journal fd: it still points at the old
            # (now unlinked) inode, so subsequent appends would be lost at reboot. Reopen
            # the fresh file - same O_APPEND discipline as at __init__.
            os.close(self._journal_fd)
            self._journal_fd = os.open(self._journal_path,
                                       os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
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
