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
import hashlib
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
# Shrinking compact (P9) two-phase-commit sidecars: the staged pair (.new), the old
# inode anchors (.old) and the token that arms boot-time recovery of an interrupted swap.
_SWAP_TOKEN_NAME = "swap.token"
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
# Iron A/B overrides for the two flush knobs; read at the point of use, applied only
# when set, so tests patching the constants above keep working (P1 pool-pool precedent).
_FLUSH_WRITERS_ENV = "FREETOKEN_FLUSH_WRITERS"
_FLUSH_GROUP_BYTES_ENV = "FREETOKEN_FLUSH_GROUP_BYTES"
_FLUSH_GROUP_BYTES_MIN = 1 << 20
# pwritev iovec cap per call (Linux IOV_MAX is 1024) and the shared zero padding tail
# source for the reserved-but-unwritten span remainder.
_PWRITEV_MAX_IOV = 512
_ZERO_BLK = bytes(_BLK)
# P9 compact copy: parallel pread->pwrite workers (P4 pipeline precedent) and the
# per-worker chunk size; FREETOKEN_COMPACT_WRITERS overrides the workers (P6 pattern).
_COMPACT_WRITERS = 8
_COMPACT_CHUNK = 32 << 20
_COMPACT_WRITERS_ENV = "FREETOKEN_COMPACT_WRITERS"
# L2 extent dedup: an extent list longer than this falls back to a private copy. The
# bound exists for pathological chains only - prefix sharing grows the list linearly
# with boundary count (boundary k references k-1 page extents + own tail + snaps), and
# a tight cap would defeat the dedup on exactly the many-boundary sessions it targets.
_MAX_RECORD_EXTENTS = 64


def _flush_writers(requested: int) -> int:
    """Concurrent pwritev bursts per flush group; FREETOKEN_FLUSH_WRITERS overrides for
    the iron A/B sweep. Unset -> the requested default (a patched constant); garbage or
    < 1 -> the default with a warning, never a silent clamp. A whitespace-only value
    counts as unset (silently). Resolve ONCE per flush and pass down."""
    raw = os.environ.get(_FLUSH_WRITERS_ENV, "").strip()
    if not raw:
        return requested
    try:
        wanted = int(raw)
    except ValueError:
        logger.warning("ignoring non-integer %s=%r; using flush writers=%d",
                       _FLUSH_WRITERS_ENV, raw, requested)
        return requested
    if wanted < 1:
        logger.warning("%s=%r below minimum 1; using flush writers=%d",
                       _FLUSH_WRITERS_ENV, raw, requested)
        return requested
    return wanted


def _flush_group_bytes(requested: int) -> int:
    """Target volume of one flush group; FREETOKEN_FLUSH_GROUP_BYTES overrides for the
    iron A/B sweep. Unset -> the requested default (a patched constant); garbage or
    < 1 MiB -> the default with a warning, never a silent clamp. A whitespace-only
    value counts as unset (silently)."""
    raw = os.environ.get(_FLUSH_GROUP_BYTES_ENV, "").strip()
    if not raw:
        return requested
    try:
        wanted = int(raw)
    except ValueError:
        logger.warning("ignoring non-integer %s=%r; using flush group_bytes=%d",
                       _FLUSH_GROUP_BYTES_ENV, raw, requested)
        return requested
    if wanted < _FLUSH_GROUP_BYTES_MIN:
        logger.warning("%s=%r below minimum %d; using flush group_bytes=%d",
                       _FLUSH_GROUP_BYTES_ENV, raw, _FLUSH_GROUP_BYTES_MIN, requested)
        return requested
    return wanted


def _compact_writers(requested: int) -> int:
    """Concurrent pread->pwrite workers of the P9 compact copy; FREETOKEN_COMPACT_WRITERS
    overrides for the iron A/B sweep. Unset -> the requested default; garbage or < 1 ->
    the default with a warning; > 64 -> clamped to 64 with a warning (P1 precedent: a
    nonsense override must not fork a thread army). Resolve ONCE per compact."""
    raw = os.environ.get(_COMPACT_WRITERS_ENV, "").strip()
    if not raw:
        return requested
    try:
        wanted = int(raw)
    except ValueError:
        logger.warning("ignoring non-integer %s=%r; using compact writers=%d",
                       _COMPACT_WRITERS_ENV, raw, requested)
        return requested
    if wanted < 1:
        logger.warning("%s=%r below minimum 1; using compact writers=%d",
                       _COMPACT_WRITERS_ENV, raw, requested)
        return requested
    if wanted > 64:
        logger.warning("%s=%r above maximum 64; clamping compact writers=64",
                       _COMPACT_WRITERS_ENV, raw)
        wanted = 64
    return wanted

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


class _FlushOverflow:
    """Flush-local tally of cap-driven L2 evictions for one flush_live phase. Lives on
    the flush pipeline, deliberately NOT in self._counters: the builder thread counts
    evictions outside the store lock, and a fresh per-flush object with a single writer
    per field (builder thread: records/bytes, flush thread: admitted) cannot lose an
    increment; flush_live reads it only after _flush_groups joined the builder."""

    def __init__(self) -> None:
        self.records = 0       # L2 records discarded to make room for this flush
        self.bytes = 0         # padded blob span those discards freed
        self.admitted = 0      # payload bytes this flush journaled (the log's Y)


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
    # L2 residency: blob record range (unpadded length) of the PRIMARY (first) extent;
    # extents lists every blob span the record consumes, in logical payload order
    # (own pages then own snaps, pages|snaps seam always extent-aligned). Empty while
    # the segment is L1-only; assigned before the journal append, immutable after.
    blob: str = ""
    l2_off: int = -1
    l2_n: int = 0
    extents: list = field(default_factory=list)   # [(off, n, crc, alg), ...]
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
    # L2 extent dedup: skip = page count covered by shared extents (no iovecs, no
    # bytes); shares = [(off, n, src_seg_id)] (src None = committed in_l2 ancestor);
    # deps = share-source seg ids, committed + same-flush (a failed source skips this
    # record too); pins = those sources' refs pins, taken at build and released by
    # _release_stage; extents = finalized [(off, n, crc, alg), ...] for the record.
    skip: int = 0
    shares: list = field(default_factory=list)
    deps: set = field(default_factory=set)
    pins: list = field(default_factory=list)
    extents: list | None = None


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
        # L2 extent refcounts, keyed by exact blob span (off, n): [refs, crc, alg].
        # Derived at replay from the surviving record set, acquired at journal time,
        # released in _discard; the table exists even when inert (same reason as the
        # counters) and is rebuilt from scratch by replay and post-compact.
        self._extents: dict[tuple[int, int], list] = {}
        # P8: record-reclaimable dead ledger - journal records a compact() would
        # actually drop, plus their padded bytes. Kept separate from _dead_bytes, whose
        # watermark seed carries physical holes no record-based compact can reclaim.
        self._dead_records = 0
        self._dead_record_bytes = 0
        # P13: dead records WITHOUT a durable tombstone in the journal. Only these can
        # resurrect on a fast-path boot (the marker skips payload verification), so they
        # alone block the shutdown marker and force the shutdown compact rewrite.
        self._marker_blockers = 0
        self._counters: Counter = Counter(offers_ok=0, offers_rej=0, offers_dedup=0,
                                          probe_hit=0,
                                          probe_miss=0, probe_midspan_only=0,
                                          probe_snapfree_skip=0,
                                          note_match=0, restore_l1=0,
                                          restore_l2=0, restore_ok=0, restore_refused=0,
                                          evictions=0, demotions=0,
                                          discards=0, tombstones=0, prefetch_begin=0,
                                          extents_shared=0,
                                          prefetch_adopt=0,
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
        # P5 overflow observability: set around _flush_groups only (see _FlushOverflow).
        # None on every non-flush path, so runtime demotes/evictions never log overflow.
        self._flush_overflow: _FlushOverflow | None = None
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
        # P9 two-phase commit: True from the arm call until the commit fully lands -
        # an OSError inside that window must NEVER lead to temp cleanup (the token and
        # anchors on disk are the only recovery evidence; see the compact except path).
        self._swap_armed = False
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
              stamp: bool = True,
              boundary_exact: bool = False) -> tuple[int, TierHandle] | None:
        """Deepest match: walk the prompt's chain keys back to front; an equal key at
        depth d implies an equal prefix [0:d] (rolling chain keys). Ties go to the most
        recently validated. stamp=False is the idle prefetch probe: no LRU refresh
        (speculative interest must not outrank real matches) and its own counters.
        boundary_exact=True (hybrid managers) serves only segments whose boundary sits
        EXACTLY at d: a mid-span match has no live GDN snapshot at d, so the restore tail
        would refuse it and mask shallower servable boundaries - the walk skips those
        depths instead. A boundary-exact segment with no snapshot bytes (a KV-only
        tip/spare offer left by a restore-finish adoption) is skipped for the same
        reason - the presence-only snapshot predicate _tier_offer's under-divergence
        guard uses; each such skip counts probe_snapfree_skip (the idle prefetch
        walk, stamp=False, counts its skips there too). Snapshots are still
        the tail gate's call (state can change between probe and restore). Candidate
        depths are the live segments' boundary
        lengths descending, not every page depth: nothing else can win, so the walk
        jumps straight over the mid-span interior. A walk that saw mid-span matches but
        no boundary-exact candidate counts one miss plus probe_midspan_only (the
        M1-class 'consulted, unservable' signal), never a hit."""
        if not self.enabled:
            return None
        now = time.monotonic_ns()
        hit_key = "probe_hit" if stamp else "prefetch_probe_hit"
        miss_key = "probe_miss" if stamp else "prefetch_probe_miss"
        with self._lock:
            if boundary_exact:
                try:
                    segs = list(self._segments.values())
                except RuntimeError:
                    # builder threads discard lock-free (account lock only): retry once
                    # on the fresh dict instead of adding a lock to the builder path
                    segs = list(self._segments.values())
                depths = sorted({s.boundary_len for s in segs
                                 if s.boundary_len <= len(page_hashes)}, reverse=True)
            else:
                depths = range(len(page_hashes), 0, -1)
            best = None
            saw_midspan = False
            for d in depths:
                for seg_id, key_depth in self._index.get(page_hashes[d - 1], []):
                    if key_depth != d:
                        continue
                    seg = self._segments.get(seg_id)
                    if seg is None or seg.boundary_len < d:
                        continue
                    if boundary_exact and seg.boundary_len != d:
                        saw_midspan = True     # servable for KV-only, never for hybrid
                        continue
                    if boundary_exact and not any(seg.snap_lens):
                        # snapshot-free: the snap gate would refuse it anyway
                        self._counters["probe_snapfree_skip"] += 1
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
            if boundary_exact and saw_midspan:
                self._counters["probe_midspan_only"] += 1
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
            self.maybe_compact()   # dead-record gate: two int compares when below
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

    # ------------------------------------------------------------ l2 extent dedup

    def _extent_acquire(self, off: int, n: int, crc, alg, pad: int) -> bool:
        """One more record consuming the blob span (off, n); the first consumer creates
        the table entry. Called with the journal append already durable. Returns True
        when the span was already referenced (a dedup event, not first ownership).
        `pad` is the extent's exclusive share of its physical span's block padding:
        0 for interior extents, the span tail for the last one - so summed padded
        sizes stay EXACT whether extents are contiguous or alone in a block."""
        ent = self._extents.get((off, n))
        if ent is None:
            ent = self._extents[(off, n)] = [0, crc, alg, pad]  # [refs, crc, alg, pad]
        ent[0] += 1
        return ent[0] > 1

    def _extent_release(self, off: int, n: int) -> int:
        """Drop one reference; the last one moves the extent's exact bytes (n + its pad
        share) from live capacity to reclaimable dead. Returns the bytes freed (0 while
        the span stays referenced by another record - capacity truthfulness under
        sharing)."""
        ent = self._extents.get((off, n))
        if ent is None:
            return 0
        ent[0] -= 1
        if ent[0] <= 0:
            del self._extents[(off, n)]
            return n + ent[3]
        return 0

    def _plan_share(self, seg: _Segment) -> tuple[list, list, int | None]:
        """Deepest prefix-proven same-chain in_l2 ancestor whose page region is extent
        aligned: its page extents are byte-identical to this segment's pages [0:M) (an
        equal chain key at depth d implies an equal page prefix - probe's store
        invariant), so they are REFERENCED instead of re-appended. Returns (shares,
        covered page count, source seg id) with shares as (off, n, src_seg_id=None) -
        the crc is resolved from the extent table at journal time, and the source seg
        id lets the flush builder pin the ancestor for the build->journal window.
        Legacy mixed spans (pages and
        snaps in one extent), crc-less ancestors and page_lens mismatches contribute
        nothing: an extent is never split, and a reference must carry a verifiable crc."""
        try:
            segs = list(self._segments.values())
        except RuntimeError:
            # builder threads evict lock-free (account lock only): retry once on the
            # fresh dict instead of adding a lock to the build path (probe precedent)
            segs = list(self._segments.values())
        best = None
        for s in segs:
            if s.seg_id == seg.seg_id or not s.in_l2 or not s.extents:
                continue
            if not self._is_prefix(s.page_keys, seg.page_keys):
                continue
            if best is None or len(s.page_keys) > len(best.page_keys):
                best = s
        if best is None or seg.page_lens[:best.boundary_len] != best.page_lens:
            return [], 0, None
        region = sum(best.page_lens)
        shares: list[tuple[int, int, int | None]] = []
        covered = 0
        for ext in best.extents:
            if covered >= region:
                break
            if covered + ext[1] > region or ext[2] is None:
                return [], 0, None     # pages|snaps seam not extent-aligned / no crc
            shares.append((ext[0], ext[1], None))
            covered += ext[1]
        if covered != region:
            return [], 0, None
        extra = 1 + (1 if sum(seg.snap_lens) else 0)   # tail-pages extent + snaps extent
        if len(shares) + extra > _MAX_RECORD_EXTENTS:
            return [], 0, None           # safety valve: private copy above the cap
        return shares, best.boundary_len, best.seg_id

    @staticmethod
    def _chain_pool_crc(pool, offs, lens, alg: str) -> int:
        """Chained payload crc over pool slices (per-page/per-snap granularity) - the
        per-extent crcs of a flush-built record are computed without any byte copy."""
        crc = 0
        mm = pool.mm
        for off, ln in zip(offs, lens):
            if ln:
                data = mm[off:off + ln]
                crc = (crc32c.crc32c(data, crc) if alg == _CRC32C_ALG
                       else zlib.crc32(data, crc))
        return crc

    def _seg_logical_crc(self, seg: _Segment, alg: str = _CRC32C_ALG) -> int:
        """Record-level crc over the FULL logical payload (pages then snaps) - unchanged
        semantics under sharing: old binaries verify it against the primary extent only
        and drop multi-extent records; single-extent records verify exactly."""
        offs = list(seg.l1_page_offs)
        lens = list(seg.page_lens)
        for off, ln in zip(seg.l1_snap_offs, seg.snap_lens):
            if ln:
                offs.append(off)
                lens.append(ln)
        return self._chain_pool_crc(self._pool, offs, lens, alg)

    def _private_extents(self, seg: _Segment, skip: int, off: int, private: bytes,
                         crc_tail: int | None = None,
                         crc_snaps: int | None = None,
                         snaps_n: int | None = None) -> list:
        """Extent list for a freshly appended span: [tail pages][snaps] as separate
        extents so the pages|snaps seam stays extent-aligned for deeper sharing.
        snaps_n overrides len(private) - tail (the flush pipeline knows the split
        without holding the private bytes). The physical span stays contiguous
        ([tail][snaps][block pad]) - the LAST extent carries the span's padding as its
        exclusive share, interior extents carry 0, so summed extent sizes equal the
        reserved bytes exactly (no per-extent block rounding double-count)."""
        tail = sum(seg.page_lens[skip:])
        if snaps_n is None:
            snaps_n = max(0, len(private) - tail)
        span_pad = -(tail + snaps_n) % _BLK
        out = []
        if tail:
            if crc_tail is None:
                crc_tail = _calc_payload_crc(private[:tail], _CRC32C_ALG)
            out.append((off, tail, crc_tail, _CRC32C_ALG, 0))
        if snaps_n:
            if crc_snaps is None:
                crc_snaps = _calc_payload_crc(private[tail:], _CRC32C_ALG)
            out.append((off + tail, snaps_n, crc_snaps, _CRC32C_ALG, 0))
        if out:
            last = out[-1]
            out[-1] = (last[0], last[1], last[2], last[3], span_pad)
        return out

    def _read_l2_payload(self, seg: _Segment) -> bytes:
        """Logical payload of an L2 segment: the ordered extents concatenated (one
        O_DIRECT read per extent; a legacy record is its single extent)."""
        if not seg.extents:
            return self._blob_read(seg.blob, seg.l2_off, seg.l2_n)
        total = sum(ext[1] for ext in seg.extents)
        buf = bytearray(total)
        pos = 0
        for ext in seg.extents:
            if ext[1]:
                buf[pos:pos + ext[1]] = self._blob_read(seg.blob, ext[0], ext[1])
            pos += ext[1]
        return bytes(buf)

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
            # the evictor sizes by the PADDED need (not the unpadded one): with extent
            # sharing a discard can free less than a victim's nominal span, so the loop
            # must keep evicting until the padded requirement genuinely fits
            self._evict_oldest_l2(padded, pending)
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

    def _journal_and_account(self, seg: _Segment,
                             extents: list[tuple[int, int, int, str]], padded: int,
                             crc: int, crc_alg: str = _CRC32C_ALG,
                             off: int | None = None) -> bool:
        """Journal one durable record and take the capacity accounting. `extents` is the
        record's ordered extent list; the journal's off/n are its PRIMARY (first)
        extent - the identity compact keep-sets and P13 tombstones match on. `padded`
        counts only the freshly appended span (shared extents add no bytes - capacity
        truthfulness under dedup). False = the blob bytes are durable but unrecoverable
        across a boot: abort cleanly (the segment keeps its L1 residency; no journal
        entry, no accounting, no extent refs)."""
        seg.blob = _BLOB_NAME
        seg.extents = list(extents)
        if seg.extents:
            seg.l2_off, seg.l2_n = seg.extents[0][0], seg.extents[0][1]
        else:
            seg.l2_off, seg.l2_n = (off if off is not None else 0), 0
        if not self._journal_append(seg, crc, crc_alg):
            seg.extents = []
            return False
        with self._account_lock:      # racing the builder thread's cap evictions
            for ext in seg.extents:
                if self._extent_acquire(ext[0], ext[1], ext[2], ext[3], ext[4]):
                    self._counters["extents_shared"] += 1
            self._ssd_used += padded
        return True

    def _write_blob(self, seg: _Segment, payload: bytes | None) -> bool:
        """Append one blob record (padded to _BLK for the O_DIRECT reader): pwrite at the
        reserved offset, ONE fdatasync, then the journal record - the durable order is
        what makes replay crash-safe. Every failure path returns False: no journal entry,
        no capacity accounting. Single-segment path (runtime offer/evict pressure); the
        shutdown flush batches segments through _write_group instead.
        L2 extent dedup (briefs/l2-snapshot-dedup.md): the deepest prefix-proven
        same-chain in_l2 ancestor's page extents are referenced and only the tail pages
        + snapshots are appended; a referenced span is already durable (its owner's
        record was journaled after an fsync)."""
        payload = payload if payload is not None else b""
        shares, skip, _src = self._plan_share(seg)
        private = payload[sum(seg.page_lens[:skip]):]
        need = len(private)
        r = self._reserve_blob_span(need)
        if r is None:
            return False
        off, padded = r
        view = memoryview(private + b"\0" * (padded - need))
        try:
            self._pwrite_span(off, view)
            os.fdatasync(self._blob_fd)
        except OSError as e:
            logger.warning("session tier: blob append failed (%s); record aborted", e)
            return False
        extents = self._private_extents(seg, skip, off, private)
        # resolve the shared extents' crc/alg/pad from the extent table (the owner
        # registered them when its own record was journaled)
        resolved = []
        for soff, sn, _src in shares:
            ent = self._extents.get((soff, sn))
            resolved.append((soff, sn, ent[1] if ent else None,
                             ent[2] if ent else None, ent[3] if ent else 0))
        return self._journal_and_account(seg, resolved + extents, padded,
                                         _calc_payload_crc(payload, _CRC32C_ALG),
                                         off=off)

    def _seg_iovecs(self, seg: _Segment, padded: int, skip_pages: int = 0) -> list:
        """Zero-copy record layout for the freshly appended span: payload iovecs straight
        from the L1 pool mmap (tail pages [skip_pages:], then snapshots) plus the zero
        padding tail; shared extents contribute no iovecs - their bytes are already in
        the blob. The pool regions stay stable until the write completes because
        flush_live holds the store lock - no offer/evict mutation can touch them
        mid-flight."""
        base = memoryview(self._pool.mm)
        bufs = [base[off:off + ln]
                for off, ln in zip(seg.l1_page_offs[skip_pages:],
                                   seg.page_lens[skip_pages:])]
        bufs.extend(base[off:off + ln]
                    for off, ln in zip(seg.l1_snap_offs, seg.snap_lens) if ln)
        pad = padded - sum(seg.page_lens[skip_pages:]) - sum(seg.snap_lens)
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
        reserve and journal looks in_l2 to the evictor) and the share-source pins come
        off (they kept the sources un-evictable from build to journal). Idempotent via
        the entry's released flag: the writer's finally and a BaseException sweep in
        _flush_groups can both reach the same stage."""
        for en in stage:
            if en.released:
                continue
            en.released = True
            pending.release(en.padded)
            en.seg.refs -= 1
            self._drop_share_pins(en.pins)
            en.pins = []

    def _pin_share_sources(self, deps: set) -> list:
        """refs-pin every share source of a pending stage entry: the journal runs on
        the writer thread AFTER this build, so an unpinned refs==0 source can be
        discarded by the builder's own cap eviction in between - its extent entries
        get released and the dependent's journal silently re-creates them (capacity
        drift, lost extents_shared, unverifiable extents). The pin makes the source
        invisible to the refs==0 victim scan for exactly the build->journal window.
        Released by _release_stage (or _drop_share_pins on non-admission)."""
        pins = []
        for sid in deps:
            src = self._segments.get(sid)
            if src is not None:
                src.refs += 1
                pins.append(src)
        return pins

    def _drop_share_pins(self, pins: list) -> None:
        for src in pins:
            src.refs -= 1

    def _plan_share_flush(self, seg: _Segment,
                          plan: dict) -> tuple[list, int, set]:
        """_plan_share over committed in_l2 segments plus earlier entries of the same
        flush (the plan map). An earlier entry's planned extents are durable before ANY
        journal record of their group (group fdatasync order), and the writer thread is
        sequential - group N's bytes are durable before group N+1's journal appends - so
        referencing them keeps the crash contract. A planned-but-failed source is
        handled by the dep gate in _write_group (the dependent record is skipped too).
        Returns (shares as (off, n, src_seg_id), covered page count, source seg ids).
        The committed ancestor's id rides in deps too: the builder pins every source
        for the build->journal window. The deepest ancestor wins: a committed ancestor
        shallower than the deepest plan-mate is dropped, a deeper committed one keeps
        the plan-mate out."""
        shares, skip, committed_src = self._plan_share(seg)
        deps: set = set()
        if committed_src is not None:
            deps.add(committed_src)
        best = None
        for pe in plan.values():
            if not self._is_prefix(pe["page_keys"], seg.page_keys):
                continue
            if best is None or len(pe["page_keys"]) > len(best["page_keys"]):
                best = pe
        if best is not None:
            m = len(best["page_keys"])
            if m > skip and seg.page_lens[:m] == best["page_lens"]:
                region = sum(best["page_lens"])
                covered = 0
                plan_shares = []
                aligned = True
                for ext in best["extents"]:
                    if covered >= region:
                        break
                    if covered + ext[1] > region:
                        aligned = False      # seam not extent-aligned: keep committed
                        break
                    plan_shares.append((ext[0], ext[1], best["sid"]))
                    covered += ext[1]
                if aligned and covered == region:
                    shares, skip = plan_shares, m
        extra = 1 + (1 if sum(seg.snap_lens) else 0)
        if len(shares) + extra > _MAX_RECORD_EXTENTS:
            return [], 0, set()
        deps.update(src for _, _, src in shares if src is not None)
        return shares, skip, deps

    def _build_group_stage(self, group: list[_Segment], pending: _PendingVolume,
                           index: int = 0,
                           plan: dict | None = None) -> list[_GroupEntry] | None:
        """Runs on the flush builder thread while the PREVIOUS group is being written:
        reserve every segment's span (cap check against the pipeline-wide pending
        volume, evicting oldest L2 under pressure - only the PRIVATE volume is
        reserved: shared extents append no bytes), pin each admitted segment with a
        refs claim (a reserved-but-unjournaled span makes the segment look in_l2 to
        the evictor), then lift each record's zero-copy iovecs and payload crc off the
        L1 pool. No payload byte is ever copied. Extent planning (plan map keyed by
        seg_id: geometry + extent slots, crcs patched in the second loop) lets a later
        entry reference earlier entries of the same flush. Returns the stage list, or
        None when nothing was admitted or the build failed - the segments keep their
        L1 residency and the group is aborted with no journal side effect."""
        stage: list[_GroupEntry] = []
        plan = plan if plan is not None else {}
        cur_seg, cur_need = group[0], 0
        cur_pins: list = []                  # pinned sources of the in-flight seg
        try:
            for seg in group:
                cur_seg, cur_need = seg, sum(seg.page_lens) + sum(seg.snap_lens)
                shares, skip, deps = self._plan_share_flush(seg, plan)
                # Pin the share sources BEFORE the reserve: the reserve's eviction
                # sweep and every later build's sweep must see them pinned, or the
                # journal on the writer thread re-creates their released extents.
                cur_pins = self._pin_share_sources(deps)
                private_need = cur_need - sum(seg.page_lens[:skip])
                r = self._reserve_blob_span(private_need, pending.value())
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
                    r = self._reserve_blob_span(private_need, 0)
                if r is None:
                    logger.warning("session tier: L2 cap reached, cannot demote seg %d",
                                   seg.seg_id)
                    self._drop_share_pins(cur_pins)
                    cur_pins = []
                    continue
                pending.add(r[1])
                seg.refs += 1                     # evictor guard, released in _write_group
                en = _GroupEntry(seg, r[0], cur_need, r[1])
                en.skip, en.shares, en.deps = skip, shares, deps
                en.pins, cur_pins = cur_pins, []
                stage.append(en)
                # Plan entry: geometry + extent slots for later entries' share walks.
                # The extent (off, n) list is final here (offsets are reserved); crc
                # slots are patched with real values in the crc loop below. The pad
                # share (last extent of the appended span) is fixed here too.
                tail_pages = cur_need - sum(seg.page_lens[:skip]) - sum(seg.snap_lens)
                snaps_n = sum(seg.snap_lens)
                span_pad = -(tail_pages + snaps_n) % _BLK
                slots = [(off, n, None, None, 0) for off, n, _ in shares]
                if tail_pages:
                    slots.append((r[0], tail_pages, None, None, 0))
                if snaps_n:
                    slots.append((r[0] + tail_pages, snaps_n, None, None, 0))
                if slots:
                    last = slots[-1]
                    slots[-1] = last[:4] + (span_pad,)
                plan[seg.seg_id] = {"sid": seg.seg_id, "page_keys": seg.page_keys,
                                    "page_lens": seg.page_lens, "extents": slots}
            if not stage:
                return None
            for en in stage:
                cur_seg, cur_need = en.seg, en.need
                en.bufs = self._seg_iovecs(en.seg, en.padded, en.skip)
                en.crc = self._seg_logical_crc(en.seg)
                en.extents = self._stage_extents(en, plan)
                plan[en.seg.seg_id]["extents"] = en.extents
        except Exception as e:                    # build failure: abort the whole group
            logger.warning(
                "session tier: flush group #%d build failed at seg %d (%d bytes, %d of "
                "%d admitted): %s; group aborted",
                index, cur_seg.seg_id, cur_need, len(stage), len(group), e)
            for en in stage:                      # doomed spans must not be shareable
                plan.pop(en.seg.seg_id, None)
            self._drop_share_pins(cur_pins)
            self._release_stage(stage, pending)
            return None
        except BaseException:                     # KI/SystemExit: unpin, then propagate
            for en in stage:
                plan.pop(en.seg.seg_id, None)
            self._drop_share_pins(cur_pins)
            self._release_stage(stage, pending)
            raise
        return stage

    def _stage_extents(self, en: _GroupEntry, plan: dict) -> list:
        """Finalize a stage entry's extent list: resolve each planned share's crc (from
        the extent table for committed ancestors, from the source entry's finalized
        slots for same-flush sources - all finalized before any journal of this flush
        can run), then append the freshly appended span's own extents. Runs on the
        builder thread inside the build's crc loop, strictly after every share source's
        own finalize."""
        resolved = []
        for off, n, src in en.shares:
            crc = alg = None
            pad = 0
            if src is None:
                ent = self._extents.get((off, n))
                if ent is not None:
                    crc, alg, pad = ent[1], ent[2], ent[3]
            else:
                for ext in plan.get(src, {}).get("extents") or []:
                    if ext[0] == off and ext[1] == n:
                        crc, alg, pad = ext[2], ext[3], ext[4]
                        break
            resolved.append((off, n, crc, alg, pad))
        tail_pages = sum(en.seg.page_lens[en.skip:])
        snaps_off = en.off + tail_pages
        snaps_n = sum(en.seg.snap_lens)
        crc_tail = self._chain_pool_crc(self._pool, en.seg.l1_page_offs[en.skip:],
                                        en.seg.page_lens[en.skip:], _CRC32C_ALG)
        crc_snaps = (self._chain_pool_crc(self._pool, en.seg.l1_snap_offs,
                                          en.seg.snap_lens, _CRC32C_ALG)
                     if snaps_n else None)
        return resolved + self._private_extents(en.seg, en.skip, en.off, b"",
                                                crc_tail=crc_tail,
                                                crc_snaps=crc_snaps,
                                                snaps_n=snaps_n)

    def _pwrite_group(self, stage: list[_GroupEntry], writers: int) -> set[int]:
        """Parallel pwritev bursts at the reserved offsets; writers comes resolved once
        per flush from the caller (see flush_live). Returns the indices whose write
        failed (their spans stay holes behind the watermark)."""
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
            with ThreadPoolExecutor(max_workers=min(writers, len(stage))) as ex:
                list(ex.map(_write_one, range(len(stage))))
        return failed

    def _write_group(self, stage: list[_GroupEntry], pending: _PendingVolume,
                     writers: int, failed_sids: set) -> int:
        """Write one built group: parallel pwritev bursts (writers resolved once per
        flush) at the reserved offsets, ONE
        fdatasync for the whole group, then the group's journal records. Durable-order
        invariant unchanged - a journal record still appears only after the blob bytes
        of its group are durable - but the crash loss window widens from one segment to
        the whole group: a SIGKILL between the fdatasync and the journal appends leaves
        the group's bytes durable and unjournaled, and replay drops that blob tail
        (segments stay in L1 in a process that survives). The pipeline-wide pending
        volume stays counted until this group settles, so a concurrent builder always
        sees a conservative cap budget. failed_sids accumulates the seg ids whose
        record did not land (failed write, failed journal append, or a dep gate skip):
        a record referencing a source that never became durable+journaled is skipped -
        its segment keeps its L1 residency for the next flush instead of journaling a
        reference replay would have to drop. Returns the demoted count."""
        if not stage:
            return 0
        count = 0
        skipped = 0
        try:
            failed = self._pwrite_group(stage, writers)
            # Group fdatasync: every byte written above is durable BEFORE any journal
            # record of the group appears.
            try:
                os.fdatasync(self._blob_fd)
            except OSError as e:
                logger.warning("session tier: blob fdatasync failed (%s); group aborted", e)
                return 0
            for i, en in enumerate(stage):
                if i in failed:
                    failed_sids.add(en.seg.seg_id)
                    continue
                if en.deps and en.deps & failed_sids:
                    # transitive: this record is skipped, so its own id blocks its
                    # dependents further down the group/flush chain
                    failed_sids.add(en.seg.seg_id)
                    skipped += 1
                    continue
                if self._journal_and_account(en.seg, en.extents, en.padded, en.crc):
                    self._release_l1(en.seg)
                    self._counters["demotions"] += 1
                    count += 1
                    ov = self._flush_overflow
                    if ov is not None:            # the Y of the overflow log line
                        ov.admitted += en.need
                else:
                    # the record did not land: block this seg's own dependents too,
                    # or they journal references to a never-journaled source span
                    failed_sids.add(en.seg.seg_id)
        finally:
            # Release the pipeline's claims whatever happened above (journaled, aborted
            # or raised): the pending volume returns to the cap budget, the refs pin
            # protects nothing once the journal attempt is over.
            self._release_stage(stage, pending)
        if skipped:
            logger.warning("session tier: flush skipped %d records whose extent "
                           "sources failed; segments stay in L1", skipped)
        return count

    def _flush_groups(self, groups: list[list[_Segment]], writers: int) -> int:
        """Group pipeline (double buffering): group N+1 is built - spans reserved,
        iovecs and payload crcs lifted off the L1 pool - while group N is being
        pwritten, fdatasync'ed and journaled. writers is resolved once per flush by
        the caller and passed down: one env warning per flush, and the logged config
        matches the applied one. Cross-group reserve serialization goes
        through the shared pending volume, so the cap accounting stays exact no matter
        how many groups are in flight. A BaseException (KI/SystemExit) mid-pipeline
        releases every built-but-unsettled stage's claims and propagates unchanged."""
        if self._blob_fd < 0:
            return 0
        pending = _PendingVolume()
        plan: dict = {}                       # flush-local extent planning map
        failed_sids: set = set()              # seg ids whose record did not land
        if len(groups) == 1:
            stage = self._build_group_stage(groups[0], pending, 0, plan)
            return (self._write_group(stage, pending, writers, failed_sids)
                    if stage else 0)
        count = 0
        built: list[_GroupEntry] | None = None
        futs: list[Future] = []
        try:
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="flush-build") as ex:
                fut = ex.submit(self._build_group_stage, groups[0], pending, 0, plan)
                futs.append(fut)
                for gi, group in enumerate(groups[1:], start=1):
                    nxt = ex.submit(self._build_group_stage, group, pending, gi, plan)
                    futs.append(nxt)
                    built = fut.result()
                    if built:
                        count += self._write_group(built, pending, writers, failed_sids)
                    fut = nxt
                built = fut.result()
                if built:
                    count += self._write_group(built, pending, writers, failed_sids)
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
        self.maybe_compact()   # discard pressure is the other dead-record check point
        return count

    def _evict_oldest_l2(self, keep: int, pending: int = 0) -> None:
        l2 = [s for s in self._segments.values() if s.in_l2 and s.refs == 0]
        while self.ssd_bytes and self._ssd_used + pending + keep > self.ssd_bytes and l2:
            victim = min(l2, key=lambda s: (s.last_validation, s.seg_id))
            freed = self._discard(victim)
            ov = self._flush_overflow          # flush-phase tally; None on runtime paths
            if ov is not None:
                ov.records += 1
                ov.bytes += freed
            l2.remove(victim)

    def _discard(self, seg: _Segment) -> int:
        """Append-only blob: the discarded spans become zero holes, reclaimable only by
        compaction once the dead-record ledger trips (maybe_compact). The P13 tombstone
        makes the discard itself durable immediately (replay kills the victim on both
        boot paths), so the hole is pure space, never a resurrection. Under extent
        sharing only PRIVATE bytes free up: each consumed extent drops one reference,
        and a span still referenced by another record stays live (and stays counted in
        _ssd_used) - capacity truthfulness under dedup. Returns the padded bytes freed
        from the capacity accounting."""
        with self._account_lock:      # racing the writer thread's journal accounting
            freed = 0
            for ext in seg.extents:
                freed += self._extent_release(ext[0], ext[1])
            self._ssd_used -= freed
            self._dead_bytes += freed
            if seg.in_l2:             # its journal record just became compact-droppable
                self._dead_records += 1
                self._dead_record_bytes += freed
                # P13: durable NOW. Builder-thread evictions race the coordinator's
                # appends lock-free: atomic O_APPEND write + order-independent tomb filter.
                if self._append_tombstone(seg.path_key, seg.l2_off, seg.l2_n):
                    self._counters["tombstones"] += 1
                else:
                    # Not durable: the discard stays compact-durable only (pre-P13), so
                    # it blocks the marker and forces the shutdown compact rewrite.
                    self._marker_blockers += 1
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
        # A landed tombstone grows the journal, so the marker size checks would
        # self-invalidate on their own; this explicit drop covers the failed-append
        # case: no on-disk trace, a stale-valid marker would resurrect the segment.
        self.invalidate_shutdown_marker()
        return freed

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
                    raw = self._read_l2_payload(seg)
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

    def count_restore_outcome(self, ok: bool) -> None:
        """P0 telemetry seam for cache.py's restore tail: served vs refused outcomes.
        The attempt counters (restore_l1/l2) stay in restore() - they count store reads."""
        with self._lock:
            self._counters["restore_ok" if ok else "restore_refused"] += 1

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
                        raw = self._read_l2_payload(seg)
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
        ov: _FlushOverflow | None = None
        # Both knobs resolve ONCE per flush: the config line and every group's pwrite
        # executor must agree, and a bad env warns at most once per flush.
        group_bytes = _flush_group_bytes(_FLUSH_GROUP_BYTES)
        writers = _flush_writers(_FLUSH_WRITERS)
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
                if group and vol + padded > group_bytes:
                    groups.append(group)
                    group, vol = [], 0
                group.append(seg)
                vol += padded
            if group:
                groups.append(group)
            if groups:
                # One line per flush phase, not per group: the iron A/B needs the
                # effective config in the log to attribute a run to the right knob.
                logger.info("session tier: flush pipeline: writers=%d group_bytes=%d",
                            writers, group_bytes)
                ov = _FlushOverflow()
                self._flush_overflow = ov
                try:
                    count = self._flush_groups(groups, writers)
                finally:
                    self._flush_overflow = None
        if ov is not None and ov.records:
            # P5: one overflow line per flush phase, only when the cap forced L2
            # evictions; the flushed-N line below stays as is.
            logger.info("session tier: flush overflow: evicted %d records / %.1f MiB "
                        "of oldest L2 to admit %d bytes",
                        ov.records, ov.bytes / 2**20, ov.admitted)
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
        stale marker's recorded size.
        B3: a VALID marker with dead records still journaled (the compact was deferred
        by in-flight blob reads, or aborted before arm) would fast-path-boot those dead
        records back as LIVE segments (discard durability) AND seed the P8 ledger 0/0 -
        the zombies would then evade the next compact forever. No marker: the next boot
        takes the full path, seeds the honest ledger, and the next compact reclaims.
        P13: a dead record WITH a durable tombstone cannot fast-path-resurrect (replay
        filters tombstones before the fast-path ledger seed), so only UNtombstoned dead
        (_marker_blockers) block the marker - tombstoned dead stay out of the way."""
        if not self.enabled or not self._journal_path:
            return
        if self._marker_blockers:
            logger.warning("session tier: shutdown marker skipped, %d dead records "
                           "without a durable tombstone (compact deferred/aborted); "
                           "next boot re-verifies and the next compact reclaims them",
                           self._marker_blockers)
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

    def _journal_append(self, seg: _Segment, crc: int,
                        crc_alg: str = _CRC32C_ALG) -> bool:
        """Durable journal record for a stored segment. off/n are the PRIMARY (first)
        extent - the record identity compact keep-sets and P13 tombstones match on.
        "extents" (additive field, briefs/l2-snapshot-dedup.md) lists every blob span
        the record consumes in logical payload order; records without it (legacy
        directories) are one private span. The record-level crc stays the LOGICAL
        payload checksum: an old binary preading the primary region only mismatches it
        for multi-extent records and drops them (clean downgrade: capability loss, no
        corruption), while single-extent records verify and boot unchanged. The frame
        format and crc are untouched."""
        rec = {
            "path_key": seg.path_key.hex(), "blen": seg.boundary_len,
            "page_keys": [k.hex() for k in seg.page_keys],
            "page_lens": seg.page_lens, "snap_lens": seg.snap_lens,
            "blob": seg.blob, "off": seg.l2_off, "n": seg.l2_n, "ts": seg.last_validation,
            # payload checksum: replay drops records whose blob region no longer matches
            # (holes after compaction, torn/short appends) - the convergence mechanism.
            # crc_alg versions the algorithm: new records carry hardware CRC32C; records
            # without the field are legacy zlib-crc32 (old catalogs replay unchanged).
            # The journal frame crc below stays zlib: see the versioning note at the top.
            "crc": crc,
        }
        if seg.extents:
            rec["extents"] = [[o, n, c, a, p] for o, n, c, a, p in seg.extents]
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

    def _append_tombstone(self, path_key: bytes, off: int, n: int) -> bool:
        """P13: one fsync'ed tombstone record - the discard is durable immediately, no
        compact needed. The record carries the victim's exact survival key
        (path_key, off, n), the same identity compact()'s keep-set matches on; replay
        kills the matching earlier record on BOTH boot paths. False = not durable
        (a marker blocker; the shutdown compact falls back to the classic rewrite)."""
        rec = {"t": 1, "path_key": path_key.hex(), "off": off, "n": n}
        payload = json.dumps(rec, sort_keys=True).encode()
        try:
            os.write(self._journal_fd,
                     struct.pack("<II", len(payload), zlib.crc32(payload)) + payload)
            os.fsync(self._journal_fd)
        except OSError as e:
            logger.warning("session tier: tombstone append failed (%s); the discard "
                           "stays compact-durable only", e)
            return False
        return True

    def replay_journal(self) -> int:
        """Boot path: rebuild the L2 index from the journal. A truncated or torn tail
        (SIGKILL mid-append) stops replay at the last complete record. Idempotent: the
        L2 index is rebuilt from scratch each call."""
        if not self.enabled or not self._journal_path:
            return 0
        self._recover_swap()
        if not os.path.exists(self._journal_path):
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
        # P13 durable tombstones: a {"t":1} record kills the earlier live record with
        # the same (path_key, off, n) identity on BOTH boot paths - the marker fast
        # path included, so a valid marker can never resurrect a tombstoned discard
        # (the basis for the P13 marker/shutdown-compact optionality). Tombstones are
        # appended after their victims (append-only journal), but the filter is
        # order-independent anyway; the tombstone records themselves never become
        # segments. Old journals carry no "t" records and replay bit-identically.
        dropped, pdropped = 0, 0
        dropped_recs: list[dict] = []
        tomb_keys = {(r["path_key"], r["off"], r["n"]) for r in records if r.get("t")}
        if tomb_keys:
            victims = [r for r in records if not r.get("t")
                       and (r["path_key"], r["off"], r["n"]) in tomb_keys]
            records = [r for r in records if not r.get("t")
                       and (r["path_key"], r["off"], r["n"]) not in tomb_keys]
            dropped = len(victims)
            dropped_recs.extend(victims)
            logger.info("session tier: replay honored %d tombstones, %d records stay "
                        "dead (compact reclaims later)", len(tomb_keys), dropped)
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
        # P8 dead-record ledger seed: how many journal records the next compact() will
        # actually drop. The P13 tombstone victims above seed it on BOTH paths (they
        # stay in the journal file until a compact). Full path: payload-dropped records
        # stay in the journal file (replay never rewrites it) and the compact parser
        # still sees them; kept records all become segments.
        if not fast:
            # Payload-crc validation: a record whose blob region no longer matches (holes
            # left by compaction, a torn append that advanced EOF) is dead - drop, never
            # resurrect. Extent records validate each DISTINCT consumed span once (the
            # shared span is read one time no matter how many records reference it);
            # legacy records validate their single private region exactly as before.
            blob_fd = -1
            if self._blob_path and os.path.exists(self._blob_path):
                try:
                    blob_fd = os.open(self._blob_path, os.O_RDONLY)
                except OSError:
                    blob_fd = -1
            if blob_fd >= 0:
                kept, pdropped_recs = [], []
                ext_ok: dict[tuple[int, int], bool] = {}
                for rec in records:
                    ok = True
                    exts = rec.get("extents")
                    if exts is not None:
                        for ext in exts:
                            key = (ext[0], ext[1])
                            good = ext_ok.get(key)
                            if good is None:
                                ecrc = ext[2] if len(ext) > 2 else None
                                ealg = ext[3] if len(ext) > 3 else None
                                good = True
                                if ecrc is not None:
                                    try:
                                        good = _match_payload_crc(
                                            os.pread(blob_fd, ext[1], ext[0]),
                                            ealg, ecrc)
                                    except OSError:
                                        good = False
                                ext_ok[key] = good
                            if not good:
                                ok = False
                                break
                    elif "crc" in rec:
                        try:
                            ok = _match_payload_crc(os.pread(blob_fd, rec["n"],
                                                             rec["off"]),
                                                    rec.get("crc_alg"), rec["crc"])
                        except OSError:
                            ok = False
                    if ok:
                        kept.append(rec)
                    else:
                        pdropped_recs.append(rec)
                os.close(blob_fd)
                pdropped = len(pdropped_recs)
                records = kept
                dropped += pdropped
                dropped_recs.extend(pdropped_recs)
                if pdropped:
                    logger.info("session tier: replay dropped %d dead/torn records",
                                pdropped)
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
                exts = rec.get("extents")
                if exts is not None:
                    seg.extents = [(e[0], e[1],
                                    e[2] if len(e) > 2 else None,
                                    e[3] if len(e) > 3 else None,
                                    e[4] if len(e) > 4 else -e[1] % _BLK)
                                   for e in exts]
                else:
                    # legacy dual-read: one private span covering pages+snaps, verified
                    # by its whole-region crc exactly as before the extent format; the
                    # span's block padding is the extent's exclusive share
                    seg.extents = [(rec["off"], rec["n"], rec.get("crc"),
                                    rec.get("crc_alg"), -rec["n"] % _BLK)]
                seg.last_validation = rec["ts"]
                self._segments[seg.seg_id] = seg
                self._touch_index(seg)
            # Extent refcounts are DERIVED, never journaled: rebuilt from the surviving
            # record set, so replay converges to the same extent liveness from any
            # journal prefix (torn tails, tombstones, payload drops - idempotent).
            self._extents = {}
            for seg in self._segments.values():
                if seg.in_l2:
                    for ext in seg.extents:
                        self._extent_acquire(ext[0], ext[1], ext[2], ext[3], ext[4])
            # Cap accounting must count only UNIQUE live extent bytes, never the blob-size
            # watermark: the blob carries holes (discards, previous generations), and a
            # watermark-seeded _ssd_used over-evicts or refuses the entire next flush
            # (P3 iron Finding 1). With sharing, per-record sums would double-count the
            # shared spans - the "l2 used" figure is the unique bytes (each extent's n
            # plus its exclusive padding share).
            live = sum(key[1] + ent[3] for key, ent in self._extents.items())
            # _blob_eof is the never-decreased watermark: exact dead even on a re-replay.
            self._dead_bytes = max(0, self._blob_eof - live)
            self._ssd_used = live
            self._dead_records = dropped
            # P8 ledger, honest under sharing: a compact drops the dead RECORDS but only
            # reclaims the extents that lose their last reference with them.
            reclaimable = 0
            for rec in dropped_recs:
                exts = rec.get("extents")
                span = [(e[0], e[1], e[4] if len(e) > 4 else -e[1] % _BLK)
                        for e in exts] if exts is not None \
                    else [(rec["off"], rec["n"], -rec["n"] % _BLK)]
                for soff, sn, spad in span:
                    if (soff, sn) not in self._extents:
                        reclaimable += sn + spad
            self._dead_record_bytes = reclaimable
            # P13: payload-dead records carry no tombstone - they alone can resurrect on
            # a fast-path boot, so they alone block the marker and the shutdown-compact
            # skip. Tombstoned victims were filtered before this point on both paths.
            self._marker_blockers = pdropped
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
        """Dead-record-gated runtime compaction: fires only when the padded bytes of
        journal records a compact would actually drop exceed the constant share of the
        blob AND the blob is above the floor; two int compares (and no locking) on every
        check point. Returns dead records dropped (0 = no-op). The _dead_bytes watermark
        (physical holes: discards, previous generations) must NOT gate this - with zero
        dead records compact() is a no-op and every ok-offer would pay a full journal
        parse for nothing (P5-iron finding 1). The holes= stat (watermark-minus-live)
        resets to 0 after a compact, which since P9 physically reclaims them in the
        same pass. P13: tombstoned victims count into _dead_record_bytes (the replay
        seed and every discard), so journal growth from tombstones is bounded by this
        same gate - a tombstone costs ~100 journal bytes against its victim's padded
        span."""
        if not self.enabled or self._blob_fd < 0:
            return 0
        if self._blob_eof < _COMPACT_FLOOR_BYTES or \
                self._dead_record_bytes <= self._blob_eof * _COMPACT_DEAD_FRACTION:
            return 0
        dropped = self.compact()
        if dropped:
            logger.info("session tier: compaction dropped %d dead records; "
                        "l2 used %.2f GiB (blob eof %.2f GiB)",
                        dropped, self._ssd_used / 2**30, self._blob_eof / 2**30)
        return dropped

    def compact(self) -> int:
        """Shutdown-checkpoint log compaction (TASK.md Component 3) with the P9 physical
        shrink: the live segments are DENSELY repacked into blob.bin.new (dead regions
        reclaimed, file == live volume) and the journal is rewritten to the new offsets;
        the pair is swapped under a two-phase commit armed by swap.token, so a SIGKILL
        at any point converges at boot to either the old or the new state (see
        _recover_swap) - replay's payload-crc check stays the last resort. With zero
        dead records this is a no-op (returns 0). P13 tombstone records are applied-
        dropped here, never carried into the new journal: their victims are gone, a
        carried tombstone would have nothing left to kill and would grow forever.
        Lock coverage: runs entirely under the store's global RLock, serializing against
        offer/evict/discard - no journal appends can happen inside. Blob READERS
        (restore, prefetch staging) read by path outside that lock under a refs pin
        taken under it; since P9 moves the offsets, the swap is DEFERRED (no-op) while
        any refs pin is held: the check and the swap sit in the same lock hold. After
        an unrecoverable armed failure ("left for boot recovery") the store keeps
        running on mismatched in-memory offsets until restart; boot recovery converges,
        and new compactions are refused while the swap token is pending."""
        if not self.enabled or self._blob_fd < 0:
            return 0
        with self._lock:
            # Never start a second swap on top of a diverged disk: a pending swap token
            # means a previous armed failure left its evidence for boot recovery - the
            # next boot converges byte-exactly, but only if that evidence survives.
            token = self._read_swap_token()
            if token is not None:
                logger.warning("session tier: compaction skipped, an armed swap token "
                               "is still pending (%s); boot recovery owns it",
                               self._describe_swap_token(token))
                return 0
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
                    if not rec.get("t")   # P13 tombstone: applied, never carried
                    and (rec["path_key"], rec["off"], rec["n"]) in live]
            if len(keep) == len(records):
                return 0                      # nothing dead: no-op (also covers empty journal)
            readers = [seg for seg in self._segments.values() if seg.refs]
            if readers:
                # A shrinking swap moves the offsets a by-path reader may be mid-read
                # on: defer to the next trigger (pins drop within one restore).
                logger.info("session tier: compaction deferred, %d blob reads in flight",
                            len(readers))
                return 0
            old_size = os.path.getsize(self._blob_path)
            live_bytes = 0
            seen_ext: set[tuple[int, int]] = set()
            for r in keep:
                span = ([(e[0], e[1], e[4] if len(e) > 4 else -e[1] % _BLK)
                         for e in r["extents"]]
                        if r.get("extents") is not None
                        else [(r["off"], r["n"], -r["n"] % _BLK)])
                for soff, sn, spad in span:
                    if (soff, sn) not in seen_ext:
                        seen_ext.add((soff, sn))
                        live_bytes += sn + spad
            logger.info("session tier: compact shrink start: %d records, live %.2f GiB "
                        "(blob %.2f GiB)", len(keep), live_bytes / 2**30,
                        old_size / 2**30)
            # Cleanup stays legal only while the commit has NOT started: an OSError in
            # the copy/journal/arm phase leaves the old pair untouched, so the staged
            # files are pure garbage. The moment _commit_swap begins (anchors, renames)
            # the swap counts as ARMED: an OSError must NEVER lead to temp cleanup -
            # the token and anchors on disk are the only evidence boot recovery can
            # finish or roll the swap back from. (W10: pre-fix cleanup after rename#1
            # erased the anchors and left the NEW blob against the OLD journal -
            # replay dropped every live record.)
            try:
                layout, total, new_extents = self._copy_blob_shrunk(keep)
                journal = self._write_journal_new(keep, layout, new_extents)
                self._arm_swap_token(total, journal, old_size)
                self._swap_armed = True
                self._commit_swap()
            except OSError as e:
                if self._swap_armed and self._rollback_armed_swap():
                    logger.warning("session tier: compact shrink rolled back after "
                                   "error (%s); blob unchanged", e)
                elif self._swap_armed:
                    logger.warning("session tier: compact shrink error after arm (%s); "
                                   "token and anchors left for boot recovery", e)
                else:
                    logger.warning("session tier: compact shrink aborted before arm "
                                   "(%s); blob unchanged", e)
                    self._cleanup_swap_temps()
                self._swap_armed = False
                return 0
            self._swap_armed = False
            # Post-commit in-memory swap: segments jump to their dense offsets with
            # remapped extent lists, and BOTH fds are reopened on the renamed files (the
            # journal-fd reopen discipline from the P3 era, extended to the blob fd -
            # both names are new inodes).
            by_key = {(seg.path_key.hex(), seg.l2_off, seg.l2_n): seg
                      for seg in self._segments.values() if seg.in_l2}
            for i, rec in enumerate(keep):
                seg = by_key[(rec["path_key"], rec["off"], rec["n"])]
                # dense layout: remapped offsets, fresh crcs (split legacy parts carry
                # their copy-time crc) and the pad share each extent owns in its run
                seg.extents = [(layout[k][0], k[1], layout[k][1], layout[k][2],
                                layout[k][3]) for k in new_extents[i]]
                seg.l2_off, seg.l2_n = seg.extents[0][0], seg.extents[0][1]
            new_fd = os.open(self._blob_path, os.O_WRONLY | os.O_CREAT, 0o644)
            try:
                new_jfd = os.open(self._journal_path,
                                  os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            except OSError:
                os.close(new_fd)      # keep BOTH old fds open and in charge
                raise
            os.close(self._blob_fd)
            os.close(self._journal_fd)
            self._blob_fd, self._journal_fd = new_fd, new_jfd
            # Dense file == UNIQUE live volume: shared spans were copied once.
            self._ssd_used = total
            # The dense file has no holes: watermark and the P8 record ledger reset.
            # The extent table is rebuilt from the remapped live set - stale entries
            # (dead spans, pre-compact offsets) must not survive into the new generation.
            self._blob_eof = os.lseek(self._blob_fd, 0, os.SEEK_END)
            self._dead_bytes = 0
            self._dead_records = 0
            self._dead_record_bytes = 0
            self._marker_blockers = 0
            self._extents = {}
            for seg in self._segments.values():
                if seg.in_l2:
                    for ext in seg.extents:
                        self._extent_acquire(ext[0], ext[1], ext[2], ext[3], ext[4])
            logger.info("session tier: compact shrink: blob %.2f GiB -> %.2f GiB",
                        old_size / 2**30, total / 2**30)
            # Tombstones are journal bookkeeping, not data records: the dropped-count
            # stays ledger-aligned (victims + payload-dead only).
            return len(records) - len(keep) - sum(1 for r in records if r.get("t"))

    def shutdown_compact(self) -> int:
        """P13: the shutdown compact is optional once every dead record carries a
        durable tombstone - replay filters tombstoned records on both boot paths, so
        skipping the rewrite cannot resurrect a discard; the P8 runtime gate reclaims
        the holes and tombstones at a later idle point. Below the P8 floor the skip
        does not apply: that gate can never fire on such a small blob, so the compact
        keeps running (the pre-P13 unconditional rewrite). A failed tombstone append
        leaves its discard compact-durable only: the un-tombstoned dead force the
        classic rewrite here (the pre-P13 shutdown, byte-for-byte)."""
        if not self.enabled or self._blob_fd < 0:
            return 0
        # The rewrite costs ~live bytes: below the P8 floor that is nearly free, so
        # small tiers keep the pre-P13 hygiene compact (their dead can never reach
        # the runtime gate); skip only where the rewrite actually saves time.
        if (self._dead_records and not self._marker_blockers
                and self._blob_eof >= _COMPACT_FLOOR_BYTES):
            logger.info("session tier: shutdown compact skipped, %d dead records are "
                        "all tombstoned; the runtime gate reclaims later",
                        self._dead_records)
            return 0
        return self.compact()

    def _fsync_dir(self) -> None:
        dfd = os.open(self.dir, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)

    def _copy_blob_shrunk(self, keep: list[dict]):
        """P9 copy pass, extent-aware (briefs/l2-snapshot-dedup.md): each DISTINCT extent
        consumed by a kept record is copied exactly once to its DENSE new offset by
        parallel pread->pwrite workers (P4 pipeline precedent, FREETOKEN_COMPACT_WRITERS)
        - a per-record copy would silently un-dedup by duplicating shared spans. Legacy
        private records are MIGRATED (the brief's direction (d) lever): their mixed
        span is split at the pages|snaps seam into two extents with copy-time crcs,
        making them shareable for future offers. Each worker crcs the payload bytes it
        copies (the first n of the run at the extent's own offset; split legacy parts
        record their fresh crc); a mismatch fails the whole copy:
        a live record whose bytes no longer verify must never be moved. The PADDED span
        is copied because the O_DIRECT reader reads block-aligned windows. Returns
        (layout, total, new_extents): layout maps source (off, n) -> [dst_off, crc,
        alg, pad_share]; new_extents[i] lists keep[i]'s remapped extent keys in order.
        OSError aborts with the old blob untouched."""
        tmp = self._blob_path + ".new"
        dst = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        src = os.open(self._blob_path, os.O_RDONLY)   # _blob_fd itself is write-only
        layout: dict[tuple[int, int], list] = {}
        new_extents: list[list[tuple[int, int]]] = []
        units: list[tuple[int, int, object, object, bool]] = []
        seen: set[tuple[int, int]] = set()
        for rec in keep:
            keys: list[tuple[int, int]] = []
            exts = rec.get("extents")
            if exts is not None:
                for e in exts:
                    key = (e[0], e[1])
                    keys.append(key)
                    if key not in seen:
                        seen.add(key)
                        units.append((e[0], e[1],
                                      e[2] if len(e) > 2 else None,
                                      e[3] if len(e) > 3 else None, False))
            else:
                n = rec["n"]
                seam = sum(rec["page_lens"])
                snaps = sum(rec["snap_lens"])
                if snaps and seam and n >= seam + snaps:
                    # legacy migration: split pages|snaps (crcs computed at copy time)
                    for soff, sn in ((rec["off"], seam), (rec["off"] + seam, n - seam)):
                        key = (soff, sn)
                        keys.append(key)
                        if key not in seen:
                            seen.add(key)
                            units.append((soff, sn, None, _CRC32C_ALG, True))
                else:
                    key = (rec["off"], n)
                    keys.append(key)
                    if key not in seen:
                        seen.add(key)
                        units.append((rec["off"], n, rec.get("crc"),
                                      rec.get("crc_alg"), False))
            new_extents.append(keys)
        # Merge physically adjacent units into RUNS (extents of one original private
        # span that all survived together): one padded copy per run preserves the
        # original contiguous layout and its single block padding - no block growth.
        # Distinct pad-free spans can abut too (adjacency alone is not same-span proof),
        # but the merged run stays byte-correct regardless: each member keeps its
        # relative offset and the run owns only the last member's padding - and a
        # cross-span merge requires the earlier span's pad to be zero, so the copied
        # total is unchanged.
        runs: list[list] = []       # [src_off, total_n, members, pad]
        for u in units:
            if runs and runs[-1][0] + runs[-1][1] == u[0]:
                runs[-1][1] += u[1]
                runs[-1][2].append(u)
                runs[-1][3] = -(runs[-1][1]) % _BLK
            else:
                runs.append([u[0], u[1], [u], -(u[1]) % _BLK])
        off = 0
        spans = []
        for run in runs:
            span = run[1] + run[3]
            for j, u in enumerate(run[2]):
                # pad share: interior extents own nothing, the run's last extent owns
                # the block padding - sums stay exact in the dense layout too
                layout[(u[0], u[1])] = [off + (u[0] - run[0]), u[2], u[3],
                                        run[3] if j == len(run[2]) - 1 else 0]
            spans.append((off, run[0], span, run))
            off += span
        total = off
        # Preallocate: block allocation off the writers' critical path (P9B iron: the
        # extent allocation stalls showed as a mid-copy dip on the NVMe).
        os.ftruncate(dst, total)

        def _copy_one(idx: int) -> None:
            dst_off, run_src, span, run = spans[idx]
            members, run_pad = run[2], run[3]
            # per-member crc accumulator: [off, n, crc, alg, compute, acc]
            acc = [[m[0], m[1], m[2], m[3], m[4], 0] for m in members]
            done = 0
            while done < span:
                n = min(_COMPACT_CHUNK, span - done)
                buf = os.pread(src, n, run_src + done)
                if len(buf) < n:      # torn historical hole: keep the layout dense,
                    buf += b"\0" * (n - len(buf))   # the crc check flags live corruption
                view = memoryview(buf)
                w = 0
                while w < n:
                    w += os.pwrite(dst, view[w:], dst_off + done + w)
                # The chunk lives in .new now: drop the source cache page so a 100+ GiB
                # copy cannot evict the rest of the page cache from under the system.
                os.posix_fadvise(src, run_src + done, n, os.POSIX_FADV_DONTNEED)
                for m in acc:
                    rel, need = m[0] - run_src, m[1]
                    lo, hi = max(rel - done, 0), min(rel + need - done, n)
                    if hi > lo:
                        take = view[lo:hi]
                        alg = m[3] or _CRC32C_ALG
                        m[5] = (crc32c.crc32c(take, value=m[5]) if alg == _CRC32C_ALG
                                else zlib.crc32(take, m[5]))
                done += n
            for m in acc:                             # whole-extent verification
                layout[(m[0], m[1])][1] = m[5]        # fresh dense-layout crc
                if not m[4] and m[2] is not None and m[5] != m[2]:
                    raise OSError(f"payload crc mismatch moving record at {m[0]}")

        try:
            workers = min(_compact_writers(_COMPACT_WRITERS), len(runs))
            if workers <= 1:
                for i in range(len(runs)):
                    _copy_one(i)
            else:
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    list(ex.map(_copy_one, range(len(runs))))
            os.fdatasync(dst)
            # The copy is on disk: release its cache too.
            os.posix_fadvise(dst, 0, total, os.POSIX_FADV_DONTNEED)
        except OSError:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        finally:
            os.close(src)
            os.close(dst)
        return layout, total, new_extents

    def _write_journal_new(self, keep: list[dict], layout: dict,
                           new_extents: list) -> bytes:
        """journal.log.new: the kept records with remapped extent lists - every extent
        at its P9 dense offset, legacy records migrated to the extent format (split
        spans carry their copy-time crcs). The record-level crc (logical payload) and
        every other field are unchanged - the payload CONTENT did not move semantics,
        only its storage locations - so replay verification behavior is identical."""
        out = bytearray()
        for i, rec in enumerate(keep):
            new_rec = dict(rec)
            exts = []
            for key in new_extents[i]:
                dst_off, crc, alg, pad = layout[key]
                exts.append([dst_off, key[1], crc, alg, pad])
            new_rec["extents"] = exts
            new_rec["off"], new_rec["n"] = exts[0][0], exts[0][1]
            payload = json.dumps(new_rec, sort_keys=True).encode()
            out += struct.pack("<II", len(payload), zlib.crc32(payload)) + payload
        tmp = self._journal_path + ".new"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            pos, view = 0, memoryview(out)
            while pos < len(out):
                pos += os.write(fd, view[pos:])
            os.fsync(fd)
        finally:
            os.close(fd)
        return bytes(out)

    def _arm_swap_token(self, blob_size: int, journal: bytes,
                        old_blob_size: int) -> None:
        """Arm the two-phase commit: from the durable rename of this token on, any crash
        is recovered by boot (_recover_swap: finish or roll back), not by luck. The OLD
        pair's sizes ride along as arm-time context (the recovery rollback/completed
        decision keys on the .old anchors, which are unambiguous even when the old
        volume coincides with the new one - W11)."""
        rec = {"blob_size": blob_size, "journal_bytes": len(journal),
               "journal_sha256": hashlib.sha256(journal).hexdigest(),
               "old_blob_size": old_blob_size,
               "old_journal_bytes": os.path.getsize(self._journal_path),
               "generation": self._marker_generation}
        tmp = os.path.join(self.dir, _SWAP_TOKEN_NAME + ".new")
        with open(tmp, "wb") as f:
            f.write(json.dumps(rec, sort_keys=True).encode())
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp, os.path.join(self.dir, _SWAP_TOKEN_NAME))
        self._fsync_dir()

    def _read_swap_token(self) -> dict | None:
        try:
            with open(os.path.join(self.dir, _SWAP_TOKEN_NAME), "rb") as f:
                mk = json.loads(f.read())
        except (OSError, ValueError):
            return None
        return mk if isinstance(mk, dict) else None

    @staticmethod
    def _describe_swap_token(token: dict) -> str:
        """One-line human form of a pending swap token for the compact-skip warning."""
        try:
            return "generation=%s blob_size=%d journal_bytes=%d" % (
                token.get("generation"), int(token["blob_size"]),
                int(token["journal_bytes"]))
        except (KeyError, TypeError, ValueError):
            return "unreadable"

    def _swap_token_matches(self, token: dict) -> bool:
        """O(journal) verification of an armed swap: the staged pair's sizes match the
        token and the staged journal is byte-identical to the one hashed at arm time.
        The blob CONTENT was already verified per record by the copy workers before the
        token was armed - a sparse hole cannot exist past that barrier - so sizes plus
        the journal hash prove the staged pair is exactly the armed state."""
        try:
            blob_ref = int(token["blob_size"])
            journal_ref = int(token["journal_bytes"])
            sha_ref = token["journal_sha256"]
        except (KeyError, TypeError, ValueError):
            return False
        blob_new = self._blob_path + ".new"
        blob = blob_new if os.path.exists(blob_new) else self._blob_path
        try:
            if os.path.getsize(blob) != blob_ref:
                return False
            journal_new = self._journal_path + ".new"
            src = journal_new if os.path.exists(journal_new) else self._journal_path
            with open(src, "rb") as f:
                data = f.read()
        except OSError:
            return False
        return (len(data) == journal_ref
                and hashlib.sha256(data).hexdigest() == sha_ref)

    def _commit_swap(self) -> None:
        """Swap the staged pair into place: hard-link anchors keep the old inodes alive
        under .old names (boot recovery's rollback path), then both renames, then the
        anchors and the token go away. The anchors and the token are unlinked ONLY after
        the renames are durable: an OSError before that point leaves them up (W10) and
        either the caller's synchronous rollback or boot recovery converges from them.
        Any SIGKILL inside leaves the token on disk and boot recovery converges - crash
        table in .tasks/boot-shutdown-io/p9-plan."""
        blob_old, journal_old = self._blob_path + ".old", self._journal_path + ".old"
        for link in (blob_old, journal_old):   # stale anchors from an older attempt
            try:
                os.unlink(link)
            except OSError:
                pass
        os.link(self._blob_path, blob_old)
        os.link(self._journal_path, journal_old)
        self._fsync_dir()
        os.rename(self._blob_path + ".new", self._blob_path)
        os.rename(self._journal_path + ".new", self._journal_path)
        self._fsync_dir()                      # renames durable BEFORE the anchors go
        for path in (blob_old, journal_old, os.path.join(self.dir, _SWAP_TOKEN_NAME)):
            try:
                os.unlink(path)
            except OSError:
                pass
        self._fsync_dir()

    def _rollback_armed_swap(self) -> bool:
        """Synchronous best-effort rollback of an ARMED swap through the .old anchors
        (same semantics as the recovery rollback branch): restore the old inodes while
        they are still anchored, then drop the staged files and the token. Returns False
        when the anchors are gone (the commit is past rollback) or the rollback itself
        errored - in both cases everything stays on disk for boot recovery."""
        blob_old, journal_old = self._blob_path + ".old", self._journal_path + ".old"
        if not (os.path.exists(blob_old) or os.path.exists(journal_old)):
            return False
        try:
            for link, path in ((blob_old, self._blob_path),
                               (journal_old, self._journal_path)):
                if os.path.exists(link):
                    os.rename(link, path)
            self._cleanup_swap_temps()
            self._fsync_dir()
            return True
        except OSError:
            return False

    def _cleanup_swap_temps(self) -> int:
        """Remove swap temp files (interrupted copy, rolled-back commit) and the legacy
        pre-P9 compact temps; returns how many names were removed."""
        removed = 0
        for name in (_BLOB_NAME + ".new", _JOURNAL_NAME + ".new",
                     _SWAP_TOKEN_NAME + ".new", _SWAP_TOKEN_NAME,
                     _BLOB_NAME + ".old", _JOURNAL_NAME + ".old",
                     _BLOB_NAME + ".compact", _JOURNAL_NAME + ".compact"):
            try:
                os.unlink(os.path.join(self.dir, name))
                removed += 1
            except OSError:
                pass
        return removed

    def _recover_swap(self) -> None:
        """Boot-time recovery of an interrupted shrinking compact, run BEFORE the journal
        is read. Token absent: the commit never armed - only temp files to clean. Token
        present and verified: finish the swap (rename whatever .new staged files are
        left). Token present but mismatched: roll back ONLY while the old pair is
        provably restorable - the .old anchors are up (the commit started but never
        reached its unlink phase), or the commit never started and the on-disk pair
        still carries the token's OLD sizes (a corrupted staged file). Otherwise the
        commit completed - complete instead; the mismatch is then post-commit damage
        outside the fault model (replay's payload-crc check is the last resort). The
        anchors are the phase indicator: stronger than sizes, which are ambiguous when
        the old volume coincides with the new one (W11). Every branch swallows its own
        errors: recovery must never kill the boot."""
        if not self._journal_path:
            return
        changed = False
        for attempt in (1, 2):                 # one retry: a transient EIO mid-rename
            try:
                token = self._read_swap_token()
                if token is None:
                    changed |= self._cleanup_swap_temps() > 0
                    break
                if not self._swap_token_matches(token):
                    # W11: the rollback-vs-completed decision must be deterministic.
                    # Roll back only while the old pair is provably restorable: either
                    # the .old anchors are still up (the commit started but never
                    # reached its unlink phase - the anchors are the phase indicator,
                    # stronger than sizes, which are ambiguous when the old volume
                    # coincides with the new one), or the commit never started at all
                    # and the on-disk pair still carries the token's OLD sizes (a
                    # corrupted staged file before any rename). Otherwise the commit
                    # completed - complete instead; the mismatch is then post-commit
                    # damage outside the fault model (replay's payload-crc check is
                    # the last resort).
                    if (os.path.exists(self._blob_path + ".old")
                            or os.path.exists(self._journal_path + ".old")
                            or self._disk_pair_is_old(token)):
                        for link, path in ((self._blob_path + ".old", self._blob_path),
                                           (self._journal_path + ".old",
                                            self._journal_path)):
                            if os.path.exists(link):
                                os.rename(link, path)
                        changed = True
                        logger.warning("session tier: swap token verification FAILED; "
                                       "rolled back to the pre-compact files")
                else:
                    blob_new = self._blob_path + ".new"
                    journal_new = self._journal_path + ".new"
                    if os.path.exists(blob_new):
                        os.rename(blob_new, self._blob_path)
                        changed = True
                    if os.path.exists(journal_new):
                        os.rename(journal_new, self._journal_path)
                        changed = True
                    logger.info("session tier: completed interrupted compact swap")
                changed |= self._cleanup_swap_temps() > 0
                break
            except OSError as e:
                if attempt == 1:
                    logger.warning("session tier: swap recovery hit %s; retrying once", e)
                    continue
                logger.warning("session tier: swap recovery failed after retry (%s); "
                               "the replay crc check converges from here", e)
                changed = True      # partial moves: resync the fds/sizes below anyway
        if changed:
            try:
                self._fsync_dir()
            except OSError:
                pass
            # __init__ opened both fds on the pre-recovery inodes and seeded the
            # provisional sizes from them - resync after any recovery move. Atomic tail:
            # open both, then close both - a failure of the second open keeps the old
            # fds open and in charge instead of leaving one closed.
            try:
                new_fd = os.open(self._blob_path, os.O_WRONLY | os.O_CREAT, 0o644)
                try:
                    new_jfd = os.open(self._journal_path,
                                      os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
                except OSError:
                    os.close(new_fd)
                    raise
                os.close(self._blob_fd)
                os.close(self._journal_fd)
                self._blob_fd, self._journal_fd = new_fd, new_jfd
                self._ssd_used = os.path.getsize(self._blob_path)
                self._blob_eof = self._ssd_used
            except OSError as e:
                logger.warning("session tier: post-recovery fd reopen failed (%s); "
                               "keeping the pre-recovery fds", e)

    def _disk_pair_is_old(self, token: dict) -> bool:
        """W11 classifier half: the on-disk pair still carries the token's OLD sizes,
        i.e. the commit never renamed anything (a corrupted staged file before the
        anchors exist is recoverable by plain cleanup). Missing legacy fields or a size
        mismatch -> False."""
        try:
            old_blob = int(token["old_blob_size"])
            old_journal = int(token["old_journal_bytes"])
        except (KeyError, TypeError, ValueError):
            return False
        try:
            return (os.path.getsize(self._blob_path) == old_blob
                    and os.path.getsize(self._journal_path) == old_journal)
        except OSError:
            return False

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

    # ----------------------------------------------------------------- metrics

    def snapshot(self) -> dict:
        """Counter + byte-gauge snapshot for the metrics export; {} when off."""
        if not self.enabled:
            return {}
        with self._lock:
            snap = dict(self._counters)
            snap.update(l1_used=self._l1_used, ssd_used=self._ssd_used,
                        blob_eof=self._blob_eof, dead_bytes=self._dead_bytes,
                        dead_records=self._dead_records,
                        dead_record_bytes=self._dead_record_bytes,
                        segments=len(self._segments), tickets=len(self._tickets))
            return snap

    def stats_line(self) -> str:
        """Compact one-liner for the scheduler's periodic batch log AND the shutdown
        final line ("session tier final: <this>"); "" when off. Stable field order:
        l1, l2, offers=ok/rej, dedup, ext (extents_shared), probes=hit/miss, midsn
        (probe_midspan_only), snapfree (probe_snapfree_skip), restore=l1/l2, ok,
        refused, evict, demote, discard, tomb, dead, holes, pf=begin/adopt/abandon,
        prej (prefetch_rej_cap)."""
        snap = self.snapshot()
        if not snap:
            return ""
        return (f"session-tier: l1={snap['l1_used'] / 2**20:.1f}MiB, "
                f"l2={snap['ssd_used'] / 2**20:.1f}MiB, "
                f"offers={snap['offers_ok']}/{snap['offers_rej']}, "
                f"dedup={snap['offers_dedup']}, ext={snap['extents_shared']}, "
                f"probes={snap['probe_hit']}/{snap['probe_miss']}, "
                f"midsn={snap['probe_midspan_only']}, "
                f"snapfree={snap['probe_snapfree_skip']}, "
                f"restore={snap['restore_l1']}(l1)/{snap['restore_l2']}(l2), "
                f"ok={snap['restore_ok']}, refused={snap['restore_refused']}, "
                        f"evict={snap['evictions']}, demote={snap['demotions']}, "
                        f"discard={snap['discards']}, tomb={snap['tombstones']}, "
                        f"dead={snap['dead_records']}rec/"
                        f"{snap['dead_record_bytes'] / 2**20:.1f}MiB, "
                        # display clamp: crash-window compact leaves dead spans above the truncated blob_eof
                        f"holes={max(0, snap['dead_bytes'] - snap['dead_record_bytes']) / 2**20:.1f}MiB, "
                        f"pf={snap['prefetch_adopt']}/{snap['prefetch_begin']}/"
                        f"{snap['prefetch_abandon']}, "
                        f"prej={snap['prefetch_rej_cap']}")
