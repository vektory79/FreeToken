"""SessionTierStore unit tests (tiering W1): content addressing, dedup, L1/L2 LRU,
journal crash-safety, byte-identical restore, off-mode. CPU only, no VRAM."""
from __future__ import annotations

import contextlib
import itertools
import os
import time
from types import SimpleNamespace

import pytest

from freetoken.scheduler.session_tier import (SessionTierStore, SnapshotSource, TierHandle,
                                              chain_page_key)

PAGE = 256
SNAP = 128
# 2 segments fit in L1 (2 * (3*256 + 128) = 1792 <= 2048); a 3rd offer forces an L1->L2 demote.
RAM_BYTES = 2048
SSD_BYTES = 6 * 4096  # room for 6 padded blob records


@contextlib.contextmanager
def _deterministic_clock():
    """Port of tests/kvcache/radix/driver.deterministic_clock: the store stamps segments
    with one time.monotonic_ns() read per touch, so a real clock's coarse resolution adds
    ties between offers and makes every LRU-order assert flaky."""
    saved = time.monotonic_ns
    time.monotonic_ns = itertools.count(1_000_000_001).__next__   # type: ignore[assignment]
    try:
        yield
    finally:
        time.monotonic_ns = saved                                 # type: ignore[assignment]


@pytest.fixture(autouse=True)
def det_clock():
    with _deterministic_clock():
        yield


def _cfg(ram=RAM_BYTES, ssd=SSD_BYTES, d=None):
    return SimpleNamespace(ram_bytes=ram, dir=d, ssd_bytes=ssd)


def _page_data(tokens):
    return repr(tokens).encode().ljust(PAGE, b".")


def _chain(token_lists):
    """Chain page keys + (key, bytes) pairs; identical prefixes produce identical keys,
    which is exactly what the caller (W2) derives from radix token-id tuples."""
    keys, pages, prev = [], [], None
    for tokens in token_lists:
        key = chain_page_key(prev, tuple(tokens))
        keys.append(key)
        pages.append((key, _page_data(tokens)))
        prev = key
    return keys, pages


def _snap(tag):
    return f"snap-{tag}".encode().ljust(SNAP, b"#")


class _SnapHandle:
    """SnapshotSource stub: W2 will hide the LinearStatePool slot -> bytes copy here."""

    def __init__(self, data: bytes):
        self._data = data

    def payload(self) -> bytes:
        return self._data


# --------------------------------------------------------------------- off mode

def test_off_mode_is_inert(tmp_path):
    store = SessionTierStore(_cfg(ram=0, ssd=0, d=None))
    assert not store.enabled
    assert store.probe([b"\x00" * 16]) is None
    assert store.offer(b"path", 1, [(b"\x00" * 16, b"x" * PAGE)]) is False
    assert store.restore(TierHandle(b"path", 1, 7)) is None
    assert store.evict_store(3) == 0
    assert store.flush_live() == 0
    assert store.replay_journal() == 0
    assert list(os.listdir(str(tmp_path))) == []  # never touched anything


# ------------------------------------------------------- addressing, dedup, probe

def test_content_addressing_dedup_and_probe(tmp_path):
    store = SessionTierStore(_cfg())
    a_keys, a_pages = _chain([(1, 0), (1, 1), (1, 2)])
    b_keys, b_pages = _chain([(1, 0), (1, 1), (7, 2), (7, 3)])  # shares the first two pages
    assert store.offer(b"path-a", 3, a_pages, _snap("a"))
    assert store.offer(b"path-b", 4, b_pages)
    # dedup: only 5 unique pages are resident in L1 (3 + 2 new), not 7
    assert store._l1_used == 5 * PAGE + SNAP

    deep = store.probe(b_keys)
    assert deep is not None and deep[0] == 4 and isinstance(deep[1], TierHandle)
    shallow = store.probe(a_keys[:2])
    assert shallow[0] == 2
    miss = store.probe([a_keys[0], a_keys[1], chain_page_key(a_keys[1], (999, 9))])
    assert miss is None or miss[0] == 2  # divergent suffix never matches deeper

    pages, snaps = store.restore(shallow[1])
    assert pages == [_page_data(t) for t in [(1, 0), (1, 1)]]  # depth-truncated restore
    assert snaps == []


def test_restore_returns_byte_identical_pages_and_snapshots(tmp_path):
    store = SessionTierStore(_cfg())
    keys, pages = _chain([(2, 0), (2, 1), (2, 2)])
    handle = _SnapHandle(_snap("h"))
    assert store.offer(b"path", 3, pages, handle)
    depth, h = store.probe(keys)
    got_pages, got_snaps = store.restore(h)
    assert depth == 3
    assert got_pages == [data for _, data in pages]
    assert got_snaps == [_snap("h")]
    assert all(isinstance(p, bytes) for p in got_pages) and \
        got_pages == [bytes(p) for p in got_pages]  # plain bytes copies, not pool views


def test_snapshot_source_handle_and_cap(tmp_path):
    store = SessionTierStore(_cfg())
    keys, pages = _chain([(3, 0)])
    assert store.offer(b"path", 1, pages, [_SnapHandle(_snap("s1")), _snap("s2")])
    _, snaps = store.restore(store.probe(keys)[1])
    assert snaps == [_snap("s1"), _snap("s2")]
    with pytest.raises(ValueError):
        store.offer(b"path2", 1, pages, [_snap("s")] * 4)


def test_reoffer_same_path_refreshes_not_duplicates(tmp_path):
    store = SessionTierStore(_cfg())
    keys, pages = _chain([(4, 0), (4, 1)])
    assert store.offer(b"path", 2, pages)
    assert store.offer(b"path", 2, pages, _snap("again"))
    assert len(store._segments) == 1
    _, snaps = store.restore(store.probe(keys)[1])
    assert snaps == [_snap("again")]


# ------------------------------------------------------------------- L1 -> L2 LRU

def test_l1_overflow_demotes_lru_to_l2(tmp_path):
    store = SessionTierStore(_cfg(d=str(tmp_path)))
    keys_a, pages_a = _chain([(1, 0), (1, 1), (1, 2)])
    keys_b, pages_b = _chain([(2, 0), (2, 1), (2, 2)])
    keys_c, pages_c = _chain([(3, 0), (3, 1), (3, 2)])
    assert store.offer(b"path-a", 3, pages_a, _snap("a"))
    assert store.offer(b"path-b", 3, pages_b, _snap("b"))
    assert store.offer(b"path-c", 3, pages_c, _snap("c"))
    # deterministic clock: a is oldest -> demoted; b and c stay in L1
    seg_a = store._by_path(b"path-a")
    assert seg_a.in_l2 and not seg_a.in_l1
    assert store._by_path(b"path-b").in_l1 and store._by_path(b"path-c").in_l1
    # the demoted segment restores byte-identical from the L2 blob
    d, h = store.probe(keys_a)
    pages, snaps = store.restore(h)
    assert d == 3
    assert pages == [data for _, data in pages_a] and snaps == [_snap("a")]


def test_note_match_reorders_lru(tmp_path):
    store = SessionTierStore(_cfg(d=str(tmp_path)))
    _, pages_a = _chain([(1, 0), (1, 1), (1, 2)])
    _, pages_b = _chain([(2, 0), (2, 1), (2, 2)])
    store.offer(b"path-a", 3, pages_a)
    store.offer(b"path-b", 3, pages_b)
    store.note_match(b"path-a", 3)  # refresh a -> b becomes the LRU victim
    assert store.evict_store(1) == 1
    assert store._by_path(b"path-a").in_l1
    assert store._by_path(b"path-b").in_l2


def test_l2_true_eviction_and_refcount_guard(tmp_path):
    store = SessionTierStore(_cfg(d=str(tmp_path)))
    chains = [_chain([(i, 0), (i, 1), (i, 2)]) for i in range(10, 19)]
    for i, (keys, pages) in enumerate(chains[:6]):
        assert store.offer(f"path-{i}".encode(), 3, pages, _snap(f"{i}"))
    # L1 holds the two newest; four older segments were demoted into the blob
    assert sum(1 for s in store._segments.values() if s.in_l2) == 4
    assert store.evict_store(2) == 2   # two more L1 segments go to L2
    # the 9th offer pushes L2 over its 6-record SSD cap -> the oldest L2 segment
    # (path-0) is truly discarded: the only path back to a re-prefill
    for i, (keys, pages) in enumerate(chains[6:], start=6):
        assert store.offer(f"path-{i}".encode(), 3, pages, _snap(f"{i}"))
    assert store.probe(chains[0][0]) is None
    assert store.probe(chains[1][0]) is not None

    # refcount guard: a pinned L1 segment is not demoted
    keys8, _ = chains[8]
    h = store.probe(keys8)[1]
    seg = store._segments[h._seg_id]
    seg.refs += 1
    assert store.evict_store(1) == 1
    assert seg.seg_id in store._segments  # pinned segment survived
    seg.refs -= 1


# ------------------------------------------------------------------------ journal

def _boot(cfg_dir):
    """Boot simulation on an existing tier directory: replay then probe."""
    store = SessionTierStore(_cfg(d=cfg_dir))
    return store, store.replay_journal()


def test_journal_replay_after_sigkill_truncated_tail(tmp_path):
    d = str(tmp_path / "tier")
    store = SessionTierStore(_cfg(d=d))
    chains = [_chain([(i, 0), (i, 1), (i, 2)]) for i in (1, 2)]
    for i, (keys, pages) in enumerate(chains, start=1):
        assert store.offer(f"path-{i}".encode(), 3, pages, _snap(f"{i}"))
    assert store.flush_live() == 2
    assert store.evict_store(2) == 2  # L1 is now empty; everything lives in L2 + journal

    # SIGKILL mid-append: a half-written record (header claims more bytes than exist)
    with open(os.path.join(d, "journal.log"), "ab") as f:
        f.write(b"\x40\x00\x00\x00partial paylo")

    boot, n = _boot(d)
    assert n == 2
    for i, (keys, pages) in enumerate(chains, start=1):
        hit = boot.probe(keys)
        assert hit is not None and hit[0] == 3
        pages_out, snaps_out = boot.restore(hit[1])
        assert pages_out == [data for _, data in pages]
        assert snaps_out == [_snap(f"{i}")]
    # torn-tail case: header present, payload corrupt
    store2 = SessionTierStore(_cfg(d=d))
    with open(os.path.join(d, "journal.log"), "ab") as f:
        f.write(b"\x10\x00\x00\x00" + b"corruptpayload!!")
    assert store2.replay_journal() == 2  # still recovers the 2 valid records, stops at the tail


def test_replay_is_idempotent(tmp_path):
    d = str(tmp_path / "tier")
    store = SessionTierStore(_cfg(d=d))
    keys, pages = _chain([(9, 0), (9, 1)])
    assert store.offer(b"path", 2, pages, _snap("x"))
    assert store.flush_live() == 1
    first = store.replay_journal()
    seg_ids_1 = sorted(store._segments)
    probes_1 = store.probe(keys)
    restore_1 = store.restore(probes_1[1])

    second = store.replay_journal()
    assert second == first
    assert sorted(store._segments) == seg_ids_1
    probes_2 = store.probe(keys)
    assert probes_2[0] == probes_1[0]
    assert store.restore(probes_2[1]) == restore_1
    third = store.replay_journal()
    assert third == first


def test_l2_only_mode_without_ram_pool(tmp_path):
    d = str(tmp_path / "tier")
    store = SessionTierStore(_cfg(ram=0, ssd=SSD_BYTES, d=d))
    assert store.enabled and store._pool is None
    keys, pages = _chain([(5, 0), (5, 1)])
    assert store.offer(b"path", 2, pages, _snap("s"))
    hit = store.probe(keys)
    pages_out, snaps_out = store.restore(hit[1])
    assert pages_out == [data for _, data in pages] and snaps_out == [_snap("s")]


def test_offer_rejected_when_both_tiers_exhausted(tmp_path):
    store = SessionTierStore(_cfg(ram=RAM_BYTES, ssd=0, d=None))   # dir unset: no L2 at all
    chains = [_chain([(i, 0), (i, 1), (i, 2)]) for i in (1, 2, 3)]
    for i, (_, pages) in enumerate(chains[:2], start=1):
        assert store.offer(f"path-{i}".encode(), 3, pages)
    # no L2 configured (ssd_bytes=0 -> no blob fd): the 3rd offer cannot fit anywhere
    assert store.offer(b"path-3", 3, chains[2][1]) is False
    assert len(store._segments) == 2


# ------------------------------------------------- Review-2: N2 / N3 / N5


def test_short_blob_append_resyncs_eof_and_next_record_is_clean(tmp_path, monkeypatch):
    """N2 fails-before: a failed append advanced the O_APPEND fd's real EOF but not
    _blob_eof, so the next record's journal offset pointed below its data and restore
    returned garbage/padding."""
    import os as _os

    from freetoken.kvcache.utils import chain_page_key

    store = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(tmp_path)))
    k1 = chain_page_key(None, (1,))
    data1 = bytes([1]) * PAGE
    real_write = _os.write
    calls = {"n": 0}

    def flaky(fd, buf):
        calls["n"] += 1
        if calls["n"] == 1:
            real_write(fd, buf[:len(buf) // 2])
            raise OSError(5, "simulated partial write")
        return real_write(fd, buf)

    monkeypatch.setattr("freetoken.scheduler.session_tier.os.write", flaky)
    assert store.offer(b"p1", 1, [(k1, data1)]) is False      # N3: no raise out of offer
    assert store.probe([k1]) is None                           # the aborted record is gone
    k2 = chain_page_key(None, (2,))
    data2 = bytes([2]) * PAGE
    assert store.offer(b"p2", 1, [(k2, data2)])                # next append on a sane offset
    depth, handle = store.probe([k2])
    assert depth == 1
    got, _ = store.restore(handle)
    assert got[0] == data2                                     # fails-before: garbage bytes


def test_offer_never_raises_on_write_or_fsync_oserror(tmp_path, monkeypatch):
    """N3: OSError from the blob write or the blob fsync must not reach the eviction hot
    path - offer returns False and leaves no journal entry."""
    from freetoken.kvcache.utils import chain_page_key

    k = chain_page_key(None, (7,))
    pages = [(k, bytes([7]) * PAGE)]
    for target in ("os.write", "os.fsync"):
        sub = tmp_path / target.replace(".", "_")
        store = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(sub)))

        def boom(*args, **kwargs):
            raise OSError(5, "simulated")

        monkeypatch.setattr(f"freetoken.scheduler.session_tier.{target}", boom)
        try:
            assert store.offer(b"p", 1, pages) is False
            assert (sub / "journal.log").read_bytes() == b""   # no entry for a failed record
        finally:
            monkeypatch.undo()


def test_compact_drops_dead_records_and_converges_after_sigkill(tmp_path):
    """N5: compaction keeps exactly the live segments; a SIGKILL after the blob rename but
    before the journal rewrite (simulated by restoring the old journal bytes) still replays
    to exactly the live set via the payload-crc check; twice-compaction and zero-dead
    compaction are no-ops."""
    from freetoken.kvcache.utils import chain_page_key

    d = tmp_path / "c"
    store = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d)))
    data = {}
    for i in (1, 2, 3):
        k = chain_page_key(None, (i,))
        data[k] = bytes([i]) * PAGE
        assert store.offer(bytes([i]) * 16, 1, [(k, data[k])])
    assert store.evict_store(1) == 1                       # seg 1 is now dead

    stale_journal = (d / "journal.log").read_bytes()
    assert store.compact() == 1                            # one dead record dropped
    assert store.compact() == 0                            # idempotent no-op

    # SIGKILL after the blob rename, before the journal rewrite:
    (d / "journal.log").write_bytes(stale_journal)
    reborn = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d)))
    assert reborn.replay_journal() == 2                    # exactly the live segments
    assert reborn.probe([chain_page_key(None, (1,))]) is None    # the dead one stays dead
    for i in (2, 3):
        k = chain_page_key(None, (i,))
        hit = reborn.probe([k])
        assert hit is not None and hit[0] == 1
        got, _ = reborn.restore(hit[1])
        assert got[0] == data[k]                           # byte-identical restores
    assert reborn.compact() == 1                           # the stale journal still lists it
    assert reborn.compact() == 0                           # idempotent


# --------------------------------------------------------------------- metrics


def test_metrics_counters_l1_path():
    store = SessionTierStore(_cfg())                       # ram only, no L2
    chains = [_chain([(i, 0), (i, 1), (i, 2)]) for i in (1, 2, 3)]
    for i, (_, pages) in enumerate(chains[:2], start=1):
        assert store.offer(f"p{i}".encode(), 3, pages, _snap(str(i)))
    assert store.offer(b"p3", 3, chains[2][1]) is False    # L1 full, no L2
    hit = store.probe(chains[0][0])
    assert store.probe([chain_page_key(None, (77,))]) is None
    store.note_match(b"p1", 3)
    store.restore(hit[1])
    snap = store.snapshot()
    assert snap["offers_ok"] == 2 and snap["offers_rej"] == 1
    assert snap["probe_hit"] == 1 and snap["probe_miss"] == 1
    assert snap["note_match"] == 1
    assert snap["restore_l1"] == 1 and snap["restore_l2"] == 0
    assert snap["l1_used"] == 2 * (3 * PAGE + SNAP)
    assert store.stats_line().startswith("session-tier: l1=")


def test_metrics_counters_demote_discard_dead_bytes(tmp_path):
    store = SessionTierStore(_cfg(d=str(tmp_path)))
    chains = [_chain([(i, 0), (i, 1), (i, 2)]) for i in (1, 2, 3)]
    for i, (_, pages) in enumerate(chains, start=1):
        assert store.offer(f"p{i}".encode(), 3, pages, _snap(str(i)))
    assert store.evict_store(4) == 4   # the 3rd offer already demoted p1: 2 more demotes
    snap = store.snapshot()            # + 2 L2 true discards (p1, then p2)
    assert snap["evictions"] == 4 and snap["demotions"] == 3 and snap["discards"] == 2
    assert snap["dead_bytes"] == 2 * 4096 and snap["ssd_used"] == 1 * 4096
    hit = store.probe(chains[2][0])    # p3 was demoted: restore counts by tier
    assert hit is not None
    store.restore(hit[1])
    snap = store.snapshot()
    assert snap["restore_l2"] == 1 and snap["restore_l1"] == 0


# ------------------------------------------------------ scheduled compaction


def test_maybe_compact_watermark_and_floor(tmp_path, monkeypatch):
    import freetoken.scheduler.session_tier as st_mod

    d = tmp_path / "wm"
    store = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d)))
    for i in (1, 2, 3, 4):
        k = chain_page_key(None, (i,))
        assert store.offer(bytes([i]) * 16, 1, [(k, bytes([i]) * PAGE)])
    assert store.evict_store(2) == 2       # L2-only mode: both are true discards
    assert store._dead_bytes == 2 * 4096 and store._blob_eof == 4 * 4096   # 50% dead
    size = (d / "blob.bin").stat().st_size
    assert store.maybe_compact() == 0      # default floor (256 MiB) blocks the rewrite
    assert (d / "blob.bin").stat().st_size == size
    monkeypatch.setattr(st_mod, "_COMPACT_FLOOR_BYTES", 4096)
    assert store.maybe_compact() == 2      # watermark + lowered floor: rewrite fires
    assert store._dead_bytes == 0 and store._ssd_used == 2 * 4096
    # live records keep their ORIGINAL offsets (holes at the head), so the truncated
    # file ends at the last live record, not at _ssd_used
    assert (d / "blob.bin").stat().st_size == 4 * 4096
    assert store._blob_eof == 4 * 4096     # resynced to the truncated file's real EOF

    monkeypatch.setattr(st_mod, "_COMPACT_DEAD_FRACTION", 0.5)
    for i in (5, 6):
        k = chain_page_key(None, (i,))
        assert store.offer(bytes([i]) * 16, 1, [(k, bytes([i]) * PAGE)])
    assert store.evict_store(1) == 1
    assert store._dead_bytes == 4096 and store._blob_eof == 6 * 4096       # 17% < 50%
    assert store.maybe_compact() == 0      # below the watermark: no trigger
    monkeypatch.setattr(st_mod, "_COMPACT_DEAD_FRACTION", 0.1)
    assert store.maybe_compact() == 1
    assert store._dead_bytes == 0


def test_compaction_cannot_interleave_a_locked_store(tmp_path, monkeypatch):
    import threading

    import freetoken.scheduler.session_tier as st_mod

    store = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(tmp_path / "ser")))
    for i in (1, 2, 3):
        k = chain_page_key(None, (i,))
        assert store.offer(bytes([i]) * 16, 1, [(k, bytes([i]) * PAGE)])
    assert store.evict_store(1) == 1
    assert store._dead_bytes == 4096       # 33% dead: over the watermark
    monkeypatch.setattr(st_mod, "_COMPACT_FLOOR_BYTES", 4096)
    done = threading.Event()

    def run():
        store.maybe_compact()
        done.set()

    with store._lock:                      # e.g. concurrent offer/restore bookkeeping
        worker = threading.Thread(target=run)
        worker.start()
        worker.join(0.3)
        assert not done.is_set()           # blocked on the global lock, never interleaved
    worker.join(10)
    assert done.is_set() and store._dead_bytes == 0


def test_compact_keeps_both_resurrected_records_after_crash_replay(tmp_path):
    """F1 fails-before: survival keyed by path_key last-wins dropped the OLDER live
    record when a crashed discard left two live segments on one path (discard leaves the
    journal record + blob region intact; replay resurrects both) - the dropped segment
    stayed indexed but its blob region became a hole, so restore returned zeros."""
    d = tmp_path / "dup"
    store = SessionTierStore(_cfg(d=str(d)))
    keys1, pages1 = _chain([(1, 0), (1, 1), (1, 2)])
    assert store.offer(b"path", 3, pages1, _snap("a"))
    assert store.flush_live() == 1          # seg1 -> L2 (journal record R1)
    assert store.evict_store(1) == 1        # seg1 discarded; R1 + blob region survive
    keys2, pages2 = _chain([(5, 0), (5, 1), (5, 2)])
    assert store.offer(b"path", 3, pages2, _snap("b"))   # re-offer same path: new seg2
    assert store.flush_live() == 1          # seg2 -> L2 (R2); journal now has R1 + R2

    # SIGKILL-style reboot: replay resurrects BOTH same-path segments
    reborn = SessionTierStore(_cfg(d=str(d)))
    assert reborn.replay_journal() == 2
    assert [s.path_key for s in reborn._segments.values()].count(b"path") == 2
    assert reborn.compact() == 0            # identity-keyed survival: nothing is dropped

    # both resurrected segments probe AND restore byte-exact
    h2 = reborn.probe(keys2)[1]             # seg2's page keys only match seg2
    assert reborn.restore(h2) == ([data for _, data in pages2], [_snap("b")])
    h1 = reborn.probe(keys1)[1]             # seg1 still indexed and readable
    assert reborn.restore(h1) == ([data for _, data in pages1], [_snap("a")])
