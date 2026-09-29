"""SessionTierStore unit tests (tiering W1): content addressing, dedup, L1/L2 LRU,
journal crash-safety, byte-identical restore, off-mode. CPU only, no VRAM."""
from __future__ import annotations

import contextlib
import itertools
import json
import logging
import os
import struct
import threading
import time
import zlib
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


# ------------------------------------------------- offer-time dedup / supersede


def test_contained_snapshot_free_offer_skipped_and_counted(tmp_path):
    """A snapshot-free offer whose pages a live deeper same-chain segment already holds
    is redundant: skipped (offers_dedup moved), the store keeps exactly one segment, and
    the covered restores still work through the deeper segment."""
    store = SessionTierStore(_cfg(d=str(tmp_path)))
    keys, pages = _chain([(1, 0), (1, 1), (1, 2)])
    assert store.offer(b"path", 3, pages)
    assert store.offer(b"path", 2, pages[:2])
    snap = store.snapshot()
    # counting convention: a dedup skip returns True, so the offer() wrapper stamps it
    # offers_ok like any accepted offer; offers_dedup is the additional breakdown
    assert snap["offers_dedup"] == 1 and snap["offers_ok"] == 2
    assert len(store._segments) == 1
    hit = store.probe(keys[:2])
    assert hit is not None and hit[0] == 2
    assert store.restore(hit[1])[0] == [data for _, data in pages[:2]]


def test_supersede_never_matches_across_divergent_chains(tmp_path):
    """Chains sharing ONLY page 0 (same first key, then divergence) are different paths:
    a deeper plain offer of chain B must not supersede the shallower plain chain-A seg
    (the prefix check diverges) - both stay, and each probe restores its own bytes."""
    store = SessionTierStore(_cfg(d=str(tmp_path)))
    keys_a, pages_a = _chain([(1, 0), (1, 1)])
    keys_b, pages_b = _chain([(1, 0), (7, 1), (7, 2)])   # shares page 0 only
    assert keys_a[0] == keys_b[0] and keys_a[1] != keys_b[1]
    assert store.offer(b"path-a", 2, pages_a)
    assert store.offer(b"path-b", 3, pages_b)
    assert len(store._segments) == 2
    assert store.snapshot()["offers_dedup"] == 0
    hit_a = store.probe(keys_a)
    assert hit_a is not None and hit_a[0] == 2
    assert store.restore(hit_a[1])[0] == [data for _, data in pages_a]
    hit_b = store.probe(keys_b)
    assert hit_b is not None and hit_b[0] == 3
    assert store.restore(hit_b[1])[0] == [data for _, data in pages_b]


def test_supersede_shallower_plain_segment_l2_accounting(tmp_path):
    """A deeper snapshot-free offer replaces a contained shallower snapshot-free one via
    _discard (dead-byte ledger, capacity accounting, index removal) and stores normally:
    no blob surgery, byte-exact restore from the new record."""
    store = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(tmp_path)))
    keys, pages = _chain([(2, 0), (2, 1)])
    assert store.offer(b"path", 1, pages[:1])
    seg1 = store._by_path(b"path")
    assert store.offer(b"path", 2, pages)
    assert seg1.seg_id not in store._segments            # superseded
    assert len(store._segments) == 1
    seg2 = store._by_path(b"path")
    assert seg2.boundary_len == 2 and seg2.in_l2
    assert store._dead_bytes == 4096                     # the padded 1-page span is dead
    assert store._ssd_used == 4096                       # only the new record counts
    assert store._index[keys[0]] == [(seg2.seg_id, 1)]   # old entries removed
    assert store._index[keys[1]] == [(seg2.seg_id, 2)]
    hit = store.probe(keys)
    assert hit is not None and hit[0] == 2
    assert store.restore(hit[1])[0] == [data for _, data in pages]


def test_supersede_releases_l1_pages_refcounted(tmp_path):
    """Superseding an L1-resident segment releases its pool pages (refcounted) before the
    _discard cleanup: no L1 leak, no L2 dead bytes (it never reached the blob)."""
    store = SessionTierStore(_cfg())
    keys, pages = _chain([(3, 0), (3, 1)])
    assert store.offer(b"path", 1, pages[:1])
    assert store.offer(b"path", 2, pages)
    assert len(store._segments) == 1
    seg = store._by_path(b"path")
    assert seg.boundary_len == 2 and seg.in_l1
    assert store._l1_used == 2 * PAGE
    assert store._pages[keys[0]].refs == 1               # old seg's page ref released
    snap = store.snapshot()
    assert snap["discards"] == 1 and snap["dead_bytes"] == 0


def test_refs_guard_blocks_supersede(tmp_path):
    """refs > 0 (a restore in flight) blocks the supersede: both segments are kept; once
    unpinned, a deeper offer supersedes every contained snapshot-free segment."""
    store = SessionTierStore(_cfg())
    keys, pages = _chain([(4, 0), (4, 1), (4, 2)])
    assert store.offer(b"path", 1, pages[:1])
    pinned = store._by_path(b"path")
    pinned.refs += 1
    assert store.offer(b"path", 2, pages[:2])
    assert len(store._segments) == 2                     # keep both under the live reader
    assert store.snapshot()["offers_dedup"] == 0
    pinned.refs -= 1
    assert store.offer(b"path", 3, pages)
    assert len(store._segments) == 1                     # both contained segs superseded
    assert store._by_path(b"path").boundary_len == 3


def test_snapshot_bearing_deeper_offer_stored_not_skipped(tmp_path):
    """The snapshot is the non-contained part: a snapshot-bearing offer is ALWAYS stored
    (and supersedes a contained shallower snapshot-free seg); a snapshot-bearing segment
    is never discarded by a deeper snapshot-free offer and keeps serving its boundary."""
    store = SessionTierStore(_cfg(d=str(tmp_path)))
    keys, pages = _chain([(5, 0), (5, 1)])
    assert store.offer(b"path", 1, pages[:1])
    assert store.offer(b"path", 2, pages, _snap("s"))
    assert len(store._segments) == 1                     # stored, plain b1 superseded
    assert store.snapshot()["offers_dedup"] == 0
    keys3 = keys + [chain_page_key(keys[1], (5, 2))]
    pages3 = pages + [(keys3[2], _page_data((5, 2)))]
    assert store.offer(b"path", 3, pages3)               # deeper plain offer
    assert len(store._segments) == 2                     # snapshot-bearing seg kept
    snap_seg = next(s for s in store._segments.values() if any(s.snap_lens))
    assert snap_seg.boundary_len == 2 and snap_seg.snap_lens == [SNAP]
    hit = store.probe(keys[:2])                          # boundary-EXACT preference intact
    assert hit is not None and hit[0] == 2
    assert store.restore(hit[1]) == ([data for _, data in pages], [_snap("s")])


def test_dedup_covers_over_snapshot_bearing_deeper_segment(tmp_path):
    """Predicate pinned explicitly: a snapshot-free offer is COVERED by a deeper live
    same-chain segment even when that segment is snapshot-bearing. Derivation: the offer
    would serve restores only at depths <= its boundary; the deeper seg serves every one
    of them (restore depth-truncates its pages, session_tier.restore), and on the hybrid
    path a snapshot-free seg serves NO restore at all (cache.py _restore_tail needs a
    snapshot at the exact _tier_snap_bound depth) - so skipping loses no capability."""
    store = SessionTierStore(_cfg(d=str(tmp_path)))
    keys, pages = _chain([(6, 0), (6, 1), (6, 2)])
    assert store.offer(b"path", 3, pages, _snap("deep"))
    assert store.offer(b"path", 2, pages[:2])            # snapshot-free, contained
    snap = store.snapshot()
    assert snap["offers_dedup"] == 1 and len(store._segments) == 1
    hit = store.probe(keys[:2])
    assert hit is not None and hit[0] == 2               # the deeper seg serves depth 2
    assert store.restore(hit[1]) == ([data for _, data in pages[:2]], [_snap("deep")])


def test_sigkill_replay_after_supersede_keeps_exactly_live_set(tmp_path):
    """Supersede leaves the superseded journal record's blob region intact until
    compaction rewrites it away (accepted discard durability, cf. the resurrected-records
    pin); after compaction a SIGKILL-style reboot replays EXACTLY the live set, and a
    supersede in the replayed history converges under the next compaction (extends the
    sigkill-replay and compact-convergence patterns)."""
    d = tmp_path / "tier"
    store = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d)))
    keys, pages = _chain([(7, 0), (7, 1)])
    assert store.offer(b"path", 1, pages[:1])
    assert store.offer(b"path", 2, pages)                # supersedes the boundary-1 record
    assert store._dead_bytes == 4096
    assert store.compact() == 1                          # dead record dropped
    assert store.compact() == 0                          # idempotent
    boot = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d)))
    assert boot.replay_journal() == 1                    # exactly the live set
    hit = boot.probe(keys)
    assert hit is not None and hit[0] == 2
    assert boot.restore(hit[1])[0] == [data for _, data in pages]

    # uncompacted crash: the superseded record resurrects at replay, a deeper offer
    # supersedes BOTH stale segments, and compaction converges to the live set
    d2 = tmp_path / "tier2"
    store2 = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d2)))
    keys, pages = _chain([(8, 0), (8, 1), (8, 2)])
    assert store2.offer(b"path", 1, pages[:1])
    assert store2.offer(b"path", 2, pages[:2])           # supersede #1
    reborn = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d2)))
    assert reborn.replay_journal() == 2                  # both regions intact pre-compact
    assert len(reborn._segments) == 2
    assert reborn.offer(b"path", 3, pages)               # supersedes both stale segments
    assert len(reborn._segments) == 1
    assert reborn.compact() == 2
    final = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d2)))
    assert final.replay_journal() == 1
    hit = final.probe(keys)
    assert final.restore(hit[1])[0] == [data for _, data in pages]


def test_offers_dedup_metric_preseeded_and_in_stats_line(tmp_path):
    """offers_dedup is pre-seeded in the Counter (snapshot() copies it whole) and lands in
    the stats_line fragment (batch log + shutdown final line via tier_stats_line)."""
    store = SessionTierStore(_cfg(d=str(tmp_path)))
    assert store.snapshot()["offers_dedup"] == 0
    assert "dedup=0" in store.stats_line()
    keys, pages = _chain([(9, 0), (9, 1)])
    assert store.offer(b"path", 2, pages)
    assert store.offer(b"path", 1, pages[:1])
    assert store.snapshot()["offers_dedup"] == 1
    assert "dedup=1" in store.stats_line()


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


def _graceful_stop(store):
    """Mirror CacheManager.shutdown_tier's marker discipline: drop any previous marker
    FIRST, flush + compact, then write the end-of-shutdown barrier."""
    store.invalidate_shutdown_marker()
    store.flush_live()
    store.compact()
    store.write_shutdown_marker()


class _LogCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@contextlib.contextmanager
def _tier_log_capture():
    """The module logger does not propagate (init_logger), so caplog cannot see it."""
    lg = logging.getLogger("freetoken.scheduler.session_tier")
    cap = _LogCapture()
    lg.addHandler(cap)
    try:
        yield cap
    finally:
        lg.removeHandler(cap)


def _counting_pread(monkeypatch):
    """Wrap the module's os.pread with a pass-through counter: the full verification
    path preads every record's payload, the fast path must not touch the blob at all."""
    import freetoken.scheduler.session_tier as st

    real = os.pread
    calls = {"n": 0}

    def counting(fd, n, off):
        calls["n"] += 1
        return real(fd, n, off)

    monkeypatch.setattr(st.os, "pread", counting)
    return calls


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


def test_short_blob_pwrite_hole_kept_watermark_past_it_next_record_clean(tmp_path, monkeypatch):
    """N2 fails-before: a failed/short blob write must not desync the tail watermark - the
    next record's journal offset must still point at its own data and restore must return
    clean bytes. The watermark intentionally STAYS past the aborted span (the hole is
    never referenced and is reclaimed by the next compaction); the O_APPEND-era lseek
    resync is gone by design, not by accident."""
    import os as _os

    from freetoken.kvcache.utils import chain_page_key

    store = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(tmp_path)))
    k1 = chain_page_key(None, (1,))
    data1 = bytes([1]) * PAGE
    real_pwrite = _os.pwrite
    calls = {"n": 0}

    def flaky(fd, buf, offset):
        calls["n"] += 1
        if calls["n"] == 1:
            real_pwrite(fd, buf[:len(buf) // 2], offset)
            raise OSError(5, "simulated partial write")
        return real_pwrite(fd, buf, offset)

    monkeypatch.setattr("freetoken.scheduler.session_tier.os.pwrite", flaky)
    assert store.offer(b"p1", 1, [(k1, data1)]) is False      # N3: no raise out of offer
    assert store.probe([k1]) is None                           # the aborted record is gone
    k2 = chain_page_key(None, (2,))
    data2 = bytes([2]) * PAGE
    assert store.offer(b"p2", 1, [(k2, data2)])                # next append on a sane offset
    depth, handle = store.probe([k2])
    assert depth == 1
    got, _ = store.restore(handle)
    assert got[0] == data2                                     # fails-before: garbage bytes


def test_offer_never_raises_on_pwrite_or_fdatasync_oserror(tmp_path, monkeypatch):
    """N3: OSError from the blob pwrite or the blob fdatasync must not reach the eviction
    hot path - offer returns False and leaves no journal entry."""
    from freetoken.kvcache.utils import chain_page_key

    k = chain_page_key(None, (7,))
    pages = [(k, bytes([7]) * PAGE)]
    for target in ("os.pwrite", "os.fdatasync"):
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


# ----------------------------------------------------------------- prefetch tickets


def _settle(store, timeout=5.0):
    """Wait until the prefetch reader finalized every queued staging item."""
    deadline = time.monotonic() + timeout
    while not store._tickets_settled():
        assert time.monotonic() < deadline, "prefetch worker did not settle"
        time.sleep(0.001)


def test_ticket_lifecycle_begin_ready_adopt_byte_exact():
    from freetoken.scheduler.session_tier import _Pool

    store = SessionTierStore(_cfg())
    keys, pages = _chain([(6, 0), (6, 1), (6, 2)])
    assert store.offer(b"path", 3, pages, _snap("t"))
    baseline = _Pool._locked_total
    hit = store.probe(keys)
    ticket = store.begin_restore(hit[1])
    assert ticket is not None and ticket.boundary == 3
    _settle(store)
    assert ticket.state == "ready"
    got = store.consume_ticket(ticket)
    assert got is not None
    adopted_pages, adopted_snaps = got
    sync_pages, sync_snaps = store.restore(hit[1])   # the sync path is the reference
    assert adopted_pages == sync_pages == [data for _, data in pages]
    assert adopted_snaps == sync_snaps == [_snap("t")]
    assert store._tickets == {}                      # consumed: nothing left staged
    assert _Pool._locked_total == baseline           # staging unlocked (no leak)
    assert store.snapshot()["prefetch_adopt"] == 1
    # contract: the staging covers the FULL boundary - a shallower handle consumes the
    # same boundary pages and the caller depth-truncates (exactly what restore does)
    ticket2 = store.begin_restore(store.probe(keys[:2])[1])
    _settle(store)
    p2, s2 = store.consume_ticket(ticket2)
    assert p2 == [data for _, data in pages] and s2 == [_snap("t")]
    assert _Pool._locked_total == baseline


def test_ticket_abandon_frees_staging_and_consumes_nothing():
    from freetoken.scheduler.session_tier import _Pool

    store = SessionTierStore(_cfg())
    keys, pages = _chain([(7, 0), (7, 1)])
    assert store.offer(b"path", 2, pages, _snap("a"))
    baseline = _Pool._locked_total
    ticket = store.begin_restore(store.probe(keys)[1])
    _settle(store)
    store.abandon_ticket(ticket)
    assert ticket.state == "abandoned" and store._tickets == {}
    assert _Pool._locked_total == baseline           # staging freed, nothing consumed
    assert store.consume_ticket(ticket) is None      # a dropped ticket is dead
    assert store.abandon_ticket(ticket) is None      # idempotent
    assert store.snapshot()["prefetch_abandon"] == 1


def test_ticket_pending_abandon_is_finalized_by_worker(tmp_path, monkeypatch):
    """Abandoning a STILL-READING ticket must not wait on the reader: the scheduler marks
    it, the worker finalize frees the staging, and consume afterwards yields nothing."""
    import threading

    from freetoken.scheduler.session_tier import _Pool

    store = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(tmp_path / "pend")))
    keys, pages = _chain([(8, 0), (8, 1)])
    assert store.offer(b"path", 2, pages, _snap("s"))
    baseline = _Pool._locked_total
    entered, release = threading.Event(), threading.Event()
    real = store._blob_read

    def slow(blob, off, n):
        entered.set()
        assert release.wait(5)
        return real(blob, off, n)

    monkeypatch.setattr(store, "_blob_read", slow)
    ticket = store.begin_restore(store.probe(keys)[1])
    assert ticket is not None
    assert entered.wait(5) and ticket.state == "pending"
    store.abandon_ticket(ticket)                     # the scheduler never waits here
    assert ticket.state == "abandoned" and store._tickets == {}
    release.set()
    _settle(store)
    assert store.consume_ticket(ticket) is None
    assert _Pool._locked_total == baseline           # freed by the worker finalize
    assert store.snapshot()["prefetch_abandon"] == 1


def test_ticket_failed_on_vanished_source(tmp_path, monkeypatch):
    """A staging read that fails (source lost / IO error) marks the ticket failed, frees
    the staging and leaves the sync restore intact as the fallback."""
    from freetoken.scheduler.session_tier import _Pool

    store = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(tmp_path / "fail")))
    keys, pages = _chain([(9, 0), (9, 1)])
    assert store.offer(b"path", 2, pages)
    baseline = _Pool._locked_total
    hit = store.probe(keys)

    def boom(blob, off, n):
        raise OSError(5, "simulated source loss")

    monkeypatch.setattr(store, "_blob_read", boom)
    ticket = store.begin_restore(hit[1])
    _settle(store)
    assert ticket.state == "failed"
    assert store.consume_ticket(ticket) is None      # adopt falls back to sync
    assert store._tickets == {} and _Pool._locked_total == baseline
    assert store.snapshot()["prefetch_fail"] == 1
    monkeypatch.undo()
    assert store.restore(hit[1]) == ([data for _, data in pages], [])


def test_ticket_dropped_when_segment_discarded(tmp_path):
    """A staged segment that gets truly discarded can never be probed again: the orphaned
    staging is dropped with it (bytes were independent, but nothing will ever consume)."""
    from freetoken.scheduler.session_tier import _Pool

    store = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(tmp_path / "disc")))
    keys, pages = _chain([(10, 0), (10, 1)])
    assert store.offer(b"path", 2, pages, _snap("d"))
    baseline = _Pool._locked_total
    ticket = store.begin_restore(store.probe(keys)[1])
    _settle(store)
    assert store.evict_store(1) == 1                 # true L2 discard of the staged segment
    assert store._tickets == {}
    assert _Pool._locked_total == baseline
    assert store.consume_ticket(ticket) is None


def test_ticket_cap_is_bounded():
    from freetoken.scheduler.session_tier import _MAX_RESTORE_TICKETS, _Pool

    assert _MAX_RESTORE_TICKETS == 2
    store = SessionTierStore(_cfg())
    chains = [_chain([(20 + i, 0), (20 + i, 1)]) for i in range(3)]
    for i, (keys, pages) in enumerate(chains):
        assert store.offer(f"p{i}".encode(), 2, pages)
    baseline = _Pool._locked_total
    t0 = store.begin_restore(store.probe(chains[0][0])[1])
    t1 = store.begin_restore(store.probe(chains[1][0])[1])
    assert t0 is not None and t1 is not None
    assert store.begin_restore(store.probe(chains[2][0])[1]) is None   # cap
    assert store.snapshot()["prefetch_rej_cap"] == 1
    _settle(store)
    store.abandon_ticket(t0)
    t2 = store.begin_restore(store.probe(chains[2][0])[1])             # a freed slot reopens
    assert t2 is not None
    _settle(store)
    for t in (t0, t1, t2):
        store.abandon_ticket(t)
    assert store._tickets == {} and _Pool._locked_total == baseline


def test_ticket_off_mode_begin_restore_is_inert():
    store = SessionTierStore(_cfg(ram=0, ssd=0, d=None))
    assert store.begin_restore(TierHandle(b"p", 1, 7)) is None
    assert store._tickets == {}


def test_prefetch_probe_does_not_stamp_lru(tmp_path):
    """The idle prefetch probe is speculative interest: it must NOT refresh the segment's
    LRU currency (only real matches/note_match do) and it counts into its own counters."""
    store = SessionTierStore(_cfg(d=str(tmp_path / "p1")))
    ka, pa = _chain([(30, 0), (30, 1)])
    kb, pb = _chain([(31, 0), (31, 1)])
    assert store.offer(b"a", 2, pa)
    assert store.offer(b"b", 2, pb)                  # b is the newer segment
    store.probe(ka, stamp=False)                     # speculative: no refresh
    assert store.evict_store(1) == 1
    assert store._by_path(b"a").in_l2                # a was STILL the LRU victim
    assert store._by_path(b"b").in_l1
    snap = store.snapshot()
    assert snap["prefetch_probe_hit"] == 1 and snap["probe_hit"] == 0
    # contrast: a stamped probe refreshes a, so b dies instead
    store2 = SessionTierStore(_cfg(d=str(tmp_path / "p2")))
    assert store2.offer(b"a", 2, pa)
    assert store2.offer(b"b", 2, pb)
    store2.probe(ka)
    assert store2.evict_store(1) == 1
    assert store2._by_path(b"b").in_l2 and store2._by_path(b"a").in_l1


def test_ticket_stress_two_threads(tmp_path, monkeypatch):
    """Thread-safety of the staging path: one thread drives begin/settle/consume-or-abandon
    while the other churns offer/probe/restore/evict/compact. Asserts are structural (they
    hold under ANY interleaving): no exceptions, ticket conservation, refs back to zero,
    staging accounting back to baseline."""
    import random
    import threading

    import freetoken.scheduler.session_tier as st_mod
    from freetoken.scheduler.session_tier import _Pool

    monkeypatch.setattr(st_mod, "_COMPACT_FLOOR_BYTES", 4096)
    store = SessionTierStore(_cfg(ram=RAM_BYTES * 8, ssd=64 * 4096,
                                  d=str(tmp_path / "stress")))
    chains = [_chain([(40 + i, 0), (40 + i, 1)]) for i in range(6)]
    for i, (keys, pages) in enumerate(chains):
        assert store.offer(f"s{i}".encode(), 2, pages, _snap(str(i)))
    baseline = _Pool._locked_total
    errs = []
    rng_a, rng_b = random.Random(11), random.Random(13)

    def driver():
        try:
            for _ in range(40):
                keys = rng_a.choice(chains)[0][:rng_a.randint(1, 2)]
                hit = store.probe(keys)
                if hit is None:
                    continue
                t = store.begin_restore(hit[1])
                if t is None:
                    continue
                assert t.ready.wait(5)
                if rng_a.random() < 0.5:
                    store.consume_ticket(t)
                else:
                    store.abandon_ticket(t)
        except Exception as e:                   # noqa: BLE001 (collected, asserted below)
            errs.append(e)

    def churn():
        try:
            for i in range(30):
                op = rng_b.choice(("probe", "restore", "evict", "offer", "compact"))
                if op == "probe":
                    store.probe(rng_b.choice(chains)[0])
                elif op == "restore":
                    hit = store.probe(rng_b.choice(chains)[0])
                    if hit is not None:
                        store.restore(hit[1])
                elif op == "evict":
                    store.evict_store(1)
                elif op == "offer":
                    j = rng_b.randrange(6)
                    store.offer(f"s{j}".encode(), 2, chains[j][1], _snap(f"r{i}"))
                else:
                    store.maybe_compact()
        except Exception as e:                   # noqa: BLE001
            errs.append(e)

    ta, tb = threading.Thread(target=driver), threading.Thread(target=churn)
    ta.start()
    tb.start()
    ta.join(30)
    tb.join(30)
    assert not ta.is_alive() and not tb.is_alive()
    assert errs == []
    assert store._tickets == {}
    snap = store.snapshot()                      # conservation: every begin settled one way
    assert snap["prefetch_begin"] == (snap["prefetch_adopt"] + snap["prefetch_abandon"]
                                      + snap["prefetch_fail"])
    assert all(seg.refs == 0 for seg in store._segments.values())
    # no staging residue: the pool's own demote accounting only ever shrinks below the
    # construction baseline, and every ticket buffer was munlocked (== pins in the
    # lifecycle/abandon tests, which do not demote)
    assert _Pool._locked_total <= baseline


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


# --------------------------------------------------- P2: clean-shutdown marker


def _marker_path(d):
    return os.path.join(str(d), "shutdown.marker")


def test_shutdown_marker_fast_path_replay(tmp_path, monkeypatch):
    """P2 (a): a completed graceful shutdown writes the fsync'ed marker; the next boot
    takes the fast path (no payload pread at all) and the data is intact."""
    d = tmp_path / "tier"
    store = SessionTierStore(_cfg(d=str(d)))
    keys, pages = _chain([(1, 0), (1, 1), (1, 2)])
    assert store.offer(b"path", 3, pages, _snap("s"))
    _graceful_stop(store)
    assert os.path.exists(_marker_path(d))

    def boom(*args, **kwargs):
        raise AssertionError("payload pread must not happen on the fast path")

    monkeypatch.setattr("freetoken.scheduler.session_tier.os.pread", boom)
    with _tier_log_capture() as cap:
        boot, n = _boot(str(d))
    assert n == 1
    assert "session tier: replay fast-path (clean shutdown marker), 1 records" in cap.messages
    hit = boot.probe(keys)
    assert hit is not None and hit[0] == 3
    out, snaps = boot.restore(hit[1])
    assert out == [x for _, x in pages] and snaps == [_snap("s")]


def test_shutdown_marker_missing_or_stale_takes_full_path(tmp_path, monkeypatch):
    """P2 (b): no marker, or a marker whose sizes no longer match (a runtime append after
    the previous stop) -> full per-record crc verification, records intact."""
    chains = [_chain([(i, 0), (i, 1), (i, 2)]) for i in (1, 2)]
    # (i) no marker: old directories without one keep the full path
    d1 = tmp_path / "no-marker"
    store = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d1)))
    for i, (keys, pages) in enumerate(chains, start=1):
        assert store.offer(f"path-{i}".encode(), 3, pages, _snap(f"{i}"))
    calls = _counting_pread(monkeypatch)
    with _tier_log_capture() as cap:
        boot, n = _boot(str(d1))
    assert n == 2 and calls["n"] == 2
    assert not any("fast-path" in m for m in cap.messages)
    for i, (keys, pages) in enumerate(chains, start=1):
        hit = boot.probe(keys)
        assert boot.restore(hit[1]) == ([x for _, x in pages], [_snap(f"{i}")])

    # (ii) stale marker: a runtime append after the graceful stop invalidates the sizes
    monkeypatch.undo()
    d2 = tmp_path / "stale"
    store = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d2)))
    for i, (keys, pages) in enumerate(chains, start=1):
        assert store.offer(f"path-{i}".encode(), 3, pages, _snap(f"{i}"))
    _graceful_stop(store)
    k3 = chain_page_key(None, (3,))
    assert store.offer(b"path-3", 1, [(k3, bytes([3]) * PAGE)])   # runtime, post-marker
    calls = _counting_pread(monkeypatch)
    boot, n = _boot(str(d2))
    assert n == 3 and calls["n"] == 3                             # full verification ran


def test_shutdown_marker_full_path_drops_dead_keeps_live(tmp_path):
    """P2 (b): a stale marker plus a corrupted payload region (torn append / hole) ->
    full path, the dead record is dropped with the crash-path report, the live ones
    restore byte-identically."""
    d = tmp_path / "dead"
    store = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d)))
    chains = [_chain([(i, 0), (i, 1), (i, 2)]) for i in (1, 2, 3)]
    for i, (keys, pages) in enumerate(chains, start=1):
        assert store.offer(f"path-{i}".encode(), 3, pages, _snap(f"{i}"))
    _graceful_stop(store)
    seg1 = store._by_path(b"path-1")
    with open(os.path.join(str(d), "blob.bin"), "r+b") as f:   # tear record 1's payload
        f.seek(seg1.l2_off)
        f.write(b"\xde" * seg1.l2_n)
    k4 = chain_page_key(None, (4,))
    assert store.offer(b"path-4", 1, [(k4, bytes([4]) * PAGE)])  # marker now stale
    with _tier_log_capture() as cap:
        boot, n = _boot(str(d))
    assert n == 3
    assert "session tier: replay dropped 1 dead/torn records" in cap.messages
    assert not any("fast-path" in m for m in cap.messages)
    assert boot.probe([chain_page_key(None, (1,))]) is None
    for i in (2, 3, 4):
        keys = chains[i - 1][0] if i <= 3 else [k4]
        hit = boot.probe(keys)
        assert hit is not None
        pages_out, _ = boot.restore(hit[1])
        assert pages_out == [x for _, x in (chains[i - 1][1] if i <= 3
                                            else [(k4, bytes([4]) * PAGE)])]


def test_shutdown_marker_invalidated_before_shutdown_crash(tmp_path, monkeypatch):
    """P2 (c): a crash mid-shutdown_tier after a VALID marker from the previous stop must
    not give a false fast path. The marker is dropped at shutdown_tier start, before any
    flush mutation; the mid-flush crash then leaves no marker (and a torn tail)."""
    d = tmp_path / "midstop"
    chains = [_chain([(i, 0), (i, 1), (i, 2)]) for i in (1, 2)]
    store = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d)))
    for i, (keys, pages) in enumerate(chains, start=1):
        assert store.offer(f"path-{i}".encode(), 3, pages, _snap(f"{i}"))
    _graceful_stop(store)
    assert os.path.exists(_marker_path(d))

    boot2 = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d)))
    assert boot2.replay_journal() == 2                           # legitimate fast path
    k3 = chain_page_key(None, (3,))
    assert boot2.offer(b"path-3", 1, [(k3, bytes([3]) * PAGE)])  # runtime since the stop

    # graceful shutdown #2 begins: the marker is gone BEFORE the flush mutates anything
    boot2.invalidate_shutdown_marker()
    assert not os.path.exists(_marker_path(d))
    # crash mid-flush: a torn journal tail (SIGKILL mid-append)
    with open(os.path.join(str(d), "journal.log"), "ab") as f:
        f.write(b"\x40\x00\x00\x00partial paylo")
    calls = _counting_pread(monkeypatch)
    with _tier_log_capture() as cap:
        boot3, n = _boot(str(d))
    assert n == 3                                                # live set intact
    assert calls["n"] == 3                                       # full verification ran
    assert not any("fast-path" in m for m in cap.messages)
    assert any("journal tail truncated/torn" in m for m in cap.messages)
    for i, (keys, pages) in enumerate(chains, start=1):
        hit = boot3.probe(keys)
        assert boot3.restore(hit[1]) == ([x for _, x in pages], [_snap(f"{i}")])
    hit = boot3.probe([k3])
    assert boot3.restore(hit[1])[0] == [bytes([3]) * PAGE]


def test_fast_path_and_full_path_identical_state(tmp_path):
    """P2 (d): on the same on-disk state the fast path and the full verification rebuild
    exactly the same store state - same segments (ids, keys, offsets, LRU stamps), index
    and byte accounting."""
    d = tmp_path / "parity"
    store = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d)))
    chains = [_chain([(i, 0), (i, 1), (i, 2)]) for i in (1, 2, 3)]
    for i, (keys, pages) in enumerate(chains, start=1):
        assert store.offer(f"path-{i}".encode(), 3, pages, _snap(f"{i}"))
    _graceful_stop(store)

    boot_fast = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d)))
    n_fast = boot_fast.replay_journal()
    os.unlink(_marker_path(d))                    # force the full path on the same data
    boot_full = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d)))
    n_full = boot_full.replay_journal()

    def state(st):
        return {sid: (seg.path_key, seg.boundary_len, tuple(seg.page_keys),
                      tuple(seg.page_lens), tuple(seg.snap_lens), seg.blob,
                      seg.l2_off, seg.l2_n, seg.last_validation)
                for sid, seg in st._segments.items()}

    assert n_fast == n_full == 3
    assert state(boot_fast) == state(boot_full)
    assert boot_fast._index == boot_full._index
    assert boot_fast._ssd_used == boot_full._ssd_used
    assert boot_fast._blob_eof == boot_full._blob_eof
    assert boot_fast._dead_bytes == boot_full._dead_bytes
    # generation: the fast path seeds it from the marker; the full path ran without one
    assert boot_fast._marker_generation >= boot_full._marker_generation


def test_fast_path_boot_seeds_ssd_used_from_live_records(tmp_path):
    """P3 iron Finding 1: a fast-path boot must seed cap accounting (_ssd_used) from the
    live journal records, like the full path - not from the blob-size watermark, which
    carries holes (discards, previous generations). A watermark-seeded _ssd_used
    over-evicts on the next stop or refuses the entire flush (tier erased).
    _blob_eof stays the never-decreased watermark: hole-safe record reservation.
    The 'session tier on' line reports the live figure on both boot paths."""
    d = str(tmp_path / "tier")
    store = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=d))
    chains = [_chain([(i, 0), (i, 1), (i, 2)]) for i in (1, 2)]
    for i, (keys, pages) in enumerate(chains, start=1):
        assert store.offer(f"p{i}".encode(), 3, pages, _snap(f"{i}"))
    # L2-only mode: offers flush straight into the blob; the graceful stop writes the
    # marker with blob_eof == getsize so the next boot takes the fast path.
    # Holes: grow the blob past the live set without journaling anything - 32 MiB so the
    # watermark and the live sum differ visibly in the GiB-rendered 'used=' log line.
    garbage = 32 * 2**20
    with open(os.path.join(d, "blob.bin"), "ab") as f:
        f.write(b"\x00" * garbage)
    _graceful_stop(store)                     # marker blob_eof == getsize -> fast path
    blob_size = os.path.getsize(os.path.join(d, "blob.bin"))
    live = sum((s.l2_n + 4096 - 1) // 4096 * 4096
               for s in store._segments.values() if s.in_l2)
    assert blob_size == live + garbage        # the watermark carries real holes

    with _tier_log_capture() as cap:
        boot = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=d))
        assert boot.replay_journal() == 2     # fast path (marker sizes match)
    # fails-before: the watermark (blob size) seeded the cap accounting
    assert boot._ssd_used == live
    assert boot._blob_eof == blob_size        # P3 invariant: watermark untouched
    assert boot._dead_bytes == garbage        # holes stay real for maybe_compact
    on = [m for m in cap.messages if "session tier on" in m]
    assert len(on) == 1 and "used=0.00 GiB" in on[0]

    # Full path on the same on-disk state: identical accounting and log figure.
    os.unlink(_marker_path(d))
    with _tier_log_capture() as cap:
        boot_full = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=d))
        assert boot_full.replay_journal() == 2
    assert boot_full._ssd_used == live
    assert boot_full._blob_eof == blob_size
    assert boot_full._dead_bytes == garbage
    on = [m for m in cap.messages if "session tier on" in m]
    assert len(on) == 1 and "used=0.00 GiB" in on[0]


def test_shutdown_marker_discard_invalidates(tmp_path, monkeypatch):
    """TP A1: a runtime discard rewrites no on-disk bytes (journal and blob eof
    untouched), so the marker's size checks cannot see it - the discard must drop the
    marker explicitly, or the next boot fast-paths a state that no longer matches the
    index and resurrects the evicted segment."""
    d = tmp_path / "disc"
    store = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d)))
    keys, pages = _chain([(1, 0), (1, 1), (1, 2)])
    assert store.offer(b"path", 3, pages, _snap("s"))
    _graceful_stop(store)
    assert os.path.exists(_marker_path(d))
    assert store.evict_store(1) == 1             # runtime discard
    assert not os.path.exists(_marker_path(d))   # dropped immediately
    calls = _counting_pread(monkeypatch)
    with _tier_log_capture() as cap:
        boot, n = _boot(str(d))
    assert n == 1                                # resurrect semantics unchanged
    assert calls["n"] == 1                       # full verification ran (no fast path)
    assert not any("fast-path" in m for m in cap.messages)


def test_read_shutdown_marker_garbage_variants(tmp_path):
    """TP B1: any garbage marker file (empty, short, bad crc, crc-valid non-dict JSON)
    reads back False without raising out of the boot replay."""
    d = tmp_path / "garbage"
    store = SessionTierStore(_cfg(d=str(d)))
    payload = json.dumps([1, 2, 3]).encode()     # crc-valid but NOT a dict
    variants = [b"",
                b"\x01\x02\x03",
                b"\x10\x00\x00\x00short",
                struct.pack("<II", 999, zlib.crc32(b"junk")) + b"junk",
                struct.pack("<II", len(payload), zlib.crc32(payload)) + payload]
    for i, framed in enumerate(variants):
        with open(_marker_path(d), "wb") as f:
            f.write(framed)
        res = store._read_shutdown_marker(0)
        # tolerate both the bool (pre-A2-fix) and (ok, parsed) tuple return shapes
        ok = res[0] if isinstance(res, tuple) else res
        assert ok is False, f"garbage variant {i} must not validate"


def test_shutdown_marker_generation_monotonic_across_stale(tmp_path, monkeypatch):
    """TP A2: a crc-valid but size-stale marker must still raise the in-memory
    generation floor - otherwise the next graceful stop emits a generation <= the
    rejected one and the monotonicity guarantee is void."""
    d = tmp_path / "gen"
    store = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d)))
    keys, pages = _chain([(1, 0), (1, 1), (1, 2)])
    assert store.offer(b"path", 3, pages, _snap("s"))
    _graceful_stop(store)                        # marker generation 1
    rec = {"journal_bytes": 123456, "blob_eof": 654321, "generation": 7}
    payload = json.dumps(rec, sort_keys=True).encode()
    with open(_marker_path(d), "wb") as f:       # stale sizes, valid crc, higher gen
        f.write(struct.pack("<II", len(payload), zlib.crc32(payload)) + payload)
    calls = _counting_pread(monkeypatch)
    boot, n = _boot(str(d))
    assert n == 1 and calls["n"] == 1            # stale sizes -> full verification
    boot.write_shutdown_marker()                 # graceful stop on this boot
    with open(_marker_path(d), "rb") as f:
        framed = f.read()
    plen, _ = struct.unpack_from("<II", framed, 0)
    mk = json.loads(framed[8:8 + plen])
    assert mk["generation"] > 7                  # strictly monotonic (pre-fix: 1)


# --------------------------------------------------------------- P3 flush write path


def _gold_payload(store, seg):
    """Sequential-path gold: the exact bytes the record occupies in the blob (payload
    plus zero padding to _BLK, as the O_DIRECT reader's aligned window expects)."""
    payload = bytearray()
    for i, ln in enumerate(seg.page_lens):
        payload += store._pool.read(seg.l1_page_offs[i], ln)
    for i, ln in enumerate(seg.snap_lens):
        if ln:
            payload += store._pool.read(seg.l1_snap_offs[i], ln)
    padded = (len(payload) + 4095) // 4096 * 4096
    return bytes(payload) + b"\0" * (padded - len(payload))


def test_flush_live_parallel_write_bytes_identical(tmp_path):
    """P3: the grouped parallel pwrite flush produces blob bytes byte-identical to the
    sequential append path - same payloads, same zero padding, journal offsets true."""
    d = str(tmp_path / "tier")
    store = SessionTierStore(_cfg(ram=16384, ssd=64 * 4096, d=d))
    chains = []
    for i in range(1, 6):
        keys, pages = _chain([(i, 0), (i, 1), (i, 2)])
        assert store.offer(f"p{i}".encode(), 3, pages, _snap(f"{i}"))
        chains.append((f"p{i}".encode(), keys, pages, f"{i}"))
    assert len([s for s in store._segments.values() if s.in_l1]) == 5
    gold = {name: _gold_payload(store, store._by_path(name)) for name, _, _, _ in chains}

    assert store.flush_live() == 5
    with open(os.path.join(d, "blob.bin"), "rb") as f:
        blob = f.read()
    boot = SessionTierStore(_cfg(ram=16384, ssd=64 * 4096, d=d))
    assert boot.replay_journal() == 5
    for name, keys, pages, tag in chains:
        seg = boot._by_path(name)
        assert seg.in_l2
        want = gold[name]
        assert blob[seg.l2_off:seg.l2_off + seg.l2_n] == want[:seg.l2_n]
        pad = blob[seg.l2_off + seg.l2_n:seg.l2_off + len(want)]
        assert pad == b"\0" * len(pad)          # padding zeros, byte-identical
        hit = boot.probe(keys)
        assert hit is not None and hit[0] == 3
        got_p, got_s = boot.restore(hit[1])
        assert got_p == [data for _, data in pages] and got_s == [_snap(tag)]


def test_flush_live_writes_segments_in_parallel(tmp_path, monkeypatch):
    """P3 fails-before: the flush path must overlap segment writes - concurrent os.pwrite
    entries prove >1 writer; the old sequential os.write append never overlaps (and never
    calls pwrite at all)."""
    import freetoken.scheduler.session_tier as st_mod

    d = str(tmp_path / "tier")
    store = SessionTierStore(_cfg(ram=16384, ssd=64 * 4096, d=d))
    for i in range(1, 5):
        keys, pages = _chain([(i, 0), (i, 1), (i, 2)])
        assert store.offer(f"p{i}".encode(), 3, pages, _snap(f"{i}"))
    assert len([s for s in store._segments.values() if s.in_l1]) == 4

    real_pwrite = os.pwrite
    state = {"entered": 0, "max": 0}
    lock = threading.Lock()

    def overlap_probe(fd, buf, offset):
        with lock:
            state["entered"] += 1
            state["max"] = max(state["max"], state["entered"])
        deadline = time.time() + 5.0
        while state["entered"] < 2 and time.time() < deadline:
            time.sleep(0.001)                   # bounded wait: never hangs the test
        try:
            return real_pwrite(fd, buf, offset)
        finally:
            with lock:
                state["entered"] -= 1

    monkeypatch.setattr(st_mod.os, "pwrite", overlap_probe)
    assert store.flush_live() == 4
    assert state["max"] >= 2                    # fails-before: sequential path gives 0/1


def test_flush_crash_between_groups_keeps_journaled_prefix(tmp_path, monkeypatch):
    """P3 crash sim: a SIGKILL after some flush journal records - the journaled records
    (their groups were made durable first) survive replay, the durable-but-unjournaled
    tail is dropped, and the journal holds only complete crc-valid records (no torn
    record is possible: blob durability precedes every journal append)."""
    d = str(tmp_path / "tier")
    store = SessionTierStore(_cfg(ram=16384, ssd=64 * 4096, d=d))
    chains = []
    for i in range(1, 5):
        keys, pages = _chain([(i, 0), (i, 1), (i, 2)])
        assert store.offer(f"p{i}".encode(), 3, pages, _snap(f"{i}"))
        chains.append((f"p{i}".encode(), keys, pages, f"{i}"))

    real_append = SessionTierStore._journal_append
    calls = {"n": 0}

    def die_after_two(self, *args):
        calls["n"] += 1
        if calls["n"] > 2:
            return False                        # "crash": the record never becomes durable
        return real_append(self, *args)

    monkeypatch.setattr(SessionTierStore, "_journal_append", die_after_two)
    assert store.flush_live() == 2              # the other two segments stay in L1
    monkeypatch.undo()

    boot = SessionTierStore(_cfg(ram=16384, ssd=64 * 4096, d=d))
    assert boot.replay_journal() == 2
    for name, keys, pages, tag in chains[:2]:
        hit = boot.probe(keys)
        assert hit is not None and hit[0] == 3
        got_p, got_s = boot.restore(hit[1])
        assert got_p == [data for _, data in pages]
        assert got_s == [_snap(tag)]
    for name, keys, pages, tag in chains[2:]:
        assert boot.probe(keys) is None         # unjournaled durable tail: dropped,
        # never resurrected

    raw = (tmp_path / "tier" / "journal.log").read_bytes()
    pos, nrec = 0, 0
    while pos + 8 <= len(raw):
        plen, crc = struct.unpack_from("<II", raw, pos)
        assert pos + 8 + plen <= len(raw)       # no torn record: header never lies
        payload = raw[pos + 8:pos + 8 + plen]
        assert zlib.crc32(payload) == crc
        pos += 8 + plen
        nrec += 1
    assert nrec == 2 and pos == len(raw)        # journaled prefix is exactly the whole file


def test_flush_group_fdatasync_failure_aborts_group(tmp_path, monkeypatch):
    """Invariant guard: if the group fdatasync fails, NO journal record of the group may
    appear - the journal only ever follows durable blob bytes."""
    import freetoken.scheduler.session_tier as st_mod

    d = str(tmp_path / "tier")
    store = SessionTierStore(_cfg(ram=16384, ssd=64 * 4096, d=d))
    for i in range(1, 4):
        keys, pages = _chain([(i, 0), (i, 1)])
        assert store.offer(f"p{i}".encode(), 2, pages, _snap(f"{i}"))

    def boom(fd):
        raise OSError(5, "simulated fdatasync failure")

    monkeypatch.setattr(st_mod.os, "fdatasync", boom)
    assert store.flush_live() == 0
    monkeypatch.undo()
    assert (tmp_path / "tier" / "journal.log").read_bytes() == b""
    assert len([s for s in store._segments.values() if s.in_l1]) == 3   # nothing lost
    assert store._ssd_used == 0                 # aborted group took no capacity accounting


def test_flush_volume_groups_all_durable(tmp_path, monkeypatch):
    """Grouping by volume: with a tiny group budget every segment becomes its own group;
    the durable order (fdatasync before that group's journal records) still yields a
    complete journal and byte-identical restores."""
    import freetoken.scheduler.session_tier as st_mod

    monkeypatch.setattr(st_mod, "_FLUSH_GROUP_BYTES", 1)
    d = str(tmp_path / "tier")
    store = SessionTierStore(_cfg(ram=16384, ssd=64 * 4096, d=d))
    chains = []
    for i in range(1, 4):
        keys, pages = _chain([(i, 0), (i, 1), (i, 2)])
        assert store.offer(f"p{i}".encode(), 3, pages, _snap(f"{i}"))
        chains.append((keys, pages))
    assert store.flush_live() == 3
    boot = SessionTierStore(_cfg(ram=16384, ssd=64 * 4096, d=d))
    assert boot.replay_journal() == 3
    for keys, pages in chains:
        hit = boot.probe(keys)
        assert hit is not None and hit[0] == 3
        assert boot.restore(hit[1])[0] == [data for _, data in pages]


# ------------------------------------------------------------ CRC versioning (P3.3)


def _journal_records_raw(path):
    """Parse a journal file end-to-end; returns (records, fully_parsed)."""
    with open(path, "rb") as f:
        raw = f.read()
    pos, recs = 0, []
    while pos + 8 <= len(raw):
        plen, crc = struct.unpack_from("<II", raw, pos)
        assert pos + 8 + plen <= len(raw)       # frame format unchanged by the versioning
        payload = raw[pos + 8:pos + 8 + plen]
        assert zlib.crc32(payload) == crc       # frame crc stays zlib for every record
        recs.append(json.loads(payload))
        pos += 8 + plen
    return recs, pos == len(raw)


def _blob_region(d, rec):
    with open(os.path.join(d, "blob.bin"), "rb") as f:
        f.seek(rec["off"])
        return f.read(rec["n"])


def _rewrite_records(d, transform):
    """Rewrite the journal file record-by-record: transform(rec) may mutate or drop the
    crc_alg field; frames stay zlib."""
    jpath = os.path.join(d, "journal.log")
    recs, _ = _journal_records_raw(jpath)
    with open(jpath, "wb") as f:
        for rec in recs:
            transform(rec)
            payload = json.dumps(rec, sort_keys=True).encode()
            f.write(struct.pack("<II", len(payload), zlib.crc32(payload)) + payload)


def test_new_records_carry_crc32c_and_verify(tmp_path):
    """P3.3: new journal records carry crc_alg=crc32c with a Castagnoli payload crc and
    verify through the replay full path; restores stay byte-exact."""
    import crc32c

    d = str(tmp_path / "tier")
    store = SessionTierStore(_cfg(ram=16384, ssd=64 * 4096, d=d))
    chains = []
    for i in range(1, 4):
        keys, pages = _chain([(i, 0), (i, 1), (i, 2)])
        assert store.offer(f"p{i}".encode(), 3, pages, _snap(f"{i}"))
        chains.append((f"p{i}".encode(), keys, pages))
    assert store.flush_live() == 3

    recs, whole = _journal_records_raw(os.path.join(d, "journal.log"))
    assert whole and len(recs) == 3
    for rec in recs:
        assert rec["crc_alg"] == "crc32c"                       # fails-before: no field
        assert rec["crc"] == crc32c.crc32c(_blob_region(d, rec))
    boot = SessionTierStore(_cfg(ram=16384, ssd=64 * 4096, d=d))
    assert boot.replay_journal() == 3                           # full-path crc32c verify
    for name, keys, pages in chains:
        hit = boot.probe(keys)
        assert hit is not None and hit[0] == 3
        assert boot.restore(hit[1])[0] == [data for _, data in pages]


def test_legacy_zlib_records_without_field_verify(tmp_path, monkeypatch):
    """Backward compat: a catalog whose records carry zlib crc32 WITHOUT crc_alg (every
    pre-P3.3 directory) takes the full verification path and replays intact."""
    d = str(tmp_path / "tier")
    store = SessionTierStore(_cfg(ram=16384, ssd=64 * 4096, d=d))
    chains = []
    for i in range(1, 4):
        keys, pages = _chain([(i, 0), (i, 1), (i, 2)])
        assert store.offer(f"p{i}".encode(), 3, pages, _snap(f"{i}"))
        chains.append((f"p{i}".encode(), keys, pages))
    assert store.flush_live() == 3

    def to_legacy(rec):
        rec.pop("crc_alg")
        rec["crc"] = zlib.crc32(_blob_region(d, rec))

    _rewrite_records(d, to_legacy)
    calls = _counting_pread(monkeypatch)
    boot = SessionTierStore(_cfg(ram=16384, ssd=64 * 4096, d=d))
    assert boot.replay_journal() == 3                           # full path, no drops
    assert calls["n"] == 3                                      # every payload re-read
    for name, keys, pages in chains:
        hit = boot.probe(keys)
        assert hit is not None and hit[0] == 3
        assert boot.restore(hit[1])[0] == [data for _, data in pages]


def test_mixed_zlib_and_crc32c_catalog(tmp_path):
    """A catalog mixing legacy zlib records (no field) and crc32c records (field) replays
    completely: the verifier selects the algorithm per record."""
    import crc32c

    d = str(tmp_path / "tier")
    store = SessionTierStore(_cfg(ram=16384, ssd=64 * 4096, d=d))
    names = []
    for i in range(1, 5):
        keys, pages = _chain([(i, 0), (i, 1), (i, 2)])
        assert store.offer(f"p{i}".encode(), 3, pages, _snap(f"{i}"))
        names.append(f"p{i}".encode())
    assert store.flush_live() == 4

    seen_algs = set()

    def mix(rec):
        data = _blob_region(d, rec)
        if len(seen_algs) % 2 == 0:             # keep alternating: even -> legacy zlib
            rec.pop("crc_alg")
            rec["crc"] = zlib.crc32(data)
            seen_algs.add("zlib")
        else:
            rec["crc"] = crc32c.crc32c(data)    # odd -> keep crc32c field
            seen_algs.add("crc32c")

    _rewrite_records(d, mix)
    assert seen_algs == {"zlib", "crc32c"}
    boot = SessionTierStore(_cfg(ram=16384, ssd=64 * 4096, d=d))
    assert boot.replay_journal() == 4
    for i, name in enumerate(names, start=1):
        keys, pages = _chain([(i, 0), (i, 1), (i, 2)])
        hit = boot.probe(keys)
        assert hit is not None and hit[0] == 3
        assert boot.restore(hit[1])[0] == [data for _, data in pages]


def test_corrupted_crc32c_record_dropped(tmp_path):
    """A crc32c record whose blob region no longer matches is dead: dropped at replay
    (with the dropped-report intact), its neighbors survive."""
    d = str(tmp_path / "tier")
    store = SessionTierStore(_cfg(ram=16384, ssd=64 * 4096, d=d))
    keys_by_name = {}
    for i in range(1, 4):
        keys, pages = _chain([(i, 0), (i, 1), (i, 2)])
        assert store.offer(f"p{i}".encode(), 3, pages, _snap(f"{i}"))
        keys_by_name[f"p{i}".encode()] = keys
    assert store.flush_live() == 3

    recs, _ = _journal_records_raw(os.path.join(d, "journal.log"))
    victim = recs[1]
    with open(os.path.join(d, "blob.bin"), "r+b") as f:   # corrupt the middle record
        f.seek(victim["off"])
        f.write(b"\xAA" * 16)
    with _tier_log_capture() as cap:
        boot = SessionTierStore(_cfg(ram=16384, ssd=64 * 4096, d=d))
        assert boot.replay_journal() == 2
    assert any("replay dropped 1 dead/torn records" in m for m in cap.messages)
    hit = boot.probe(keys_by_name[b"p2"])
    assert hit is None                                     # the victim stays dead
    for name, i in ((b"p1", 1), (b"p3", 3)):
        keys, pages = _chain([(i, 0), (i, 1), (i, 2)])
        hit = boot.probe(keys)
        assert hit is not None and hit[0] == 3
        assert boot.restore(hit[1])[0] == [data for _, data in pages]


def test_crc32c_records_take_p2_fast_path(tmp_path):
    """P2 fast-path intact: after a graceful stop the boot replays crc32c records via
    the marker with ZERO payload re-reads; the full path stays the crash fallback."""
    import freetoken.scheduler.session_tier as st_mod

    d = str(tmp_path / "tier")
    store = SessionTierStore(_cfg(ram=16384, ssd=64 * 4096, d=d))
    for i in range(1, 3):
        keys, pages = _chain([(i, 0), (i, 1), (i, 2)])
        assert store.offer(f"p{i}".encode(), 3, pages, _snap(f"{i}"))
    _graceful_stop(store)

    real_pread = os.pread
    counter = {"n": 0}

    def counting(fd, n, off):
        counter["n"] += 1
        return real_pread(fd, n, off)

    st_mod.os.pread = counting
    try:
        boot, n = _boot(d)
        assert n == 2
        assert counter["n"] == 0                    # fast path: no payload crc re-read
    finally:
        st_mod.os.pread = real_pread


# ----------------------------------------------------- TP fixes (review A/B, P3)


def test_flush_live_survives_cap_eviction_mid_iteration(tmp_path, monkeypatch):
    """A1 fails-before: flush_live iterated self._segments.values() directly while a
    mid-loop cap eviction (_flush_group -> _reserve_blob_span -> _evict_oldest_l2 ->
    _discard) deleted from the same dict -> RuntimeError. Port of boot-shutdown-io/
    p3-plan/repro_flushlive_dict.py: with the shrunk cap, reserving the first flush
    segment evicts BOTH runtime-demoted L2 victims mid-iteration."""
    import freetoken.scheduler.session_tier as st_mod

    d = str(tmp_path / "tier")
    monkeypatch.setattr(st_mod, "_FLUSH_GROUP_BYTES", 2048)   # force a mid-loop group flush
    store = SessionTierStore(_cfg(ram=2048, ssd=64 * 4096, d=d))
    for i in range(1, 5):
        keys, pages = _chain([(i, 0), (i, 1), (i, 2)])
        assert store.offer(f"p{i}".encode(), 3, pages, _snap(f"{i}"))
    assert [s.seg_id for s in store._segments.values() if s.in_l2]  # runtime L2 victims

    store.ssd_bytes = 4096                    # reserving s3 evicts BOTH L2 victims
    # ... mid-iteration, then s3's fresh span is itself the victim s4's reserve needs
    count = store.flush_live()                # fails-before: RuntimeError here
    assert count == 2                         # s3 written, then sacrificed for s4
    assert not [s for s in store._segments.values() if s.in_l2 and s.seg_id <= 2]
    boot = SessionTierStore(_cfg(ram=2048, ssd=64 * 4096, d=d))
    # p1/p2 records resurrect too: a runtime discard rewrites no journal, and their blob
    # regions were never overwritten (documented resurrect semantics).
    assert boot.replay_journal() == 4
    for i in (1, 2, 3, 4):
        keys, pages = _chain([(i, 0), (i, 1), (i, 2)])
        hit = boot.probe(keys)
        assert hit is not None and hit[0] == 3
        assert boot.restore(hit[1])[0] == [data for _, data in pages]


def test_partial_pwrite_failure_keeps_segment_in_l1(tmp_path, monkeypatch):
    """B1: one segment's pwrite fails inside a group -> flush_live counts M-1, the
    journal holds only successes, the failed segment stays in L1 with clean blob fields,
    _ssd_used accounts M-1 records, boot replays M-1 and the survivors are byte-exact."""
    import freetoken.scheduler.session_tier as st_mod

    d = str(tmp_path / "tier")
    store = SessionTierStore(_cfg(ram=16384, ssd=64 * 4096, d=d))
    chains = []
    for i in range(1, 5):
        keys, pages = _chain([(i, 0), (i, 1), (i, 2)])
        assert store.offer(f"p{i}".encode(), 3, pages, _snap(f"{i}"))
        chains.append((f"p{i}".encode(), keys, pages))

    real_pwrite = os.pwrite
    calls = {"n": 0}

    def flaky(fd, buf, offset):
        calls["n"] += 1
        if calls["n"] == 1:                   # exactly one group segment fails
            raise OSError(5, "simulated pwrite failure")
        return real_pwrite(fd, buf, offset)

    monkeypatch.setattr(st_mod.os, "pwrite", flaky)
    assert store.flush_live() == 3            # the failed one is not counted
    monkeypatch.undo()
    assert store._ssd_used == 3 * 4096        # no accounting for the aborted span
    failed = [s for s in store._segments.values() if s.in_l1]
    assert len(failed) == 1 and failed[0].blob == "" and failed[0].l2_off == -1
    boot = SessionTierStore(_cfg(ram=16384, ssd=64 * 4096, d=d))
    assert boot.replay_journal() == 3
    for name, keys, pages in chains:
        hit = boot.probe(keys)
        if name.decode() == failed[0].path_key.decode():
            assert hit is None                # unjournaled: dropped, not resurrected
        else:
            assert hit is not None and hit[0] == 3
            assert boot.restore(hit[1])[0] == [data for _, data in pages]


def test_failed_write_hole_skipped_by_next_group_no_dropped(tmp_path, monkeypatch):
    """B2: a failed group-1 write leaves a hole; the next group reserves AFTER it (the
    watermark intentionally stays past the hole). Boot takes the full path, no record
    references the hole, and 'replay dropped' is ABSENT - nothing looks dead."""
    import freetoken.scheduler.session_tier as st_mod

    d = str(tmp_path / "tier")
    monkeypatch.setattr(st_mod, "_FLUSH_GROUP_BYTES", 1)   # every segment = own group
    store = SessionTierStore(_cfg(ram=16384, ssd=64 * 4096, d=d))
    chains = []
    for i in range(1, 4):
        keys, pages = _chain([(i, 0), (i, 1), (i, 2)])
        assert store.offer(f"p{i}".encode(), 3, pages, _snap(f"{i}"))
        chains.append((f"p{i}".encode(), keys, pages))

    real_pwrite = os.pwrite
    calls = {"n": 0}

    def flaky(fd, buf, offset):
        calls["n"] += 1
        if calls["n"] == 1:                   # group 1's single segment fails
            raise OSError(5, "simulated pwrite failure")
        return real_pwrite(fd, buf, offset)

    monkeypatch.setattr(st_mod.os, "pwrite", flaky)
    assert store.flush_live() == 2
    monkeypatch.undo()

    recs, whole = _journal_records_raw(os.path.join(d, "journal.log"))
    assert whole and len(recs) == 2
    assert all(rec["off"] >= 4096 for rec in recs)   # the hole [0, 4096) is unreferenced
    with _tier_log_capture() as cap:
        boot = SessionTierStore(_cfg(ram=16384, ssd=64 * 4096, d=d))
        assert boot.replay_journal() == 2
    assert not any("replay dropped" in m for m in cap.messages)   # nothing looks dead
    for name, keys, pages in chains[1:]:
        hit = boot.probe(keys)
        assert hit is not None and hit[0] == 3
        assert boot.restore(hit[1])[0] == [data for _, data in pages]


def test_marker_valid_after_failed_flush_boot_fast_path(tmp_path, monkeypatch):
    """B3: a graceful stop after a flush with a failed write leaves a VALID marker (the
    compacted sizes are self-consistent); boot takes the fast path with zero payload
    re-reads and the unjournaled failed segment is dropped. Triage A2 pin: the watermark
    is NOT trimmed on failure - trimming would race the parallel writers and break the
    never-decreased invariant; the hole is reclaimed by compaction instead."""
    import freetoken.scheduler.session_tier as st_mod

    d = str(tmp_path / "tier")
    monkeypatch.setattr(st_mod, "_FLUSH_GROUP_BYTES", 1)   # deterministic group order
    store = SessionTierStore(_cfg(ram=16384, ssd=64 * 4096, d=d))
    keys_by_name = {}
    for i in range(1, 4):
        keys, pages = _chain([(i, 0), (i, 1), (i, 2)])
        assert store.offer(f"p{i}".encode(), 3, pages, _snap(f"{i}"))
        keys_by_name[f"p{i}".encode()] = keys

    victim_page0 = _page_data((1, 0))         # p1's first page: fail its writes forever
    real_pwrite = os.pwrite

    def flaky(fd, buf, offset):
        if bytes(buf[:PAGE]) == victim_page0:
            raise OSError(5, "simulated pwrite failure")
        return real_pwrite(fd, buf, offset)

    monkeypatch.setattr(st_mod.os, "pwrite", flaky)
    assert store.flush_live() == 2            # p1 fails, p2/p3 journal
    _graceful_stop(store)                     # p1's retry fails too; marker written
    monkeypatch.undo()

    calls = _counting_pread(monkeypatch)
    boot, n = _boot(d)
    assert n == 2 and calls["n"] == 0         # fast path accepted, no payload re-reads
    assert boot.probe(keys_by_name[b"p1"]) is None    # unjournaled tail: dropped
    for name in (b"p2", b"p3"):
        i = int(name[1:])
        keys, pages = _chain([(i, 0), (i, 1), (i, 2)])
        hit = boot.probe(keys)
        assert hit is not None and hit[0] == 3
        assert boot.restore(hit[1])[0] == [data for _, data in pages]


def test_flush_compact_offer_boot_two_generations(tmp_path):
    """Triage (7)+(B8): flush -> compact -> runtime offer -> boot-2. The runtime offer
    writes PAST the compact-resynced _blob_eof (the lseek resync at compact), its record
    lands at the truncated file's real EOF, and BOTH generations of records (pre-compact
    survivors and the post-compact offer) replay and restore byte-exactly."""
    d = tmp_path / "tier"
    store = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d)))   # L2-only runtime offers
    for i in (1, 2, 3):
        keys, pages = _chain([(i, 0), (i, 1), (i, 2)])
        assert store.offer(f"p{i}".encode(), 3, pages, _snap(f"{i}"))
    assert store.evict_store(1) == 1          # p1 is now dead
    assert store.compact() == 1               # blob truncated, _blob_eof resynced
    eof = store._blob_eof
    assert eof == 3 * 4096                    # p1's head hole + two live records at
    # ... their ORIGINAL offsets: the file ends at the last live record
    keys4, pages4 = _chain([(4, 0), (4, 1), (4, 2)])
    assert store.offer("p4".encode(), 3, pages4, _snap("4"))
    recs, whole = _journal_records_raw(str(d / "journal.log"))
    assert whole and len(recs) == 3
    assert recs[-1]["off"] == eof             # the offer wrote past the resynced EOF

    boot = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d)))
    assert boot.replay_journal() == 3         # both generations alive
    for i in (2, 3, 4):
        keys, pages = _chain([(i, 0), (i, 1), (i, 2)])
        hit = boot.probe(keys)
        assert hit is not None and hit[0] == 3
        assert boot.restore(hit[1])[0] == [data for _, data in pages]


def test_group_reserve_respects_ssd_cap(tmp_path):
    """Cap-regression fix: _ssd_used lags until the group's journal records land, so the
    group's reserve loop must count its pending reservations - otherwise the whole group
    reserves past the ssd cap (HEAD enforced a strict per-segment cap: at exhaustion a
    segment is REFUSED and stays in L1). Accounting must converge with what the journal
    actually occupies, and _blob_eof (never decreased) is unaffected."""
    d = str(tmp_path / "tier")
    store = SessionTierStore(_cfg(ram=16384, ssd=4 * 4096, d=d))   # cap = 4 records
    chains = []
    for i in range(1, 9):
        keys, pages = _chain([(i, 0), (i, 1), (i, 2)])
        assert store.offer(f"p{i}".encode(), 3, pages, _snap(f"{i}"))
        chains.append((f"p{i}".encode(), keys, pages))
    assert len([s for s in store._segments.values() if s.in_l1]) == 8

    assert store.flush_live() == 4            # fails-before: 8, the whole group over cap
    assert store._ssd_used == 4 * 4096        # exactly the cap, never past it
    recs, whole = _journal_records_raw(os.path.join(d, "journal.log"))
    assert whole and len(recs) == 4           # only the admitted segments are journaled
    failed = [s for s in store._segments.values() if s.in_l1]
    assert len(failed) == 4                   # rejected segments stay in L1 (HEAD way)
    assert all(s.blob == "" and s.l2_off == -1 for s in failed)
    for i in range(5, 9):
        keys, _ = _chain([(i, 0), (i, 1), (i, 2)])
        assert store.probe(keys) is not None  # still served from L1

    boot = SessionTierStore(_cfg(ram=16384, ssd=4 * 4096, d=d))
    assert boot.replay_journal() == 4
    assert boot._ssd_used == 4 * 4096         # accounting converged with the journal
    for i in range(1, 5):
        keys, pages = _chain([(i, 0), (i, 1), (i, 2)])
        hit = boot.probe(keys)
        assert hit is not None and hit[0] == 3
        assert boot.restore(hit[1])[0] == [data for _, data in pages]
