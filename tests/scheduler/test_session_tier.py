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
