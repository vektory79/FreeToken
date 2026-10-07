"""SessionTierStore unit tests (tiering W1): content addressing, dedup, L1/L2 LRU,
journal crash-safety, byte-identical restore, off-mode. CPU only, no VRAM."""
from __future__ import annotations

import contextlib
import itertools
import json
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


@pytest.fixture(autouse=True)
def _flush_env_clean(monkeypatch):
    """P1 lesson (env-config-hygiene): a shell-exported FREETOKEN_* env var silently
    overrides what a test thinks it is testing - drop all three knobs unless the test
    sets them explicitly."""
    monkeypatch.delenv("FREETOKEN_FLUSH_WRITERS", raising=False)
    monkeypatch.delenv("FREETOKEN_FLUSH_GROUP_BYTES", raising=False)
    monkeypatch.delenv("FREETOKEN_COMPACT_WRITERS", raising=False)


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


def test_probe_boundary_exact_skips_mid_span_returns_deepest_boundary(tmp_path):
    """P1 (M2c shape at HW depths 1657/1393/1328): the hybrid probe contract skips
    mid-span matches - a segment with no live GDN snapshot at the matched depth can never
    serve a hybrid restore (the snap gate would refuse) - and returns the DEEPEST
    boundary-exact candidate instead (1328 over 1200). The default walk is untouched:
    KV-only managers keep mid-span matches (restore depth-truncates the pages)."""
    store = SessionTierStore(_cfg(ram=1 << 21))
    keys, pages = _chain([(1, i) for i in range(1657)])
    assert store.offer(b"path-tip", 1657, pages, _snap("tip"))
    assert store.offer(b"path-old", 1328, pages[:1328], _snap("old"))
    assert store.offer(b"path-shallow", 1200, pages[:1200], _snap("sh"))

    request = keys[:1393] + [chain_page_key(keys[1392], (9, 9)),
                             chain_page_key(chain_page_key(keys[1392], (9, 9)), (9, 9))]
    # default walk unchanged: the deepest common page wins, mid-span included (KV-only)
    mid = store.probe(request)
    assert mid is not None and mid[0] == 1393 and mid[1].path_key == b"path-tip"
    # hybrid walk: the 1393 match is inside the tip (its snapshot lives at 1657) -
    # skipped; the deepest boundary-exact candidate is the 1328 segment
    hit = store.probe(request, boundary_exact=True)
    assert hit is not None and hit[0] == 1328 and hit[1].path_key == b"path-old"
    got, snaps = store.restore(hit[1])
    assert got == [data for _, data in pages[:1328]] and snaps == [_snap("old")]


def test_probe_boundary_exact_without_candidate_misses_once_and_counts(tmp_path):
    """P1 honesty: with ONLY mid-span matches reachable the hybrid walk reports 'no
    candidate' - one probe_miss, no per-depth inflation - and the mid-span-only shape
    keeps its own counter instead of pretending a servable hit. The sighting needs the
    walk to CROSS a deeper segment mid-span at a visited boundary depth below the
    divergence (here: the boundary-3 walk depth sees the boundary-5 segment)."""
    store = SessionTierStore(_cfg(ram=1 << 12))
    keys, pages = _chain([(7, i) for i in range(5)])
    assert store.offer(b"path-deep", 5, pages, _snap("deep"))
    x_keys, x_pages = _chain([(7, 0), (7, 1), (9, 2)])
    assert store.offer(b"path-x", 3, x_pages)
    request = keys[:4] + [chain_page_key(keys[3], (8, 8))]
    assert store.probe(request, boundary_exact=True) is None
    snap = store.snapshot()
    assert snap["probe_miss"] == 1 and snap["probe_hit"] == 0
    assert snap["probe_midspan_only"] == 1       # the boundary-5 seg crossed at depth 3
    mid = store.probe(request)                   # default walk still returns the mid-span 4
    assert mid is not None and mid[0] == 4 and mid[1].path_key == b"path-deep"
    snap = store.snapshot()
    assert snap["probe_miss"] == 1 and snap["probe_hit"] == 1
    assert snap["probe_midspan_only"] == 1


def test_probe_boundary_exact_skips_snapshot_free_serves_shallower_bearer(tmp_path):
    """W2 fails-before: a snapshot-FREE boundary-exact segment (a KV-only tip/spare
    offer left by a restore-finish adoption) used to win the boundary-exact walk and be
    refused at the tail's snap gate, masking the shallower snapshot-bearing candidate
    below it. The KV-only default walk still serves the deep snapshot-free segment."""
    store = SessionTierStore(_cfg(ram=1 << 12))
    keys, pages = _chain([(7, i) for i in range(5)])
    assert store.offer(b"path-bare", 5, pages)          # snapshot-free tip: no snap arg
    assert store.offer(b"path-bear", 3, pages[:3], _snap("bear"))
    hit = store.probe(keys, boundary_exact=True)
    assert hit is not None and hit[0] == 3 and hit[1].path_key == b"path-bear"
    got, snaps = store.restore(hit[1])
    assert got == [data for _, data in pages[:3]] and snaps == [_snap("bear")]
    bare = store.probe(keys)                            # KV-only regression: walk unchanged
    assert bare is not None and bare[0] == 5 and bare[1].path_key == b"path-bare"
    assert store.restore(bare[1])[1] == []              # restorable, just snapshot-free


def test_probe_snapfree_skip_counts_only_snapshot_free_skips(tmp_path):
    """W3 (review 1298048): the snapshot-free boundary-exact skip has its own preseeded
    counter - it counts ONLY that skip: a plain boundary-exact hit, the KV-only walk and
    a mid-span-only sighting (no snapshot-free candidate involved) leave it untouched."""
    store = SessionTierStore(_cfg(ram=1 << 12))
    keys, pages = _chain([(7, i) for i in range(5)])
    assert store.offer(b"path-bare", 5, pages)          # snapshot-free tip: no snap arg
    assert store.offer(b"path-bear", 3, pages[:3], _snap("bear"))
    hit = store.probe(keys[:3], boundary_exact=True)    # plain exact hit: no skip
    assert hit is not None and hit[0] == 3 and hit[1].path_key == b"path-bear"
    assert store.snapshot()["probe_snapfree_skip"] == 0
    hit = store.probe(keys, boundary_exact=True)        # the W2 shape: depth-5 snapshot-free
    assert hit is not None and hit[0] == 3              # candidate skipped (counted once),
    assert store.snapshot()["probe_snapfree_skip"] == 1 # the shallower bearer still wins
    bare = store.probe(keys)                            # KV-only walk serves the bare tip
    assert bare is not None and bare[0] == 5            # itself - no skip counted there
    assert store.snapshot()["probe_snapfree_skip"] == 1
    # mid-span-only sighting with NO snapshot-free candidate: probe_midspan_only owns it
    store2 = SessionTierStore(_cfg(ram=1 << 12))
    d_keys, d_pages = _chain([(7, i) for i in range(5)])
    assert store2.offer(b"path-deep", 5, d_pages, _snap("deep"))
    s_keys, s_pages = _chain([(7, 0), (7, 1), (9, 2)])
    assert store2.offer(b"path-side", 3, s_pages, _snap("side"))
    request = d_keys[:4] + [chain_page_key(d_keys[3], (8, 8))]
    assert store2.probe(request, boundary_exact=True) is None
    snap = store2.snapshot()
    assert snap["probe_midspan_only"] == 1
    assert snap["probe_snapfree_skip"] == 0


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

    # uncompacted crash: P13 - the superseded record is tombstoned at discard, so it
    # stays dead across the reboot (pre-P13 it resurrected and a deeper offer had to
    # supersede BOTH stale segments before compaction converged)
    d2 = tmp_path / "tier2"
    store2 = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d2)))
    keys, pages = _chain([(8, 0), (8, 1), (8, 2)])
    assert store2.offer(b"path", 1, pages[:1])
    assert store2.offer(b"path", 2, pages[:2])           # supersede #1
    reborn = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d2)))
    assert reborn.replay_journal() == 1                  # the tombstone holds pre-compact
    assert len(reborn._segments) == 1
    assert reborn.offer(b"path", 3, pages)               # supersedes both stale segments
    assert len(reborn._segments) == 1
    assert reborn.compact() == 2
    final = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d2)))
    assert final.replay_journal() == 1
    hit = final.probe(keys)
    assert final.restore(hit[1])[0] == [data for _, data in pages]


def test_offers_dedup_metric_preseeded_and_in_stats_line(tmp_path):
    """offers_dedup is pre-seeded in the Counter (snapshot() copies it whole) and lands in
    the stats_line fragment (batch log + shutdown final line via tier_stats_line).
    probe_midspan_only must be pre-seeded too: snapshot() copies the Counter whole, so an
    unseeded key is missing (KeyError) until its first sighting."""
    store = SessionTierStore(_cfg(d=str(tmp_path)))
    assert store.snapshot()["offers_dedup"] == 0
    assert store.snapshot()["probe_midspan_only"] == 0   # pre-seeded before any sighting
    assert store.snapshot()["probe_snapfree_skip"] == 0  # W3: pre-seeded like its neighbor
    line = store.stats_line()
    assert "dedup=0" in line
    assert "midsn=0" in line and "prej=0" in line and "snapfree=0" in line
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


@pytest.fixture
def tier_log_capture(log_capture_factory):
    """Module binding of the conftest factory: this module's non-propagating logger."""
    return log_capture_factory("freetoken.scheduler.session_tier")


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
    # the records live in L2 + journal now (flush demoted them): a torn tail must keep
    # them all - no discards here, so no tombstones either

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


def test_compact_drops_dead_records_and_crc_last_resort(tmp_path):
    """N5 (P9-reworked): compaction keeps exactly the live segments and SHRINKS the file
    to the live volume; twice-compaction and zero-dead compaction are no-ops. The
    pre-P9 'stale journal against the new blob' crash window is unreachable now (the
    swap token completes the pair at boot - see the SIGKILL matrix), so the payload-crc
    last resort is asserted against a hand-tampered pair: stale offsets mismatch and
    drop, garbage never resurrects."""
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
    assert (d / "blob.bin").stat().st_size == 2 * 4096     # P9: file == live volume
    assert store.compact() == 0                            # idempotent no-op

    # hand-tampered pair (stale journal + shrunk blob): every stale offset mismatches
    # (moved or past-eof) and replay drops all of them
    (d / "journal.log").write_bytes(stale_journal)
    reborn = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d)))
    assert reborn.replay_journal() == 0                    # crc last resort, no garbage
    assert reborn.probe([chain_page_key(None, (1,))]) is None
    # all 3 journal records are now dead (their holders were dropped): the next
    # compact reclaims the WHOLE file down to the (empty) live volume
    assert reborn.compact() == 3
    assert (d / "blob.bin").stat().st_size == 0


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
    # P9: the two survivors are DENSELY repacked - the file IS the live volume now
    assert (d / "blob.bin").stat().st_size == 2 * 4096
    assert store._blob_eof == 2 * 4096     # resynced to the shrunk file's real EOF

    monkeypatch.setattr(st_mod, "_COMPACT_DEAD_FRACTION", 0.5)
    for i in (5, 6):
        k = chain_page_key(None, (i,))
        assert store.offer(bytes([i]) * 16, 1, [(k, bytes([i]) * PAGE)])
    assert store.evict_store(1) == 1
    assert store._dead_bytes == 4096 and store._blob_eof == 4 * 4096       # 25% < 50%
    assert store.maybe_compact() == 0      # below the watermark: no trigger
    monkeypatch.setattr(st_mod, "_COMPACT_DEAD_FRACTION", 0.1)
    assert store.maybe_compact() == 1
    assert store._dead_bytes == 0
    assert store._blob_eof == 3 * 4096     # shrunk again to the live volume


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


# ------------------------------------------------- P8: record-based compact gate


def _boot_with_watermark_holes(tmp_path, subdir="p8", garbage=32 * 2**20):
    """P5-iron finding-1 state: watermark holes (blob-tail garbage that survived a
    graceful stop) but ZERO dead journal records - the state that used to gate
    maybe_compact on every ok-offer while compact() no-op'ed."""
    d = str(tmp_path / subdir)
    store = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=d))
    for i in (1, 2, 3):
        k = chain_page_key(None, (i,))
        assert store.offer(bytes([i]) * 16, 1, [(k, bytes([i]) * PAGE)])
    with open(os.path.join(d, "blob.bin"), "ab") as f:
        f.write(b"\x00" * garbage)
    _graceful_stop(store)                 # compact no-op (0 dead records); marker matches
    return d, garbage


def _counting_compact(store, monkeypatch):
    """Pass-through counters around the instance's _parse_journal/compact."""
    calls = {"parse": 0, "compact": 0}
    real_parse, real_compact = store._parse_journal, store.compact

    def parse():
        calls["parse"] += 1
        return real_parse()

    def compact():
        calls["compact"] += 1
        return real_compact()

    monkeypatch.setattr(store, "_parse_journal", parse)
    monkeypatch.setattr(store, "compact", compact)
    return calls


def test_watermark_holes_zero_dead_records_offer_skips_compact(tmp_path, monkeypatch):
    """P8 (a): watermark holes (72-88 GiB in the iron run) with zero dead journal
    records must never gate compaction - an ok-offer must not parse the journal nor
    call compact: every ok-offer used to pay a full _parse_journal for a guaranteed
    no-op (P5-iron: 52 offers x 1.5 MiB reads + 270 JSON records each)."""
    import freetoken.scheduler.session_tier as st_mod

    d, garbage = _boot_with_watermark_holes(tmp_path)
    monkeypatch.setattr(st_mod, "_COMPACT_FLOOR_BYTES", 4096)
    store = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=d))
    assert store.replay_journal() == 3    # fast path: no discard ever dropped the marker
    assert store._dead_bytes == garbage   # the watermark dead stays, exactly as before
    calls = _counting_compact(store, monkeypatch)

    assert store.maybe_compact() == 0     # record gate: nothing reclaimable
    assert calls == {"parse": 0, "compact": 0}
    k = chain_page_key(None, (9,))
    assert store.offer(bytes([9]) * 16, 1, [(k, bytes([9]) * PAGE)])
    assert calls == {"parse": 0, "compact": 0}   # the ok-offer stayed cheap
    hit = store.probe([k])
    assert hit is not None and hit[0] == 1
    # ledger exactness (fix side): the watermark holes never count as dead records
    assert store._dead_records == 0 and store._dead_record_bytes == 0


def test_dead_records_still_gate_compact_and_drop_exactly_them(tmp_path, monkeypatch):
    """P8 (b): real dead records must compact exactly as before (P3-semantics guard):
    the ledger counts runtime discards, the rewrite drops exactly those records,
    survivors keep their original offsets and restore byte-exact."""
    import freetoken.scheduler.session_tier as st_mod

    d, _ = _boot_with_watermark_holes(tmp_path, subdir="p8b", garbage=8192)
    store = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=d))
    assert store.replay_journal() == 3
    assert store.evict_store(2) == 2      # default floor (256 MiB) blocks the runtime compact
    assert store._dead_records == 2 and store._dead_record_bytes == 2 * 4096
    monkeypatch.setattr(st_mod, "_COMPACT_FLOOR_BYTES", 4096)
    assert store.maybe_compact() == 2     # fires on the record ledger, as before
    assert store._dead_records == 0 and store._dead_record_bytes == 0
    assert store._dead_bytes == 0 and store._ssd_used == 4096
    assert store.compact() == 0           # idempotent no-op, as before
    k = chain_page_key(None, (3,))
    hit = store.probe([k])
    assert hit is not None
    got, _ = store.restore(hit[1])
    assert got[0] == bytes([3]) * PAGE    # the survivor is byte-exact


def test_stats_line_dead_is_record_honest_and_holes_separate(tmp_path):
    """P8 (c): stats_line must not show watermark holes as dead: dead= counts the
    records a compact would drop (+ their bytes), the unreclaimable holes get their
    own gauge; the 'session tier final:' stop line prints exactly this line."""
    d, garbage = _boot_with_watermark_holes(tmp_path, subdir="p8c")
    store = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=d))
    store.replay_journal()                # fast path, zero dead records
    line = store.stats_line()
    assert "dead=0rec/0.0MiB" in line
    assert f"holes={garbage / 2**20:.1f}MiB" in line
    assert f"dead={garbage / 2**20:.1f}MiB" not in line   # the misleading figure is gone
    assert store.evict_store(1) == 1
    line = store.stats_line()
    assert "dead=1rec/" in line           # the discarded record is the real dead
    assert f"holes={garbage / 2**20:.1f}MiB" in line      # holes unchanged by the discard


def test_full_path_boot_counts_payload_dropped_records(tmp_path, monkeypatch):
    """P8 (d, boot side): a payload-crc-dead record stays in the journal file after a
    crash boot (replay never rewrites it) and a later compact parser still sees it -
    the ledger must count it at boot, or the compact gate under-reports."""
    import freetoken.scheduler.session_tier as st_mod

    d, _ = _boot_with_watermark_holes(tmp_path, subdir="p8d", garbage=8192)
    with open(os.path.join(d, "blob.bin"), "r+b") as f:   # corrupt segs 1+2 payloads
        f.write(b"\xde" * 256)
        f.seek(4096)
        f.write(b"\xde" * 256)
    os.unlink(_marker_path(d))            # force the full verification path
    boot = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=d))
    assert boot.replay_journal() == 1     # the dead records are dropped, not resurrected
    assert boot._dead_records == 2 and boot._dead_record_bytes == 2 * 4096
    monkeypatch.setattr(st_mod, "_COMPACT_FLOOR_BYTES", 4096)
    assert boot.maybe_compact() == 2      # the ledger gates exactly what compact drops
    assert boot._dead_records == 0
    reborn = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=d))
    assert reborn.replay_journal() == 1   # the journal no longer lists the dead records


def test_stats_line_holes_never_negative(tmp_path):
    """P8 review: in the crashed-compact window (SIGKILL after the blob rename, before
    the journal rewrite) the old journal carries dead records whose spans lie ABOVE
    the truncated blob_eof; replay drops them (payload crc reads past eof), so
    _dead_record_bytes can exceed the watermark-seeded _dead_bytes - holes must clamp
    to 0 for display, never render negative."""
    d = tmp_path / "holes"
    store = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d)))
    keys = []
    for i in (1, 2, 3, 4):
        k = chain_page_key(None, (i,))
        keys.append(k)
        assert store.offer(bytes([i]) * 16, 1, [(k, bytes([i]) * PAGE)])
    for k in keys[:2]:                    # validate the LOWER segs: evictions hit the TOP
        hit = store.probe([k])
        store.restore(hit[1])
    assert store.evict_store(2) == 2      # dead records sit ABOVE the survivors
    stale_journal = (d / "journal.log").read_bytes()
    assert store.compact() == 2           # blob truncated: eof == live == 2 * 4096
    (d / "journal.log").write_bytes(stale_journal)   # SIGKILL before the journal rewrite
    boot = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d)))
    assert boot.replay_journal() == 2     # dead records above eof: payload-crc drops
    assert boot._dead_bytes == 0 and boot._dead_record_bytes == 2 * 4096
    line = boot.stats_line()
    assert "holes=0.0MiB" in line
    assert "=-" not in line               # no negative gauge anywhere in the line


# ------------------------------------------------- P9: shrinking compact swap


class _Crash(Exception):
    """Simulated SIGKILL inside the swap: compact's handled-abort path catches OSError,
    so the crash must be a different kind to leave the half-done on-disk state behind."""


def _crash_on_fs(monkeypatch, predicate):
    """Wrap the module-visible os link/rename/unlink/pwrite so any call whose args
    satisfy predicate(name, args) raises _Crash (a simulated SIGKILL mid-swap)."""
    import freetoken.scheduler.session_tier as st
    real = {n: getattr(os, n) for n in ("link", "rename", "unlink", "pwrite")}

    def make(n, orig):
        def hooked(*a, **k):
            if predicate(n, a):
                raise _Crash(f"simulated SIGKILL in {n}")
            return orig(*a, **k)
        return hooked

    for n, orig in real.items():
        monkeypatch.setattr(st.os, n, make(n, orig))


def _tier_files(d):
    return sorted(p.name for p in os.scandir(str(d)))


def _shrink_fixture(root, n=3, dead=(1,), blob_above_floor=False):
    """n offers in L2-only mode; the `dead` indexes evicted (true discards -> dead
    records). Survivors are validated first so LRU evicts exactly the dead ones.
    blob_above_floor extends blob.bin with a sparse tail so the watermark (_blob_eof)
    sits just above the P8 compact floor - the shape of a tier that once grew past
    the floor; the P13 shutdown-skip gate reads that watermark, not the live set."""
    d = root / "tier"
    store = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d)))
    data = {}
    for i in range(1, n + 1):
        k = chain_page_key(None, (i,))
        data[k] = bytes([i]) * PAGE
        assert store.offer(bytes([i]) * 16, 1, [(k, data[k])])
    if dead:
        for i in range(1, n + 1):
            if i in dead:
                continue
            hit = store.probe([chain_page_key(None, (i,))])
            store.restore(hit[1])
        assert store.evict_store(len(dead)) == len(dead)
    if blob_above_floor:
        import freetoken.scheduler.session_tier as st_mod
        target = st_mod._COMPACT_FLOOR_BYTES + 4096
        with open(d / "blob.bin", "r+b") as f:
            f.seek(target - 1)
            f.write(b"\0")                  # sparse: prior generations, no live bytes
        store._blob_eof = os.fstat(store._blob_fd).st_size
    return d, store, data


def _restore_all(store, data, live):
    for i in live:
        hit = store.probe([chain_page_key(None, (i,))])
        assert hit is not None, f"record {i} missing"
        assert store.restore(hit[1])[0] == [data[chain_page_key(None, (i,))]]


def test_compact_shrinks_file_to_live_volume(tmp_path):
    """P9 (a): after compact the blob file is EXACTLY the live volume (dense repack),
    not the old watermark; the offsets moved and everything still restores."""
    d, store, data = _shrink_fixture(tmp_path, n=4, dead=(1, 4))
    assert (d / "blob.bin").stat().st_size == 4 * 4096
    assert store.compact() == 2
    assert (d / "blob.bin").stat().st_size == 2 * 4096      # file == live volume
    assert store._blob_eof == 2 * 4096 and store._ssd_used == 2 * 4096
    assert _tier_files(d) == ["blob.bin", "journal.log"]    # no swap debris
    boot = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d)))
    assert boot.replay_journal() == 2
    _restore_all(boot, data, [2, 3])


def test_compact_copy_is_byte_exact_vs_serial_reference(tmp_path):
    """P9 (c): the parallel copy lands the same bytes a serial reference copy would:
    each live span is byte-identical at its NEW dense offset, crc fields unchanged."""
    d, store, data = _shrink_fixture(tmp_path, n=3, dead=(2,))
    old_blob = (d / "blob.bin").read_bytes()
    before = {rec["path_key"]: rec for rec in store._parse_journal()}
    assert store.compact() == 1
    new_blob = (d / "blob.bin").read_bytes()
    after = store._parse_journal()
    assert len(after) == 2
    dense = 0
    for rec in after:
        old_rec = before[rec["path_key"]]
        span = (rec["n"] + 4095) // 4096 * 4096
        assert rec["off"] == dense                          # dense packing, no gaps
        assert old_rec["crc"] == rec["crc"]                 # content crc unchanged
        assert (new_blob[rec["off"]:rec["off"] + span]
                == old_blob[old_rec["off"]:old_rec["off"] + span])
        dense += span
    assert len(new_blob) == dense


_CRASH_WINDOWS = {
    # commit not armed: boot cleanup -> old state (the dead record still replays)
    "copy": lambda n, a: n == "pwrite",
    "jnew": None,                       # store-method patch: after journal.new durable
    "token": lambda n, a: n == "rename" and str(a[0]).endswith("swap.token.new"),
    # token armed: boot recovery completes the swap
    "link-blob": lambda n, a: n == "link" and str(a[0]).endswith("blob.bin"),
    "link-journal": lambda n, a: n == "link" and str(a[0]).endswith("journal.log"),
    "rename-blob": lambda n, a: n == "rename" and str(a[0]).endswith("blob.bin.new"),
    "rename-journal": lambda n, a: n == "rename" and str(a[0]).endswith("journal.log.new"),
    "unlink-old": lambda n, a: n == "unlink" and str(a[0]).endswith(".old"),
    "unlink-token": lambda n, a: n == "unlink" and str(a[0]).endswith("swap.token"),
}


@pytest.mark.parametrize("window", sorted(_CRASH_WINDOWS))
def test_compact_sigkill_matrix_converges(tmp_path, monkeypatch, window):
    """P9 (b): a crash at EVERY step of the swap converges at boot to either the old or
    the shrunk state - live records byte-exact, journal consistent, no swap debris.
    Windows before the token arms leave the old state; after, recovery completes."""
    d, store, data = _shrink_fixture(tmp_path, n=4, dead=(1,))
    if window == "copy":
        calls = {"n": 0}
        real_pwrite = os.pwrite

        def flaky(fd, buf, off):
            calls["n"] += 1
            if calls["n"] >= 2:
                raise _Crash("mid-copy")
            return real_pwrite(fd, buf, off)

        monkeypatch.setattr("freetoken.scheduler.session_tier.os.pwrite", flaky)
    elif window == "jnew":
        real_jnew = store._write_journal_new

        def boom(keep, layout, new_extents):
            real_jnew(keep, layout, new_extents)
            raise _Crash("after journal.new")

        monkeypatch.setattr(store, "_write_journal_new", boom)
    else:
        _crash_on_fs(monkeypatch, _CRASH_WINDOWS[window])
    with pytest.raises(_Crash):
        store.compact()
    monkeypatch.undo()
    boot = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d)))
    n = boot.replay_journal()
    shrunk = (d / "blob.bin").stat().st_size == 3 * 4096
    if window in ("copy", "jnew", "token"):
        assert not shrunk and n == 3        # old state: the P13 tombstone keeps the
                                            # discarded record dead (pre-P13: replayed)
    else:
        assert shrunk and n == 3            # recovery completed the swap
    assert _tier_files(d) == ["blob.bin", "journal.log"]
    _restore_all(boot, data, [2, 3, 4])


def test_boot_recovery_completes_armed_swap(tmp_path, monkeypatch, tier_log_capture):
    """P9 (g1): crash after the token armed, staged pair intact -> boot recovery
    completes the swap and logs it; the result is the clean shrunk state."""
    d, store, data = _shrink_fixture(tmp_path, n=3, dead=(1,))
    _crash_on_fs(monkeypatch, lambda n, a: n == "link")
    with pytest.raises(_Crash):
        store.compact()
    monkeypatch.undo()
    assert os.path.exists(str(d / "swap.token"))
    assert os.path.exists(str(d / "blob.bin.new"))
    boot = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d)))
    with tier_log_capture() as cap:
        assert boot.replay_journal() == 2
    assert any("completed interrupted compact swap" in m for m in cap.messages)
    assert (d / "blob.bin").stat().st_size == 2 * 4096
    assert _tier_files(d) == ["blob.bin", "journal.log"]
    _restore_all(boot, data, [2, 3])


def test_boot_recovery_rolls_back_broken_staged_pair(tmp_path, monkeypatch, tier_log_capture):
    """P9 (g2): armed token + corrupted staged blob (truncated) -> boot rolls back to
    the .old inodes; the OLD layout is intact bit-for-bit (and the discarded record
    stays dead - its tombstone is in the rolled-back journal), swap debris is gone."""
    d, store, data = _shrink_fixture(tmp_path, n=3, dead=(1,))
    _crash_on_fs(monkeypatch, lambda n, a: n == "link")
    with pytest.raises(_Crash):
        store.compact()
    monkeypatch.undo()
    with open(str(d / "blob.bin.new"), "r+b") as f:
        f.truncate(4096)
    boot = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d)))
    with tier_log_capture() as cap:
        assert boot.replay_journal() == 2
    assert any("rolled back" in m for m in cap.messages)
    assert (d / "blob.bin").stat().st_size == 3 * 4096
    assert _tier_files(d) == ["blob.bin", "journal.log"]
    _restore_all(boot, data, [2, 3])


def test_boot_recovery_rolls_back_stale_staged_journal(tmp_path, monkeypatch, tier_log_capture):
    """P9 (g3): armed token + corrupted staged JOURNAL (sha mismatch) -> same rollback."""
    d, store, data = _shrink_fixture(tmp_path, n=3, dead=(1,))
    _crash_on_fs(monkeypatch, lambda n, a: n == "link")
    with pytest.raises(_Crash):
        store.compact()
    monkeypatch.undo()
    with open(str(d / "journal.log.new"), "r+b") as f:
        f.seek(20)
        b = f.read(1)
        f.seek(20)
        f.write(bytes([b[0] ^ 0xFF]))
    boot = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d)))
    with tier_log_capture() as cap:
        assert boot.replay_journal() == 2   # old journal intact; the tombstone holds
    assert any("rolled back" in m for m in cap.messages)
    assert (d / "blob.bin").stat().st_size == 3 * 4096
    _restore_all(boot, data, [2, 3])


def test_boot_cleans_legacy_and_swap_temps(tmp_path):
    """Boot with no token removes pre-P9 compact debris and interrupted P9 temps."""
    d = tmp_path / "tier"
    d.mkdir()
    (d / "blob.bin").write_bytes(b"")
    (d / "journal.log").write_bytes(b"")
    (d / "blob.bin.compact").write_bytes(b"junk")   # pre-P9 crash debris
    (d / "blob.bin.new").write_bytes(b"junk")       # interrupted P9 copy
    store = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d)))
    assert store.replay_journal() == 0
    assert _tier_files(d) == ["blob.bin", "journal.log"]


def test_compact_resets_p8_ledger_and_stats(tmp_path):
    """P9 (d): after the shrink the P8 record ledger and the watermark holes are all
    zero and the stats line reports the honest zero state."""
    d, store, data = _shrink_fixture(tmp_path, n=4, dead=(1, 2))
    assert store._dead_records == 2 and store._dead_record_bytes == 2 * 4096
    assert store._dead_bytes == 2 * 4096
    assert store.compact() == 2
    snap = store.snapshot()
    assert snap["dead_records"] == 0 and snap["dead_record_bytes"] == 0
    assert snap["dead_bytes"] == 0 and snap["blob_eof"] == 2 * 4096
    line = store.stats_line()
    assert "dead=0rec/0.0MiB" in line and "holes=0.0MiB" in line


def test_graceful_stop_after_shrink_marker_fast_path(tmp_path, monkeypatch, tier_log_capture):
    """P9 (e): the shutdown discipline (invalidate -> flush -> compact -> marker) leaves
    a VALID marker with the SHRUNK getsize; the next boot fast-paths with zero preads."""
    d, store, data = _shrink_fixture(tmp_path, n=3, dead=(1,))
    _graceful_stop(store)
    assert os.path.exists(_marker_path(d))
    assert (d / "blob.bin").stat().st_size == 2 * 4096

    def boom(*args, **kwargs):
        raise AssertionError("payload pread must not happen on the fast path")

    monkeypatch.setattr("freetoken.scheduler.session_tier.os.pread", boom)
    with tier_log_capture() as cap:
        boot, n = _boot(str(d))
    assert n == 2
    assert any("replay fast-path (clean shutdown marker)" in m for m in cap.messages)
    _restore_all(boot, data, [2, 3])


def test_append_after_shrink_survives_boot_double_reopen(tmp_path):
    """P9 (f): after BOTH files were renamed, the store's blob and journal fds must be
    re-opened on the new inodes: a post-compact append (new blob span + journal record)
    must survive a crash boot byte-exactly (stale-fd regression, P3-era discipline)."""
    d, store, data = _shrink_fixture(tmp_path, n=3, dead=(1,))
    assert store.compact() == 1
    k4 = chain_page_key(None, (4,))
    data4 = bytes([4]) * PAGE
    assert store.offer(bytes([4]) * 16, 1, [(k4, data4)])   # L2-only: direct append
    recs, whole = _journal_records_raw(str(d / "journal.log"))
    assert whole and len(recs) == 3
    assert recs[-1]["off"] == 2 * 4096                      # appended at the dense EOF
    boot = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d)))
    assert boot.replay_journal() == 3
    hit = boot.probe([k4])
    assert boot.restore(hit[1])[0] == [data4]
    assert (d / "blob.bin").stat().st_size == 3 * 4096


def test_compact_deferred_while_blob_read_in_flight(tmp_path, tier_log_capture):
    """P9 (h): a shrinking swap must not move offsets under a by-path reader (restore /
    prefetch staging read outside the store lock): with a refs pin held, compact is a
    deferred no-op; released, it runs."""
    d, store, data = _shrink_fixture(tmp_path, n=3, dead=(1,))
    hit = store.probe([chain_page_key(None, (2,))])
    seg = store._segments[hit[1]._seg_id]
    seg.refs += 1
    with tier_log_capture() as cap:
        assert store.compact() == 0
    assert any("compaction deferred" in m for m in cap.messages)
    assert (d / "blob.bin").stat().st_size == 3 * 4096      # unchanged
    seg.refs -= 1
    assert store.compact() == 1
    assert (d / "blob.bin").stat().st_size == 2 * 4096


def test_compact_oserror_mid_commit_converges_to_old_pair(tmp_path, monkeypatch, tier_log_capture):
    """Fix W10: a runtime OSError INSIDE _commit_swap after the arm (rename#2 here)
    must roll back synchronously through the anchors - the pre-fix cleanup erased the
    anchors and the token and left the NEW blob against the OLD journal: the next boot
    dropped every live record."""
    import freetoken.scheduler.session_tier as st

    d, store, data = _shrink_fixture(tmp_path, n=3, dead=(1,))
    real_rename = os.rename

    def flaky(src, dst):
        if str(src).endswith("journal.log.new"):
            raise OSError(5, "simulated EIO on the journal rename")
        return real_rename(src, dst)

    monkeypatch.setattr(st.os, "rename", flaky)
    with tier_log_capture() as cap:
        assert store.compact() == 0
    monkeypatch.undo()
    assert any("rolled back" in m for m in cap.messages)
    assert not any("left for boot recovery" in m for m in cap.messages)
    assert _tier_files(d) == ["blob.bin", "journal.log"]   # no anchors/token debris
    boot = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d)))
    assert boot.replay_journal() == 2                       # the OLD pair is intact;
                                                            # the tombstone holds
    _restore_all(boot, data, [2, 3])


def test_compact_oserror_before_arm_aborts_clean(tmp_path, monkeypatch, tier_log_capture):
    """Fix (pre-arm path): an OSError while the commit has not started (the copy here)
    aborts cleanly - the old pair is untouched, the staged garbage is cleaned up, and
    the deferred dead records simply remain for the next compact."""
    import freetoken.scheduler.session_tier as st

    d, store, data = _shrink_fixture(tmp_path, n=3, dead=(1,))

    def flaky(fd, buf, off):
        raise OSError(5, "simulated EIO in the copy")

    monkeypatch.setattr(st.os, "pwrite", flaky)
    with tier_log_capture() as cap:
        assert store.compact() == 0
    monkeypatch.undo()
    assert any("aborted before arm" in m for m in cap.messages)
    assert _tier_files(d) == ["blob.bin", "journal.log"]
    boot = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d)))
    assert boot.replay_journal() == 2                       # old pair intact, tombstone holds
    _restore_all(boot, data, [2, 3])


def test_boot_recovery_rollback_with_anchors(tmp_path, monkeypatch, tier_log_capture):
    """Fix (recovery classification): crash between the renames (anchors up) + a
    corrupted staged journal (sha mismatch) -> the boot rolls back to the OLD pair
    byte-exactly instead of renaming the corrupted journal into place."""
    d, store, data = _shrink_fixture(tmp_path, n=3, dead=(1,))
    _crash_on_fs(monkeypatch,
                 lambda n, a: n == "rename" and str(a[0]).endswith("journal.log.new"))
    with pytest.raises(_Crash):
        store.compact()
    monkeypatch.undo()
    assert os.path.exists(str(d / "blob.bin.old"))          # anchors are up
    with open(str(d / "journal.log.new"), "r+b") as f:
        f.seek(20)
        b = f.read(1)
        f.seek(20)
        f.write(bytes([b[0] ^ 0xFF]))                        # corrupt the staged journal
    boot = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d)))
    with tier_log_capture() as cap:
        assert boot.replay_journal() == 2   # rolled back to the old pair; tombstone holds
    assert any("rolled back" in m for m in cap.messages)
    assert _tier_files(d) == ["blob.bin", "journal.log"]
    _restore_all(boot, data, [2, 3])


def test_marker_written_when_deferred_dead_all_tombstoned(tmp_path, monkeypatch, tier_log_capture):
    """P13 refines B3: a deferred compact leaves dead records journaled, but with a
    durable tombstone each they cannot fast-path-resurrect (replay filters tombstones
    before the fast-path ledger seed), so the marker stays honest and WRITES. The
    pre-P13 skip remains for un-tombstoned dead - covered by the blocker test below."""
    d, store, data = _shrink_fixture(tmp_path, n=3, dead=(1,), blob_above_floor=True)
    hit = store.probe([chain_page_key(None, (2,))])
    seg = store._segments[hit[1]._seg_id]
    seg.refs += 1                       # a blob read in flight: compact WOULD defer
    with tier_log_capture() as cap:
        _graceful_stop_p13(store)
    seg.refs -= 1
    assert any("shutdown compact skipped" in m for m in cap.messages)
    assert os.path.exists(_marker_path(d))       # tombstoned dead no longer block B3
    calls = _counting_pread(monkeypatch)         # fast path: zero payload re-reads
    boot = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d)))
    assert boot.replay_journal() == 2
    assert calls["n"] == 0
    assert boot.probe([chain_page_key(None, (1,))]) is None   # the tombstone holds
    _restore_all(boot, data, [2, 3])


def test_compact_writers_env_clamped(monkeypatch, tier_log_capture):
    """Fix B5 (P1 precedent): a nonsense override must not fork a thread army - clamped
    to 64 with a warning; in-range values pass through."""
    import freetoken.scheduler.session_tier as st

    monkeypatch.setenv("FREETOKEN_COMPACT_WRITERS", "1000")
    with tier_log_capture() as cap:
        assert st._compact_writers(8) == 64
    assert any("clamping" in m for m in cap.messages)
    monkeypatch.setenv("FREETOKEN_COMPACT_WRITERS", "63")
    assert st._compact_writers(8) == 63
    monkeypatch.delenv("FREETOKEN_COMPACT_WRITERS")
    assert st._compact_writers(8) == 8


def test_second_compact_skipped_while_token_pending(tmp_path, monkeypatch, tier_log_capture):
    """Fix: an armed failure that cannot roll back ('left for boot recovery') leaves the
    token + anchors + staged pair up; a SECOND compact must NOT start on top of the
    diverged disk - warning + rc=0, the recovery evidence survives, and the boot
    converges through the token byte-exactly."""
    import freetoken.scheduler.session_tier as st

    d, store, data = _shrink_fixture(tmp_path, n=3, dead=(1,))
    real_rename = os.rename

    def flaky(src, dst):        # everything fails EXCEPT the token's own rename: the
        if str(src).endswith("swap.token.new"):   # arm lands, the commit cannot, and
            return real_rename(src, dst)           # the rollback cannot either
        raise OSError(5, "simulated EIO on every other rename")

    monkeypatch.setattr(st.os, "rename", flaky)
    with tier_log_capture() as cap:
        assert store.compact() == 0
    monkeypatch.undo()
    assert any("left for boot recovery" in m for m in cap.messages)
    for name in ("swap.token", "blob.bin.old", "blob.bin.new", "journal.log.new"):
        assert os.path.exists(str(d / name)), name

    # the second compact on the SAME store: refused, evidence untouched
    with tier_log_capture() as cap2:
        assert store.compact() == 0
    assert any("compaction skipped, an armed swap token is still pending" in m
               for m in cap2.messages)
    assert any("generation=" in m for m in cap2.messages)
    for name in ("swap.token", "blob.bin.old", "blob.bin.new", "journal.log.new"):
        assert os.path.exists(str(d / name)), name

    # boot: recovery completes the armed swap; live records byte-exact
    boot = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d)))
    assert boot.replay_journal() == 2
    assert _tier_files(d) == ["blob.bin", "journal.log"]
    assert (d / "blob.bin").stat().st_size == 2 * 4096
    _restore_all(boot, data, [2, 3])


def test_recovery_reopen_failure_keeps_old_fds(tmp_path, monkeypatch, tier_log_capture):
    """Atomic tail reopen in recovery: if the second open fails, the old fds stay open
    and in charge (warned) - no fd is left closed or stale."""
    import freetoken.scheduler.session_tier as st

    d, store, data = _shrink_fixture(tmp_path, n=3, dead=(1,))
    _crash_on_fs(monkeypatch,
                 lambda n, a: n == "rename" and str(a[0]).endswith("blob.bin.new"))
    with pytest.raises(_Crash):
        store.compact()             # crash between the renames: boot completes the swap
    monkeypatch.undo()

    boot = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d)))
    real_open = os.open

    def flaky(path, flags, *args):
        if str(path).endswith("journal.log") and flags & os.O_WRONLY:
            raise OSError(5, "simulated EIO on the journal reopen")
        return real_open(path, flags, *args)

    monkeypatch.setattr(st.os, "open", flaky)
    with tier_log_capture() as cap:
        assert boot.replay_journal() == 2
    monkeypatch.undo()
    assert any("post-recovery fd reopen failed" in m for m in cap.messages)
    assert boot._blob_fd >= 0 and boot._journal_fd >= 0
    os.close(boot._blob_fd)                     # both fds valid: no double-close damage
    os.close(boot._journal_fd)


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
    """F1 (P13-updated): survival keyed by EXACT identity (path, offset, length) stays
    the compact rule, but the crashed-discard double-live-segment scenario cannot arise
    anymore: the discard appends a durable tombstone, so the reboot replays exactly ONE
    live segment per path and the discarded one stays dead (pre-P13 both resurrected
    and last-wins survival would hole the older live segment's blob region)."""
    d = tmp_path / "dup"
    store = SessionTierStore(_cfg(d=str(d)))
    keys1, pages1 = _chain([(1, 0), (1, 1), (1, 2)])
    assert store.offer(b"path", 3, pages1, _snap("a"))
    assert store.flush_live() == 1          # seg1 -> L2 (journal record R1)
    assert store.evict_store(1) == 1        # seg1 discarded; the tombstone lands too
    keys2, pages2 = _chain([(5, 0), (5, 1), (5, 2)])
    assert store.offer(b"path", 3, pages2, _snap("b"))   # re-offer same path: new seg2
    assert store.flush_live() == 1          # seg2 -> L2 (R2); journal now has R1 + R2 + T1

    # SIGKILL-style reboot: the tombstone keeps R1 dead, exactly one segment is live
    reborn = SessionTierStore(_cfg(d=str(d)))
    assert reborn.replay_journal() == 1
    assert [s.path_key for s in reborn._segments.values()].count(b"path") == 1
    assert reborn.compact() == 1            # R1 + T1 reclaimed, R2 kept

    # the surviving segment probes AND restores byte-exact; the dead one never returns
    h2 = reborn.probe(keys2)[1]             # seg2's page keys only match seg2
    assert reborn.restore(h2) == ([data for _, data in pages2], [_snap("b")])
    assert reborn.probe(keys1) is None      # seg1 stays discarded across boots


# --------------------------------------------------- P2: clean-shutdown marker


def _marker_path(d):
    return os.path.join(str(d), "shutdown.marker")


def test_shutdown_marker_fast_path_replay(tmp_path, monkeypatch, tier_log_capture):
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
    with tier_log_capture() as cap:
        boot, n = _boot(str(d))
    assert n == 1
    assert "session tier: replay fast-path (clean shutdown marker), 1 records" in cap.messages
    hit = boot.probe(keys)
    assert hit is not None and hit[0] == 3
    out, snaps = boot.restore(hit[1])
    assert out == [x for _, x in pages] and snaps == [_snap("s")]


def test_shutdown_marker_missing_or_stale_takes_full_path(tmp_path, monkeypatch, tier_log_capture):
    """P2 (b): no marker, or a marker whose sizes no longer match (a runtime append after
    the previous stop) -> full per-record crc verification, records intact."""
    chains = [_chain([(i, 0), (i, 1), (i, 2)]) for i in (1, 2)]
    # (i) no marker: old directories without one keep the full path
    d1 = tmp_path / "no-marker"
    store = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d1)))
    for i, (keys, pages) in enumerate(chains, start=1):
        assert store.offer(f"path-{i}".encode(), 3, pages, _snap(f"{i}"))
    calls = _counting_pread(monkeypatch)
    with tier_log_capture() as cap:
        boot, n = _boot(str(d1))
    assert n == 2 and calls["n"] == 4   # 2 records x 2 extents (per-extent verify)
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
    assert n == 3 and calls["n"] == 5     # 2 snapped records x 2 + 1 bare x 1


def test_shutdown_marker_full_path_drops_dead_keeps_live(tmp_path, tier_log_capture):
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
    with tier_log_capture() as cap:
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


def test_shutdown_marker_invalidated_before_shutdown_crash(tmp_path, monkeypatch, tier_log_capture):
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
    with tier_log_capture() as cap:
        boot3, n = _boot(str(d))
    assert n == 3                                                # live set intact
    assert calls["n"] == 5          # 2 snapped records x 2 + 1 bare x 1 extents
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


def test_fast_path_boot_seeds_ssd_used_from_live_records(tmp_path, tier_log_capture):
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

    with tier_log_capture() as cap:
        boot = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=d))
        assert boot.replay_journal() == 2     # fast path (marker sizes match)
    # fails-before: the watermark (blob size) seeded the cap accounting
    assert boot._ssd_used == live
    assert boot._blob_eof == blob_size        # P3 invariant: watermark untouched
    assert boot._dead_bytes == garbage        # holes stay real for reservations/cap accounting
    on = [m for m in cap.messages if "session tier on" in m]
    assert len(on) == 1 and "used=0.00 GiB" in on[0]

    # Full path on the same on-disk state: identical accounting and log figure.
    os.unlink(_marker_path(d))
    with tier_log_capture() as cap:
        boot_full = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=d))
        assert boot_full.replay_journal() == 2
    assert boot_full._ssd_used == live
    assert boot_full._blob_eof == blob_size
    assert boot_full._dead_bytes == garbage
    on = [m for m in cap.messages if "session tier on" in m]
    assert len(on) == 1 and "used=0.00 GiB" in on[0]


def test_shutdown_marker_discard_invalidates(tmp_path, monkeypatch, tier_log_capture):
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
    with tier_log_capture() as cap:
        boot, n = _boot(str(d))
    assert n == 0                                # P13: the tombstone keeps it dead
    assert calls["n"] == 0                       # no live records left to verify
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
    assert n == 1 and calls["n"] == 2   # stale sizes -> full per-extent verification
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
        # the record's physical bytes = its ordered extents (shared spans included)
        # concatenated; unshared records are one contiguous span - byte-identical to
        # the sequential path's [payload][zero padding] layout
        got = b""
        end = 0
        for ext in seg.extents:
            got += blob[ext[0]:ext[0] + ext[1]]
            end = max(end, ext[0] + ext[1])
        assert got == want[:len(got)]
        last = seg.extents[-1]
        pad = blob[last[0] + last[1]:last[0] + last[1] + last[4]]
        assert pad == b"\0" * len(pad)          # padding zeros, byte-identical
        hit = boot.probe(keys)
        assert hit is not None and hit[0] == 3
        got_p, got_s = boot.restore(hit[1])
        assert got_p == [data for _, data in pages] and got_s == [_snap(tag)]


def test_flush_live_writes_segments_in_parallel(tmp_path, monkeypatch):
    """P3 fails-before: the flush path must overlap segment writes - concurrent os.pwritev
    entries prove >1 writer; the old sequential os.write append never overlaps (and never
    calls pwritev at all)."""
    import freetoken.scheduler.session_tier as st_mod

    d = str(tmp_path / "tier")
    store = SessionTierStore(_cfg(ram=16384, ssd=64 * 4096, d=d))
    for i in range(1, 5):
        keys, pages = _chain([(i, 0), (i, 1), (i, 2)])
        assert store.offer(f"p{i}".encode(), 3, pages, _snap(f"{i}"))
    assert len([s for s in store._segments.values() if s.in_l1]) == 4

    real_pwritev = os.pwritev
    state = {"entered": 0, "max": 0}
    lock = threading.Lock()

    def overlap_probe(fd, buffers, offset):
        with lock:
            state["entered"] += 1
            state["max"] = max(state["max"], state["entered"])
        deadline = time.time() + 5.0
        while state["entered"] < 2 and time.time() < deadline:
            time.sleep(0.001)                   # bounded wait: never hangs the test
        try:
            return real_pwritev(fd, buffers, offset)
        finally:
            with lock:
                state["entered"] -= 1

    monkeypatch.setattr(st_mod.os, "pwritev", overlap_probe)
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
    """The record's LOGICAL payload bytes: extents records concatenate their ordered
    extent regions, legacy records read their single private span."""
    with open(os.path.join(d, "blob.bin"), "rb") as f:
        exts = rec.get("extents")
        if exts is not None:
            out = b""
            for e in exts:
                f.seek(e[0])
                out += f.read(e[1])
            return out
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
    assert calls["n"] == 6                     # every payload re-read, per extent
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


def test_corrupted_crc32c_record_dropped(tmp_path, tier_log_capture):
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
    with tier_log_capture() as cap:
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
    mid-loop cap eviction (_flush_groups -> _reserve_blob_span -> _evict_oldest_l2 ->
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
    # P13: s1/s2 (runtime demotes) and s3 (sacrificed mid-flush) all carry durable
    # tombstones - the pre-P13 documented resurrect semantics is gone; only s4 lives.
    assert boot.replay_journal() == 1
    for i in (1, 2, 3):
        keys, pages = _chain([(i, 0), (i, 1), (i, 2)])
        assert boot.probe(keys) is None
    keys, pages = _chain([(4, 0), (4, 1), (4, 2)])
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

    real_pwritev = os.pwritev
    calls = {"n": 0}

    def flaky(fd, buffers, offset):
        calls["n"] += 1
        if calls["n"] == 1:                   # exactly one group segment fails
            raise OSError(5, "simulated pwrite failure")
        return real_pwritev(fd, buffers, offset)

    monkeypatch.setattr(st_mod.os, "pwritev", flaky)
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


def test_failed_write_hole_skipped_by_next_group_no_dropped(tmp_path, monkeypatch, tier_log_capture):
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

    real_pwritev = os.pwritev
    calls = {"n": 0}

    def flaky(fd, buffers, offset):
        calls["n"] += 1
        if calls["n"] == 1:                   # group 1's single segment fails
            raise OSError(5, "simulated pwrite failure")
        return real_pwritev(fd, buffers, offset)

    monkeypatch.setattr(st_mod.os, "pwritev", flaky)
    assert store.flush_live() == 2
    monkeypatch.undo()

    recs, whole = _journal_records_raw(os.path.join(d, "journal.log"))
    assert whole and len(recs) == 2
    assert all(rec["off"] >= 4096 for rec in recs)   # the hole [0, 4096) is unreferenced
    with tier_log_capture() as cap:
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
    real_pwritev = os.pwritev

    def flaky(fd, buffers, offset):
        if bytes(buffers[0][:PAGE]) == victim_page0:
            raise OSError(5, "simulated pwrite failure")
        return real_pwritev(fd, buffers, offset)

    monkeypatch.setattr(st_mod.os, "pwritev", flaky)
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
    assert store.compact() == 1               # blob shrunk, _blob_eof resynced
    eof = store._blob_eof
    assert eof == 2 * 4096                    # P9: the two live records are dense -
    # ... the file IS the live volume, the next append lands at its EOF
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


# ------------------------------------------------------------ flush producer pipeline (P4)


def test_flush_pipeline_builds_next_group_during_previous_write(tmp_path, monkeypatch):
    """P4 fails-before: the flush is a pipeline - the build of group N+1 (span reserves,
    zero-copy iovecs, payload crc off the L1 pool) completes while group N is still
    being written. Group 0's write blocks until group 1's build event fires; a
    sequential flush can never satisfy that wait (the bounded wait raises). The event
    order is deterministic - the timeout is only a deadlock guard, not a timing
    assertion."""
    import freetoken.scheduler.session_tier as st_mod

    monkeypatch.setattr(st_mod, "_FLUSH_GROUP_BYTES", 1)   # every segment = own group
    d = str(tmp_path / "tier")
    store = SessionTierStore(_cfg(ram=16384, ssd=64 * 4096, d=d))
    for i in range(1, 4):
        keys, pages = _chain([(i, 0), (i, 1), (i, 2)])
        assert store.offer(f"p{i}".encode(), 3, pages, _snap(f"{i}"))

    real_build = SessionTierStore._build_group_stage
    real_write = SessionTierStore._write_group
    lock = threading.Lock()
    events: list[tuple[str, int]] = []
    idx = {"build": 0, "write": 0}
    built_next = threading.Event()

    def build(self, group, pending, index=0, plan=None):
        r = real_build(self, group, pending, index, plan)
        with lock:
            i = idx["build"]
            idx["build"] += 1
            events.append(("build", i))
        if i == 1:
            built_next.set()
        return r

    def write(self, stage, pending, writers, failed_sids=()):
        with lock:
            i = idx["write"]
            idx["write"] += 1
            events.append(("write_start", i))
        if i == 0 and not built_next.wait(timeout=10.0):
            raise AssertionError("pipeline: group 1 was not built during group 0's write")
        r = real_write(self, stage, pending, writers, failed_sids)
        with lock:
            events.append(("write_end", i))
        return r

    monkeypatch.setattr(SessionTierStore, "_build_group_stage", build)
    monkeypatch.setattr(SessionTierStore, "_write_group", write)
    assert store.flush_live() == 3
    monkeypatch.undo()
    b1 = events.index(("build", 1))
    assert b1 < events.index(("write_end", 0))   # overlap happened
    assert events.index(("write_start", 1)) > b1  # ...and group 1 writes after its build

    boot = SessionTierStore(_cfg(ram=16384, ssd=64 * 4096, d=d))
    assert boot.replay_journal() == 3


def test_flush_group_build_failure_aborts_group_cleanly(tmp_path, monkeypatch, tier_log_capture):
    """P4: a failure during group BUILD (lifting iovecs / crc off the L1 pool) aborts
    the whole group with no journal side effect: the segments keep their L1 residency
    with clean blob fields, the builder's refs pins are released, the pending volume
    returns to the cap budget, and a retry after the transient failure converges to a
    byte-exact restore."""
    import freetoken.scheduler.session_tier as st_mod

    monkeypatch.setattr(st_mod, "_FLUSH_GROUP_BYTES", 1)   # every segment = own group
    d = str(tmp_path / "tier")
    store = SessionTierStore(_cfg(ram=16384, ssd=64 * 4096, d=d))
    for i in range(1, 4):
        keys, pages = _chain([(i, 0), (i, 1), (i, 2)])
        assert store.offer(f"p{i}".encode(), 3, pages, _snap(f"{i}"))

    real_iovecs = SessionTierStore._seg_iovecs
    victim = store._by_path(b"p1").seg_id

    def flaky(self, seg, padded, skip_pages=0):
        if seg.seg_id == victim:
            raise OSError(5, "simulated pool page read failure")
        return real_iovecs(self, seg, padded, skip_pages)

    monkeypatch.setattr(SessionTierStore, "_seg_iovecs", flaky)
    with tier_log_capture() as cap:
        assert store.flush_live() == 2
    monkeypatch.undo()
    assert any("flush group #0 build failed at seg" in m and "group aborted" in m
               for m in cap.messages)   # B3: operator sees which group/segment failed

    failed = [s for s in store._segments.values() if s.in_l1]
    assert len(failed) == 1 and failed[0].seg_id == victim
    assert failed[0].blob == "" and failed[0].l2_off == -1
    assert failed[0].refs == 0                 # the builder's refs pin was released
    assert store._ssd_used == 2 * 4096         # no accounting for the aborted group
    recs, whole = _journal_records_raw(os.path.join(d, "journal.log"))
    assert whole and len(recs) == 2            # only the built+written groups journaled

    assert store.flush_live() == 1             # retry after the transient failure
    assert store._ssd_used == 3 * 4096
    boot = SessionTierStore(_cfg(ram=16384, ssd=64 * 4096, d=d))
    assert boot.replay_journal() == 3
    for i in range(1, 4):
        keys, pages = _chain([(i, 0), (i, 1), (i, 2)])
        hit = boot.probe(keys)
        assert hit is not None and hit[0] == 3
        assert boot.restore(hit[1])[0] == [data for _, data in pages]


def test_pwritev_short_writes_complete_record(tmp_path, monkeypatch):
    """P4: the pwritev loop must drive short writes to completion - the blob record is
    byte-exact even when every call reports only half the bytes written (the rewritten
    overlap converges to the same bytes)."""
    import freetoken.scheduler.session_tier as st_mod

    d = str(tmp_path / "tier")
    store = SessionTierStore(_cfg(ram=16384, ssd=64 * 4096, d=d))
    keys, pages = _chain([(1, 0), (1, 1), (1, 2)])
    assert store.offer(b"p1", 3, pages, _snap("1"))
    gold = _gold_payload(store, store._by_path(b"p1"))

    real_pwritev = os.pwritev

    def short(fd, buffers, offset):
        n = real_pwritev(fd, buffers, offset)
        return max(1, n // 2) if n > 1 else n

    monkeypatch.setattr(st_mod.os, "pwritev", short)
    assert store.flush_live() == 1
    monkeypatch.undo()

    with open(os.path.join(d, "blob.bin"), "rb") as f:
        blob = f.read()
    boot = SessionTierStore(_cfg(ram=16384, ssd=64 * 4096, d=d))
    assert boot.replay_journal() == 1
    seg = boot._by_path(b"p1")
    assert blob[seg.l2_off:seg.l2_off + len(gold)] == gold
    hit = boot.probe(keys)
    assert hit is not None and hit[0] == 3
    got_p, got_s = boot.restore(hit[1])
    assert got_p == [data for _, data in pages] and got_s == [_snap("1")]


# ----------------------------------------------------- P4 TP fixes: pipeline hardening


def test_flush_drain_retry_admits_next_group_after_previous_settles(tmp_path, monkeypatch):
    """P4 drain-retry: a cap refusal caused SOLELY by the previous group's still-
    unjournaled pending volume is retried after the drain, reproducing the P3 sacrifice
    semantics (a later group sacrifices an earlier journaled segment). Deterministic:
    group 0's write is blocked until group 1's first reserve attempt is observed
    refusing on pending>0, so the refusal cannot race away; a sequential/no-drain flush
    leaves group 1 refused -> count 1 -> fail."""
    import freetoken.scheduler.session_tier as st_mod

    monkeypatch.setattr(st_mod, "_FLUSH_GROUP_BYTES", 1)   # every segment = own group
    d = str(tmp_path / "tier")
    store = SessionTierStore(_cfg(ram=2048, ssd=64 * 4096, d=d))
    for i in range(1, 5):
        keys, pages = _chain([(i, 0), (i, 1), (i, 2)])
        assert store.offer(f"p{i}".encode(), 3, pages, _snap(f"{i}"))
    assert len([s for s in store._segments.values() if s.in_l2]) == 2   # s1, s2 runtime
    store.ssd_bytes = 4096          # s4's reserve must wait for s3's pending volume

    real_reserve = SessionTierStore._reserve_blob_span
    real_write = SessionTierStore._write_group
    lock = threading.Lock()
    reserve_log: list[tuple[int, int, object]] = []
    refused = threading.Event()

    def reserve(self, need, pending=0):
        r = real_reserve(self, need, pending)
        with lock:
            reserve_log.append((need, pending, r))
        if r is None and pending > 0:
            refused.set()
        return r

    def write(self, stage, pending, writers, failed_sids=()):
        if not refused.wait(timeout=10.0):
            raise AssertionError("group 1's reserve never refused on pending")
        return real_write(self, stage, pending, writers, failed_sids)

    monkeypatch.setattr(SessionTierStore, "_reserve_blob_span", reserve)
    monkeypatch.setattr(SessionTierStore, "_write_group", write)
    assert store.flush_live() == 2
    monkeypatch.undo()

    with lock:
        log = list(reserve_log)
    assert any(need == 896 and pend > 0 and r is None for need, pend, r in log)
    assert any(need == 896 and pend == 0 and r is not None for need, pend, r in log)
    assert not [s for s in store._segments.values() if s.seg_id == 3]  # s3 sacrificed
    assert [s for s in store._segments.values() if s.seg_id == 4][0].in_l2
    boot = SessionTierStore(_cfg(ram=2048, ssd=64 * 4096, d=d))
    assert boot.replay_journal() == 1   # P13 tombstones: s1/s2/s3 stay dead, s4 lives
    keys4, pages4 = _chain([(4, 0), (4, 1), (4, 2)])
    hit = boot.probe(keys4)
    assert hit is not None and hit[0] == 3
    assert boot.restore(hit[1])[0] == [data for _, data in pages4]


def test_flush_drain_timeout_warns_and_retries_with_pending_zero(tmp_path, monkeypatch, tier_log_capture):
    """Timeout fallback honesty: a wait_drained timeout logs a warning (what was waited
    for, how much pending is left) and retries the cap check with pending=0 - which may
    transiently over-commit the ssd cap by up to one group volume (documented in
    _PendingVolume; shutdown-flush only)."""
    import freetoken.scheduler.session_tier as st_mod

    monkeypatch.setattr(st_mod, "_FLUSH_GROUP_BYTES", 1)
    d = str(tmp_path / "tier")
    store = SessionTierStore(_cfg(ram=2048, ssd=64 * 4096, d=d))
    for i in range(1, 5):
        keys, pages = _chain([(i, 0), (i, 1), (i, 2)])
        assert store.offer(f"p{i}".encode(), 3, pages, _snap(f"{i}"))
    store.ssd_bytes = 4096

    real_write = SessionTierStore._write_group
    refused = threading.Event()

    def write(self, stage, pending, writers, failed_sids=()):
        if not refused.wait(timeout=10.0):
            raise AssertionError("group 1's reserve never refused on pending")
        return real_write(self, stage, pending, writers, failed_sids)

    def fake_wait(self, timeout):
        return False                       # simulate the 60 s timeout elapsing

    monkeypatch.setattr(st_mod._PendingVolume, "wait_drained", fake_wait)

    real_reserve = SessionTierStore._reserve_blob_span

    def reserve(self, need, pending=0):
        r = real_reserve(self, need, pending)
        if r is None and pending > 0:
            refused.set()
        return r

    monkeypatch.setattr(SessionTierStore, "_reserve_blob_span", reserve)
    monkeypatch.setattr(SessionTierStore, "_write_group", write)
    with tier_log_capture() as cap:
        assert store.flush_live() == 2
    monkeypatch.undo()
    assert any("waited 60 s" in m and "still unjournaled" in m for m in cap.messages)
    assert store._ssd_used == 2 * 4096    # s3 + s4 both live: the documented over-commit


# ---------------------------------------------------- P5 flush overflow observability


def test_flush_overflow_log_counts_cap_evictions(tmp_path, monkeypatch, tier_log_capture):
    """P5 fails-before: a flush-phase cap eviction was invisible (no counter, no log).
    With the shrunk cap, s3's reserve evicts both runtime L2 victims and s4's
    drain-retry reserve sacrifices s3; the flush must log exactly one overflow line
    whose figures match the actual discards: 3 records / 3 padded 4 KiB spans evicted,
    2 x 896 payload bytes admitted."""
    import freetoken.scheduler.session_tier as st_mod

    monkeypatch.setattr(st_mod, "_FLUSH_GROUP_BYTES", 1)   # every segment = own group
    d = str(tmp_path / "tier")
    store = SessionTierStore(_cfg(ram=2048, ssd=64 * 4096, d=d))
    for i in range(1, 5):
        keys, pages = _chain([(i, 0), (i, 1), (i, 2)])
        assert store.offer(f"p{i}".encode(), 3, pages, _snap(f"{i}"))
    assert len([s for s in store._segments.values() if s.in_l2]) == 2   # runtime demotes
    store.ssd_bytes = 4096                    # every flush reserve must evict

    with tier_log_capture() as cap:
        assert store.flush_live() == 2        # s3 and s4 journaled (s3 sacrificed for s4)
    assert len(store._segments) == 1          # s1, s2, s3 discarded, only s4 survives
    overflow = [m for m in cap.messages if "flush overflow" in m]
    assert overflow == ["session tier: flush overflow: evicted 3 records / 0.0 MiB "
                        "of oldest L2 to admit 1792 bytes"]


def test_flush_without_evictions_logs_no_overflow_line(tmp_path, tier_log_capture):
    """P5: silence when the cap never bites. Runtime L1->L2 demotes (offer pressure,
    _write_blob path) are not flush evictions and must not produce the line; the
    existing flushed-N line is untouched; an idle repeat flush logs nothing at all."""
    d = str(tmp_path / "tier")
    store = SessionTierStore(_cfg(ram=2048, ssd=64 * 4096, d=d))
    for i in range(1, 5):
        keys, pages = _chain([(i, 0), (i, 1), (i, 2)])
        assert store.offer(f"p{i}".encode(), 3, pages, _snap(f"{i}"))

    with tier_log_capture() as cap:
        assert store.flush_live() == 2        # only s3, s4 are still in L1
    assert not [m for m in cap.messages if "flush overflow" in m]
    assert any("flushed 2 live segments" in m for m in cap.messages)
    assert store._ssd_used == 4 * 4096        # s1..s4 journaled, nothing evicted

    with tier_log_capture() as cap:          # repeat flush: nothing live, full silence
        assert store.flush_live() == 0
    assert cap.messages == []


def test_flush_overflow_counts_drain_retry_eviction(tmp_path, monkeypatch, tier_log_capture):
    """P5: the eviction made by the drain-retry reserve (the pending=0 fallback after
    a refusal caused by the previous group's still-unjournaled volume) lands in the
    same overflow tally. Refusal-first ordering is forced deterministically: group 0's
    write blocks until group 1's reserve has refused on pending>0, so the sacrifice of
    s3 can only come from the retry."""
    import freetoken.scheduler.session_tier as st_mod

    monkeypatch.setattr(st_mod, "_FLUSH_GROUP_BYTES", 1)   # every segment = own group
    d = str(tmp_path / "tier")
    store = SessionTierStore(_cfg(ram=2048, ssd=64 * 4096, d=d))
    for i in range(1, 5):
        keys, pages = _chain([(i, 0), (i, 1), (i, 2)])
        assert store.offer(f"p{i}".encode(), 3, pages, _snap(f"{i}"))
    store.ssd_bytes = 4096

    real_reserve = SessionTierStore._reserve_blob_span
    real_write = SessionTierStore._write_group
    refused = threading.Event()

    def reserve(self, need, pending=0):
        r = real_reserve(self, need, pending)
        if r is None and pending > 0:
            refused.set()
        return r

    def write(self, stage, pending, writers, failed_sids=()):
        if not refused.wait(timeout=10.0):
            raise AssertionError("group 1's reserve never refused on pending")
        return real_write(self, stage, pending, writers, failed_sids)

    monkeypatch.setattr(SessionTierStore, "_reserve_blob_span", reserve)
    monkeypatch.setattr(SessionTierStore, "_write_group", write)
    with tier_log_capture() as cap:
        assert store.flush_live() == 2
    monkeypatch.undo()

    assert refused.is_set()                   # the drain-retry path was taken
    assert not [s for s in store._segments.values() if s.seg_id == 3]   # s3 sacrificed
    overflow = [m for m in cap.messages if "flush overflow" in m]
    assert overflow == ["session tier: flush overflow: evicted 3 records / 0.0 MiB "
                        "of oldest L2 to admit 1792 bytes"]


def test_flush_pipeline_baseexception_releases_built_stages(tmp_path, monkeypatch):
    """A SystemExit/KI mid-pipeline must not leak the built-but-unwritten stages: refs
    pins and pending volume are released for the stage the write never ran on AND for
    every built-but-unconsumed future, and the exception propagates unchanged - so a
    retry flush_live loses no segments."""
    import freetoken.scheduler.session_tier as st_mod

    monkeypatch.setattr(st_mod, "_FLUSH_GROUP_BYTES", 1)   # every segment = own group
    d = str(tmp_path / "tier")
    store = SessionTierStore(_cfg(ram=16384, ssd=64 * 4096, d=d))
    for i in range(1, 4):
        keys, pages = _chain([(i, 0), (i, 1), (i, 2)])
        assert store.offer(f"p{i}".encode(), 3, pages, _snap(f"{i}"))

    def boom(self, stage, pending, writers, failed_sids=()):
        raise SystemExit("simulated exit mid-pipeline")

    monkeypatch.setattr(SessionTierStore, "_write_group", boom)
    with pytest.raises(SystemExit):
        store.flush_live()
    monkeypatch.undo()

    assert (tmp_path / "tier" / "journal.log").read_bytes() == b""   # nothing journaled
    assert store._ssd_used == 0
    assert all(s.refs == 0 for s in store._segments.values())        # no leaked pins
    assert all(s.blob == "" and s.l2_off == -1 for s in store._segments.values())
    assert store.flush_live() == 3        # retry loses no segments
    boot = SessionTierStore(_cfg(ram=16384, ssd=64 * 4096, d=d))
    assert boot.replay_journal() == 3


def test_account_lock_stress_no_lost_updates(tmp_path):
    """The _account_lock guards the two concurrent _ssd_used/_dead_bytes mutation sites
    (the writer thread's journal accounting vs the builder thread's cap-eviction
    discard): M interleaved real _journal_and_account and real _discard calls must
    converge exactly - zero lost updates."""
    from freetoken.scheduler.session_tier import _Segment

    d = str(tmp_path / "tier")
    store = SessionTierStore(_cfg(ram=0, ssd=4 << 20, d=d))    # L2-only runtime offers
    m = 300
    for i in range(m):
        keys, pages = _chain([(i, 0)])
        assert store.offer(f"p{i}".encode(), 1, pages)
    l2 = [s for s in list(store._segments.values()) if s.in_l2]
    assert len(l2) == m
    base_ssd, base_dead = store._ssd_used, store._dead_bytes

    def journal_side():
        for i in range(m):
            seg = _Segment(seg_id=-(i + 1), path_key=bytes([i % 256]) * 8,
                           boundary_len=1, page_keys=[], page_lens=[], snap_lens=[])
            assert store._journal_and_account(seg, [(0, 0, 0, "crc32c", 0)],
                                               4096, 0, off=0)

    def discard_side():
        for seg in l2:
            store._discard(seg)

    t1 = threading.Thread(target=journal_side)
    t2 = threading.Thread(target=discard_side)
    t1.start(); t2.start(); t1.join(); t2.join()

    assert store._ssd_used == base_ssd + m * 4096 - m * 4096   # exact: nothing lost
    assert store._dead_bytes == base_dead + m * 4096
    # Deterministic binding: BOTH real mutation sites must be under the lock (the
    # stress above detects lost updates only probabilistically).
    import inspect
    src = (inspect.getsource(SessionTierStore._journal_and_account)
           + inspect.getsource(SessionTierStore._discard))
    assert src.count("_account_lock") >= 2


def test_pwritev_iov_chunking_byte_exact(tmp_path, monkeypatch):
    """The _pwritev_span cursor must split a record across _PWRITEV_MAX_IOV-sized
    pwritev calls without losing or duplicating a byte: a many-page segment written
    with an iovec cap of 2 lands byte-exact in the blob and restores byte-exactly."""
    import freetoken.scheduler.session_tier as st_mod

    monkeypatch.setattr(st_mod, "_PWRITEV_MAX_IOV", 2)
    d = str(tmp_path / "tier")
    store = SessionTierStore(_cfg(ram=16384, ssd=64 * 4096, d=d))
    keys, pages = _chain([(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5)])
    assert store.offer(b"p1", 6, pages, _snap("1"))
    gold = _gold_payload(store, store._by_path(b"p1"))

    assert store.flush_live() == 1
    with open(os.path.join(d, "blob.bin"), "rb") as f:
        blob = f.read()
    boot = SessionTierStore(_cfg(ram=16384, ssd=64 * 4096, d=d))
    assert boot.replay_journal() == 1
    seg = boot._by_path(b"p1")
    assert blob[seg.l2_off:seg.l2_off + len(gold)] == gold
    hit = boot.probe(keys)
    assert hit is not None and hit[0] == 6
    got_p, got_s = boot.restore(hit[1])
    assert got_p == [data for _, data in pages] and got_s == [_snap("1")]


def test_flush_shared_dedup_page_across_groups_byte_exact(tmp_path, monkeypatch):
    """Two segments sharing deduped L1 pages (equal prefix) flushed in DIFFERENT groups:
    the zero-copy iovecs of both records read the same pool regions, the shared page is
    released only after the second segment's group settles, and both records land
    byte-exact with boot-time probe/restore intact for both segments."""
    import freetoken.scheduler.session_tier as st_mod

    monkeypatch.setattr(st_mod, "_FLUSH_GROUP_BYTES", 1)   # every segment = own group
    d = str(tmp_path / "tier")
    store = SessionTierStore(_cfg(ram=16384, ssd=64 * 4096, d=d))
    keys_a, pages_a = _chain([(1, 0), (1, 1), (1, 2)])
    keys_b, pages_b = _chain([(1, 0), (1, 1), (7, 2)])     # shares the first two pages
    assert store.offer(b"pa", 3, pages_a, _snap("a"))
    assert store.offer(b"pb", 3, pages_b, _snap("b"))
    shared = keys_a[:2]
    assert store._pages[shared[0]].refs == 2 and store._pages[shared[1]].refs == 2
    gold_a = _gold_payload(store, store._by_path(b"pa"))
    gold_b = _gold_payload(store, store._by_path(b"pb"))

    assert store.flush_live() == 2
    assert not store._pages                    # shared pages freed after both groups
    assert store._l1_used == 0
    with open(os.path.join(d, "blob.bin"), "rb") as f:
        blob = f.read()
    boot = SessionTierStore(_cfg(ram=16384, ssd=64 * 4096, d=d))
    assert boot.replay_journal() == 2
    for name, keys, pages, gold in ((b"pa", keys_a, pages_a, gold_a),
                                    (b"pb", keys_b, pages_b, gold_b)):
        seg = boot._by_path(name)
        assert blob[seg.l2_off:seg.l2_off + len(gold)] == gold
        hit = boot.probe(keys)
        assert hit is not None and hit[0] == 3
        got_p, got_s = boot.restore(hit[1])
        assert got_p == [data for _, data in pages]
        assert got_s == [_snap(name.decode()[1])]


# ---------------------------------------------------- P6: flush env knobs


def _big_pages(tag):
    """One 1 MiB page: a segment's padded volume (~1.004 MiB) exceeds the 1 MiB env
    floor, so FREETOKEN_FLUSH_GROUP_BYTES=1 MiB deterministically makes every such
    segment its own group while the 1 GiB default keeps one group."""
    data = bytes([0x41 + (tag % 26)]) * (1 << 20)
    key = chain_page_key(None, (tag, 0))
    return [key], [(key, data)]


def _big_store(d):
    return SessionTierStore(_cfg(ram=8 * 1024 * 1024, ssd=8 * 1024 * 1024, d=d))


def _wrap_executor(monkeypatch, st_mod):
    """Record every ThreadPoolExecutor the module builds: (max_workers,
    thread_name_prefix, map item count). The pwrite executor has no thread_name_prefix;
    the P4 flush-builder is 'flush-build'."""
    real = st_mod.ThreadPoolExecutor
    calls = []

    def factory(*a, **kw):
        ex = real(*a, **kw)
        entry = [kw.get("max_workers", a[0] if a else None),
                 kw.get("thread_name_prefix"), None]
        calls.append(entry)
        real_map = ex.map

        def counting_map(fn, iterable, *margs):
            items = list(iterable)
            entry[2] = len(items)
            return real_map(fn, items, *margs)

        ex.map = counting_map
        return ex

    monkeypatch.setattr(st_mod, "ThreadPoolExecutor", factory)
    return calls


def test_flush_env_unset_uses_module_constants(tmp_path, monkeypatch):
    """P6 (a/d): with both env knobs unset the resolvers return the module constants,
    INCLUDING patched ones - the existing monkeypatch.setattr(_FLUSH_GROUP_BYTES, ...)
    tests keep working."""
    import freetoken.scheduler.session_tier as st_mod

    assert st_mod._flush_writers(st_mod._FLUSH_WRITERS) == st_mod._FLUSH_WRITERS
    assert (st_mod._flush_group_bytes(st_mod._FLUSH_GROUP_BYTES)
            == st_mod._FLUSH_GROUP_BYTES)
    monkeypatch.setattr(st_mod, "_FLUSH_WRITERS", 5)
    monkeypatch.setattr(st_mod, "_FLUSH_GROUP_BYTES", 12345)
    assert st_mod._flush_writers(5) == 5
    assert st_mod._flush_group_bytes(12345) == 12345


def test_flush_env_valid_writers_pools_pwrite_executor(tmp_path, monkeypatch):
    """P6 (a): FREETOKEN_FLUSH_WRITERS=1 reaches the flush path - the pwrite executor
    of a 2-segment group is built with max_workers=1 (vs >= 2 unset), the flush is
    byte-exact on boot."""
    import freetoken.scheduler.session_tier as st_mod

    monkeypatch.setenv("FREETOKEN_FLUSH_WRITERS", "1")
    d = str(tmp_path / "tier")
    store = _big_store(d)
    chains = []
    for i in (1, 2):
        keys, pages = _big_pages(i)
        assert store.offer(f"p{i}".encode(), 1, pages, _snap(f"{i}"))
        chains.append((keys, pages))

    calls = _wrap_executor(monkeypatch, st_mod)
    assert store.flush_live() == 2
    pwrite = [c for c in calls if c[1] is None]        # no thread_name_prefix
    assert len(pwrite) == 1
    assert pwrite[0][0] == 1 and pwrite[0][2] >= 2     # env applied, both stage entries

    boot, n = _boot(d)
    assert n == 2
    for keys, pages in chains:
        hit = boot.probe(keys)
        assert hit is not None and hit[0] == 1
        assert boot.restore(hit[1])[0] == [data for _, data in pages]


def test_flush_env_valid_group_bytes_splits_groups(tmp_path, monkeypatch):
    """P6 (a): FREETOKEN_FLUSH_GROUP_BYTES=1 MiB reaches the grouping loop - two ~1 MiB
    segments land in TWO groups (the P4 builder executor appears) vs ONE group unset;
    both runs flush byte-exact."""
    import freetoken.scheduler.session_tier as st_mod

    chains = [(_big_pages(i)) for i in (1, 2)]
    for env_group_bytes, expect_builder in (("1048576", True), (None, False)):
        if env_group_bytes is not None:
            monkeypatch.setenv("FREETOKEN_FLUSH_GROUP_BYTES", env_group_bytes)
        else:
            monkeypatch.delenv("FREETOKEN_FLUSH_GROUP_BYTES", raising=False)
        d = str(tmp_path / f"tier-{int(expect_builder)}")
        store = _big_store(d)
        for i, (keys, pages) in enumerate(chains, start=1):
            assert store.offer(f"p{i}".encode(), 1, pages, _snap(f"{i}"))
        calls = _wrap_executor(monkeypatch, st_mod)
        assert store.flush_live() == 2
        builders = [c for c in calls if c[1] == "flush-build"]
        assert bool(builders) is expect_builder
        boot, n = _boot(d)
        assert n == 2
        for i, (keys, pages) in enumerate(chains, start=1):
            hit = boot.probe(keys)
            assert hit is not None and hit[0] == 1
            assert boot.restore(hit[1])[0] == [data for _, data in pages]


def test_flush_env_below_minimum_falls_back_with_warning(monkeypatch, tier_log_capture):
    """P6 (b): a set-but-below-minimum env knob falls back to the default with a
    warning, never a silent clamp or a zero/negative value reaching the pipeline."""
    import freetoken.scheduler.session_tier as st_mod

    monkeypatch.setenv("FREETOKEN_FLUSH_WRITERS", "0")
    monkeypatch.setenv("FREETOKEN_FLUSH_GROUP_BYTES", "1")   # below the 1 MiB floor
    with tier_log_capture() as cap:
        assert st_mod._flush_writers(st_mod._FLUSH_WRITERS) == st_mod._FLUSH_WRITERS
        assert (st_mod._flush_group_bytes(st_mod._FLUSH_GROUP_BYTES)
                == st_mod._FLUSH_GROUP_BYTES)
    warnings = [m for m in cap.messages if "FREETOKEN_FLUSH_" in m]
    assert len(warnings) == 2
    assert any("below minimum 1" in m and "FREETOKEN_FLUSH_WRITERS" in m
               for m in warnings)
    assert any("below minimum 1048576" in m and "FREETOKEN_FLUSH_GROUP_BYTES" in m
               for m in warnings)


def test_flush_env_garbage_falls_back_with_warning(monkeypatch, tier_log_capture):
    """P6 (c): non-integer env values fall back to the default with a warning."""
    import freetoken.scheduler.session_tier as st_mod

    for name in ("FREETOKEN_FLUSH_WRITERS", "FREETOKEN_FLUSH_GROUP_BYTES"):
        for garbage in ("abc", "12abc", ""):
            if garbage:
                monkeypatch.setenv(name, garbage)
            else:
                monkeypatch.delenv(name, raising=False)
            with tier_log_capture() as cap:
                w = st_mod._flush_writers(st_mod._FLUSH_WRITERS)
                g = st_mod._flush_group_bytes(st_mod._FLUSH_GROUP_BYTES)
            assert w == st_mod._FLUSH_WRITERS and g == st_mod._FLUSH_GROUP_BYTES
            garbage_warnings = [m for m in cap.messages if name in m
                                and "non-integer" in m]
            assert bool(garbage_warnings) is bool(garbage)   # unset is silent
            if garbage:
                assert garbage in garbage_warnings[0]


def test_flush_config_log_line_one_per_flush(tmp_path, monkeypatch, tier_log_capture):
    """P6 (e): the flush logs ONE config line with the effective writers/group_bytes,
    once per flush phase (not per group) - the iron A/B attributes a run to the right
    knob even when the log tail is skimmed."""

    monkeypatch.setenv("FREETOKEN_FLUSH_WRITERS", "3")
    monkeypatch.setenv("FREETOKEN_FLUSH_GROUP_BYTES", "1048576")   # 2 groups
    d = str(tmp_path / "tier")
    store = _big_store(d)
    for i in (1, 2):
        keys, pages = _big_pages(i)
        assert store.offer(f"p{i}".encode(), 1, pages, _snap(f"{i}"))
    with tier_log_capture() as cap:
        assert store.flush_live() == 2
    config = [m for m in cap.messages if "flush pipeline" in m]
    assert len(config) == 1
    assert "writers=3" in config[0] and "group_bytes=1048576" in config[0]


def test_flush_env_garbage_writers_warns_once_per_flush(tmp_path, monkeypatch, tier_log_capture):
    """TP fix fails-before: writers resolved once per flush and passed down. Four
    384 KiB segments with group_bytes=1 MiB make TWO multi-segment groups; before the
    fix a garbage writers env warned at flush start AND once per group's pwrite
    executor (G+1 = 3 lines) - now exactly ONE warning per flush, and the logged
    config matches the applied pool size."""

    monkeypatch.setenv("FREETOKEN_FLUSH_WRITERS", "abc")
    monkeypatch.setenv("FREETOKEN_FLUSH_GROUP_BYTES", "1048576")   # groups of 2
    d = str(tmp_path / "tier")
    store = _big_store(d)
    for i in range(1, 5):
        data = bytes([0x41 + i]) * (384 * 1024)
        keys = [chain_page_key(None, (i, 0))]
        assert store.offer(f"p{i}".encode(), 1, [(keys[0], data)], _snap(f"{i}"))
    with tier_log_capture() as cap:
        assert store.flush_live() == 4
    writer_warnings = [m for m in cap.messages
                       if "FREETOKEN_FLUSH_WRITERS" in m and "non-integer" in m]
    assert len(writer_warnings) == 1


# ------------------------------------------------- P13: durable tombstone records


def _frame(rec: dict) -> bytes:
    """One journal frame (the framing every record shares - zlib crc32, unchanged)."""
    payload = json.dumps(rec, sort_keys=True).encode()
    return struct.pack("<II", len(payload), zlib.crc32(payload)) + payload


def _graceful_stop_p13(store):
    """Mirror CacheManager.shutdown_tier's P13 discipline: drop any previous marker
    FIRST, flush, OPTIONAL compact, then the end-of-shutdown barrier."""
    store.invalidate_shutdown_marker()
    store.flush_live()
    store.shutdown_compact()
    store.write_shutdown_marker()


def test_tombstone_discard_survives_restart_without_compact(tmp_path, monkeypatch):
    """P13 fails-before: an evicted L2 record must stay dead across a boot that ran no
    compact at all (and left no marker). Pre-P13 the record resurrected: the journal
    still listed it and its payload crc still verified over the append-only blob.
    The tombstone append must also be fsync'ed to be durable: removing os.fsync from
    _append_tombstone has to fail the fsync-count assert below."""
    fsyncs = {"n": 0}
    real_fsync = os.fsync

    def counting_fsync(fd):
        fsyncs["n"] += 1
        return real_fsync(fd)

    monkeypatch.setattr("freetoken.scheduler.session_tier.os.fsync", counting_fsync)
    d, store, data = _shrink_fixture(tmp_path, n=3, dead=())
    before = fsyncs["n"]
    assert store.evict_store(1) == 1            # discards record 1, one tombstone
    assert store._counters["tombstones"] == 1
    assert fsyncs["n"] > before                 # the tombstone append fsynced first
    boot, n = _boot(str(d))
    assert n == 2                               # the tombstone keeps record 1 dead
    assert boot.probe([chain_page_key(None, (1,))]) is None
    _restore_all(boot, data, [2, 3])


def test_legacy_records_replay_unchanged_next_to_hand_framed_tombstone(tmp_path):
    """P13 compatibility: existing records gain NO new fields (a store-written journal
    has no "t" keys); a hand-framed tombstone is honored; a legacy zlib-crc32 record
    without crc_alg (the pre-P3 shape) still verifies and replays live next to them;
    the filter is order-independent (the tombstone lands after the legacy record but
    kills an earlier one)."""
    d, store, data = _shrink_fixture(tmp_path, n=3, dead=())
    recs, whole = _journal_records_raw(str(d / "journal.log"))
    assert whole and len(recs) == 3
    assert all("t" not in rec for rec in recs)          # the old format is untouched
    victim = recs[0]

    k9 = chain_page_key(None, (9,))
    data9 = bytes([9]) * PAGE
    with open(d / "blob.bin", "r+b") as f:
        f.seek(0, os.SEEK_END)
        off9 = f.tell()
        f.write(data9.ljust(4096, b"\0"))       # the full padded span, like _write_blob
    legacy = {"path_key": k9.hex(), "blen": 1, "page_keys": [k9.hex()],
              "page_lens": [PAGE], "snap_lens": [], "blob": "blob.bin",
              "off": off9, "n": PAGE, "ts": 1,
              "crc": zlib.crc32(data9)}                 # no crc_alg: the pre-P3 shape
    with open(d / "journal.log", "ab") as f:
        f.write(_frame(legacy))
        f.write(_frame({"t": 1, "path_key": victim["path_key"],
                        "off": victim["off"], "n": victim["n"]}))

    boot, n = _boot(str(d))
    assert n == 3                              # rec1 tombstoned away; 2, 3 and 9 live
    assert boot.probe([chain_page_key(None, (1,))]) is None
    hit = boot.probe([k9])
    assert hit is not None and hit[0] == 1     # the legacy record is live
    assert boot.restore(hit[1])[0] == [data9]
    _restore_all(boot, data, [2, 3])


def test_shutdown_compact_skipped_all_dead_tombstoned_marker_fast_path(tmp_path, monkeypatch, tier_log_capture):
    """P13 core: with every dead record tombstoned and the watermark above the P8
    floor the shutdown compact SKIPS (no blob rewrite), the marker still WRITES (they
    cannot fast-path-resurrect), and the next boot fast-paths with zero preads, the
    victim gone and the P8 ledger honest."""
    import freetoken.scheduler.session_tier as st_mod

    d, store, data = _shrink_fixture(tmp_path, n=3, dead=(1,), blob_above_floor=True)
    assert store._blob_eof >= st_mod._COMPACT_FLOOR_BYTES   # the skip gate's premise
    with tier_log_capture() as cap:
        _graceful_stop_p13(store)
    assert any("shutdown compact skipped" in m for m in cap.messages)
    assert os.path.exists(_marker_path(d))
    assert ((d / "blob.bin").stat().st_size
            == st_mod._COMPACT_FLOOR_BYTES + 4096)          # NOT rewritten

    def boom(*args, **kwargs):
        raise AssertionError("payload pread must not happen on the fast path")

    monkeypatch.setattr("freetoken.scheduler.session_tier.os.pread", boom)
    with tier_log_capture() as cap:
        boot, n = _boot(str(d))
    assert n == 2
    assert any("replay fast-path (clean shutdown marker)" in m for m in cap.messages)
    assert any("replay honored 1 tombstones" in m for m in cap.messages)
    assert boot.probe([chain_page_key(None, (1,))]) is None
    assert boot._dead_records == 1 and boot._dead_record_bytes == 4096   # honest ledger
    _restore_all(boot, data, [2, 3])


def test_shutdown_compact_below_floor_rewrites_small_tier(tmp_path, tier_log_capture):
    """P13 gate refine (fails-before): a tier whose blob stays below the P8 compact
    floor keeps the pre-P13 shutdown hygiene - the compact RUNS even though every
    dead record is already tombstoned (the runtime gate can never fire down there,
    so the skip would leave the holes and the tombstones unreclaimed forever)."""
    import freetoken.scheduler.session_tier as st_mod

    d, store, data = _shrink_fixture(tmp_path, n=3, dead=(1,))
    assert store._blob_eof < st_mod._COMPACT_FLOOR_BYTES    # below the floor
    with tier_log_capture() as cap:
        _graceful_stop_p13(store)
    assert any("compact shrink" in m for m in cap.messages)
    assert not any("shutdown compact skipped" in m for m in cap.messages)
    assert (d / "blob.bin").stat().st_size == 2 * 4096      # holes reclaimed
    assert os.path.exists(_marker_path(d))
    boot, n = _boot(str(d))
    assert n == 2 and boot._dead_records == 0               # ledger reset by the rewrite
    assert boot.probe([chain_page_key(None, (1,))]) is None
    recs, whole = _journal_records_raw(str(d / "journal.log"))
    assert whole and len(recs) == 2
    assert all("t" not in rec for rec in recs)              # tombstones not carried
    _restore_all(boot, data, [2, 3])


def test_full_replay_ledger_mixed_tombstoned_and_payload_dead(tmp_path, tier_log_capture):
    """P13 ledger exactness on the FULL replay path with mixed dead: a tombstoned
    victim (v) and a payload-crc-dead record (pdropped) each feed the P8 ledger
    exactly once (dead = v + pdropped), while only the untombstoned payload-dead
    block the marker (blockers = pdropped) - a split the live count n cannot show."""
    d, store, data = _shrink_fixture(tmp_path, n=4, dead=(1,))          # v = 1
    with open(d / "blob.bin", "r+b") as f:      # corrupt record 2: pdropped = 1
        f.seek(4096)
        f.write(b"\\xde" * 256)
    with tier_log_capture() as cap:
        boot, n = _boot(str(d))                 # full path: no marker was ever written
    assert n == 2                               # live: 3, 4
    assert any("replay honored 1 tombstones" in m for m in cap.messages)
    assert any("replay dropped 1 dead/torn records" in m for m in cap.messages)
    assert boot._dead_records == 2              # dead = v + pdropped = 1 + 1
    assert boot._dead_record_bytes == 2 * 4096  # both padded spans
    assert boot._marker_blockers == 1           # blockers = pdropped, not the victim
    assert boot.probe([chain_page_key(None, (1,))]) is None
    assert boot.probe([chain_page_key(None, (2,))]) is None
    _restore_all(boot, data, [3, 4])


def test_tombstone_append_failure_forces_shutdown_compact(tmp_path, monkeypatch):
    """A failed tombstone append (EIO) leaves the discard compact-durable only: the
    un-tombstoned dead become a marker blocker and the shutdown compact falls back to
    the classic rewrite (the pre-P13 shutdown, byte-for-byte). After it the marker is
    legal again (the ledger and the blockers reset with the rewritten journal)."""
    d, store, data = _shrink_fixture(tmp_path, n=3, dead=())
    for i in (2, 3):                        # keep 2/3 warm: LRU evicts exactly 1
        hit = store.probe([chain_page_key(None, (i,))])
        store.restore(hit[1])
    monkeypatch.setattr(store, "_append_tombstone", lambda pk, off, n: False)
    assert store.evict_store(1) == 1            # the patched append bypasses the logger
    assert store._marker_blockers == 1
    _graceful_stop_p13(store)
    assert (d / "blob.bin").stat().st_size == 2 * 4096      # the rewrite RAN
    assert os.path.exists(_marker_path(d))                  # legal again after compact
    monkeypatch.undo()
    calls = _counting_pread(monkeypatch)
    boot, n = _boot(str(d))
    assert n == 2 and calls["n"] == 0                       # fast path: journal == live
    _restore_all(boot, data, [2, 3])


def test_marker_skipped_when_blocker_and_compact_deferred(tmp_path, monkeypatch, tier_log_capture):
    """B3 survives for un-tombstoned dead: a failed tombstone append + a deferred
    compact (blob reads in flight) leave dead records WITHOUT tombstones - the marker
    must not write, or a fast-path boot would resurrect them (the pre-P13 behavior)."""
    d, store, data = _shrink_fixture(tmp_path, n=3, dead=())
    for i in (2, 3):
        hit = store.probe([chain_page_key(None, (i,))])
        store.restore(hit[1])
    monkeypatch.setattr(store, "_append_tombstone", lambda pk, off, n: False)
    assert store.evict_store(1) == 1
    monkeypatch.undo()
    hit = store.probe([chain_page_key(None, (2,))])
    seg = store._segments[hit[1]._seg_id]
    seg.refs += 1                       # a blob read in flight: compact defers
    with tier_log_capture() as cap:
        _graceful_stop_p13(store)
    seg.refs -= 1
    assert any("compaction deferred" in m for m in cap.messages)
    assert any("marker skipped" in m for m in cap.messages)
    assert not os.path.exists(_marker_path(d))
    boot, n = _boot(str(d))             # full path; the zombie returns (no tombstone)
    assert n == 3                       # the pre-P13 discard durability, unchanged


def test_torn_tombstone_never_applied_converges(tmp_path, tier_log_capture):
    """Crash window of a torn tombstone: a torn tombstone tail fails the framing crc,
    replay stops at the last complete record and converges to the pre-discard state -
    the same semantics as a torn live-record append (no corruption, no live loss)."""
    d, store, data = _shrink_fixture(tmp_path, n=3, dead=())
    with open(d / "journal.log", "ab") as f:
        f.write(struct.pack("<I", 18) + b'{"t":1,"pat')   # torn tombstone tail
    with tier_log_capture() as cap:
        boot, n = _boot(str(d))
    assert any("journal tail truncated/torn" in m for m in cap.messages)
    assert n == 3
    _restore_all(boot, data, [1, 2, 3])


def test_tombstone_for_unknown_victim_is_harmless(tmp_path):
    """A tombstone whose victim is already reclaimed (compact dropped both) is a no-op
    at replay: nothing crashes, nothing extra is dropped."""
    d, store, data = _shrink_fixture(tmp_path, n=3, dead=(1,))
    assert store.compact() == 1                     # victim + tombstone reclaimed
    with open(d / "journal.log", "ab") as f:
        f.write(_frame({"t": 1, "path_key": (bytes([1]) * 16).hex(),
                        "off": 0, "n": PAGE}))
    boot, n = _boot(str(d))
    assert n == 2
    _restore_all(boot, data, [2, 3])


def test_tombstone_counter_in_snapshot_and_stats_line(tmp_path):
    """Every durable tombstone is counted (snapshot counter + stats_line fragment)."""
    d, store, data = _shrink_fixture(tmp_path, n=3, dead=(1,))
    assert store.snapshot()["tombstones"] == 1
    assert "tomb=1" in store.stats_line()
    fresh = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(tmp_path / "fresh")))
    assert fresh.snapshot()["tombstones"] == 0
    assert "tomb=0" in fresh.stats_line()


# ------------------------------------------------- L2 extent dedup (l2-snapshot-dedup)

def _extent_chain_store(d, boundaries):
    """L2-only store holding `boundaries` snapshot-bearing same-chain boundaries:
    boundaries = [(tag, n_pages, snap_tag), ...]; returns (store, per-boundary
    (path_key, keys, pages) list). Each boundary is offered under its own path key."""
    store = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d)))
    out = []
    token_lists = []
    for i in range(max(n for _, n, _ in boundaries)):
        token_lists.append((7, i))
    keys, pages = _chain(token_lists)
    for tag, n, snap_tag in boundaries:
        assert store.offer(f"path-{tag}".encode(), n, pages[:n], _snap(snap_tag))
        out.append((f"path-{tag}".encode(), keys[:n], pages[:n], _snap(snap_tag)))
    return store, out


def test_extent_write_dedups_contained_snapshot_bearing_offer(tmp_path):
    """Core dedup: a deeper snapshot-boundary offer references the shallower boundary's
    page extents and appends ONLY its tail pages + own snapshot - the unit-scale
    analogue of 23.81 -> ~7 GiB. offers_dedup (snapshot-free path) stays untouched."""
    d = tmp_path / "tier"
    store, segs = _extent_chain_store(d, [(1, 2, "a"), (2, 4, "b")])
    snap = store.snapshot()
    assert snap["extents_shared"] == 1          # one page extent referenced, not copied
    assert snap["offers_dedup"] == 0            # snapshot-bearing offers are always stored
    assert snap["ssd_used"] == 2 * 4096         # unique bytes: b1 span + b2 private span
    seg1 = store._segments[min(store._segments)]
    recs, whole = _journal_records_raw(os.path.join(str(d), "journal.log"))
    assert whole and len(recs) == 2
    assert [len(r["extents"]) for r in recs] == [2, 3]   # [pages][snaps] / [shared][tail][snaps]
    assert recs[1]["off"] == recs[0]["off"] and recs[1]["n"] == sum(recs[0]["page_lens"])
    # both boundaries restore byte-exact
    for path_key, keys, pages, s in segs:
        hit = store.probe(keys)
        assert hit is not None and hit[0] == len(keys)
        assert store.restore(hit[1]) == ([data for _, data in pages], [s])


def test_extent_restore_byte_exact_across_shared_tail_seam(tmp_path):
    """Every boundary depth of shared records restores byte-exact - depth truncation
    crosses the shared/tail extent seam - and the boundary-EXACT probe preference is
    unchanged; prefetch staging across extents lands byte-identical too."""
    d = tmp_path / "tier"
    store, segs = _extent_chain_store(d, [(1, 1, "a"), (2, 2, "b"), (3, 4, "c")])
    # b2 references b1's page extent; b3 references b1's + b2's tail-page extents
    assert store.snapshot()["extents_shared"] == 3
    for path_key, keys, pages, s in segs:
        hit = store.probe(keys)
        assert hit is not None and hit[0] == len(keys)      # boundary-exact still wins
        assert store.restore(hit[1]) == ([data for _, data in pages], [s])
    deepest = segs[-1]
    ticket = store.begin_restore(TierHandle(deepest[0], len(deepest[1]),
                                            store._by_path(deepest[0]).seg_id))
    assert ticket is not None
    _settle(store)
    got = store.consume_ticket(ticket)
    assert got == ([data for _, data in deepest[2]], [deepest[3]])


def test_extent_replay_derives_refcounts_orphan_span_converges_dead(tmp_path):
    """Refcounts are derived from the surviving record set on boot; a span appended but
    never journaled (crash before the record) is an unreferenced orphan: never counted
    as live, reclaimed by the next compaction that fires."""
    d = tmp_path / "tier"
    store, segs = _extent_chain_store(d, [(1, 2, "a"), (2, 4, "b")])
    with open(d / "blob.bin", "r+b") as f:
        f.seek(0, os.SEEK_END)
        f.write(b"\xab" * 4096)                 # orphan span, no journal record
    boot = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d)))
    assert boot.replay_journal() == 2
    assert boot._ssd_used == 2 * 4096           # the orphan is not live capacity
    assert boot._dead_bytes == 4096             # watermark minus live
    assert len(boot._extents) == 4              # pages+snaps of b1, private pages+snaps of b2
    recs, _ = _journal_records_raw(os.path.join(str(d), "journal.log"))
    shared = (recs[0]["extents"][0][0], recs[0]["extents"][0][1])
    assert boot._extents[shared][0] == 2        # referenced by BOTH records
    # reclaim: discard b1 (tombstone -> dead record), compact reclaims b1-only bytes
    # AND the orphan; the shared span survives via b2 and is copied exactly once
    seg1 = [s for s in boot._segments.values()
            if s.path_key == segs[0][0]][0]
    boot.evict_store(1)
    assert boot._dead_records == 1
    assert boot.compact() == 1
    # the shared page extent (512 B, interior) is copied alone -> own aligned block:
    # 4096 (aligned shared span) + 4096 (b2's private tail+snap run)
    assert boot._ssd_used == 2 * 4096
    hit = boot.probe(segs[1][1])
    assert boot.restore(hit[1]) == ([data for _, data in segs[1][2]], [segs[1][3]])
    final = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d)))
    assert final.replay_journal() == 1
    assert final._ssd_used == 2 * 4096
    hit = final.probe(segs[1][1])
    assert final.restore(hit[1]) == ([data for _, data in segs[1][2]], [segs[1][3]])


def test_extent_torn_journal_tail_after_extent_record(tmp_path):
    """A SIGKILL mid-journal-append after an extent record: replay keeps exactly the
    complete prefix, the torn record's private bytes stay durable-but-unjournaled
    holes, and the surviving shared records restore byte-exact."""
    d = tmp_path / "tier"
    store, segs = _extent_chain_store(d, [(1, 1, "a"), (2, 2, "b"), (3, 4, "c")])
    raw = (d / "journal.log").read_bytes()
    pos, sizes = 0, []
    while pos + 8 <= len(raw):
        plen, _ = struct.unpack_from("<II", raw, pos)
        pos += 8 + plen
        sizes.append(pos)
    with open(d / "journal.log", "r+b") as f:
        f.truncate(sizes[1] + 5)                # mid-third-record
    boot = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d)))
    assert boot.replay_journal() == 2
    recs, _ = _journal_records_raw(os.path.join(str(d), "journal.log"))
    # the surviving records still share: b2's first extent IS b1's pages span
    assert recs[1]["extents"][0][0] == recs[0]["extents"][0][0]
    assert boot._extents[(recs[0]["extents"][0][0],
                          recs[0]["extents"][0][1])][0] == 2
    for path_key, keys, pages, s in segs[:2]:
        hit = boot.probe(keys)
        assert hit is not None
        assert boot.restore(hit[1]) == ([data for _, data in pages], [s])
    hit = boot.probe(segs[2][1])
    assert hit is None or hit[0] < len(segs[2][1])   # the torn boundary never serves


def test_extent_dual_read_legacy_journal_boots_and_compact_migrates(tmp_path):
    """Backward compat: a journal whose records carry NO extents field (every pre-L2
    directory) boots via dual-read, restores byte-exact; the first compact migrates the
    records to the extent format (pages|snaps split) after which a contained offer
    SHARES the migrated page extents."""
    d = tmp_path / "tier"
    store = SessionTierStore(_cfg(ram=16384, ssd=1 << 20, d=str(d)))
    chains = []
    for i in range(1, 4):
        keys, pages = _chain([(i, 0), (i, 1), (i, 2)])
        assert store.offer(f"p{i}".encode(), 3, pages, _snap(f"{i}"))
        chains.append((f"p{i}".encode(), keys, pages, _snap(f"{i}")))
    assert store.flush_live() == 3
    # rewrite to the legacy format: single whole-payload span with a whole-region crc
    # (stripping extents alone would leave the LOGICAL crc over a partial region - the
    # exact clean-downgrade drop an old binary performs on multi-extent records)
    import crc32c as _crc32c_mod

    def to_legacy(rec):
        whole = sum(rec["page_lens"]) + sum(rec["snap_lens"])
        with open(os.path.join(str(d), "blob.bin"), "rb") as f:
            f.seek(rec["off"])
            body = f.read(whole)
        rec.pop("extents")
        rec["n"] = whole
        rec["crc"] = _crc32c_mod.crc32c(body)
        rec["crc_alg"] = "crc32c"

    _rewrite_records(str(d), to_legacy)
    boot = SessionTierStore(_cfg(ram=16384, ssd=1 << 20, d=str(d)))
    assert boot.replay_journal() == 3
    assert boot._ssd_used == 3 * 4096
    for name, keys, pages, s in chains:
        hit = boot.probe(keys)
        assert boot.restore(hit[1]) == ([data for _, data in pages], [s])
    assert boot.compact() == 0                  # nothing dead: no-op, no migration
    recs, _ = _journal_records_raw(os.path.join(str(d), "journal.log"))
    assert all("extents" not in r for r in recs)
    # force the migration: discard one record -> compact fires and rewrites the format
    victim = boot._by_path(b"p1")
    boot.evict_store(1)
    assert boot.compact() == 1
    recs, _ = _journal_records_raw(os.path.join(str(d), "journal.log"))
    assert len(recs) == 2 and all(len(r["extents"]) == 2 for r in recs)
    hit = boot.probe(chains[1][1])
    assert boot.restore(hit[1]) == ([data for _, data in chains[1][2]], [chains[1][3]])
    # a contained snapshot-bearing offer now SHARES the migrated page extents
    boot = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d)))   # L2-only offers
    assert boot.replay_journal() == 2
    keys1, pages1 = chains[1][1], chains[1][2]
    keys4 = keys1 + [chain_page_key(keys1[-1], (1, 3))]
    pages4 = pages1 + [(keys4[3], _page_data((1, 3)))]
    assert boot.offer(b"p2deep", 4, pages4, _snap("deep"))
    assert boot.snapshot()["extents_shared"] >= 1
    hit = boot.probe(keys4)
    assert boot.restore(hit[1]) == ([data for _, data in pages4], [_snap("deep")])


def test_extent_discard_frees_only_unique_and_shared_span_survives(tmp_path):
    """Discard of a record holding shared extents frees ONLY its private bytes and
    decrements the shared refcounts; the sharer keeps serving byte-exact. When the last
    referrer goes, the shared span's bytes become dead exactly once."""
    d = tmp_path / "tier"
    store, segs = _extent_chain_store(d, [(1, 2, "a"), (2, 4, "b")])
    assert store._ssd_used == 2 * 4096
    store.evict_store(1)                        # discards the LRU b1
    snap = store.snapshot()
    # b1's snap extent died with it (128 + its 3456 pad share); the pages extent stays
    # LIVE - b2 still references it (512 bytes remain counted in capacity)
    assert snap["ssd_used"] == 512 + 4096
    assert snap["dead_bytes"] == 128 + 3456
    hit = store.probe(segs[1][1])
    assert store.restore(hit[1]) == ([data for _, data in segs[1][2]], [segs[1][3]])
    store.evict_store(1)                        # b2: private + the now-lone shared span
    snap = store.snapshot()
    assert snap["ssd_used"] == 0
    assert snap["dead_bytes"] == 2 * 4096       # everything b1+b2 ever appended
    assert store._extents == {}


def test_extent_compact_keeps_shared_span_and_dedupes_copy(tmp_path):
    """Compaction with extents: the shared span survives while ANY referencing record
    survives, is copied exactly once (dense file == unique live volume), and the
    resurrected-two-records-per-path identity rule still holds byte-exactly."""
    d = tmp_path / "tier"
    store, segs = _extent_chain_store(d, [(1, 2, "a"), (2, 4, "b")])
    b2_seg = store._segments[max(store._segments)]
    store.evict_store(1)                        # b1 dead (tombstoned), shared span stays
    assert store.compact() == 1
    # dense == unique live volume: the shared 512 B page span copies to its own
    # aligned block (4096) plus b2's private tail+snap run (4096)
    assert (d / "blob.bin").stat().st_size == 2 * 4096
    assert store._ssd_used == 2 * 4096
    hit = store.probe(segs[1][1])
    assert store.restore(hit[1]) == ([data for _, data in segs[1][2]], [segs[1][3]])
    final = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d)))
    assert final.replay_journal() == 1
    hit = final.probe(segs[1][1])
    assert final.restore(hit[1]) == ([data for _, data in segs[1][2]], [segs[1][3]])


def test_extent_flush_shares_across_groups(tmp_path, monkeypatch):
    """The shutdown flush (the measured 23.81 GiB seam) shares too: every boundary
    after the first appends only its private span even when each segment is its own
    volume group; boot replays and restores byte-exact at every depth."""
    import freetoken.scheduler.session_tier as st_mod

    monkeypatch.setattr(st_mod, "_FLUSH_GROUP_BYTES", 1)   # every segment = own group
    d = str(tmp_path / "tier")
    store = SessionTierStore(_cfg(ram=16384, ssd=1 << 20, d=d))
    keys, pages = _chain([(7, 0), (7, 1), (7, 2), (7, 3)])
    want = []
    for i, tag in ((1, "a"), (2, "b"), (4, "c")):
        assert store.offer(f"p{i}".encode(), i, pages[:i], _snap(tag))
        want.append((f"p{i}".encode(), keys[:i], pages[:i], _snap(tag)))
    assert store.flush_live() == 3
    snap = store.snapshot()
    assert snap["extents_shared"] == 3          # b2 -> 1 share, b3 -> 2 page extents
    assert snap["ssd_used"] == 3 * 4096         # unique bytes only
    boot = SessionTierStore(_cfg(ram=16384, ssd=1 << 20, d=d))
    assert boot.replay_journal() == 3
    assert boot._ssd_used == 3 * 4096
    for name, kws, pgs, s in want:
        hit = boot.probe(kws)
        assert hit is not None and hit[0] == len(kws)
        assert boot.restore(hit[1]) == ([data for _, data in pgs], [s])


def test_extent_flush_dep_failure_keeps_dependent_in_l1(tmp_path, monkeypatch):
    """A flush entry whose extent SOURCE failed its write is not journaled either (its
    reference would point at a hole): both segments keep their L1 residency and the
    retried flush converges byte-exact."""
    import freetoken.scheduler.session_tier as st_mod

    monkeypatch.setattr(st_mod, "_FLUSH_GROUP_BYTES", 1)   # every segment = own group
    d = str(tmp_path / "tier")
    store = SessionTierStore(_cfg(ram=16384, ssd=1 << 20, d=d))
    keys, pages = _chain([(7, 0), (7, 1), (7, 2)])
    assert store.offer(b"p1", 1, pages[:1], _snap("a"))
    assert store.offer(b"p2", 2, pages[:2], _snap("b"))

    real = st_mod.os.pwritev
    state = {"n": 0}

    def fail_first(fd, buffers, offset):
        state["n"] += 1
        if state["n"] == 1:
            raise OSError(5, "simulated write failure")
        return real(fd, buffers, offset)

    monkeypatch.setattr(st_mod.os, "pwritev", fail_first)
    assert store.flush_live() == 0              # b1 failed; b2 dep-skipped
    monkeypatch.undo()
    assert len([s for s in store._segments.values() if s.in_l1]) == 2
    assert store._ssd_used == 0
    assert store.flush_live() == 2              # clean retry
    boot = SessionTierStore(_cfg(ram=16384, ssd=1 << 20, d=d))
    assert boot.replay_journal() == 2
    hit = boot.probe(keys[:2])
    assert boot.restore(hit[1]) == ([data for _, data in pages[:2]], [_snap("b")])


def test_extent_capacity_gauges_and_stats(tmp_path):
    """Capacity gauges stay exact across offer/discard/compact with sharing and the
    extents_shared counter is pre-seeded and rendered in the stats line."""
    d = tmp_path / "tier"
    store = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d)))
    assert store.snapshot()["extents_shared"] == 0
    assert "ext=0" in store.stats_line()
    keys, pages = _chain([(7, 0), (7, 1), (7, 2), (7, 3)])
    assert store.offer(b"p1", 2, pages[:2], _snap("a"))
    assert store.offer(b"p2", 4, pages, _snap("b"))
    line = store.stats_line()
    assert "ext=1" in line
    snap = store.snapshot()
    assert snap["ssd_used"] == 2 * 4096 and snap["dead_bytes"] == 0
    store.evict_store(1)
    snap = store.snapshot()
    # p1's snap extent died (128 + 3456 pad share); the shared pages extent stays
    # live via p2 (512) plus p2's private span (4096)
    assert snap["ssd_used"] == 512 + 4096 and snap["dead_bytes"] == 128 + 3456
    assert snap["dead_records"] == 1 and snap["dead_record_bytes"] == 128 + 3456
    assert store.compact() == 1
    snap = store.snapshot()
    # dense: aligned shared span + p2's private run
    assert snap["ssd_used"] == 2 * 4096 and snap["dead_bytes"] == 0
    assert snap["dead_records"] == 0 and snap["dead_record_bytes"] == 0


def test_extent_eviction_under_cap_frees_private_only(tmp_path):
    """Cap pressure with shared records: evictions free only unique bytes but victims
    always leave the candidate list - no livelock, offers keep succeeding, and the
    accounting never claims more freed space than the blob actually gave up."""
    d = tmp_path / "tier"
    store = SessionTierStore(_cfg(ram=0, ssd=6 * 4096, d=str(d)))
    keys, pages = _chain([(7, 0), (7, 1), (7, 2), (7, 3)])
    assert store.offer(b"p1", 2, pages[:2], _snap("a"))
    assert store.offer(b"p2", 4, pages, _snap("b"))
    assert store._ssd_used == 2 * 4096
    # a same-chain even deeper boundary still fits within the cap by sharing
    assert store.offer(b"p3", 3, pages[:3], _snap("c"))
    assert store._ssd_used == 3 * 4096
    assert store.snapshot()["extents_shared"] >= 2   # b2 and b3 share b1's page extent
    # divergent chains have nothing to share and push the cap: the oldest L2 records
    # are truly discarded (only their unique bytes count as freed)
    for i in range(10, 14):
        fk, fp = _chain([(i, 0), (i, 1)])
        assert store.offer(f"q{i}".encode(), 2, fp, _snap(f"q{i}"))
    assert store._ssd_used <= 6 * 4096
    assert store.probe(keys[:1]) is None or store.probe(keys) is not None


def test_extent_share_source_pinned_against_builder_eviction(tmp_path):
    """F1 regression (builder-evict vs journal): between the stage build and the
    journal, the builder thread's lock-free cap eviction can pick the refs==0 share
    source; _discard releases its extent entries and the dependent's journal then
    silently re-creates them - _ssd_used loses the shared span's bytes (can go
    negative on the dependent's later discard) and extents_shared misses the event.
    The build-time pin keeps the source out of the refs==0 victim scan; it is
    released with the stage. Pre-fix the victim scan below finds the source and
    the journal drifts: ssd_used 4096 instead of 8192, extents_shared 0."""
    import freetoken.scheduler.session_tier as st_mod

    d = tmp_path / "tier"
    store = SessionTierStore(_cfg(ram=16384, ssd=1 << 20, d=str(d)))
    keys, pages = _chain([(7, 0), (7, 1), (7, 2), (7, 3)])
    assert store.offer(b"p1", 2, pages[:2], _snap("a"))
    assert store.offer(b"p2", 4, pages, _snap("b"))
    seg1 = store._by_path(b"p1")
    seg2 = store._by_path(b"p2")
    assert store._demote_one_lru()             # deterministic clock: p1 goes to L2
    assert seg1.in_l2 and seg1.refs == 0 and seg2.in_l1
    # build the flush stage for the dependent exactly as the builder thread does
    pending = st_mod._PendingVolume()
    stage = store._build_group_stage([seg2], pending, 0, {})
    assert stage is not None and stage[0].shares
    # the builder's lock-free victim scan, exactly as _evict_oldest_l2 runs it
    for v in [s for s in store._segments.values() if s.in_l2 and s.refs == 0]:
        store._discard(v)
    assert store._write_group(stage, pending, 1, set()) == 1
    assert seg1.refs == 0                      # pin released with the stage
    snap = store.snapshot()
    # (i) byte truth: p1's span (4096) + p2's private span (4096), no eviction drift
    assert snap["ssd_used"] == 2 * 4096
    # (ii) the share event is counted
    assert snap["extents_shared"] == 1
    # (iii) every consumed extent carries a verifiable crc - no unverified extents
    recs, whole = _journal_records_raw(os.path.join(str(d), "journal.log"))
    assert whole and len(recs) == 2            # no tombstone: the source survived
    assert all(e[2] is not None for e in recs[1]["extents"])
    # both boundaries still serve byte-exact
    hit = store.probe(keys[:2])
    assert store.restore(hit[1]) == ([data for _, data in pages[:2]], [_snap("a")])
    hit = store.probe(keys)
    assert store.restore(hit[1]) == ([data for _, data in pages], [_snap("b")])


def test_extent_flush_journal_failure_blocks_dependents(tmp_path, monkeypatch):
    """F2 regression: a journal-append failure for a share source must land the
    source in failed_sids, or the dependent's record is journaled referencing a
    span no journal record owns (boot-safe - replay drops it - but the in-memory
    unique-byte accounting misses the span until replay/compact, contradicting the
    dep-gate contract). The dependent is skipped and both segments keep their L1
    residency for the retried flush."""
    d = str(tmp_path / "tier")
    store = SessionTierStore(_cfg(ram=16384, ssd=1 << 20, d=d))
    keys, pages = _chain([(7, 0), (7, 1), (7, 2)])
    assert store.offer(b"p1", 1, pages[:1], _snap("a"))
    assert store.offer(b"p2", 2, pages[:2], _snap("b"))
    seg1 = store._by_path(b"p1")

    armed = {"on": True}

    def fail_source_journal(seg, crc, crc_alg=None):
        if armed["on"] and seg is seg1:
            armed["on"] = False
            return False
        return True

    monkeypatch.setattr(store, "_journal_append", fail_source_journal)
    assert store.flush_live() == 0             # p1's record failed; p2 dep-skipped
    assert all(s.in_l1 for s in store._segments.values())
    assert store._ssd_used == 0
    recs, _ = _journal_records_raw(os.path.join(d, "journal.log"))
    assert recs == []
    monkeypatch.undo()
    assert store.flush_live() == 2             # clean retry: both land, p2 shares
    assert store._ssd_used == 2 * 4096         # unique bytes only
    boot = SessionTierStore(_cfg(ram=16384, ssd=1 << 20, d=d))
    assert boot.replay_journal() == 2
    for kw, pgs, s in ((keys[:1], pages[:1], _snap("a")),
                       (keys[:2], pages[:2], _snap("b"))):
        hit = boot.probe(kw)
        assert hit is not None and hit[0] == len(kw)
        assert boot.restore(hit[1]) == ([data for _, data in pgs], [s])


def test_extent_crc_mismatch_drops_all_consumers_only(tmp_path):
    """G1: one corrupted shared span drops EVERY record consuming it (per-extent
    validation is cached per distinct span) while records on untouched spans
    survive - the drop is scoped by extent consumption, not by record."""
    d = tmp_path / "tier"
    store, segs = _extent_chain_store(d, [(1, 2, "a"), (2, 4, "b")])
    ik, ip = _chain([(9, 0), (9, 1)])
    assert store.offer(b"q", 2, ip, _snap("q"))   # independent chain, own extents
    recs, _ = _journal_records_raw(os.path.join(str(d), "journal.log"))
    shared = recs[0]["extents"][0]             # b1's page span, consumed by b2 too
    with open(d / "blob.bin", "r+b") as f:
        f.seek(shared[0])
        f.write(b"\\xde\\xad\\xbe\\xef")          # corrupt the shared bytes
    boot = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d)))
    assert boot.replay_journal() == 1          # both b1 and b2 dropped, q survives
    assert boot.probe(segs[0][1]) is None
    assert boot.probe(segs[1][1]) is None
    hit = boot.probe(ik)
    assert hit is not None and hit[0] == len(ik)
    assert boot.restore(hit[1]) == ([data for _, data in ip], [_snap("q")])
    assert (shared[0], shared[1]) not in boot._extents


def test_extent_beyond_blob_eof_dropped_no_crash(tmp_path):
    """G2: a truncated blob makes a consumed extent's pread come up short - treated
    exactly like a crc mismatch (drop, no crash); extents entirely below the cut
    still verify and their records serve byte-exact."""
    d = tmp_path / "tier"
    store, segs = _extent_chain_store(d, [(1, 2, "a"), (2, 4, "b")])
    recs, _ = _journal_records_raw(os.path.join(str(d), "journal.log"))
    # cut inside b2's LAST extent (its snapshot): b1's bytes all lie below the cut
    last = recs[1]["extents"][-1]
    os.truncate(d / "blob.bin", last[0] + last[1] // 2)
    boot = SessionTierStore(_cfg(ram=0, ssd=1 << 20, d=str(d)))
    assert boot.replay_journal() == 1
    hit = boot.probe(segs[0][1])
    assert boot.restore(hit[1]) == ([data for _, data in segs[0][2]], [segs[0][3]])
    assert boot._by_path(segs[1][0]) is None    # b2's record dropped (short pread)


def test_extent_cap_overflow_falls_back_to_private_copy(tmp_path, monkeypatch):
    """G3: a tiny _MAX_RECORD_EXTENTS makes every share walk exceed the cap - the
    planners return empty and the record lands as an honest FULL private copy (no
    truncation, no crash, nothing shared)."""
    import freetoken.scheduler.session_tier as st_mod

    monkeypatch.setattr(st_mod, "_MAX_RECORD_EXTENTS", 1)
    d = tmp_path / "tier"
    store, segs = _extent_chain_store(d, [(1, 2, "a"), (2, 4, "b")])
    snap = store.snapshot()
    assert snap["extents_shared"] == 0
    assert snap["ssd_used"] == 2 * 4096        # both spans privately padded in full
    recs, _ = _journal_records_raw(os.path.join(str(d), "journal.log"))
    assert recs[1]["extents"][0][:2] != recs[0]["extents"][0][:2]   # no reference
    for seg in segs:                           # full payloads restore byte-exact
        hit = store.probe(seg[1])
        assert store.restore(hit[1]) == ([data for _, data in seg[2]], [seg[3]])
