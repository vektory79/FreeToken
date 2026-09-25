"""P2b integration: CacheManager hybrid path (match_req -> cache_req donate -> prefix hit).
CPU, real LinearStatePool + page_table, hand-built Reqs. Exercises the two-currency wiring
without the full scheduler/engine."""
from __future__ import annotations

import contextlib
import itertools
import time
from types import SimpleNamespace

import pytest
import torch

from freetoken.core import Req, SamplingParams
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.scheduler.cache import CacheManager


@contextlib.contextmanager
def _deterministic_clock():
    """Port of tests/kvcache/radix/driver.deterministic_clock (B7): the trees stamp nodes with
    one time.monotonic_ns() read per walk, so a real clock's coarse resolution adds ties
    BETWEEN walks and makes stamp-ordering asserts flaky."""
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


def _pool(num_slots=16):
    g = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate="silu",
    )
    return LinearStatePool(group=g, num_slots=num_slots, dtype=torch.bfloat16,
                           device=torch.device("cpu"), tp_size=1)


def _pend(ids):
    # int32 to match production Req.input_ids dtype (fast_compare_key needs consistent dtype)
    t = torch.tensor(ids, dtype=torch.int32)
    return SimpleNamespace(input_ids=t, input_len=len(ids))


def test_hybrid_cache_manager_donate_then_hit():
    pool = _pool()
    page_table = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, page_table, "hybrid_radix", linear_state_pool=pool)
    assert cm.is_hybrid

    # cold match on an empty tree
    mr = cm.match_req(_pend([1, 2, 3, 4, 5]))
    assert mr.cuda_handle.cached_len == 0 and mr.mamba_value is None

    # admit req A: allocate live + ping-pong, stage KV pages, mark a ×N snapshot at boundary 4
    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    page_table[0, :4] = torch.tensor([100, 101, 102, 103], dtype=torch.int32)
    reqA = Req(input_ids=torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32), table_idx=0,
               cached_len=4, output_len=1, uid=0, sampling_params=SamplingParams(),
               cache_handle=mr.cuda_handle)
    reqA.linear_slot_idx, reqA.mamba_ping_pong = live, pp
    reqA.mamba_next_track_idx = 1            # flipped from 0 in build_fla_metadata; frozen = pp[0]
    reqA.mamba_last_track_seqlen = 4
    cm.lock(mr.cuda_handle)

    free_before = pool.num_free_slots
    cm.cache_req(reqA, finished=False)       # donate pp[0] at boundary 4; replace it in the pair
    # pp[0] donated to the tree; a fresh replacement was alloc'd -> net free-slot count unchanged
    assert pool.num_free_slots == free_before - 1  # one replacement alloc'd (donated slot now tree-owned)
    assert reqA.mamba_ping_pong[0] != pp[0]        # slot 0 replaced; pp[0] now lives in the tree

    # req B shares the [1,2,3,4] prefix -> HIT: returns the donated snapshot + reused KV
    mrB = cm.match_req(_pend([1, 2, 3, 4, 9]))
    assert mrB.cuda_handle.cached_len == 4
    assert mrB.mamba_value == pp[0]
    assert mrB.cuda_handle.get_matched_indices().tolist() == [100, 101, 102, 103]


def test_hybrid_finish_frozen_donate_page_aligned_l_at_page_size_gt_1():
    """page_size>1 sibling of donate_then_hit: the finish-frozen donate inserts the frozen
    snapshot at a page-aligned L while the unaligned cached_len skips the live donate, so a
    repeat request must still reuse the prefix and its snapshot."""
    pool = _pool()
    ps = 8
    pt = torch.zeros(4, 128, dtype=torch.int32)
    cm = CacheManager(16, ps, pt, "hybrid_radix", linear_state_pool=pool)

    mr = cm.match_req(_pend(list(range(1, 13))))   # 12 tokens: one full page + 4
    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    pt[0, 8:12] = ps                               # tokens 8..11 live on the second page
    req = Req(input_ids=torch.arange(1, 14, dtype=torch.int32), table_idx=0,
              cached_len=12, output_len=1, uid=0, sampling_params=SamplingParams(),
              cache_handle=mr.cuda_handle)
    req.linear_slot_idx, req.mamba_ping_pong = live, pp
    req.mamba_next_track_idx = 1                   # frozen = pp[0]
    req.mamba_last_track_seqlen = ps               # page-aligned frozen boundary
    cm.lock(mr.cuda_handle)

    free_before = pool.num_free_slots
    cm.cache_req(req, finished=True)
    assert pool.num_free_slots == free_before + 2  # pp[1] and the live slot freed; tree owns pp[0]
    assert req.mamba_ping_pong is None             # the frozen branch consumed the pair

    mr2 = cm.match_req(_pend(list(range(1, 9)) + [99]))
    assert mr2.cuda_handle.cached_len == ps
    assert mr2.mamba_value == pp[0]


def test_hybrid_finish_frozen_donate_skipped_for_unaligned_l():
    """page_size>1: with an unaligned tracked L and an unaligned cached_len the finish commit
    donates nothing (both inserts would attach an over-advanced state to a shorter node):
    every slot returns to the pool and the tree stays empty."""
    pool = _pool()
    ps = 8
    pt = torch.zeros(4, 128, dtype=torch.int32)
    cm = CacheManager(16, ps, pt, "hybrid_radix", linear_state_pool=pool)

    mr = cm.match_req(_pend(list(range(1, 13))))   # 12 tokens: one full page + 4
    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    pt[0, 8:12] = ps                               # tokens 8..11 live on the second page
    req = Req(input_ids=torch.arange(1, 14, dtype=torch.int32), table_idx=0,
              cached_len=12, output_len=1, uid=0, sampling_params=SamplingParams(),
              cache_handle=mr.cuda_handle)
    req.linear_slot_idx, req.mamba_ping_pong = live, pp
    req.mamba_next_track_idx = 1                   # frozen = pp[0]
    req.mamba_last_track_seqlen = 7                # align_down(12, 8) = 8 != 12 and != 7
    cm.lock(mr.cuda_handle)

    free_before = pool.num_free_slots
    cm.cache_req(req, finished=True)
    assert pool.num_free_slots == free_before + 3  # live + both ping-pong slots returned
    assert pool.num_free_slots == pool.num_slots - 1   # nothing tree-owned

    mr2 = cm.match_req(_pend(list(range(1, 13)) + [99]))
    assert mr2.cuda_handle.cached_len == 0
    assert mr2.mamba_value is None


def test_chunk_commit_skips_unaligned_l_at_page_size_gt_1():
    """page_size>1 sibling of the unaligned-L finish skip, for the finished=False chunk
    commit (cache.py:386-391): a misaligned tracked L is consumed with NO donation -- the
    req keeps its KV pages and GDN slots and the tree stays empty until the next aligned
    boundary (or the finish-donate) commits instead."""
    pool = _pool()
    ps = 8
    pt = torch.zeros(4, 128, dtype=torch.int32)
    cm = CacheManager(16, ps, pt, "hybrid_radix", linear_state_pool=pool)

    mr = cm.match_req(_pend(list(range(1, 13))))   # 12 tokens: one full page + 4
    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    pt[0, 8:12] = ps                               # tokens 8..11 live on the second page
    req = Req(input_ids=torch.arange(1, 14, dtype=torch.int32), table_idx=0,
              cached_len=12, output_len=1, uid=0, sampling_params=SamplingParams(),
              cache_handle=mr.cuda_handle)
    req.linear_slot_idx, req.mamba_ping_pong = live, pp
    req.mamba_next_track_idx = 1                   # frozen = pp[0]
    req.mamba_last_track_seqlen = 7                # align_down(7, 8) = 0 != 7
    cm.lock(mr.cuda_handle)

    free_before = pool.num_free_slots
    free_pages_before = len(cm.free_slots)
    cm.cache_req(req, finished=False)              # chunk-commit path: misaligned-L skip
    assert req.mamba_last_track_seqlen is None     # the mark is consumed even on the skip
    assert pool.num_free_slots == free_before      # no donate, no replacement alloc
    assert pool.num_free_slots == pool.num_slots - 1 - 3  # nothing tree-owned: req holds live+pair
    assert len(cm.free_slots) == free_pages_before  # req keeps its KV pages: nothing freed

    mr2 = cm.match_req(_pend(list(range(1, 13)) + [99]))
    assert mr2.cuda_handle.cached_len == 0
    assert mr2.mamba_value is None
    cm.prefix_cache.check_integrity()


def test_hybrid_finish_donates_live_slot():
    pool = _pool()
    page_table = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, page_table, "hybrid_radix", linear_state_pool=pool)

    mr = cm.match_req(_pend([7, 8, 9, 10]))
    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    page_table[1, :3] = torch.tensor([200, 201, 202], dtype=torch.int32)
    req = Req(input_ids=torch.tensor([7, 8, 9, 10], dtype=torch.int32), table_idx=1,
              cached_len=3, output_len=1, uid=1, sampling_params=SamplingParams(),
              cache_handle=mr.cuda_handle)
    req.linear_slot_idx, req.mamba_ping_pong = live, pp
    cm.lock(mr.cuda_handle)

    cm.cache_req(req, finished=True)         # donate the live slot directly (final state)
    # ping-pong pair freed; live slot kept (now owned by the tree)
    mr2 = cm.match_req(_pend([7, 8, 9, 10]))
    assert mr2.cuda_handle.cached_len == 3 and mr2.mamba_value == live


def test_chunk_commit_clears_mamba_last_track_seqlen_before_decode():
    """The commit at the prefill/decode boundary must clear L: decode never tracks, and
    snapshot_toolcall_anchor only freezes an anchor while L is None (consume-if-None)."""
    pool = _pool()
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, pt, "hybrid_radix", linear_state_pool=pool)

    mr = cm.match_req(_pend([1, 2, 3, 4, 5]))
    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    pt[0, :4] = torch.tensor([100, 101, 102, 103], dtype=torch.int32)
    req = Req(input_ids=torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32), table_idx=0,
              cached_len=4, output_len=1, uid=0, sampling_params=SamplingParams(),
              cache_handle=mr.cuda_handle)
    req.linear_slot_idx, req.mamba_ping_pong = live, pp
    req.mamba_next_track_idx = 1
    req.mamba_last_track_seqlen = 4
    cm.lock(mr.cuda_handle)

    cm.cache_req(req, finished=False)
    assert req.mamba_last_track_seqlen is None


def test_free_req_slots_idempotent():
    """C2: a finish/abort double-free of the same request must NOT push its GDN slots twice."""
    pool = _pool()
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, pt, "hybrid_radix", linear_state_pool=pool)
    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    req = Req(input_ids=torch.tensor([1, 2, 3], dtype=torch.int32), table_idx=0, cached_len=2,
              output_len=1, uid=0, sampling_params=SamplingParams(), cache_handle=None)
    req.linear_slot_idx, req.mamba_ping_pong = live, pp
    base = pool.num_free_slots
    cm._free_req_slots(req)
    assert pool.num_free_slots == base + 3        # live + 2 ping-pong returned once
    cm._free_req_slots(req)                        # second free (abort/finish race)
    assert pool.num_free_slots == base + 3         # idempotent: nothing pushed twice


def test_rebuild_reclaims_donated_gdn_slots():
    """C5: a runtime cache rebuild must return the discarded tree's GDN snapshot slots (idle)."""
    pool = _pool(num_slots=16)
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, pt, "hybrid_radix", linear_state_pool=pool)
    mr = cm.match_req(_pend([7, 8, 9, 10]))
    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    pt[1, :3] = torch.tensor([200, 201, 202], dtype=torch.int32)
    req = Req(input_ids=torch.tensor([7, 8, 9, 10], dtype=torch.int32), table_idx=1, cached_len=3,
              output_len=1, uid=1, sampling_params=SamplingParams(), cache_handle=mr.cuda_handle)
    req.linear_slot_idx, req.mamba_ping_pong = live, pp
    cm.lock(mr.cuda_handle)
    cm.cache_req(req, finished=True)              # donates `live` to the tree, frees ping-pong
    assert pool.num_free_slots < pool.num_slots - 1   # a slot is now tree-owned
    cm.rebuild(64, pt)                            # idle rebuild discards the tree
    assert pool.num_free_slots == pool.num_slots - 1  # all GDN slots reclaimed (no leak)


def test_prefill_chunk_ends_on_a_page_boundary():
    """A hybrid chunk must end page-aligned: the snapshot commit skips any other boundary."""
    from freetoken.scheduler.prefill import ChunkedReq, PrefillAdder
    from freetoken.scheduler.table import TableManager
    from freetoken.scheduler.utils import PendingReq

    pool = _pool()
    pt = torch.zeros(4, 512, dtype=torch.int32)
    cm = CacheManager(64, 64, pt, "hybrid_radix", linear_state_pool=pool)
    assert cm.prefill_chunk_align == 64
    tm = TableManager(max_running_reqs=4, page_table=pt)
    pending = PendingReq(0, torch.arange(300, dtype=torch.int32), SamplingParams(max_tokens=1))

    adder = PrefillAdder(token_budget=100, reserved_size=0, cache_manager=cm, table_manager=tm)
    req = adder.try_add_one(pending)
    assert isinstance(req, ChunkedReq) and req.extend_len == 64

    # a budget below one page keeps the unaligned chunk rather than stalling the request
    adder = PrefillAdder(token_budget=40, reserved_size=0, cache_manager=cm, table_manager=tm)
    assert adder.try_add_one(pending).extend_len == 40


def _chunked_setup(num_slots=16, prompt_len=300, page_size=64, num_pages=32, uid=0):
    """Hybrid manager + table + one pending prompt, sized for chunked-prefill flows."""
    from freetoken.scheduler.table import TableManager
    from freetoken.scheduler.utils import PendingReq

    pool = _pool(num_slots)
    pt = torch.zeros(8, num_pages * page_size, dtype=torch.int32)
    cm = CacheManager(num_pages, page_size, pt, "hybrid_radix", linear_state_pool=pool)
    tm = TableManager(max_running_reqs=8, page_table=pt)
    pending = PendingReq(uid, torch.arange(prompt_len, dtype=torch.int32),
                         SamplingParams(max_tokens=1))
    return cm, pool, pt, tm, pending


def _forward_chunk(adder, pending, track_len, next_track_idx):
    """One chunked-prefill step in production order: create the chunk (the hybrid per-chunk
    donation of the PRIOR chunk fires inside try_add_one), make its pages real, stamp the
    xCHUNK track state _build_track_metadata leaves behind (boundary + flip, only when the
    extend crosses a boundary), advance cached_len (engine complete_one). Returns
    (req, inherited_track_len) -- the mark the chunk inherited before its own forward,
    i.e. what the donation consumed."""
    from freetoken.scheduler.prefill import ChunkedReq

    req = adder.try_add_one(pending)
    inherited_track = req.mamba_last_track_seqlen
    adder.cache_manager.allocate_paged([req])
    if track_len is not None:
        req.mamba_last_track_seqlen = track_len
        req.mamba_next_track_idx = next_track_idx
    req.complete_one()
    pending.chunked_req = req if isinstance(req, ChunkedReq) else None
    return req, inherited_track


def _snapshot_chain(cm):
    """The tree's boundary nodes along the single inserted chain, root-first."""
    chain, node = [], cm.prefix_cache.root
    while node.children:
        assert len(node.children) == 1
        node = next(iter(node.children.values()))
        chain.append(node)
    return chain


def test_prefill_continuation_donates_prior_boundary_before_inheriting():
    """AMENDMENT 2: creating a continuation commits the PRIOR chunk's tracked x64 boundary
    BEFORE inheriting its state (hybrid only): the continuation sees the post-commit handle
    and ping-pong tuple and a consumed track mark. Committing at the drain instead would
    double-free: the continuation has already copied (overlap) the pre-commit state."""
    from freetoken.scheduler.prefill import ChunkedReq, PrefillAdder
    from freetoken.scheduler.table import TableManager
    from freetoken.scheduler.utils import PendingReq

    pool = _pool()
    pt = torch.zeros(4, 512, dtype=torch.int32)
    cm = CacheManager(64, 64, pt, "hybrid_radix", linear_state_pool=pool)
    tm = TableManager(max_running_reqs=4, page_table=pt)
    pending = PendingReq(0, torch.arange(300, dtype=torch.int32), SamplingParams(max_tokens=1))

    adder = PrefillAdder(token_budget=100, reserved_size=0, cache_manager=cm, table_manager=tm)
    chunk1 = adder.try_add_one(pending)
    assert isinstance(chunk1, ChunkedReq) and chunk1.extend_len == 64
    assert chunk1.mamba_last_track_seqlen is None  # fresh admit starts untracked
    chunk1.mamba_last_track_seqlen = 64            # the x64 track fired during chunk 1's forward
    chunk1.mamba_next_track_idx = 1                # flipped from 0 in build_fla_metadata; frozen = pp[0]
    cm.allocate_paged([chunk1])                    # mirror _launch_req: make the chunk's pages real
    chunk1.complete_one()                          # cached_len 0 -> 64
    pp0 = chunk1.mamba_ping_pong[0]
    pending.chunked_req = chunk1

    # B3-2b: a budget bounce (image-whole path) AFTER the donation, then retry: the second
    # pass's donation is a no-op (the mark was consumed) and the inherited state is identical.
    pending.mm_items = [SimpleNamespace(offsets=[(100, 200)])]
    bounced = PrefillAdder(token_budget=100, pass_budget=300, reserved_size=0,
                           cache_manager=cm, table_manager=tm, keep_images_whole=True)
    assert bounced.try_add_one(pending) is None    # bounced in _add_one_req, donation already ran
    bounce_state = (pool.num_free_slots, chunk1.cache_handle.cached_len,
                    chunk1.mamba_last_track_seqlen, chunk1.mamba_ping_pong)
    pending.mm_items = None

    adder = PrefillAdder(token_budget=100, reserved_size=0, cache_manager=cm, table_manager=tm)
    req2 = adder.try_add_one(pending)
    assert req2 is not None
    assert (pool.num_free_slots, chunk1.cache_handle.cached_len,
            chunk1.mamba_last_track_seqlen, chunk1.mamba_ping_pong) == bounce_state
    # the donation consumed the mark and rebound the chain BEFORE the inheritance
    assert req2.mamba_last_track_seqlen is None
    assert req2.cached_len == 64
    assert req2.cache_handle.cached_len == 64      # rebound to the donated boundary node
    assert req2.mamba_ping_pong[0] != pp0          # donated slot replaced before inheriting

    # the boundary is reusable by any request sharing the prefix
    mr = cm.match_req(_pend(list(range(64)) + [99]))
    assert mr.cuda_handle.cached_len == 64
    assert mr.mamba_value == pp0

    # chain integration: the final chunk still commits its own boundary at the drain, and the
    # aligned finish donates the live slot at the next boundary -- no double donation
    cm.allocate_paged([req2])
    req2.complete_one()                            # cached_len 64 -> 128
    live_id = req2.linear_slot_idx
    free_before = pool.num_free_slots
    cm.cache_req(req2, finished=True)
    assert pool.num_free_slots == free_before + 2  # the pair freed; live donated at 128
    mr2 = cm.match_req(_pend(list(range(128)) + [99]))
    assert mr2.cuda_handle.cached_len == 128
    assert mr2.mamba_value == live_id
    cm.prefix_cache.check_integrity()


def test_chunked_req_donates_each_intermediate_boundary():
    """AMENDMENT 2 T-A: a 3-chunk ChunkedReq flow must attach each intermediate chunk's
    tracked x64 snapshot to the tree at its page-aligned boundary as the NEXT chunk is
    created. Pre-fix the ChunkedReq drain skip left ZERO intermediate snapshots -- only the
    final chunk's own boundary ever landed (fails-before)."""
    from freetoken.scheduler.prefill import PrefillAdder

    cm, pool, pt, tm, pending = _chunked_setup(prompt_len=300)
    adder = PrefillAdder(token_budget=128, reserved_size=0, cache_manager=cm, table_manager=tm)
    chunk1, _ = _forward_chunk(adder, pending, 64, 1)       # chunk [0,128): tracks 64, flip -> 1
    pp0 = chunk1.mamba_ping_pong[0]

    adder = PrefillAdder(token_budget=128, reserved_size=0, cache_manager=cm, table_manager=tm)
    chunk2, inherited = _forward_chunk(adder, pending, 192, 0)   # chunk [128,256): tracks 192
    # creating chunk 2 donated chunk 1's boundary 64 before inheriting its state
    a = next(iter(cm.prefix_cache.root.children.values()))
    assert a.length == 64 and a.mamba_value == pp0          # page-aligned end 64, snapshot attached
    assert inherited is None                                # chunk 1's mark was consumed by the donation
    assert chunk2.cache_handle.cached_len == 64             # rebound to the donated node
    assert chunk2.mamba_ping_pong[0] != pp0                 # donated slot replaced before inheriting
    pp1 = chunk2.mamba_ping_pong[1]

    adder = PrefillAdder(token_budget=128, reserved_size=0, cache_manager=cm, table_manager=tm)
    chunk3, inherited3 = _forward_chunk(adder, pending, None, 0)   # final chunk [256,300): no x64 track
    b = next(iter(a.children.values()))
    assert a.length + b.length == 192 and b.mamba_value == pp1   # boundary 192, page-aligned end
    assert b.mamba_ref_count == 1 and a.mamba_ref_count == 0     # newest locked, prior evictable
    assert inherited3 is None                               # chunk 2's mark consumed too
    assert chunk3.cache_handle.cached_len == 192
    assert chunk3.mamba_ping_pong == chunk2.mamba_ping_pong      # inherited the post-donation pair
    # pool: 15 allocatable = 3 req slots + 2 tree snapshots + 10 free
    assert pool.num_free_slots == 10
    assert cm.prefix_cache.mamba_evictable_size == 1 and cm.prefix_cache.mamba_protected == 1
    cm.prefix_cache.check_integrity()


def test_midhistory_divergence_matches_deepest_live_boundary():
    """AMENDMENT 2 T-B: after a 3-chunk prefill donates boundaries 64 and 192, a repeat of
    the full prefix reuses to 192, and a request sharing only the first ~1.5 chunks
    (divergence at 100) matches the deepest LIVE boundary below the divergence: cached_len
    == 64. Pre-fix: 0 -- a full re-prefill (fails-before)."""
    from freetoken.scheduler.prefill import PrefillAdder

    cm, pool, pt, tm, pending = _chunked_setup(prompt_len=300)
    for k, track in enumerate((64, 192, None)):
        adder = PrefillAdder(token_budget=128, reserved_size=0, cache_manager=cm,
                             table_manager=tm)
        nxt = None if track is None else (1 if k == 0 else 0)
        chunk, _ = _forward_chunk(adder, pending, track, nxt)
    chain = _snapshot_chain(cm)
    pp0, pp1 = chain[0].mamba_value, chain[1].mamba_value
    assert (chain[0].length, chain[0].length + chain[1].length) == (64, 192)
    assert pp0 is not None and pp1 is not None

    # full-prefix repeat: reuse up to the deepest committed boundary
    mr = cm.match_req(_pend(list(range(300))))
    assert mr.cuda_handle.cached_len == 192 and mr.mamba_value == pp1
    # mid-history rewrite: the client replays [0,100) then diverges
    ids2 = list(range(100)) + list(range(9000, 9008))
    mr2 = cm.match_req(_pend(ids2))
    assert mr2.cuda_handle.cached_len == 64                 # deepest live boundary <= 100
    assert mr2.mamba_value == pp0


def test_per_chunk_donation_then_finish_keeps_pool_integrity():
    """AMENDMENT 2 T-C (double-free regression). Reconstructed hazard: at the DRAIN the
    continuation has already copied (overlap) the pre-commit handle and ping-pong tuple, so
    a drain-point cache_req (a) leaves the donated snapshot slot id in the continuation's
    stale tuple, which _free_req_slots frees again at finish while the tree owns it, and
    (b) frees [stale_handle.cached_len, prefix_len) -- the pages the prior commit already
    adopted into the tree -- a second time. The fix commits at continuation creation, so
    the donate-then-finish flow must keep both pools exactly conserved."""
    from freetoken.scheduler.prefill import PrefillAdder

    cm, pool, pt, tm, pending = _chunked_setup(prompt_len=300)
    chunks = []
    for k, track in enumerate((64, 192, None)):
        adder = PrefillAdder(token_budget=128, reserved_size=0, cache_manager=cm,
                             table_manager=tm)
        nxt = None if track is None else (1 if k == 0 else 0)
        chunk, _ = _forward_chunk(adder, pending, track, nxt)
        chunks.append(chunk)
    chunk1 = chunks[0]

    # B3-2a: re-committing an already-committed chunk (its L was consumed -> None) must be
    # a no-op: the early return fires before any tree/pool/refcount mutation.
    tree_before = [(n.length, n.mamba_value, n.mamba_ref_count, n.ref_count)
                   for n in _snapshot_chain(cm)]
    free_before_recommit = pool.num_free_slots
    cm.cache_req(chunk1, finished=False)
    assert pool.num_free_slots == free_before_recommit
    assert [(n.length, n.mamba_value, n.mamba_ref_count, n.ref_count)
            for n in _snapshot_chain(cm)] == tree_before

    free_before_finish = pool.num_free_slots
    cm.cache_req(chunk, finished=True)     # finish: unaligned live-donate (300) skipped
    # exactly live + both ping-pong slots returned once; the tree keeps its 2 donated slots
    assert pool.num_free_slots == free_before_finish + 3
    tree = cm.prefix_cache
    assert tree.mamba_evictable_size + tree.mamba_protected == 2
    assert chunk.mamba_ping_pong is None and chunk.linear_slot_idx is None
    cm.check_integrity()                   # page + GDN-slot conservation, tree structure
    cm._free_req_slots(chunk)              # abort/finish race: second free must be a no-op
    assert pool.num_free_slots == free_before_finish + 3


def test_donation_fund_pressure_evicts_stalest_validated_boundary():
    """AMENDMENT 2 T-D: a 9-slot pool (8 allocatable: live+2pp per req + 1 padding sink)
    holds only 4 unlocked boundary donations; a 10-chunk prompt must fire evict_mamba
    mid-chain and kill the STALEST-VALIDATED boundary first (FIFO by snapshot_lru -- the
    amended Fix-3 ordering becomes load-bearing), never the just-locked tip."""
    from freetoken.scheduler.prefill import PrefillAdder

    cm, pool, pt, tm, pending = _chunked_setup(num_slots=9, prompt_len=1280)
    next_idx = 0
    chunk = None
    for k in range(10):                     # chunks of 128; boundary k = k*128 + 64
        adder = PrefillAdder(token_budget=128, reserved_size=0, cache_manager=cm,
                             table_manager=tm)
        chunk, _ = _forward_chunk(adder, pending, k * 128 + 64, 1 - next_idx)
        next_idx = 1 - next_idx
    cm.cache_req(chunk, finished=False)     # the final chunk's drain commit (boundary 1216)
    chain = _snapshot_chain(cm)
    assert len(chain) == 10 and [c.length for c in chain] == [64] + [128] * 9
    dead, live, tip = chain[:5], chain[5:9], chain[9]
    assert all(c.mamba_value is None for c in dead)      # FIFO by last validation
    assert all(c.mamba_value is not None for c in live)
    assert tip.mamba_value is not None and tip.mamba_ref_count == 1   # just-locked tip survives
    assert pool.num_free_slots == 0                      # steady state: 3 req + 5 tree = 8
    cm.prefix_cache.check_integrity()
    # the aligned finish (1280) donates the live state at boundary 1280; full conservation
    cm.cache_req(chunk, finished=True)
    assert pool.num_free_slots == 2
    assert cm.prefix_cache.mamba_evictable_size == 6 and cm.prefix_cache.mamba_protected == 0
    cm.check_integrity()


def test_finish_does_not_double_donate_final_boundary():
    """AMENDMENT 2 T-E: the final chunk's own boundary is donated ONCE (at its drain
    commit); the finish's pending-frozen branch must not re-attach at any position (the
    commit consumed the mark) and the unaligned live donate is skipped, so the tree keeps
    exactly the two donated boundaries with their original slots."""
    from freetoken.scheduler.prefill import PrefillAdder

    cm, pool, pt, tm, pending = _chunked_setup(prompt_len=200)
    adder = PrefillAdder(token_budget=128, reserved_size=0, cache_manager=cm, table_manager=tm)
    chunk1, _ = _forward_chunk(adder, pending, 64, 1)       # chunk [0,128): tracks 64
    pp0 = chunk1.mamba_ping_pong[0]
    adder = PrefillAdder(token_budget=128, reserved_size=0, cache_manager=cm, table_manager=tm)
    chunk2, _ = _forward_chunk(adder, pending, 192, 0)      # final chunk [128,200): tracks 192
    pp1 = chunk2.mamba_ping_pong[1]
    cm.cache_req(chunk2, finished=False)                    # the drain commit donates 192
    chain = _snapshot_chain(cm)
    assert [c.mamba_value for c in chain] == [pp0, pp1]
    assert chain[1].mamba_ref_count == 1                    # the drain commit locked the tip
    cm.cache_req(chunk2, finished=True)                     # L is None: no pending-frozen re-donate
    assert [c.mamba_value for c in chain] == [pp0, pp1]     # unchanged: no double donation
    assert pool.num_free_slots == 13                        # 15 - 2 tree-owned
    cm.check_integrity()


def test_finish_pending_frozen_dedup_keeps_existing_snapshot():
    """AMENDMENT 2 T-E (dedup backstop): a finish whose tracked L names an ALREADY-donated
    boundary must dedup (mamba_exist) -- the tree keeps the ORIGINAL snapshot and the
    donated slot stays req-owned (freed with the pair), never double-owned. The forced
    stale L is defensive legacy state, unreachable in the new design (the drain commit
    clears the mark; a decode-time anchor can only freeze at page-aligned positions
    above every prefill boundary) -- kept to pin the dedup path."""
    from freetoken.scheduler.prefill import PrefillAdder

    cm, pool, pt, tm, pending = _chunked_setup(prompt_len=200)
    adder = PrefillAdder(token_budget=128, reserved_size=0, cache_manager=cm, table_manager=tm)
    chunk1, _ = _forward_chunk(adder, pending, 64, 1)
    pp0 = chunk1.mamba_ping_pong[0]
    adder = PrefillAdder(token_budget=128, reserved_size=0, cache_manager=cm, table_manager=tm)
    chunk2, _ = _forward_chunk(adder, pending, 192, 0)
    pp1 = chunk2.mamba_ping_pong[1]
    cm.cache_req(chunk2, finished=False)                    # drain commit donates 192
    # force the collision: a stale L naming the already-donated boundary 64
    chunk2.mamba_last_track_seqlen = 64
    cm.cache_req(chunk2, finished=True)
    chain = _snapshot_chain(cm)
    assert [c.mamba_value for c in chain] == [pp0, pp1]     # the original snapshots survive
    assert pool.num_free_slots == 13                        # the colliding slot returned with the pair
    cm.check_integrity()


def test_aligned_final_boundary_finish_live_donate():
    """B3-3: an ALIGNED final boundary (192 = chunks 64+128). The drain commit donates the
    final chunk's tracked L=128; the finish's live-donate then fires (192 is page-aligned)
    onto a FRESH attach node -- a single chain can never dedup its own live donate, since
    every chain boundary is < cached_len -- so the live slot is DONATED and only the pair
    is freed (+2). A re-prefill of the same prompt (phase B) whose admission matches 128
    (a 191-token match cannot reach the 192 node's 64-token key) finishes with the
    live-donate landing on the EXISTING 192 node -> mamba_exist=True -> keep_live=False:
    live + pair freed (+3), the tree's snapshots untouched (no double-donate/double-free)."""
    from freetoken.scheduler.prefill import ChunkedReq, PrefillAdder
    from freetoken.scheduler.utils import PendingReq

    cm, pool, pt, tm, pending = _chunked_setup(prompt_len=192)
    adder = PrefillAdder(token_budget=100, reserved_size=0, cache_manager=cm, table_manager=tm)
    chunk1, _ = _forward_chunk(adder, pending, 64, 1)       # chunk [0,64): tracks 64
    pp0 = chunk1.mamba_ping_pong[0]
    adder = PrefillAdder(token_budget=128, reserved_size=0, cache_manager=cm, table_manager=tm)
    chunk2, _ = _forward_chunk(adder, pending, 128, 0)      # FINAL [64,192): extend 128 -> L=128
    assert not isinstance(chunk2, ChunkedReq)
    pp1 = chunk2.mamba_ping_pong[1]
    cm.cache_req(chunk2, finished=False)                    # the drain commit donates L=128
    chain = _snapshot_chain(cm)
    assert [n.length for n in chain] == [64, 64]            # spans; cumulative ends 64, 128
    assert sum(n.length for n in chain) == 128              # the drain-donated boundary L=128
    assert [n.mamba_value for n in chain] == [pp0, pp1]

    live_id = chunk2.linear_slot_idx
    free_before = pool.num_free_slots
    cm.cache_req(chunk2, finished=True)                     # the aligned live-donate fires at 192
    chain = _snapshot_chain(cm)
    assert len(chain) == 3 and chain[2].mamba_value == live_id   # the live slot DONATED at 192
    assert pool.num_free_slots == free_before + 2           # only the pair freed (keep_live=True)
    # (the live slot ref is always cleared by _free_req_slots; tree ownership is the
    # chain[2].mamba_value == live_id assert above plus the +2, not +3, pool count)
    cm.check_integrity()

    # phase B: a re-prefill of the same prompt; its aligned finish live-donate dedups
    pending_b = PendingReq(1, torch.arange(192, dtype=torch.int32), SamplingParams(max_tokens=1))
    adder = PrefillAdder(token_budget=128, reserved_size=0, cache_manager=cm, table_manager=tm)
    req_b = adder.try_add_one(pending_b)
    assert req_b is not None and not isinstance(req_b, ChunkedReq) and req_b.cached_len == 128
    cm.allocate_paged([req_b])
    req_b.complete_one()                                    # cached_len 128 -> 192
    free_before_b = pool.num_free_slots
    cm.cache_req(req_b, finished=True)                      # the live-donate dedups at 192
    assert pool.num_free_slots == free_before_b + 3         # live + pair freed; nothing donated
    assert [n.mamba_value for n in _snapshot_chain(cm)] == [pp0, pp1, live_id]   # untouched
    assert req_b.mamba_ping_pong is None and req_b.linear_slot_idx is None
    cm.check_integrity()


def test_naive_cache_does_not_align_prefill_chunks():
    """The alignment hook is hybrid-only; every other cache keeps the raw budget chunk."""
    from freetoken.scheduler.prefill import ChunkedReq, PrefillAdder
    from freetoken.scheduler.table import TableManager
    from freetoken.scheduler.utils import PendingReq

    pt = torch.zeros(4, 512, dtype=torch.int32)
    cm = CacheManager(64, 64, pt, "radix")
    assert cm.prefill_chunk_align == 1
    tm = TableManager(max_running_reqs=4, page_table=pt)
    adder = PrefillAdder(token_budget=100, reserved_size=0, cache_manager=cm, table_manager=tm)
    pending = PendingReq(0, torch.arange(300, dtype=torch.int32), SamplingParams(max_tokens=1))
    chunk1 = adder.try_add_one(pending)
    assert isinstance(chunk1, ChunkedReq) and chunk1.extend_len == 100

    # B3-1: the is_hybrid gate keeps the per-chunk donation hybrid-only: a non-hybrid
    # continuation must not cache_req at creation (the drain skip still covers it), so the
    # tree stays empty and the admission handle is not rebound until the final chunk.
    cm.allocate_paged([chunk1])
    chunk1.complete_one()                             # cached_len 0 -> 100
    pending.chunked_req = chunk1
    adder = PrefillAdder(token_budget=100, reserved_size=0, cache_manager=cm, table_manager=tm)
    chunk2 = adder.try_add_one(pending)
    assert isinstance(chunk2, ChunkedReq) and chunk2.cached_len == 100
    assert len(cm.prefix_cache.root_node.children) == 0    # no insert at continuation creation
    assert chunk2.cache_handle is chunk1.cache_handle      # the admission handle, not rebound

    # the final chunk commits at its own drain: only then does the tree fill
    cm.allocate_paged([chunk2])
    chunk2.complete_one()                             # cached_len 100 -> 200
    pending.chunked_req = chunk2
    adder = PrefillAdder(token_budget=100, reserved_size=0, cache_manager=cm, table_manager=tm)
    final = adder.try_add_one(pending)                # [200,300): the last chunk is a plain Req
    assert not isinstance(final, ChunkedReq)
    cm.allocate_paged([final])
    final.complete_one()                              # cached_len 200 -> 300
    cm.cache_req(final, finished=False)
    assert len(cm.prefix_cache.root_node.children) == 1    # the final drain commit inserted
    assert final.cache_handle is not chunk1.cache_handle   # rebound by the final commit
    assert final.cache_handle.cached_len == 256            # align_down(300, 64)


def test_pool_sizing_covers_4mr_floor():
    """C6: pool must reserve the 4-slot-per-request non-evictable floor even at a tiny ratio."""
    from types import SimpleNamespace
    from freetoken.kvcache.linear_state_pool import _linear_pool_num_slots
    for mr in (1, 8, 64):
        c = SimpleNamespace(max_running_req=mr, cache_type="hybrid_radix",
                            linear_state_cache_ratio=0.1)
        assert _linear_pool_num_slots(c) >= 4 * mr + 1, (mr, _linear_pool_num_slots(c))


def test_chunk_commit_dedup_refreshes_boundary_lru():
    """Fix-3, scheduler level: a repeat chunked-prefill that dedups at an existing boundary
    re-validates it, so the deepest shared boundary leaves the commit walk's uniform tic
    strictly behind the shallower boundary (pre-fix they tie and the LRU order is arbitrary)."""
    pool = _pool()
    pt = torch.zeros(4, 512, dtype=torch.int32)
    cm = CacheManager(64, 64, pt, "hybrid_radix", linear_state_pool=pool)

    ids129 = list(range(1, 130))            # Req needs device_len > cached_len

    def _commit(uid, table_idx, cached_len, track_len, pages_base):
        mr = cm.match_req(_pend(ids129[:cached_len]))
        live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
        pt[table_idx, :cached_len] = torch.arange(pages_base, pages_base + cached_len,
                                                  dtype=torch.int32)
        req = Req(input_ids=torch.tensor(ids129, dtype=torch.int32), table_idx=table_idx,
                  cached_len=cached_len, output_len=1, uid=uid,
                  sampling_params=SamplingParams(), cache_handle=mr.cuda_handle)
        req.linear_slot_idx, req.mamba_ping_pong = live, pp
        req.mamba_next_track_idx = 1               # frozen = pp[0]
        req.mamba_last_track_seqlen = cached_len   # the x64 boundary this chunk tracked
        cm.lock(mr.cuda_handle)
        cm.cache_req(req, finished=False)
        return mr, req

    _commit(0, 0, 64, 64, 100)                     # req 1: chunk commit at boundary 64 (B1)
    _commit(1, 1, 128, 128, 200)                   # req 2: chunk commit at boundary 128 (B2)
    _commit(2, 2, 128, 128, 300)                   # req 3: same prefix -> DEDUP at B2

    b1 = next(iter(cm.prefix_cache.root.children.values()))
    b2 = next(iter(b1.children.values()))
    assert b1.length == 64 and b2.length == 64 and b2.mamba_value is not None
    assert b2.timestamp > b1.timestamp

    # the boundary snapshots stay live (mamba-locked by the open chunk reqs) across the
    # repeated chunked-prefill requests, so a re-prefill of the same prefix still resumes
    assert b1.mamba_value is not None and b2.mamba_value is not None
    mr = cm.match_req(_pend(ids129))          # match_req strips the last token: 128 tokens
    assert mr.cuda_handle.cached_len == 128 and mr.mamba_value is not None
    cm.prefix_cache.check_integrity()


def test_pool_exhaustion_evicts_the_stale_validated_boundary():
    """Amended Fix-3, scheduler level: a 7-slot pool (6 allocatable -- the pool keeps one
    padding sink) makes ensure_mamba_slots actually exhaust at req4's chunk commit (the
    real commit order: insert -> unlock -> match -> lock -> ensure). At the ensure B64 was
    last validated at req3's dedup, the off-path decoy D1 at reqD, and B128 by req4's own
    admission match -- all three share the commit walk's tic. Post-fix the victim is the
    stalest-VALIDATED one (B64) and the just-re-validated B128 survives; pre-fix the decoy's
    stale walk-tic timestamp makes IT the victim and the ordering is lost (fails-before)."""
    pool = _pool(num_slots=7)
    pt = torch.zeros(5, 512, dtype=torch.int32)
    cm = CacheManager(64, 64, pt, "hybrid_radix", linear_state_pool=pool)
    ids193 = list(range(1, 194))            # Req needs device_len > cached_len
    idsD = list(range(1000, 1065))          # decoy branch: different first page, off-path

    def _commit(uid, table_idx, ids, cached_len, track_len, pages_base, finish):
        mr = cm.match_req(_pend(ids[:cached_len + 1]))
        live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
        pt[table_idx, :cached_len] = torch.arange(pages_base, pages_base + cached_len,
                                                  dtype=torch.int32)
        req = Req(input_ids=torch.tensor(ids[:cached_len + 1], dtype=torch.int32),
                  table_idx=table_idx, cached_len=cached_len, output_len=1, uid=uid,
                  sampling_params=SamplingParams(), cache_handle=mr.cuda_handle)
        req.linear_slot_idx, req.mamba_ping_pong = live, pp
        req.mamba_next_track_idx = 1               # frozen = pp[0]
        req.mamba_last_track_seqlen = track_len
        cm.lock(mr.cuda_handle)
        cm.cache_req(req, finished=finish)

    _commit(0, 0, ids193, 64, 64, 100, True)       # B64: live slot donated at the x64 boundary
    _commit(1, 1, ids193, 128, 128, 200, True)     # B128 born; its finish dedup validates it
    _commit(2, 2, ids193, 64, 64, 300, True)       # repeat @64: B64 dedup-validated (newest pre-r4)
    _commit(3, 3, idsD, 64, 64, 600, True)         # off-path decoy D1: validated once, never re-walked
    _commit(4, 4, ids193, 192, 192, 400, False)    # fresh B192 -> ensure_mamba_slots must evict

    b64 = next(iter(cm.prefix_cache.root.children.values()))
    b128 = next(iter(b64.children.values()))
    b192 = next(iter(b128.children.values()))
    d1 = next(c for c in cm.prefix_cache.root.children.values() if c is not b64)
    assert (b64.length, b128.length, b192.length, d1.length) == (64, 64, 64, 64)
    assert b64.mamba_value is None                 # the stalest-VALIDATED tie-class peer died
    assert b128.mamba_value is not None            # re-validated by req4's admission match
    assert d1.mamba_value is not None              # validated after B64: survives victim #1
    assert b192.mamba_value is not None            # the just-donated tip is locked, untouched
    mr = cm.match_req(_pend(ids193))               # match_req strips the last token: 192
    assert mr.cuda_handle.cached_len == 192 and mr.mamba_value == b192.mamba_value
    cm.prefix_cache.check_integrity()


# ------------------------------------------------- W2: session-tier interception + restore
class _FakeKVPool:
    """Minimal MHAKVCache-family stand-in: the `_kv_buffer` layout _KVPageBytes codes over
    (no scale buffers, no swa paging)."""

    swa_paged = False

    def __init__(self, pages=256, page_size=1):
        self._kv_buffer = torch.zeros(2, 2, pages, page_size, 2, 4, dtype=torch.bfloat16)


def _tiered_cm(pool, kvpool, pt, ram=1 << 22):
    from freetoken.scheduler.cache import SessionTierCfg

    return CacheManager(64, 1, pt, "hybrid_radix", linear_state_pool=pool, swa_pool=kvpool,
                        session_tier_cfg=SessionTierCfg(ram_bytes=ram))


def _donate_one_snapshot(cm, pool, kvpool, pt, ids=(1, 2, 3, 4, 5), boundary=4, uid=0):
    """Cold admission + one x-boundary snapshot donation on pages 100..103, with a
    recognizable KV payload in the fake pool."""
    mr = cm.match_req(_pend(list(ids)))          # cold admission tracks the session
    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    pt[uid, :boundary] = torch.tensor([100, 101, 102, 103][:boundary], dtype=torch.int32)
    original = {}
    for j, page in enumerate(range(100, 100 + boundary)):
        kvpool._kv_buffer[:, :, page] = float(page) * 1.5
        original[page] = kvpool._kv_buffer[:, :, page].contiguous().cpu() \
            .view(torch.uint8).numpy().tobytes()
    req = Req(input_ids=torch.tensor(list(ids), dtype=torch.int32), table_idx=uid,
              cached_len=boundary, output_len=1, uid=uid, sampling_params=SamplingParams(),
              cache_handle=mr.cuda_handle)
    req.linear_slot_idx, req.mamba_ping_pong = live, pp
    req.mamba_next_track_idx = 1
    req.mamba_last_track_seqlen = boundary
    cm.lock(mr.cuda_handle)
    cm.cache_req(req, finished=False)            # donate the frozen slot at the boundary
    cm.unlock(req.cache_handle)                  # request keeps running elsewhere: the
    return original, pp[0], req                  # committed snapshot is unlocked/evictable


def test_session_tier_intercepts_evict_mamba_victim_in_adaptive_set():
    """A tip victim of a tracked session is offered to the store (bytes received) and the
    core freeing is unchanged."""
    pool, kvpool = _pool(), _FakeKVPool()
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = _tiered_cm(pool, kvpool, pt)
    original, donated, req = _donate_one_snapshot(cm, pool, kvpool, pt)
    assert cm.tier_store is not None and cm._tier_on

    frees_before = len(cm.free_slots)
    cm.ensure_mamba_slots(pool.num_slots)        # pool pressure: the tip must go
    assert len(cm.free_slots) == frees_before + 4        # core freeing unchanged
    assert pool.num_free_slots == 12             # the snapshot slot came back

    chain = cm._chain_keys(torch.tensor([1, 2, 3, 4], dtype=torch.int32))
    hit = cm.tier_store.probe(chain)
    assert hit is not None and hit[0] == 4       # the demoted tip is recoverable
    pages, snaps = cm.tier_store.restore(hit[1])
    assert len(pages) == 4 and len(snaps) == 1
    for j, page in enumerate((100, 101, 102, 103)):
        assert pages[j] == original[page]        # byte-identical KV demotion
    from freetoken.scheduler.cache import _SlotSnapshot

    assert snaps[0] == _SlotSnapshot(pool, donated).payload()   # snapshot bytes received


def test_session_tier_victim_outside_adaptive_set_freed_as_today():
    """With the store ON but the session untracked (no admission match), the victim is
    freed exactly as today and the store stays empty."""
    pool, kvpool = _pool(), _FakeKVPool()
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = _tiered_cm(pool, kvpool, pt)
    _donate_one_snapshot(cm, pool, kvpool, pt)
    cm._tier_sessions.clear()                    # simulate an untracked session

    frees_before = len(cm.free_slots)
    cm.ensure_mamba_slots(pool.num_slots)
    assert len(cm.free_slots) == frees_before + 4
    assert pool.num_free_slots == 12
    assert cm.tier_store.probe(
        cm._chain_keys(torch.tensor([1, 2, 3, 4], dtype=torch.int32))) is None


def test_session_tier_off_mode_store_never_called():
    """No cfg (or a disabled cfg) -> no store at all; eviction behaves as today."""
    from freetoken.scheduler.cache import SessionTierCfg

    for cfg in (None, SessionTierCfg()):
        pool, kvpool = _pool(), _FakeKVPool()
        pt = torch.zeros(4, 64, dtype=torch.int32)
        cm = CacheManager(64, 1, pt, "hybrid_radix", linear_state_pool=pool, swa_pool=kvpool,
                          session_tier_cfg=cfg)
        assert cm.tier_store is None and not cm._tier_on
        _donate_one_snapshot(cm, pool, kvpool, pt)
        frees_before = len(cm.free_slots)
        cm.ensure_mamba_slots(pool.num_slots)
        assert len(cm.free_slots) == frees_before + 4 and pool.num_free_slots == 12


def test_session_tier_restore_feeds_normal_insert_path():
    """Cold match after the demotion: the store hit restores byte-identical KV pages + the
    snapshot slot WITHOUT any re-prefill of those tokens, consuming exactly the free pages
    and one slot the replaced re-prefill would."""
    pool, kvpool = _pool(), _FakeKVPool()
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = _tiered_cm(pool, kvpool, pt)
    original, donated, req = _donate_one_snapshot(cm, pool, kvpool, pt)
    cm.ensure_mamba_slots(pool.num_slots)        # demote the tip: the tree is empty again
    assert cm.prefix_cache.full_evictable == 0

    frees_after_evict = len(cm.free_slots)
    mr = cm.match_req(_pend([1, 2, 3, 4, 9]))    # cold tree, warm store
    assert mr.cuda_handle.cached_len == 4
    assert mr.mamba_value is not None
    assert mr.mamba_value >= 1                   # a real pool slot (never the padding sink)
    matched = mr.cuda_handle.get_matched_indices()
    # the restore takes free-list HEAD pages [0,1,2,3]: eviction appended [100..103] AFTER
    # the untouched arange head, so this assert depends on that head-order (torch.cat append).
    assert matched.tolist() == [0, 1, 2, 3]      # pages restored from the free-list head
    for j in range(4):
        assert kvpool._kv_buffer[:, :, j].contiguous().cpu() \
            .view(torch.uint8).numpy().tobytes() == original[100 + j]
    assert len(cm.free_slots) == frees_after_evict - 4   # budget: same as the re-prefill
    assert pool.num_free_slots == 12 - 1                 # + one snapshot slot

    # the normal path takes over: a commit of the restored prefix dedups against nothing new
    # and the tree re-grows through plain insert()/cache_req (exercised by the handle shape).
    assert mr.cuda_handle.node.is_root()         # no tree node yet: lock target is inert


def test_session_tier_offer_pages_units_at_page_size_gt_1():
    """B2 fails-before: at page_size>1 the offer seam must take PAGES (len(chain_keys));
    pre-fix it took the TOKEN boundary_len and SessionTierStore.offer raised ValueError."""
    from freetoken.scheduler.cache import SessionTierCfg

    pool, kvpool = _pool(), _FakeKVPool(pages=16, page_size=8)
    pt = torch.zeros(4, 128, dtype=torch.int32)
    cm = CacheManager(16, 8, pt, "hybrid_radix", linear_state_pool=pool, swa_pool=kvpool,
                      session_tier_cfg=SessionTierCfg(ram_bytes=1 << 22))
    ids = list(range(1, 14))
    mr = cm.match_req(_pend(ids))                # cold admission tracks the session (tokens)
    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    pt[0, :8] = 0                                # page 0 covers tokens 0..7
    pt[0, 8:12] = 8                              # page 1 covers tokens 8..11
    for pg in range(2):
        kvpool._kv_buffer[:, :, pg] = float(pg) * 1.5
    req = Req(input_ids=torch.tensor(ids, dtype=torch.int32), table_idx=0, cached_len=12,
              output_len=1, uid=0, sampling_params=SamplingParams(), cache_handle=mr.cuda_handle)
    req.linear_slot_idx, req.mamba_ping_pong = live, pp
    req.mamba_next_track_idx = 1
    req.mamba_last_track_seqlen = 8              # finish-frozen donate at the aligned boundary
    cm.lock(mr.cuda_handle)
    cm.cache_req(req, finished=True)
    cm.ensure_mamba_slots(pool.num_slots)        # tiered eviction: must not raise
    chain = cm._chain_keys(torch.tensor(ids[:8], dtype=torch.int32))
    hit = cm.tier_store.probe(chain)
    assert hit is not None and hit[0] == 1       # probe depth is PAGES
    pages, snaps = cm.tier_store.restore(hit[1])
    assert len(pages) == 1 and len(snaps) == 1
    mr2 = cm.match_req(_pend(list(range(1, 9)) + [99]))
    assert mr2.cuda_handle.cached_len == 8 and mr2.mamba_value is not None


def test_session_tier_adaptive_set_excludes_stale_grid_points():
    """M5 fails-before: the under-divergence slot is ONE boundary (the deepest seen);
    pre-fix every boundary <= divergence depth was offered (stale grid points)."""
    from freetoken.kvcache.hybrid_radix_cache import VictimPath
    from freetoken.kvcache.utils import chain_page_key
    from freetoken.scheduler.cache import SessionTierCfg

    pool, kvpool = _pool(), _FakeKVPool()
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, pt, "hybrid_radix", linear_state_pool=pool, swa_pool=kvpool,
                      session_tier_cfg=SessionTierCfg(ram_bytes=1 << 22))
    k = [chain_page_key(None, (1,))]
    for t in (2, 3, 4):
        k.append(chain_page_key(k[-1], (t,)))
    cm._tier_sessions[k[0]] = [4, 2, 0, None]    # tip 4 tok, divergence depth 2 tok

    def vp(depth_pages, bl_tokens):
        return VictimPath(k[depth_pages - 1], bl_tokens, tuple(k[:depth_pages]),
                          torch.arange(depth_pages, dtype=torch.int32), None)

    offered = []
    real_offer = cm.tier_store.offer

    def spy(path_key, boundary, kv_pages, snapshot_slot=None):
        offered.append(boundary)
        return real_offer(path_key, boundary, kv_pages, snapshot_slot)

    cm.tier_store.offer = spy
    cm._tier_offer([vp(1, 4)])                   # tip
    cm._tier_offer([vp(2, 2)])                   # deepest under-divergence boundary
    cm._tier_offer([vp(1, 1)])                   # stale intermediate grid point
    assert offered == [1, 2]                     # the stale point never reaches the store
    # offer-time supersede collapses the snapshot-free tip into the deeper segment
    held = {seg.path_key for seg in cm.tier_store._segments.values()}
    assert held == {k[1]}


def test_tier_offers_across_turns_collapse_nested_plain_segments(tmp_path):
    """Manager-level offer supersede: per-turn cumulative snapshot-free boundaries of one
    session collapse to the deepest store segment, a re-offer of an already contained
    boundary is skipped (offers_dedup), and the shutdown flush keeps exactly that one
    segment - no nested redundant plain segments in the blob."""
    from freetoken.kvcache.hybrid_radix_cache import VictimPath
    from freetoken.kvcache.utils import chain_page_key
    from freetoken.scheduler.cache import SessionTierCfg

    pool, kvpool = _pool(), _FakeKVPool()
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, pt, "hybrid_radix", linear_state_pool=pool, swa_pool=kvpool,
                      session_tier_cfg=SessionTierCfg(ram_bytes=1 << 22, dir=str(tmp_path)))
    k = [chain_page_key(None, (1,))]
    for t in (2, 3, 4):
        k.append(chain_page_key(k[-1], (t,)))

    def vp(depth_pages, bl_tokens):
        return VictimPath(k[depth_pages - 1], bl_tokens, tuple(k[:depth_pages]),
                          torch.arange(depth_pages, dtype=torch.int32), None)

    cm._tier_sessions[k[0]] = [4, 0, 0, None]
    cm._tier_offer([vp(1, 4)])                   # turn-1 tip
    cm._tier_sessions[k[0]] = [8, 4, 0, None]
    cm._tier_offer([vp(2, 8)])                   # turn-2 tip: supersedes turn-1's segment
    cm._tier_sessions[k[0]] = [12, 8, 0, None]
    cm._tier_offer([vp(3, 12)])                  # turn-3 tip: supersedes turn-2's
    store = cm.tier_store
    assert len(store._segments) == 1
    assert next(iter(store._segments.values())).boundary_len == 3
    cm._tier_offer([vp(2, 8)])                   # re-offer of a contained boundary
    assert store.snapshot()["offers_dedup"] == 1
    assert len(store._segments) == 1
    assert cm.shutdown_tier() == 1               # flush_live: still exactly one segment
    assert len(store._segments) == 1


def test_abandon_restore_returns_pages_and_slot():
    """B4 fails-before: a restored match refused at admission leaked its pages + slot;
    abandon_restore returns both, idempotently, without double-freeing a COW-consumed slot."""
    pool, kvpool = _pool(), _FakeKVPool()
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = _tiered_cm(pool, kvpool, pt)
    _donate_one_snapshot(cm, pool, kvpool, pt)
    cm.ensure_mamba_slots(pool.num_slots)

    mr = cm.match_req(_pend([1, 2, 3, 4, 9]))    # restored: 4 pages + 1 slot held
    assert mr.cuda_handle.cached_len == 4
    pages_held, slots_held = len(cm.free_slots), pool.num_free_slots
    cm.abandon_restore(mr.cuda_handle)           # admission refused
    assert len(cm.free_slots) == pages_held + 4 and pool.num_free_slots == slots_held + 1
    cm.abandon_restore(mr.cuda_handle)           # idempotent
    assert len(cm.free_slots) == pages_held + 4 and pool.num_free_slots == slots_held + 1

    mr2 = cm.match_req(_pend([1, 2, 3, 4, 9]))   # a second restore, this one COW-consumed
    cm.tier_release_restore_slot(mr2.mamba_value)   # what _restore_linear_states does
    pool.free(mr2.mamba_value)
    pages_mid = len(cm.free_slots)
    cm.abandon_restore(mr2.cuda_handle)          # late/edge abandon: pages only
    assert len(cm.free_slots) == pages_mid + 4
    assert pool.num_free_slots == slots_held + 1  # slot NOT double-freed


def test_session_tier_restore_fallbacks_leak_nothing():
    """m8: free-page shortage, slot shortage and a missing snapshot each fall back to the
    normal path with zero page/slot leakage."""
    pool, kvpool = _pool(), _FakeKVPool()
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = _tiered_cm(pool, kvpool, pt)
    _donate_one_snapshot(cm, pool, kvpool, pt)
    cm.ensure_mamba_slots(pool.num_slots)        # tree empty, store warm (snapshot bound=4)
    base_pages, base_slots = len(cm.free_slots), pool.num_free_slots

    saved = cm.free_slots
    cm.free_slots = saved[:2]                    # free-page shortage
    mr = cm.match_req(_pend([1, 2, 3, 4, 9]))
    assert mr.cuda_handle.cached_len == 0 and mr.mamba_value is None
    assert len(cm.free_slots) == 2 and pool.num_free_slots == base_slots
    cm.free_slots = saved

    held = pool.alloc(pool.num_free_slots)       # slot shortage
    mr = cm.match_req(_pend([1, 2, 3, 4, 9]))
    assert mr.cuda_handle.cached_len == 0 and mr.mamba_value is None
    assert len(cm.free_slots) == base_pages and pool.num_free_slots == 0
    pool.free(held)

    from freetoken.kvcache.hybrid_radix_cache import VictimPath
    from freetoken.kvcache.utils import chain_page_key

    chain4 = cm._chain_keys(torch.tensor([1, 2, 3, 4], dtype=torch.int32))
    st = cm._tier_sessions[chain4[0]]
    st[1] = 2                                    # divergence depth 2 tokens
    cm._tier_offer([VictimPath(chain4[1], 2, tuple(chain4[:2]),
                               torch.arange(2, dtype=torch.int32), None)])  # KV-only seg
    mr = cm.match_req(_pend([1, 2, 9]))          # probe hits the KV-only boundary
    assert mr.cuda_handle.cached_len == 0 and mr.mamba_value is None   # no snapshot there
    assert len(cm.free_slots) == base_pages and pool.num_free_slots == base_slots


def test_store_l2_offset_after_discard_reads_real_data(tmp_path):
    """B3 fails-before: _discard shrank _ssd_used while the blob fd is O_APPEND (writes at
    real EOF), so a post-discard offer recorded an offset BELOW EOF and restore returned
    another segment's bytes/padding."""
    from types import SimpleNamespace

    from freetoken.kvcache.utils import chain_page_key
    from freetoken.scheduler.session_tier import SessionTierStore

    store = SessionTierStore(
        SimpleNamespace(ram_bytes=0, dir=str(tmp_path), ssd_bytes=1 << 20))
    content = {}
    for i in (1, 2, 3):
        k = chain_page_key(None, (i,))
        content[k] = bytes([i]) * 4096
        assert store.offer(bytes([i]) * 16, 1, [(k, content[k])])
    assert store.evict_store(1) == 1            # discard seg 1: capacity accounting shrinks
    k4 = chain_page_key(None, (4,))
    content[k4] = bytes([4]) * 4096
    assert store.offer(bytes([4]) * 16, 1, [(k4, content[k4])])   # appends at real EOF
    hit = store.probe([k4])
    assert hit is not None and hit[0] == 1
    pages, _ = store.restore(hit[1])
    assert pages[0] == content[k4]              # byte-identical after the discard


# ------------------------------------------------- Review-2: N1 / N4


def test_tier_admission_short_prompt_guard_page_size_gt_1():
    """N1 fails-before: a prompt shorter than page_size raised IndexError in
    _tier_admission on every admission while the tier was on."""
    from freetoken.scheduler.cache import SessionTierCfg

    pool, kvpool = _pool(), _FakeKVPool(pages=16, page_size=8)
    pt = torch.zeros(4, 128, dtype=torch.int32)
    cm = CacheManager(16, 8, pt, "hybrid_radix", linear_state_pool=pool, swa_pool=kvpool,
                      session_tier_cfg=SessionTierCfg(ram_bytes=1 << 22))
    mr = cm.match_req(_pend([1, 2, 3]))          # 3 tokens < page_size 8
    assert mr.cuda_handle.cached_len == 0 and mr.mamba_value is None
    assert cm._tier_sessions == {}               # nothing indexed, nothing crashed


def test_tier_admission_bookkeeping_divergence_tip_promotion_cold():
    """N4: driven through match flows only - a cold first match tracks the session with
    zeros, a full match promotes the tip (old tip becomes the spare), a shallow match
    records the divergence depth. Two snapshot boundaries (4 and 8) make a nonzero
    shallow match possible (hybrid matches truncate to snapshot boundaries)."""
    pool, kvpool = _pool(), _FakeKVPool()
    pt = torch.zeros(8, 64, dtype=torch.int32)
    cm = _tiered_cm(pool, kvpool, pt)
    cm.match_req(_pend([1, 2, 3, 4]))            # cold: tracks the session, zeros
    k1 = next(iter(cm._tier_sessions))
    assert cm._tier_sessions[k1] == [0, 0, 0, None]

    _donate_one_snapshot(cm, pool, kvpool, pt, ids=(1, 2, 3, 4, 5))     # boundary 4

    # a second snapshot boundary at 8: finish-donate a req whose frozen mark is 8
    mr2 = cm.match_req(_pend(list(range(1, 10))))   # stripped 8: truncated to boundary 4
    live2, pp2 = pool.alloc(1)[0], tuple(pool.alloc(2))
    pt[1, :4] = torch.tensor([100, 101, 102, 103], dtype=torch.int32)
    pt[1, 4:8] = torch.tensor([104, 105, 106, 107], dtype=torch.int32)
    req2 = Req(input_ids=torch.tensor(list(range(1, 10)), dtype=torch.int32), table_idx=1,
               cached_len=8, output_len=1, uid=1, sampling_params=SamplingParams(),
               cache_handle=mr2.cuda_handle)
    req2.linear_slot_idx, req2.mamba_ping_pong = live2, pp2
    req2.mamba_next_track_idx = 1
    req2.mamba_last_track_seqlen = 8
    cm.lock(mr2.cuda_handle)
    cm.cache_req(req2, finished=True)            # donate at the boundary-8 node

    cm.match_req(_pend(list(range(1, 10))))      # full match: cached_len 8 == len(ids) 8
    assert cm._tier_sessions[k1] == [8, 0, 4, None]        # tip 8, old tip 4 = spare

    cm.match_req(_pend([1, 2, 3, 4, 9]))         # shallow: truncated to boundary 4
    assert cm._tier_sessions[k1] == [8, 4, 4, None]        # divergence depth 4 recorded

    cm.match_req(_pend([9, 9, 9, 9]))            # a different session: tracked cold
    assert len(cm._tier_sessions) == 2


def test_session_tier_boot_replay_restores_snapshot_segment(tmp_path):
    """Arm-3 boot restore: after a graceful shutdown (flush_live + compact) a FRESH
    CacheManager over the same tier dir must restore the journal-replayed segment on the
    first cold match. Fails-before: _tier_snap_bound is process-local and was never
    rehydrated at boot, so try_restore always fell back to a full re-prefill."""
    from freetoken.scheduler.cache import SessionTierCfg

    pool, kvpool = _pool(), _FakeKVPool()
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cfg = SessionTierCfg(ram_bytes=1 << 22, dir=str(tmp_path))
    cm = CacheManager(64, 1, pt, "hybrid_radix", linear_state_pool=pool, swa_pool=kvpool,
                      session_tier_cfg=cfg)
    original, _donated, _req = _donate_one_snapshot(cm, pool, kvpool, pt)
    cm.ensure_mamba_slots(pool.num_slots)        # wash: the tip demotes into the store
    assert cm.shutdown_tier() == 1               # graceful shutdown: flush L1 -> L2
    assert any(s.in_l2 for s in cm.tier_store._segments.values())

    pool2, kvpool2 = _pool(), _FakeKVPool()      # reboot: fresh manager, same tier dir
    pt2 = torch.zeros(4, 64, dtype=torch.int32)
    cm2 = CacheManager(64, 1, pt2, "hybrid_radix", linear_state_pool=pool2, swa_pool=kvpool2,
                       session_tier_cfg=SessionTierCfg(ram_bytes=1 << 22, dir=str(tmp_path)))
    mr = cm2.match_req(_pend([1, 2, 3, 4, 9]))   # cold tree, warm L2: first post-boot turn
    assert mr.cuda_handle.cached_len == 4 and mr.mamba_value is not None
    # the restore lands on the fresh free-list HEAD pages [0..3], not the donor's indices
    for j in range(4):
        assert kvpool2._kv_buffer[:, :, j].contiguous().cpu() \
            .view(torch.uint8).numpy().tobytes() == original[100 + j]


def test_kv_page_bytes_scale_layouts_roundtrip():
    """Arm-3 hardware: the real attention pool family is MHAKV (kv [2, L, page, ps, H, D],
    fp8 scale [2, L, pages*ps, H]) AND DSA/MLA (kv [1, L, page, ps, 1, D], fp8 scale
    [L, pages*ps]). Pre-fix the codec's scale assert indexed s.shape[2], which IndexError'd
    on the 2-D DSA scale at boot (crash seen on hardware, run_arm3.sh first launch)."""
    from freetoken.scheduler.cache import _KVPageBytes, SessionTierCfg

    class _FakeMHAScale:
        swa_paged = False

        def __init__(self):
            self._kv_buffer = torch.zeros(2, 2, 16, 4, 2, 4, dtype=torch.bfloat16)
            self._scale_buffer = torch.zeros(2, 2, 16 * 4, 2, dtype=torch.float32)
            self._block_scale_buffer = None

    class _FakeDSA:
        swa_paged = False

        def __init__(self):
            self._kv_buffer = torch.zeros(1, 2, 16, 4, 1, 8, dtype=torch.uint8)
            self._scale_buffer = torch.zeros(2, 16 * 4, dtype=torch.float32)
            self._block_scale_buffer = None

    for pool in (_FakeMHAScale(), _FakeDSA()):
        codec = _KVPageBytes(pool)
        for page in (0, 7, 15):
            n = len(codec.read_page(page))
            blob = bytes((page * 31 + i) % 256 for i in range(n))
            codec.write_page(page, blob)
            assert codec.read_page(page) == blob

    # the boot hook must not raise on the DSA-shaped pool (CacheManager tier init)
    pool, kvpool = _pool(), _FakeDSA()
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, pt, "hybrid_radix", linear_state_pool=pool, swa_pool=kvpool,
                      session_tier_cfg=SessionTierCfg(ram_bytes=1 << 22))
    assert cm._tier_on and cm._page_bytes is not None


def test_session_tier_restored_finish_adopts_pages_nonaligned():
    """Arm-3 hardware (896-page leak -> integrity kill): a restored request whose prefill
    covered only the page tail finishes with a NON-page-aligned cached_len; the finish's
    else branch skipped the insert entirely, so the restore's own fresh pages were neither
    tree-owned nor freed. page_size 8, tail 1 token."""
    from freetoken.scheduler.cache import SessionTierCfg

    pool, kvpool = _pool(), _FakeKVPool(pages=16, page_size=8)
    pt = torch.zeros(4, 128, dtype=torch.int32)
    cm = CacheManager(16, 8, pt, "hybrid_radix", linear_state_pool=pool, swa_pool=kvpool,
                      session_tier_cfg=SessionTierCfg(ram_bytes=1 << 22))
    ids = list(range(1, 10))                     # 9 tokens: 8 aligned + 1 tail
    mr = cm.match_req(_pend(ids))                # cold admission tracks the session
    pages = cm._allocate(2)                      # page bases for [0:8) and the tail
    pt[0, :8] = pages[0]
    tail_page = int(pages[1])
    pt[0, 8:9] = tail_page
    for pg in range(2):
        kvpool._kv_buffer[:, :, int(pages[pg]) // 8] = float(pg) * 1.5
    req = Req(input_ids=torch.tensor(ids + [99], dtype=torch.int32), table_idx=0, cached_len=9,
              output_len=1, uid=0, sampling_params=SamplingParams(), cache_handle=mr.cuda_handle)
    req.linear_slot_idx, req.mamba_ping_pong = pool.alloc(1)[0], tuple(pool.alloc(2))
    req.mamba_next_track_idx = 1
    req.mamba_last_track_seqlen = 8              # chunk-commit the x8 boundary + donate
    cm.lock(mr.cuda_handle)
    cm.cache_req(req, finished=False)
    cm.cache_req(req, finished=True)             # finish the donor (tail freed, slots back)
    cm.ensure_mamba_slots(pool.num_slots)        # wash: the tip demotes, tree empty
    chain = cm._chain_keys(torch.tensor(ids[:8], dtype=torch.int32))
    assert cm.tier_store.probe(chain) is not None

    mr2 = cm.match_req(_pend(ids))               # restored admission: 8 tokens + 1 slot
    assert mr2.cuda_handle.cached_len == 8 and mr2.mamba_value is not None
    restored_pages = mr2.cuda_handle.get_matched_indices().clone()
    req2 = Req(input_ids=torch.tensor(ids + [99], dtype=torch.int32), table_idx=0, cached_len=9,
               output_len=1, uid=1, sampling_params=SamplingParams(),
               cache_handle=mr2.cuda_handle)
    req2.linear_slot_idx = pool.alloc(1)[0]
    req2.mamba_ping_pong = tuple(pool.alloc(2))
    cm.lock(mr2.cuda_handle)
    pt[0, :8] = restored_pages                   # what the admission writes (prefill.py)
    pt[0, 8:9] = int(cm._allocate(1)[0])         # the tail page (really allocated)
    cm.cache_req(req2, finished=True)
    cm.check_integrity()                         # pre-fix: 1 restored page leaked
    assert cm.prefix_cache.full_evictable == 8   # the tree owns the restored page now
    m = cm.prefix_cache.match_prefix(torch.tensor(ids[:8], dtype=torch.int32))
    assert m.cached_len == 0                     # KV-only node: reuse goes via the store


def test_session_tier_restore_refuses_tree_owned_span():
    """Arm-3 HW bug 4 (253-page leak -> integrity kill): the store's deepest boundary can be
    SHALLOWER than the tree's surviving match and overlap KV-only tombstones (both keep their
    pages). Pre-fix try_restore fired whenever cached_len < len(ids), replaced the DEEPER tree
    match with a shallower restore, and the restored pages (duplicates of tree-owned KV)
    leaked at the commit-time dedup. The restore must fire only when its span is wholly
    novel; otherwise the normal (deeper) path serves the request."""
    pool, kvpool = _pool(), _FakeKVPool()
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = _tiered_cm(pool, kvpool, pt)
    ids13 = list(range(1, 14))                   # 13 tokens; tree chain [0:11) (match 11)
    pages = cm._allocate(11)
    for pg in range(11):
        kvpool._kv_buffer[:, :, int(pages[pg])] = float(pg) * 1.5

    slots = [pool.alloc(1)[0] for _ in range(3)]
    cm.prefix_cache.insert(torch.tensor(ids13[:4], dtype=torch.int32), pages[:4], slots[0])
    cm.prefix_cache.insert(torch.tensor(ids13[:8], dtype=torch.int32), pages[:8], slots[1])
    cm.prefix_cache.insert(torch.tensor(ids13[:11], dtype=torch.int32), pages[:11], slots[2])
    cm.match_req(_pend(ids13))                   # tracks the session (admission bookkeeping)

    er = cm.prefix_cache.evict_mamba(2, derive_paths=True)   # tombstone [0:4) and [4:8)
    cm._tier_offer(er.victim_paths)              # store: boundaries 4 and 8 (snapshots live)
    cm.linear_state_pool.free(er.mamba_slots)
    cm._free(er.kv_indices)

    mr = cm.match_req(_pend(ids13))              # tree match 11 vs store depth 8
    assert mr.cuda_handle.cached_len == 11       # pre-fix: the shallower restore replaced it
    assert mr.mamba_value not in cm._tier_restore_slots

    req = Req(input_ids=torch.tensor(ids13 + [99], dtype=torch.int32), table_idx=0,
              cached_len=13, output_len=1, uid=0, sampling_params=SamplingParams(),
              cache_handle=mr.cuda_handle)
    req.linear_slot_idx, req.mamba_ping_pong = pool.alloc(1)[0], tuple(pool.alloc(2))
    pt[0, :11] = mr.cuda_handle.get_matched_indices()
    pt[0, 11:13] = cm._allocate(2)
    cm.lock(mr.cuda_handle)
    cm.cache_req(req, finished=True)
    cm.check_integrity()                         # pre-fix: 8 restored dup pages leaked
    hit = cm.tier_store.probe(cm._chain_keys(torch.tensor(ids13[:12], dtype=torch.int32)))
    assert hit is not None and hit[0] == 8       # the store segment is intact, just unused


def test_session_tier_snapshot_only_rerestore_over_adopted_span():
    """Arm-3 phase-D (byte-identity run): after a restored turn's finish adopted its pages as
    KV-only nodes, a repeat of the same prompt can neither tree-match (no live snapshot) nor
    full-restore (the span is tree-owned) - pre-fix it fell back to a full 57k re-prefill
    (HW: cached=0, 77.5s, byte-different response). The restore must go snapshot-only: the
    tree pages stay put and the store's boundary snapshot revives the tip."""
    pool, kvpool = _pool(), _FakeKVPool()
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = _tiered_cm(pool, kvpool, pt)
    ids = [1, 2, 3, 4, 5]
    mr = cm.match_req(_pend(ids))                # cold: tracks the session
    pages = cm._allocate(4)
    pt[0, :4] = pages
    pt[0, 4:5] = cm._allocate(1)[0]
    for pg in range(4):
        kvpool._kv_buffer[:, :, int(pages[pg])] = float(pg) * 1.5
    req = Req(input_ids=torch.tensor(ids + [99], dtype=torch.int32), table_idx=0,
              cached_len=5, output_len=1, uid=0, sampling_params=SamplingParams(),
              cache_handle=mr.cuda_handle)
    req.linear_slot_idx, req.mamba_ping_pong = pool.alloc(1)[0], tuple(pool.alloc(2))
    req.mamba_next_track_idx = 1
    req.mamba_last_track_seqlen = 4
    cm.lock(mr.cuda_handle)
    cm.cache_req(req, finished=False)            # chunk commit [0:4) + snapshot donation
    cm.cache_req(req, finished=True)             # non-aligned finish: normal path, tail freed
    cm.ensure_mamba_slots(pool.num_slots)        # wash: tip -> store, tree empty

    mr1 = cm.match_req(_pend(ids))               # full restore: fresh pages, node=root
    assert mr1.cuda_handle.cached_len == 4 and mr1.cuda_handle.tier_restored
    restored_pages = mr1.cuda_handle.get_matched_indices().clone()
    req2 = Req(input_ids=torch.tensor(ids + [99], dtype=torch.int32), table_idx=0,
               cached_len=5, output_len=1, uid=1, sampling_params=SamplingParams(),
               cache_handle=mr1.cuda_handle)
    req2.linear_slot_idx, req2.mamba_ping_pong = pool.alloc(1)[0], tuple(pool.alloc(2))
    cm.lock(mr1.cuda_handle)
    pt[0, :4] = restored_pages                   # admission rows: the restore's fresh pages
    pt[0, 4:5] = cm._allocate(1)[0]
    cm.cache_req(req2, finished=True)            # adopts [0:5) as a KV-only node

    free_before = len(cm.free_slots)
    mr2 = cm.match_req(_pend(ids))               # KV-only span: no live match, no fresh pages
    assert mr2.cuda_handle.cached_len == 4       # snapshot-only restore revives the boundary
    assert mr2.mamba_value is not None and mr2.mamba_value in cm._tier_restore_slots
    assert len(cm.free_slots) == free_before     # zero pages taken
    cm.check_integrity()                         # ledger balanced


# ----------------------------- S-wave: per-handle pending + QSA page codec


def test_pending_restore_is_per_handle():
    """S-wave fails-before: _tier_pending_restore was a single slot, so a second restore
    overwrote the first pending record and abandoning the FIRST admission leaked its 4
    pages + GDN slot permanently. Per-handle records return exactly their own."""
    pool, kvpool = _pool(), _FakeKVPool()
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = _tiered_cm(pool, kvpool, pt)
    _donate_one_snapshot(cm, pool, kvpool, pt)
    cm.ensure_mamba_slots(pool.num_slots)

    mr1 = cm.match_req(_pend([1, 2, 3, 4, 9]))   # restore #1
    assert mr1.cuda_handle.cached_len == 4 and mr1.mamba_value is not None
    mr2 = cm.match_req(_pend([1, 2, 3, 4, 9]))   # restore #2 overwrote the record (pre-fix)
    assert mr2.cuda_handle.cached_len == 4 and mr2.mamba_value is not None
    pages_held, slots_held = len(cm.free_slots), pool.num_free_slots
    cm.abandon_restore(mr1.cuda_handle)          # refuse #1 AFTER #2 restored
    assert len(cm.free_slots) == pages_held + 4 and pool.num_free_slots == slots_held + 1
    cm.abandon_restore(mr1.cuda_handle)          # idempotent
    assert len(cm.free_slots) == pages_held + 4 and pool.num_free_slots == slots_held + 1
    cm.lock(mr2.cuda_handle)                     # admission #2: consumes only its own entry
    cm.abandon_restore(mr2.cuda_handle)          # already consumed by lock: no double free
    assert len(cm.free_slots) == pages_held + 4 and pool.num_free_slots == slots_held + 1
    assert cm._tier_pending_restore == {}


def _qsa_pool(monkeypatch, *, mrope=False, num_pages=16, page_size=16, index_ratio=4):
    """Real QSAKVCache (Qwen3.8-Flash-Next geometry scaled down): 4 sparse slabs, ratio 4,
    so page_size 16 -> 4 slab rows per page. TP singleton like tests/kvcache/test_qsa_pool."""
    from freetoken.distributed.info import DistributedInfo
    from freetoken.kvcache.qsa_pool import QSAKVCache

    monkeypatch.setattr("freetoken.kvcache.mha_pool.get_tp_info",
                        lambda: DistributedInfo(rank=0, size=1))
    return QSAKVCache(
        num_kv_heads=2, num_layers=8, head_dim=64, num_pages=num_pages,
        page_size=page_size, dtype=torch.bfloat16, device=torch.device("cpu"),
        index_head_dim=32, num_index_layers=4, index_ratio=index_ratio,
        num_req_slots=4, mrope=mrope, layer_ids=(1, 3, 5, 7),
    )


def test_tier_activates_qsa_pool(monkeypatch):
    """QSA codec wave: the interim blanket refusal is replaced by the complete codec, so a
    healthy QSAKVCache (slab + rope tiers present) now ACTIVATES the tier. Fails-before by
    construction: pre-change _tier_refusal_reason refused every QSA pool (the restored
    pages would serve stale slab rows -> wrong block selection on the QSA layers)."""
    from freetoken.scheduler.cache import SessionTierCfg, _tier_refusal_reason

    qsa = _qsa_pool(monkeypatch, mrope=True)
    assert _tier_refusal_reason(qsa) is None
    pool, pt = _pool(), torch.zeros(4, 256, dtype=torch.int32)
    cm = CacheManager(16, 16, pt, "hybrid_radix", linear_state_pool=pool, swa_pool=qsa,
                      session_tier_cfg=SessionTierCfg(ram_bytes=1 << 22))
    assert cm._tier_on and cm.tier_store is not None
    assert cm._page_bytes._cmp is qsa._cmp_k_buffer
    assert cm._page_bytes._rope is qsa._rope_positions
    mr = cm.match_req(_pend([1, 2, 3]))          # hooks stay inert exactly like off-mode
    assert mr.cuda_handle.cached_len == 0 and mr.mamba_value is None


def test_qsa_refusal_narrows_to_uncoverable_variants(monkeypatch):
    """The refusal is no longer keyed on the pool TYPE: a healthy QSA pool passes; only an
    uncoverable layout refuses loudly with the tier off. page_size % index_ratio != 0 needs
    no refusal branch - the QSA pool constructor rejects it (ValueError) before the tier
    ever sees the pool; a detached slab is the defensive case this pin keeps loud."""
    from freetoken.scheduler import cache as cache_mod
    from freetoken.scheduler.cache import SessionTierCfg, _tier_refusal_reason

    qsa = _qsa_pool(monkeypatch)
    assert _tier_refusal_reason(qsa) is None
    qsa._cmp_k_buffer = None                     # uncoverable variant (defensive)
    assert _tier_refusal_reason(qsa) is not None
    seen = []
    monkeypatch.setattr(cache_mod.logger, "warning",
                        lambda msg, *a: seen.append(msg % a if a else msg))
    pool, pt = _pool(), torch.zeros(4, 256, dtype=torch.int32)
    cm = CacheManager(16, 16, pt, "hybrid_radix", linear_state_pool=pool, swa_pool=qsa,
                      session_tier_cfg=SessionTierCfg(ram_bytes=1 << 22))
    assert cm.tier_store is None and not cm._tier_on
    assert any("session tier refused" in m for m in seen)


def test_kv_page_bytes_qsa_layout_roundtrip(monkeypatch):
    """The QSA page codec demotes the K/V page TOGETHER with its compressed-slab rows
    [page*ps/ratio, (page+1)*ps/ratio) and its rope rows [page*ps, (page+1)*ps). The
    per-request scratch rows (past cmp_scratch_base) must stay OUT of the page bytes: a
    restored tenant rebuilds them in its own forwards. Non-mrope pools have no rope tier;
    quantized pools add the generic scale buffers on top (fake variant, CUDA-free)."""
    from freetoken.kvcache.qsa_pool import QSAKVCache
    from freetoken.scheduler.cache import _KVPageBytes

    for mrope in (True, False):
        qsa = _qsa_pool(monkeypatch, mrope=mrope)
        codec = _KVPageBytes(qsa)
        n = len(codec.read_page(2))
        kv = qsa._kv_buffer[:, :, 2].numel() * qsa._kv_buffer.element_size()
        slab = qsa._cmp_k_buffer[:, 8:12, :].numel() * qsa._cmp_k_buffer.element_size()
        rope = (16 * 3 * 4) if mrope else 0      # _rope_positions rows are int32
        assert n == kv + slab + rope
        original = codec.read_page(2)
        blob = bytes((i * 37 + 5) % 256 for i in range(n))
        codec.write_page(2, blob)
        assert codec.read_page(2) == blob
        # the write landed in page 2's slab rows only: groups of other pages and the
        # scratch rows stay zero
        assert qsa._cmp_k_buffer[:, :8, :].abs().sum().item() == 0
        assert qsa._cmp_k_buffer[:, 12:, :].abs().sum().item() == 0
        codec.write_page(2, original)
        assert codec.read_page(2) == original

    # quantized variant: fp8-style scale buffers locate at dim 2 exactly like the MHAKV
    # family; the slab + rope append AFTER them (same order on read and write)
    qsa = QSAKVCache.__new__(QSAKVCache)         # capability probe, no runtime init
    qsa._index_ratio = 2
    qsa._kv_buffer = torch.zeros(2, 2, 8, 4, 2, 4, dtype=torch.bfloat16)
    qsa._scale_buffer = torch.zeros(2, 2, 32, 2, dtype=torch.float32)
    qsa._block_scale_buffer = None
    qsa._cmp_k_buffer = torch.zeros(3, 16 + 2, 6, dtype=torch.bfloat16)
    qsa._rope_positions = torch.zeros(32, 3, dtype=torch.int32)
    codec = _KVPageBytes(qsa)
    n = len(codec.read_page(5))
    assert n == (2 * 2 * 4 * 2 * 4 * 2 + 2 * 2 * 4 * 2 * 4 + 3 * 2 * 6 * 2 + 4 * 3 * 4)
    blob = bytes((i * 11 + 3) % 256 for i in range(n))
    codec.write_page(5, blob)
    assert codec.read_page(5) == blob


def test_kv_page_bytes_kpooldsa_layout_roundtrip():
    """ITEM-3: the KpoolDSA page codec demotes the latent page + its 2-D fp8 scales
    TOGETHER with the 1/ratio page-owned index-shadow rows [page*ps/ratio,
    (page+1)*ps/ratio) per indexer layer. The per-request scratch rows (past
    cmp_scratch_base) and the tail rings stay OUT of the page bytes: per-forward state a
    restored tenant rebuilds (dsa_pool.KpoolDSAKVCache docstring). Fails-before by
    construction: the W2 matrix class-refused KpoolDSAKVCache, so no codec existed."""
    from freetoken.kvcache.dsa_pool import KpoolDSAKVCache
    from freetoken.scheduler.cache import _KVPageBytes

    kpool = KpoolDSAKVCache(latent_dim=16, num_layers=2, num_pages=8, page_size=4,
                            dtype=torch.bfloat16, device=torch.device("cpu"),
                            index_head_dim=6, num_index_layers=3, index_ratio=2,
                            num_req_slots=4, kv_quant="fp8")
    codec = _KVPageBytes(kpool)
    n = len(codec.read_page(2))
    kv = kpool._kv_buffer[:, :, 2].numel()                 # uint8 latents
    scale = kpool._scale_buffer[:, 8:12].numel() * 4       # fp32 token rows
    shadow = kpool._index_k_buffer[:, 4:6, :].numel() * 2  # bf16 shadow rows
    assert n == kv + scale + shadow
    original = codec.read_page(2)
    blob = bytes((i * 41 + 7) % 256 for i in range(n))
    codec.write_page(2, blob)
    assert codec.read_page(2) == blob
    # the write landed in page 2's shadow rows only: other rows, the scratch rows (past
    # cmp_scratch_base = 16) and the tail rings stay zero
    assert kpool._index_k_buffer[:, :4, :].abs().sum().item() == 0
    assert kpool._index_k_buffer[:, 6:, :].abs().sum().item() == 0
    assert kpool._tail_k.abs().sum().item() == 0 and kpool._tail_gate.abs().sum().item() == 0
    codec.write_page(2, original)
    assert codec.read_page(2) == original


def test_session_tier_qsa_restore_roundtrip_bit_exact(monkeypatch):
    """End-to-end QSA seam: an eviction offer reads kv+slab+rope bytes for the victim's
    pages; a restored admission writes them back BIT-EXACTLY into fresh pages. Fails-before
    by construction: pre-change the guard refused activation; with the old KV-only codec
    the restored pages would serve a hostile next tenant's stale slab rows. The whole pool
    is zeroed after the wash so any tier the codec forgets comes back as zeros, not as the
    donor's surviving values."""
    from freetoken.scheduler.cache import SessionTierCfg

    qsa = _qsa_pool(monkeypatch, mrope=True)
    pool, pt = _pool(), torch.zeros(4, 256, dtype=torch.int32)
    cm = CacheManager(16, 16, pt, "hybrid_radix", linear_state_pool=pool, swa_pool=qsa,
                      session_tier_cfg=SessionTierCfg(ram_bytes=1 << 22))
    assert cm._tier_on                           # activation itself fails-before
    ids = list(range(1, 65))                     # 64 tokens = 4 pages
    mr = cm.match_req(_pend(ids))                # cold admission tracks the session
    pages = cm._allocate(4)                      # token bases [0,16,32,48]: one per page
    for p in range(4):
        pt[0, p * 16:(p + 1) * 16] = int(pages[p])
    rps = 16 // qsa.index_ratio                  # slab rows per page
    for p in range(4):
        qsa._kv_buffer[:, :, p] = float(p) * 1.5
        qsa._cmp_k_buffer[:, p * rps:(p + 1) * rps, :] = float(p) + 0.25
        qsa._rope_positions[p * 16:(p + 1) * 16, :] = torch.arange(
            p * 16, (p + 1) * 16, dtype=torch.int32).unsqueeze(1)
    kv_ref = [qsa._kv_buffer[:, :, p].clone() for p in range(4)]
    slab_ref = [qsa._cmp_k_buffer[:, p * rps:(p + 1) * rps, :].clone() for p in range(4)]
    rope_ref = [qsa._rope_positions[p * 16:(p + 1) * 16].clone() for p in range(4)]

    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    req = Req(input_ids=torch.tensor(ids + [99], dtype=torch.int32), table_idx=0,
              cached_len=64, output_len=1, uid=0, sampling_params=SamplingParams(),
              cache_handle=mr.cuda_handle)
    req.linear_slot_idx, req.mamba_ping_pong = live, pp
    req.mamba_next_track_idx = 1
    req.mamba_last_track_seqlen = 64             # donate the x64 snapshot
    cm.lock(mr.cuda_handle)
    cm.cache_req(req, finished=False)
    cm.unlock(req.cache_handle)
    cm.ensure_mamba_slots(pool.num_slots)        # wash: the tip demotes to the store
    hit = cm.tier_store.probe(cm._chain_keys(torch.tensor(ids, dtype=torch.int32)))
    assert hit is not None and hit[0] == 4       # 4 pages offered

    # hostile next tenant: every tier of every page zeroed; the restore must rebuild ALL
    # of them, not just the K/V page the phase-1 codec knew
    qsa._kv_buffer.zero_()
    qsa._cmp_k_buffer.zero_()
    qsa._rope_positions.zero_()

    mr2 = cm.match_req(_pend(ids + [99]))        # restored admission: 64 tokens + 1 slot
    assert mr2.cuda_handle.cached_len == 64 and mr2.cuda_handle.tier_restored
    assert mr2.mamba_value is not None           # the boundary snapshot rides along
    matched = mr2.cuda_handle.get_matched_indices()
    for p in range(4):
        page = int(matched[p * 16]) // 16
        assert torch.equal(qsa._kv_buffer[:, :, page], kv_ref[p])
        assert torch.equal(qsa._cmp_k_buffer[:, page * rps:(page + 1) * rps, :], slab_ref[p])
        assert torch.equal(qsa._rope_positions[page * 16:(page + 1) * 16], rope_ref[p])
    # the restore's own scratch rows stay zero (per-request tier, rebuilt per forward)
    assert qsa._cmp_k_buffer[:, qsa.cmp_scratch_base:, :].abs().sum().item() == 0


# --------------------- W2 generic brief: KV-only tier on a plain radix manager


def _kvonly_cm(kvpool, pt, ram=1 << 22, num_pages=64, page_size=1):
    from freetoken.scheduler.cache import SessionTierCfg

    return CacheManager(num_pages, page_size, pt, "radix", swa_pool=kvpool,
                        session_tier_cfg=SessionTierCfg(ram_bytes=ram))


def _kv_bytes(kvpool, page: int) -> bytes:
    return kvpool._kv_buffer[:, :, page].contiguous().cpu() \
        .view(torch.uint8).numpy().tobytes()


def _fill_pages(cm, kvpool, pt, ids, uid=0):
    """Allocate len(ids) page bases, stamp recognizable bytes into the pool and the pt row
    (ps==1: one token per page). Returns (page bases, {page: original bytes})."""
    pages = cm._allocate(len(ids))
    pt[uid, :len(ids)] = pages
    originals = {}
    for pg in pages:
        pg = int(pg)
        kvpool._kv_buffer[:, :, pg] = float(pg) * 1.5
        originals[pg] = _kv_bytes(kvpool, pg)
    return pages, originals


def test_kvonly_tier_activates_on_plain_radix_and_stays_off_elsewhere():
    """Generic brief: a plain radix manager (no LinearStatePool) with tier flags runs the
    KV-only currency; naive has no tree (no boundaries to index) and stays off."""
    from freetoken.scheduler.cache import SessionTierCfg

    kvpool = _FakeKVPool()
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = _kvonly_cm(kvpool, pt)
    assert cm._tier_on and cm.tier_store is not None and cm._page_bytes is not None
    mr = cm.match_req(_pend([1, 2, 3]))          # hooks run on the plain path
    assert mr.cuda_handle.cached_len == 0 and mr.mamba_value is None
    assert len(cm._tier_sessions) == 1           # admission bookkeeping ran

    naive = CacheManager(64, 1, pt, "naive", swa_pool=_FakeKVPool(),
                         session_tier_cfg=SessionTierCfg(ram_bytes=1 << 22))
    assert naive.tier_store is None and not naive._tier_on


def test_kvonly_refusal_matrix_loud(monkeypatch):
    """Coverage matrix: only page-owned tiers byte-encode. BSA (index-key slab) holds a
    tier the codec must not silently drop -> loud refusal; a pool without _kv_buffer
    (Hybrid-SWA / DSV4 families) falls to the generic no-codec warning. KpoolDSA is
    COVERED since the ITEM-3 codec branch (page-owned 1/ratio index shadow; scratch rows
    and tail rings are per-forward state) - it activates like QSA."""
    from freetoken.distributed.info import DistributedInfo
    from freetoken.kvcache.bsa_pool import BSAKVCache
    from freetoken.kvcache.dsa_pool import KpoolDSAKVCache
    from freetoken.scheduler import cache as cache_mod
    from freetoken.scheduler.cache import SessionTierCfg, _tier_refusal_reason

    monkeypatch.setattr("freetoken.kvcache.mha_pool.get_tp_info",
                        lambda: DistributedInfo(rank=0, size=1))
    bsa = BSAKVCache(num_kv_heads=2, num_layers=2, head_dim=16, num_pages=4, page_size=1,
                     dtype=torch.bfloat16, device=torch.device("cpu"),
                     index_head_dim=8, num_index_layers=2)
    kpool = KpoolDSAKVCache(latent_dim=16, num_layers=2, num_pages=4, page_size=1,
                            dtype=torch.bfloat16, device=torch.device("cpu"),
                            index_head_dim=8, num_index_layers=2,
                            index_ratio=1, num_req_slots=4)
    assert _tier_refusal_reason(bsa) is not None
    assert _tier_refusal_reason(kpool) is None   # ITEM-3: covered by the codec branch
    seen = []
    monkeypatch.setattr(cache_mod.logger, "warning",
                        lambda msg, *a: seen.append(msg % a if a else msg))
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, pt, "radix", swa_pool=bsa,
                      session_tier_cfg=SessionTierCfg(ram_bytes=1 << 22))
    assert cm.tier_store is None and not cm._tier_on
    assert any("session tier refused" in m for m in seen)
    seen.clear()
    cm2 = CacheManager(64, 1, pt, "radix",
                       swa_pool=SimpleNamespace(swa_paged=False),   # no _kv_buffer
                       session_tier_cfg=SessionTierCfg(ram_bytes=1 << 22))
    assert cm2.tier_store is None and not cm2._tier_on
    assert any("no paged-KV byte codec" in m for m in seen)


def test_kvonly_adaptive_set_degenerates_to_tip():
    """KV-only tip-only set: without snapshots the tip segment serves every shallower
    resume (probe matches the common chain prefix; restore depth-truncates its pages), so
    the hybrid under-divergence slot is pure overhead. Fails-before: no offer seam ran for
    the plain path (store empty). Discriminates the rules: with a divergence at 4, the
    hybrid keep-set would also keep the boundary-4 victim; tip-only must drop it."""
    kvpool = _FakeKVPool()
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = _kvonly_cm(kvpool, pt)
    ids = list(range(1, 9))                      # 8 tokens = 2 tree nodes
    cm.match_req(_pend(ids + [9]))               # tip 8 tracked
    cm.match_req(_pend([1, 2, 3, 4, 99, 98]))    # divergence at 4 (st[1] = 4)
    pages, originals = _fill_pages(cm, kvpool, pt, ids)
    cm.prefix_cache.insert_prefix(torch.tensor(ids[:4], dtype=torch.int32), pages[:4])
    cm.prefix_cache.insert_prefix(torch.tensor(ids, dtype=torch.int32), pages)

    cm._allocate(61)                             # pressure: one eviction pops BOTH leaves
                                                 # (B first, then exposed A as a leaf) and
                                                 # auto-offers both through the same seam
    chain8 = cm._chain_keys(torch.tensor(ids, dtype=torch.int32))
    held = {seg.path_key for seg in cm.tier_store._segments.values()}
    assert held == {chain8[-1]}                  # tip-only: the boundary-4 victim dropped

    # justification: a resume diverging at 4 restores the shared [0:4) pages from the TIP
    # segment - the under-divergence boundary would have stored the same bytes again
    hit = cm.tier_store.probe(cm._chain_keys(torch.tensor(ids[:4], dtype=torch.int32)))
    assert hit is not None and hit[0] == 4
    got, snaps = cm.tier_store.restore(hit[1])
    assert snaps == []
    assert got == [originals[int(pg)] for pg in pages[:4]]


def test_kvonly_restore_feeds_normal_insert_path():
    """Cold tree + warm store: the store hit restores byte-identical KV pages WITHOUT a
    re-prefill, the restored handle feeds the normal insert path (the commit's
    insert_prefix adopts the pages and the tree re-grows), consuming exactly the free
    pages the replaced re-prefill would. Fails-before: cached_len stayed 0 (no restore
    on the plain path)."""
    kvpool = _FakeKVPool()
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = _kvonly_cm(kvpool, pt)
    ids = list(range(1, 13))
    cm.match_req(_pend(ids))                     # track the session (cold)
    pages, originals = _fill_pages(cm, kvpool, pt, ids)
    cm.prefix_cache.insert_prefix(torch.tensor(ids, dtype=torch.int32), pages)
    evicted, victims = cm.prefix_cache.evict_paths(1)
    cm._free(evicted[::cm.page_size])
    cm._tier_offer(victims)                      # wash: tree empty, store warm
    assert cm.prefix_cache.size_info.total_size == 0

    frees_before = len(cm.free_slots)
    mr = cm.match_req(_pend(ids + [99]))
    assert mr.cuda_handle.cached_len == 12
    assert mr.mamba_value is None                # KV-only: no slot rides along
    matched = mr.cuda_handle.get_matched_indices()
    assert matched.numel() == 12
    for j in range(12):
        assert _kv_bytes(kvpool, int(matched[j])) == originals[int(pages[j])]
    assert len(cm.free_slots) == frees_before - 12   # budget: same as the re-prefill

    cm.lock(mr.cuda_handle)
    pt[0, :12] = matched                         # what the admission writes (prefill.py)
    req = Req(input_ids=torch.tensor(ids + [99], dtype=torch.int32), table_idx=0,
              cached_len=12, output_len=1, uid=0, sampling_params=SamplingParams(),
              cache_handle=mr.cuda_handle)
    cm.cache_req(req, finished=True)
    cm.check_integrity()                         # ledger balanced after the adoption
    m = cm.prefix_cache.match_prefix(torch.tensor(ids, dtype=torch.int32))
    assert m.cuda_handle.cached_len == 12        # the tree re-grew over the restored span


def test_kvonly_restore_restores_only_the_unowned_suffix():
    """owned_prefix interplay: the tree owns [0:4) (a live KV-only node) and the store's
    boundary is deeper (8): the restore takes only the missing [4:8) fresh pages - the
    tree's pages stay in place and ride the handle. Fails-before: no restore on the plain
    path (cached_len 0)."""
    from freetoken.kvcache.hybrid_radix_cache import VictimPath

    kvpool = _FakeKVPool()
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = _kvonly_cm(kvpool, pt)
    ids = list(range(1, 13))
    cm.match_req(_pend([1, 2, 3, 4, 5]))         # cold admission tracks the session
    pages, originals = _fill_pages(cm, kvpool, pt, ids)
    cm.prefix_cache.insert_prefix(torch.tensor(ids[:4], dtype=torch.int32), pages[:4])
    chain = cm._chain_keys(torch.tensor(ids, dtype=torch.int32))
    cm._tier_offer([VictimPath(chain[7], 8, tuple(chain[:8]), pages[:8], None)])
    cm._free(pages[4:])                          # the offer covered [0:8); drop the rest

    mr = cm.match_req(_pend(ids + [99]))
    assert mr.cuda_handle.cached_len == 8
    matched = mr.cuda_handle.get_matched_indices()
    assert matched[:4].tolist() == pages[:4].tolist()    # tree-owned prefix untouched
    for j in range(4, 8):
        assert _kv_bytes(kvpool, int(matched[j])) == originals[int(pages[j])]
    assert len(cm.free_slots) == 64 - 4 - 4      # only 4 fresh pages taken

    cm.lock(mr.cuda_handle)
    pt[0, :8] = matched
    req = Req(input_ids=torch.tensor(ids + [99], dtype=torch.int32), table_idx=0,
              cached_len=8, output_len=1, uid=0, sampling_params=SamplingParams(),
              cache_handle=mr.cuda_handle)
    cm.cache_req(req, finished=True)
    cm.check_integrity()
    m = cm.prefix_cache.match_prefix(torch.tensor(ids[:8], dtype=torch.int32))
    assert m.cuda_handle.cached_len == 8


def test_kvonly_abandon_restore_returns_pages():
    """A restored-but-refused KV-only admission returns its fresh pages (no slot); a
    second restore is unaffected; lock() consumes the record so a late abandon is a
    no-op. Fails-before: no restore ran (cached_len 0)."""
    kvpool = _FakeKVPool()
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = _kvonly_cm(kvpool, pt)
    ids = list(range(1, 13))
    cm.match_req(_pend(ids))
    pages, originals = _fill_pages(cm, kvpool, pt, ids)
    cm.prefix_cache.insert_prefix(torch.tensor(ids, dtype=torch.int32), pages)
    evicted, victims = cm.prefix_cache.evict_paths(1)
    cm._free(evicted[::cm.page_size])
    cm._tier_offer(victims)

    mr = cm.match_req(_pend(ids + [99]))
    assert mr.cuda_handle.cached_len == 12
    held = len(cm.free_slots)                    # 52: the 12 restored pages are held
    cm.abandon_restore(mr.cuda_handle)
    assert len(cm.free_slots) == held + 12
    assert cm._tier_pending_restore == {}
    cm.abandon_restore(mr.cuda_handle)           # idempotent
    assert len(cm.free_slots) == held + 12

    mr2 = cm.match_req(_pend(ids + [99]))        # a second restore after the refusal
    assert mr2.cuda_handle.cached_len == 12
    pages_mid = len(cm.free_slots)
    cm.lock(mr2.cuda_handle)                     # admission consumes the record
    cm.abandon_restore(mr2.cuda_handle)          # no double return
    assert len(cm.free_slots) == pages_mid
    assert cm._tier_pending_restore == {}


def test_kvonly_restore_refuses_tree_owned_span():
    """Regression pin: the store's boundary can be SHALLOWER than the tree's surviving
    match (both keep their pages); the restore must fire only when its span is wholly
    novel, else restored duplicates of tree-owned KV leak at the commit-time dedup."""
    from freetoken.kvcache.hybrid_radix_cache import VictimPath

    kvpool = _FakeKVPool()
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = _kvonly_cm(kvpool, pt)
    ids13 = list(range(1, 14))
    cm.match_req(_pend(ids13))                   # tracks the session
    pages, originals = _fill_pages(cm, kvpool, pt, ids13[:11])
    cm.prefix_cache.insert_prefix(torch.tensor(ids13[:11], dtype=torch.int32), pages)
    chain = cm._chain_keys(torch.tensor(ids13, dtype=torch.int32))
    cm._tier_offer([VictimPath(chain[7], 8, tuple(chain[:8]), pages[:8], None)])

    held = len(cm.free_slots)                    # 53: the offer only read bytes
    mr = cm.match_req(_pend(ids13 + [99]))
    assert mr.cuda_handle.cached_len == 11       # the deeper tree match serves the request
    assert len(cm.free_slots) == held            # nothing taken
    assert cm._tier_pending_restore == {}
    hit = cm.tier_store.probe(chain)
    assert hit is not None and hit[0] == 8       # the store segment is intact, just unused


def test_kvonly_restore_units_at_page_size_gt_1():
    """ps>1 units: probe depth is PAGES, cached_len TOKENS; a partially tree-owned span
    restores only the missing page. Pins both the plain-path wiring and the units fix
    (pre-fix the gate compared tokens to pages and refused the restore)."""
    from freetoken.kvcache.hybrid_radix_cache import VictimPath
    from freetoken.scheduler.cache import SessionTierCfg

    kvpool = _FakeKVPool(pages=16, page_size=4)
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(16, 4, pt, "radix", swa_pool=kvpool,
                      session_tier_cfg=SessionTierCfg(ram_bytes=1 << 22))
    ids = list(range(1, 9))                      # 8 tokens = 2 pages
    cm.match_req(_pend([1, 2, 3, 4, 5]))         # cold admission tracks the session
    pages = cm._allocate(2)                      # page bases 0 and 4
    pt[0, :4] = pages[0]
    pt[0, 4:8] = pages[1]
    for pg in range(2):
        kvpool._kv_buffer[:, :, pg] = float(pg) * 1.5
    originals = [_kv_bytes(kvpool, pg) for pg in range(2)]
    cm.prefix_cache.insert_prefix(torch.tensor(ids[:4], dtype=torch.int32), pt[0, :4])
    chain = cm._chain_keys(torch.tensor(ids, dtype=torch.int32))
    cm._tier_offer([VictimPath(chain[-1], 8, tuple(chain), pt[0, :8], None)])
    cm._free(pt[0, 4:8])                         # the offer covered both pages

    mr = cm.match_req(_pend(ids + [99]))
    assert mr.cuda_handle.cached_len == 8        # 2 pages, not 2 tokens
    matched = mr.cuda_handle.get_matched_indices()
    assert matched[:4].tolist() == pt[0, :4].tolist()    # tree-owned page untouched
    assert _kv_bytes(kvpool, int(matched[4]) // 4) == originals[1]
    assert len(cm.free_slots) == 16 - 1 - 1      # one fresh page taken

    cm.lock(mr.cuda_handle)
    pt[0, :8] = matched
    req = Req(input_ids=torch.tensor(ids + [99], dtype=torch.int32), table_idx=0,
              cached_len=8, output_len=1, uid=0, sampling_params=SamplingParams(),
              cache_handle=mr.cuda_handle)
    cm.cache_req(req, finished=True)
    cm.check_integrity()
    m = cm.prefix_cache.match_prefix(torch.tensor(ids, dtype=torch.int32))
    assert m.cuda_handle.cached_len == 8


def test_kvonly_off_mode_untouched():
    """No tier cfg (or a disabled cfg) on a plain radix manager: zero new behavior - no
    store, no admission bookkeeping, plain evict returns pages exactly as today."""
    from freetoken.scheduler.cache import SessionTierCfg

    for cfg in (None, SessionTierCfg()):
        kvpool = _FakeKVPool()
        pt = torch.zeros(4, 64, dtype=torch.int32)
        cm = CacheManager(64, 1, pt, "radix", swa_pool=kvpool, session_tier_cfg=cfg)
        assert cm.tier_store is None and not cm._tier_on and cm._page_bytes is None
        mr = cm.match_req(_pend([1, 2, 3]))
        assert mr.cuda_handle.cached_len == 0
        assert cm._tier_sessions == {}           # no admission bookkeeping
        pages = cm._allocate(4)
        cm.prefix_cache.insert_prefix(torch.tensor([1, 2, 3, 4], dtype=torch.int32), pages)
        evicted = cm.prefix_cache.evict(1)       # plain evict: no derive seam
        cm._free(evicted[::cm.page_size])
        assert len(cm.free_slots) == 64          # full ledger, no store touched


def test_kvonly_plain_pool_codec_roundtrip():
    """The plain MHAKV family (kv + fp8-style scale sidecar) byte-encodes through the
    same _KVPageBytes and activates a KV-only manager. Fails-before by construction: the
    plain manager never activated, so the scale-bearing pool had no tier there."""
    from freetoken.scheduler.cache import _KVPageBytes, SessionTierCfg

    class _FakeMHAScale:
        swa_paged = False

        def __init__(self):
            self._kv_buffer = torch.zeros(2, 2, 16, 4, 2, 4, dtype=torch.bfloat16)
            self._scale_buffer = torch.zeros(2, 2, 16 * 4, 2, dtype=torch.float32)
            self._block_scale_buffer = None

    kvpool = _FakeMHAScale()
    codec = _KVPageBytes(kvpool)
    n = len(codec.read_page(3))
    blob = bytes((i * 29 + 7) % 256 for i in range(n))
    codec.write_page(3, blob)
    assert codec.read_page(3) == blob

    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(16, 4, pt, "radix", swa_pool=kvpool,
                      session_tier_cfg=SessionTierCfg(ram_bytes=1 << 22))
    assert cm._tier_on and cm._page_bytes is not None


def test_kvonly_shutdown_offers_live_boundaries():
    """The graceful-shutdown seam offers the KV-only tree's live tip per tracked session
    (the hybrid seam offers its snapshot boundaries); without it the tip dies with the
    process and the post-reboot first turn falls back to a full re-prefill. Fails-before:
    the plain path never reached the shutdown offer (store stayed empty)."""
    kvpool = _FakeKVPool()
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = _kvonly_cm(kvpool, pt)
    ids = list(range(1, 9))
    cm.match_req(_pend(ids))                     # track the session (cold)
    pages, originals = _fill_pages(cm, kvpool, pt, ids)
    cm.prefix_cache.insert_prefix(torch.tensor(ids, dtype=torch.int32), pages)

    assert cm.shutdown_tier() == 0               # L1-only store: nothing to flush to L2
    chain8 = cm._chain_keys(torch.tensor(ids, dtype=torch.int32))
    held = {seg.path_key for seg in cm.tier_store._segments.values()}
    assert held == {chain8[-1]}                  # the live tip was offered before the flush
    hit = cm.tier_store.probe(chain8)
    assert hit is not None and hit[0] == 8
    got, snaps = cm.tier_store.restore(hit[1])
    assert got == [originals[int(pg)] for pg in pages] and snaps == []


# ------------------------------------------------------- idle restore prefetch (phase 2)


def _settle_tier(cm, timeout=5.0):
    import time as _time

    deadline = _time.monotonic() + timeout
    store = cm.tier_store
    while not store._tickets_settled():
        assert _time.monotonic() < deadline, "prefetch worker did not settle"
        _time.sleep(0.001)


def _washed_hybrid_cm():
    """Donated tip washed out of the tree: tree empty, store warm, session tracked with
    its last seen stripped ids [1, 2, 3, 4]."""
    pool, kvpool = _pool(), _FakeKVPool()
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = _tiered_cm(pool, kvpool, pt)
    original, donated, _req = _donate_one_snapshot(cm, pool, kvpool, pt)
    cm.ensure_mamba_slots(pool.num_slots)            # tiered eviction: the tip goes
    assert cm.prefix_cache.full_evictable == 0
    return cm, pool, kvpool, original


def test_prefetch_idle_stages_and_adopt_skips_sync_read():
    """The idle hook stages the tracked session's tip (bytes + reservation); the next
    admission ADOPTS the staging through the shared restore tail: byte-exact pages, the
    RESERVED slot rides the match, and the sync store read never runs. Fails-before:
    no prefetch seam existed (cached_len came from the sync restore instead)."""
    cm, pool, kvpool, original = _washed_hybrid_cm()
    key = cm._chain_keys(torch.tensor([1, 2, 3, 4], dtype=torch.int32))[0]
    free_before, slots_before = len(cm.free_slots), pool.num_free_slots
    cm.prefetch_tier_idle()
    entry = cm._tier_prefetch[key]
    assert entry is not None and entry.res_slot is not None
    _settle_tier(cm)
    assert entry.ticket.state == "ready"
    assert len(cm.free_slots) == free_before - 4     # reservation carved at prefetch START
    assert pool.num_free_slots == slots_before - 1   # the snapshot slot is reserved too
    assert cm._tier_reserved_pages == 4

    mr = cm.match_req(_pend([1, 2, 3, 4, 9]))        # admission adopts the staging
    assert mr.cuda_handle.cached_len == 4
    assert mr.mamba_value == entry.res_slot          # the RESERVED slot, not a fresh alloc
    matched = mr.cuda_handle.get_matched_indices()
    assert matched.tolist() == [0, 1, 2, 3]          # the reserved head pages
    for j in range(4):
        assert kvpool._kv_buffer[:, :, j].contiguous().cpu() \
            .view(torch.uint8).numpy().tobytes() == original[100 + j]
    snap = cm.tier_store.snapshot()
    assert snap["prefetch_adopt"] == 1
    assert snap["restore_l1"] == 0 and snap["restore_l2"] == 0   # the sync read never ran
    assert cm._tier_prefetch == {} and cm._tier_reserved_pages == 0
    assert len(cm.free_slots) == free_before - 4     # converted, not re-spent
    assert pool.num_free_slots == slots_before - 1
    assert mr.cuda_handle.node.is_root()


def test_prefetch_adopt_then_abandon_returns_pages_and_slot():
    """abandon_restore on an ADOPTED match returns the reserved pages and slot exactly
    like the sync path's B4 pin - and never double-returns (the entry was consumed)."""
    cm, pool, kvpool, _original = _washed_hybrid_cm()
    cm.prefetch_tier_idle()
    _settle_tier(cm)
    free_staged, slots_staged = len(cm.free_slots), pool.num_free_slots
    mr = cm.match_req(_pend([1, 2, 3, 4, 9]))
    assert mr.cuda_handle.cached_len == 4            # adopted
    cm.abandon_restore(mr.cuda_handle)               # admission refused
    assert len(cm.free_slots) == free_staged + 4
    assert pool.num_free_slots == slots_staged + 1
    assert cm._tier_prefetch == {} and cm._tier_reserved_pages == 0
    assert cm._tier_pending_restore == {}
    cm.abandon_restore(mr.cuda_handle)               # idempotent
    assert len(cm.free_slots) == free_staged + 4 and pool.num_free_slots == slots_staged + 1


def test_prefetch_reservation_returned_on_direct_drop():
    """Dropping a ticket without an admission (supersede/reap/shutdown path) returns the
    whole reservation and frees the staging - nothing consumed."""
    kvpool = _FakeKVPool()
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = _kvonly_cm(kvpool, pt)
    ids = list(range(1, 13))
    cm.match_req(_pend(ids + [99]))                  # seen ids cover the full 12-page tip
    pages, _originals = _fill_pages(cm, kvpool, pt, ids)
    cm.prefix_cache.insert_prefix(torch.tensor(ids, dtype=torch.int32), pages)
    evicted, victims = cm.prefix_cache.evict_paths(1)
    cm._free(evicted[::cm.page_size])
    cm._tier_offer(victims)                          # wash: tree empty, store warm
    key = cm._chain_keys(torch.tensor(ids, dtype=torch.int32))[0]
    cm.prefetch_tier_idle()
    _settle_tier(cm)
    assert key in cm._tier_prefetch
    assert len(cm.free_slots) == 64 - 12 and cm._tier_reserved_pages == 12
    cm._tier_prefetch_drop(key)
    assert len(cm.free_slots) == 64 and cm._tier_reserved_pages == 0
    assert cm._tier_prefetch == {} and cm.tier_store._tickets == {}
    assert cm.tier_store.snapshot()["prefetch_abandon"] == 1


def test_prefetch_kvonly_adopt_byte_exact_and_ledger_balanced():
    """KV-only adopt: staged pages land byte-exact, the handle feeds the normal insert
    path, and the exact idle ledger stays balanced WHILE the reservation is live (the
    reserved pages sit outside free_slots until adopt)."""
    kvpool = _FakeKVPool()
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = _kvonly_cm(kvpool, pt)
    ids = list(range(1, 13))
    cm.match_req(_pend(ids + [99]))
    pages, originals = _fill_pages(cm, kvpool, pt, ids)
    cm.prefix_cache.insert_prefix(torch.tensor(ids, dtype=torch.int32), pages)
    evicted, victims = cm.prefix_cache.evict_paths(1)
    cm._free(evicted[::cm.page_size])
    cm._tier_offer(victims)
    cm.prefetch_tier_idle()
    _settle_tier(cm)
    assert len(cm._tier_prefetch) == 1 and cm._tier_reserved_pages == 12
    cm.check_integrity()                             # the ledger term carries the reservation

    mr = cm.match_req(_pend(ids + [99]))
    assert mr.cuda_handle.cached_len == 12 and mr.mamba_value is None
    matched = mr.cuda_handle.get_matched_indices()
    for j in range(12):
        assert _kv_bytes(kvpool, int(matched[j])) == originals[int(pages[j])]
    snap = cm.tier_store.snapshot()
    assert snap["prefetch_adopt"] == 1
    assert snap["restore_l1"] == 0 and snap["restore_l2"] == 0
    assert len(cm.free_slots) == 52   # the reservation was converted, not re-spent
    assert cm._tier_reserved_pages == 0 and cm._tier_prefetch == {}

    cm.lock(mr.cuda_handle)
    pt[0, :12] = matched
    req = Req(input_ids=torch.tensor(ids + [99], dtype=torch.int32), table_idx=0,
              cached_len=12, output_len=1, uid=0, sampling_params=SamplingParams(),
              cache_handle=mr.cuda_handle)
    cm.cache_req(req, finished=True)
    cm.check_integrity()                             # balanced after the adoption too
    m = cm.prefix_cache.match_prefix(torch.tensor(ids, dtype=torch.int32))
    assert m.cuda_handle.cached_len == 12            # the tree re-grew over the restored span


def test_prefetch_reoffer_supersedes_ticket():
    """A same-prefix re-offer supersedes a live staging: ticket dropped, staging freed,
    reservation returned, nothing consumed."""
    from freetoken.kvcache.hybrid_radix_cache import VictimPath

    cm, pool, kvpool, _original = _washed_hybrid_cm()
    key = cm._chain_keys(torch.tensor([1, 2, 3, 4], dtype=torch.int32))[0]
    cm.prefetch_tier_idle()
    _settle_tier(cm)
    assert key in cm._tier_prefetch
    free_staged, slots_staged = len(cm.free_slots), pool.num_free_slots
    chain4 = cm._chain_keys(torch.tensor([1, 2, 3, 4], dtype=torch.int32))
    vp = VictimPath(chain4[-1], 4, tuple(chain4), torch.arange(4, dtype=torch.int32), None)
    cm._tier_offer([vp])                             # same prefix re-offered
    assert cm._tier_prefetch == {} and cm._tier_reserved_pages == 0
    assert len(cm.free_slots) == free_staged + 4 and pool.num_free_slots == slots_staged + 1
    assert cm.tier_store._tickets == {}
    # the next idle re-stages from the refreshed segment
    cm.prefetch_tier_idle()
    _settle_tier(cm)
    assert key in cm._tier_prefetch


def test_prefetch_skipped_when_tree_owns_the_tip():
    """No speculative waste: while the tree still owns the boundary (the next turn would
    be a tree hit, not a restore) the idle hook stages nothing and reserves nothing."""
    pool, kvpool = _pool(), _FakeKVPool()
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = _tiered_cm(pool, kvpool, pt)
    _donate_one_snapshot(cm, pool, kvpool, pt)       # tip is LIVE in the tree
    free_before, slots_before = len(cm.free_slots), pool.num_free_slots
    cm.prefetch_tier_idle()
    assert cm._tier_prefetch == {} and cm._tier_reserved_pages == 0
    assert len(cm.free_slots) == free_before and pool.num_free_slots == slots_before
    assert cm.tier_store.snapshot()["prefetch_begin"] == 0


def test_prefetch_starvation_guard_skips_and_off_mode_noops():
    """The reserve guard keeps at least half of the then-free pages for live allocation
    (a speculative staging can never pinch admission to zero); the off-mode hook is an
    early return with zero behavior change."""
    cm, pool, kvpool, _original = _washed_hybrid_cm()
    saved = cm.free_slots
    cm.free_slots = saved[:6]                        # fresh(4) * 2 > 6: guard trips
    cm.prefetch_tier_idle()
    assert cm._tier_prefetch == {} and cm._tier_reserved_pages == 0
    assert len(cm.free_slots) == 6
    cm.free_slots = saved

    cm2 = CacheManager(64, 1, torch.zeros(4, 64, dtype=torch.int32), "hybrid_radix",
                       linear_state_pool=_pool(), swa_pool=_FakeKVPool(),
                       session_tier_cfg=None)
    assert cm2._tier_on is False
    cm2.prefetch_tier_idle()                         # a no-op, never touches anything


# ------------------------------------------- review findings F1-F6 (async prefetch TP)


def test_kv_page_bytes_refreshed_after_pool_rebuild():
    """F1 fails-before: a runtime KV resize reallocates the pool's tensors (engine
    rebuild_from_config -> CacheManager.rebuild); the codec captured _scale_buffer refs at
    init, so pre-fix restore wrote the scale rows into the DEAD tensor while the live pool
    served zeros, and a post-rebuild offer read stale scale bytes. The rebuild must refresh
    _page_bytes."""
    from freetoken.scheduler.cache import SessionTierCfg

    class _FakeMHAScalePool:
        swa_paged = False

        def __init__(self, pages=8):
            self._kv_buffer = torch.zeros(2, 2, pages, 1, 2, 4, dtype=torch.bfloat16)
            self._scale_buffer = torch.zeros(2, 2, pages, 2, dtype=torch.float32)

    kvpool = _FakeMHAScalePool(pages=8)
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, pt, "radix", swa_pool=kvpool,
                      session_tier_cfg=SessionTierCfg(ram_bytes=1 << 22))
    ids = list(range(1, 5))
    cm.match_req(_pend(ids + [9]))               # track the session
    got = cm._allocate(4)
    pt[0, :4] = got
    for pg in got.tolist():
        kvpool._kv_buffer[:, :, int(pg)] = float(pg) * 1.5
        kvpool._scale_buffer[:, :, int(pg)] = float(pg) + 0.25
    old_kv, old_scale = kvpool._kv_buffer.clone(), kvpool._scale_buffer.clone()
    cm.prefix_cache.insert_prefix(torch.tensor(ids, dtype=torch.int32), got)
    evicted, victims = cm.prefix_cache.evict_paths(1)
    cm._free(evicted[::cm.page_size])
    cm._tier_offer(victims)                      # bytes read through the PRE-rebuild tensors

    # engine-style runtime resize: the pool swaps in NEW tensors (same shapes)
    kvpool._kv_buffer = torch.zeros_like(kvpool._kv_buffer)
    kvpool._scale_buffer = torch.zeros_like(kvpool._scale_buffer)
    cm.rebuild(64, pt)

    mr = cm.match_req(_pend(ids + [9]))          # tree empty: the store restores
    assert mr.cuda_handle.cached_len == 4
    matched = mr.cuda_handle.get_matched_indices()
    for j, pg in enumerate(got.tolist()):
        pg = int(pg)
        assert torch.equal(kvpool._kv_buffer[:, :, int(matched[j])], old_kv[:, :, pg])
        # stale-ref corruption leaves the LIVE scale rows zero (written into the dead tensor)
        assert torch.equal(kvpool._scale_buffer[:, :, int(matched[j])], old_scale[:, :, pg])
    # and a post-rebuild OFFER reads the live tensors (not the stale refs)
    parts = [kvpool._kv_buffer[:, :, 5], kvpool._scale_buffer[:, :, 5]]
    direct = b"".join(p.contiguous().cpu().view(torch.uint8).numpy().tobytes() for p in parts)
    assert cm._page_bytes.read_page(5) == direct


def test_prefetch_reap_returns_reservation_after_discard(tmp_path):
    """F2 fails-before: the store's _discard marks the staged ticket abandoned but the
    manager entry kept holding its reservation - the idle reap only matched "failed", so
    the reserved pages leaked forever. The reap must release both states."""
    from freetoken.scheduler.cache import SessionTierCfg

    kvpool = _FakeKVPool()
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, pt, "radix", swa_pool=kvpool,
                      session_tier_cfg=SessionTierCfg(ram_bytes=1 << 22,
                                                      dir=str(tmp_path / "tier")))
    ids = list(range(1, 13))
    cm.match_req(_pend(ids + [99]))
    pages, _originals = _fill_pages(cm, kvpool, pt, ids)
    cm.prefix_cache.insert_prefix(torch.tensor(ids, dtype=torch.int32), pages)
    evicted, victims = cm.prefix_cache.evict_paths(1)
    cm._free(evicted[::cm.page_size])
    cm._tier_offer(victims)
    key = cm._chain_keys(torch.tensor(ids, dtype=torch.int32))[0]
    cm.prefetch_tier_idle()
    _settle_tier(cm)
    assert len(cm.free_slots) == 52 and cm._tier_reserved_pages == 12

    assert cm.tier_store.evict_store(1) == 1     # demote the staged segment to L2
    assert cm.tier_store.evict_store(1) == 1     # true discard: the ticket is invalidated
    assert key in cm._tier_prefetch              # the entry still holds the reservation
    cm.prefetch_tier_idle()                      # the idle reap returns it (no re-offer)
    assert cm._tier_prefetch == {} and cm._tier_reserved_pages == 0
    assert len(cm.free_slots) == 64
    assert cm.tier_store._tickets == {}


def test_prefetch_walk_does_not_refresh_tree_lru():
    """F5 fails-before: the idle prefetch's owned_prefix walk re-stamped the tree's shared
    nodes on every idle pass (speculative interest outranking real admissions); with the
    no-stamp walk the node's timestamp is untouched by prefetch_tier_idle."""
    kvpool = _FakeKVPool()
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = _kvonly_cm(kvpool, pt)
    ids = list(range(1, 13))
    cm.match_req(_pend(ids + [99]))
    pages, _originals = _fill_pages(cm, kvpool, pt, ids)
    cm.prefix_cache.insert_prefix(torch.tensor(ids[:8], dtype=torch.int32), pages[:8])
    chain = cm._chain_keys(torch.tensor(ids, dtype=torch.int32))
    from freetoken.kvcache.hybrid_radix_cache import VictimPath

    cm._tier_offer([VictimPath(chain[-1], 12, tuple(chain), pages, None)])
    cm._free(pages[8:])
    node, owned, _ = cm.prefix_cache.owned_prefix(torch.tensor(ids, dtype=torch.int32))
    assert owned == 8
    before = node.timestamp
    cm.prefetch_tier_idle()                      # the speculative walk runs in here
    _settle_tier(cm)
    key = cm._chain_keys(torch.tensor(ids, dtype=torch.int32))[0]
    assert key in cm._tier_prefetch              # the walk ran (probe hit, entry staged)
    assert node.timestamp == before              # ...but did not refresh the node's LRU


def test_prefetch_adopt_surplus_when_tree_grew():
    """F6a: the tree GREW over the prefix between begin_restore and adopt - the adopt
    consumes only the fresh pages it needs and returns the surplus reservation."""
    kvpool = _FakeKVPool()
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = _kvonly_cm(kvpool, pt)
    ids = list(range(1, 13))
    cm.match_req(_pend(ids + [99]))
    pages, originals = _fill_pages(cm, kvpool, pt, ids)
    cm.prefix_cache.insert_prefix(torch.tensor(ids, dtype=torch.int32), pages)
    evicted, victims = cm.prefix_cache.evict_paths(1)
    cm._free(evicted[::cm.page_size])
    cm._tier_offer(victims)                      # wash: tree empty, store warm (boundary 12)
    key = cm._chain_keys(torch.tensor(ids, dtype=torch.int32))[0]
    cm.prefetch_tier_idle()
    _settle_tier(cm)
    assert len(cm.free_slots) == 52 and cm._tier_reserved_pages == 12

    tree_pages = cm._allocate(4)                 # the tree (re)grows over [0:4)
    cm.prefix_cache.insert_prefix(torch.tensor(ids[:4], dtype=torch.int32), tree_pages)

    mr = cm.match_req(_pend(ids + [99]))         # adopt: fresh 8 of the 12 reserved
    assert mr.cuda_handle.cached_len == 12
    matched = mr.cuda_handle.get_matched_indices()
    assert matched[:4].tolist() == tree_pages.tolist()    # tree-owned prefix rides
    for j in range(4, 12):
        assert _kv_bytes(kvpool, int(matched[j])) == originals[int(pages[j])]
    assert len(cm.free_slots) == 48 + 4          # 4 tree pages - 12 reserved + 4 surplus
    assert cm._tier_prefetch == {} and cm._tier_reserved_pages == 0
    assert cm.tier_store.snapshot()["prefetch_adopt"] == 1


def test_prefetch_topup_when_tree_shrank():
    """F6b: the tree SHRANK between begin_restore and adopt (its owned pages washed) - the
    adopt tops up the missing pages from the free list and stays byte-exact."""
    kvpool = _FakeKVPool()
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = _kvonly_cm(kvpool, pt)
    ids = list(range(1, 13))
    cm.match_req(_pend(ids + [99]))
    pages, originals = _fill_pages(cm, kvpool, pt, ids)
    cm.prefix_cache.insert_prefix(torch.tensor(ids[:4], dtype=torch.int32), pages[:4])
    chain = cm._chain_keys(torch.tensor(ids, dtype=torch.int32))
    from freetoken.kvcache.hybrid_radix_cache import VictimPath

    cm._tier_offer([VictimPath(chain[-1], 12, tuple(chain), pages, None)])
    cm._free(pages[4:])
    cm.prefetch_tier_idle()                      # fresh = 12 - 4 = 8 reserved
    _settle_tier(cm)
    assert cm._tier_reserved_pages == 8 and len(cm.free_slots) == 64 - 4 - 8

    evicted = cm.prefix_cache.evict(4)           # the tree shrinks back to empty
    cm._free(evicted[::cm.page_size])
    mr = cm.match_req(_pend(ids + [99]))         # adopt: fresh 12 > reserved 8 -> top up 4
    assert mr.cuda_handle.cached_len == 12
    matched = mr.cuda_handle.get_matched_indices()
    for j in range(12):
        assert _kv_bytes(kvpool, int(matched[j])) == originals[int(pages[j])]
    assert cm._tier_prefetch == {} and cm._tier_reserved_pages == 0
    assert len(cm.free_slots) == 64 - 4 - 8 - 4 + 4   # the 4 topped-up pages are spent


def test_tier_seen_ids_ring_cap():
    """F6c fails-before: the last-seen-ids ring holds at most 64 sessions; the oldest is
    evicted (and any prefetch entry for it released) instead of growing unbounded."""
    kvpool = _FakeKVPool()
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = _kvonly_cm(kvpool, pt)
    for i in range(66):
        cm.match_req(_pend([1000 + i, 1001 + i]))   # distinct first-page chain keys
    assert len(cm._tier_seen_ids) == 64
    oldest = cm._chain_keys(torch.tensor([1000, 1001], dtype=torch.int32))[0]
    assert oldest not in cm._tier_seen_ids
    newest = cm._chain_keys(torch.tensor([1064, 1065], dtype=torch.int32))[0]
    assert newest in cm._tier_seen_ids


def test_bug1_restored_decode_page_crossing_no_leak():
    """BUG-1 fails-before: a tier-restored request whose DECODE allocates a page beyond the
    restored span (finish cached_len non-aligned AND past the restored boundary + tail
    page). The non-aligned finish's tier-restored insert adopted [free_upto, insert_len)
    but the trailing free still started at the OLD restored boundary - the just-adopted
    span went back on the free list while tree-owned: free + cache == num_pages + 1 and
    the idle integrity check killed the scheduler (Qwen3.8, 62400-token restore, 64-token
    decode)."""
    from freetoken.scheduler.cache import SessionTierCfg

    pool, kvpool = _pool(), _FakeKVPool(pages=16, page_size=8)
    pt = torch.zeros(4, 128, dtype=torch.int32)
    cm = CacheManager(16, 8, pt, "hybrid_radix", linear_state_pool=pool, swa_pool=kvpool,
                      session_tier_cfg=SessionTierCfg(ram_bytes=1 << 22))
    ids = list(range(1, 10))                     # 8 aligned + 1 tail token (the prompt)
    mr = cm.match_req(_pend(ids))
    pages = cm._allocate(2)
    pt[0, :8] = pages[0]
    pt[0, 8:9] = int(pages[1])
    for pg in range(2):
        kvpool._kv_buffer[:, :, int(pages[pg]) // 8] = float(pg) * 1.5
    req = Req(input_ids=torch.tensor(ids + [99], dtype=torch.int32), table_idx=0, cached_len=9,
              output_len=1, uid=0, sampling_params=SamplingParams(), cache_handle=mr.cuda_handle)
    req.linear_slot_idx, req.mamba_ping_pong = pool.alloc(1)[0], tuple(pool.alloc(2))
    req.mamba_next_track_idx = 1
    req.mamba_last_track_seqlen = 8
    cm.lock(mr.cuda_handle)
    cm.cache_req(req, finished=False)
    cm.cache_req(req, finished=True)
    cm.ensure_mamba_slots(pool.num_slots)        # wash: store warm at boundary 8
    chain = cm._chain_keys(torch.tensor(ids[:8], dtype=torch.int32))
    assert cm.tier_store.probe(chain) is not None

    mr2 = cm.match_req(_pend(ids))               # restored admission: boundary 8 restored
    assert mr2.cuda_handle.cached_len == 8 and mr2.mamba_value is not None
    restored_pages = mr2.cuda_handle.get_matched_indices().clone()
    # the prompt tail (1 token -> page 1) + decode crossing into page 2: finish at 17
    ids17 = ids + [90 + i for i in range(9)]     # 17 tokens of device content
    req2 = Req(input_ids=torch.tensor(ids17 + [99], dtype=torch.int32), table_idx=0, cached_len=17,
               output_len=9, uid=1, sampling_params=SamplingParams(),
               cache_handle=mr2.cuda_handle)
    req2.linear_slot_idx = pool.alloc(1)[0]
    req2.mamba_ping_pong = tuple(pool.alloc(2))
    cm.lock(mr2.cuda_handle)
    pt[0, :8] = restored_pages                   # the restored span (prefill.py)
    pt[0, 8:9] = int(cm._allocate(1)[0])         # the prompt-tail page
    pt[0, 9:10] = int(cm._allocate(1)[0])        # the DECODE page (the crossing)
    cm.cache_req(req2, finished=True)
    cm.check_integrity()                         # pre-fix: free + cache == num_pages + 1
    assert cm.prefix_cache.full_evictable == 16  # restored page + tail page tree-owned
    assert len(cm.free_slots) == 14              # 16 pages - the 2 tree-owned (16 tokens)
    m = cm.prefix_cache.match_prefix(torch.tensor(ids[:16], dtype=torch.int32))
    assert m.cached_len == 0                     # KV-only node: reuse goes via the store
    _, owned, _ = cm.prefix_cache.owned_prefix(torch.tensor(ids17[:16], dtype=torch.int32))
    assert owned == 16                           # the tree owns the adopted span


def test_bug3_refused_restore_leaves_tree_ledger_and_store_intact():
    """BUG-3 fails-before probe: resume-1-A restores fine; resume-1-B probe-HITs the store
    (shared system prefix) but its restore is REFUSED (fresh > free). The refusal must not
    disturb A's tree span, the ledger, or the store; B's subsequent admission may wash A
    under pressure, but only through the offer seam (A stays recoverable)."""
    from freetoken.kvcache.hybrid_radix_cache import VictimPath
    from freetoken.scheduler.cache import SessionTierCfg

    pool, kvpool = _pool(num_slots=8), _FakeKVPool(pages=16)
    pt = torch.zeros(4, 16, dtype=torch.int32)
    cm = CacheManager(16, 1, pt, "hybrid_radix", linear_state_pool=pool, swa_pool=kvpool,
                      session_tier_cfg=SessionTierCfg(ram_bytes=1 << 22))
    A = [1, 2, 3, 4, 5, 6, 7, 8]                 # 8 pages
    B = [1, 2, 3, 4] + [9] * 12                  # 16 pages sharing A's [1..4] prefix

    # fill: A's tip donated + washed into the store (snapshot bound 8)
    mr_a = cm.match_req(_pend(A + [50]))
    pages_a = cm._allocate(8)
    pt[0, :8] = pages_a
    req_a = Req(input_ids=torch.tensor(A + [50], dtype=torch.int32), table_idx=0, cached_len=8,
                output_len=1, uid=0, sampling_params=SamplingParams(), cache_handle=mr_a.cuda_handle)
    req_a.linear_slot_idx, req_a.mamba_ping_pong = pool.alloc(1)[0], tuple(pool.alloc(2))
    req_a.mamba_next_track_idx = 1
    req_a.mamba_last_track_seqlen = 8
    cm.lock(mr_a.cuda_handle)
    cm.cache_req(req_a, finished=False)
    cm.cache_req(req_a, finished=True)
    cm.ensure_mamba_slots(pool.num_slots)        # wash A: offered with its snapshot
    chain_a = cm._chain_keys(torch.tensor(A, dtype=torch.int32))
    assert cm.tier_store.probe(chain_a) is not None
    assert cm.prefix_cache.full_evictable == 0   # tree empty (post-reboot equivalent)

    # B's tip into the store too (as the fill left it): tracked + offered with a live slot
    cm.match_req(_pend(B + [51]))                # tracks B (st[0] = 16)
    pages_b = cm._allocate(16)
    slot_b = pool.alloc(1)[0]
    chain_b = cm._chain_keys(torch.tensor(B, dtype=torch.int32))
    cm._tier_offer([VictimPath(chain_b[-1], 16, tuple(chain_b), pages_b, slot_b)])
    cm._free(pages_b)
    pool.free([slot_b])
    base_free = len(cm.free_slots)

    # resume-1-A: restore succeeds, A back in the tree
    mr2 = cm.match_req(_pend(A + [50]))
    assert mr2.cuda_handle.cached_len == 8 and mr2.mamba_value is not None
    restored = mr2.cuda_handle.get_matched_indices().clone()
    req2 = Req(input_ids=torch.tensor(A + [50], dtype=torch.int32), table_idx=0, cached_len=8,
               output_len=1, uid=1, sampling_params=SamplingParams(), cache_handle=mr2.cuda_handle)
    req2.linear_slot_idx = pool.alloc(1)[0]
    req2.mamba_ping_pong = tuple(pool.alloc(2))
    cm.lock(mr2.cuda_handle)
    pt[0, :8] = restored
    cm.cache_req(req2, finished=True)            # A finishes: span adopted, unlocked
    assert cm.prefix_cache.full_evictable == 8

    # resume-1-B: probe HIT at depth 16, owned 4, fresh 12 > free 8 -> the free-page refusal
    free_before = len(cm.free_slots)
    mr_b = cm.match_req(_pend(B + [51]))         # the refusal under test
    hit = cm.tier_store.probe(chain_b)
    assert hit is not None and hit[0] == 16      # the probe DID hit (precondition)
    assert mr_b.cuda_handle.cached_len == 0      # refused: the normal path took over
    assert cm.prefix_cache.match_prefix(torch.tensor(A, dtype=torch.int32)).cached_len == 8
    cm.check_integrity()                         # ledger untouched by the refusal
    assert len(cm.free_slots) == free_before
    assert cm._tier_pending_restore == {}

    # B's admission washes A under pressure - only via the offer seam: A stays recoverable
    cm.ensure_mamba_slots(pool.num_slots)        # the arm's admission pressure
    cm.check_integrity()
    hit_a = cm.tier_store.probe(chain_a)
    assert hit_a is not None and hit_a[0] == 8   # A recoverable from the store
    got, snaps = cm.tier_store.restore(hit_a[1])
    assert len(got) == 8 and len(snaps) == 1     # A's span + snapshot intact in the store


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: PASS")
