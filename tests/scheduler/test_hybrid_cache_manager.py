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

    cm._tier_offer([vp(1, 4)])                   # tip
    cm._tier_offer([vp(2, 2)])                   # deepest under-divergence boundary
    cm._tier_offer([vp(1, 1)])                   # stale intermediate grid point
    held = {seg.path_key for seg in cm.tier_store._segments.values()}
    assert held == {k[0], k[1]}                  # exactly the adaptive set


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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: PASS")
