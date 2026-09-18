"""P2b integration: CacheManager hybrid path (match_req -> cache_req donate -> prefix hit).
CPU, real LinearStatePool + page_table, hand-built Reqs. Exercises the two-currency wiring
without the full scheduler/engine."""
from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.core import Req, SamplingParams
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.scheduler.cache import CacheManager


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


def test_prefill_continuation_forwards_mamba_last_track_seqlen():
    """A continuation Req must inherit the x64 boundary its previous chunk tracked: the Req
    that finishes prefill is the one whose L drives the finish-frozen donate, so a dropped
    field leaves every multi-chunk prompt with a short final chunk unable to donate."""
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
    pending.chunked_req = chunk1

    adder = PrefillAdder(token_budget=100, reserved_size=0, cache_manager=cm, table_manager=tm)
    req2 = adder.try_add_one(pending)
    assert req2 is not None
    assert req2.mamba_last_track_seqlen == 64
    assert req2.cached_len == 64

    # chain integration: the Req that finishes prefill carries the forwarded L into its finish
    # commit, so the frozen donate lands at the x64 boundary and a repeat request hits it
    cm.allocate_paged([req2])
    req2.complete_one()                            # cached_len 64 -> 128
    free_before = pool.num_free_slots
    cm.cache_req(req2, finished=True)
    assert pool.num_free_slots == free_before + 1  # pp[1] freed; tree owns pp[0] and the live slot

    mr = cm.match_req(_pend(list(range(64)) + [99]))
    assert mr.cuda_handle.cached_len == 64
    assert mr.mamba_value == chunk1.mamba_ping_pong[0]


def test_naive_cache_does_not_align_prefill_chunks():
    """The alignment hook is hybrid-only; every other cache keeps the raw budget chunk."""
    from freetoken.scheduler.prefill import PrefillAdder
    from freetoken.scheduler.table import TableManager
    from freetoken.scheduler.utils import PendingReq

    pt = torch.zeros(4, 512, dtype=torch.int32)
    cm = CacheManager(64, 64, pt, "radix")
    assert cm.prefill_chunk_align == 1
    tm = TableManager(max_running_reqs=4, page_table=pt)
    adder = PrefillAdder(token_budget=100, reserved_size=0, cache_manager=cm, table_manager=tm)
    pending = PendingReq(0, torch.arange(300, dtype=torch.int32), SamplingParams(max_tokens=1))
    assert adder.try_add_one(pending).extend_len == 100


def test_pool_sizing_covers_4mr_floor():
    """C6: pool must reserve the 4-slot-per-request non-evictable floor even at a tiny ratio."""
    from types import SimpleNamespace
    from freetoken.kvcache.linear_state_pool import _linear_pool_num_slots
    for mr in (1, 8, 64):
        c = SimpleNamespace(max_running_req=mr, cache_type="hybrid_radix",
                            linear_state_cache_ratio=0.1)
        assert _linear_pool_num_slots(c) >= 4 * mr + 1, (mr, _linear_pool_num_slots(c))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: PASS")
