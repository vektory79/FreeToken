"""Abort vs overlap-scheduling races.

Under overlap scheduling a batch launches one iteration before _process_last_data drains
it, and an abort message is processed in between. Freeing the request's resources inside
the abort handler while its forward is in flight used to corrupt state: the hybrid
prefill-commit dereferenced the None'd GDN ping-pong slots (TypeError killed the
scheduler); plain radix would silently re-read the freed page-table row.

The scheduler now uses the SGLang-style single-owner design: the abort handler frees
immediately ONLY when the request has no forward in flight (not in ``self._last_data``'s
batch); otherwise it just sets ``req.aborted`` and _process_last_data frees the request
when the batch drains, after copy_done.synchronize(). A ``table_idx != -1`` sentinel on
the prefix-commit remains as defense-in-depth.

Tests drive the real (unbound) Scheduler methods against CPU-built hybrid managers.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.core import Batch, Req, SamplingParams
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.message import AbortBackendMsg, DetokenizeMsg
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.decode import DecodeManager
from freetoken.scheduler.prefill import ChunkedReq, PrefillManager
from freetoken.scheduler.scheduler import Scheduler
from freetoken.scheduler.status import RequestTimingTracker
from freetoken.scheduler.table import TableManager
from freetoken.scheduler.utils import PendingReq

UID = 2


def _pool(num_slots=16):
    g = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate="silu",
    )
    return LinearStatePool(group=g, num_slots=num_slots, dtype=torch.bfloat16,
                           device=torch.device("cpu"), tp_size=1)


def _setup():
    """Hybrid managers + a stub Scheduler `self` for the real unbound methods."""
    pool = _pool()
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, pt, "hybrid_radix", linear_state_pool=pool)
    tm = TableManager(max_running_reqs=4, page_table=pt)
    dm = DecodeManager(page_size=1)
    pm = PrefillManager(cm, tm, dm)
    sent = []
    stub = SimpleNamespace(
        cache_manager=cm,
        table_manager=tm,
        decode_manager=dm,
        prefill_manager=pm,
        finished_reqs=set(),
        eos_token_ids=set(),
        toolcall_anchor_id=None,
        config=SimpleNamespace(page_size=1),
        status_reporter=SimpleNamespace(report_batch=lambda *_, **__: None),
        # frozen clock: drains measure a 0 ms forward, so the reporter receives no timing
        timing_tracker=RequestTimingTracker(clock=lambda: 0.0),
        send_result=sent.extend,
        _kv_usage_pages=cm.page_usage,
        _mamba_slot_usage=lambda: None,
        _swa_token_usage=lambda: None,
        _gpu_mem_bytes=lambda: 0,
        _match_stop_str=lambda _req: None,
        _pending_abort_acks=set(),
        _last_data=None,
    )
    stub._free_req_resources = lambda req: Scheduler._free_req_resources(stub, req)
    return pool, cm, tm, dm, pm, sent, stub


def _launch_req(pool, cm, tm, prompt, *, cls=Req, track_seqlen=None):
    """A launched (forward in flight) hybrid req: handle locked, pages allocated,
    GDN slots held, cached_len advanced -- the state _process_last_data will drain."""
    mr = cm.match_req(SimpleNamespace(input_ids=prompt, input_len=len(prompt),
                                      ))
    req = cls(input_ids=prompt, table_idx=tm.allocate(), cached_len=0, output_len=4,
              uid=UID, sampling_params=SamplingParams(max_tokens=4),
              cache_handle=mr.cuda_handle)
    req.linear_slot_idx = pool.alloc(1)[0]
    req.mamba_ping_pong = tuple(pool.alloc(2))
    req.mamba_next_track_idx = 1
    cm.lock(mr.cuda_handle)
    cm.allocate_paged([req])
    req.complete_one()
    req.mamba_last_track_seqlen = track_seqlen
    return req


def _as_last_data(batch):
    # (ForwardInput-ish, ForwardOutput-ish, launch clock): the launch timestamp is the
    # third element the drain unpacks since the measured-forward-duration change.
    return (
        SimpleNamespace(batch=batch),
        (None, torch.tensor([42], dtype=torch.int32),
         SimpleNamespace(synchronize=lambda: None)),
        0.0,
    )


def test_abort_inflight_final_chunk_marks_then_drains():
    """Abort while the final prefill chunk (plain Req, already in running_reqs) is in
    flight: the handler only marks; the same iteration's drain frees exactly once."""
    pool, cm, tm, dm, _pm, sent, stub = _setup()
    req = _launch_req(pool, cm, tm, torch.arange(1, 13, dtype=torch.int32),
                      track_seqlen=8)
    batch = Batch(reqs=[req], phase="prefill")
    dm.filter_reqs(batch.reqs)                  # _forward: joins running_reqs at launch
    stub._last_data = _as_last_data(batch)      # overlap_loop exposes the un-drained batch

    Scheduler._process_one_msg(stub, AbortBackendMsg(uid=UID))
    assert req.aborted and req.table_idx != -1  # marked, NOT freed under the forward
    assert req.mamba_ping_pong is not None
    assert req not in dm.running_reqs
    assert UID in stub._pending_abort_acks
    free_after_mark = pool.num_free_slots

    Scheduler._process_last_data(stub, stub._last_data)
    assert req.table_idx == -1                  # freed at the drain point
    assert pool.num_free_slots > free_after_mark
    assert req in stub.finished_reqs
    assert sent == []                           # no DetokenizeMsg: abort ack stays terminal
    match_ids = torch.cat([req.input_ids[:8], torch.tensor([99], dtype=torch.int32)])
    mr = cm.match_req(SimpleNamespace(input_ids=match_ids, input_len=match_ids.numel()))
    assert mr.cuda_handle.cached_len == 8       # the frozen donate of the carried L fired at drain
    assert mr.mamba_value is not None
    cm.check_integrity()


def test_abort_inflight_intermediate_chunk_marks_then_drains():
    """Abort mid-chunked-prefill: prefill_manager.abort_req pops the pending continuation
    (no next chunk launches) and returns the in-flight ChunkedReq, which is marked and
    freed when its batch drains."""
    pool, cm, tm, _dm, pm, sent, stub = _setup()
    prompt = torch.arange(1, 13, dtype=torch.int32)
    chunk = _launch_req(pool, cm, tm, prompt[:8], cls=ChunkedReq)
    pending = PendingReq(uid=UID, input_ids=prompt,
                         sampling_params=SamplingParams(max_tokens=4))
    pending.chunked_req = chunk
    pm.pending_list = [pending]
    batch = Batch(reqs=[chunk], phase="prefill")
    stub._last_data = _as_last_data(batch)

    Scheduler._process_one_msg(stub, AbortBackendMsg(uid=UID))
    assert pm.pending_list == []                # continuation gone: no next chunk
    assert chunk.aborted and chunk.table_idx != -1

    Scheduler._process_last_data(stub, stub._last_data)
    assert chunk.table_idx == -1
    assert sent == []                           # chunks never reply
    cm.check_integrity()


def test_abort_starved_decode_req_frees_immediately():
    """A request with no forward in flight (e.g. a decode req starved behind a long
    chunked prefill) is freed by the abort handler right away -- deferring would leak
    until its next batch, which strict prefill-priority puts arbitrarily far away."""
    pool, cm, tm, dm, _pm, _sent, stub = _setup()
    req = _launch_req(pool, cm, tm, torch.arange(1, 13, dtype=torch.int32))
    dm.filter_reqs([req])
    # the un-drained batch belongs to some other request's prefill
    stub._last_data = (SimpleNamespace(batch=SimpleNamespace(reqs=[])), None)
    # mid-decode request with an accumulated timing entry (no drain will ever run for it)
    req.decode_batch_idx = 2
    stub.timing_tracker.on_decode_drained(SimpleNamespace(reqs=[req]), 100.0)
    base_free = pool.num_free_slots

    Scheduler._process_one_msg(stub, AbortBackendMsg(uid=UID))
    assert not req.aborted
    assert req.table_idx == -1                  # freed immediately, no drain needed
    assert stub.timing_tracker.pop(UID) == (0.0, 0.0, False)  # timing entry released too
    assert pool.num_free_slots > base_free
    assert req not in dm.running_reqs
    cm.check_integrity()


def test_prefix_commit_sentinel_guard():
    """Defense-in-depth: even if some future path frees a req early (bypassing the
    aborted mark), the finished=False prefix-commit must skip a freed req instead of
    dereferencing its None'd GDN slots (the original crash, cache.py _cache_req_hybrid)."""
    pool, cm, tm, dm, _pm, sent, stub = _setup()
    req = _launch_req(pool, cm, tm, torch.arange(1, 13, dtype=torch.int32),
                      track_seqlen=8)
    batch = Batch(reqs=[req], phase="prefill")
    dm.filter_reqs(batch.reqs)

    aborted = dm.abort_req(UID)
    assert aborted is req
    Scheduler._free_req_resources(stub, aborted)   # freed WITHOUT the aborted mark
    assert req.table_idx == -1 and req.mamba_ping_pong is None
    free_after_abort = pool.num_free_slots

    Scheduler._process_last_data(stub, _as_last_data(batch))  # pre-guard: TypeError

    assert pool.num_free_slots == free_after_abort  # nothing double-freed
    cm.check_integrity()
    assert [m.uid for m in sent] == [UID]  # un-marked path still publishes the token


def test_final_prefill_commit_clears_track_seqlen_and_donates():
    """The finished=False prefill commit at the drain (scheduler.py:398) must consume and
    clear the req's mamba_last_track_seqlen: snapshot_toolcall_anchor only fires when the
    mark is None at decode entry, so a leftover L would silently disable tool-call
    anchors. With L page-aligned and the ping-pong pair present, the chunk-commit
    donation must also land (frozen snapshot reusable at prefix L). Drives the real
    _process_last_data drain on the final-chunk state (plain Req in running_reqs);
    chunk admission itself is not needed to reach the commit. L=8 is a scaled stand-in,
    not a tracker-producible boundary: the tracker emits cached_len + k*64
    (linear.py:120-127)."""
    pool, cm, tm, dm, _pm, sent, stub = _setup()
    req = _launch_req(pool, cm, tm, torch.arange(1, 13, dtype=torch.int32),
                      track_seqlen=8)
    frozen = req.mamba_ping_pong[0]     # mamba_next_track_idx==1 -> the forward wrote pp[0]
    batch = Batch(reqs=[req], phase="prefill")
    dm.filter_reqs(batch.reqs)          # final-chunk state: plain Req in running_reqs

    free_before = pool.num_free_slots
    Scheduler._process_last_data(stub, _as_last_data(batch))

    assert req.mamba_last_track_seqlen is None, \
        "final prefill commit must clear the pending x64 track mark before decode"
    assert req.table_idx != -1 and req not in stub.finished_reqs  # finished=False path ran
    assert [m.finished for m in sent] == [False]
    # slot conservation: the donated frozen slot is tree-owned and replaced in the pair
    assert pool.num_free_slots == free_before - 1
    assert req.mamba_ping_pong[0] != frozen
    match_ids = torch.cat([req.input_ids[:8], torch.tensor([99], dtype=torch.int32)])
    mr = cm.match_req(SimpleNamespace(input_ids=match_ids, input_len=match_ids.numel()))
    assert mr.cuda_handle.cached_len == 8
    assert mr.mamba_value == frozen
    # KV-page conservation for the still-running req: the 8 committed pages are
    # tree-owned (the req's row aliases them); its cached_len-8 tail pages stay
    # req-exclusive. check_integrity is idle-only accounting, so assert the exact count.
    assert len(cm.free_slots) + 8 + (req.cached_len - 8) == cm.num_pages
    # RA1: the donated node must be lock-protected (cache.py:408 inc_lock), or evict_mamba
    # (candidate filter mamba_ref_count == 0) could reclaim it under the still-running req.
    assert cm.prefix_cache.full_protected == 8
    assert mr.cuda_handle.node.mamba_ref_count >= 1
    cm.prefix_cache.check_integrity()   # structural only: cm.check_integrity() sees tail pages


def test_post_terminal_overlap_step_is_dropped():
    """Overlap scheduling launches one more decode step for a request that already
    terminated (filter_reqs keeps it while output budget remains). The extra drain
    must not append its token, emit a second DetokenizeMsg, or free twice."""
    from freetoken.message import DetokenizeMsg

    pool, cm, tm, dm, _pm, sent, stub = _setup()
    stub.eos_token_ids = {42}  # the drained token (42) finishes the request by EOS
    req = _launch_req(pool, cm, tm, torch.arange(1, 13, dtype=torch.int32),
                      track_seqlen=8)
    dm.filter_reqs([req])

    Scheduler._process_last_data(stub, _as_last_data(Batch(reqs=[req], phase="prefill")))
    assert req in stub.finished_reqs and req.table_idx == -1
    terminal = [m for m in sent if isinstance(m, DetokenizeMsg)]
    assert len(terminal) == 1 and terminal[0].finished

    # The overlap extra step: the same req sits in the next batch's drain.
    output_len_before = req.output_len
    Scheduler._process_last_data(stub, _as_last_data(Batch(reqs=[req], phase="decode")))
    assert [m for m in sent if isinstance(m, DetokenizeMsg)] == terminal  # no 2nd msg
    assert req.output_len == output_len_before                           # no append
    cm.check_integrity()


def test_drain_attributes_forward_time_to_the_terminal_reply():
    """The final chunk's forward time lands in the terminal reply's timings: the tracker
    is fed BEFORE the terminal pop, so even a single-chunk prompt reports prefill_ms
    (pre-fix ordering popped first and the only chunk's time was lost)."""
    pool, cm, tm, dm, _pm, sent, stub = _setup()
    stub.timing_tracker = RequestTimingTracker(clock=lambda: 2.0)  # t_launch 0.0 -> 2000 ms
    # burn the engine-start warmup slot so the drained batch is a regular one
    stub.timing_tracker.on_prefill_drained(
        SimpleNamespace(log_new_tokens=1, log_req_new_tokens={999: 1}), 0.0)
    req = _launch_req(pool, cm, tm, torch.arange(1, 13, dtype=torch.int32))
    batch = Batch(reqs=[req], phase="prefill")
    batch.log_new_tokens = 12
    batch.log_req_new_tokens = {UID: 12}
    dm.filter_reqs(batch.reqs)
    stub.eos_token_ids = {42}  # the drained token (42) finishes the request by EOS

    Scheduler._process_last_data(stub, _as_last_data(batch))

    terminal = [m for m in sent if isinstance(m, DetokenizeMsg) and m.finished]
    assert len(terminal) == 1
    assert terminal[0].prefill_ms == pytest.approx(2000.0)
    assert terminal[0].prompt_warmup is False
    assert stub.timing_tracker.pop(UID) == (0.0, 0.0, False)  # entry released


def test_abort_drain_purges_tracker_entry_without_resurrection():
    """In-flight abort: attribution still sees the aborted req listed in the batch, so the
    pre-attribution pop alone resurrects the uid; the post-attribution purge must win and
    leave no tracker entry (no growth for aborted requests)."""
    pool, cm, tm, dm, _pm, sent, stub = _setup()
    stub.timing_tracker = RequestTimingTracker(clock=lambda: 5.0)  # drain -> 5000 ms
    req = _launch_req(pool, cm, tm, torch.arange(1, 13, dtype=torch.int32))
    req.decode_batch_idx = 3  # aborted mid-decode: attribution would credit this step
    batch = Batch(reqs=[req], phase="decode")
    dm.filter_reqs(batch.reqs)
    stub._last_data = _as_last_data(batch)

    Scheduler._process_one_msg(stub, AbortBackendMsg(uid=UID))
    assert req.aborted  # marked, NOT freed under the forward

    Scheduler._process_last_data(stub, stub._last_data)

    assert req.table_idx == -1
    assert sent == []                           # abort ack stays the terminal reply
    assert stub.timing_tracker.pop(UID) == (0.0, 0.0, False)  # purged, not resurrected


def test_chunked_prefill_accumulates_ms_across_chunks_into_terminal_reply():
    """Each chunk's own-window forward time accumulates per uid; the final chunk's drain
    publishes the sum on the terminal reply. Also pins the own-window span: chunk 2 is
    measured against chunk 1's drain (2.0 s), not its own launch (which would give 3.0)."""
    pool, cm, tm, dm, _pm, sent, stub = _setup()
    clock = {"t": 0.0}
    stub.timing_tracker = RequestTimingTracker(clock=lambda: clock["t"])
    # burn the engine-start warmup slot so both chunks carry regular (non-warmup) time
    stub.timing_tracker.on_prefill_drained(
        SimpleNamespace(log_new_tokens=1, log_req_new_tokens={999: 1}), 0.0)

    prompt = torch.arange(1, 13, dtype=torch.int32)
    chunk = _launch_req(pool, cm, tm, prompt[:8], cls=ChunkedReq)
    mid = Batch(reqs=[chunk], phase="prefill")
    mid.log_new_tokens = 8
    mid.log_req_new_tokens = {UID: 8}
    clock["t"] = 1.0  # chunk 1 own window: 1.0 s -> 1000 ms
    Scheduler._process_last_data(stub, _as_last_data(mid))
    assert sent == []                           # intermediate chunks never reply

    req = _launch_req(pool, cm, tm, prompt)     # final chunk: plain Req, same uid
    final = Batch(reqs=[req], phase="prefill")
    final.log_new_tokens = 12
    final.log_req_new_tokens = {UID: 12}
    dm.filter_reqs(final.reqs)
    stub.eos_token_ids = {42}  # the drained token (42) finishes the request by EOS
    clock["t"] = 3.0  # chunk 2 own window: 3.0 - 1.0 (previous drain) = 2.0 s -> 2000 ms
    Scheduler._process_last_data(stub, _as_last_data(final))

    terminal = [m for m in sent if isinstance(m, DetokenizeMsg) and m.finished]
    assert len(terminal) == 1
    assert terminal[0].prefill_ms == pytest.approx(3000.0)  # 1000 + 2000 across chunks
    assert terminal[0].prompt_warmup is False
    assert stub.timing_tracker.pop(UID) == (0.0, 0.0, False)  # entry released


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: PASS")
