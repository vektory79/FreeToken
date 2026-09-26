from __future__ import annotations

import pytest
from types import SimpleNamespace

from freetoken.scheduler.status import (
    RequestTimingTracker,
    SchedulerStatusReporter,
    _usage_ratio,
)


def _reporter(interval=40):
    logs: list[str] = []
    clock = {"t": 0.0}
    rep = SchedulerStatusReporter(
        log=logs.append,
        clock=lambda: clock["t"],
        decode_log_interval=interval,
    )
    return rep, logs, clock


def _req(extend, cached):
    return SimpleNamespace(extend_len=extend, cached_len=cached)


def _prefill_batch(new_tokens, cached_tokens, n_seqs):
    # The reporter must read the schedule-time snapshot (log_new_tokens/log_cached_tokens),
    # NOT the live reqs: by report time forward's complete_one() has advanced them to
    # decode state. Live reqs here carry deliberately-wrong values to prove that.
    reqs = [_req(extend=1, cached=10_000) for _ in range(n_seqs)]
    return SimpleNamespace(
        is_prefill=True, is_decode=False, reqs=reqs,
        log_new_tokens=new_tokens, log_cached_tokens=cached_tokens,
    )


def _decode_batch(n):
    return SimpleNamespace(is_prefill=False, is_decode=True, reqs=[_req(1, 0) for _ in range(n)])


def test_prefill_line_reports_tokens_and_throughput():
    rep, logs, clock = _reporter()
    clock["t"] = 0.5  # 30 new tokens over 0.5s -> 60 tok/s
    rep.report_batch(
        _prefill_batch(new_tokens=30, cached_tokens=12, n_seqs=2),
        running_reqs=2, queue_reqs=1, kv_used_pages=50, kv_total_pages=200, page_size=16,
    )
    assert len(logs) == 1
    line = logs[0]
    assert "#new-seq: 2" in line
    assert "#new-token: 30" in line  # snapshot, not the live reqs' extend_len (1 each)
    assert "#cached-token: 12" in line  # snapshot, not the live reqs' cached_len (10000 each)
    assert "token usage: 0.25" in line
    assert "#running-req: 2" in line
    assert "#queue-req: 1" in line
    assert "input throughput (token/s): 60.00" in line


def test_mamba_slots_reported_only_when_provided():
    rep, logs, clock = _reporter()
    clock["t"] = 1.0
    # non-hybrid (mamba_slots=None): no mamba field
    rep.report_batch(
        _prefill_batch(new_tokens=10, cached_tokens=0, n_seqs=1),
        running_reqs=1, queue_reqs=0, kv_used_pages=1, kv_total_pages=10, page_size=1,
    )
    assert "mamba" not in logs[-1]
    # hybrid: #mamba-slot: used/total and usage ratio
    clock["t"] = 2.0
    rep.report_batch(
        _prefill_batch(new_tokens=10, cached_tokens=0, n_seqs=1),
        running_reqs=1, queue_reqs=0, kv_used_pages=1, kv_total_pages=10, page_size=1,
        mamba_slots=(37, 256),
    )
    assert "#mamba-slot: 37/256" in logs[-1]
    assert "mamba usage: 0.14" in logs[-1]


def test_swa_tokens_reported_only_when_provided():
    rep, logs, clock = _reporter(interval=1)
    clock["t"] = 1.0
    # non-SWA (swa_tokens=None): no swa field
    rep.report_batch(
        _prefill_batch(new_tokens=10, cached_tokens=0, n_seqs=1),
        running_reqs=1, queue_reqs=0, kv_used_pages=1, kv_total_pages=10, page_size=1,
    )
    assert "swa" not in logs[-1]
    # SWA: #swa-token: used/total and usage ratio, on both prefill and decode lines
    clock["t"] = 2.0
    rep.report_batch(
        _prefill_batch(new_tokens=10, cached_tokens=0, n_seqs=1),
        running_reqs=1, queue_reqs=0, kv_used_pages=1, kv_total_pages=10, page_size=1,
        swa_tokens=(8448, 76800),
    )
    assert "#swa-token: 8448/76800" in logs[-1]
    assert "swa usage: 0.11" in logs[-1]
    clock["t"] = 3.0
    rep.report_batch(_decode_batch(1), running_reqs=1, queue_reqs=0,
                     kv_used_pages=1, kv_total_pages=10, page_size=1,
                     swa_tokens=(8448, 76800))
    assert "#swa-token: 8448/76800" in logs[-1]
    assert "swa usage: 0.11" in logs[-1]


def test_decode_lines_are_throttled_to_every_nth_forward():
    rep, logs, clock = _reporter(interval=3)
    for i, t in enumerate((1.0, 1.5), start=1):
        clock["t"] = t
        rep.report_batch(_decode_batch(2), running_reqs=2, queue_reqs=0,
                         kv_used_pages=60, kv_total_pages=200, page_size=16)
        assert logs == [], f"should not log before the interval (forward {i})"
    clock["t"] = 2.0  # 3rd forward -> log; 6 tokens over 2.0s gap -> 3 tok/s
    rep.report_batch(_decode_batch(2), running_reqs=2, queue_reqs=4,
                     kv_used_pages=62, kv_total_pages=200, page_size=16)
    assert len(logs) == 1
    line = logs[0]
    assert "#running-req: 2" in line
    assert "#queue-req: 4" in line
    assert "#token: 992" in line  # 62 pages * 16
    assert "token usage: 0.31" in line
    assert "gen throughput (token/s): 3.00" in line


def test_decode_counter_resets_each_interval():
    rep, logs, clock = _reporter(interval=2)
    clock["t"] = 1.0
    rep.report_batch(_decode_batch(5), running_reqs=5, queue_reqs=0,
                     kv_used_pages=1, kv_total_pages=10, page_size=1)
    clock["t"] = 2.0  # first emission: 10 tokens over 2.0s -> 5 tok/s
    rep.report_batch(_decode_batch(5), running_reqs=5, queue_reqs=0,
                     kv_used_pages=1, kv_total_pages=10, page_size=1)
    assert "gen throughput (token/s): 5.00" in logs[-1]
    # next window is measured from the previous emission, with a reset token count
    clock["t"] = 3.0
    rep.report_batch(_decode_batch(3), running_reqs=3, queue_reqs=0,
                     kv_used_pages=1, kv_total_pages=10, page_size=1)
    clock["t"] = 4.0  # 6 tokens over (4.0-2.0)=2.0s -> 3 tok/s
    rep.report_batch(_decode_batch(3), running_reqs=3, queue_reqs=0,
                     kv_used_pages=1, kv_total_pages=10, page_size=1)
    assert "gen throughput (token/s): 3.00" in logs[-1]


def test_zero_gap_and_zero_total_are_guarded():
    rep, logs, clock = _reporter(interval=1)
    # gap == 0 (clock unchanged since construction) and total == 0 must not raise
    rep.report_batch(_decode_batch(4), running_reqs=4, queue_reqs=0,
                     kv_used_pages=0, kv_total_pages=0, page_size=1)
    line = logs[-1]
    assert "gen throughput (token/s): 0.00" in line
    assert "token usage: 0.00" in line
    assert "#token: 0" in line  # owned-KV (dsv4) reports 0/0 pages


def test_interval_is_clamped_to_at_least_one():
    rep, _, _ = _reporter(interval=0)
    assert rep.decode_log_interval == 1
    rep_neg, _, _ = _reporter(interval=-5)
    assert rep_neg.decode_log_interval == 1


def test_usage_ratio_guard():
    assert _usage_ratio(0, 0) == 0.0
    assert _usage_ratio(5, 0) == 0.0
    assert _usage_ratio(5, 10) == 0.5


# ------------------------------------- measured forward duration (W2/W3 wave)
def test_prefill_throughput_uses_measured_forward_duration():
    # Regression for the last-chunk artifact: the report gap (5.2s) suggested an
    # impossible ~1563 tok/s while the measured forward took the full 8s.
    rep, logs, clock = _reporter()
    clock["t"] = 1.0
    rep.report_batch(
        _prefill_batch(new_tokens=8128, cached_tokens=0, n_seqs=1),
        running_reqs=1, queue_reqs=0, kv_used_pages=1, kv_total_pages=10, page_size=1,
    )
    clock["t"] = 6.2  # report gap = 5.2s; the batch's forward itself measured 8.0s
    rep.report_batch(
        _prefill_batch(new_tokens=8128, cached_tokens=0, n_seqs=1),
        running_reqs=1, queue_reqs=0, kv_used_pages=1, kv_total_pages=10, page_size=1,
        forward_ms=8000.0,
    )
    assert "input throughput (token/s): 1016.00" in logs[-1]  # 8128 / 8.0
    assert "1563" not in logs[-1]  # the gap-based artifact value must be gone


def test_prefill_throughput_falls_back_to_gap_without_measurement():
    rep, logs, clock = _reporter()
    clock["t"] = 0.5  # 30 new tokens over a 0.5s gap -> 60 tok/s (legacy behavior)
    rep.report_batch(
        _prefill_batch(new_tokens=30, cached_tokens=12, n_seqs=2),
        running_reqs=2, queue_reqs=1, kv_used_pages=50, kv_total_pages=200, page_size=16,
    )
    assert "input throughput (token/s): 60.00" in logs[-1]


def test_warmup_prefill_line_is_marked_and_placeholders_unchanged():
    rep, logs, clock = _reporter()
    clock["t"] = 1.0
    rep.report_batch(
        _prefill_batch(new_tokens=30, cached_tokens=0, n_seqs=1),
        running_reqs=1, queue_reqs=0, kv_used_pages=1, kv_total_pages=10, page_size=1,
        forward_ms=3000.0,
        warmup=True,
    )
    assert logs[-1].endswith(", warmup: true")
    # the parsed throughput field keeps its place in the line
    assert "input throughput (token/s): 10.00" in logs[-1]
    clock["t"] = 2.0
    rep.report_batch(
        _prefill_batch(new_tokens=30, cached_tokens=0, n_seqs=1),
        running_reqs=1, queue_reqs=0, kv_used_pages=1, kv_total_pages=10, page_size=1,
        forward_ms=3000.0,
    )
    assert "warmup" not in logs[-1]


def test_decode_window_throughput_uses_measured_batch_durations():
    # First decode window after a prefill chunk: the wall gap includes the prefill and
    # understates the rate; the measured decode window excludes it.
    rep, logs, clock = _reporter(interval=2)
    clock["t"] = 1.0
    rep.report_batch(
        _decode_batch(2), running_reqs=2, queue_reqs=0,
        kv_used_pages=60, kv_total_pages=200, page_size=16,
        forward_ms=100.0,
    )
    assert logs == []  # throttled
    clock["t"] = 1.5
    rep.report_batch(  # a prefill chunk lands inside the decode window
        _prefill_batch(new_tokens=8000, cached_tokens=0, n_seqs=1),
        running_reqs=2, queue_reqs=0, kv_used_pages=60, kv_total_pages=200, page_size=16,
    )
    clock["t"] = 9.0  # wall gap 8.0s; measured decode window = 0.1 + 0.1 = 0.2s
    rep.report_batch(
        _decode_batch(2), running_reqs=2, queue_reqs=0,
        kv_used_pages=60, kv_total_pages=200, page_size=16,
        forward_ms=100.0,
    )
    assert "gen throughput (token/s): 20.00" in logs[-1]  # 4 tokens / 0.2s
    assert "0.50" not in logs[-1]  # the gap-based value (4 / 8.0) must be gone


# ------------------------------------------------- per-request timing tracker
def test_tracker_attributes_prefill_by_token_share_and_marks_warmup():
    tracker = RequestTimingTracker(clock=lambda: 0.0)
    warmup_batch = SimpleNamespace(log_new_tokens=100, log_req_new_tokens={1: 80, 2: 20})
    assert tracker.on_prefill_drained(warmup_batch, 1000.0) is True  # first prefill = warmup
    # warmup time is excluded from the rate, but the touched requests stay marked
    assert tracker.pop(1) == (0.0, 0.0, True)

    second = SimpleNamespace(log_new_tokens=50, log_req_new_tokens={2: 50})
    assert tracker.on_prefill_drained(second, 500.0) is False
    # req 2: warmup-chunk share dropped, steady-state share kept; flag still marks it
    assert tracker.pop(2) == (pytest.approx(500.0), 0.0, True)


def test_tracker_splits_prefill_time_proportionally_to_tokens():
    tracker = RequestTimingTracker(clock=lambda: 0.0)
    # burn the engine-start warmup slot so the measured batch is a regular one
    tracker.on_prefill_drained(SimpleNamespace(log_new_tokens=1, log_req_new_tokens={9: 1}), 0.0)
    batch = SimpleNamespace(log_new_tokens=100, log_req_new_tokens={1: 75, 2: 25})
    assert tracker.on_prefill_drained(batch, 1000.0) is False
    assert tracker.pop(1) == (pytest.approx(750.0), 0.0, False)
    assert tracker.pop(2) == (pytest.approx(250.0), 0.0, False)


def test_tracker_skips_each_requests_first_decode_step():
    tracker = RequestTimingTracker(clock=lambda: 0.0)
    first = SimpleNamespace(reqs=[
        SimpleNamespace(uid=1, decode_batch_idx=1),   # first decode forward: skipped
        SimpleNamespace(uid=2, decode_batch_idx=2),
    ])
    tracker.on_decode_drained(first, 100.0)
    second = SimpleNamespace(reqs=[
        SimpleNamespace(uid=1, decode_batch_idx=2),
        SimpleNamespace(uid=2, decode_batch_idx=3),
    ])
    tracker.on_decode_drained(second, 60.0)
    assert tracker.pop(1) == (0.0, pytest.approx(30.0), False)
    assert tracker.pop(2) == (0.0, pytest.approx(80.0), False)
    assert tracker.pop(1) == (0.0, 0.0, False)  # pop clears


def test_tracker_batch_forward_ms_measures_launch_to_drain():
    now = {"t": 1.0}
    tracker = RequestTimingTracker(clock=lambda: now["t"])
    launch = tracker.clock()
    now["t"] = 1.25
    assert tracker.batch_forward_ms(launch) == pytest.approx(250.0)
