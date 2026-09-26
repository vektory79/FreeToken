from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from freetoken.core import Batch


@dataclass
class SchedulerStatusReporter:
    log: Callable[[str], None]
    clock: Callable[[], float] = time.perf_counter
    decode_log_interval: int = 40
    _last_prefill_time: float = field(init=False)
    _last_decode_time: float = field(init=False)
    _decode_forward_count: int = field(default=0, init=False)
    _decode_generated_tokens: int = field(default=0, init=False)
    _decode_window_seconds: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        now = self.clock()
        self._last_prefill_time = now
        self._last_decode_time = now
        self.decode_log_interval = max(1, self.decode_log_interval)

    def report_batch(
        self,
        batch: Batch,
        *,
        running_reqs: int,
        queue_reqs: int,
        kv_used_pages: int,
        kv_total_pages: int,
        page_size: int,
        mamba_slots: tuple[int, int] | None = None,
        swa_tokens: tuple[int, int] | None = None,
        tier_fn: Callable[[], str] | None = None,
        forward_ms: float | None = None,
        warmup: bool = False,
    ) -> None:
        if batch.is_prefill:
            self._report_prefill(
                batch,
                running_reqs=running_reqs,
                queue_reqs=queue_reqs,
                kv_used_pages=kv_used_pages,
                kv_total_pages=kv_total_pages,
                mamba_slots=mamba_slots,
                swa_tokens=swa_tokens,
                tier_fn=tier_fn,
                forward_ms=forward_ms,
                warmup=warmup,
            )
        elif batch.is_decode:
            self._report_decode(
                batch,
                running_reqs=running_reqs,
                queue_reqs=queue_reqs,
                kv_used_pages=kv_used_pages,
                kv_total_pages=kv_total_pages,
                page_size=page_size,
                mamba_slots=mamba_slots,
                swa_tokens=swa_tokens,
                tier_fn=tier_fn,
                forward_ms=forward_ms,
            )

    def _report_prefill(
        self,
        batch: Batch,
        *,
        running_reqs: int,
        queue_reqs: int,
        kv_used_pages: int,
        kv_total_pages: int,
        mamba_slots: tuple[int, int] | None = None,
        swa_tokens: tuple[int, int] | None = None,
        tier_fn: Callable[[], str] | None = None,
        forward_ms: float | None = None,
        warmup: bool = False,
    ) -> None:
        now = self.clock()
        gap = now - self._last_prefill_time
        self._last_prefill_time = now
        # Read the schedule-time snapshot: by report time the forward's complete_one() has
        # advanced each req's cached_len to device_len, so reading the reqs here would log
        # decode-state values (#new-token == #reqs, #cached-token == full prompt).
        new_tokens = batch.log_new_tokens
        cached_tokens = batch.log_cached_tokens
        # Measured launch->drain forward duration, not the gap between reports: a report
        # gap mixes in host/idle time and, on the last chunk, is taken while the tail is
        # still draining - the ~1600 tok/s artifact. Gap stays as fallback when the batch
        # carried no measurement.
        measured = forward_ms / 1000.0 if forward_ms is not None and forward_ms > 0 else 0.0
        duration = measured if measured > 0 else gap
        input_throughput = new_tokens / duration if duration > 0 else 0.0
        # Trailing marker only: parsed fields (and the throughput value) keep their place.
        warmup_mark = ", warmup: true" if warmup else ""
        self.log(
            f"Prefill batch, "
            f"#new-seq: {len(batch.reqs)}, "
            f"#new-token: {new_tokens}, "
            f"#cached-token: {cached_tokens}, "
            f"token usage: {_usage_ratio(kv_used_pages, kv_total_pages):.2f}, "
            f"{_swa_msg(swa_tokens)}"
            f"{_mamba_msg(mamba_slots)}"
            f"{tier_fn() if tier_fn else ''}"
            f"#running-req: {running_reqs}, "
            f"#queue-req: {queue_reqs}, "
            f"input throughput (token/s): {input_throughput:.2f}"
            f"{warmup_mark}"
        )

    def _report_decode(
        self,
        batch: Batch,
        *,
        running_reqs: int,
        queue_reqs: int,
        kv_used_pages: int,
        kv_total_pages: int,
        page_size: int,
        mamba_slots: tuple[int, int] | None = None,
        swa_tokens: tuple[int, int] | None = None,
        tier_fn: Callable[[], str] | None = None,
        forward_ms: float | None = None,
    ) -> None:
        self._decode_forward_count += 1
        self._decode_generated_tokens += len(batch.reqs)
        if forward_ms is not None and forward_ms > 0:
            self._decode_window_seconds += forward_ms / 1000.0
        if self._decode_forward_count % self.decode_log_interval != 0:
            return   # tier_fn stays unevaluated for the discarded 39/40 batches

        now = self.clock()
        gap = now - self._last_decode_time
        self._last_decode_time = now
        # Denominator = sum of measured decode-forward durations, not the wall gap: a gap
        # spanning a prefill chunk / transition adds that time to the first window after it
        # and understates the decode rate (the TTFT-class first-window outlier). Gap stays
        # as fallback when no batch in the window carried a measurement.
        window = self._decode_window_seconds if self._decode_window_seconds > 0 else gap
        gen_throughput = self._decode_generated_tokens / window if window > 0 else 0.0
        self._decode_generated_tokens = 0
        self._decode_window_seconds = 0.0
        self.log(
            f"Decode batch, "
            f"#running-req: {running_reqs}, "
            f"#token: {kv_used_pages * page_size}, "
            f"token usage: {_usage_ratio(kv_used_pages, kv_total_pages):.2f}, "
            f"{_swa_msg(swa_tokens)}"
            f"{_mamba_msg(mamba_slots)}"
            f"{tier_fn() if tier_fn else ''}"
            f"gen throughput (token/s): {gen_throughput:.2f}, "
            f"#queue-req: {queue_reqs}"
        )


def _usage_ratio(used: int, total: int) -> float:
    return used / total if total > 0 else 0.0


@dataclass
class RequestTimingTracker:
    """Per-request forward-time accumulation feeding the llama.cpp-style timings block.

    The scheduler measures each batch's forward duration (launch -> drain, after
    copy_done.synchronize) and hands the batch in here: prefill time is attributed by each
    request's token share, decode time by equal share. Two exclusions keep the rates
    honest: the engine-start warmup prefill batch (JIT/autotune cost, marked in the log)
    is excluded from prompt time, and each request's first decode step (TTFT-class CUDA
    graph/sync overhead) from predicted time. Token counts are unaffected.
    """

    clock: Callable[[], float] = time.perf_counter
    _prefill_ms: dict[int, float] = field(default_factory=dict)
    _decode_ms: dict[int, float] = field(default_factory=dict)
    _warmup_uids: set[int] = field(default_factory=set)
    _seen_prefill: bool = False
    _last_drain: float | None = field(default=None)

    def batch_forward_ms(self, launch_time: float) -> float:
        """Own-window forward duration in ms, measured launch -> drain (after the event
        sync).

        Under overlap scheduling batch N launches while N-1 still runs, so a plain
        launch->drain span includes the queue wait behind N-1 and ~doubles the GPU-bound
        steady-state duration; drain - max(launch, previous drain) counts only this
        batch's own window. The first batch's predecessor is its own launch.
        """
        drain = self.clock()
        prev = self._last_drain
        start = launch_time if prev is None else max(launch_time, prev)
        self._last_drain = drain
        return max(0.0, (drain - start) * 1000.0)

    def on_prefill_drained(self, batch: Batch, forward_ms: float) -> bool:
        """Attribute a drained prefill batch's time; returns True when it was the warmup."""
        warmup = not self._seen_prefill
        self._seen_prefill = True
        total = batch.log_new_tokens
        if total <= 0 or forward_ms <= 0:
            return warmup
        ms_per_token = forward_ms / total
        for uid, tokens in batch.log_req_new_tokens.items():
            if tokens <= 0:
                continue
            if not warmup:
                self._prefill_ms[uid] = self._prefill_ms.get(uid, 0.0) + tokens * ms_per_token
            else:
                self._warmup_uids.add(uid)
        return warmup

    def on_decode_drained(self, batch: Batch, forward_ms: float) -> None:
        """Attribute a drained decode batch's time, skipping each request's first step."""
        if forward_ms <= 0 or not batch.reqs:
            return
        share = forward_ms / len(batch.reqs)
        for req in batch.reqs:
            # decode_batch_idx is incremented at prepare time, so 1 = this was the request's
            # first decode forward: its cost is TTFT-class overhead, not steady decode.
            if getattr(req, "decode_batch_idx", 0) <= 1:
                continue
            self._decode_ms[req.uid] = self._decode_ms.get(req.uid, 0.0) + share

    def pop(self, uid: int) -> tuple[float, float, bool]:
        """Accumulated (prefill_ms, decode_ms, prompt_warmup) for a terminal request."""
        prefill = self._prefill_ms.pop(uid, 0.0)
        decode = self._decode_ms.pop(uid, 0.0)
        warmup = uid in self._warmup_uids
        self._warmup_uids.discard(uid)
        return prefill, decode, warmup


def _mamba_msg(mamba_slots: tuple[int, int] | None) -> str:
    """GDN-state (mamba) pool occupancy for hybrid models; empty for the rest."""
    if mamba_slots is None:
        return ""
    used, total = mamba_slots
    return f"#mamba-slot: {used}/{total}, mamba usage: {_usage_ratio(used, total):.2f}, "


def _swa_msg(swa_tokens: tuple[int, int] | None) -> str:
    """Window (swa) pool occupancy for SWA models; empty for the rest."""
    if swa_tokens is None:
        return ""
    used, total = swa_tokens
    return f"#swa-token: {used}/{total}, swa usage: {_usage_ratio(used, total):.2f}, "
