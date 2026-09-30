"""io_uring deep-queue reader for the FTW bank load (kb/cases/boot-shutdown-io P7).

A blocking-preadv pool worker holds QD~1 per thread, which flattened the bank phase at
~5 GB/s while the device proved 6+ (fio 1xQD32x1M); one raw-syscall io_uring ring keeps
QD 32-64 reads in flight with a single ``io_uring_enter`` per batch. Raw x86_64 syscalls
(setup=425, enter=426) instead of liburing: no library dependency, and the SQ/CQ layout
is taken from the offsets the kernel returns, never hardcoded.

Design decisions fixed before the code (brief P7, .tasks/boot-shutdown-io/p7-plan):

- Buffer lifetime: an SQE's receive buffer is a ctypes export of the caller's
  destination memoryview, created when the SQE is prepared and released only after its
  CQE is reaped; a short read requeues the remainder (re-exporting then). Destinations
  themselves live until ``on_done`` -- strictly longer than any in-flight read.
- Thread-safety: the ring lives entirely on the calling thread (one submitter/reaper;
  ``submit_wait`` is the only blocking call). No io_uring state is touched from CUDA or
  pin threads; ``on_done`` (progress bar + ``PinPipeline.submit``, a thread-safe enqueue)
  runs here exactly like it ran on pool threads in the direct path.
- Fallback: only ring initialization raises :class:`IoUringUnavailable` (no kernel
  support, seccomp, fd exhaustion) and downgrades to the pool path; errors on reaped
  reads propagate like the pool path -- first error re-raised after the ring drains,
  the failed task skips its single ``on_done``.
"""

from __future__ import annotations

import ctypes
import errno
import mmap
import os
import platform
import struct
from collections import deque

from freetoken.utils import init_logger

logger = init_logger(__name__)

_SYS_IO_URING_SETUP = 425  # x86_64 syscall numbers, stable kernel ABI
_SYS_IO_URING_ENTER = 426
_IORING_OFF_SQ_RING = 0
_IORING_OFF_CQ_RING = 0x8000000
_IORING_OFF_SQES = 0x10000000
_IORING_ENTER_GETEVENTS = 1
_IORING_OP_READV = 1
_SQE_SIZE = 64
_CQE_SIZE = 16
# opcode flags ioprio fd | off addr | len rw_flags | user_data | zero the rest
_SQE_PACK = "<BBHiQQIIQ24x"
_CQE_PACK = "<Qii"  # user_data res flags


class IoUringUnavailable(OSError):
    """The ring cannot be set up at all (ENOSYS/EINVAL/EPERM/...); the caller falls back to the pool."""


def _syscall(nr: int, *args) -> int:
    r = _libc.syscall(nr, *args)
    if r < 0:
        e = ctypes.get_errno()
        raise OSError(e, errno.errorcode.get(e, str(e)))
    return r


_libc = ctypes.CDLL(None, use_errno=True)
_libc.syscall.restype = ctypes.c_long


class _SqRingOff(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint32) for n in
                ("head", "tail", "ring_mask", "ring_entries", "flags", "dropped", "array")] + [
                ("resv", ctypes.c_uint32 * 3)]


class _CqRingOff(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint32) for n in
                ("head", "tail", "ring_mask", "ring_entries", "overflow", "cqes", "flags")] + [
                ("resv", ctypes.c_uint32 * 3)]


class _SetupParams(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint32) for n in
                ("sq_entries", "cq_entries", "flags", "sq_thread_cpu", "sq_thread_idle",
                 "features", "wq_fd")] + [
                ("resv", ctypes.c_uint32 * 3), ("sq_off", _SqRingOff), ("cq_off", _CqRingOff)]


class _IoVec(ctypes.Structure):
    _fields_ = [("base", ctypes.c_void_p), ("len", ctypes.c_size_t)]


class _Ring:
    """Minimal SQ/CQ ring over raw syscalls. Single-threaded by contract: the object
    must be constructed, driven and closed from ONE thread (the submitter/reaper)."""

    def __init__(self, entries: int):
        # Plain user-space stores (SQE fields, SQ tail, CQ head) are correct on x86 because
        # TSO orders them and the io_uring_enter syscall boundary serializes against the
        # kernel; on weakly ordered ISAs (ARM) they race without liburing's barriers.
        if platform.machine() not in ("x86_64", "amd64"):
            raise IoUringUnavailable(
                f"io_uring ring supports x86_64 only, not {platform.machine()}"
            )
        self.sq = self.cq = self.sqes = None
        try:
            params = _SetupParams()
            fd = _syscall(_SYS_IO_URING_SETUP, entries, ctypes.byref(params))
            try:
                sq_sz = params.sq_off.array + params.sq_entries * 4
                cq_sz = params.cq_off.cqes + params.cq_entries * _CQE_SIZE
                self.sq = mmap.mmap(fd, sq_sz, offset=_IORING_OFF_SQ_RING)
                self.cq = mmap.mmap(fd, cq_sz, offset=_IORING_OFF_CQ_RING)
                self.sqes = mmap.mmap(fd, params.sq_entries * _SQE_SIZE, offset=_IORING_OFF_SQES)
            except (OSError, ValueError, OverflowError) as exc:
                for m in (self.sqes, self.cq, self.sq):  # unmap the done ones, reverse order
                    if m is not None:
                        try:
                            m.close()
                        except (OSError, ValueError):
                            pass
                os.close(fd)
                raise IoUringUnavailable(f"ring mmap failed: {exc}") from exc
        except IoUringUnavailable:
            raise
        except OSError as exc:
            raise IoUringUnavailable(str(exc)) from exc
        if params.sq_entries < 1:
            os.close(fd)
            raise IoUringUnavailable("kernel returned an empty SQ ring")
        self._fd = fd
        self._so = params.sq_off
        self._co = params.cq_off
        self._entries = params.sq_entries
        self.granted = params.sq_entries  # what the kernel actually handed out (Info TP)

    def _u32(self, buf, off: int) -> int:
        return int.from_bytes(buf[off:off + 4], "little")

    def prep_readv(self, fd: int, iov_addr: int, offset: int, user_data: int) -> bool:
        """Fill the next SQE with one READV over ``iov_addr``; False when the SQ ring is full."""
        tail = self._u32(self.sq, self._so.tail)
        if tail - self._u32(self.sq, self._so.head) >= self._entries:
            return False
        idx = tail & self._u32(self.sq, self._so.ring_mask)
        struct.pack_into(_SQE_PACK, self.sqes, idx * _SQE_SIZE,
                         _IORING_OP_READV, 0, 0, fd, offset, iov_addr, 1, 0, user_data)
        struct.pack_into("<I", self.sq, self._so.array + idx * 4, idx)
        struct.pack_into("<I", self.sq, self._so.tail, tail + 1)
        return True

    def submit_wait(self, to_submit: int, min_complete: int = 1) -> int:
        """``io_uring_enter`` with GETEVENTS (to_submit may be 0: pure wait); retries EINTR."""
        while True:
            try:
                return _syscall(_SYS_IO_URING_ENTER, self._fd, to_submit, min_complete,
                                _IORING_ENTER_GETEVENTS, None, None)
            except OSError as exc:
                if exc.errno != errno.EINTR:
                    raise

    def reap(self) -> list[tuple[int, int]]:
        """Pop every completion available now as ``(user_data, res)``; the CQ head is
        published only after the batch is read (the kernel never reorders below it)."""
        head = self._u32(self.cq, self._co.head)
        tail = self._u32(self.cq, self._co.tail)
        if head == tail:
            return []
        mask = self._u32(self.cq, self._co.ring_mask)
        base = self._co.cqes
        out = []
        while head != tail:
            out.append(struct.unpack_from(_CQE_PACK, self.cq,
                                          base + (head & mask) * _CQE_SIZE)[:2])
            head += 1
        struct.pack_into("<I", self.cq, self._co.head, head)
        return out

    def close(self) -> None:
        for m in (self.sq, self.cq, self.sqes):
            if m is None:
                continue
            try:
                m.close()
            except (OSError, ValueError):
                pass
        try:
            os.close(self._fd)
        except (OSError, AttributeError):
            pass


class _Rec:
    """One in-flight read; the record owns the ctypes export so the kernel never writes
    into memory Python could reclaim before the CQE."""

    __slots__ = ("fd", "file_off", "dest_off", "nbytes", "state", "buf", "iov")

    def __init__(self, fd: int, file_off: int, dest_off: int, nbytes: int, state: dict):
        self.fd = fd
        self.file_off = file_off
        self.dest_off = dest_off
        self.nbytes = nbytes
        self.state = state
        self.buf = None
        self.iov = None


def _complete(rec: _Rec, res: int, queue: deque, err: list) -> bool:
    """Apply one reaped CQE. Returns True when the job is finally settled (done or
    failed); a short read requeues its remainder and returns False."""
    state = rec.state
    rec.buf = None  # release the ctypes export in every branch; a requeue re-exports
    rec.iov = None
    if res < 0:
        state["failed"] = True
        err.append(OSError(f"shard fd {rec.fd}: io_uring read failed: "
                           f"{errno.errorcode.get(-res, -res)} at offset {rec.file_off}"))
        return True
    if res == 0:
        state["failed"] = True
        err.append(OSError(f"unexpected EOF reading FTW: fd {rec.fd} at offset "
                           f"{rec.file_off} (got 0/{rec.nbytes} bytes)"))
        return True
    if res < rec.nbytes:
        # resume at the running offset, O_DIRECT-legal (all ranges stay block-aligned)
        rec.file_off += res
        rec.dest_off += res
        rec.nbytes -= res
        queue.appendleft((rec.fd, rec.file_off, rec.dest_off, rec.nbytes, state))
        return False
    state["left"] -= 1
    if state["left"] == 0 and not state["failed"]:
        try:
            state["on_done"]()
        except BaseException as exc:  # surfaced after the drain, like the pool path
            err.append(exc)
    return True


def _drain(ring: _Ring, queue: deque, qd: int, err: list) -> None:
    """Single submitter/reaper loop; runs on the calling thread only."""
    in_flight: dict[int, _Rec] = {}
    next_id = 0
    total = len(queue)
    done = 0
    submitted = 0  # SQEs prepared since the last enter
    while done < total:
        while len(in_flight) + submitted < qd and queue:
            fd, fo, do, ln, state = queue.popleft()
            rec = _Rec(fd, fo, do, ln, state)
            try:
                # the export pins the destination region until the CQE is reaped
                rec.buf = (ctypes.c_char * ln).from_buffer(state["dest"], do)
            except (BufferError, ValueError, TypeError) as exc:
                raise OSError(f"cannot map FTW destination for io_uring read: {exc}") from exc
            rec.iov = _IoVec(ctypes.addressof(rec.buf), ln)
            if not ring.prep_readv(fd, ctypes.addressof(rec.iov), fo, next_id):
                rec.buf = None  # kernel clamp defense: sq_entries may come back < qd, so the SQ ring can fill
                queue.appendleft((fd, fo, do, ln, state))
                break
            in_flight[next_id] = rec
            next_id += 1
            submitted += 1
        if submitted == 0 and not in_flight:
            raise OSError("io_uring ring made no progress (jobs pending, nothing in flight)")
        ring.submit_wait(submitted, 1)
        submitted = 0
        for user_data, res in ring.reap():
            if _complete(in_flight.pop(user_data), res, queue, err):
                done += 1


def _settle(on_dones, err: list) -> None:
    """Fire each zero-job settle exactly once; a failure is recorded, not raised
    (the first error surfaces after the drain, like the pool path)."""
    for on_done in on_dones:
        try:
            on_done()
        except BaseException as exc:  # noqa: BLE001
            err.append(exc)


def run_ring_reads(reader, planned, tasks, *, workers: int, chunk: int, qd: int) -> None:
    """Execute every planned sub-read through one io_uring ring (P7 backend).

    ``planned``/``tasks`` are the ``(dest, jobs)`` / ``(dest, entry, on_done)`` lists
    from ``_run_shared_reads``. Raises :class:`IoUringUnavailable` only from ring setup;
    read errors are raised after the ring drains (the failed task skips its settle).
    Zero-job entries settle exactly like read ones: inline when no ring is needed at
    all, otherwise right after the ring is built and never before it -- so the pool
    fallback can never double-settle a bank.
    """
    for file in {j[0] for _, jobs in planned for j in jobs}:
        reader._fd(file)
    queue: deque = deque()
    empties = []  # zero-job on_dones: total stays read-only, settles never lost
    for (dest, jobs), (_dest, _entry, on_done) in zip(planned, tasks):
        if not jobs:
            empties.append(on_done)
            continue
        state = {"left": len(jobs), "failed": False, "on_done": on_done, "dest": dest}
        for file, fo, do, ln in jobs:
            queue.append((reader._fd(file), fo, do, ln, state))
    err: list[BaseException] = []
    if not queue:  # every task zero-job: settle inline, the ring is never built
        _settle(empties, err)
        if err:
            raise err[0]
        return
    ring = _Ring(qd)  # IoUringUnavailable escapes only from here (nothing settled yet)
    logger.info(
        f"FTW bank reads: {len(tasks)} tasks, {len(queue)} jobs, pool={workers}, "
        f"sub-chunk={chunk // 1024} KiB, backend=io_uring, qd={qd}, granted={ring.granted}"
    )
    _settle(empties, err)  # ring path committed: zero-job entries settle like read ones
    try:
        _drain(ring, queue, qd, err)
    finally:
        ring.close()
    if err:
        raise err[0]
