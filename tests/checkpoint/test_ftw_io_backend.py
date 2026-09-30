"""P7 io_uring backend for the FTW bank phase (kb/cases/boot-shutdown-io P7).

FREETOKEN_FTW_IO_BACKEND=io_uring routes the shared bank reads through one raw-syscall
io_uring ring (QD-deep, single thread) instead of the blocking-preadv pool. Every test
runs on a synthetic CPU-only FTW fixture written straight through FTWWriter; the ring
tests need O_DIRECT to be accepted by the fixture filesystem (the reader probe falls
back to mmap elsewhere, and the ring never engages there -- asserted too).

The pin contract (exactly one settle per bank) is asserted under the io_uring backend,
not assumed: the settle happens in on_done, which the ring loop calls exactly once
per task.
"""

from __future__ import annotations

import collections
import ctypes
import logging
import mmap as mmap_mod
import os
import threading

import pytest
import torch

from freetoken.checkpoint import ftw as ftw_mod
from freetoken.checkpoint import ftw_uring
from freetoken.checkpoint.ftw import (
    FTWReader,
    FTWWriter,
    _ALPHA_NAMES,
    layer_bank_entry_name,
    load_ftw_banks,
)

_LAYERS = 4
_E = 4
_FLAT_E = 5
_FLAT_ROWS = 3
_PER_LAYER_BYTES = [
    (2 << 20) + 1234,  # > 2 MiB sub-read cap: multiple sub-reads + ragged tail
    2 * (2 << 20) + 7,
    8192,
    4096,  # exactly one block
]


def _uring_skip_reason():
    """One probe per test session: raw-ring tests skip (not fail) where io_uring is missing."""
    try:
        ring = ftw_uring._Ring(1)
        ring.close()
    except Exception as exc:  # noqa: BLE001
        return f"io_uring ring unavailable on this box: {exc}"
    return None


_URING_SKIP = _uring_skip_reason()


class _FakeReader:
    """run_ring_reads only needs ``_fd(file)``; hand it one real fd over a small blob."""

    def __init__(self, fd):
        self._fd_value = fd

    def _fd(self, file):
        return self._fd_value


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    # keep the whole wave off the GPU driver (pins no-op) and pin the env for every test:
    # stray FREETOKEN_* exports from an iron shell must not flip a backend assertion
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("FREETOKEN_SKIP_BANK_PIN", "1")
    monkeypatch.delenv("FREETOKEN_BANK_POOL_WORKERS", raising=False)
    monkeypatch.delenv(ftw_mod._IO_BACKEND_ENV, raising=False)
    monkeypatch.delenv(ftw_mod._IO_QD_ENV, raising=False)


def _pattern(shape, seed):
    n = 1
    for s in shape:
        n *= s
    t = (torch.arange(n, dtype=torch.int64) * 7 + seed) % 251
    return t.to(torch.uint8).view(*shape)


def _write_banks_ftw(out_dir, *, shard_limit=8 << 30):
    """Per-layer gate_up/down entries x4 layers, alpha vectors, one flat multi-layer
    entry (its per-layer windows are not align-aligned). shard_limit small -> shards."""
    out_dir.mkdir(parents=True, exist_ok=True)
    w = FTWWriter(str(out_dir), shard_limit=shard_limit)
    expect = {"alpha": {}, "per_layer": {}, "flat": None}
    seed = 0
    for name, n in zip(_ALPHA_NAMES, (1000, 509)):
        t = _pattern((n,), seed)
        seed += 1
        w.add_tensor(name, t, kind="experts_bank")
        expect["alpha"][name] = t
    for layer, lbytes in enumerate(_PER_LAYER_BYTES):
        for role in ("gate_up", "down"):
            t = _pattern((_E, lbytes), seed)
            seed += 1
            w.add_tensor(layer_bank_entry_name(role, layer), t, kind="experts_bank")
            expect["per_layer"][(role, layer)] = t
    t = _pattern((_LAYERS * _FLAT_E, _FLAT_ROWS), seed)
    w.add_tensor("gate_up_scales", t, kind="experts_bank")
    expect["flat"] = t
    w.finalize({"quant_format": "bf16", "expert_bank_num_layers": _LAYERS})
    return expect


def _assert_banks_exact(banks, expect):
    assert torch.equal(banks.gate_up_alpha, expect["alpha"]["gate_up_alpha"])
    assert torch.equal(banks.down_alpha, expect["alpha"]["down_alpha"])
    for (role, layer), want in expect["per_layer"].items():
        assert torch.equal(banks.sources[role][layer], want), (role, layer)
    flat = expect["flat"]
    for layer in range(_LAYERS):
        want = flat[layer * _FLAT_E:(layer + 1) * _FLAT_E]
        assert torch.equal(banks.sources["gate_up_scale"][layer], want), ("flat", layer)


def test_backend_env_resolver(monkeypatch, caplog):
    f = ftw_mod._resolve_io_backend
    env = ftw_mod._IO_BACKEND_ENV
    monkeypatch.delenv(env, raising=False)
    assert f() == "direct"  # unset -> direct: the io_uring path is opt-in
    monkeypatch.setenv(env, "   ")
    assert f() == "direct"  # whitespace-only = unset
    for raw in ("io_uring", "auto", "direct"):
        monkeypatch.setenv(env, raw)
        assert f() == raw, raw
    monkeypatch.setenv(env, "garbage")
    with caplog.at_level(logging.WARNING, logger="freetoken.checkpoint.ftw"):
        assert f() == "direct"  # garbage -> default, never a crash
    assert "garbage" in caplog.text and "backend=direct" in caplog.text


def test_io_qd_resolver(monkeypatch, caplog):
    f = ftw_mod._io_qd
    env = ftw_mod._IO_QD_ENV
    monkeypatch.delenv(env, raising=False)
    assert f() == 32  # unset -> the fio-precedent default
    monkeypatch.setenv(env, "   ")
    assert f() == 32
    for raw, want in (("64", 64), ("1", 1), ("128", 128), ("0", 1), ("-3", 1), ("999", 128)):
        monkeypatch.setenv(env, raw)
        assert f() == want, raw
    monkeypatch.setenv(env, "abc")
    with caplog.at_level(logging.WARNING, logger="freetoken.checkpoint.ftw"):
        assert f() == 32
    assert "abc" in caplog.text
    monkeypatch.setenv(env, "999")
    with caplog.at_level(logging.WARNING, logger="freetoken.checkpoint.ftw"):
        assert f() == 128
    assert "clamped" in caplog.text  # clamping warns, it is not silent


def test_ring_reads_byte_exact_io_uring_backend(tmp_path, monkeypatch, caplog):
    out_dir = tmp_path / "ckpt"
    expect = _write_banks_ftw(out_dir)
    monkeypatch.setenv(ftw_mod._IO_BACKEND_ENV, "io_uring")
    with caplog.at_level(logging.INFO, logger="freetoken.checkpoint.ftw"):
        banks = load_ftw_banks(str(out_dir), num_layers=_LAYERS, chunk=1 << 12)  # 4 KiB jobs: a deep real queue
    if "backend=io_uring" not in caplog.text:
        pytest.skip("fixture filesystem rejected O_DIRECT; the ring never engages (mmap fallback)")
    _assert_banks_exact(banks, expect)
    assert "ignoring unknown" not in caplog.text

    # cross-check one per-layer range against an independent serial read_into
    reader = FTWReader(str(out_dir))
    entry = reader.tensors[layer_bank_entry_name("gate_up", 0)]
    ref = ftw_mod._transient_buffer(entry["nbytes"])
    reader.read_into(memoryview(ref), entry, workers=1)
    ref_t = torch.frombuffer(ref, dtype=torch.uint8, count=entry["nbytes"])
    assert torch.equal(ref_t.view(_E, _PER_LAYER_BYTES[0]), banks.sources["gate_up"][0])
    reader.close()


def test_ring_qd_env_flows_into_config_line(tmp_path, monkeypatch, caplog):
    out_dir = tmp_path / "ckpt"
    expect = _write_banks_ftw(out_dir)
    monkeypatch.setenv(ftw_mod._IO_BACKEND_ENV, "io_uring")
    monkeypatch.setenv(ftw_mod._IO_QD_ENV, "5")
    with caplog.at_level(logging.INFO, logger="freetoken.checkpoint.ftw"):
        banks = load_ftw_banks(str(out_dir), num_layers=_LAYERS)
    if "backend=io_uring" not in caplog.text:
        pytest.skip("fixture filesystem rejected O_DIRECT; the ring never engages (mmap fallback)")
    _assert_banks_exact(banks, expect)
    assert "backend=io_uring, qd=5" in caplog.text
    assert "pool=8," in caplog.text  # the fallback pool size stays reported in the same line


def test_ring_pin_contract_one_submit_per_bank(tmp_path, monkeypatch):
    from freetoken.moe.host_banks import PinPipeline

    out_dir = tmp_path / "ckpt"
    expect = _write_banks_ftw(out_dir)
    monkeypatch.setenv(ftw_mod._IO_BACKEND_ENV, "io_uring")
    submits = []
    real = PinPipeline.submit

    def spy(self, bank, *args, **kwargs):
        submits.append(id(bank))
        return real(self, bank, *args, **kwargs)

    monkeypatch.setattr(PinPipeline, "submit", spy)
    banks = load_ftw_banks(str(out_dir), num_layers=_LAYERS)
    _assert_banks_exact(banks, expect)
    n_banks = len(_ALPHA_NAMES) + 2 * _LAYERS + _LAYERS  # alphas + per-layer + flat layers
    assert len(submits) == n_banks
    assert set(collections.Counter(submits).values()) == {1}  # exactly one settle per bank


def test_ring_setup_failure_falls_back_to_pool(tmp_path, monkeypatch, caplog):
    out_dir = tmp_path / "ckpt"
    expect = _write_banks_ftw(out_dir)
    monkeypatch.setenv(ftw_mod._IO_BACKEND_ENV, "io_uring")
    tries = []

    def dead_ring(entries):
        tries.append(entries)
        raise ftw_uring.IoUringUnavailable(1, "Operation not permitted")

    monkeypatch.setattr(ftw_uring, "_Ring", dead_ring)
    created = []
    real_exec = ftw_mod.ThreadPoolExecutor

    class SpyExecutor(real_exec):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(ftw_mod, "ThreadPoolExecutor", SpyExecutor)
    with caplog.at_level(logging.INFO, logger="freetoken.checkpoint.ftw"):
        banks = load_ftw_banks(str(out_dir), num_layers=_LAYERS)
    _assert_banks_exact(banks, expect)
    assert len(tries) == 1  # the ring was attempted exactly once per load
    assert "io_uring unavailable" in caplog.text  # the ONE downgrade warning
    assert "backend=direct" in caplog.text  # the pool path reports what actually ran
    assert len(created) == 1  # the fallback is the ordinary one shared pool


def test_garbage_env_uses_pool(tmp_path, monkeypatch, caplog):
    out_dir = tmp_path / "ckpt"
    expect = _write_banks_ftw(out_dir)
    monkeypatch.setenv(ftw_mod._IO_BACKEND_ENV, "garbage")
    with caplog.at_level(logging.INFO, logger="freetoken.checkpoint.ftw"):
        banks = load_ftw_banks(str(out_dir), num_layers=_LAYERS)
    _assert_banks_exact(banks, expect)
    assert "ignoring unknown" in caplog.text
    assert "backend=direct" in caplog.text
    assert "backend=io_uring" not in caplog.text


def test_mmap_reader_never_builds_a_ring(tmp_path, monkeypatch, caplog):
    out_dir = tmp_path / "ckpt"
    expect = _write_banks_ftw(out_dir)
    monkeypatch.setenv(ftw_mod._IO_BACKEND_ENV, "io_uring")
    real_open = ftw_mod.os.open
    real_direct = getattr(ftw_mod.os, "O_DIRECT", 0)

    def no_direct_open(path, flags, *a, **k):
        if real_direct and flags & real_direct:
            raise OSError("simulated: O_DIRECT unsupported")
        return real_open(path, flags, *a, **k)

    monkeypatch.setattr(ftw_mod.os, "open", no_direct_open)
    built = []

    def no_ring(entries):
        built.append(entries)
        raise AssertionError("ring built in mmap mode")

    monkeypatch.setattr(ftw_uring, "_Ring", no_ring)
    with caplog.at_level(logging.INFO, logger="freetoken.checkpoint.ftw"):
        banks = load_ftw_banks(str(out_dir), num_layers=_LAYERS)
    _assert_banks_exact(banks, expect)
    assert not built
    assert "backend=mmap" in caplog.text
    assert "io_uring unavailable" not in caplog.text  # not a downgrade: the ring never applies
    assert "io_uring backend ignored under mmap fallback" in caplog.text  # A/B attribution


def test_ring_read_error_drains_and_raises(tmp_path, monkeypatch):
    from freetoken.moe.host_banks import PinPipeline

    out_dir = tmp_path / "ckpt"
    _write_banks_ftw(out_dir, shard_limit=1 << 20)  # several shards: only the last one goes bad
    monkeypatch.setenv(ftw_mod._IO_BACKEND_ENV, "io_uring")
    reader0 = FTWReader(str(out_dir))
    bad_file = reader0.shards[-1]["file"]
    reader0.close()

    real_fd = FTWReader._fd

    def bad_tail(self, file):
        if file == bad_file:
            return 999999  # never a real fd: every read on that shard reaps EBADF
        return real_fd(self, file)

    monkeypatch.setattr(FTWReader, "_fd", bad_tail)
    submits = []
    real_submit = PinPipeline.submit

    def spy_submit(self, bank, *a, **k):
        submits.append(id(bank))
        return real_submit(self, bank, *a, **k)

    monkeypatch.setattr(PinPipeline, "submit", spy_submit)
    with pytest.raises(OSError, match="EBADF"):  # re-raised after the ring drains
        load_ftw_banks(str(out_dir), num_layers=_LAYERS)
    assert len(submits) < len(_ALPHA_NAMES) + 2 * _LAYERS + _LAYERS  # failed tasks skip their settle


def test_no_thread_leak(tmp_path, monkeypatch):
    out_dir = tmp_path / "ckpt"
    expect = _write_banks_ftw(out_dir)
    monkeypatch.setenv(ftw_mod._IO_BACKEND_ENV, "io_uring")
    before = threading.active_count()
    banks = load_ftw_banks(str(out_dir), num_layers=_LAYERS)
    _assert_banks_exact(banks, expect)
    assert threading.active_count() <= before  # the ring runs on the calling thread; only growth could be ours


def test_complete_short_requeues_eof_and_error_flag():
    err = []

    # short read: requeue the remainder, no settle yet
    state = {"left": 2, "failed": False, "on_done": None, "dest": None}
    queue = collections.deque()
    rec = ftw_uring._Rec(7, 4096, 0, 8192, state)
    assert not ftw_uring._complete(rec, 4096, queue, err)
    assert list(queue) == [(7, 8192, 4096, 4096, state)]  # running offset advanced on both sides
    assert state["left"] == 2

    # EOF (res == 0): failed, error recorded, no on_done
    state2 = {"left": 1, "failed": False, "on_done": None, "dest": None}
    rec2 = ftw_uring._Rec(7, 8192, 4096, 4096, state2)
    assert ftw_uring._complete(rec2, 0, collections.deque(), err)
    assert state2["failed"] and "unexpected EOF" in str(err[-1])

    # kernel error (res < 0): failed, error recorded, no on_done
    state3 = {"left": 1, "failed": False, "on_done": None, "dest": None}
    rec3 = ftw_uring._Rec(7, 0, 0, 4096, state3)
    assert ftw_uring._complete(rec3, -5, collections.deque(), err)
    assert state3["failed"] and "EIO" in str(err[-1])

    # full read: settles; the last job of a healthy task fires on_done exactly once
    err2 = []
    done = []
    state4 = {"left": 1, "failed": False, "on_done": lambda: done.append(1), "dest": None}
    rec4 = ftw_uring._Rec(7, 0, 0, 4096, state4)
    assert ftw_uring._complete(rec4, 4096, collections.deque(), err2)
    assert done == [1] and not err2


def test_ring_batch_reads_with_sq_wrap(tmp_path):
    if _URING_SKIP:
        pytest.skip(_URING_SKIP)
    path = tmp_path / "blob.bin"
    payload = bytes(range(256)) * 256  # 64 KiB, deterministic
    path.write_bytes(payload)
    fd = os.open(str(path), os.O_RDONLY)
    try:
        ring = ftw_uring._Ring(4)  # QD 4 for 16 jobs: the SQ ring must wrap
        try:
            buf = mmap_mod.mmap(-1, len(payload))
            try:
                # a real destination: _drain maps every job's region out of this view itself
                dest = memoryview(buf)
                state = {"left": 16, "failed": False, "on_done": lambda: None, "dest": dest}
                queue = collections.deque(
                    (fd, i * 4096, i * 4096, 4096, state) for i in range(16)
                )
                err = []
                ftw_uring._drain(ring, queue, 4, err)
                assert not err and not queue
                assert bytes(buf) == payload  # every byte landed via its own SQE
                dest.release()  # drop the view export before the buffer can close
            finally:
                buf.close()
        finally:
            ring.close()
    finally:
        os.close(fd)


def test_ring_eof_reads_res_zero_past_eof(tmp_path):
    if _URING_SKIP:
        pytest.skip(_URING_SKIP)
    path = tmp_path / "blob.bin"
    path.write_bytes(b"z" * 4096)
    fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECT", 0))
    try:
        ring = ftw_uring._Ring(2)
        try:
            buf = mmap_mod.mmap(-1, 8192)
            try:
                export = (ctypes.c_char * 8192).from_buffer(buf)
                iov = ftw_uring._IoVec(ctypes.addressof(export), 8192)
                assert ring.prep_readv(fd, ctypes.addressof(iov), 4096, 0)  # fully past EOF
                ring.submit_wait(1, 1)
                cqes = ring.reap()
                assert len(cqes) == 1 and cqes[0][0] == 0
                assert cqes[0][1] == 0  # EOF lands as res == 0, which _complete turns into an error
                del export  # release the ctypes export before the buffer can close
            finally:
                buf.close()
        finally:
            ring.close()
    finally:
        os.close(fd)


def test_ring_settles_zero_job_task_alongside_reads(tmp_path):
    """Mixed plan (one reading task + one zero-job task): BOTH settles fire exactly once.
    A lost empty-settle here is the silent-pin IMA trap (invisible to CPU tests)."""
    if _URING_SKIP:
        pytest.skip(_URING_SKIP)
    path = tmp_path / "blob.bin"
    path.write_bytes(b"w" * 8192)
    fd = os.open(str(path), os.O_RDONLY)
    try:
        dest = mmap_mod.mmap(-1, 8192)
        try:
            mv = memoryview(dest)
            calls = []
            planned = [(mv, [(path.name, 0, 0, 4096)]), (mv, [])]
            tasks = [
                (mv, {"global_off": 0, "nbytes": 4096}, lambda: calls.append("read")),
                (mv, {"global_off": 0, "nbytes": 0}, lambda: calls.append("empty")),
            ]
            ftw_uring.run_ring_reads(_FakeReader(fd), planned, tasks,
                                     workers=1, chunk=4096, qd=4)
            assert bytes(dest)[:4096] == b"w" * 4096
            assert sorted(calls) == ["empty", "read"]  # exactly one settle per task
            mv.release()
        finally:
            dest.close()
    finally:
        os.close(fd)


def test_ring_on_done_error_raises_after_drain(tmp_path):
    """A failing settle must not lose the other settles nor skip the raise (pool parity)."""
    if _URING_SKIP:
        pytest.skip(_URING_SKIP)
    path = tmp_path / "blob.bin"
    path.write_bytes(b"w" * 8192)
    fd = os.open(str(path), os.O_RDONLY)
    try:
        dest = mmap_mod.mmap(-1, 8192)
        try:
            mv = memoryview(dest)
            calls = []

            def bad_settle():
                calls.append("bad")
                raise ValueError("settle boom")

            planned = [(mv, [(path.name, 0, 0, 4096)]), (mv, [])]
            tasks = [
                (mv, {"global_off": 0, "nbytes": 4096}, bad_settle),
                (mv, {"global_off": 0, "nbytes": 0}, lambda: calls.append("ok")),
            ]
            with pytest.raises(ValueError, match="settle boom"):
                ftw_uring.run_ring_reads(_FakeReader(fd), planned, tasks,
                                         workers=1, chunk=4096, qd=4)
            assert calls == ["ok", "bad"] or calls == ["bad", "ok"]  # both settles ran once
            mv.release()
        finally:
            dest.close()
    finally:
        os.close(fd)


def test_ring_rejects_non_x86(monkeypatch):
    """Plain user-space stores are only correct on x86 (TSO): weakly ordered ISAs must
    get the pool fallback, never a silently racing ring."""
    monkeypatch.setattr(ftw_uring.platform, "machine", lambda: "aarch64")
    with pytest.raises(ftw_uring.IoUringUnavailable, match="x86_64"):
        ftw_uring._Ring(4)
