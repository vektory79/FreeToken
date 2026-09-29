"""P1 bank-load read shape: ONE bounded shared read pool with layer-ordered sub-reads
capped at ``_BANK_SUB_CHUNK`` instead of the old fork of per-task nested pools (up to
128 concurrent reads; kb/cases/boot-shutdown-io P1).

Every test runs on a synthetic CPU-only FTW fixture written straight through
FTWWriter (per-layer + flat-layout + alpha entries), and holds in both reader
backends (O_DIRECT when the filesystem accepts it, mmap fallback otherwise).
The pin contract (exactly one settle per bank) is asserted, not assumed.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter

import pytest
import torch

from freetoken.checkpoint import ftw as ftw_mod
from freetoken.checkpoint.ftw import (
    FTWReader,
    FTWWriter,
    _ALPHA_NAMES,
    layer_bank_entry_name,
    load_ftw_banks,
)

_LAYERS = 4
_E = 4  # experts per per-layer bank entry
_FLAT_E = 5  # experts in the flat-layout entry (its 15 B layer window is never align-aligned)
_FLAT_ROWS = 3
_PER_LAYER_BYTES = [
    (2 << 20) + 1234,  # > 2 MiB sub-read cap: multiple sub-reads + ragged tail
    2 * (2 << 20) + 7,
    8192,
    4096,  # exactly one block
]


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    # keep the whole wave off the GPU driver: no CUDA context, pins no-op;
    # and no stray env override: the pool tests pin the default pool size
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("FREETOKEN_SKIP_BANK_PIN", "1")
    monkeypatch.delenv("FREETOKEN_BANK_POOL_WORKERS", raising=False)


def _pattern(shape, seed):
    n = 1
    for s in shape:
        n *= s
    t = (torch.arange(n, dtype=torch.int64) * 7 + seed) % 251
    return t.to(torch.uint8).view(*shape)


def _write_banks_ftw(out_dir, *, with_flat=True):
    """Per-layer entries for banks gate_up/down x4 layers, alpha vectors, and (optionally)
    one flat multi-layer entry whose per-layer windows are not align-aligned (head-pad path)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    w = FTWWriter(str(out_dir))
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
    if with_flat:
        t = _pattern((_LAYERS * _FLAT_E, _FLAT_ROWS), seed)
        w.add_tensor("gate_up_scales", t, kind="experts_bank")  # canonical role gate_up_scale
        expect["flat"] = t
    w.finalize({"quant_format": "bf16", "expert_bank_num_layers": _LAYERS})
    return expect


def test_bank_bytes_exact_under_subchunking(tmp_path):
    out_dir = tmp_path / "ckpt"
    expect = _write_banks_ftw(out_dir)
    banks = load_ftw_banks(str(out_dir), num_layers=_LAYERS)
    assert torch.equal(banks.gate_up_alpha, expect["alpha"]["gate_up_alpha"])
    assert torch.equal(banks.down_alpha, expect["alpha"]["down_alpha"])
    for (role, layer), want in expect["per_layer"].items():
        got = banks.sources[role][layer]
        assert got.dtype is torch.uint8
        assert torch.equal(got, want), (role, layer)
    flat = expect["flat"]
    for layer in range(_LAYERS):
        want = flat[layer * _FLAT_E:(layer + 1) * _FLAT_E]
        assert torch.equal(banks.sources["gate_up_scale"][layer], want), ("flat", layer)

    # cross-check one per-layer range against an independent serial read_into
    reader = FTWReader(str(out_dir))
    entry = reader.tensors[layer_bank_entry_name("gate_up", 0)]
    ref = ftw_mod._transient_buffer(entry["nbytes"])
    reader.read_into(memoryview(ref), entry, workers=1)
    ref_t = torch.frombuffer(ref, dtype=torch.uint8, count=entry["nbytes"])
    assert torch.equal(ref_t.view(_E, _PER_LAYER_BYTES[0]), banks.sources["gate_up"][0])
    reader.close()


def test_load_ftw_banks_uses_one_bounded_shared_pool(tmp_path, monkeypatch):
    out_dir = tmp_path / "ckpt"
    _write_banks_ftw(out_dir)
    created = []
    peak = {"n": 0, "peak": 0}
    lock = threading.Lock()
    real_exec = ftw_mod.ThreadPoolExecutor

    class SpyExecutor(real_exec):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

        def submit(self, fn, *args, **kwargs):
            def wrapped(*a, **kw):
                with lock:
                    peak["n"] += 1
                    peak["peak"] = max(peak["peak"], peak["n"])
                try:
                    return fn(*a, **kw)
                finally:
                    with lock:
                        peak["n"] -= 1

            return super().submit(wrapped, *args, **kwargs)

    monkeypatch.setattr(ftw_mod, "ThreadPoolExecutor", SpyExecutor)
    calls = []
    real_read_into = FTWReader.read_into

    def spy_read_into(self, dest, entry, **kwargs):
        calls.append(entry.get("name", entry["global_off"]))
        return real_read_into(self, dest, entry, **kwargs)

    monkeypatch.setattr(FTWReader, "read_into", spy_read_into)
    banks = load_ftw_banks(str(out_dir), num_layers=_LAYERS, chunk=4096)
    assert banks is not None
    assert calls == []  # the bank phase never goes through per-task read_into pools
    assert len(created) == 1  # one shared pool, not one nested pool per task
    workers = created[0]._max_workers
    assert workers == 8  # exact default (the autouse fixture delenv'd the override)
    assert peak["peak"] <= workers  # bounded concurrency; the old fork peaked near 128


def test_load_submits_jobs_layer_major(tmp_path, monkeypatch):
    out_dir = tmp_path / "ckpt"
    _write_banks_ftw(out_dir)
    offs = []
    real_exec = ftw_mod.ThreadPoolExecutor

    class SpyExecutor(real_exec):
        def submit(self, fn, *args, **kwargs):
            # _run_shared_reads submits (reader._read_job, dest, job); read jobs only
            if len(args) == 2 and isinstance(args[1], tuple) and len(args[1]) == 4:
                offs.append(args[1][1])  # file_off of (file, file_off, dest_off, length)
            return super().submit(fn, *args, **kwargs)

    monkeypatch.setattr(ftw_mod, "ThreadPoolExecutor", SpyExecutor)
    load_ftw_banks(str(out_dir), num_layers=_LAYERS)
    assert offs, "no read jobs captured"

    reader = FTWReader(str(out_dir))

    def layer_of(off):
        for name, e in reader.tensors.items():
            m = ftw_mod._LAYER_ENTRY_RE.match(name)
            if m and e["global_off"] <= off < e["global_off"] + e["nbytes"]:
                return int(m.group("layer"))
        return None  # alpha/flat-window jobs have no unambiguous layer

    seq = [l for l in (layer_of(o) for o in offs) if l is not None]
    assert seq
    assert seq == sorted(seq)  # every layer's jobs are submitted before the next layer's


def test_pin_contract_one_submit_per_bank(tmp_path, monkeypatch):
    from freetoken.moe.host_banks import PinPipeline

    out_dir = tmp_path / "ckpt"
    _write_banks_ftw(out_dir)
    submits = []
    real = PinPipeline.submit

    def spy(self, bank, *args, **kwargs):
        submits.append(id(bank))
        return real(self, bank, *args, **kwargs)

    monkeypatch.setattr(PinPipeline, "submit", spy)
    banks = load_ftw_banks(str(out_dir), num_layers=_LAYERS)
    n_banks = len(_ALPHA_NAMES) + 2 * _LAYERS + _LAYERS  # alphas + per-layer + flat layers
    assert len(submits) == n_banks
    assert set(Counter(submits).values()) == {1}  # exactly one settle per bank (note-count trap)
    assert banks.layer_residency == ["pinned"] * _LAYERS


def test_load_phase_cpu_time_not_regressed(tmp_path):
    out_dir = tmp_path / "ckpt"
    _write_banks_ftw(out_dir, with_flat=False)
    reader = FTWReader(str(out_dir))
    entries = list(reader.entries("experts_bank"))

    t0 = time.perf_counter()
    for e in entries:
        buf = ftw_mod._transient_buffer(e["nbytes"])
        reader.read_into(memoryview(buf), e, workers=1, chunk=1 << 14)
    serial = time.perf_counter() - t0
    reader.close()

    t0 = time.perf_counter()
    banks = load_ftw_banks(str(out_dir), num_layers=_LAYERS, chunk=1 << 14)
    pooled = time.perf_counter() - t0
    assert banks is not None
    assert pooled <= serial * 6.0 + 1.0  # relative guard with timing slack: pooling must not slow the phase down


def test_bank_pool_workers_env_parameterization(monkeypatch, caplog):
    f = ftw_mod._bank_pool_workers
    env = ftw_mod._BANK_POOL_WORKERS_ENV
    monkeypatch.delenv(env, raising=False)
    assert f(8) == 8  # unset -> requested
    for raw, want in (("4", 4), ("0", 1), ("-5", 1), ("32", 16), ("1", 1)):
        monkeypatch.setenv(env, raw)
        assert f(8) == want, raw  # valid passthrough; silent-clamp values land on the 1..16 edge
    monkeypatch.setenv(env, "   ")
    assert f(8) == 8  # whitespace-only = unset
    monkeypatch.setenv(env, "abc")
    with caplog.at_level(logging.WARNING, logger="freetoken.checkpoint.ftw"):
        assert f(6) == 6  # invalid -> requested, never a crash
    assert "abc" in caplog.text
    monkeypatch.setenv(env, "32")
    with caplog.at_level(logging.WARNING, logger="freetoken.checkpoint.ftw"):
        assert f(8) == 16
    assert "clamped" in caplog.text  # clamping warns, it is not silent


def test_bank_reads_config_logged(tmp_path, caplog):
    out_dir = tmp_path / "ckpt"
    _write_banks_ftw(out_dir, with_flat=False)
    with caplog.at_level(logging.INFO, logger="freetoken.checkpoint.ftw"):
        banks = load_ftw_banks(str(out_dir), num_layers=_LAYERS)
    assert banks is not None
    text = caplog.text
    assert "10 tasks" in text  # 2 alphas + 2 banks x 4 layers
    assert "pool=8" in text  # default pool (env delenv'd by the autouse fixture)
    assert "sub-chunk=2048 KiB" in text  # min(_DEFAULT_CHUNK, _BANK_SUB_CHUNK)
    assert "backend=" in text  # direct|mmap, per what the probe settled


@pytest.mark.parametrize("exc_cls", [OSError, KeyboardInterrupt])
def test_read_job_failure_drains_and_raises(tmp_path, monkeypatch, exc_cls):
    out_dir = tmp_path / "ckpt"
    _write_banks_ftw(out_dir, with_flat=False)
    n_tasks = len(_ALPHA_NAMES) + 2 * _LAYERS
    chunk = min(ftw_mod._DEFAULT_CHUNK, ftw_mod._BANK_SUB_CHUNK)
    reader0 = FTWReader(str(out_dir))
    planned_total, target = 0, None
    for e in reader0.entries("experts_bank"):
        jobs = reader0._plan_jobs(memoryview(b""), e, chunk)
        planned_total += len(jobs)
        if target is None and len(jobs) >= 2:
            target = jobs[1]  # exactly one deterministic job fails
    reader0.close()
    assert target is not None

    executed = []
    real_job = FTWReader._read_job

    def flaky(self, dest, job):
        executed.append(job)
        if job == target:
            raise exc_cls("boom")
        return real_job(self, dest, job)

    monkeypatch.setattr(FTWReader, "_read_job", flaky)
    from freetoken.moe.host_banks import PinPipeline

    submits = []
    real_submit = PinPipeline.submit

    def spy_submit(self, bank, *a, **k):
        submits.append(id(bank))
        return real_submit(self, bank, *a, **k)

    monkeypatch.setattr(PinPipeline, "submit", spy_submit)
    with pytest.raises(exc_cls, match="boom"):  # re-raised as-is, not wrapped
        load_ftw_banks(str(out_dir), num_layers=_LAYERS)
    assert len(executed) == planned_total  # the pool drains: every planned job ran
    assert len(submits) == n_tasks - 1  # only the failed task skips its settle
    assert not [t for t in threading.enumerate() if "ftw-bank-read" in t.name]


def test_mmap_backend_parity(tmp_path, monkeypatch, caplog):
    out_dir = tmp_path / "ckpt"
    expect = _write_banks_ftw(out_dir)
    real_open = ftw_mod.os.open
    real_direct = getattr(ftw_mod.os, "O_DIRECT", 0)

    def no_direct_open(path, flags, *a, **k):
        if real_direct and flags & real_direct:
            raise OSError("simulated: O_DIRECT unsupported")
        return real_open(path, flags, *a, **k)

    monkeypatch.setattr(ftw_mod.os, "open", no_direct_open)
    created = []
    real_exec = ftw_mod.ThreadPoolExecutor

    class SpyExecutor(real_exec):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(ftw_mod, "ThreadPoolExecutor", SpyExecutor)
    with caplog.at_level(logging.INFO, logger="freetoken.checkpoint.ftw"):
        banks = load_ftw_banks(str(out_dir), num_layers=_LAYERS)
    assert "backend=mmap" in caplog.text  # the probe was forced onto the fallback
    assert len(created) == 1  # the fallback path also goes through the one shared pool
    for (role, layer), want in expect["per_layer"].items():
        assert torch.equal(banks.sources[role][layer], want), (role, layer)
    for layer in range(_LAYERS):
        want = expect["flat"][layer * _FLAT_E:(layer + 1) * _FLAT_E]
        assert torch.equal(banks.sources["gate_up_scale"][layer], want), ("flat", layer)
