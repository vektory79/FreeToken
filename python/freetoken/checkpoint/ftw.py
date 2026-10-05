"""FreeToken Weight (FTW) checkpoint: one O_DIRECT-friendly on-disk format for a whole model.

The format is a single *logical contiguous byte region* of all tensors, sliced *physically*
into shard files of at most ``shard_limit`` bytes (default 8 GiB, for HF/filesystem
friendliness). It exists because reading the original safetensors back fast is awkward:
tensors are packed with no alignment, so an individual tensor can't be O_DIRECT-read at an
arbitrary offset; and the earlier per-bank cache prototype worked around that by giving every expert bank
its own file -- which doesn't cover dense weights and turns a model's long tail of tiny
tensors (norms, biases, router) into hundreds of tiny I/Os.

FTW fixes both:

* **Aligned.** Every tensor starts at a 4096-aligned region offset and is padded to 4096;
  shards are cut at 4096-aligned boundaries. So any tensor (or any shard-local slice of one)
  is read with offset, length (rounded up to 4096), and destination all block-aligned --
  exactly what O_DIRECT requires. A tensor larger than a shard simply spans shards; because
  both its start and the shard boundary are aligned, each piece stays aligned.
* **Unified.** It holds dense weights as ``kind="weight"`` (exactly what a model's
  ``iter_weights`` yields -- post fusion/TP-shard, fed straight to ``load_state_dict``) and
  the offload expert state as ``kind="experts_bank"`` (post backend-repack -- the per-expert
  weight banks plus, distinguished only by their reserved names, the alpha scale vectors;
  the FTW content). The converter runs the per-model loaders once; this reader is
  model-agnostic.

Layout on disk::

    <dir>/freetoken_weight.json        # index: tensors[] + shards[] + meta
    <dir>/freetoken-00000.ftw         # the byte region, sliced <= shard_limit
    <dir>/freetoken-00001.ftw
    <dir>/config.json, tokenizer*, ...# copied so the dir is a self-contained checkpoint
"""

from __future__ import annotations

import json
import math
import mmap
import os
import re
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

INDEX_NAME = "freetoken_weight.json"
FORMAT_TAG = "freetoken_weight"
FORMAT_VERSION = 1
ALIGN = 4096  # O_DIRECT block alignment (== page size on this platform)
DEFAULT_SHARD_LIMIT = 8 << 30  # 8 GiB; must be a multiple of ALIGN
_SHARD_FMT = "freetoken-{:05d}.ftw"
_DEFAULT_CHUNK = 8 << 20
# Bank-load read shape (P1): ONE shared bounded pool instead of a per-task nested-pool
# fork. Sub-reads are capped at _BANK_SUB_CHUNK so the pool's few workers always have a
# deep queue of ~1-2 MiB preadv in flight (the fio 1xQD32x1M precedent) rather than a
# few 8 MiB ones; FREETOKEN_BANK_POOL_WORKERS overrides the pool size for the iron A/B.
_BANK_SUB_CHUNK = 2 << 20
_BANK_POOL_WORKERS_ENV = "FREETOKEN_BANK_POOL_WORKERS"
# Bank-read backend (P7): io_uring keeps QD 32-64 reads in flight from one thread where a
# blocking-preadv pool worker holds QD~1 (the fio 1xQD32x1M = 6.45 GB/s precedent);
# FREETOKEN_FTW_IO_BACKEND=io_uring|auto|direct opts in, garbage -> default + warning.
_IO_BACKEND_ENV = "FREETOKEN_FTW_IO_BACKEND"
_IO_QD_ENV = "FREETOKEN_FTW_IO_QD"
_IO_QD_DEFAULT = 32
_IO_BACKENDS = ("io_uring", "auto", "direct")
_ALPHA_NAMES = ("gate_up_alpha", "down_alpha")
# Per-layer expert-bank entry name (converter streaming path, see checkpoint/convert.py):
# each layer of a bank is its own FTW tensor instead of one flat [num_layers*E, ...] region.
_LAYER_ENTRY_RE = re.compile(r"^(?P<base>.+)#L(?P<layer>\d{5})$")


def layer_bank_entry_name(bank_name: str, layer_id: int) -> str:
    """Name of one per-layer ``experts_bank`` FTW entry; :func:`load_ftw_banks` groups
    entries matching ``_LAYER_ENTRY_RE`` back into a per-layer bank list by base name."""
    return f"{bank_name}#L{layer_id:05d}"


def _pread_into(fd: int, mv: memoryview, offset: int) -> None:
    """POSIX positional read into ``mv`` at ``offset``, looping over any short preadv.

    preadv may return short (a signal, or the EOF-adjacent tail); the loop resumes
    at the running offset, which stays O_DIRECT-legal: the writer pads every tensor
    to ALIGN and cuts shards at ALIGN boundaries, so direct-IO short reads land on
    block boundaries. EOF before the buffer is filled raises ``OSError`` — a
    truncated shard must not silently load garbage weights."""
    done = 0
    total = len(mv)
    while done < total:
        n = os.preadv(fd, [mv[done:]], offset + done)
        if n == 0:
            raise OSError(
                f"unexpected EOF reading FTW: got {done}/{total} bytes at offset {offset}"
            )
        done += n


def _align_up(n: int, a: int = ALIGN) -> int:
    return (n + a - 1) // a * a


def _dtype_str(dt: torch.dtype) -> str:
    return str(dt).removeprefix("torch.")


def _dtype_of(s: str) -> torch.dtype:
    return getattr(torch, s)


def _elsize(dt: torch.dtype) -> int:
    return torch.empty((), dtype=dt).element_size()


def is_ftw_checkpoint(path: str) -> bool:
    """True if ``path`` is a directory holding a FreeToken Weight (FTW) index."""
    return os.path.isfile(os.path.join(path, INDEX_NAME))


def ftw_tensor_names(path: str, *kinds: str) -> list[str]:
    """Names the FTW index lists for ``kinds`` (every kind when none is given)."""
    keep = set(kinds)
    with open(os.path.join(path, INDEX_NAME)) as f:
        return [t["name"] for t in json.load(f)["tensors"] if not keep or t["kind"] in keep]


def ftw_quant_format(path: str) -> str | None:
    """The ``quant_format`` an FTW checkpoint's expert banks were packed for; None when ``path`` is not an FTW checkpoint or holds no banks."""
    if not is_ftw_checkpoint(path):
        return None
    with open(os.path.join(path, INDEX_NAME)) as f:
        return json.load(f).get("quant_format")


# ============================== writer ==============================
class FTWWriter:
    """Stream tensors into the FTW, rolling shard files at ``shard_limit``.

    Tensors are written in call order into one logical byte stream; each is padded to
    ``ALIGN`` so the next starts aligned. A tensor that doesn't fit the current shard's
    remaining room is split across shards (the split point is the shard boundary, which is
    aligned). Call :meth:`add_tensor` for each tensor, then :meth:`finalize`.
    """

    def __init__(self, out_dir: str, *, shard_limit: int = DEFAULT_SHARD_LIMIT):
        assert shard_limit % ALIGN == 0, "shard_limit must be a multiple of ALIGN"
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.shard_limit = shard_limit
        self._tensors: list[dict] = []
        self._shards: list[dict] = []
        self._global = 0  # running FTW offset (incl. padding)
        self._f = None  # current shard file handle
        self._shard_idx = -1
        self._shard_start = 0  # FTW offset where the current shard began
        self._cur = 0  # bytes written to the current shard

    def _roll(self) -> None:
        if self._f is not None:
            self._shards.append({"file": _SHARD_FMT.format(self._shard_idx),
                                 "global_off": self._shard_start, "nbytes": self._cur})
            self._f.close()
        self._shard_idx += 1
        self._shard_start = self._global
        self._cur = 0
        self._f = open(os.path.join(self.out_dir, _SHARD_FMT.format(self._shard_idx)), "wb")

    def _write_raw(self, data: memoryview) -> None:
        """Write ``data`` into the FTW byte stream, splitting across shards at the limit."""
        if self._f is None:
            self._roll()
        off = 0
        n = len(data)
        while off < n:
            if self._cur == self.shard_limit:
                self._roll()
            take = min(n - off, self.shard_limit - self._cur)
            self._f.write(data[off:off + take])
            off += take
            self._cur += take
            self._global += take

    def add_tensor(self, name: str, tensor: torch.Tensor, kind: str = "weight") -> None:
        t = tensor.detach().cpu().contiguous()
        raw = t.reshape(-1).view(torch.uint8)
        nbytes = int(raw.numel())
        # A small tensor (<= shard) never splits: roll early so it lands whole in one shard.
        if self._f is None or (nbytes <= self.shard_limit
                               and self._cur + nbytes > self.shard_limit):
            self._roll()
        global_off = self._global
        assert global_off % ALIGN == 0, "tensor start must be aligned (invariant)"
        self._write_raw(memoryview(raw.numpy()))
        self._tensors.append({"name": name, "kind": kind, "dtype": _dtype_str(t.dtype),
                              "shape": list(t.shape), "global_off": global_off, "nbytes": nbytes})
        # pad to ALIGN so the next tensor starts aligned
        pad = _align_up(self._global) - self._global
        if pad:
            self._write_raw(memoryview(bytes(pad)))

    def finalize(self, meta: dict) -> dict:
        if self._f is not None:
            self._shards.append({"file": _SHARD_FMT.format(self._shard_idx),
                                 "global_off": self._shard_start, "nbytes": self._cur})
            self._f.close()
            self._f = None
        index = {"format": FORMAT_TAG, "version": FORMAT_VERSION, "align": ALIGN,
                 "shard_limit": self.shard_limit, "total_bytes": self._global,
                 "tensors": self._tensors, "shards": self._shards, **meta}
        tmp = os.path.join(self.out_dir, INDEX_NAME + ".tmp")
        with open(tmp, "w") as f:
            json.dump(index, f)
        os.replace(tmp, os.path.join(self.out_dir, INDEX_NAME))
        return index


# ============================== reader ==============================
class FTWReader:
    """Random-access reader over an FTW checkpoint.

    Maps a tensor's logical byte range to one-or-more shard-file ranges (split at shard
    boundaries) and reads each piece with chunked multi-threaded O_DIRECT directly into the
    destination buffer. Offsets/lengths are all 4096-aligned (lengths rounded up into the
    rounded-up destination), so O_DIRECT is always legal -- including the tail of a tensor
    (the rounding reads into the region's padding, which is discarded by the tensor view)."""

    def __init__(self, path: str):
        with open(os.path.join(path, INDEX_NAME)) as f:
            self.index = json.load(f)
        assert self.index.get("format") == FORMAT_TAG, f"not a {FORMAT_TAG}: {path}"
        self.dir = path
        self.shards = sorted(self.index["shards"], key=lambda s: s["global_off"])
        self.tensors = {t["name"]: t for t in self.index["tensors"]}
        self._fds: dict[str, int] = {}
        self._maps: dict[str, tuple[mmap.mmap, memoryview]] = {}
        # O_DIRECT (DMA straight from disk, bypassing the page cache) is the fast path but a
        # perf choice, not a correctness one. Some filesystems reject it at open with EINVAL
        # (tmpfs, many overlay/network mounts) and the flag is Linux-only; when it's absent
        # we fall back to mmap (below), NOT to chunked buffered preadv -- a whole-shard
        # mapping + kernel readahead copies far faster than per-chunk page-cache reads.
        # 0 here means "O_DIRECT unavailable -> use the mmap path".
        self._direct = getattr(os, "O_DIRECT", 0)
        self._probed = False
        self._lock = threading.Lock()  # load_ftw_banks calls read_into concurrently

    def meta(self, key: str, default=None):
        return self.index.get(key, default)

    def entries(self, *kinds: str) -> list[dict]:
        keep = set(kinds)
        return [t for t in self.index["tensors"] if not keep or t["kind"] in keep]

    def _ensure_mode(self) -> None:
        """Resolve the read backend once: keep O_DIRECT if the filesystem accepts it, else
        drop to the mmap fallback. Thread-safe -- ``_probed`` is published only after
        ``_direct`` is final, so a concurrent reader never races onto a stale direct path."""
        if self._probed:
            return
        with self._lock:
            if self._probed:
                return
            if self._direct and self.shards:
                try:
                    os.close(os.open(os.path.join(self.dir, self.shards[0]["file"]),
                                     os.O_RDONLY | self._direct))
                except OSError:
                    self._direct = 0
                    logger.warning("O_DIRECT unsupported on %s; using mmap fallback for "
                                   "FTW load", self.dir)
            self._probed = True

    def _fd(self, file: str) -> int:
        fd = self._fds.get(file)
        if fd is None:
            with self._lock:  # first-open only; chunk reads reuse the cached fd lock-free
                fd = self._fds.get(file)
                if fd is None:
                    fd = os.open(os.path.join(self.dir, file), os.O_RDONLY | self._direct)
                    self._fds[file] = fd
        return fd

    def _map(self, file: str) -> memoryview:
        entry = self._maps.get(file)
        if entry is None:
            with self._lock:
                entry = self._maps.get(file)
                if entry is None:
                    fd = os.open(os.path.join(self.dir, file), os.O_RDONLY)
                    try:
                        m = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
                    finally:
                        os.close(fd)  # the mapping keeps its own reference to the file
                    try:
                        m.madvise(mmap.MADV_SEQUENTIAL)  # kernel readahead for streaming
                    except (AttributeError, OSError):
                        pass
                    entry = (m, memoryview(m))
                    self._maps[file] = entry
        return entry[1]

    def close(self) -> None:
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()
        for m, mv in self._maps.values():
            mv.release()
            m.close()
        self._maps.clear()

    def _pieces(self, global_off: int, nbytes: int):
        """Yield (file, file_off, dest_off, length) covering [global_off, +nbytes),
        split at shard boundaries. All file_off/dest_off are ALIGN-aligned."""
        dest_off = 0
        remaining = nbytes
        pos = global_off
        for sh in self.shards:
            s0, s1 = sh["global_off"], sh["global_off"] + sh["nbytes"]
            if pos >= s1 or remaining <= 0:
                continue
            if pos < s0:  # regions are contiguous; a gap means a corrupt index
                raise ValueError("FTW gap / misordered shards")
            take = min(remaining, s1 - pos)
            yield sh["file"], pos - s0, dest_off, take
            pos += take
            dest_off += take
            remaining -= take
        if remaining:
            raise ValueError("tensor range exceeds FTW shards")

    def _plan_jobs(self, dest: memoryview, entry: dict, chunk: int) -> list[tuple]:
        """Split one entry's byte range into (file, file_off, dest_off, length) read
        jobs -- all ALIGN-aligned, the tail rounded up into the region's padding."""
        jobs = []
        for file, file_off, dest_off, length in self._pieces(entry["global_off"], entry["nbytes"]):
            rlen = _align_up(length)  # round the tail up; padding is in-region, harmless
            for c in range(0, rlen, chunk):
                jobs.append((file, file_off + c, dest_off + c, min(chunk, rlen - c)))
        return jobs

    def _read_job(self, dest: memoryview, job: tuple) -> None:
        """Execute one planned read job -- the shared code of every read path."""
        file, fo, do, ln = job
        if self._direct:
            try:
                _pread_into(self._fd(file), dest[do:do + ln], fo)
            except OSError as e:
                raise OSError(f"shard {file}: {e}") from e
        else:
            mv = self._map(file)
            if fo + ln > len(mv):
                raise OSError(
                    f"unexpected EOF reading FTW: shard {file} has "
                    f"{len(mv)} bytes, need {ln} at offset {fo}"
                )
            dest[do:do + ln] = mv[fo:fo + ln]

    def read_into(self, dest: memoryview, entry: dict, *, workers: int = 8,
                  chunk: int = _DEFAULT_CHUNK) -> None:
        """Read one tensor's bytes into ``dest`` (length >= entry nbytes rounded to ALIGN)."""
        self._ensure_mode()
        jobs = self._plan_jobs(dest, entry, chunk)

        # Open/map each distinct shard once, single-threaded, so the pool only reuses handles.
        touch = self._fd if self._direct else self._map
        for file in {j[0] for j in jobs}:
            touch(file)

        if len(jobs) <= 1:
            for j in jobs:
                self._read_job(dest, j)
        else:
            with ThreadPoolExecutor(workers) as ex:
                list(ex.map(lambda j: self._read_job(dest, j), jobs))


def _bank_pool_workers(requested: int) -> int:
    """Size of the one shared bank-read pool; FREETOKEN_BANK_POOL_WORKERS overrides for the iron A/B sweep."""
    raw = os.environ.get(_BANK_POOL_WORKERS_ENV, "").strip()
    if not raw:
        return requested
    try:
        wanted = int(raw)
    except ValueError:
        logger.warning(
            f"ignoring non-integer {_BANK_POOL_WORKERS_ENV}={raw!r}; using pool={requested}"
        )
        return requested
    clamped = max(1, min(16, wanted))
    if clamped != wanted:  # 0/-5 -> 1, 64 -> 16: never clamp silently
        logger.warning(
            f"{_BANK_POOL_WORKERS_ENV}={raw!r} out of range 1..16, clamped to pool={clamped}"
        )
    return clamped


def _resolve_io_backend() -> str:
    """P7 bank-read backend; FREETOKEN_FTW_IO_BACKEND=io_uring|auto|direct (unset -> direct).
    ``auto`` is an alias of ``io_uring``: same ring attempt, same one-warning fallback."""
    raw = os.environ.get(_IO_BACKEND_ENV, "").strip()
    if not raw:
        return "direct"
    if raw in _IO_BACKENDS:
        return raw
    logger.warning(f"ignoring unknown {_IO_BACKEND_ENV}={raw!r}; using backend=direct")
    return "direct"


def _io_qd() -> int:
    """Queue depth of the io_uring ring; FREETOKEN_FTW_IO_QD overrides (1..128, clamped with warning)."""
    raw = os.environ.get(_IO_QD_ENV, "").strip()
    if not raw:
        return _IO_QD_DEFAULT
    try:
        wanted = int(raw)
    except ValueError:
        logger.warning(f"ignoring non-integer {_IO_QD_ENV}={raw!r}; using qd={_IO_QD_DEFAULT}")
        return _IO_QD_DEFAULT
    clamped = max(1, min(128, wanted))
    if clamped != wanted:  # 0/-3 -> 1, 999 -> 128: never clamp silently
        logger.warning(f"{_IO_QD_ENV}={raw!r} out of range 1..128, clamped to qd={clamped}")
    return clamped


def _run_shared_reads(reader: FTWReader, tasks, *, workers: int, chunk: int) -> None:
    """Read every ``(dest, entry, on_done)`` task through ONE bounded queue: the io_uring
    ring when FREETOKEN_FTW_IO_BACKEND opts in (QD-deep, one thread), else the shared pool.

    The queue holds all sub-read jobs at once (the deep queue), so the submitter always
    has the next read ready; jobs are pulled in submission order, which the caller submits
    layer-major, so the concurrent reads land on nearby FTW regions instead of fanning a
    nested pool per task across every shard. ``on_done`` runs exactly once per task after
    its last job completes (the pin step); a failed job skips it, and the first error is
    re-raised after the queue drains."""
    reader._ensure_mode()
    planned = [(dest, reader._plan_jobs(dest, entry, chunk)) for dest, entry, _ in tasks]
    touch = reader._fd if reader._direct else reader._map
    for file in {j[0] for _, jobs in planned for j in jobs}:
        touch(file)

    backend = _resolve_io_backend()
    if backend != "direct" and reader._direct:
        try:
            from freetoken.checkpoint import ftw_uring
        except ImportError as exc:
            logger.warning(
                f"{_IO_BACKEND_ENV}={backend}: io_uring module unavailable ({exc}); using pool fallback"
            )
        else:
            try:
                ftw_uring.run_ring_reads(reader, planned, tasks, workers=workers, chunk=chunk, qd=_io_qd())
                return
            except ftw_uring.IoUringUnavailable as exc:
                logger.warning(
                    f"{_IO_BACKEND_ENV}={backend}: io_uring unavailable ({exc}); using pool fallback"
                )
    elif backend != "direct":
        logger.warning(
            f"{_IO_BACKEND_ENV}={backend}: io_uring backend ignored under mmap fallback"
        )

    logger.info(
        f"FTW bank reads: {len(tasks)} tasks, {sum(len(j) for _, j in planned)} jobs, "
        f"pool={workers}, sub-chunk={chunk // 1024} KiB, "
        f"backend={'direct' if reader._direct else 'mmap'}"
    )

    err: list[BaseException] = []
    ex = ThreadPoolExecutor(workers, thread_name_prefix="ftw-bank-read")
    try:
        for (dest, jobs), (_dest, _entry, on_done) in zip(planned, tasks):
            if not jobs:  # a zero-byte entry: settle it exactly like a read one
                on_done()
                continue
            state = {"left": len(jobs), "failed": False}
            lock = threading.Lock()

            def _after(fut, state=state, lock=lock, on_done=on_done):
                exc = fut.exception()
                with lock:
                    state["failed"] |= exc is not None
                    state["left"] -= 1
                    last, failed = state["left"] == 0, state["failed"]
                if exc is not None and not err:
                    err.append(exc)
                if last and not failed:
                    try:
                        on_done()
                    except BaseException as exc2:
                        err.append(exc2)

            for job in jobs:
                ex.submit(reader._read_job, dest, job).add_done_callback(_after)
    finally:
        ex.shutdown(wait=True)
    if err:
        raise err[0]


def _transient_buffer(nbytes: int) -> mmap.mmap:
    return mmap.mmap(-1, _align_up(nbytes))


def iter_ftw_weights(path: str, *, kinds=("weight",), keep: Callable[[str], bool] | None = None,
                       workers: int = 8, chunk: int = _DEFAULT_CHUNK, prefetch: int = 2):
    """Yield ``(name, host_tensor)`` for the requested kinds, reading each tensor via
    chunked O_DIRECT. A background thread prefetches the next ``prefetch`` tensors so the
    disk stays busy while the consumer copies the current one to the GPU. Transient buffers
    are freed as the consumer advances (peak host mem ~ prefetch+1 tensors). An entry whose
    name ``keep`` rejects is dropped before any of its bytes are read."""
    import queue
    import threading

    from freetoken.utils.progress import byte_bar

    reader = FTWReader(path)
    entries = reader.entries(*kinds)
    if keep is not None:
        entries = [e for e in entries if keep(e["name"])]
    q: queue.Queue = queue.Queue(maxsize=max(1, prefetch))
    _DONE = object()
    err: list[BaseException] = []
    cancel = threading.Event()

    def _put(item) -> bool:
        # A plain q.put would deadlock teardown: if the consumer stops with the queue
        # full (early break out of the generator, or an exception mid-load), close()
        # runs the finally below, which joins this thread while it waits for queue
        # space forever. Poll the cancel flag instead of blocking indefinitely.
        while not cancel.is_set():
            try:
                q.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _producer():
        try:
            for e in entries:
                buf = _transient_buffer(e["nbytes"])
                reader.read_into(memoryview(buf), e, workers=workers, chunk=chunk)
                dt = _dtype_of(e["dtype"])
                t = torch.frombuffer(buf, dtype=dt, count=e["nbytes"] // _elsize(dt))
                # a 0-d entry (a per-tensor scale) comes back 0-d, not [1]
                if not _put((e["name"], t.view(*e["shape"]) if e["shape"] else t.view(()), buf, e["nbytes"])):
                    return
        except BaseException as ex:  # surface to consumer
            err.append(ex)
        finally:
            _put(_DONE)

    th = threading.Thread(target=_producer, name="FTW-prefetch", daemon=True)
    th.start()
    bar = byte_bar(sum(e["nbytes"] for e in entries), "Loading weights (FTW)")
    try:
        while True:
            item = q.get()
            if item is _DONE:
                break
            name, tensor, buf, nbytes = item
            yield name, tensor
            bar.update(nbytes)
            del tensor, buf  # buffer reclaimable once the consumer drops the tensor
    finally:
        bar.close()
        cancel.set()
        th.join()
        reader.close()
    if err:
        raise err[0]


def _split_bank_entries(reader: FTWReader, path: str, num_layers: int):
    """``(all, alphas, flat regions, {base: {layer: entry}})`` of the ``experts_bank`` entries; None when there are none."""
    bank_entries = reader.entries("experts_bank")
    if not bank_entries:
        return None

    alpha_entries = [e for e in bank_entries if e["name"] in _ALPHA_NAMES]
    row_entries = [e for e in bank_entries if e["name"] not in _ALPHA_NAMES]

    meta_layers = reader.meta("expert_bank_num_layers")
    if meta_layers is not None and meta_layers != num_layers:
        raise RuntimeError(
            f"{path!r} was converted with {meta_layers} expert-bank layers but the "
            f"model config says num_moe_layers={num_layers}; the checkpoint does not "
            "match its config"
        )

    # Split row entries into the two layouts by name.
    flat_entries: list[dict] = []
    per_layer_groups: dict[str, dict[int, dict]] = {}
    for e in row_entries:
        m = _LAYER_ENTRY_RE.match(e["name"])
        if m is None:
            flat_entries.append(e)
            continue
        per_layer_groups.setdefault(m.group("base"), {})[int(m.group("layer"))] = e

    mixed = {e["name"] for e in flat_entries} & per_layer_groups.keys()
    assert not mixed, f"FTW bank(s) mix flat and per-layer row layouts: {sorted(mixed)}"
    for base, by_layer in per_layer_groups.items():
        assert sorted(by_layer) == list(range(num_layers)), (
            f"FTW bank {base!r} has per-layer entries for layers {sorted(by_layer)}, "
            f"expected exactly range({num_layers})"
        )
    return bank_entries, alpha_entries, flat_entries, per_layer_groups


def load_ftw_banks(
    path: str, *, num_layers: int, workers: int = 8, chunk: int = _DEFAULT_CHUNK,
    layer_residency: list[str] | None = None,
):
    """Reconstruct the offload :class:`ExpertBanks` from the FTW's ``experts_bank``
    entries, on the per-layer host bank contract (one ``[num_experts, ...]``
    HostBank per layer per bank; see ``moe.offload_cache.set_bank_sources``).

    Reads go through ONE bounded shared pool (``workers`` threads, sub-reads capped at
    ``_BANK_SUB_CHUNK``, tasks submitted layer-major); ``FREETOKEN_BANK_POOL_WORKERS``
    overrides the pool size. The pin contract (exactly one settle per bank) is unchanged.

    ``layer_residency`` (default: all pinned) settles each layer's banks per its ``HostResidency`` label as reads complete: PINNED -> cudaHostRegister, LOCKED -> mlock (CPU-executor resident, no pin quota spent).
    The applied labels are echoed back on ``ExpertBanks.layer_residency``.

    Two on-disk row layouts, distinguished per bank name (a file never mixes them for
    the same name -- checked below):

    * **Flat region** (pre-existing files, and non-streamable formats): one entry per
      bank, ONE contiguous ``[num_layers * num_experts, ...]`` region. ``num_layers``
      isn't part of that region's shape, so the caller passes it
      (``ModelConfig.num_moe_layers`` -- FTW checkpoints carry the model's config.json);
      the ``expert_bank_num_layers`` index meta the converter records is used as a
      cross-check when present. A layer's byte range within the region generally is
      NOT 4096-aligned (only the whole region's start is guaranteed aligned) -- read it
      via its ALIGNED enclosing window ``[align_down(off), align_up(off+len))`` into a
      page-aligned scratch HostBank, and view the real per-layer tensor as a
      head-offset slice.
    * **Per-layer** (streamable-format conversion, see :mod:`freetoken.checkpoint.convert`):
      one entry per ``(bank, layer)``, name ``f"{bank_name}#L{layer_id:05d}"``. Each was
      written by its own ``add_tensor`` call, so its start is already ALIGN-aligned --
      no windowing/head-pad needed, read straight into a HostBank shaped like the entry.

    Alphas (``gate_up_alpha``/``down_alpha``) stay flat ``[num_layers*num_experts]``
    vectors, unaffected by the row split (fixed GPU residency; see
    ``cache_budget.expert_bytes_per_slot``).
    """
    from freetoken.moe.host_banks import (
        HostBank, HostResidency, PinPipeline, alloc_banks, born_pinned_default,
    )
    from freetoken.utils.progress import byte_bar

    residency = layer_residency or [HostResidency.PINNED.value] * num_layers
    assert len(residency) == num_layers, (len(residency), num_layers)

    # PINNED layers are born-pinned (cudaHostAlloc) where that wins (see born_pinned_default); LOCKED/PAGEABLE layers stay lazy mmaps
    born = born_pinned_default()

    def _backing(layer_id: int) -> str:
        if born and residency[layer_id] == HostResidency.PINNED.value:
            return "cuda"
        return "mmap"

    reader = FTWReader(path)
    try:
        split = _split_bank_entries(reader, path, num_layers)
    except BaseException:
        reader.close()
        raise
    if split is None:
        reader.close()
        return None
    bank_entries, alpha_entries, flat_entries, per_layer_groups = split

    # gguf banks: role-ordered (gate, up, down) ggml type per bank layer, persisted by
    # the converter; absent -> None for every other format (nvfp4 loads exactly as before)
    quant_format = reader.meta("quant_format")
    meta_gguf_types = reader.meta("gguf_types")
    if quant_format == "gguf" and meta_gguf_types is None:
        reader.close()
        raise RuntimeError(
            f"{path!r} packs gguf expert banks but its index has no gguf_types: it was "
            "converted by an older build; re-convert with ft checkpoint"
        )
    if meta_gguf_types is not None:
        if (
            not isinstance(meta_gguf_types, list)
            or len(meta_gguf_types) != num_layers
            or not all(
                isinstance(row, (list, tuple))
                and len(row) == 3
                and all(isinstance(t, int) for t in row)
                for row in meta_gguf_types
            )
        ):
            reader.close()
            raise RuntimeError(
                f"{path!r} records malformed gguf_types meta (one (gate, up, down) int "
                f"triple per MoE layer expected, got {meta_gguf_types!r}); the checkpoint "
                "does not match its config"
            )
        meta_gguf_types = tuple(tuple(int(t) for t in row) for row in meta_gguf_types)

    # Alphas: unchanged, one flat HostBank per entry.
    alpha_specs = {e["name"]: (tuple(e["shape"]), _dtype_of(e["dtype"])) for e in alpha_entries}
    alpha_hb = alloc_banks(alpha_specs)

    # Row banks: one padded-window HostBank per (name, layer_id) for the flat layout, plus
    # how to carve the real [num_experts, *row_shape] tensor out of its head; ``None`` marks
    # a per-layer entry (direct view, no carving needed).
    row_hb: dict[str, list] = {}
    row_view_args: dict[str, list] = {}
    row_jobs = []  # (name, HostBank, window_off, window_len, layer_bytes) -- flat layout
    layer_jobs = []  # (name, HostBank, entry) -- per-layer layout, direct aligned read

    for e in flat_entries:
        name = e["name"]
        total, *row_shape = e["shape"]
        assert total % num_layers == 0, (name, total, num_layers)
        num_experts = total // num_layers
        dtype = _dtype_of(e["dtype"])
        row_bytes = (math.prod(row_shape) if row_shape else 1) * _elsize(dtype)
        layer_bytes = num_experts * row_bytes
        assert layer_bytes * num_layers == e["nbytes"], (name, layer_bytes, num_layers, e["nbytes"])
        row_hb[name] = []
        row_view_args[name] = []
        for layer_id in range(num_layers):
            off = e["global_off"] + layer_id * layer_bytes
            win_off = (off // ALIGN) * ALIGN
            win_end = _align_up(off + layer_bytes)
            head_pad = off - win_off
            bank = HostBank((win_end - win_off,), torch.uint8, backing=_backing(layer_id))
            row_hb[name].append(bank)
            row_view_args[name].append((head_pad, layer_bytes, num_experts, tuple(row_shape), dtype))
            row_jobs.append((name, bank, win_off, win_end - win_off, layer_bytes, layer_id))

    for base, by_layer in per_layer_groups.items():
        row_hb[base] = []
        row_view_args[base] = []
        for layer_id in range(num_layers):
            e = by_layer[layer_id]
            assert e["global_off"] % ALIGN == 0, (base, layer_id, e["global_off"])  # writer invariant
            bank = HostBank(tuple(e["shape"]), _dtype_of(e["dtype"]), backing=_backing(layer_id))
            row_hb[base].append(bank)
            row_view_args[base].append(None)
            layer_jobs.append((base, bank, e, layer_id))

    total_bytes = sum(e["nbytes"] for e in bank_entries)
    bar = byte_bar(total_bytes, "Loading expert banks (FTW)")

    # Tasks are submitted layer-major (alphas first, they are tiny): each worker of the
    # one shared pool keeps reading nearby FTW regions instead of fanning a nested pool
    # per task across every shard. Each bank settles exactly once as its own read
    # completes, overlapping cudaHostRegister with the remaining reads.
    try:
        with PinPipeline() as pins:

            def _on_done(bank, layer_id, nbytes):
                def _finish():
                    bar.update(nbytes)
                    if layer_id is None:
                        pins.submit(bank)
                    else:
                        pins.submit(bank, residency[layer_id])
                return _finish

            tasks = []  # (layer_id, name, dest, entry, on_done)
            for e in alpha_entries:
                bank = alpha_hb[e["name"]]
                tasks.append((-1, e["name"], bank.memoryview(), e,
                              _on_done(bank, None, e["nbytes"])))
            for name, bank, win_off, win_len, layer_bytes, layer_id in row_jobs:
                entry = {"global_off": win_off, "nbytes": win_len}
                tasks.append((layer_id, name, bank.memoryview(), entry,
                              _on_done(bank, layer_id, layer_bytes)))
            for name, bank, entry, layer_id in layer_jobs:
                tasks.append((layer_id, name, bank.memoryview(), entry,
                              _on_done(bank, layer_id, entry["nbytes"])))
            tasks.sort(key=lambda t: (t[0], t[1]))
            _run_shared_reads(
                reader, [(dest, entry, on_done) for _, _, dest, entry, on_done in tasks],
                workers=_bank_pool_workers(workers), chunk=min(chunk, _BANK_SUB_CHUNK),
            )
    finally:
        bar.close()
        reader.close()

    sources: dict[str, list] = {}
    for name, banks in row_hb.items():
        views = []
        for bank, view_args in zip(banks, row_view_args[name]):
            if view_args is None:  # per-layer entry: already shaped [num_experts, ...]
                views.append(bank.tensor)
                continue
            head_pad, layer_bytes, num_experts, row_shape, dtype = view_args
            raw = bank.tensor[head_pad:head_pad + layer_bytes].view(dtype)
            views.append(raw.view(num_experts, *row_shape) if row_shape else raw.view(num_experts))
        sources[name] = views

    from freetoken.moe.legacy_format import canonical_role, kind_kernel_for
    from freetoken.moe.expert_banks import ExpertBanks

    # the file names the banks the legacy way; the quant_format tag names the
    # (kind, kernel) they were packed for - except "gguf": those banks carry no
    # QuantKind/kernel (raw ggml blocks) and load with kind=None
    sources = {canonical_role(name): views for name, views in sources.items()}
    kind, kernel = kind_kernel_for(quant_format) if quant_format is not None else (None, None)

    # a failed mlock leaves a LOCKED layer pageable; the log and labels report what the banks actually settled at
    applied = list(residency)
    for banks in row_hb.values():
        for layer_id, bank in enumerate(banks):
            if (applied[layer_id] == HostResidency.LOCKED.value
                    and bank.residency is not HostResidency.LOCKED):
                applied[layer_id] = HostResidency.PAGEABLE.value
    unpinned = [i for i, r in enumerate(applied) if r != HostResidency.PINNED.value]
    if unpinned:
        by_layer = [0] * num_layers
        for name, banks in row_hb.items():
            for layer_id, bank in enumerate(banks):
                by_layer[layer_id] += bank.nbytes
        locked = [i for i in unpinned if applied[i] == HostResidency.LOCKED.value]
        pageable = [i for i in unpinned if i not in set(locked)]
        pinned_b = sum(b for i, b in enumerate(by_layer) if i not in set(unpinned))
        locked_b = sum(by_layer[i] for i in locked)
        pageable_part = ""
        if pageable:
            pageable_b = sum(by_layer[i] for i in pageable)
            pageable_part = (
                f" + {pageable_b / 2**30:.2f} GiB pageable "
                f"(lock failed, {len(pageable)} CPU layers: {pageable})"
            )
        logger.info(
            f"MoE bank split residency: {pinned_b / 2**30:.2f} GiB pinned "
            f"({'born-pinned cudaHostAlloc' if born else 'cudaHostRegister'}, "
            f"{num_layers - len(unpinned)} GPU layers) + "
            f"{locked_b / 2**30:.2f} GiB OS-locked ({len(locked)} CPU layers: {locked})"
            f"{pageable_part}"
        )

    # alphas are the small per-expert scale vectors, distinguished by their reserved names
    # (not a separate kind); everything else under experts_bank is a weight source.
    alpha_kw = {n: alpha_hb[n].tensor for n in alpha_hb}
    return ExpertBanks(
        quant_format, sources, **alpha_kw,
        layer_residency=applied, kind=kind, kernel=kernel, gguf_types=meta_gguf_types,
    )


def load_ftw_banks_to_device(path: str, *, num_layers: int, device: torch.device, workers: int = 8, chunk: int = _DEFAULT_CHUNK):
    """:func:`load_ftw_banks` for resident experts: every bank lands on ``device``, one (bank, layer) at a time through a transient host buffer."""
    from freetoken.moe.expert_banks import ExpertBanks
    from freetoken.moe.legacy_format import canonical_role, kind_kernel_for
    from freetoken.utils.progress import byte_bar

    reader = FTWReader(path)
    try:
        split = _split_bank_entries(reader, path, num_layers)
        if split is None:
            return None
        bank_entries, alpha_entries, flat_entries, per_layer_groups = split

        # (name, layer_id or None for an alpha, read offset, read length, head pad, shape, dtype)
        jobs = [(e["name"], None, e["global_off"], e["nbytes"], 0, tuple(e["shape"]), _dtype_of(e["dtype"])) for e in alpha_entries]
        for e in flat_entries:
            total, *row_shape = e["shape"]
            assert total % num_layers == 0, (e["name"], total, num_layers)
            layer_bytes = total // num_layers * math.prod(row_shape) * _elsize(_dtype_of(e["dtype"]))
            assert layer_bytes * num_layers == e["nbytes"], (e["name"], layer_bytes, num_layers, e["nbytes"])
            for layer_id in range(num_layers):
                off = e["global_off"] + layer_id * layer_bytes
                win_off = (off // ALIGN) * ALIGN
                jobs.append((e["name"], layer_id, win_off, _align_up(off + layer_bytes) - win_off, off - win_off, (total // num_layers, *row_shape), _dtype_of(e["dtype"])))
        for base, by_layer in per_layer_groups.items():
            for layer_id in range(num_layers):
                e = by_layer[layer_id]
                jobs.append((base, layer_id, e["global_off"], e["nbytes"], 0, tuple(e["shape"]), _dtype_of(e["dtype"])))

        def _read(job) -> torch.Tensor:
            _name, _layer, off, nbytes, head_pad, shape, dtype = job
            buf = _transient_buffer(nbytes)
            reader.read_into(memoryview(buf), {"global_off": off, "nbytes": nbytes}, workers=workers, chunk=chunk)
            count = math.prod(shape) * _elsize(dtype)
            return torch.frombuffer(buf, dtype=torch.uint8, count=head_pad + count)[head_pad:].view(dtype).view(shape)

        sources: dict[str, list] = {}
        alphas: dict[str, torch.Tensor] = {}
        bar = byte_bar(sum(e["nbytes"] for e in bank_entries), "Loading expert banks (FTW)")
        try:
            # read the next (bank, layer) while the current one copies to the device
            with ThreadPoolExecutor(1) as ex:
                ahead = ex.submit(_read, jobs[0])
                for k, (name, layer_id, *_rest) in enumerate(jobs):
                    host = ahead.result()
                    if k + 1 < len(jobs):
                        ahead = ex.submit(_read, jobs[k + 1])
                    if layer_id is None:
                        alphas[name] = host.to(device)
                    else:
                        sources.setdefault(canonical_role(name), [None] * num_layers)[layer_id] = host.to(device)
                    bar.update(host.numel() * host.element_size())
                    del host
        finally:
            bar.close()
        quant_format = reader.meta("quant_format")
    finally:
        reader.close()

    kind, kernel = kind_kernel_for(quant_format) if quant_format is not None else (None, None)
    return ExpertBanks(quant_format, sources, **alphas, kind=kind, kernel=kernel)


__all__ = [
    "INDEX_NAME", "FORMAT_TAG", "FORMAT_VERSION", "ALIGN", "DEFAULT_SHARD_LIMIT",
    "is_ftw_checkpoint", "ftw_tensor_names", "FTWWriter", "FTWReader",
    "iter_ftw_weights", "load_ftw_banks", "load_ftw_banks_to_device", "layer_bank_entry_name",
]
