"""Shared GGUF access helpers: detection, metadata, and tensor enumeration.

Thin layer over ``gguf.GGUFReader`` (gguf-py). Metadata is read into a plain dict
keyed by the GGUF field name (``general.architecture``, ``gemma4.block_count`` ...);
tensors are exposed as ``GgufTensor`` records carrying the *torch* shape (ggml dims
reversed), the ggml quant type, and a zero-copy ``uint8`` view of the packed block
bytes laid out as ``[rows, row_bytes]`` (rows = product of all but the fastest ggml
dim; row_bytes spans whole quant blocks of the fastest dim).
"""

from __future__ import annotations

import functools
import os
import struct
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import torch


def is_gguf_path(model_path: str) -> bool:
    """A single ``.gguf`` file (the only GGUF layout FreeToken loads directly)."""
    return isinstance(model_path, str) and os.path.isfile(model_path) and model_path.endswith(
        ".gguf"
    )


# Canonical name of the metadata-only GGUF that ``convert_checkpoint`` drops into an FTW
# dir built from a bare ``.gguf`` source. A GGUF carries its config AND tokenizer in the
# file's KV section, not sibling files, so a converted checkpoint has nowhere else to read
# them from -- this file is the header + KV bytes verbatim (tensor_count patched to 0, no
# tensor infos, no weight data), letting the FTW dir resolve config/tokenizer the exact
# same way the original ``.gguf`` file does.
FTW_METADATA_GGUF = "source_metadata.gguf"
# Records whether the source carried an untied "output.weight" head (the tensor table
# is stripped from metadata-only gguf files, so the fact travels as a KV).
OUTPUT_WEIGHT_PRESENT_KV = "freetoken.output_weight_present"


def gguf_config_source(model_path: str) -> str | None:
    """The ``.gguf`` file to source config/tokenizer/metadata from, or ``None``.

    A bare ``.gguf`` file resolves to itself; an FTW dir carrying a
    :data:`FTW_METADATA_GGUF` resolves to that embedded metadata file. This is the single
    seam config/tokenizer dispatch uses to decide "this checkpoint is GGUF-config-sourced"
    -- a real file and a converted-FTW dir both land on a genuine ``.gguf`` path the reader
    can parse, so no downstream code learns about the FTW wrapper.
    """
    if is_gguf_path(model_path):
        return model_path
    if isinstance(model_path, str) and os.path.isdir(model_path):
        cand = os.path.join(model_path, FTW_METADATA_GGUF)
        if os.path.isfile(cand):
            return cand
    return None


def write_metadata_gguf(source_gguf: str, dest_path: str, extra_kvs: dict | None = None) -> None:
    """Write a metadata-only GGUF: the source's header + KV section byte-for-byte, with
    ``tensor_count`` patched to 0 (no tensor infos, no weight data). Reading only the
    header+KV is cheap; the multi-GB tensor data is never touched.

    ``extra_kvs`` appends further KVs into the copied KV section (used by the converter
    to carry the mmproj clip.* tower facts into an FTW dir). Values pack as bool /
    int32 / float32 / string / flat arrays of those - exactly what the config hooks
    read back; anything else is a caller bug and fails the re-parse check below.

    Validates by re-parsing: the copy must list zero tensors and expose the identical KV
    key set (the KV *bytes* are copied verbatim, so identical keys imply identical values).
    """
    import gguf

    reader = gguf.GGUFReader(source_gguf)
    assert reader.tensors, f"{source_gguf}: no tensors to bound the KV section"
    src_keys = {k for k in reader.fields if not k.startswith("GGUF.")}
    extra = extra_kvs or {}
    present = any(t.name == "output.weight" for t in reader.tensors)
    # an extra KV clashing with a source KV would write a duplicate record the re-parse
    # silently dedupes; clashing with the reserved fact would silently REPLACE it
    collision = set(extra) & (src_keys | {OUTPUT_WEIGHT_PRESENT_KV})
    if collision:
        raise ValueError(
            f"{source_gguf}: extra KV keys collide with the source's own KV section: "
            f"{sorted(collision)}"
        )
    appends = {OUTPUT_WEIGHT_PRESENT_KV: present, **extra}
    # The first tensor-info record starts exactly where the KV section ends (GGUF places no
    # padding between KV and tensor infos; padding is only before the tensor *data*).
    kv_end = int(reader.tensors[0].field.offset)
    buf = bytearray(reader.data[:kv_end].tobytes())  # header + all KV, verbatim
    buf[8:16] = b"\x00" * 8  # tensor_count is a u64 at byte 8; 0 is byte-order agnostic
    # The tensor table is dropped, but config derivation needs one fact from it (an
    # untied output head shows up only as an "output.weight" tensor). Append it as an
    # extra KV and bump kv_count (u64 at byte 16). Little-endian only -- the re-parse
    # below fails loudly on a big-endian source.
    buf += b"".join(_pack_kv(k, v) for k, v in sorted(appends.items()))
    struct.pack_into("<Q", buf, 16, struct.unpack_from("<Q", buf, 16)[0] + len(appends))
    tmp = f"{dest_path}.tmp"
    with open(tmp, "wb") as f:
        f.write(buf)
    os.replace(tmp, dest_path)

    check = gguf.GGUFReader(dest_path)
    assert not check.tensors, "metadata gguf still lists tensors after patch"
    dst_keys = {k for k in check.fields if not k.startswith("GGUF.")}
    assert dst_keys == src_keys | set(appends), (
        f"metadata gguf KV keys differ from source: "
        f"missing {sorted(src_keys - dst_keys)}, extra {sorted(dst_keys - src_keys - set(appends))}"
    )


def _pack_kv(key: str, value) -> bytes:
    """One GGUF KV record: u64 key length + key + u32 value type + payload."""
    import gguf

    VT = gguf.GGUFValueType
    key_bytes = key.encode()
    out = struct.pack("<Q", len(key_bytes)) + key_bytes
    if isinstance(value, bool):
        vtype, payload = VT.BOOL, struct.pack("<?", value)
    elif isinstance(value, int):
        vtype, payload = VT.INT32, struct.pack("<i", value)
    elif isinstance(value, float):
        vtype, payload = VT.FLOAT32, struct.pack("<f", value)
    elif isinstance(value, str):
        vtype = VT.STRING
        payload = struct.pack("<Q", len(value)) + value.encode()
    elif isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"{key}: empty-array KV packing is not supported")
        elem = value[0]
        # GGUF arrays are single-typed; [1, 2.0] packed by the first element's type
        # would silently truncate the rest, so name the mix instead
        kinds = sorted({type(v).__name__ for v in value})
        if len(kinds) > 1:
            raise ValueError(f"{key}: heterogeneous array KV ({', '.join(kinds)})")
        if isinstance(elem, bool):
            etype, pack_one = VT.BOOL, lambda v: struct.pack("<?", v)
        elif isinstance(elem, int):
            etype, pack_one = VT.INT32, lambda v: struct.pack("<i", v)
        elif isinstance(elem, float):
            etype, pack_one = VT.FLOAT32, lambda v: struct.pack("<f", v)
        else:
            raise ValueError(f"{key}: unsupported array element type {type(elem).__name__}")
        vtype = VT.ARRAY
        payload = struct.pack("<I", etype) + struct.pack("<Q", len(value)) + b"".join(pack_one(v) for v in value)
    else:
        raise ValueError(f"{key}: unsupported KV value type {type(value).__name__}")
    return out + struct.pack("<I", int(vtype)) + payload


@dataclass(frozen=True)
class GgufTensor:
    name: str
    shape: tuple[int, ...]  # torch order (ggml dims reversed)
    ggml_type: int
    rows: int  # product of shape[:-1] over the *ggml* layout = blocks-major rows
    row_bytes: int  # packed bytes per row (whole quant blocks of the fastest dim)
    _raw: np.ndarray  # uint8 view, shape [rows, row_bytes]

    def packed(self) -> torch.Tensor:
        """Zero-copy ``[rows, row_bytes]`` uint8 tensor of the native block bytes."""
        return torch.from_numpy(self._raw)


def _field_value(reader, name: str) -> Any:
    field = reader.fields.get(name)
    if field is None:
        return None
    return field.contents()


@functools.cache
def _reader(model_path: str):
    import gguf

    return gguf.GGUFReader(model_path)


@functools.cache
def load_gguf_metadata(model_path: str) -> dict[str, Any]:
    """All GGUF KV metadata as ``{field_name: python_value}`` (arrays -> lists)."""
    reader = _reader(model_path)
    return {name: field.contents() for name, field in reader.fields.items()}


def gguf_architecture(model_path: str) -> str:
    arch = _field_value(_reader(model_path), "general.architecture")
    if arch is None:
        raise ValueError(f"GGUF file {model_path} has no general.architecture")
    return str(arch)


def iter_gguf_tensors(model_path: str) -> Iterator[GgufTensor]:
    """Yield every tensor with its torch shape, ggml type, and packed block bytes."""
    import gguf

    reader = _reader(model_path)
    for t in reader.tensors:
        ne = [int(s) for s in t.shape]  # ggml order, fastest dim first
        torch_shape = tuple(reversed(ne))
        block, type_size = gguf.GGML_QUANT_SIZES[t.tensor_type]
        n_fast = ne[0]
        if n_fast % block != 0:
            raise ValueError(
                f"{t.name}: fastest dim {n_fast} not a multiple of block {block} "
                f"for {t.tensor_type.name}"
            )
        row_bytes = n_fast // block * type_size
        rows = int(np.prod(ne[1:])) if len(ne) > 1 else 1
        # gguf-py returns quantized tensors as raw uint8 but F32/F16 as typed arrays;
        # normalize everything to a flat byte view before shaping into [rows, row_bytes].
        flat = np.ascontiguousarray(t.data).reshape(-1).view(np.uint8)
        raw = flat.reshape(rows, row_bytes)
        yield GgufTensor(
            name=t.name,
            shape=torch_shape,
            ggml_type=int(t.tensor_type),
            rows=rows,
            row_bytes=row_bytes,
            _raw=raw,
        )


def gguf_tensor_names(model_path: str) -> set[str]:
    return {t.name for t in _reader(model_path).tensors}


__all__ = [
    "is_gguf_path",
    "FTW_METADATA_GGUF",
    "OUTPUT_WEIGHT_PRESENT_KV",
    "gguf_config_source",
    "write_metadata_gguf",
    "GgufTensor",
    "load_gguf_metadata",
    "gguf_architecture",
    "iter_gguf_tensors",
    "gguf_tensor_names",
]
