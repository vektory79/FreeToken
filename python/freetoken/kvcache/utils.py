"""Shared kvcache helpers (layering: kvcache-level, importable by scheduler)."""
from __future__ import annotations

import hashlib
import struct

__all__ = ["chain_page_key"]


def chain_page_key(prev_key: bytes | None, tokens: tuple[int, ...]) -> bytes:
    """Incremental hash over one page's token-id tuple and its ancestor key.

    Deliberate approximation: KV bytes are a deterministic function of (token ids,
    position), so hashing token-id tuples stands in for hashing KV content - this keeps
    the D2H copy of KV bytes off the hot path; chain hashing makes equal keys at depth
    d imply equal prefixes [0:d].
    """
    h = hashlib.blake2b(digest_size=16)
    if prev_key is not None:
        h.update(prev_key)
    h.update(struct.pack("<I", len(tokens)))
    for t in tokens:
        h.update(struct.pack("<q", t))
    return h.digest()
