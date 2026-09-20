"""Regression: the "gguf" quant_format tag maps to (None, None) in kind_kernel_for.

"gguf" banks are raw packed ggml blocks with no QuantKind/kernel behind the tag;
the tag must not join LEGACY_FORMAT (the inverse (kind, kernel)-to-tag map) - it
lives only in _KIND_KERNEL. Before the fix kind_kernel_for("gguf") raised
KeyError("gguf") in load_ftw_banks, the hard blocker of the gguf -> FTW serve
fast path (task S gap 1).
"""


def test_kind_kernel_for_gguf_maps_to_none_none():
    from freetoken.moe.legacy_format import kind_kernel_for

    assert kind_kernel_for("gguf") == (None, None)
