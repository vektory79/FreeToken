"""GGUF -> FTW serve fast path (Task S gaps 1+2+3): kind_kernel_for("gguf"),
gguf_types persistence through the converter index, the load_ftw_banks roundtrip,
and the FTW analog of the pre-load --moe-cache-auto floor gate.

The fixture mirrors the real GLM-5.3-Flash GGUF heterogeneity at toy geometry:
3 width signatures (IQ3_XXS, IQ3_XXS, IQ4_XS) x5 / (IQ4_XS, IQ4_XS, Q6_K) x1 /
(IQ3_XXS, IQ3_XXS, Q6_K) x2 over 8 MoE bank layers, with H=512 != I=256
(non-square: gate/up pack 256 output rows of 512 input cols, down packs 512 rows
of 256 - square fixtures mask real-geometry bugs, see the fixture-crafting notes).
The conversion runs the real convert_checkpoint sink path; offload_expert_method
is stubbed to the value the real gguf path always produces (None - gguf MoE
layers carry no quant method), keeping the test off toy-dim meta-model builds.
convert_checkpoint hard-inits a CUDA context, so the converted-fixture tests
skip on CUDA-less hosts (pure-meta tests run anywhere).
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "models"))

# reuse the proven writer/metadata and the triple-signature gate fixtures
from test_gguf_expert_banks import _TRIPLE_SIGS  # noqa: E402
from test_glm5_next_gguf import _ITER_METADATA, _write_iter_gguf  # noqa: E402

_FT_BLOCK_COUNT = 9  # 8 trunk layers (all MoE) + blk.8 as the skipped MTP block
_FT_H = 512  # embedding_length: gate/up pack 512 input cols
_FT_I = 256  # expert_feed_forward_length: down packs 256 input cols
_FT_E = 4
_FT_METADATA = {
    **_ITER_METADATA,
    "glm5next.block_count": _FT_BLOCK_COUNT,
    "glm5next.embedding_length": _FT_H,
    # DSA ids (3, 7) fall out of the parser's i % 4 == 3 rule; i == 8 is the MTP block
    "glm5next.attention.head_count_kv": [1 if (i % 4 == 3 or i == 8) else 0 for i in range(9)],
    "glm5next.leading_dense_block_count": 0,
    "glm5next.swiglu_clamp_exp": [10.0] * _FT_BLOCK_COUNT,
    "glm5next.swiglu_clamp_shexp": [10.0] * _FT_BLOCK_COUNT,
}
_ROW_BYTES = {18: 98, 23: 136, 14: 210}  # packed bytes per 256-element block


def _write_ftw_source_gguf(path) -> str:
    import gguf

    qt = {
        8: gguf.GGMLQuantizationType.Q8_0,
        18: gguf.GGMLQuantizationType.IQ3_XXS,
        23: gguf.GGMLQuantizationType.IQ4_XS,
        14: gguf.GGMLQuantizationType.Q6_K,
    }
    entries = {
        # vocab x hidden Q8_0 embedding: the config parse's only tensor fact
        "token_embd.weight": (np.zeros((256, _FT_H // 32 * 34), np.uint8), qt[8]),
    }
    for layer, (gt, ut, dt) in enumerate(_TRIPLE_SIGS):
        # layer 0 inserts down first: role assembly must follow the explicit
        # gate/up/down order, never the file's tensor encounter order
        order = (
            (("down", dt), ("up", ut), ("gate", gt))
            if layer == 0
            else (("gate", gt), ("up", ut), ("down", dt))
        )
        for role, type_id in order:
            ne0, rows = (_FT_I, _FT_H) if role == "down" else (_FT_H, _FT_I)
            # 3D like the real file's stacked expert tensors: (E, rows, packed row bytes)
            raw = np.zeros((_FT_E, rows, ne0 // 256 * _ROW_BYTES[type_id]), np.uint8)
            entries[f"blk.{layer}.ffn_{role}_exps.weight"] = (raw, qt[type_id])
    return _write_iter_gguf(path, metadata=_FT_METADATA, tensors=entries)


@pytest.fixture(scope="module")
def ftw_gguf_ckpt(tmp_path_factory):
    if not torch.cuda.is_available():
        pytest.skip("convert_checkpoint hard-inits a CUDA context")
    import freetoken.engine.engine as engine_mod
    from freetoken.checkpoint.convert import convert_checkpoint

    src = _write_ftw_source_gguf(tmp_path_factory.mktemp("ftw-gguf-src") / "banks.gguf")
    out = tmp_path_factory.mktemp("ftw-gguf-out") / "ckpt"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(engine_mod, "offload_expert_method", lambda config: None)
        convert_checkpoint(src, str(out))
    return src, str(out)


def test_ftw_gguf_roundtrip(ftw_gguf_ckpt):
    from freetoken.checkpoint.ftw import ALIGN, load_ftw_banks
    from freetoken.models.gguf.reader import iter_gguf_tensors

    src, out = ftw_gguf_ckpt
    banks = load_ftw_banks(out, num_layers=8)
    assert banks is not None
    assert banks.quant_format == "gguf"
    assert banks.kind is None and banks.kernel is None
    assert banks.gate_up_alpha is None and banks.down_alpha is None
    assert sorted(banks.sources) == ["down", "gate", "up"]
    assert banks.gguf_types == tuple(tuple(s) for s in _TRIPLE_SIGS)
    assert banks.layer_residency == ["pinned"] * 8
    for layer, sig in enumerate(_TRIPLE_SIGS):
        for role, tid in zip(("gate", "up", "down"), sig):
            t = banks.sources[role][layer]
            ne0, rows = (_FT_I, _FT_H) if role == "down" else (_FT_H, _FT_I)
            assert t.dtype is torch.uint8
            assert tuple(t.shape) == (_FT_E, rows, ne0 // 256 * _ROW_BYTES[tid])

    # bank bytes are the source file's packed rows verbatim, per explicit role
    # (the down-first insertion order of layer 0 makes an encounter-order bug swap bytes)
    by_name = {t.name: t for t in iter_gguf_tensors(src)}
    for layer, sig in enumerate(_TRIPLE_SIGS):
        for role, tid in zip(("gate", "up", "down"), sig):
            ne0, rows = (_FT_I, _FT_H) if role == "down" else (_FT_H, _FT_I)
            raw = by_name[f"blk.{layer}.ffn_{role}_exps.weight"].packed()
            assert torch.equal(banks.sources[role][layer], raw.reshape(_FT_E, rows, -1))

    # the converter index carries the role-ordered types and honors ALIGN-4096 entries
    with open(os.path.join(out, "freetoken_weight.json"), encoding="utf-8") as f:
        index = json.load(f)
    assert index["quant_format"] == "gguf"
    assert index["gguf_types"] == [list(s) for s in _TRIPLE_SIGS]
    assert index["expert_bank_num_layers"] == 8
    bank_entries = [t for t in index["tensors"] if t["kind"] == "experts_bank"]
    assert len(bank_entries) == 24  # 3 role banks x 8 layers
    assert all(t["global_off"] % ALIGN == 0 for t in bank_entries)


def test_ftw_without_gguf_meta_keeps_legacy_semantics(tmp_path):
    # absent gguf_types meta -> None (nvfp4/bf16 checkpoints load exactly as before),
    # and the quant_format tag still resolves kind/kernel through LEGACY_FORMAT
    from freetoken.checkpoint.ftw import FTWWriter, load_ftw_banks
    from freetoken.layers.quantization import QuantKind

    out = tmp_path / "ckpt"
    w = FTWWriter(str(out))
    for name, shape in (("gate_up", (4, 8, 3)), ("down", (4, 5, 2))):
        for layer in range(2):
            w.add_tensor(
                f"{name}#L{layer:05d}", torch.zeros(shape, dtype=torch.uint8), kind="experts_bank"
            )
    w.finalize({"quant_format": "bf16", "expert_bank_num_layers": 2})
    banks = load_ftw_banks(str(out), num_layers=2)
    assert banks.quant_format == "bf16"
    assert banks.kind is QuantKind.NONE and banks.kernel == "fused"
    assert banks.gguf_types is None
    assert sorted(banks.sources) == ["down", "gate_up"]


@pytest.mark.parametrize(
    ("meta", "why"),
    [
        ({"gguf_types": [[18, 18, 23]]}, "wrong layer count"),  # 1 != 2 MoE layers
        ({"gguf_types": {"0": [18, 18, 23], "1": [18, 18, 23]}}, "dict-encoded meta"),
        ({"gguf_types": [[18, 18], [18, 18, 23]]}, "short row"),
        ({"gguf_types": [[18, 18, 23], [18, 18, "x"]]}, "non-int type id"),
        ({}, "old-converter dir without the meta"),
    ],
)
def test_ftw_gguf_types_meta_rejected(tmp_path, meta, why):
    # malformed gguf_types meta must fail at LOAD time (with the reader closed), not
    # later with a cryptic unpack/IndexError; a gguf dir without the meta was made by
    # an older converter build and must say so instead of dying at serve init
    from freetoken.checkpoint.ftw import FTWWriter, load_ftw_banks

    out = tmp_path / "ckpt"
    w = FTWWriter(str(out))
    for name, shape in (("gate", (4, 8, 3)), ("up", (4, 8, 3)), ("down", (4, 5, 2))):
        for layer in range(2):
            w.add_tensor(
                f"{name}#L{layer:05d}", torch.zeros(shape, dtype=torch.uint8), kind="experts_bank"
            )
    w.finalize({"quant_format": "gguf", **meta})
    with pytest.raises(RuntimeError, match="gguf_types"):
        load_ftw_banks(str(out), num_layers=2)

