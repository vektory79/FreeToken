"""GGUF -> FTW serve fast path (Task S gaps 1+2+3): kind_kernel_for("gguf"),
gguf_types persistence through the converter index, the load_ftw_banks roundtrip,
and the FTW analog of the pre-load --moe-cache-auto floor gate.

The fixture mirrors the real GLM-5.3-Flash GGUF heterogeneity at toy geometry:
3 width signatures (IQ3_XXS, IQ3_XXS, IQ4_XS) x5 / (IQ4_XS, IQ4_XS, Q6_K) x1 /
(IQ3_XXS, IQ3_XXS, Q6_K) x2 over 8 MoE bank layers, with H=512 != I=256
(non-square: gate/up pack 256 output rows of 512 input cols, down packs 512 rows
of 256 - square fixtures mask real-geometry bugs, see the fixture-crafting notes).
The conversion runs the real convert_checkpoint sink path on CPU (device="cpu")
- no CUDA context needed for the toy geometry, so the converted-fixture tests
run on CPU-only hosts too; the pin settle of load_ftw_banks is skipped there
(the byte-level checks do not need it). Pure-meta tests run anywhere.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "models"))

# reuse the proven writer/metadata and the triple-signature gate fixtures
# (inline copies: test_gguf_expert_banks rides the amputated old expert-sources API
# and is module-skipped for the expert-banks deferred-port wave)
_TRIPLE_SIGS = [(18, 18, 23)] * 5 + [(23, 23, 14)] + [(18, 18, 14)] * 2
_TRIPLE_GROUP_SLOT_BYTES = [169_984, 246_784, 207_872]


def _gate_engine_and_config(path, kv_reserve_tokens):
    # Engine.__new__ stub (same idiom as test_engine_resolve_auto_moe_cache_size_maps_kwargs)
    # + an auto-plan config over the triple-signature gguf: budget 89,000,000 B,
    # cache_per_page 98,304, per-slot (layer-0 width) 169,984 -> kv_reserve_tokens
    # picks the envelope: 14,080 -> 14 slots, 14,048 -> 15, 14,016 -> 16.
    import torch

    from freetoken.engine.engine import Engine
    from freetoken.kvcache.mha_pool import MHAKVCache
    from freetoken.models.config import KVCacheGroupSpec

    class StubModelConfig:
        num_experts = 4
        num_moe_layers = 8
        first_k_dense_replace = 0
        expert_quant = "none"
        moe_weight_format = "gguf"
        hidden_size = 512
        moe_intermediate_size = 256
        decode_target = "gpu"

        def kv_cache_group_specs(self):
            return [KVCacheGroupSpec(
                name="full", layer_ids=(0, 1, 2), num_kv_heads=8, head_dim=64, sliding_window=None,
            )]

        def linear_attention_group(self):
            return None

    class StubConfig:
        dtype = torch.float16
        page_size = 16
        max_running_req = 4
        hybrid_swa_cache_mode = "auto"
        memory_ratio = 0.9
        moe_cache_auto = True
        use_dummy_weight = False
        model_path = path
        moe_prefill_overlap = True
        moe_strategy = "offload"
        moe_cpu_layers = None
        expert_load = "auto"
        swa_full_tokens_ratio = 0.2
        swa_num_pages_override = None
        model_config = StubModelConfig()

        class tp_info:
            size = 1

    StubConfig.kv_reserve_tokens = kv_reserve_tokens  # class bodies cannot close over locals

    engine = Engine.__new__(Engine)  # bypass __init__/GPU
    engine._baseline_free = 100_000_000
    engine._weights_bytes = 1_000_000
    engine._pool_cls = MHAKVCache
    return engine, StubConfig()


from test_glm5_next_gguf import (  # noqa: E402
    _CLIP_METADATA,
    _CONV,
    _HCN,
    _HD,
    _IDXH,
    _IDXD,
    _ITER_METADATA,
    _KL,
    _KPOOL,
    _NH,
    _PROJ,
    _QL,
    _QK,
    _TOKENIZER_METADATA,
    _VD,
    _full_vision_fixture_data,
    _write_iter_gguf,
    _write_vision_mmproj,
)

_FT_BLOCK_COUNT = 9  # 8 trunk layers (all MoE) + blk.8 as the skipped MTP block
_FT_H = 512  # embedding_length: gate/up pack 512 input cols
_FT_I = 256  # expert_feed_forward_length: down packs 256 input cols
_FT_E = 4
_FT_METADATA = {
    **_ITER_METADATA,
    **_TOKENIZER_METADATA,  # image-serving facts ride the main gguf metadata (phase 3)
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
    # pypi gguf-py (fixture writer only, freetoken vendors its own reader):
    # the baseline tree kept this whole module uncollected behind the D4 module
    # skip, so these tests skip rather than error when the package is absent
    gguf = pytest.importorskip("gguf")

    qt = {
        8: gguf.GGMLQuantizationType.Q8_0,
        18: gguf.GGMLQuantizationType.IQ3_XXS,
        23: gguf.GGMLQuantizationType.IQ4_XS,
        14: gguf.GGMLQuantizationType.Q6_K,
    }

    def q8(rows, ne0):
        # packed Q8_0 payload stub: (rows, ne0 // 32 * 34) zero bytes
        return (np.zeros((rows, ne0 // 32 * 34), np.uint8), qt[8])

    # dense tensors for every mapped module name: the new stack builds quant
    # schemes by walking to_gguf(name) per module, so a convertible file must
    # carry the full trunk (shapes mirror test_glm5_next_gguf._iter_tensor_set)
    hc_rows, hc_cols = (2 + _HCN) * _HCN, _HCN * _FT_H
    kda_layers = (0, 1, 2, 4, 5, 6)  # head_count_kv == 0 under the i % 4 == 3 rule
    dsa_layers = (3, 7)
    entries = {
        # vocab x hidden Q8_0 embedding: the config parse's only tensor fact
        "token_embd.weight": (np.zeros((256, _FT_H // 32 * 34), np.uint8), qt[8]),
        # untied Q6_K head + final norm: iter_gguf_weights consumes both
        "output.weight": (np.zeros((256, _FT_H // 256 * 210), np.uint8), qt[14]),
        "output_norm.weight": (np.full(_FT_H, 0.5, np.float32), None),
    }
    for layer in range(8):
        entries[f"blk.{layer}.attn_norm.weight"] = (np.full(_FT_H, 0.25, np.float32), None)
        entries[f"blk.{layer}.ffn_norm.weight"] = (np.full(_FT_H, 0.75, np.float32), None)
        entries[f"blk.{layer}.hc_attn_fn.weight"] = q8(hc_rows, hc_cols)
        entries[f"blk.{layer}.hc_attn_base.weight"] = (np.arange(hc_rows, dtype=np.float32) / 8, None)
        entries[f"blk.{layer}.hc_attn_scale.weight"] = (np.array([0.1, 0.2, 0.3], np.float32), None)
        entries[f"blk.{layer}.hc_ffn_fn.weight"] = q8(hc_rows, hc_cols)
        entries[f"blk.{layer}.hc_ffn_base.weight"] = (np.arange(hc_rows, dtype=np.float32) / 8, None)
        entries[f"blk.{layer}.hc_ffn_scale.weight"] = (np.array([0.4, 0.5, 0.6], np.float32), None)
        entries[f"blk.{layer}.ffn_gate_inp.weight"] = (np.zeros((_FT_E, _FT_H), np.float32), None)
        entries[f"blk.{layer}.exp_probs_b.bias"] = (np.zeros(_FT_E, np.float32), None)
        entries[f"blk.{layer}.ffn_gate_shexp.weight"] = q8(_FT_I, _FT_H)
        entries[f"blk.{layer}.ffn_up_shexp.weight"] = q8(_FT_I, _FT_H)
        entries[f"blk.{layer}.ffn_down_shexp.weight"] = q8(_FT_H, _FT_I)
    for layer in kda_layers:
        for rows, suffix in (
            (_PROJ, "attn_q"), (_PROJ, "attn_k"), (_PROJ, "attn_v"),
            (_NH, "ssm_beta"), (_HD, "ssm_f_a"), (_HD, "ssm_g_a"),
        ):
            entries[f"blk.{layer}.{suffix}.weight"] = q8(rows, _FT_H)
        for part in ("q", "k", "v"):
            entries[f"blk.{layer}.ssm_conv1d_{part}.weight"] = (np.full((_PROJ, 1, _CONV), 1.0, np.float32), None)
        entries[f"blk.{layer}.ssm_f_b.weight"] = q8(_PROJ, _HD)
        entries[f"blk.{layer}.ssm_g_b.weight"] = q8(_PROJ, _HD)
        entries[f"blk.{layer}.attn_output.weight"] = q8(_FT_H, _PROJ)
        entries[f"blk.{layer}.ssm_a"] = (np.full(_NH, -1.0, np.float32), None)  # A_log = log(-x)
        entries[f"blk.{layer}.ssm_dt.bias"] = (np.zeros(_PROJ, np.float32), None)
        entries[f"blk.{layer}.ssm_norm.weight"] = (np.full(_HD, 0.5, np.float32), None)
    for layer in dsa_layers:
        entries[f"blk.{layer}.attn_q_a.weight"] = q8(_QL, _FT_H)
        entries[f"blk.{layer}.attn_q_a_norm.weight"] = (np.full(_QL, 0.5, np.float32), None)
        entries[f"blk.{layer}.attn_q_b.weight"] = q8(_NH * _QK, _QL)
        entries[f"blk.{layer}.attn_kv_a_mqa.weight"] = q8(_KL, _FT_H)
        entries[f"blk.{layer}.attn_kv_a_norm.weight"] = (np.full(_KL, 0.5, np.float32), None)
        entries[f"blk.{layer}.attn_output.weight"] = q8(_FT_H, _NH * _VD)
        entries[f"blk.{layer}.attn_k_b.weight"] = (np.zeros((_NH, _KL, _QK // 32 * 34), np.uint8), qt[8])
        entries[f"blk.{layer}.attn_v_b.weight"] = (np.zeros((_NH, _VD, _KL // 32 * 34), np.uint8), qt[8])
        entries[f"blk.{layer}.indexer.attn_k.weight"] = q8(_IDXD, _FT_H)
        entries[f"blk.{layer}.indexer.attn_q_b.weight"] = q8(_IDXH * _IDXD, _QL)
        entries[f"blk.{layer}.indexer.proj.weight"] = (np.zeros((_IDXH, _FT_H), np.float32), None)
        entries[f"blk.{layer}.indexer.k_norm.weight"] = (np.full(_IDXD, 0.5, np.float32), None)
        entries[f"blk.{layer}.indexer.k_norm.bias"] = (np.zeros(_IDXD, np.float32), None)
        entries[f"blk.{layer}.indexer_compressor_gate.weight"] = q8(_IDXD, _FT_H)
        entries[f"blk.{layer}.indexer_compressor_ape.weight"] = (np.zeros((_KPOOL, _IDXD), np.float32), None)
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
    from freetoken.checkpoint.convert import convert_checkpoint

    src = _write_ftw_source_gguf(tmp_path_factory.mktemp("ftw-gguf-src") / "banks.gguf")
    out = tmp_path_factory.mktemp("ftw-gguf-out") / "ckpt"
    convert_checkpoint(src, str(out), device="cpu")
    return src, str(out)


def test_ftw_gguf_roundtrip(ftw_gguf_ckpt, monkeypatch):
    if not torch.cuda.is_available():
        monkeypatch.setenv("FREETOKEN_SKIP_BANK_PIN", "1")  # CPU-only: no pin settle
    from freetoken.checkpoint.ftw import ALIGN, load_ftw_banks
    from freetoken.models.gguf.reader import gguf_tensor_index
    from freetoken.models.gguf.tensors import packed_rows

    src, out = ftw_gguf_ckpt
    banks = load_ftw_banks(out, num_layers=8)
    assert banks is not None
    assert banks.quant_format == "gguf"
    assert banks.kind is None and banks.kernel is None
    assert banks.gate_up_alpha is None and banks.down_alpha is None
    # the merged runtime schema (same-type gate/up stacks pack into one gate_up bank;
    # the pre-re-platform three-role on-disk layout was retired with it)
    assert sorted(banks.sources) == ["down", "gate_up"]
    assert banks.layer_residency == ["pinned"] * 8
    for layer, (gt, ut, dt) in enumerate(_TRIPLE_SIGS):
        gu = banks.sources["gate_up"][layer]
        dn = banks.sources["down"][layer]
        assert gu.dtype is torch.uint8
        assert tuple(gu.shape) == (_FT_E, 2 * _FT_I, _FT_H // 256 * _ROW_BYTES[gt])
        assert tuple(dn.shape) == (_FT_E, _FT_H, _FT_I // 256 * _ROW_BYTES[dt])

    # bank bytes are the source file's packed rows verbatim, gate rows then up rows
    # inside the merged gate_up bank (the down-first insertion order of layer 0 makes
    # an encounter-order bug swap bytes)
    by_name = gguf_tensor_index(src)
    for layer, (gt, ut, dt) in enumerate(_TRIPLE_SIGS):
        gu = banks.sources["gate_up"][layer]
        gate = packed_rows(by_name[f"blk.{layer}.ffn_gate_exps.weight"]).reshape(_FT_E, _FT_I, -1)
        up = packed_rows(by_name[f"blk.{layer}.ffn_up_exps.weight"]).reshape(_FT_E, _FT_I, -1)
        dn = banks.sources["down"][layer]
        down = packed_rows(by_name[f"blk.{layer}.ffn_down_exps.weight"]).reshape(_FT_E, _FT_H, -1)
        assert torch.equal(gu[:, :_FT_I], gate)
        assert torch.equal(gu[:, _FT_I:], up)
        assert torch.equal(dn, down)

    # the converter index carries the role-ordered types and honors ALIGN-4096 entries
    with open(os.path.join(out, "freetoken_weight.json"), encoding="utf-8") as f:
        index = json.load(f)
    assert index["quant_format"] == "gguf"
    assert index["gguf_types"] == [list(s) for s in _TRIPLE_SIGS]
    assert index["expert_bank_num_layers"] == 8
    bank_entries = [t for t in index["tensors"] if t["kind"] == "experts_bank"]
    assert len(bank_entries) == 16  # 2 merged banks x 8 layers
    assert all(t["global_off"] % ALIGN == 0 for t in bank_entries)


def test_ftw_without_gguf_meta_keeps_legacy_semantics(tmp_path, monkeypatch):
    # absent gguf_types meta -> None (nvfp4/bf16 checkpoints load exactly as before),
    # and the quant_format tag still resolves kind/kernel through LEGACY_FORMAT
    monkeypatch.setenv("FREETOKEN_SKIP_BANK_PIN", "1")  # CPU-only: no pin settle
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
    # the gguf meta rides layer_tags now; a legacy bf16 dir carries none
    assert banks.layer_tags is None
    assert sorted(banks.sources) == ["down", "gate_up"]


def test_ftw_scan_matches_loaded_bank_grouping(ftw_gguf_ckpt):
    # exactness bridge: the index-only scan's grouping and per-group slot widths
    # would equal the grouping the loaded banks see; the scan targets
    # gguf_ftw_signature_groups + Engine._group_bank_layers, both still absent
    pytest.skip("FTW-index signature scan not re-landed (feature request, outside D1-D5)")
    from freetoken.checkpoint.ftw import load_ftw_banks
    from freetoken.engine.cache_budget import expert_bytes_per_slot
    from freetoken.engine.engine import Engine
    from freetoken.moe.expert_banks import gguf_ftw_signature_groups

    _, out = ftw_gguf_ckpt
    banks = load_ftw_banks(out, num_layers=8)
    scan_groups, scan_slots = gguf_ftw_signature_groups(
        out, SimpleNamespace(num_experts=_FT_E, num_moe_layers=8)
    )
    ns = SimpleNamespace(sources=banks.sources)
    groups = Engine._group_bank_layers(ns, 8)
    assert scan_groups == groups == [[0, 1, 2, 3, 4], [5], [6, 7]]
    assert scan_slots == [
        expert_bytes_per_slot({n: [banks.sources[n][l] for l in m] for n in banks.sources})
        for m in groups
    ]
    assert scan_slots == _TRIPLE_GROUP_SLOT_BYTES


def test_ftw_early_floor_gate_rejects_before_bank_load(ftw_gguf_ckpt, monkeypatch):
    # pins the FTW branch of the gate's position inside _init_offload_moe_cache:
    # an unfundable plan must raise off the index alone, before load_expert_banks
    # the gate's gguf_signature_groups answers only on a bare .gguf (an FTW dir
    # keeps the documented degrade-not-raise path); the FTW-index scan stays out
    pytest.skip("FTW-index signature scan not re-landed (feature request, outside D1-D5)")
    import freetoken.engine.engine as engine_mod

    _, out = ftw_gguf_ckpt
    engine, config = _gate_engine_and_config(out, kv_reserve_tokens=14_080)  # envelope 14 < need 15
    engine.model = object()
    engine._host_tables_bytes = 0

    def _sentinel(*args, **kwargs):
        raise AssertionError("load_expert_banks ran before the floor gate")

    monkeypatch.setattr(engine_mod, "shared_offload_method", lambda model: None)
    monkeypatch.setattr(engine_mod, "load_expert_banks", _sentinel)
    with pytest.raises(ValueError, match="byte-weighted minimum of 15"):
        engine._init_offload_moe_cache(config)


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


def test_ftw_gguf_types_meta_accepted(tmp_path, monkeypatch):
    # the positive half of the guard (the parametrization above pins the rejections
    # only): a well-formed triple per MoE layer loads and lands on banks.gguf_types
    monkeypatch.setenv("FREETOKEN_SKIP_BANK_PIN", "1")  # CPU-only: no pin settle
    from freetoken.checkpoint.ftw import FTWWriter, load_ftw_banks

    out = tmp_path / "ckpt"
    w = FTWWriter(str(out))
    for name, shape in (("gate_up", (4, 16, 3)), ("down", (4, 5, 2))):
        for layer in range(2):
            w.add_tensor(
                f"{name}#L{layer:05d}", torch.zeros(shape, dtype=torch.uint8), kind="experts_bank"
            )
    w.finalize({"quant_format": "gguf", "gguf_types": [[18, 18, 23], [18, 18, 23]]})
    banks = load_ftw_banks(str(out), num_layers=2)
    assert banks.quant_format == "gguf"
    assert banks.kind is None and banks.kernel is None
    assert sorted(banks.sources) == ["down", "gate_up"]
    assert banks.layer_residency == ["pinned"] * 2


def test_converter_gguf_types_triple_form():
    # the converter-side contract without pypi-gguf/CUDA: layer_tags -> int triples
    # in the OLD key/shape the capability gates read (the gate type repeats for up)
    from freetoken.checkpoint.convert import gguf_type_triples

    tags = [("ggml", "IQ3_XXS", "IQ4_XS"), ("ggml", "IQ4_XS", "Q6_K")]
    assert gguf_type_triples(tags) == [[18, 18, 23], [23, 23, 14]]
    assert gguf_type_triples([]) == []


@pytest.mark.parametrize(
    ("meta", "why"),
    [
        ({}, "old-converter dir without the meta: re-convert contract"),
        ({"gguf_types": [[18, 18, 23]]}, "malformed meta: wrong layer count"),
    ],
)
def test_ftw_gguf_types_meta_rejected_to_device(tmp_path, meta, why):
    # the resident path carries the same contract: a gguf checkpoint reaching
    # load_ftw_banks_to_device must raise (re-convert / config mismatch), not
    # silently load kind=None banks; the guard sits before any bank read, so
    # this runs on CPU and never needs a device
    from freetoken.checkpoint.ftw import FTWWriter, load_ftw_banks_to_device

    out = tmp_path / "ckpt"
    w = FTWWriter(str(out))
    for name, shape in (("gate", (4, 8, 3)), ("up", (4, 8, 3)), ("down", (4, 5, 2))):
        for layer in range(2):
            w.add_tensor(
                f"{name}#L{layer:05d}", torch.zeros(shape, dtype=torch.uint8), kind="experts_bank"
            )
    w.finalize({"quant_format": "gguf", **meta})
    with pytest.raises(RuntimeError, match="gguf_types"):
        load_ftw_banks_to_device(str(out), num_layers=2, device=torch.device("cpu"))


def test_ftw_persisted_types_feed_capability_gates(ftw_gguf_ckpt):
    # the hybrid gate + --moe-cpu-layers auto consult gguf_expert_bank_types, which is
    # bare-.gguf-only: on an FTW dir it must fall back to the persisted index meta
    # (the fix that unblocks --moe-strategy hybrid for converted checkpoints)
    from freetoken.moe.expert_banks import gguf_expert_bank_types

    _, out = ftw_gguf_ckpt
    types = gguf_expert_bank_types(out, SimpleNamespace(num_moe_layers=8))
    assert types == {i: tuple(s) for i, s in enumerate(_TRIPLE_SIGS)}
    # and the FTW floor-gate scan stays silent over the same index-less formats
    assert gguf_expert_bank_types("/nonexistent-dir", SimpleNamespace(num_moe_layers=8)) is None


# ------------------------------------------------------------------------------
# Phase 4: the converter packs the vision tower into the FTW (USER DECISION 4)
# ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ftw_gguf_vision_ckpt(tmp_path_factory):
    """A converted GGUF FTW built WITH --mmproj: tower tensors + merged metadata."""
    from freetoken.checkpoint.convert import convert_checkpoint

    src = _write_ftw_source_gguf(tmp_path_factory.mktemp("ftw-gguf-vision-src") / "banks.gguf")
    mmproj = _write_vision_mmproj(
        tmp_path_factory.mktemp("ftw-gguf-vision-mmproj") / "mmproj.gguf",
        metadata=dict(_CLIP_METADATA),
        tensors=_full_vision_fixture_data(),
    )
    out = tmp_path_factory.mktemp("ftw-gguf-vision-out") / "ckpt"
    # CPU conversion: the tower tensors stream through the same load_weight path
    # as a GPU conversion; no CUDA context needed for the toy geometry
    convert_checkpoint(src, str(out), device="cpu", mmproj_path=mmproj)
    return mmproj, str(out)


def test_ftw_gguf_vision_tensors_roundtrip(ftw_gguf_vision_ckpt):
    """Every tower tensor the mmproj reader yields lands in the FTW under its
    FreeToken visual.* name with the reader's exact shape - the packing is the
    load_weight stream, so a drift shows up as a missing/renamed tensor."""
    import json

    from freetoken.models.glm5_next.gguf import iter_gguf_vision_weights

    mmproj, out = ftw_gguf_vision_ckpt
    expect = dict(iter_gguf_vision_weights(mmproj))
    index = json.load(open(os.path.join(out, "freetoken_weight.json")))
    got = {
        t["name"]: tuple(t["shape"])
        for t in index["tensors"]
        if t["name"].startswith("visual.")
    }
    assert set(got) == set(expect)
    for name, tensor in expect.items():
        assert got[name] == tuple(tensor.shape), name


def test_ftw_gguf_vision_metadata_carrier(ftw_gguf_vision_ckpt):
    """The metadata-only carrier merges the mmproj's clip.* KVs and keeps the main
    GGUF's tokenizer facts (chat template + vocab) byte-verbatim alongside."""
    import math

    from freetoken.models.gguf.reader import FTW_METADATA_GGUF, load_gguf_metadata
    from freetoken.models.glm5_next.gguf import _IMAGE_TOKEN

    mmproj, out = ftw_gguf_vision_ckpt
    carrier = load_gguf_metadata(os.path.join(out, FTW_METADATA_GGUF))
    mm_meta = load_gguf_metadata(mmproj)
    clip_keys = [k for k in carrier if k.startswith("clip.")]
    assert sorted(clip_keys) == sorted(k for k in mm_meta if k.startswith("clip."))
    assert carrier["clip.has_vision_encoder"] is True
    assert math.isclose(carrier["clip.vision.swiglu_limit"], mm_meta["clip.vision.swiglu_limit"], rel_tol=1e-6)
    assert carrier["clip.vision.image_mean"] == pytest.approx(mm_meta["clip.vision.image_mean"], abs=1e-6)
    # the image-serving facts come from the main GGUF KVs, already in the carrier
    assert "tokenizer.chat_template" in carrier
    assert carrier["tokenizer.ggml.tokens"][154854] == _IMAGE_TOKEN


def test_ftw_gguf_vision_boot_without_mmproj(ftw_gguf_vision_ckpt):
    """The converted FTW boots multimodal with NO --mmproj: config path resolves the
    vision spec from the merged carrier, and the mm-processor builds its fallback
    image processor from the baked-in kwargs - the source .gguf is never opened."""
    from freetoken.mm.processor import get_mm_processor
    from freetoken.models.weight import ftw_lacks_vision
    from freetoken.utils import cached_load_hf_config

    _, out = ftw_gguf_vision_ckpt
    shim = cached_load_hf_config(out)
    assert shim.architectures == ["Glm5NextGGUFForConditionalGeneration"]
    assert shim.mmproj_path is None
    assert shim.vision_config.depth == 24
    assert shim.vision_config.patch_size == 14
    assert shim.image_token_id == 154854
    assert shim.image_processor_kwargs["patch_size"] == 14
    assert shim.image_processor_kwargs["min_image_tokens"] == 16
    assert ftw_lacks_vision(out) is False
    proc = get_mm_processor(out)
    assert proc is not None
    assert proc.placeholder == [154854]
    assert proc._image_processor().patch_size == 14


def test_ftw_gguf_text_only_boot_unchanged(ftw_gguf_ckpt):
    """A checkpoint converted WITHOUT --mmproj keeps the pre-phase-4 shape: text-only
    spec, no tower tensors, ftw_lacks_vision still flags it."""
    from freetoken.models.weight import ftw_lacks_vision
    from freetoken.utils import cached_load_hf_config

    _, out = ftw_gguf_ckpt
    shim = cached_load_hf_config(out)
    assert shim.architectures == ["Glm5NextGGUFForCausalLM"]
    assert getattr(shim, "vision_config", None) is None
    assert getattr(shim, "image_token_id", None) is None
    assert ftw_lacks_vision(out) is True


# ------------------------------------------------------------------------------
# gguf->FTW boot: checkpoint_quant_config resolves schemes without the source's
# tensor table (the metadata-only carrier lists zero tensors)
# ------------------------------------------------------------------------------


def _assert_ftw_schemes_resolve(out: str, *, legacy: bool) -> None:
    from freetoken.layers.quantization.scheme import gguf_types as scheme_types
    from freetoken.models.register import checkpoint_quant_config, get_model_spec
    from freetoken.utils import cached_load_hf_config

    hf = cached_load_hf_config(out)
    quant = checkpoint_quant_config(out, hf, get_model_spec(hf.architectures[0]))
    # dense half: the convert contract's pinned types (token_embd Q8_0, output
    # Q6_K, every mapped linear Q8_0); float modules resolve scheme-less
    assert scheme_types(quant.scheme_for("model.embed_tokens")) == (8,)
    assert scheme_types(quant.scheme_for("lm_head")) == (14,)
    assert scheme_types(quant.scheme_for("model.layers.0.self_attn.in_proj")) == (8,) * 6
    assert scheme_types(quant.scheme_for("model.layers.3.self_attn.q_a_proj")) == (8,)
    # the old carrier stores the indexer's wq_b dequantized (see ftw_tensor_types)
    wq_b = quant.scheme_for("model.layers.3.self_attn.indexer.wq_b")
    assert (scheme_types(wq_b) == (8,)) if not legacy else wq_b is None
    assert scheme_types(quant.scheme_for("model.layers.0.mlp.shared_experts.gate_proj")) == (8,)
    assert quant.scheme_for("model.layers.0.mlp.gate") is None  # bf16 router
    # expert half: the persisted (gate, up, down) triples per MoE layer
    assert scheme_types(quant.scheme_for("model.layers.0.mlp.experts")) == (18, 18, 23)
    assert scheme_types(quant.scheme_for("model.layers.5.mlp.experts")) == (23, 23, 14)


def test_ftw_quant_config_resolves_schemes(ftw_gguf_ckpt, monkeypatch):
    # the gguf->FTW boot residual: the config stage must not open a tensor table
    # the FTW does not have; new conversions record the source table in the index
    def _no_index(*args, **kwargs):
        raise AssertionError("gguf_tensor_index consulted on the FTW config path")

    monkeypatch.setattr("freetoken.models.gguf.reader.gguf_tensor_index", _no_index)
    _, out = ftw_gguf_ckpt
    _assert_ftw_schemes_resolve(out, legacy=False)
    # the conversion records the source's dense tensor table in the index
    with open(os.path.join(out, "freetoken_weight.json"), encoding="utf-8") as f:
        index = json.load(f)
    assert index["gguf_tensor_types"]["token_embd.weight"] == 8
    assert index["gguf_tensor_types"]["output.weight"] == 14
    assert index["gguf_tensor_types"]["blk.0.ffn_gate_exps.weight"] == 18


def test_ftw_quant_config_old_carrier_contract_rebuild(ftw_gguf_ckpt, tmp_path, monkeypatch):
    # an old-converter index (no gguf_tensor_types meta - the shape of the on-disk
    # 136G FTW-vision carrier) still resolves: dense from the family's convert
    # contract, experts from the persisted gguf_types triples; no re-conversion
    def _no_index(*args, **kwargs):
        raise AssertionError("gguf_tensor_index consulted on the FTW config path")

    monkeypatch.setattr("freetoken.models.gguf.reader.gguf_tensor_index", _no_index)
    _, out = ftw_gguf_ckpt
    old = tmp_path / "old-carrier"
    old.mkdir()
    with open(os.path.join(out, "freetoken_weight.json"), encoding="utf-8") as f:
        index = json.load(f)
    index.pop("gguf_tensor_types", None)
    with open(old / "freetoken_weight.json", "w", encoding="utf-8") as f:
        json.dump(index, f)
    shutil.copy(os.path.join(out, "source_metadata.gguf"), old / "source_metadata.gguf")
    _assert_ftw_schemes_resolve(str(old), legacy=True)


def test_ftw_quant_config_no_contract_old_carrier_says_reconvert(ftw_gguf_ckpt, tmp_path, monkeypatch):
    # an old-converter index for a family that pins no dense contract fails with
    # the re-convert direction, not a KeyError out of an empty type table
    import freetoken.models.register as register_mod
    from freetoken.models.register import checkpoint_quant_config, get_model_spec
    from freetoken.utils import cached_load_hf_config

    _, out = ftw_gguf_ckpt
    old = tmp_path / "old-no-hook"
    old.mkdir()
    with open(os.path.join(out, "freetoken_weight.json"), encoding="utf-8") as f:
        index = json.load(f)
    index.pop("gguf_tensor_types", None)
    with open(old / "freetoken_weight.json", "w", encoding="utf-8") as f:
        json.dump(index, f)
    shutil.copy(os.path.join(out, "source_metadata.gguf"), old / "source_metadata.gguf")
    real = register_mod._load_attr

    def _no_hook(module, attr):
        if attr == "ftw_tensor_types":
            raise AttributeError(module)
        return real(module, attr)

    monkeypatch.setattr(register_mod, "_load_attr", _no_hook)
    hf = cached_load_hf_config(str(old))
    with pytest.raises(RuntimeError, match="re-convert"):
        checkpoint_quant_config(str(old), hf, get_model_spec(hf.architectures[0]))


def test_ftw_replay_renames_legacy_qweight_segments(tmp_path):
    # the pre-re-platform carriers store packed ggml rows under the old "qweight"
    # segment leaf (the on-disk 136G FTW-vision shape); the replay must yield the
    # runtime's weight/weight_N vocabulary the GGUF methods create_weights build
    from freetoken.checkpoint.ftw import FTWWriter, iter_ftw_weights

    out = tmp_path / "legacy-names"
    w = FTWWriter(str(out))
    w.add_tensor("model.embed_tokens.qweight", torch.zeros((4, 8), dtype=torch.uint8))
    w.add_tensor("model.norm.weight", torch.zeros(4, dtype=torch.bfloat16))
    w.finalize({})
    got = {name: tuple(t.shape) for name, t in iter_ftw_weights(str(out), workers=1)}
    assert got == {"model.embed_tokens.weight": (4, 8), "model.norm.weight": (4,)}


def test_ftw_legacy_gate_up_banks_merge_on_load(tmp_path, monkeypatch):
    # pre-re-platform gguf carriers store three per-layer banks (gate/up/down);
    # the loader must replay them as the merged gate_up the runtime schema reads
    # (halves interleaved per expert), folding through per-layer scratch
    monkeypatch.setenv("FREETOKEN_SKIP_BANK_PIN", "1")  # CPU-only: no pin settle
    from freetoken.checkpoint.ftw import FTWWriter, load_ftw_banks

    out = tmp_path / "legacy-banks"
    w = FTWWriter(str(out))
    # bank entries are whole 4096-byte regions on real carriers (the writer's
    # alignment invariant), so the slice reads stay O_DIRECT-legal; every byte
    # position carries its own value (mod a prime) so a half-swap cannot pass
    idx = torch.arange(4096, dtype=torch.long)
    gate_t = (idx % 251).to(torch.uint8).view(2, 4, 512)
    up_t = ((idx + 37) % 251).to(torch.uint8).view(2, 4, 512)
    down_t = ((idx + 74) % 251).to(torch.uint8).view(2, 4, 512)
    for layer in range(2):
        w.add_tensor(f"gate#L{layer:05d}", gate_t.clone(), kind="experts_bank")
        w.add_tensor(f"up#L{layer:05d}", up_t.clone(), kind="experts_bank")
        w.add_tensor(f"down#L{layer:05d}", down_t.clone(), kind="experts_bank")
    w.finalize({"quant_format": "gguf", "gguf_types": [[18, 18, 23], [18, 18, 23]]})
    # the reader pulls through O_DIRECT, which bypasses the page cache the
    # buffered writes above still sit in: sync or the reads race the writeback
    os.sync()
    banks = load_ftw_banks(str(out), num_layers=2)
    assert sorted(banks.sources) == ["down", "gate_up"]
    for layer in range(2):
        gu = banks.sources["gate_up"][layer]
        assert tuple(gu.shape) == (2, 8, 512)
        assert torch.equal(gu[:, :4], gate_t)
        assert torch.equal(gu[:, 4:], up_t)
    # boot-time parity: the engine loads inside torch.inference_mode, so the merged
    # banks must be normal tensors the reader threads can still fold into
    with torch.inference_mode():
        banks = load_ftw_banks(str(out), num_layers=2)
    assert sorted(banks.sources) == ["down", "gate_up"]
    for layer in range(2):
        gu = banks.sources["gate_up"][layer]
        assert torch.equal(gu[:, :4], gate_t)
        assert torch.equal(gu[:, 4:], up_t)


def test_convert_mmproj_non_clip_fails_fast(tmp_path):
    from test_glm5_next_gguf import _write_mmproj_gguf
    from freetoken.checkpoint.convert import convert_checkpoint

    src = _write_ftw_source_gguf(tmp_path / "banks.gguf")
    bad = _write_mmproj_gguf(tmp_path / "wrong.gguf", arch="glm5next")
    with pytest.raises(ValueError, match="!= 'clip'"):
        convert_checkpoint(src, str(tmp_path / "ckpt"), device="cpu", mmproj_path=bad)


def test_pad_legacy_rows_segment_leaf_index(monkeypatch):
    # a legacy carrier's segment leaves are weight_N after the qweight replay;
    # the pad path must read the run index from the tail of the leaf, not the
    # separator, or int("_") raises on every multi-segment legacy tensor
    import freetoken.layers.quantization.configs as configs_mod
    import freetoken.layers.quantization.scheme as scheme_mod
    from freetoken.checkpoint.ftw import _pad_legacy_rows
    from freetoken.layers.quantization.ggml_types import pad_rows
    from freetoken.layers.quantization.registry import QuantKind

    class _Scheme:
        kind = QuantKind.GGUF

    class _Cfg:
        def scheme_for(self, prefix):
            return _Scheme()

    monkeypatch.setattr(configs_mod, "get_quant_config", lambda: _Cfg())
    monkeypatch.setattr(scheme_mod, "gguf_types", lambda scheme: (8, 8))  # two q8_0 segments
    # one q8_0 block is 34 bytes per 32 values: 24 blocks -> k=768, 768 % 512 -> pad > 0
    tensor = torch.zeros(4, 816, dtype=torch.uint8)
    out = _pad_legacy_rows({"name": "blk.3.ffn_up.weight_1"}, tensor)
    pad = pad_rows(8, (816 // 34) * 32)
    assert pad > 0
    assert out.shape == (4 + pad, 816)
    assert torch.equal(out[:4], tensor)
    assert (out[4:] == 0).all()

    # the un-suffixed first segment keeps its index-0 path
    out0 = _pad_legacy_rows({"name": "blk.3.ffn_up.weight"}, tensor)
    assert out0.shape == (4 + pad, 816)


def test_ftw_legacy_gate_up_zero_scratch_remap(tmp_path, monkeypatch):
    # remap-safe shapes (ALIGN-aligned expert blocks, nbytes == E*cx): gate/up read
    # jobs land directly in the merged bank via the affine remap (_merge_pair_jobs)
    # - no scratch, no fold - and the result must be byte-identical to the old
    # scratch+fold algorithm (gate -> [:, :g], up -> [:, g:]), with chunk both
    # larger and smaller than the expert block
    monkeypatch.setenv("FREETOKEN_SKIP_BANK_PIN", "1")  # CPU-only: no pin settle
    import freetoken.checkpoint.ftw as ftw_mod
    from freetoken.checkpoint.ftw import FTWWriter, load_ftw_banks

    E, g_rows, u_rows, row_bytes = 3, 16, 8, 512
    g_cx, u_cx = g_rows * row_bytes, u_rows * row_bytes  # 8192 / 4096, ALIGN-aligned
    gate_t = (torch.arange(E * g_cx) % 251).to(torch.uint8).view(E, g_rows, row_bytes)
    up_t = ((torch.arange(E * u_cx) + 61) % 251).to(torch.uint8).view(E, u_rows, row_bytes)
    down_t = ((torch.arange(E * u_cx) + 122) % 251).to(torch.uint8).view(E, u_rows, row_bytes)

    out = tmp_path / "legacy-zero-scratch"
    w = FTWWriter(str(out))
    for layer in range(2):
        w.add_tensor(f"gate#L{layer:05d}", gate_t.clone(), kind="experts_bank")
        w.add_tensor(f"up#L{layer:05d}", up_t.clone(), kind="experts_bank")
        w.add_tensor(f"down#L{layer:05d}", down_t.clone(), kind="experts_bank")
    w.finalize({"quant_format": "gguf", "gguf_types": [[18, 18, 23]] * 2})
    # the reader pulls through O_DIRECT: sync or the reads race the writeback
    os.sync()

    calls = []
    orig = ftw_mod._merge_pair_jobs

    def _spy(reader, entry, cx, base, stride, chunk):
        jobs = orig(reader, entry, cx, base, stride, chunk)
        calls.append((entry["name"], cx, base, stride, jobs))
        return jobs

    monkeypatch.setattr(ftw_mod, "_merge_pair_jobs", _spy)

    def _check(banks, tag):
        assert sorted(banks.sources) == ["down", "gate_up"], tag
        for layer in range(2):
            gu = banks.sources["gate_up"][layer]
            assert tuple(gu.shape) == (E, g_rows + u_rows, row_bytes), tag
            # byte-identical to the old scratch+fold algorithm
            assert torch.equal(gu[:, :g_rows], gate_t), tag
            assert torch.equal(gu[:, g_rows:], up_t), tag

    # chunk larger than the expert block: one job per expert
    banks = load_ftw_banks(str(out), num_layers=2)
    _check(banks, "chunk > cx")
    assert [c[0] for c in calls] == ["gate#L00000", "up#L00000", "gate#L00001", "up#L00001"]
    assert {c[1] for c in calls} == {g_cx, u_cx}
    merged_bytes = E * (g_cx + u_cx)
    for name, cx, base, _stride, jobs in calls:
        assert len(jobs) == E  # one job per expert
        assert sum(j[3] for j in jobs) == E * cx  # exact cover, no pad tail read
        end = max(j[2] + j[3] for j in jobs)
        assert end <= merged_bytes, (name, end)
        if name.startswith("up"):
            assert end == merged_bytes  # the up half ends exactly at the merged tail

    # chunk smaller than the expert block: several jobs per expert, same bytes;
    # boot-time parity: the engine loads inside inference_mode
    calls.clear()
    with torch.inference_mode():
        banks = load_ftw_banks(str(out), num_layers=2, chunk=4096)
    _check(banks, "chunk < cx")
    assert len(calls) == 4
    for name, cx, base, _stride, jobs in calls:
        per_expert = 2 if name.startswith("gate") else 1  # 8192/4096 vs 4096/4096
        assert len(jobs) == E * per_expert, (name, len(jobs))
        assert sum(j[3] for j in jobs) == E * cx

    # no job straddles an expert block: each job's dest run fits one block
    for name, cx, base, _stride, jobs in calls:
        for _f, _fo, do, ln in jobs:
            off = do - base
            blk = off // (g_cx + u_cx)
            assert blk == (off + ln - 1) // (g_cx + u_cx), (name, do, ln)
