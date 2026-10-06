"""glm5_next (GLM-5.3-Flash) GGUF config shim: ``glm5next.*`` metadata -> ModelConfig.

Runs off a synthetic glm5next GGUF written with gguf-py: the KV section mirrors the
real checkpoint's metadata (the exact llama.cpp key spellings from
research/phase1-config-keys.md), and token_embd.weight is a stub whose ggml
shape[-1] is the vocab -- the only tensor fact build_gguf_shim consumes. The
147 GB real checkpoint is never opened.
"""

from __future__ import annotations

import os

import warnings

from typing import TYPE_CHECKING

import gguf
import numpy as np
import pytest
import torch

from freetoken.attention.base import AttnType
from freetoken.models.config import FullAttentionGroupConfig, LinearGatedDeltaGroupConfig
from freetoken.models.glm5_next.args import DSA_LAYER, KDA_LAYER

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

_NUM_LAYERS = 45  # block_count 46 - nextn_predict_layers 1: the trunk
_DSA_IDS = tuple(range(3, _NUM_LAYERS, 4))  # 3, 7, ..., 43
_KDA_IDS = tuple(i for i in range(_NUM_LAYERS) if i not in _DSA_IDS)
_VOCAB = 154880
_CONTEXT = 1_048_576

# attention.head_count_kv over block_count (46): 0 marks the KDA/recurrent layer
# class. The NextN draft block blk.45 is a DSA layer and falls outside the trunk
# slice -- same split as the real file's metadata.
_HEAD_COUNT_KV = [1 if (i % 4 == 3 or i == _NUM_LAYERS) else 0 for i in range(46)]

# tokenizer KVs the image-serving path (phase 3) consumes: the chat template must
# render the image placeholder and the vocab must map it to the pinned id 154854
# (the real checkpoint's layout). Only merged into default-metadata writes: guard
# tests with custom metadata never boot multimodal.
_TOKENIZER_METADATA: dict = {
    "tokenizer.chat_template": "{%- macro emit_image() -%}<|image|>{%- endmacro -%}",
    "tokenizer.ggml.tokens": ["<pad>"] * _VOCAB,
}
_TOKENIZER_METADATA["tokenizer.ggml.tokens"][154854] = "<|image|>"

_GLM5NEXT_METADATA: dict = {
    "glm5next.block_count": 46,
    "glm5next.nextn_predict_layers": 1,
    "glm5next.context_length": _CONTEXT,
    "glm5next.embedding_length": 4096,
    "glm5next.feed_forward_length": 12288,
    "glm5next.leading_dense_block_count": 3,
    "glm5next.vocab_size": _VOCAB,
    "glm5next.attention.head_count": 64,
    "glm5next.attention.head_count_kv": _HEAD_COUNT_KV,
    "glm5next.attention.q_lora_rank": 1536,
    "glm5next.attention.kv_lora_rank": 512,
    "glm5next.attention.key_length": 512,
    "glm5next.attention.key_length_mla": 256,
    "glm5next.attention.value_length": 512,
    "glm5next.attention.value_length_mla": 256,
    "glm5next.attention.layer_norm_rms_epsilon": 9.999999747378752e-06,
    "glm5next.attention.layer_norm_epsilon": 1e-6,
    "glm5next.rope.dimension_count": 0,
    "glm5next.attention.indexer.head_count": 32,
    "glm5next.attention.indexer.key_length": 128,
    "glm5next.attention.indexer.top_k": 2048,
    "glm5next.attention.indexer.kpool": 4,
    "glm5next.kda.head_dim": 128,
    "glm5next.kda.gate_lower_bound": -5.0,
    "glm5next.ssm.conv_kernel": 4,
    "glm5next.hyper_connection.count": 4,
    "glm5next.hyper_connection.sinkhorn_iterations": 20,
    "glm5next.hyper_connection.epsilon": 9.999999974752427e-07,
    "glm5next.expert_count": 288,
    "glm5next.expert_used_count": 8,
    "glm5next.expert_feed_forward_length": 2048,
    "glm5next.expert_shared_count": 1,
    "glm5next.expert_shared_feed_forward_length": 2048,
    "glm5next.expert_weights_scale": 2.5,
    "glm5next.expert_weights_norm": True,
    "glm5next.expert_gating_func": 2,
    "glm5next.swiglu_clamp_exp": [10.0] * 46,
    "glm5next.swiglu_clamp_shexp": [10.0] * 46,
}


def _write_gguf(path, *, arch="glm5next", metadata=None, untied=False) -> str:
    import gguf

    meta = _GLM5NEXT_METADATA if metadata is None else metadata
    if metadata is None:
        # default-metadata writes double as multimodal-boot fixtures (phase 3)
        meta = {**meta, **_TOKENIZER_METADATA}
    w = gguf.GGUFWriter(str(path), arch)
    for key, val in sorted(meta.items()):
        if isinstance(val, bool):
            w.add_bool(key, val)
        elif isinstance(val, list):
            w.add_array(key, val)
        elif isinstance(val, str):
            w.add_string(key, val)
        elif isinstance(val, float):
            w.add_float32(key, val)
        else:
            # int32 so negative-valued guard fixtures survive the round-trip
            w.add_int32(key, val)
    # token_embd stub: ggml dims (hidden stub, vocab); the writer reverses numpy
    # shape into ggml dims, so shape[-1] as _vocab_size reads it is the vocab.
    w.add_tensor(
        "token_embd.weight",
        np.zeros(4 * _VOCAB, dtype=np.float16),
        raw_shape=(_VOCAB, 4),
        raw_dtype=gguf.GGMLQuantizationType.F16,
    )
    if untied:
        # output.weight stub, same shape as the embedding: its presence is the only
        # fact build_gguf_shim needs to untie (the real checkpoint ships it).
        w.add_tensor(
            "output.weight",
            np.zeros(4 * _VOCAB, dtype=np.float16),
            raw_shape=(_VOCAB, 4),
            raw_dtype=gguf.GGMLQuantizationType.F16,
        )
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)


@pytest.fixture(scope="session")
def glm5next_gguf(tmp_path_factory) -> str:
    return _write_gguf(tmp_path_factory.mktemp("glm5next-gguf") / "glm5next.gguf")


def _parse(path) -> object:
    from freetoken.models.gguf.config import build_gguf_shim
    from freetoken.models.glm5_next.gguf import parse_gguf_config

    return parse_gguf_config(build_gguf_shim(path))


def test_parse_gguf_config_maps_metadata_to_glm5_args(glm5next_gguf):
    cfg = _parse(glm5next_gguf)
    args = cfg.glm5_args

    # Trunk sizing: block_count includes the NextN draft block, per-layer arrays
    # span block_count and are sliced to the trunk.
    assert cfg.num_layers == _NUM_LAYERS
    assert args.dsa_layer_ids == _DSA_IDS
    assert args.kda_layer_ids == _KDA_IDS
    assert args.layer_types.count(KDA_LAYER) == 34
    assert args.layer_types.count(DSA_LAYER) == 11

    # Dense/MoE split rides on leading_dense_block_count.
    assert cfg.first_k_dense_replace == 3
    assert args.mlp_layer_types == ("dense",) * 3 + ("sparse",) * (_NUM_LAYERS - 3)

    # MLA (NoPE: rope.dimension_count is UNMAPPED, qk_rope_head_dim stays 0).
    assert (args.q_lora_rank, args.kv_lora_rank) == (1536, 512)
    assert (args.qk_nope_head_dim, args.qk_rope_head_dim, args.v_head_dim) == (256, 0, 256)
    assert args.latent_dim == 512
    assert args.mla_nope is True
    assert cfg.head_dim == 512
    assert cfg.rotary_config.rotary_dim == 0
    assert cfg.attn_sm_scale == pytest.approx(256**-0.5)

    # Indexer: 32 x 128, top_k 2048 selects kpool 4, tail pool always selected.
    assert (args.index_n_heads, args.index_head_dim, args.index_topk) == (32, 128, 2048)
    assert args.index_kpool == 4
    assert args.index_kpool_compress is True
    assert args.index_kpool_always_select_tail is True
    assert args.indexer_types == ("full",) * _NUM_LAYERS
    assert args.indexer_rope_interleave is False

    # KDA: one head count feeds the MLA and the KDA alike.
    assert args.num_heads == 64
    assert args.linear_num_heads == 64
    assert (args.linear_head_dim, args.linear_conv_kernel_dim) == (128, 4)
    assert args.linear_lower_bound == -5.0

    # mHC: 4 streams, sinkhorn 20; GGUF never carries the tau/post-mult keys.
    assert args.mhc is True
    assert args.mhc_num_residual_streams == 4
    assert args.mhc_sinkhorn_iterations == 20
    assert args.hc_eps == pytest.approx(1e-6, rel=1e-6)
    assert args.mhc_tau == 0.05
    assert args.mhc_post_mult_value == 2.0

    # Experts: 288 routed / 8 active / 1 shared / inter 2048, sigmoid-norm scaling.
    assert cfg.num_experts == 288
    assert cfg.num_experts_per_tok == 8
    assert cfg.moe_intermediate_size == 2048
    assert cfg.n_shared_experts == 1
    assert cfg.routed_scaling_factor == 2.5
    assert cfg.norm_topk_prob is True

    # Scalars: vocab from token_embd.weight shape, swiglu clamp flips hidden_act.
    assert cfg.vocab_size == _VOCAB
    assert cfg.hidden_size == 4096
    assert cfg.intermediate_size == 12288
    assert args.max_position == _CONTEXT
    assert args.norm_eps == pytest.approx(1e-5, rel=1e-6)
    assert args.swiglu_limit == 10.0
    assert cfg.swiglu_limit == 10.0
    assert cfg.hidden_act == "swiglu_clamp"
    assert cfg.tie_word_embeddings is True
    assert cfg.expert_quant == "none"  # GGUF carries no quantization_config

    # Attention groups, ordered by first layer id (0 < 3).
    linear, full = cfg.attention_groups
    assert isinstance(linear, LinearGatedDeltaGroupConfig)
    assert linear.variant == "kda"
    assert linear.layer_ids == _KDA_IDS
    assert (linear.num_key_heads, linear.key_head_dim) == (64, 128)
    assert linear.conv_kernel_dim == 4
    assert isinstance(full, FullAttentionGroupConfig)
    assert full.layer_ids == _DSA_IDS
    assert (full.mla, full.head_dim, full.num_kv_heads) == (True, 512, 1)
    assert (full.index_head_dim, full.num_index_layers, full.index_ratio) == (128, 11, 4)
    assert cfg.attn_type_for_layer(0) == AttnType.LINEAR
    assert cfg.attn_type_for_layer(3) == AttnType.DSA
    assert cfg.has_linear_attention and cfg.has_hybrid_attention
    assert cfg.architectures == ["Glm5NextGGUFForCausalLM"]


def test_build_gguf_shim_accepts_glm5next_and_rejects_unknown_arch(glm5next_gguf, tmp_path):
    from freetoken.models.gguf.config import GGUF_ARCH_TO_REGISTRY, build_gguf_shim

    assert GGUF_ARCH_TO_REGISTRY["glm5next"] == "Glm5NextGGUFForCausalLM"
    shim = build_gguf_shim(glm5next_gguf)
    assert shim.architectures == ["Glm5NextGGUFForCausalLM"]
    assert shim.model_type == "glm5next"
    # vocab comes off token_embd.weight's shape (ggml [hidden, vocab])
    assert shim.vocab_size == _VOCAB
    assert shim.tie_word_embeddings is True  # no output.weight tensor
    assert shim.metadata["glm5next.block_count"] == 46

    other = tmp_path / "unknown.gguf"
    _write_gguf(other, arch="gpt2", metadata={})
    with pytest.raises(ValueError, match="'gpt2' is not supported"):
        build_gguf_shim(str(other))


def test_registry_resolves_the_glm5next_gguf_spec():
    from freetoken.models.glm5_next import iter_gguf_weights, parse_gguf_config
    from freetoken.models.glm5_next import gguf as gguf_module
    from freetoken.models.register import get_model_spec

    spec = get_model_spec("Glm5NextGGUFForCausalLM")
    assert spec.module == "freetoken.models.glm5_next"
    # same model classes as the HF path; only the config/weight loaders swap.
    assert spec.model_cls == "Glm5NextForCausalLM"
    assert spec.parse_config == "parse_gguf_config"
    assert spec.iter_weights == "iter_gguf_weights"
    # the package exports what the spec names (test_models_registry checks callable-ness
    # for every registry entry; this pins the glm5next pair to the gguf adapter).
    assert parse_gguf_config is gguf_module.parse_gguf_config
    assert iter_gguf_weights is gguf_module.iter_gguf_weights


def test_engine_model_config_resolves_over_the_frozen_shim(glm5next_gguf):
    # model_config clears the unbuilt encoder sections on a copy of hf_config; for
    # GGUF that copy is the frozen GgufConfigShim, whose generated __setattr__
    # raises FrozenInstanceError on ANY assignment - every ft serve *.gguf died here.
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    cfg = EngineConfig(
        model_path=glm5next_gguf, tp_info=DistributedInfo(rank=0, size=1), dtype=torch.bfloat16
    )
    model_config = cfg.model_config
    assert model_config.architectures == ["Glm5NextGGUFForCausalLM"]
    # the shim itself stays untouched: the mmproj facts default to text-only, no
    # phantom audio section, metadata intact
    shim = cfg.hf_config
    assert shim.vision_config is None and shim.mmproj_path is None
    assert not hasattr(shim, "audio_config")
    assert shim.vocab_size == _VOCAB and shim.tie_word_embeddings is True
    assert shim.metadata["glm5next.block_count"] == 46


def test_iter_gguf_weights_rejects_expert_inclusion():
    # routed experts only serve from the offload cache (same contract as the HF reader);
    # the iterator is a generator, so the guard fires on first next() like weight.py's
    # ``yield from`` consumption.
    from freetoken.models.glm5_next import iter_gguf_weights

    with pytest.raises(ValueError, match="include_moe_experts=True is not supported"):
        list(
            iter_gguf_weights("unused.gguf", None, include_moe_experts=True, include_non_moe=True)
        )


def test_missing_required_metadata_key_fails_loudly(tmp_path):
    meta = {
        k: v for k, v in _GLM5NEXT_METADATA.items()
        if k != "glm5next.attention.q_lora_rank"
    }
    path = _write_gguf(tmp_path / "missing-key.gguf", metadata=meta)
    with pytest.raises(KeyError, match="missing GGUF metadata key glm5next.attention.q_lora_rank"):
        _parse(path)


def test_per_layer_varying_scalar_rejected(tmp_path):
    # llama accepts per-layer arrays for key_or_arr scalars; FreeToken carries one
    # value, so a non-uniform array must be refused, not folded.
    meta = {**_GLM5NEXT_METADATA, "glm5next.feed_forward_length": [12288, 12288, 2048]}
    path = _write_gguf(tmp_path / "varying.gguf", metadata=meta)
    with pytest.raises(ValueError, match="varies per layer"):
        _parse(path)


def test_uniform_per_layer_array_folds_to_scalar(tmp_path):
    meta = {**_GLM5NEXT_METADATA, "glm5next.attention.head_count": [64] * 46}
    path = _write_gguf(tmp_path / "uniform.gguf", metadata=meta)
    args = _parse(path).glm5_args
    assert args.num_heads == 64
    assert args.linear_num_heads == 64


def test_per_layer_array_must_span_block_count(tmp_path):
    meta = {**_GLM5NEXT_METADATA, "glm5next.block_count": 45}
    path = _write_gguf(tmp_path / "short-array.gguf", metadata=meta)
    with pytest.raises(ValueError, match="has 46 entries for block_count 45"):
        _parse(path)


def test_all_kda_trunk_rejected(tmp_path):
    # llama.cpp requires 0 < n_recr < n_layer; an all-KDA trunk has no DSA layer.
    meta = {**_GLM5NEXT_METADATA, "glm5next.attention.head_count_kv": [0] * 46}
    path = _write_gguf(tmp_path / "all-kda.gguf", metadata=meta)
    with pytest.raises(ValueError, match="KDA layers of 45"):
        _parse(path)


def test_indexer_topk_must_be_kpool_divisible(tmp_path):
    meta = {**_GLM5NEXT_METADATA, "glm5next.attention.indexer.top_k": 2047}
    path = _write_gguf(tmp_path / "topk.gguf", metadata=meta)
    with pytest.raises(ValueError, match="not divisible"):
        _parse(path)


def test_nextn_layer_count_must_leave_a_trunk(tmp_path):
    meta = {
        **_GLM5NEXT_METADATA,
        "glm5next.block_count": 1,
        "glm5next.nextn_predict_layers": 1,
    }
    path = _write_gguf(tmp_path / "no-trunk.gguf", metadata=meta)
    with pytest.raises(ValueError, match="nextn_predict_layers 1 outside 0..0"):
        _parse(path)


def test_untied_file_marks_tie_word_embeddings_false(tmp_path):
    # The real checkpoint ships output.weight (untied); Phase 2 maps it to lm_head.
    from freetoken.models.gguf.config import build_gguf_shim

    path = _write_gguf(tmp_path / "untied.gguf", untied=True)
    assert build_gguf_shim(path).tie_word_embeddings is False
    assert _parse(path).tie_word_embeddings is False


def test_scalar_head_count_kv_broadcasts_over_layers(tmp_path):
    # key_or_arr: a scalar applies to every layer (llama-model.cpp); the resulting
    # uniform split trips the strict-minority check instead of a TypeError.
    meta = {**_GLM5NEXT_METADATA, "glm5next.attention.head_count_kv": 0}
    path = _write_gguf(tmp_path / "scalar-kv.gguf", metadata=meta)
    with pytest.raises(ValueError, match="45 KDA layers of 45") as excinfo:
        _parse(path)
    assert "scalar head_count_kv applies to every layer" in str(excinfo.value)


def test_scalar_helper_rejects_empty_and_varying_arrays():
    # An empty per-layer array never survives a gguf-py round-trip (dropped at
    # write), so the guard is exercised directly on the folding helper.
    from freetoken.models.glm5_next.gguf import _scalar

    with pytest.raises(ValueError, match="empty per-layer array"):
        _scalar([], "swiglu_clamp_exp")
    with pytest.raises(ValueError, match="varies per layer"):
        _scalar([2048, 2048, 1024], "expert_feed_forward_length")
    assert _scalar([2048] * 46, "expert_feed_forward_length") == 2048
    assert _scalar(8, "expert_used_count") == 8


@pytest.mark.parametrize("kpool", [0, -4])
def test_indexer_kpool_must_be_positive(tmp_path, kpool):
    # A negative kpool would pass the modulo check and silently disable compression.
    meta = {**_GLM5NEXT_METADATA, "glm5next.attention.indexer.kpool": kpool}
    path = _write_gguf(tmp_path / f"kpool-{kpool}.gguf", metadata=meta)
    with pytest.raises(ValueError, match="must be > 0"):
        _parse(path)


@pytest.mark.parametrize("dense", [46, -1])
def test_leading_dense_block_count_must_be_in_range(tmp_path, dense):
    meta = {**_GLM5NEXT_METADATA, "glm5next.leading_dense_block_count": dense}
    path = _write_gguf(tmp_path / f"dense-{dense}.gguf", metadata=meta)
    with pytest.raises(ValueError, match="leading_dense_block_count"):
        _parse(path)


def test_expert_gating_func_must_be_sigmoid(tmp_path):
    # 1 == llama.cpp SOFTMAX; FreeToken only implements the sigmoid/noaux_tc router.
    meta = {**_GLM5NEXT_METADATA, "glm5next.expert_gating_func": 1}
    path = _write_gguf(tmp_path / "gating.gguf", metadata=meta)
    with pytest.raises(ValueError, match="expert_gating_func 1"):
        _parse(path)


def test_roped_variant_rejected(tmp_path):
    # glm5next.cpp asserts n_rot() == 0 ("nope-only"); a nonzero rope dim must not
    # silently serve the NoPE model.
    meta = {**_GLM5NEXT_METADATA, "glm5next.rope.dimension_count": 64}
    path = _write_gguf(tmp_path / "roped.gguf", metadata=meta)
    with pytest.raises(ValueError, match="nope-only"):
        _parse(path)


def test_metadata_vocab_mismatch_rejected(tmp_path):
    meta = {**_GLM5NEXT_METADATA, "glm5next.vocab_size": 154881}
    path = _write_gguf(tmp_path / "vocab.gguf", metadata=meta)
    with pytest.raises(ValueError, match="vocab_size 154881"):
        _parse(path)


def test_shared_expert_ffn_mismatch_rejected(tmp_path):
    # FreeToken derives the shared-expert width like llama (n_ff_exp * n_shared).
    meta = {**_GLM5NEXT_METADATA, "glm5next.expert_shared_feed_forward_length": 4096}
    path = _write_gguf(tmp_path / "shexp-ffn.gguf", metadata=meta)
    with pytest.raises(ValueError, match="expert_shared_feed_forward_length 4096"):
        _parse(path)


def test_indexer_norm_epsilon_mismatch_rejected(tmp_path):
    # FreeToken hardcodes the indexer k-norm eps to 1e-6 (attention.py).
    meta = {**_GLM5NEXT_METADATA, "glm5next.attention.layer_norm_epsilon": 2e-6}
    path = _write_gguf(tmp_path / "ln-eps.gguf", metadata=meta)
    with pytest.raises(ValueError, match="layer_norm_epsilon"):
        _parse(path)


def test_shexp_clamp_mismatch_rejected(tmp_path):
    # FreeToken carries one clamp for routed and shared experts.
    meta = {**_GLM5NEXT_METADATA, "glm5next.swiglu_clamp_shexp": [20.0] * 46}
    path = _write_gguf(tmp_path / "shexp-clamp.gguf", metadata=meta)
    with pytest.raises(ValueError, match="swiglu_clamp_shexp"):
        _parse(path)


def test_two_nextn_layers_shrink_the_trunk(tmp_path):
    meta = {**_GLM5NEXT_METADATA, "glm5next.nextn_predict_layers": 2}
    path = _write_gguf(tmp_path / "nextn2.gguf", metadata=meta)
    cfg = _parse(path)
    args = cfg.glm5_args
    assert cfg.num_layers == 44
    assert args.layer_types.count(KDA_LAYER) == 33
    assert args.layer_types.count(DSA_LAYER) == 11
    assert args.mlp_layer_types == ("dense",) * 3 + ("sparse",) * 41


def test_negative_nextn_predict_layers_rejected(tmp_path):
    # A negative nextn would inflate num_layers past block_count and silently turn
    # the draft block into a trunk layer.
    meta = {**_GLM5NEXT_METADATA, "glm5next.nextn_predict_layers": -1}
    path = _write_gguf(tmp_path / "nextn-neg.gguf", metadata=meta)
    with pytest.raises(ValueError, match="nextn_predict_layers -1 outside 0..45"):
        _parse(path)


def test_indexer_topk_must_be_positive(tmp_path):
    meta = {**_GLM5NEXT_METADATA, "glm5next.attention.indexer.top_k": -8}
    path = _write_gguf(tmp_path / "topk-neg.gguf", metadata=meta)
    with pytest.raises(ValueError, match="must be > 0"):
        _parse(path)


@pytest.mark.parametrize(
    "optional_key",
    [
        "glm5next.expert_gating_func",
        "glm5next.rope.dimension_count",
        "glm5next.vocab_size",
        "glm5next.attention.layer_norm_epsilon",
        "glm5next.expert_shared_feed_forward_length",
        "glm5next.swiglu_clamp_shexp",
    ],
)
def test_optional_guard_keys_may_be_absent(tmp_path, optional_key):
    # The mismatch guards are presence-conditional: a converter that omits the key
    # must parse, not trip the guard.
    meta = {k: v for k, v in _GLM5NEXT_METADATA.items() if k != optional_key}
    path = _write_gguf(tmp_path / "absent-key.gguf", metadata=meta)
    cfg = _parse(path)
    assert cfg.num_layers == _NUM_LAYERS


# =====================================================================================
# Phase 2: iter_gguf_weights / iter_gguf_expert_sources over a synthetic weight file.
#
# The Phase 1 fixture above is metadata-only (the config shim consumes one tensor
# fact). The weight iterator consumes the tensor table, so this section adds a
# second session-scoped fixture with a small trunk: layer 0 KDA + dense FFN, layer 1
# KDA + MoE, layer 2 DSA + MoE, and blk.3 as the skipped MTP/NextN block. Every
# config dim is shrunk (head 4, hidden 64, kv_lora 32, moe ffn 256) so the Q8_0
# blocks stay tiny; all geometry asserts read THESE dims, never the real
# checkpoint's, and the 147 GB gguf is never opened.
# =====================================================================================

_VOCAB_S = 256
_H = 256  # embedding_length (a multiple of 256 so the Q6_K head table packs)
_NH = 4  # attention.head_count
_HD = 32  # kda.head_dim -> KDA proj = _NH * _HD
_QL = 32  # attention.q_lora_rank
_KL = 32  # attention.kv_lora_rank
_QK = 32  # attention.key_length_mla (qk_nope)
_VD = 32  # attention.value_length_mla
_DFF = 128  # feed_forward_length (dense FFN)
_EFF = 256  # expert_feed_forward_length: Q6_K down bank needs ne0 % 256
_NE = 4  # expert_count
_IDXH = 2  # indexer head_count
_IDXD = 32  # indexer key_length
_KPOOL = 4
_HCN = 2  # hyper_connection.count -> hc_dim 128, hc_mix 8
_CONV = 4  # ssm.conv_kernel
_PROJ = _NH * _HD  # KDA d_inner
_IT_BLOCK_COUNT = 4
_IT_LAYERS = 3  # trunk layers 0..2; blk.3 is the MTP block
_KDA_ITER_LAYERS = (0, 1)
_DSA_ITER_LAYERS = (2,)
_MOE_ITER_LAYERS = (1, 2)

_ITER_METADATA = {
    **_GLM5NEXT_METADATA,
    "glm5next.block_count": _IT_BLOCK_COUNT,
    "glm5next.vocab_size": _VOCAB_S,
    "glm5next.embedding_length": _H,
    "glm5next.attention.head_count": _NH,
    "glm5next.attention.head_count_kv": [0, 0, 1, 1],
    "glm5next.attention.q_lora_rank": _QL,
    "glm5next.attention.kv_lora_rank": _KL,
    "glm5next.attention.key_length": _QK,
    "glm5next.attention.key_length_mla": _QK,
    "glm5next.attention.value_length": _VD,
    "glm5next.attention.value_length_mla": _VD,
    "glm5next.attention.indexer.head_count": _IDXH,
    "glm5next.attention.indexer.key_length": _IDXD,
    "glm5next.attention.indexer.top_k": 8,
    "glm5next.attention.indexer.kpool": _KPOOL,
    "glm5next.kda.head_dim": _HD,
    "glm5next.feed_forward_length": _DFF,
    "glm5next.leading_dense_block_count": 1,
    "glm5next.expert_count": _NE,
    "glm5next.expert_used_count": 2,
    "glm5next.expert_feed_forward_length": _EFF,
    "glm5next.expert_shared_count": 1,
    "glm5next.expert_shared_feed_forward_length": _EFF,
    "glm5next.hyper_connection.count": _HCN,
    "glm5next.swiglu_clamp_exp": [10.0] * _IT_BLOCK_COUNT,
    "glm5next.swiglu_clamp_shexp": [10.0] * _IT_BLOCK_COUNT,
}

# A_log ground truth: the gguf stores -exp(A_log) per head (kimi-k3 convention).
_SSM_A_TRUE = np.array([0.5, -1.0, 2.0, -0.25], np.float32)
# Q6_K down bank payload: random bytes are fine, the bank must pass through verbatim.
# Q6_K down bank: payload bytes random but the super-block scale (bytes 208:210)
# pinned to a sane fp16 value - random d would decode to inf/NaN and poison the
# moe_vec parity smoke (the dequant reference AND the kernel both chain through it).
_DOWN_BANK_BYTES = np.zeros((_NE, _H, 210), np.uint8)
_rng = np.random.default_rng(1234)
_DOWN_BANK_BYTES[:, :, :208] = _rng.integers(0, 256, (_NE, _H, 208), dtype=np.uint8)
_DOWN_BANK_BYTES[:, :, 208:210] = np.frombuffer(np.float16(1.5).tobytes(), np.uint8)


def _q8_vals(rows, cols, tag):
    """Element values whose Q8_0 payload encodes (tag, row, col) in every int8."""
    vals = (tag * 1000 + np.arange(rows)[:, None] * cols + np.arange(cols)) % 254 - 127
    return vals.astype(np.int8)


def _pack_q8_0(vals):
    """Q8_0 blocks (fp16 scale 1.0 + int8 quants) shaped for gguf-py raw writes: the
    uint8 byte shape [..., row_bytes] converts back to the ggml element shape."""
    cols = vals.shape[-1]
    assert cols % 32 == 0
    nb = cols // 32
    flat = np.ascontiguousarray(vals, dtype=np.int8).reshape(-1, cols)
    out = np.zeros((flat.shape[0], nb * 34), dtype=np.uint8)
    out[:, 1::34] = 0x3C  # fp16 scale 1.0, little-endian
    for b in range(nb):
        out[:, b * 34 + 2:b * 34 + 34] = flat[:, b * 32:(b + 1) * 32].view(np.uint8)
    return out.reshape(vals.shape[:-1] + (nb * 34,))


def _bank_vals(tag):
    """Routed-expert bank fills: expert e's rows encode (tag, e, ffn, col)."""
    e = np.arange(_NE)[:, None, None]
    f = np.arange(_EFF)[None, :, None]
    c = np.arange(_H)[None, None, :]
    return ((tag * 1000 + e * 2000 + f * _H + c) % 254 - 127).astype(np.int8)


def _iter_tensor_set() -> dict:
    """{gguf tensor name -> (payload ndarray, gguf raw dtype or None)} for the trunk.

    Float arrays carry the torch shape (C-order, ggml ne0 last); quantized payloads
    are uint8 byte shapes that gguf-py converts back to the ggml element shape."""
    import gguf

    q8 = gguf.GGMLQuantizationType.Q8_0
    q6k = gguf.GGMLQuantizationType.Q6_K
    ts: dict = {}

    def add(name, arr, qt=None):
        ts[name] = (arr, qt)

    # globals: Q8_0 embedding, untied F16 head, F32 final norm
    add("token_embd.weight", _pack_q8_0(_q8_vals(_VOCAB_S, _H, 0)), q8)
    # untied Q6_K head table (the convert contract; bytes are opaque to the tests)
    add("output.weight", np.zeros((_VOCAB_S, _H // 256 * 210), np.uint8), q6k)
    add("output_norm.weight", np.full(_H, 0.5, np.float32))

    hc_rows, hc_cols = (2 + _HCN) * _HCN, _HCN * _H
    for n in range(_IT_LAYERS):
        add(f"blk.{n}.attn_norm.weight", np.full(_H, 0.25, np.float32))
        add(f"blk.{n}.ffn_norm.weight", np.full(_H, 0.75, np.float32))
        add(f"blk.{n}.hc_attn_fn.weight", _pack_q8_0(_q8_vals(hc_rows, hc_cols, n)), q8)
        add(f"blk.{n}.hc_attn_base.weight", np.arange(hc_rows, dtype=np.float32) / 8)
        add(f"blk.{n}.hc_attn_scale.weight", np.array([0.1, 0.2, 0.3], np.float32))
        add(f"blk.{n}.hc_ffn_fn.weight", _pack_q8_0(_q8_vals(hc_rows, hc_cols, 10 + n)), q8)
        add(f"blk.{n}.hc_ffn_base.weight", np.arange(hc_rows, dtype=np.float32) / 8)
        add(f"blk.{n}.hc_ffn_scale.weight", np.array([0.4, 0.5, 0.6], np.float32))

    for n in _KDA_ITER_LAYERS:
        for tag, suffix, rows in (
            (0, "attn_q", _PROJ), (1, "attn_k", _PROJ), (2, "attn_v", _PROJ),
            (3, "ssm_beta", _NH), (4, "ssm_f_a", _HD), (5, "ssm_g_a", _HD),
        ):
            add(f"blk.{n}.{suffix}.weight", _pack_q8_0(_q8_vals(rows, _H, tag)), q8)
        for tag, part in ((1.0, "q"), (2.0, "k"), (3.0, "v")):
            add(f"blk.{n}.ssm_conv1d_{part}.weight", np.full((_PROJ, 1, _CONV), tag, np.float32))
        add(f"blk.{n}.ssm_f_b.weight", _pack_q8_0(_q8_vals(_PROJ, _HD, 10)), q8)
        add(f"blk.{n}.ssm_g_b.weight", _pack_q8_0(_q8_vals(_PROJ, _HD, 11)), q8)
        add(f"blk.{n}.attn_output.weight", _pack_q8_0(_q8_vals(_H, _PROJ, 12)), q8)
        add(f"blk.{n}.ssm_a", (-np.exp(_SSM_A_TRUE)).astype(np.float32))
        add(f"blk.{n}.ssm_dt.bias", np.arange(_PROJ, dtype=np.float32) / 16 - 4)
        add(f"blk.{n}.ssm_norm.weight", np.full(_HD, 0.5, np.float32))

    add("blk.0.ffn_gate.weight", _pack_q8_0(_q8_vals(_DFF, _H, 20)), q8)
    add("blk.0.ffn_up.weight", _pack_q8_0(_q8_vals(_DFF, _H, 21)), q8)
    add("blk.0.ffn_down.weight", _pack_q8_0(_q8_vals(_H, _DFF, 22)), q8)

    for n in _MOE_ITER_LAYERS:
        add(f"blk.{n}.ffn_gate_inp.weight", (
            np.arange(_NE, dtype=np.float32)[:, None] + np.arange(_H)[None, :] / 64
        ).astype(np.float32))
        add(f"blk.{n}.exp_probs_b.bias", np.array([0.01, -0.02, 0.03, -0.04], np.float32))
        add(f"blk.{n}.ffn_gate_shexp.weight", _pack_q8_0(_q8_vals(_EFF, _H, 30)), q8)
        add(f"blk.{n}.ffn_up_shexp.weight", _pack_q8_0(_q8_vals(_EFF, _H, 31)), q8)
        add(f"blk.{n}.ffn_down_shexp.weight", _pack_q8_0(_q8_vals(_H, _EFF, 32)), q8)
        add(f"blk.{n}.ffn_gate_exps.weight", _pack_q8_0(_bank_vals(0)), q8)
        add(f"blk.{n}.ffn_up_exps.weight", _pack_q8_0(_bank_vals(7)), q8)
        add(f"blk.{n}.ffn_down_exps.weight", _DOWN_BANK_BYTES, q6k)

    add("blk.2.attn_q_a.weight", _pack_q8_0(_q8_vals(_QL, _H, 40)), q8)
    add("blk.2.attn_q_a_norm.weight", np.full(_QL, 0.5, np.float32))
    add("blk.2.attn_q_b.weight", _pack_q8_0(_q8_vals(_NH * _QK, _QL, 41)), q8)
    add("blk.2.attn_kv_a_mqa.weight", _pack_q8_0(_q8_vals(_KL, _H, 42)), q8)
    add("blk.2.attn_kv_a_norm.weight", np.full(_KL, 0.5, np.float32))
    add("blk.2.attn_output.weight", _pack_q8_0(_q8_vals(_H, _NH * _VD, 43)), q8)
    # kv_b pieces: attn_k_b reads [head, kv_lora, qk] and stores h*kl + kv (constant
    # along qk, so the fused rows expose the transpose); attn_v_b reads
    # [head, v_head, kv_lora] and stores -128 + h*kl + v (disjoint sign range).
    k_b = np.broadcast_to(
        np.arange(_NH)[:, None, None] * _KL + np.arange(_KL)[None, :, None],
        (_NH, _KL, _QK),
    ).astype(np.int8)
    v_b = np.broadcast_to(
        -128 + np.arange(_NH)[:, None, None] * _KL + np.arange(_VD)[None, :, None],
        (_NH, _VD, _KL),
    ).astype(np.int8)
    add("blk.2.attn_k_b.weight", _pack_q8_0(k_b), q8)
    add("blk.2.attn_v_b.weight", _pack_q8_0(v_b), q8)

    add("blk.2.indexer.attn_k.weight", _pack_q8_0(_q8_vals(_IDXD, _H, 50)), q8)
    add("blk.2.indexer.attn_q_b.weight", _pack_q8_0(_q8_vals(_IDXH * _IDXD, _QL, 51)), q8)
    add("blk.2.indexer.proj.weight", (
        np.arange(_IDXH, dtype=np.float32)[:, None] + np.arange(_H)[None, :] / 64
    ).astype(np.float32))
    add("blk.2.indexer.k_norm.weight", np.full(_IDXD, 0.5, np.float32))
    add("blk.2.indexer.k_norm.bias", np.arange(_IDXD, dtype=np.float32) / 8 - 2)
    add("blk.2.indexer_compressor_gate.weight", _pack_q8_0(_q8_vals(_IDXD, _H, 52)), q8)
    add("blk.2.indexer_compressor_ape.weight", (
        np.arange(_KPOOL, dtype=np.float32)[:, None] + np.arange(_IDXD)[None, :] / 64
    ).astype(np.float32))

    # MTP draft block blk.3: block weights + .nextn glue + a bank; all skipped.
    add("blk.3.attn_norm.weight", np.full(_H, 0.25, np.float32))
    add("blk.3.nextn.eh_proj.weight", _pack_q8_0(_q8_vals(_PROJ, _H, 60)), q8)
    add("blk.3.nextn.enorm.weight", np.full(_H, 0.25, np.float32))
    add("blk.3.ffn_gate_exps.weight", _pack_q8_0(_bank_vals(0)), q8)
    return ts


def _write_iter_gguf(path, *, metadata=None, tensors=None) -> str:
    import gguf

    meta = _ITER_METADATA if metadata is None else metadata
    if metadata is None:
        meta = {**meta, **_TOKENIZER_METADATA}
    entries = _iter_tensor_set() if tensors is None else tensors
    w = gguf.GGUFWriter(str(path), "glm5next")
    for key, val in sorted(meta.items()):
        if isinstance(val, bool):
            w.add_bool(key, val)
        elif isinstance(val, list):
            w.add_array(key, val)
        elif isinstance(val, str):
            w.add_string(key, val)
        elif isinstance(val, float):
            w.add_float32(key, val)
        else:
            # int32 so negative-valued guard fixtures survive the round-trip
            w.add_int32(key, val)
    for name, (arr, qt) in entries.items():
        w.add_tensor(name, arr, raw_dtype=qt)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)


@pytest.fixture(scope="session")
def glm5next_iter_gguf(tmp_path_factory) -> str:
    return _write_iter_gguf(tmp_path_factory.mktemp("glm5next-iter") / "glm5next-iter.gguf")


@pytest.fixture(autouse=True)
def _single_rank_tp():
    # iter_gguf_weights enforces TP=1 via get_tp_info; weight-reader tests run TP=1
    # (same guarded idiom as the kda/model/snapshot tests).
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _iter_params(path) -> dict:
    from freetoken.models.glm5_next import iter_gguf_weights

    return dict(
        iter_gguf_weights(path, None, include_moe_experts=False, include_non_moe=True)
    )


def _expected_iter_names() -> set:
    names = {"model.embed_tokens.qweight", "lm_head.qweight", "model.norm.weight"}
    for n in range(_IT_LAYERS):
        names |= {
            f"model.layers.{n}.{s}"
            for s in (
                "input_layernorm.weight", "post_attention_layernorm.weight",
                "hc_attn_fn", "hc_attn_base", "hc_attn_scale",
                "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale",
            )
        }
    for n in _KDA_ITER_LAYERS:
        names |= {
            f"model.layers.{n}.self_attn.{s}"
            for s in (
                "in_proj.qweight", "conv1d.weight", "f_b_proj.qweight",
                "g_b_proj.qweight", "o_proj.qweight", "o_norm.weight",
                "A_log", "dt_bias",
            )
        }
    names |= {
        f"model.layers.2.self_attn.{s}"
        for s in (
            "q_a_proj.qweight", "q_a_layernorm.weight", "q_b_proj.qweight",
            "kv_a_proj_with_mqa.qweight", "kv_a_layernorm.weight", "o_proj.qweight",
            "kv_b_proj.weight", "indexer.wk.weight", "indexer.wq_b.weight",
            "indexer.weights_proj.weight", "indexer.k_norm.weight",
            "indexer.k_norm.bias", "indexer.index_kpool_compress_gate",
            "indexer.index_kpool_compress_ape",
        )
    }
    names |= {
        f"model.layers.0.mlp.{s}"
        for s in ("gate_proj.qweight", "up_proj.qweight", "down_proj.qweight")
    }
    for n in _MOE_ITER_LAYERS:
        names |= {
            f"model.layers.{n}.mlp.{s}"
            for s in (
                "gate.weight", "e_score_correction_bias",
                "shared_experts.gate_proj.qweight", "shared_experts.up_proj.qweight",
                "shared_experts.down_proj.qweight",
            )
        }
    return names


def test_iter_gguf_weights_full_trunk_names_and_casts(glm5next_iter_gguf):
    out = _iter_params(glm5next_iter_gguf)
    assert set(out) == _expected_iter_names()
    # blk.45-style MTP tensors (block weights, .nextn glue, a bank) never surface
    assert not any(name.startswith("model.layers.3.") for name in out)
    for name, t in out.items():
        if name.endswith(".qweight"):
            assert t.dtype == torch.uint8 and t.dim() == 2, name
        else:
            assert t.dtype in (torch.bfloat16, torch.float32), name
    # packed geometry follows the fixture's own dims, not the real file's
    assert out["model.embed_tokens.qweight"].shape == (_VOCAB_S, _H // 32 * 34)
    assert out["lm_head.qweight"].shape == (_VOCAB_S, _H // 256 * 210)  # Q6_K rows
    assert out["model.layers.1.self_attn.f_b_proj.qweight"].shape == (_PROJ, _HD // 32 * 34)
    assert out["model.layers.1.self_attn.o_proj.qweight"].shape == (_H, _PROJ // 32 * 34)
    # dense-name yields: bf16 norms, fp32 where the family keeps fp32
    assert out["model.layers.0.input_layernorm.weight"].dtype == torch.bfloat16
    assert out["model.layers.0.self_attn.o_norm.weight"].dtype == torch.bfloat16
    assert out["model.layers.0.hc_attn_fn"].dtype == torch.float32
    assert out["model.layers.0.hc_attn_fn"][0, 0].item() == -127.0  # Q8_0 dequant, scale 1
    assert out["model.layers.1.mlp.e_score_correction_bias"].dtype == torch.float32
    assert out["model.layers.2.self_attn.indexer.wk.weight"].dtype == torch.bfloat16


def test_iter_gguf_weights_in_proj_fusion_matches_consumer_split(glm5next_iter_gguf):
    out = _iter_params(glm5next_iter_gguf)
    fused = out["model.layers.1.self_attn.in_proj.qweight"]
    # Glm5NextKDA._in_proj_split = [P, P, P, H, D, D] (kda.py) slices the fused GEMM
    # in the q|k|v|b|f_a|g_a concat order of weight.py _KDA_IN_PROJ; every piece is
    # Q8_0 with ne0 == hidden, so the packed dim-0 concat is byte-exact.
    assert fused.dtype == torch.uint8
    assert fused.shape == (3 * _PROJ + _NH + 2 * _HD, _H // 32 * 34)
    row = 0
    for suffix, rows, tag in (
        ("attn_q", _PROJ, 0), ("attn_k", _PROJ, 1), ("attn_v", _PROJ, 2),
        ("ssm_beta", _NH, 3), ("ssm_f_a", _HD, 4), ("ssm_g_a", _HD, 5),
    ):
        expected = torch.from_numpy(_pack_q8_0(_q8_vals(rows, _H, tag)))
        assert torch.equal(fused[row:row + rows], expected), f"slot {suffix} out of order"
        row += rows
    assert row == fused.shape[0]


def test_iter_gguf_weights_conv1d_channel_order(glm5next_iter_gguf):
    out = _iter_params(glm5next_iter_gguf)
    fused = out["model.layers.0.self_attn.conv1d.weight"]
    # q|k|v conv pieces F32 in the gguf, cat on the channel axis, bf16 out.
    assert fused.dtype == torch.bfloat16
    assert fused.shape == (3 * _PROJ, 1, _CONV)
    for i, tag in enumerate((1.0, 2.0, 3.0)):
        assert torch.equal(
            fused[i * _PROJ:(i + 1) * _PROJ],
            torch.full((_PROJ, 1, _CONV), tag).to(torch.bfloat16),
        ), f"conv slot {i} out of order"


def test_iter_gguf_weights_kv_b_axis_convention(glm5next_iter_gguf):
    out = _iter_params(glm5next_iter_gguf)
    fused = out["model.layers.2.self_attn.kv_b_proj.weight"]
    # attn_k_b stores [head, kv_lora, qk] and fuses with the last-two-axes transpose;
    # attn_v_b stores [head, v_head, kv_lora] as-is; per head the rows are
    # [k(qk); v(v_head)] over kv_lora (attention.py takes w[:, :qk] = w_uk).
    assert fused.dtype == torch.bfloat16
    assert fused.shape == (_NH * (_QK + _VD), _KL)
    expected = torch.empty(_NH * (_QK + _VD), _KL, dtype=torch.bfloat16)
    for h in range(_NH):
        k_rows = h * (_QK + _VD) + torch.arange(_QK)
        expected[k_rows] = (torch.arange(_KL) + h * _KL).to(torch.bfloat16)
        v_rows = h * (_QK + _VD) + _QK + torch.arange(_VD)
        expected[v_rows] = (torch.arange(-128, -128 + _VD) + h * _KL).unsqueeze(1).to(torch.bfloat16)
    assert torch.equal(fused, expected)


def test_iter_gguf_weights_a_log_dt_bias_and_underscore_params(glm5next_iter_gguf):
    out = _iter_params(glm5next_iter_gguf)
    a_log = out["model.layers.1.self_attn.A_log"]
    assert a_log.dtype == torch.float32 and a_log.shape == (_NH,)
    torch.testing.assert_close(a_log, torch.from_numpy(_SSM_A_TRUE), rtol=1e-6, atol=1e-7)
    # round trip: the gguf stores -exp(A_log); log(-x) must invert it in fp32
    torch.testing.assert_close(
        -torch.exp(a_log), torch.from_numpy(-np.exp(_SSM_A_TRUE)), rtol=1e-6, atol=1e-7
    )
    # ssm_dt ships as a .bias tensor but maps to the plain dt_bias param, fp32
    dt = out["model.layers.1.self_attn.dt_bias"]
    assert dt.dtype == torch.float32
    torch.testing.assert_close(
        dt, torch.arange(_PROJ, dtype=torch.float32) / 16 - 4, rtol=0, atol=0
    )
    # indexer_compressor_* underscore spellings map to the raw (no .weight) params
    gate = out["model.layers.2.self_attn.indexer.index_kpool_compress_gate"]
    ape = out["model.layers.2.self_attn.indexer.index_kpool_compress_ape"]
    assert gate.dtype == torch.bfloat16 and gate.shape == (_IDXD, _H)
    assert ape.dtype == torch.float32 and ape.shape == (_KPOOL, _IDXD)


@pytest.mark.parametrize(
    ("bogus", "match"),
    [
        ("blk.1.mystery.weight", r"unmapped glm5next GGUF tensor: blk\.1\.mystery\.weight"),
        ("mystery.weight", r"unmapped glm5next GGUF tensor: mystery\.weight"),
    ],
)
def test_iter_gguf_weights_unknown_tensor_names_the_tensor(tmp_path, bogus, match):
    ts = _iter_tensor_set()
    ts[bogus] = (np.full(_H, 0.25, np.float32), None)
    path = _write_iter_gguf(tmp_path / "unknown.gguf", tensors=ts)
    with pytest.raises(ValueError, match=match):
        _iter_params(path)


def test_iter_gguf_weights_a_log_guard_rejects_nonnegative_ssm_a(tmp_path):
    # ssm_a holds -exp(A_log) per head; a >= 0 entry cannot yield a real A_log.
    ts = _iter_tensor_set()
    ts["blk.0.ssm_a"] = (np.array([-1.0, -2.0, 0.5, -0.5], np.float32), None)
    path = _write_iter_gguf(tmp_path / "bad-a-log.gguf", tensors=ts)
    with pytest.raises(ValueError, match=r"blk\.0\.ssm_a: ssm_a holds -exp\(A_log\) but 1 entries are >= 0"):
        _iter_params(path)


@pytest.mark.parametrize(
    ("drop", "match"),
    [
        ("blk.1.ssm_g_a.weight", r"incomplete in_proj groups \[\(1, \['b', 'f_a', 'k', 'q', 'v'\]\)\]"),
        ("blk.1.ssm_conv1d_v.weight", r"incomplete conv1d groups \[\(1, \['k', 'q'\]\)\]"),
        ("blk.2.attn_v_b.weight", r"incomplete kv_b groups \[\(2, \['k'\]\)\]"),
    ],
)
def test_iter_gguf_weights_incomplete_fusion_group_fails_at_end_of_stream(tmp_path, drop, match):
    # The implementation buffers fusion pieces and checks completeness when the
    # generator is exhausted: a ValueError listing the layer and the slots that
    # arrived (a bare assert would vanish under python -O and yield a partial set).
    ts = _iter_tensor_set()
    del ts[drop]
    path = _write_iter_gguf(tmp_path / "incomplete.gguf", tensors=ts)
    with pytest.raises(ValueError, match=match):
        _iter_params(path)


def test_iter_gguf_weights_kda_piece_on_a_dsa_layer_fails_at_end_of_stream(tmp_path):
    # _IN_PROJ_SLOTS is consulted before the layer-class tables, so a KDA-only
    # suffix on a DSA layer buffers silently and can only fail at end of stream.
    ts = _iter_tensor_set()
    ts["blk.2.attn_q.weight"] = ts["blk.0.attn_q.weight"]
    path = _write_iter_gguf(tmp_path / "kda-on-dsa.gguf", tensors=ts)
    with pytest.raises(ValueError, match=r"incomplete in_proj groups \[\(2, \['q'\]\)\]"):
        _iter_params(path)


def test_iter_gguf_weights_mixed_row_bytes_in_proj_rejected(tmp_path):
    # F8 guard: the packed concat needs uniform row_bytes; an F16 ssm_beta among
    # Q8_0 pieces must fail loudly, not inside torch.cat with a tensor-less error.
    ts = _iter_tensor_set()
    ts["blk.1.ssm_beta.weight"] = (np.full((_NH, _H), 0.5, np.float16), None)
    path = _write_iter_gguf(tmp_path / "mixed-rb.gguf", tensors=ts)
    with pytest.raises(ValueError, match=r"layer 1 in_proj pieces mix row_bytes"):
        _iter_params(path)


def test_iter_gguf_expert_sources_yields_moe_banks_verbatim(glm5next_iter_gguf):
    import gguf as gguf_mod
    from freetoken.models.glm5_next import iter_gguf_expert_sources

    path = glm5next_iter_gguf
    entries = list(iter_gguf_expert_sources(path, _parse(path)))
    # one entry per MoE trunk layer, in stream order; the dense layer has no banks
    # and blk.3 (MTP) is outside the trunk
    assert [layer for layer, _ in entries] == list(_MOE_ITER_LAYERS)
    for layer, slots in entries:
        assert set(slots) == {"gate", "up", "down"}
        gate, up, down = slots["gate"], slots["up"], slots["down"]
        assert gate.shape == (_NE, _EFF, _H) and up.shape == (_NE, _EFF, _H)
        assert down.shape == (_NE, _H, _EFF)
        # ggml type rides untouched: gate/up Q8_0, down Q6_K superblocks
        assert gate.ggml_type == int(gguf_mod.GGMLQuantizationType.Q8_0)
        assert down.ggml_type == int(gguf_mod.GGMLQuantizationType.Q6_K)
        # packed() is the stacked bank: rows run expert-major within the bank, so a
        # dim-0 slice IS one expert's contiguous packed rows (no repack, no dequant).
        assert gate.packed().shape == (_NE * _EFF, _H // 32 * 34)
        assert down.packed().shape == (_H * _NE, 210)
        gate_bytes = gate.packed()
        up_bytes = up.packed()
        for e in range(_NE):
            want_gate = torch.from_numpy(_pack_q8_0(_bank_vals(0)[e:e + 1])).reshape(_EFF, -1)
            assert torch.equal(gate_bytes[e * _EFF:(e + 1) * _EFF], want_gate), (layer, e)
            want_up = torch.from_numpy(_pack_q8_0(_bank_vals(7)[e:e + 1])).reshape(_EFF, -1)
            assert torch.equal(up_bytes[e * _EFF:(e + 1) * _EFF], want_up), (layer, e)
        assert torch.equal(
            down.packed(), torch.from_numpy(_DOWN_BANK_BYTES.reshape(_H * _NE, 210))
        )


def test_iter_gguf_expert_sources_incomplete_bank_group_fails(tmp_path):
    ts = _iter_tensor_set()
    del ts["blk.2.ffn_down_exps.weight"]
    path = _write_iter_gguf(tmp_path / "bankless.gguf", tensors=ts)
    from freetoken.models.glm5_next import iter_gguf_expert_sources

    with pytest.raises(ValueError, match=r"incomplete expert bank groups \[\(2, \['gate', 'up'\]\)\]"):
        list(iter_gguf_expert_sources(path, _parse(path)))


def test_iter_gguf_expert_sources_reject_bank_on_dense_layer(tmp_path):
    # a bank under leading_dense_block_count is a corrupt file; dropping it silently
    # would desync the bank count the offload cache allocates.
    ts = _iter_tensor_set()
    ts["blk.0.ffn_down_exps.weight"] = ts["blk.1.ffn_down_exps.weight"]
    path = _write_iter_gguf(tmp_path / "dense-bank.gguf", tensors=ts)
    from freetoken.models.glm5_next import iter_gguf_expert_sources

    with pytest.raises(ValueError, match=r"ffn_down_exps.weight on dense layer 0"):
        list(iter_gguf_expert_sources(path, _parse(path)))


def test_iter_gguf_weights_rejects_tp_gt_1(glm5next_iter_gguf, monkeypatch):
    # the loader emits full fused tensors; TP sharding is not implemented, same
    # status as the HF reader (weight.py raises the same NotImplementedError).
    from freetoken.models.glm5_next import iter_gguf_weights

    class _TwoRank:
        size = 2

        def is_primary(self):
            return True

    monkeypatch.setattr("freetoken.distributed.get_tp_info", lambda: _TwoRank())
    with pytest.raises(NotImplementedError, match="supports TP=1 only"):
        list(
            iter_gguf_weights(
                glm5next_iter_gguf, None, include_moe_experts=False, include_non_moe=True
            )
        )


def test_convert_swaps_packed_modules_and_keeps_dense_fusions(tmp_path):
    # Phase 4 entry criterion: parse_gguf_config sets the gguf marker and the convert
    # swaps exactly the .qweight-yielded params for GGUF-quant ops, mirroring gemma4's
    # is_gguf_model/convert pair. Fusion outputs (kv_b_proj, conv1d) and the bf16/fp32
    # rows stay on the normal path. The fixture's F16 head + 64-dim hidden are
    # iterator-only conveniences -- the converted head follows the real file's Q6_K
    # output convention, which needs a 256-multiple hidden, hence the metadata
    # override below.
    import gguf as gguf_mod

    from freetoken.layers.gguf import GGUFEmbedding, GGUFLinear
    from freetoken.models.gguf.dequant import row_bytes as _rb
    from freetoken.models.glm5_next.gguf import GGUFLMHead
    from freetoken.models.glm5_next.model import Glm5NextForCausalLM

    hidden = 256  # Q6_K tables pack 256-wide blocks; the real file's 4096 qualifies
    meta = {**_ITER_METADATA, "glm5next.embedding_length": hidden}
    path = _write_iter_gguf(tmp_path / "convert.gguf", metadata=meta)
    config = _parse(path)
    assert config.moe_weight_format == "gguf"
    # construction alone converts: the __init__ hook sees the gguf marker
    model = Glm5NextForCausalLM(config)

    assert isinstance(model.model.embed_tokens, GGUFEmbedding)
    assert isinstance(model.lm_head, GGUFLMHead)
    assert model.lm_head.qweight.shape == (_VOCAB_S, _rb(hidden, gguf_mod.GGMLQuantizationType.Q6_K))
    l0, l1, l2 = model.model.layers.op_list[0], model.model.layers.op_list[1], model.model.layers.op_list[2]
    # KDA layer 0: the four packed linears + the dense mlp swap
    for owner, attr in (
        (l0.self_attn, "in_proj"),
        (l0.self_attn, "f_b_proj"),
        (l0.self_attn, "g_b_proj"),
        (l0.self_attn, "o_proj"),
        (l0.mlp, "gate_proj"),
        (l0.mlp, "up_proj"),
        (l0.mlp, "down_proj"),
    ):
        assert isinstance(getattr(owner, attr), GGUFLinear), attr
    assert l0.self_attn.in_proj.qweight.shape == (3 * _PROJ + _NH + 2 * _HD, _rb(hidden, gguf_mod.GGMLQuantizationType.Q8_0))
    # DSA layer 2: the four packed linears swap; kv_b_proj stays dense
    for attr in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj"):
        assert isinstance(getattr(l2.self_attn, attr), GGUFLinear), attr
    assert not isinstance(l2.self_attn.kv_b_proj, GGUFLinear)
    assert isinstance(l2.mlp.shared_experts.gate_proj, GGUFLinear)
    assert not isinstance(l2.mlp.gate, GGUFLinear)  # the router stays dense
    # the loader-visible state dict exposes packed keys; dense keys unchanged
    sd = model.state_dict()
    assert "lm_head.qweight" in sd and "lm_head.weight" not in sd
    assert "model.layers.1.self_attn.in_proj.qweight" in sd
    assert "model.layers.2.self_attn.kv_b_proj.weight" in sd
    assert "model.layers.0.self_attn.A_log" in sd
    # the convert must leave every non-.qweight param exactly as built
    from freetoken.layers import LinearReplicated

    assert not isinstance(l0.self_attn.conv1d, GGUFLinear)
    assert isinstance(l0.self_attn.conv1d.weight, torch.Tensor)
    assert l0.self_attn.conv1d.weight.shape == (3 * _PROJ, 1, _CONV)
    assert not isinstance(l2.self_attn.indexer.wk, GGUFLinear)
    assert isinstance(l2.self_attn.indexer.wk, LinearReplicated)
    assert l0.hc_attn_fn.dtype == torch.float32
    assert not isinstance(l1.mlp.gate, GGUFLinear)
    assert isinstance(l1.mlp.gate, LinearReplicated)
    assert l1.mlp.e_score_correction_bias.dtype == torch.float32


def test_convert_rejects_tied_gguf(tmp_path):
    # a tied variant (no output.weight) would die late on a bare lm_head.qweight
    # KeyError -- the config guard refuses it up front instead.
    ts = _iter_tensor_set()
    del ts["output.weight"]
    path = _write_iter_gguf(tmp_path / "tied.gguf", tensors=ts)
    config = _parse(path)
    assert config.tie_word_embeddings is True
    from freetoken.models.glm5_next.model import Glm5NextForCausalLM

    with pytest.raises(ValueError, match="requires the untied file"):
        Glm5NextForCausalLM(config)


@pytest.mark.parametrize(
    ("tensor_name", "payload", "raw_dtype", "match"),
    [
    (
        "token_embd.weight",
        np.zeros((_VOCAB_S, 144), np.uint8),  # ne0 = 144/18*32 = hidden
        gguf.GGMLQuantizationType.Q4_0,
        r"token_embd\.weight is Q4_0; the convert swap builds the embedding as Q8_0",
    ),
    (
        "output.weight",
        np.zeros((_VOCAB_S, 272), np.uint8),  # ne0 = 272/34*32 = hidden
        gguf.GGMLQuantizationType.Q8_0,
        r"output\.weight is Q8_0; the convert swap builds the untied head as Q6_K",
    ),
    (
        "blk.0.ffn_gate.weight",
        np.zeros((_DFF, 144), np.uint8),
        gguf.GGMLQuantizationType.Q4_0,
        r"blk\.0\.ffn_gate\.weight is Q4_0; the convert swap builds packed linears as Q8_0",
    ),
    ],
)
def test_iter_gguf_weights_rejects_wrong_dense_type(
    tmp_path, tensor_name, payload, raw_dtype, match
):
    # C1: the convert hardcodes a per-target packed type (embed Q8_0, untied head
    # Q6_K, linears Q8_0); a variant file deviating on any dense tensor must fail at
    # ITERATION time naming the tensor and the expected type, never as a late
    # loader shape assert. Payload byte widths keep ne0 == the fixture hidden.
    ts = _iter_tensor_set()
    ts[tensor_name] = (payload, raw_dtype)
    path = _write_iter_gguf(tmp_path / "wrong-dense-type.gguf", tensors=ts)
    with pytest.raises(ValueError, match=match):
        _iter_params(path)


def test_dequant_q8_0_reference_matches_handcrafted_blocks():
    from freetoken.models.gguf.dequant import GGML_Q8_0, dequant_q8_0, dequantize

    # handcrafted block: fp16 scale 0.5 + 32 int8 quants -> w = d * q
    q = np.array([127, -128, 1, -1, 0, 42, -42, 7] + [0] * 24, dtype=np.int8)
    raw = np.concatenate([np.array([0.5], np.float16).view(np.uint8), q.view(np.uint8)])
    out = dequant_q8_0(torch.from_numpy(raw.copy()), torch.float32)
    torch.testing.assert_close(out, torch.from_numpy(q.astype(np.float32) * np.float32(0.5)))

    # multi-block: expected values recomputed by an independent numpy decode
    rng = np.random.default_rng(11)
    n = 7
    scales = rng.uniform(-3, 3, n).astype(np.float16)
    quants = rng.integers(-128, 128, (n, 32)).astype(np.int8)
    raw = np.zeros((n, 34), np.uint8)
    raw[:, 0:2] = scales.view(np.uint8).reshape(n, 2)
    raw[:, 2:34] = quants.view(np.uint8)
    ref = quants.astype(np.float32) * scales.astype(np.float32).reshape(n, 1)
    out = dequant_q8_0(torch.from_numpy(raw), torch.float32).reshape(n, 32)
    torch.testing.assert_close(out, torch.from_numpy(ref))
    # dequantize dispatch + bf16 cast agree with the fp32 reference rounded once
    bf = dequantize(torch.from_numpy(raw.reshape(-1)), GGML_Q8_0, torch.bfloat16).reshape(n, 32)
    assert torch.equal(bf, torch.from_numpy(ref).to(torch.bfloat16))


# =====================================================================================
# Phase 3: embedded tokenizer (tokenizer.ggml.* KV -> PreTrainedTokenizerFast)
# =====================================================================================


def _write_tokenizer_gguf(
    tmp_path,
    *,
    tokens,
    merges,
    special_ids,
    chat_template=None,
    token_type=None,
    arch="glm5next",
    scores=None,
):
    import gguf as gguf_mod

    path = tmp_path / f"tokenizer-{arch}.gguf"
    w = gguf_mod.GGUFWriter(str(path), arch)
    w.add_array("tokenizer.ggml.tokens", list(tokens))
    w.add_array("tokenizer.ggml.token_type", token_type or [1] * len(tokens))
    w.add_array("tokenizer.ggml.merges", list(merges))
    if scores is not None:
        w.add_array("tokenizer.ggml.scores", list(scores))
    w.add_string("tokenizer.ggml.model", "gpt2")
    w.add_string("tokenizer.ggml.pre", "glm4")
    for key, tid in special_ids.items():
        w.add_uint32(f"tokenizer.ggml.{key}_token_id", tid)
    if chat_template is not None:
        w.add_string("tokenizer.chat_template", chat_template)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.close()
    return str(path)


def test_load_tokenizer_from_gguf_metadata(tmp_path):
    # Byte-level alphabet + one merge whose result is in the vocab: encode() splits
    # to single byte tokens (the merge never applies to the sample text) and
    # decode() is lossless, so the round trip exercises the converter path without
    # depending on merge ranks. gguf-py drops empty arrays on write, so the merge
    # list must be non-empty.
    from transformers import PreTrainedTokenizerFast
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    from freetoken.utils.hf import load_eos_token_ids, load_tokenizer

    byte_chars = list(bytes_to_unicode().values())
    tokens = byte_chars + ["!!", "<eos>", "<pad>", "[gMASK]"]
    eos_id, pad_id, bos_id, unk_id = 257, 258, 259, 0
    template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    path = _write_tokenizer_gguf(
        tmp_path,
        tokens=tokens,
        # gguf-py drops empty arrays on write; a harmless real merge keeps the KV
        merges=["! !"],
        special_ids={"eos": eos_id, "padding": pad_id, "bos": bos_id, "unknown": unk_id},
        chat_template=template,
    )

    tok = load_tokenizer(path)
    assert isinstance(tok, PreTrainedTokenizerFast)
    assert (tok.bos_token_id, tok.eos_token_id, tok.pad_token_id, tok.unk_token_id) == (
        bos_id,
        eos_id,
        pad_id,
        unk_id,
    )
    assert tok.bos_token == "[gMASK]" and tok.eos_token == "<eos>"
    assert tok.chat_template == template

    text = "hello world! 42"
    ids = tok.encode(text)
    assert all(0 <= i < len(tokens) for i in ids)
    assert tok.decode(ids) == text

    # eos-ids helper: the formal eos plus the vocab <eos>; no <turn|> in this vocab,
    # so the gemma4 turn-end branch is a no-op for glm5next.
    assert load_eos_token_ids(path, tok) == {eos_id}


def test_glm5next_gguf_tokenizer_digit_split_and_gemma4_gate(tmp_path):
    """No-reference pin: glm5next gets the glm4 pre-tokenizer, gemma4 does not.

    GGUFGPTConverter attaches the GPT-2 ByteLevel regex, which glues the leading
    space onto digit runs (" in 100 days" pre-tokenizes as "Ġ100") - a pre-token
    shape the glm4-trained vocab has no merges for, so numerals in user input
    byte-split at encode. For glm5next the converter pre_tokenizer is replaced
    with the reference glm4 scheme; the expected pre-splits below are the
    reference tokenizer's own pieces (verified against its tokenizer.json).
    """
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    from freetoken.utils.hf import load_tokenizer

    byte_chars = list(bytes_to_unicode().values())
    tokens = byte_chars + ["!!", "<eos>", "<pad>", "[gMASK]"]
    tok = load_tokenizer(
        _write_tokenizer_gguf(
            tmp_path,
            tokens=tokens,
            merges=["! !"],
            special_ids={"eos": 258, "padding": 259, "bos": 260, "unknown": 0},
        )
    )
    pt = tok.backend_tokenizer.pre_tokenizer
    # the reference glm4 Sequence (Split Isolated + ByteLevel), not the GPT-2 regex
    assert repr(pt).startswith("Sequence") and "Isolated" in repr(pt)
    # digit runs split into 1-3 char pre-tokens, no leading space glued to digits
    assert pt.pre_tokenize_str("in 100 days") == [
        ("in", (0, 2)),
        ("Ġ", (2, 3)),
        ("100", (3, 6)),
        ("Ġdays", (6, 11)),
    ]
    assert pt.pre_tokenize_str("2^10") == [
        ("2", (0, 1)),
        ("^", (1, 2)),
        ("10", (2, 4)),
    ]
    assert pt.pre_tokenize_str("2+2") == [
        ("2", (0, 1)),
        ("+", (1, 2)),
        ("2", (2, 3)),
    ]
    assert pt.pre_tokenize_str("12345") == [("123", (0, 3)), ("45", (3, 5))]
    assert tok.decode(tok.encode("in 100 days")) == "in 100 days"

    # gemma4: converter default preserved - no Isolated glm4 split, digits keep
    # the GPT-2 shape (a standalone digit-run piece must not appear)
    gtok = load_tokenizer(
        _write_tokenizer_gguf(
            tmp_path,
            arch="gemma4",
            tokens=["▁hello", "▁world", "h", "i", "2", "+", "<eos>"],
            merges=["a b"],
            scores=[0.0] * 7,
            special_ids={"eos": 6},
        )
    )
    gpt = gtok.backend_tokenizer.pre_tokenizer
    assert "Isolated" not in repr(gpt)
    assert ("100", (3, 6)) not in gpt.pre_tokenize_str("in 100 days")


_GLM5NEXT_TOKENIZER_REF = os.environ.get("FREETOKEN_GLM5NEXT_TOKENIZER_REF", "")
needs_ref_tokenizer = pytest.mark.skipif(
    not (_GLM5NEXT_TOKENIZER_REF and os.path.isdir(_GLM5NEXT_TOKENIZER_REF)),
    reason=(
        "FREETOKEN_GLM5NEXT_TOKENIZER_REF not set to the RedHatAI GLM-5.3-Flash dir "
        "(with tokenizer.json); set it to round-trip against the reference tokenizer"
    ),
)


@needs_ref_tokenizer
def test_glm5next_gguf_tokenizer_matches_reference(tmp_path):
    """Round-trip the gguf-embedded tokenizer against the RedHatAI reference.

    The synthetic gguf carries the REFERENCE's own flat vocab (model vocab + added
    tokens, in id order) and its full merge list, so any tokenization difference
    isolates the converter path rather than vocab drift. Known special-token
    handling: the serve path re-encodes rendered chat text, so the converted
    tokenizer registers the gguf special tokens (token_type walk) as atomic
    AddedTokens - asserted below via the render->encode check. The converter path
    now ports the reference glm4 pre_tokenizer (was: the GPT-2 ByteLevel regex,
    which mangled numerals in user input); digit and code samples that used to be
    pinned divergences assert identity below, and the boundary split is reported
    via warnings.warn, not masked.
    """
    import json

    from transformers import AutoTokenizer

    from freetoken.utils.hf import load_eos_token_ids, load_tokenizer

    ref_dir = _GLM5NEXT_TOKENIZER_REF
    ref = AutoTokenizer.from_pretrained(ref_dir)
    with open(os.path.join(ref_dir, "tokenizer.json"), encoding="utf-8") as f:
        tj = json.load(f)

    # flat vocab in id order: model vocab + added tokens (the gguf convention)
    flat = dict(tj["model"]["vocab"])
    for t in tj["added_tokens"]:
        flat[t["content"]] = t["id"]
    size = max(flat.values()) + 1
    tokens = ["<|gguf_pad|>"] * size
    for tok_, i in flat.items():
        tokens[i] = tok_
    assert all(t != "<|gguf_pad|>" for t in tokens)

    added = {t["content"]: t["id"] for t in tj["added_tokens"]}
    eos_id = added["<|endoftext|>"]
    bos_id = added["[gMASK]"]
    merges = [f"{a} {b}" for a, b in tj["model"]["merges"]]
    # F1: the gguf's token_type array marks the special tokens CONTROL - the
    # converter registers them as atomic AddedTokens on the fast tokenizer.
    token_types = [1] * len(tokens)
    for tok_id in added.values():
        token_types[tok_id] = 3  # GGMLTokenTyPE.CONTROL
    path = _write_tokenizer_gguf(
        tmp_path,
        tokens=tokens,
        merges=merges,
        special_ids={
            "eos": eos_id,
            "bos": bos_id,
            "padding": eos_id,
            "unknown": eos_id,
            # the real file declares eom/eot ids; the eos helper unions them (F2)
            "eom": added["<|observation|>"],
            "eot": added["<|user|>"],
        },
        token_type=token_types,
        chat_template=ref.chat_template,
    )

    tok = load_tokenizer(path)
    assert tok.eos_token_id == eos_id and tok.bos_token_id == bos_id
    # F2: glm5next ends turns with <eot>/<eom> - the helper unions the gguf-declared
    # eom/eot ids with eos (reference stop set {154820, 154827, 154829}).
    assert load_eos_token_ids(path, tok) == {154820, 154827, 154829}

    # Prose samples MUST match the reference exactly (verified: en/ru/cjk identical).
    prose = [
        "The quick brown fox jumps over the lazy dog.",
        "Съешь ещё этих мягких французских булок, да выпей чаю.",
        "你好，世界！GLM-5.3-Flash 是一个混合专家模型。",
    ]
    # Digit survival: pre-fix the converter kept the GPT-2 ByteLevel regex and the
    # glm4-trained vocab had no merges for its "Ġ100"-style digit pre-tokens, so
    # numerals in user input byte-split into garbage ("in 100 days" -> "in <blank>
    # days", "2^10" -> "<image?> ^10"). With the ported glm4 pre_tokenizer these
    # must match the reference exactly (no unk, lossless).
    digits = [
        "in 100 days",
        "2^10",
        "2+2",
        "What is 2+2?",
        "for i in range(10):\n    x[i] = i * 2  # 12345",
    ]
    # reference-shape pins (stable constants of the released reference tokenizer)
    assert ref.tokenize("in 100 days") == ["in", "Ġ", "100", "Ġdays"]
    assert ref.tokenize("2+2") == ["2", "+", "2"]

    for s in prose + digits:
        ref_toks = ref.tokenize(s)
        ids = tok.encode(s)
        got_toks = tok.convert_ids_to_tokens(ids)
        assert got_toks == ref_toks, (
            f"sample {s[:20]!r} diverges from the reference: "
            f"gguf={got_toks[:10]} ref={ref_toks[:10]}"
        )
        assert tok.decode(ids) == s
        assert tok.unk_token_id not in ids

    code = "def f(x):\n    return x + 12345  # comment"
    ref_code = ref.tokenize(code)
    # FLIPPED PIN (was a known divergence): pre-fix the gguf split '(x' as '(' + 'x'
    # and kept the newline separate from '):'; the glm4 pre_tokenizer port makes
    # the gguf side identical to the reference ('(x' and '):Ċ' grouped). The
    # reference-side piece pin stays as the drift guard.
    assert ref_code[:4] == ["def", "Ġf", "(x", "):Ċ"], ref_code[:10]
    assert tok.convert_ids_to_tokens(tok.encode(code)) == ref_code
    assert tok.decode(tok.encode(code)) == code  # lossless

    # F3: the serve path renders the chat template to TEXT and re-encodes it
    # (apply_chat_template(tokenize=False) -> encode(..., add_special_tokens=False));
    # every special token the template emits must map to its single registered id
    # (no byte-splitting) - pinned by the token_type registration in tokenizer.py.
    messages = [{"role": "user", "content": "hi"}]
    rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok.encode(rendered, add_special_tokens=False)
    rendered_specials = [t for t in sorted(added, key=lambda k: added[k]) if t and t in rendered]
    assert rendered_specials, f"no special tokens in rendered template: {rendered[:120]!r}"
    for st in rendered_specials:
        assert added[st] in enc, (
            f"special token {st!r} byte-split across the rendered template encode"
        )

    # special-token boundary: reported, not masked -- the gguf converter has no
    # added_tokens machinery, so the literal special string byte-splits.
    s = "<|user|>hello"
    got_toks = tok.convert_ids_to_tokens(tok.encode(s))
    boundary_id = added["<|user|>"]
    if boundary_id not in tok.encode(s):
        warnings.warn(
            f"glm5next gguf tokenizer boundary divergence: {s!r} encodes as "
            f"{got_toks[:8]!r}; the special id {boundary_id} is only reachable "
            "via explicit ids (the gguf converter has no added_tokens machinery)",
            stacklevel=1,
        )
    assert tok.decode(tok.encode(s)) == s  # still lossless


# ------------------------------------------------------------------------------
# --mmproj: clip-architecture vision file -> multimodal GGUF spec + VisionConfig
# ------------------------------------------------------------------------------

# clip.* KVs mirror the real mmproj-BF16.gguf metadata (the llama.cpp spellings);
# the fields FreeToken's VisionConfig consumes are exactly the mapped nine.
_CLIP_METADATA: dict = {
    "clip.has_vision_encoder": True,
    "clip.projector_type": "glm5next",
    "clip.use_silu": True,
    "clip.vision.attention.head_count": 16,
    "clip.vision.attention.layer_norm_epsilon": 1e-05,
    "clip.vision.block_count": 24,
    "clip.vision.embedding_length": 1024,
    "clip.vision.feed_forward_length": 4096,
    "clip.vision.image_size": 448,
    "clip.vision.patch_size": 14,
    "clip.vision.projection_dim": 4096,
    "clip.vision.spatial_merge_size": 2,
    "clip.vision.swiglu_limit": 10.0,
    "clip.vision.image_mean": [0.48145467, 0.45782751, 0.40821072],
    "clip.vision.image_std": [0.26862955, 0.26130259, 0.27577710],
    "general.name": "Glm-5.3-Flash",
    "general.type": "mmproj",
}

# Header-shape stubs: the config path reads tensor NAMES and SHAPES only, never the
# weight data, so the derived-field checks run against tiny buffers carrying the
# load-bearing axes (patch-embed input channels at shape[1], the temporal slice
# count, the qkv bias presence). Contents are unused.
_MMPROJ_TENSORS: dict = {
    # two temporal slices of the Conv3d patch embedding (numpy shape == torch order)
    "v.patch_embd.weight": (np.zeros((2, 3, 4, 4), dtype=np.float32), gguf.GGMLQuantizationType.F32),
    "v.patch_embd.weight.1": (np.zeros((2, 3, 4, 4), dtype=np.float32), gguf.GGMLQuantizationType.F32),
    # attention_bias=True fact: the tower's qkv carries a bias tensor
    "v.blk.0.attn_qkv.bias": (np.zeros(3072, dtype=np.float32), gguf.GGMLQuantizationType.F32),
}


def _write_mmproj_gguf(path, *, metadata=None, tensors=None, arch="clip") -> str:
    w = gguf.GGUFWriter(str(path), arch)
    meta = _CLIP_METADATA if metadata is None else metadata
    for key, val in sorted(meta.items()):
        if isinstance(val, bool):
            w.add_bool(key, val)
        elif isinstance(val, list):
            w.add_array(key, val)
        elif isinstance(val, str):
            w.add_string(key, val)
        elif isinstance(val, float):
            w.add_float32(key, val)
        else:
            w.add_int32(key, val)
    entries = _MMPROJ_TENSORS if tensors is None else tensors
    for name, (arr, qt) in entries.items():
        w.add_tensor(name, arr, raw_dtype=qt)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)


@pytest.fixture(scope="session")
def mmproj_gguf(tmp_path_factory) -> str:
    return _write_mmproj_gguf(tmp_path_factory.mktemp("glm5next-mmproj") / "mmproj.gguf")


def test_mmproj_builds_multimodal_spec_and_full_vision_config(glm5next_gguf, mmproj_gguf):
    from freetoken.models.gguf.config import build_gguf_shim
    from freetoken.utils import cached_load_hf_config

    shim = build_gguf_shim(glm5next_gguf, mmproj_path=mmproj_gguf)
    assert shim.architectures == ["Glm5NextGGUFForConditionalGeneration"]
    assert shim.mmproj_path == mmproj_gguf
    assert shim.vision_config is not None
    # the flag travels through the same entry point the engine uses
    assert cached_load_hf_config(
        glm5next_gguf, mmproj_path=mmproj_gguf
    ).architectures == ["Glm5NextGGUFForConditionalGeneration"]

    cfg = _parse_shim(shim)
    vc = cfg.vision_config
    # all 13 fields, values pinned to the NVFP4 reference vision_config
    assert (vc.depth, vc.hidden_size, vc.num_heads) == (24, 1024, 16)
    assert (vc.intermediate_size, vc.out_hidden_size) == (4096, 4096)
    assert (vc.patch_size, vc.spatial_merge_size) == (14, 2)
    assert (vc.temporal_patch_size, vc.in_channels) == (2, 3)
    assert vc.projection_intermediate_size == 10240
    assert vc.rms_norm_eps == pytest.approx(1e-5, rel=1e-5)
    assert vc.swiglu_limit == pytest.approx(10.0)
    assert vc.attention_bias is True
    assert cfg.is_multimodal


def _parse_shim(shim) -> "ModelConfig":
    from freetoken.models.glm5_next.gguf import parse_gguf_config

    return parse_gguf_config(shim)


def test_gguf_without_mmproj_registry_and_config_unchanged(glm5next_gguf):
    """No flag: the text-only spec, shim fields and config are byte-identical to the pre-mmproj path."""
    from freetoken.models.gguf.config import build_gguf_shim

    shim = build_gguf_shim(glm5next_gguf)
    assert shim.architectures == ["Glm5NextGGUFForCausalLM"]
    assert shim.mmproj_path is None
    assert shim.vision_config is None
    cfg = _parse(glm5next_gguf)
    assert cfg.vision_config is None
    assert not cfg.is_multimodal


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("clip.vision.block_count", 12),
        ("clip.vision.embedding_length", 768),
        ("clip.vision.attention.head_count", 8),
        ("clip.vision.feed_forward_length", 2048),
        ("clip.vision.projection_dim", 2048),
        ("clip.vision.patch_size", 16),
        ("clip.vision.spatial_merge_size", 1),
        ("clip.vision.swiglu_limit", 7.0),
    ],
)
def test_mmproj_pinned_reference_mismatch_fails_fast(glm5next_gguf, tmp_path, field, value):
    from freetoken.models.gguf.config import build_gguf_shim

    meta = dict(_CLIP_METADATA, **{field: value})
    p = _write_mmproj_gguf(tmp_path / "pinned.gguf", metadata=meta)
    with pytest.raises(ValueError, match="reference"):
        build_gguf_shim(glm5next_gguf, mmproj_path=p)


def test_mmproj_missing_file_fails_fast(glm5next_gguf, tmp_path):
    from freetoken.models.gguf.config import build_gguf_shim

    with pytest.raises(ValueError, match="not found"):
        build_gguf_shim(glm5next_gguf, mmproj_path=str(tmp_path / "absent.gguf"))


def test_mmproj_non_clip_arch_fails_fast(glm5next_gguf, tmp_path):
    from freetoken.models.gguf.config import build_gguf_shim

    p = _write_mmproj_gguf(tmp_path / "wrong.gguf", arch="glm5next")
    with pytest.raises(ValueError, match="'clip'"):
        build_gguf_shim(glm5next_gguf, mmproj_path=p)


def test_mmproj_missing_clip_key_fails_fast(glm5next_gguf, tmp_path):
    from freetoken.models.gguf.config import build_gguf_shim

    meta = {k: v for k, v in _CLIP_METADATA.items() if k != "clip.vision.block_count"}
    p = _write_mmproj_gguf(tmp_path / "incomplete.gguf", metadata=meta)
    with pytest.raises(ValueError, match="clip.vision.block_count"):
        build_gguf_shim(glm5next_gguf, mmproj_path=p)


def test_mmproj_derived_field_gate_shape_mismatch_fails_fast(glm5next_gguf, tmp_path):
    from freetoken.models.gguf.config import build_gguf_shim

    # mm.gate.weight at a wrong output width: the pinned projection_intermediate_size
    # check reads the tensor header shape even though the metadata never carries it
    tensors = dict(_MMPROJ_TENSORS)
    tensors["mm.gate.weight"] = (np.zeros((16, 8), dtype=np.float16), gguf.GGMLQuantizationType.F16)
    p = _write_mmproj_gguf(tmp_path / "gate.gguf", tensors=tensors)
    with pytest.raises(ValueError, match="projection_intermediate_size"):
        build_gguf_shim(glm5next_gguf, mmproj_path=p)


def test_mmproj_temporal_slice_count_mismatch_fails_fast(glm5next_gguf, tmp_path):
    from freetoken.models.gguf.config import build_gguf_shim

    # a single patch-embed slice: temporal_patch_size is pinned to 2 (two Conv3d slices)
    tensors = {
        "v.patch_embd.weight": (np.zeros((2, 3, 4, 4), dtype=np.float32), gguf.GGMLQuantizationType.F32),
        "v.blk.0.attn_qkv.bias": (np.zeros(3072, dtype=np.float32), gguf.GGMLQuantizationType.F32),
    }
    p = _write_mmproj_gguf(tmp_path / "temporal.gguf", tensors=tensors)
    with pytest.raises(ValueError, match="temporal_patch_size"):
        build_gguf_shim(glm5next_gguf, mmproj_path=p)


def test_mmproj_missing_qkv_bias_fails_fast(glm5next_gguf, tmp_path):
    from freetoken.models.gguf.config import build_gguf_shim

    tensors = {
        k: v for k, v in _MMPROJ_TENSORS.items() if k != "v.blk.0.attn_qkv.bias"
    }
    p = _write_mmproj_gguf(tmp_path / "nobias.gguf", tensors=tensors)
    with pytest.raises(ValueError, match="attention_bias"):
        build_gguf_shim(glm5next_gguf, mmproj_path=p)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("clip.has_vision_encoder", False, "has_vision_encoder"),
        ("clip.projector_type", "glm4v", "projector_type"),
    ],
)
def test_mmproj_tower_facts_mismatch_fails_fast(glm5next_gguf, tmp_path, field, value, match):
    from freetoken.models.gguf.config import build_gguf_shim

    meta = dict(_CLIP_METADATA, **{field: value})
    p = _write_mmproj_gguf(tmp_path / "facts.gguf", metadata=meta)
    with pytest.raises(ValueError, match=match):
        build_gguf_shim(glm5next_gguf, mmproj_path=p)


@pytest.mark.parametrize("field", ["clip.has_vision_encoder", "clip.projector_type"])
def test_mmproj_missing_tower_fact_fails_fast(glm5next_gguf, tmp_path, field):
    from freetoken.models.gguf.config import build_gguf_shim

    meta = {k: v for k, v in _CLIP_METADATA.items() if k != field}
    p = _write_mmproj_gguf(tmp_path / "nofact.gguf", metadata=meta)
    with pytest.raises(ValueError, match=field.split(".")[1]):
        build_gguf_shim(glm5next_gguf, mmproj_path=p)


def test_mmproj_patch_slices_without_base_tensor_fails_fast(glm5next_gguf, tmp_path):
    """Slices .N present but the base v.patch_embd.weight absent: a named error, not a KeyError."""
    from freetoken.models.gguf.config import build_gguf_shim

    tensors = {
        "v.patch_embd.weight.1": (np.zeros((2, 3, 4, 4), dtype=np.float32), gguf.GGMLQuantizationType.F32),
        "v.blk.0.attn_qkv.bias": (np.zeros(3072, dtype=np.float32), gguf.GGMLQuantizationType.F32),
    }
    p = _write_mmproj_gguf(tmp_path / "noslice.gguf", tensors=tensors)
    with pytest.raises(ValueError, match="base"):
        build_gguf_shim(glm5next_gguf, mmproj_path=p)


def test_engine_config_mmproj_spec_resolves_and_config_builds(glm5next_gguf, mmproj_gguf):
    """Phase 2 lifted the boot gate: the mmproj boot resolves the vision spec and the
    config path builds the multimodal ModelConfig (the tower itself streams from the
    mmproj file through load_weight during boot)."""
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig
    from freetoken.models.register import get_model_spec

    cfg = EngineConfig(
        model_path=glm5next_gguf,
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        mmproj_path=mmproj_gguf,
    )
    assert cfg.hf_config.architectures == ["Glm5NextGGUFForConditionalGeneration"]
    assert cfg.model_spec is get_model_spec("Glm5NextGGUFForConditionalGeneration")
    assert cfg.model_spec.encoders
    assert cfg.model_config.is_multimodal
    assert cfg.model_config.vision_config is not None


def test_engine_config_threads_mmproj_into_hf_config(glm5next_gguf, mmproj_gguf):
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig
    from freetoken.models.register import get_model_spec

    cfg = EngineConfig(
        model_path=glm5next_gguf,
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        mmproj_path=mmproj_gguf,
    )
    assert cfg.hf_config.architectures == ["Glm5NextGGUFForConditionalGeneration"]
    assert get_model_spec("Glm5NextGGUFForConditionalGeneration").encoders


def test_engine_config_disabled_vision_drops_mmproj_section(glm5next_gguf, mmproj_gguf):
    """--mm-disable vision on an mmproj boot: the frozen shim's vision_config section is
    dropped (dataclasses.replace), so the config serves text-only."""
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig
    from freetoken.mm.config import MultimodalConfig

    cfg = EngineConfig(
        model_path=glm5next_gguf,
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        mmproj_path=mmproj_gguf,
        mm=MultimodalConfig(disabled_encoders=frozenset({"vision"})),
    )
    assert cfg.active_encoders == ()
    assert cfg.model_config.vision_config is None


def test_iter_gguf_weights_include_vision_contract(glm5next_iter_gguf, mmproj_gguf, monkeypatch):
    """load_weight passes include_vision once spec.encoders is non-empty (weight.py);
    the reader must accept it, and include_vision=False must never open the mmproj."""
    import inspect

    import freetoken.models.gguf.reader as reader_mod
    from freetoken.models.glm5_next import iter_gguf_weights
    from freetoken.models.register import _load_attr, get_model_spec

    spec = get_model_spec("Glm5NextGGUFForConditionalGeneration")
    assert spec.encoders
    assert "include_vision" in inspect.signature(iter_gguf_weights).parameters

    # spy on the raw gguf reader (every metadata/tensor access funnels through it)
    real_reader = reader_mod._reader.__wrapped__
    opened: list[str] = []

    def spy_reader(path):
        opened.append(str(path))
        return real_reader(path)

    monkeypatch.setattr(reader_mod, "_reader", spy_reader)

    reader = _load_attr(spec.module, spec.iter_weights)
    out = dict(
        reader(
            glm5next_iter_gguf,
            None,
            include_moe_experts=False,
            include_non_moe=True,
            include_vision=False,
        )
    )
    assert out and not any(name.startswith("visual.") for name in out)
    assert mmproj_gguf not in opened  # the mmproj file was never even mmap'd


def test_iter_vision_weights_gguf_branch_reads_mmproj(glm5next_gguf, vision_mmproj_gguf):
    """Phase 2: the dispatcher routes a GGUF checkpoint with an mmproj file to the
    GGUF vision reader (the phase-1 NotImplementedError stub is gone)."""
    from freetoken.models.glm5_next import iter_vision_weights

    out = dict(iter_vision_weights(glm5next_gguf, None, mmproj_path=vision_mmproj_gguf))
    assert out and all(name.startswith("visual.") for name in out)
    assert "visual.blocks.0.attn.qkv.weight" in out


def test_iter_vision_weights_mmproj_requires_gguf(tmp_path):
    from freetoken.models.glm5_next import iter_vision_weights

    with pytest.raises(ValueError, match="GGUF checkpoint"):
        list(iter_vision_weights(str(tmp_path), None, mmproj_path="mmproj.gguf"))


def test_iter_vision_weights_gguf_without_mmproj_fails_fast(glm5next_gguf):
    from freetoken.models.glm5_next import iter_vision_weights

    with pytest.raises(ValueError, match="--mmproj"):
        list(iter_vision_weights(glm5next_gguf, None))


def test_iter_vision_weights_without_mmproj_keeps_hf_reader(tmp_path, monkeypatch):
    """No mmproj on a HF checkpoint dir: the dispatcher falls through to the HF
    safetensors reader exactly as before. A GGUF path instead fails fast (a bare
    .gguf carries no tower), see test_iter_vision_weights_gguf_without_mmproj_fails_fast."""
    import freetoken.models.glm5_next.weight as weight_mod
    from freetoken.models.glm5_next import iter_vision_weights

    def _hf_reader(path):
        raise RuntimeError("HF reader reached")

    monkeypatch.setattr(weight_mod, "download_hf_weight", _hf_reader)
    with pytest.raises(RuntimeError, match="HF reader"):
        list(iter_vision_weights(str(tmp_path / "hf-model"), None))


# ------------------------------------------------------------------------------
# iter_gguf_vision_weights: mmproj tensor mapping (phase 2)
# ------------------------------------------------------------------------------

# reader-fixture dims: hidden, ffn, out_hidden (downsample/merger width),
# projection_intermediate (merger MLP width), patch spatial. Deliberately small
# and NON-square across the three width signatures (hidden / ffn / merger) so a
# transposed or axis-swapped read cannot alias (kb fixture-crafting lesson).
_VH, _VF, _VO, _VP, _VKH, _VKW = 8, 16, 12, 20, 2, 2


def _vision_fixture_data() -> dict[str, tuple[np.ndarray, bool]]:
    """name -> (torch-order float32 values, stored as BF16). Weights are BF16-stored
    exactly like the real mmproj (biases/norms/patch-embedding are F32 there); the
    analytic values distinguish segments, axes and temporal slices."""
    data: dict[str, tuple[np.ndarray, bool]] = {}

    def add(name: str, arr: np.ndarray, bf16: bool) -> None:
        data[name] = (arr, bf16)

    # block 0, all 14 tensors: the fused qkv carries 1/2/3 per q|k|v segment on the
    # OUTPUT axis (rows) and 10/20/30 on the bias - pins the no-transpose read
    qkv = np.zeros((3 * _VH, _VH), dtype=np.float32)
    qkv[:_VH] = 1.0
    qkv[_VH : 2 * _VH] = 2.0
    qkv[2 * _VH :] = 3.0
    add("v.blk.0.attn_qkv.weight", qkv, True)
    qkvb = np.zeros(3 * _VH, dtype=np.float32)
    qkvb[:_VH] = 10.0
    qkvb[_VH : 2 * _VH] = 20.0
    qkvb[2 * _VH :] = 30.0
    add("v.blk.0.attn_qkv.bias", qkvb, False)
    add("v.blk.0.attn_out.weight", np.arange(_VH * _VH, dtype=np.float32).reshape(_VH, _VH), True)
    add("v.blk.0.attn_out.bias", np.arange(_VH, dtype=np.float32), False)
    add("v.blk.0.attn_q_norm.weight", np.arange(4, dtype=np.float32), False)
    add("v.blk.0.attn_k_norm.weight", np.arange(4, dtype=np.float32) + 1, False)
    add("v.blk.0.ln1.weight", np.arange(_VH, dtype=np.float32) + 0.5, False)
    add("v.blk.0.ln2.weight", np.arange(_VH, dtype=np.float32) + 1.5, False)
    add("v.blk.0.ffn_gate.weight", np.arange(_VF * _VH, dtype=np.float32).reshape(_VF, _VH), True)
    add("v.blk.0.ffn_gate.bias", np.arange(_VF, dtype=np.float32) + 1, False)
    add("v.blk.0.ffn_up.weight", np.arange(_VF * _VH, dtype=np.float32).reshape(_VF, _VH) + 100, True)
    add("v.blk.0.ffn_up.bias", np.arange(_VF, dtype=np.float32) + 2, False)
    add("v.blk.0.ffn_down.weight", np.arange(_VH * _VF, dtype=np.float32).reshape(_VH, _VF), True)
    add("v.blk.0.ffn_down.bias", np.arange(_VH, dtype=np.float32) + 3, False)

    # patch embedding: two temporal slices with distinct fillers (1.0 / 2.0)
    add("v.patch_embd.weight", np.full((_VH, 3, _VKH, _VKW), 1.0, dtype=np.float32), False)
    add("v.patch_embd.weight.1", np.full((_VH, 3, _VKH, _VKW), 2.0, dtype=np.float32), False)
    add("v.patch_embd.bias", np.arange(_VH, dtype=np.float32) + 4, False)
    add("v.post_ln.weight", np.arange(_VH, dtype=np.float32) + 5, False)

    add(
        "mm.patch_merger.weight",
        np.arange(_VO * _VH * _VKH * _VKW, dtype=np.float32).reshape(_VO, _VH, _VKH, _VKW),
        False,
    )
    add("mm.patch_merger.bias", np.arange(_VO, dtype=np.float32) + 6, False)
    add("mm.model.fc.weight", np.arange(_VO * _VO, dtype=np.float32).reshape(_VO, _VO), True)
    add("mm.gate.weight", np.arange(_VP * _VO, dtype=np.float32).reshape(_VP, _VO), True)
    add("mm.up.weight", np.arange(_VP * _VO, dtype=np.float32).reshape(_VP, _VO) + 50, True)
    add("mm.down.weight", np.arange(_VO * _VP, dtype=np.float32).reshape(_VO, _VP), True)
    add("mm.post_norm.weight", np.arange(_VO, dtype=np.float32) + 7, False)
    add("mm.post_norm.bias", np.arange(_VO, dtype=np.float32) + 8, False)
    return data


def _bf16_raw(arr: np.ndarray) -> np.ndarray:
    """torch-order float32 values -> flat uint8 bf16 bytes in ggml ne0-fastest order
    (a C-flatten of the torch-order array already runs ne0 fastest). Truncation to
    bf16 is exact only for values < 256 (bf16 carries 8 mantissa bits); every value
    an assert compares numerically sits below that bound, while the >=256 aranges
    (patch merger, full-fixture merger MLP) are asserted by name/shape only - the
    numeric cross-check in scripts/verify_mmproj_tensors.py compares bf16 on both
    sides, so the rounding cancels."""
    u32 = np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32)
    bf = (u32 >> 16).astype("<u2")
    return np.ascontiguousarray(bf.reshape(-1)).view(np.uint8)


def _write_vision_mmproj(path, *, metadata=None, tensors=None) -> str:
    """Full one-block mmproj with analytic values; block_count 1 matches the single block."""
    w = gguf.GGUFWriter(str(path), "clip")
    meta = dict(_CLIP_METADATA, **{"clip.vision.block_count": 1}) if metadata is None else metadata
    for key, val in sorted(meta.items()):
        if isinstance(val, bool):
            w.add_bool(key, val)
        elif isinstance(val, list):
            w.add_array(key, val)
        elif isinstance(val, str):
            w.add_string(key, val)
        elif isinstance(val, float):
            w.add_float32(key, val)
        else:
            w.add_int32(key, val)
    for name, (arr, bf16) in (tensors if tensors is not None else _vision_fixture_data()).items():
        if bf16:
            # gguf-py takes a torch-order shape and writes the dims reversed; with a
            # uint8 payload the last dim is a BYTE count, divided by the type size
            # before the reversal, so it must carry the input width * 2 bytes
            byte_shape = arr.shape[:-1] + (arr.shape[-1] * 2,)
            w.add_tensor(name, _bf16_raw(arr), raw_shape=byte_shape, raw_dtype=gguf.GGMLQuantizationType.BF16)
        else:
            w.add_tensor(name, arr)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)


@pytest.fixture(scope="session")
def vision_mmproj_gguf(tmp_path_factory) -> str:
    return _write_vision_mmproj(tmp_path_factory.mktemp("glm5next-vision") / "mmproj-vision.gguf")


def _full_vision_fixture_data() -> dict[str, tuple[np.ndarray, bool]]:
    """The tiny analytic tower replicated across all 24 pinned blocks, with the merger
    MLP widened to the pinned projection_intermediate_size: the only mmproj that both
    the config parser (pinned metadata + gate-width cross-check) and the reader
    (every block present) accept, so the load_weight boot path can consume it."""
    data: dict[str, tuple[np.ndarray, bool]] = {}
    for layer in range(_MMPROJ_PINNED_DEPTH := 24):
        for name, spec in _vision_fixture_data().items():
            if name.startswith("v.blk."):
                data[name.replace("v.blk.0.", f"v.blk.{layer}.")] = spec
    for name, spec in _vision_fixture_data().items():
        if not name.startswith("v.blk."):
            data[name] = spec
    proj_inter = 10240
    data["mm.gate.weight"] = (np.arange(proj_inter * _VO, dtype=np.float32).reshape(proj_inter, _VO), True)
    data["mm.up.weight"] = (np.arange(proj_inter * _VO, dtype=np.float32).reshape(proj_inter, _VO) + 50, True)
    data["mm.down.weight"] = (np.arange(_VO * proj_inter, dtype=np.float32).reshape(_VO, proj_inter), True)
    return data


@pytest.fixture(scope="session")
def full_vision_mmproj_gguf(tmp_path_factory) -> str:
    return _write_vision_mmproj(
        tmp_path_factory.mktemp("glm5next-vision-full") / "mmproj-vision-full.gguf",
        metadata=dict(_CLIP_METADATA),
        tensors=_full_vision_fixture_data(),
    )


def _vision_params(mmproj) -> dict:
    from freetoken.models.glm5_next.gguf import iter_gguf_vision_weights

    return dict(iter_gguf_vision_weights(mmproj))


def test_gguf_vision_reader_full_name_map(vision_mmproj_gguf):
    from freetoken.models.glm5_next.gguf import _VISION_BLOCK_MAP

    out = _vision_params(vision_mmproj_gguf)
    expected = {
        "visual.patch_embed.proj.weight",
        "visual.patch_embed.proj.bias",
        "visual.post_layernorm.weight",
        "visual.downsample.weight",
        "visual.downsample.bias",
        "visual.merger.proj.weight",
        "visual.merger.post_projection_norm.weight",
        "visual.merger.post_projection_norm.bias",
        "visual.merger.gate_proj.weight",
        "visual.merger.up_proj.weight",
        "visual.merger.down_proj.weight",
    }
    expected |= {f"visual.blocks.0.{sfx}" for sfx in _VISION_BLOCK_MAP.values()}
    assert set(out) == expected
    assert len(out) == 25  # 14 block tensors + 11 tower-level params
    assert all(t.dtype == torch.bfloat16 for t in out.values())


def test_gguf_vision_reader_axis_orders_and_temporal_stack(vision_mmproj_gguf):
    out = _vision_params(vision_mmproj_gguf)
    # every linear lands (out, in) with the three width signatures distinguishable
    assert out["visual.blocks.0.mlp.gate_proj.weight"].shape == (_VF, _VH)
    assert out["visual.blocks.0.mlp.down_proj.weight"].shape == (_VH, _VF)
    assert out["visual.merger.gate_proj.weight"].shape == (_VP, _VO)
    assert out["visual.merger.down_proj.weight"].shape == (_VO, _VP)
    # Conv2d downsample keeps (out, in, kh, kw)
    assert out["visual.downsample.weight"].shape == (_VO, _VH, _VKH, _VKW)
    # Conv3d patch embedding: the temporal axis is STACKED IN at dim 2, not cat-stretched
    w = out["visual.patch_embed.proj.weight"]
    assert w.shape == (_VH, 3, 2, _VKH, _VKW)
    assert (w[:, :, 0] == 1.0).all() and (w[:, :, 1] == 2.0).all()
    # the brief's mm.post_norm candidate was wrong: the wide biasful LayerNorm
    # is the merger's post_projection_norm; the tower RMSNorm comes from v.post_ln
    assert out["visual.merger.post_projection_norm.weight"].shape == (_VO,)
    assert out["visual.merger.post_projection_norm.bias"].shape == (_VO,)
    assert out["visual.post_layernorm.weight"].shape == (_VH,)


def test_gguf_vision_reader_qkv_row_order(vision_mmproj_gguf):
    """llama.cpp stores attn_qkv with q|k|v on the output axis; the reader must not
    transpose: vision.py unbinds (S, 3, H*D) over dim 1 == row-major q|k|v."""
    out = _vision_params(vision_mmproj_gguf)
    w = out["visual.blocks.0.attn.qkv.weight"].float()
    assert (w[:_VH] == 1.0).all() and (w[_VH : 2 * _VH] == 2.0).all() and (w[2 * _VH :] == 3.0).all()
    b = out["visual.blocks.0.attn.qkv.bias"].float()
    assert (b[:_VH] == 10.0).all() and (b[_VH : 2 * _VH] == 20.0).all() and (b[2 * _VH :] == 30.0).all()


def test_gguf_vision_reader_casts_follow_the_reference(vision_mmproj_gguf):
    """BF16 weights keep their bf16 bits; F32 norms/biases cast to bf16 exactly like
    the reference _iter_vision (which casts every visual.* tensor to bf16)."""
    out = _vision_params(vision_mmproj_gguf)
    gate = out["visual.blocks.0.mlp.gate_proj.weight"]
    assert gate.dtype == torch.bfloat16
    assert gate.float()[0, 0].item() == 0.0 and gate.float()[-1, -1].item() == float(_VF * _VH - 1)
    up = out["visual.blocks.0.mlp.up_proj.weight"]
    assert up.float()[0, 0].item() == 100.0
    assert out["visual.blocks.0.norm1.weight"].dtype == torch.bfloat16


def test_gguf_vision_reader_rejects_unknown_tensor(tmp_path):
    tensors = dict(_vision_fixture_data())
    tensors["v.blk.0.weird.weight"] = (np.ones(4, dtype=np.float32), False)
    p = _write_vision_mmproj(tmp_path / "weird.gguf", tensors=tensors)
    with pytest.raises(ValueError, match="unmapped mmproj tensor: v.blk.0.weird.weight"):
        _vision_params(p)


def test_gguf_vision_reader_requires_every_block(tmp_path):
    meta = dict(_CLIP_METADATA, **{"clip.vision.block_count": 2})
    p = _write_vision_mmproj(tmp_path / "gap.gguf", metadata=meta)
    with pytest.raises(ValueError, match=r"vision blocks \[1\]"):
        _vision_params(p)


def test_gguf_vision_reader_requires_complete_patch_slices(tmp_path):
    tensors = {
        name: spec
        for name, spec in _vision_fixture_data().items()
        if name not in ("v.patch_embd.weight", "v.patch_embd.weight.1")
    }
    tensors["v.patch_embd.weight.1"] = (np.full((_VH, 3, _VKH, _VKW), 2.0, dtype=np.float32), False)
    p = _write_vision_mmproj(tmp_path / "noslice.gguf", tensors=tensors)
    with pytest.raises(ValueError, match="temporal frames"):
        _vision_params(p)


def test_gguf_vision_reader_rejects_duplicate_patch_slice_index(tmp_path):
    """The base tensor and an explicit .0 slice both claim temporal frame 0: a
    duplicate must fail fast instead of silently picking one for the stack."""
    tensors = dict(_vision_fixture_data())
    tensors["v.patch_embd.weight.0"] = (np.full((_VH, 3, _VKH, _VKW), 9.0, dtype=np.float32), False)
    p = _write_vision_mmproj(tmp_path / "dup.gguf", tensors=tensors)
    with pytest.raises(ValueError, match="duplicate patch-embedding slice index 0"):
        _vision_params(p)


def test_gguf_vision_reader_rejects_block_beyond_depth(tmp_path):
    meta = dict(_CLIP_METADATA, **{"clip.vision.block_count": 1})
    tensors = dict(_vision_fixture_data())
    tensors["v.blk.1.ln1.weight"] = (np.arange(_VH, dtype=np.float32), False)
    p = _write_vision_mmproj(tmp_path / "extra.gguf", metadata=meta, tensors=tensors)
    with pytest.raises(ValueError, match="vision block 1 outside"):
        _vision_params(p)


def test_gguf_vision_reader_rejects_non_clip_arch(tmp_path):
    from freetoken.models.glm5_next.gguf import iter_gguf_vision_weights

    p = _write_gguf(tmp_path / "main.gguf")  # a glm5next gguf, not a clip mmproj
    with pytest.raises(ValueError, match="'glm5next'"):
        list(iter_gguf_vision_weights(p))


def test_iter_gguf_weights_streams_vision_after_trunk(glm5next_iter_gguf, vision_mmproj_gguf):
    from freetoken.models.glm5_next import iter_gguf_weights

    names = [
        name
        for name, _ in iter_gguf_weights(
            glm5next_iter_gguf,
            None,
            include_moe_experts=False,
            include_non_moe=True,
            include_vision=True,
            mmproj_path=vision_mmproj_gguf,
        )
    ]
    assert any(name.startswith("visual.") for name in names)
    # the stacked patch weight closes the stream: text trunk flushes its buffers first
    assert names[-1] == "visual.patch_embed.proj.weight"


def test_iter_gguf_weights_include_vision_requires_mmproj(glm5next_iter_gguf):
    from freetoken.models.glm5_next import iter_gguf_weights

    with pytest.raises(ValueError, match="--mmproj"):
        list(
            iter_gguf_weights(
                glm5next_iter_gguf, None, include_moe_experts=False, include_non_moe=True, include_vision=True
            )
        )


def test_load_weight_threads_mmproj_to_the_gguf_reader(glm5next_iter_gguf, full_vision_mmproj_gguf):
    """Boot wiring: load_weight resolves the spec WITH the mmproj fact, so the vision
    spec is selected and the tower tensors stream from the mmproj file. Needs the
    full fixture: the boot config path runs parse_mmproj_vision_config, whose pinned
    metadata and gate-width cross-check reject the one-block tiny tower."""
    from freetoken.models.weight import load_weight

    out = dict(
        load_weight(
            glm5next_iter_gguf,
            torch.device("cpu"),
            include_moe_experts=False,
            include_vision=True,
            mmproj_path=full_vision_mmproj_gguf,
        )
    )
    assert any(name.startswith("visual.") for name in out)
    assert "visual.patch_embed.proj.weight" in out


def test_load_vision_weight_threads_mmproj_to_the_gguf_reader(glm5next_iter_gguf, full_vision_mmproj_gguf, monkeypatch):
    """Encoder-only path (scripts/ftw_hotfix.py read_tower): load_vision_weight must
    forward mmproj_path to the family reader, so the GGUF branch reaches
    iter_gguf_vision_weights instead of dying on the dispatcher's '--mmproj' guard.
    Needs the full fixture: the spec resolution runs the boot config path, whose
    pinned metadata and gate-width cross-check reject the one-block tiny tower."""
    import freetoken.models.glm5_next.gguf as gguf_mod
    from freetoken.models.weight import load_vision_weight

    seen: dict = {}
    real_reader = gguf_mod.iter_gguf_vision_weights

    def spy_reader(mmproj_path, device=None):
        seen["mmproj_path"] = mmproj_path
        return real_reader(mmproj_path, device)

    monkeypatch.setattr(gguf_mod, "iter_gguf_vision_weights", spy_reader)
    out = dict(
        load_vision_weight(glm5next_iter_gguf, torch.device("cpu"), mmproj_path=full_vision_mmproj_gguf)
    )
    assert seen["mmproj_path"] == full_vision_mmproj_gguf
    assert any(name.startswith("visual.") for name in out)
    assert "visual.patch_embed.proj.weight" in out


# ------------------------------------------------------------------------------
# Phase 3: image processor + image-serving facts from the mmproj/tokenizer metadata
# ------------------------------------------------------------------------------


def test_mmproj_image_processor_config_full_map(mmproj_gguf):
    from freetoken.models.glm5_next.gguf import parse_mmproj_image_processor_config

    kw = parse_mmproj_image_processor_config(mmproj_gguf)
    assert kw["patch_size"] == 14
    assert kw["merge_size"] == 2
    assert kw["temporal_patch_size"] == 2
    assert kw["min_image_tokens"] == 16
    assert kw["max_image_tokens"] == 8000
    assert kw["image_mean"] == pytest.approx([0.48145467, 0.45782751, 0.40821072], abs=1e-6)
    assert kw["image_std"] == pytest.approx([0.26862955, 0.26130259, 0.27577710], abs=1e-6)


def test_mmproj_image_processor_config_missing_key_fails_fast(tmp_path):
    from freetoken.models.glm5_next.gguf import parse_mmproj_image_processor_config

    meta = {k: v for k, v in _CLIP_METADATA.items() if k != "clip.vision.image_mean"}
    p = _write_mmproj_gguf(tmp_path / "nonorm.gguf", metadata=meta)
    with pytest.raises(ValueError, match="clip.vision.image_mean"):
        parse_mmproj_image_processor_config(p)


def test_mmproj_image_processor_config_bad_geometry_fails_fast(tmp_path):
    from freetoken.models.glm5_next.gguf import parse_mmproj_image_processor_config

    meta = {**_CLIP_METADATA, "clip.vision.image_size": 450}
    p = _write_mmproj_gguf(tmp_path / "badgeom.gguf", metadata=meta)
    with pytest.raises(ValueError, match="multiple"):
        parse_mmproj_image_processor_config(p)


def test_gguf_image_serving_resolves_pinned_token(glm5next_gguf):
    from freetoken.models.glm5_next.gguf import resolve_gguf_image_serving
    from freetoken.models.gguf.reader import load_gguf_metadata

    assert resolve_gguf_image_serving(glm5next_gguf, load_gguf_metadata(glm5next_gguf)) == 154854


def _write_metadata_only_gguf(path, metadata) -> str:
    import gguf

    w = gguf.GGUFWriter(str(path), "glm5next")
    for key, val in sorted(metadata.items()):
        if isinstance(val, bool):
            w.add_bool(key, val)
        elif isinstance(val, list):
            w.add_array(key, val)
        elif isinstance(val, str):
            w.add_string(key, val)
        elif isinstance(val, float):
            w.add_float32(key, val)
        else:
            w.add_int32(key, val)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.close()
    return str(path)


def test_gguf_image_serving_missing_template_fails_fast(tmp_path):
    from freetoken.models.glm5_next.gguf import resolve_gguf_image_serving

    p = _write_metadata_only_gguf(tmp_path / "notmpl.gguf", dict(_GLM5NEXT_METADATA))
    from freetoken.models.gguf.reader import load_gguf_metadata

    with pytest.raises(ValueError, match="chat template"):
        resolve_gguf_image_serving(p, load_gguf_metadata(p))


def test_gguf_image_serving_template_without_placeholder_fails_fast(tmp_path):
    from freetoken.models.glm5_next.gguf import resolve_gguf_image_serving
    from freetoken.models.gguf.reader import load_gguf_metadata

    meta = {
        **_GLM5NEXT_METADATA,
        "tokenizer.chat_template": "plain template, no image macro",
        "tokenizer.ggml.tokens": _TOKENIZER_METADATA["tokenizer.ggml.tokens"],
    }
    p = _write_metadata_only_gguf(tmp_path / "noimg.gguf", meta)
    with pytest.raises(ValueError, match="placeholder"):
        resolve_gguf_image_serving(p, load_gguf_metadata(p))


def test_gguf_image_serving_vocab_without_token_fails_fast(tmp_path):
    from freetoken.models.glm5_next.gguf import resolve_gguf_image_serving
    from freetoken.models.gguf.reader import load_gguf_metadata

    meta = {
        **_GLM5NEXT_METADATA,
        "tokenizer.chat_template": _TOKENIZER_METADATA["tokenizer.chat_template"],
        "tokenizer.ggml.tokens": ["a", "b", "c"],
    }
    p = _write_metadata_only_gguf(tmp_path / "novocab.gguf", meta)
    with pytest.raises(ValueError, match="vocab carries no"):
        resolve_gguf_image_serving(p, load_gguf_metadata(p))


def test_gguf_image_serving_wrong_token_id_fails_fast(tmp_path):
    from freetoken.models.glm5_next.gguf import resolve_gguf_image_serving
    from freetoken.models.gguf.reader import load_gguf_metadata

    toks = ["x"] * 12
    toks[3] = "<|image|>"
    meta = {
        **_GLM5NEXT_METADATA,
        "tokenizer.chat_template": _TOKENIZER_METADATA["tokenizer.chat_template"],
        "tokenizer.ggml.tokens": toks,
    }
    p = _write_metadata_only_gguf(tmp_path / "wrongid.gguf", meta)
    with pytest.raises(ValueError, match="3 != the GLM-5.3-Flash reference 154854"):
        resolve_gguf_image_serving(p, load_gguf_metadata(p))


def test_mmproj_boot_carries_image_token_and_processor_path(glm5next_gguf, mmproj_gguf):
    from freetoken.models.gguf.config import build_gguf_shim
    from freetoken.models.glm5_next.gguf import parse_gguf_config

    shim = build_gguf_shim(glm5next_gguf, mmproj_path=mmproj_gguf)
    assert shim.image_token_id == 154854
    cfg = parse_gguf_config(shim)
    assert cfg.image_token_id == 154854
    assert cfg.mmproj_path == mmproj_gguf
    assert cfg.vision_config is not None


def test_get_mm_processor_text_only_gguf_is_none(glm5next_gguf):
    from freetoken.mm.processor import get_mm_processor

    assert get_mm_processor(glm5next_gguf) is None


def test_get_mm_processor_reraises_operator_errors(glm5next_gguf, tmp_path):
    """The blanket except that turns foreign configs into None must not swallow the
    explicit --mmproj operator errors: a bad flag is a boot stop, not a silent
    text-only downgrade."""
    from freetoken.mm.processor import get_mm_processor

    bad = _write_mmproj_gguf(tmp_path / "wrong.gguf", arch="glm5next")
    with pytest.raises(ValueError, match="!= 'clip'"):
        get_mm_processor(glm5next_gguf, mmproj_path=bad)
    with pytest.raises(ValueError, match="not found"):
        get_mm_processor(glm5next_gguf, mmproj_path=str(tmp_path / "absent.gguf"))


def test_mm_processor_fallback_builds_image_processor_from_mmproj(glm5next_gguf, mmproj_gguf):
    from freetoken.mm.processor import get_mm_processor

    proc = get_mm_processor(glm5next_gguf, mmproj_path=mmproj_gguf)
    assert proc is not None
    assert proc.placeholder == [154854]
    assert proc.merge == 2
    assert proc.patch_dim == 3 * 2 * 14 * 14
    ip = proc._image_processor()
    assert type(ip).__name__ == "Glm5NextImageProcessor"
    assert ip.patch_size == 14
    assert ip.merge_size == 2
    assert ip.temporal_patch_size == 2
    assert ip.min_image_tokens == 16
    assert ip.max_image_tokens == 8000
    assert list(ip.image_mean) == pytest.approx([0.48145467, 0.45782751, 0.40821072], abs=1e-6)


def test_mm_processor_preprocessor_config_file_takes_precedence(tmp_path):
    """The NVFP4-style path (a directory shipping preprocessor_config.json) must keep
    using the file: the mmproj fallback only fills the GGUF-shaped gap."""
    import json

    from types import SimpleNamespace

    from freetoken.mm.config import MultimodalConfig
    from freetoken.mm.processors.glm5_next import Glm5NextMMProcessor

    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps({"image_processor_type": "Glm5NextImageProcessor", "min_image_tokens": 123})
    )
    hf_config = SimpleNamespace(
        vision_config=SimpleNamespace(
            spatial_merge_size=2, in_channels=3, temporal_patch_size=2, patch_size=14
        ),
        image_token_id=7,
        mmproj_path=None,
    )
    proc = Glm5NextMMProcessor(hf_config, str(tmp_path), MultimodalConfig())
    assert proc._image_processor().min_image_tokens == 123


def test_mm_processor_mmproj_boot_wraps_image_into_mm_item(glm5next_gguf, mmproj_gguf):
    """Full CPU pass over the fallback: metadata-built image processor, MMItem wire
    format and the placeholder expansion the chat template slots into."""
    from PIL import Image

    from freetoken.mm.processor import get_mm_processor

    proc = get_mm_processor(glm5next_gguf, mmproj_path=mmproj_gguf)
    item = proc.process([Image.new("RGB", (448, 448), (127, 127, 127))])[0]
    assert item.feature.shape == (1024, 1176)
    assert item.feature.dtype == torch.bfloat16
    assert item.model_specific_data["grid_thw"] == [1, 32, 32]
    assert len(proc.prompt_replacement(item).full) == 256


# ------------------------------------------------------------------------------
# Phase 4: the converted-FTW metadata carrier boots vision without --mmproj
# ------------------------------------------------------------------------------


def _merged_carrier(dst, glm5next_gguf, extra_kvs) -> str:
    from freetoken.models.gguf.reader import write_metadata_gguf

    write_metadata_gguf(glm5next_gguf, dst, extra_kvs=extra_kvs)
    return str(dst)


def test_metadata_gguf_extra_kv_roundtrip(glm5next_gguf, tmp_path):
    """The converter packs the tower facts as raw KV records into the byte-verbatim
    carrier: every supported scalar/array type must read back value-identical."""
    import math

    from freetoken.models.gguf.reader import load_gguf_metadata

    kvs = {
        "ft.t.bool": True,
        "ft.t.int": 24,
        "ft.t.float": 1e-5,
        "ft.t.str": "glm5next",
        "ft.t.floats": [0.48145467, 0.45782751, 0.40821072],
        "ft.t.ints": [2, 3],
    }
    p = _merged_carrier(tmp_path / "carrier.gguf", glm5next_gguf, kvs)
    m = load_gguf_metadata(p)
    assert m["ft.t.bool"] is True
    assert m["ft.t.int"] == 24
    assert math.isclose(m["ft.t.float"], 1e-5, rel_tol=1e-6)
    assert m["ft.t.str"] == "glm5next"
    assert m["ft.t.floats"] == pytest.approx(kvs["ft.t.floats"], abs=1e-7)
    assert list(m["ft.t.ints"]) == [2, 3]


def test_ftw_carrier_boots_vision_without_mmproj(glm5next_gguf, mmproj_gguf, tmp_path):
    """A carrier with the mmproj clip.* KVs merged resolves the multimodal spec, the
    vision config, the pinned image token and the baked processor kwargs - the same
    facts a real converted FTW carries - with no mmproj file anywhere."""
    from freetoken.models.gguf.reader import load_gguf_metadata
    from freetoken.mm.processor import get_mm_processor
    from freetoken.utils import cached_load_hf_config

    mm_meta = load_gguf_metadata(mmproj_gguf)
    p = _merged_carrier(
        tmp_path / "carrier.gguf", glm5next_gguf,
        {k: v for k, v in mm_meta.items() if k.startswith("clip.")},
    )
    shim = cached_load_hf_config(p)
    assert shim.architectures == ["Glm5NextGGUFForConditionalGeneration"]
    assert shim.mmproj_path is None
    assert shim.vision_config.depth == 24
    assert shim.image_token_id == 154854
    assert shim.image_processor_kwargs["merge_size"] == 2
    proc = get_mm_processor(p)
    assert proc.placeholder == [154854]
    ip = proc._image_processor()
    assert ip.patch_size == 14
    assert list(ip.image_mean) == pytest.approx(mm_meta["clip.vision.image_mean"], abs=1e-6)


def test_metadata_gguf_extra_kv_collision_fails_fast(glm5next_gguf, tmp_path):
    """An extra KV clashing with a source KV (or the reserved output-weight fact) would
    write a duplicate record the re-parse silently dedupes - name the clash instead."""
    import pytest

    from freetoken.models.gguf.reader import OUTPUT_WEIGHT_PRESENT_KV, write_metadata_gguf

    with pytest.raises(ValueError, match="general.architecture"):
        write_metadata_gguf(glm5next_gguf, tmp_path / "a.gguf", extra_kvs={"general.architecture": "clip"})
    with pytest.raises(ValueError, match=OUTPUT_WEIGHT_PRESENT_KV):
        write_metadata_gguf(glm5next_gguf, tmp_path / "b.gguf", extra_kvs={OUTPUT_WEIGHT_PRESENT_KV: True})


def test_pack_kv_rejects_bad_arrays(tmp_path):
    from freetoken.models.gguf.reader import _pack_kv

    with pytest.raises(ValueError, match="heterogeneous"):
        _pack_kv("ft.mix", [1, 2.0])
    with pytest.raises(ValueError, match="empty-array"):
        _pack_kv("ft.empty", [])
    with pytest.raises(ValueError, match="unsupported KV value type"):
        _pack_kv("ft.dict", {"a": 1})


def test_ftw_carrier_partial_vision_metadata_fails_fast(glm5next_gguf, mmproj_gguf, tmp_path):
    """A hand-trimmed carrier (tower flag present, fields missing) fails fast instead
    of silently building a half-configured tower."""
    from freetoken.models.gguf.reader import load_gguf_metadata
    from freetoken.utils import cached_load_hf_config

    mm_meta = load_gguf_metadata(mmproj_gguf)
    partial = {k: v for k, v in mm_meta.items() if k.startswith("clip.") and k != "clip.vision.patch_size"}
    p = _merged_carrier(tmp_path / "partial.gguf", glm5next_gguf, partial)
    with pytest.raises(ValueError, match="clip.vision.patch_size"):
        cached_load_hf_config(p)


def test_raw_gguf_with_tower_metadata_fails_fast(glm5next_gguf, tmp_path):
    """clip.* KVs on a checkpoint GGUF whose tensor table is the LLM (not a metadata
    carrier) mean the tower file is missing: demand --mmproj, never serve silently."""
    from freetoken.utils import cached_load_hf_config

    meta = {**_GLM5NEXT_METADATA, **_TOKENIZER_METADATA, **_CLIP_METADATA}
    p = _write_gguf(tmp_path / "towelkeys.gguf", metadata=meta)
    with pytest.raises(ValueError, match="--mmproj"):
        cached_load_hf_config(p)


def test_ftw_text_only_carrier_unchanged(glm5next_gguf, tmp_path):
    """No clip.* KVs in the carrier: the pre-phase-4 text-only boot path, byte for
    byte (old FTWs converted before this campaign must not change shape)."""
    from freetoken.utils import cached_load_hf_config

    p = _merged_carrier(tmp_path / "textonly.gguf", glm5next_gguf, None)
    shim = cached_load_hf_config(p)
    assert shim.architectures == ["Glm5NextGGUFForCausalLM"]
    assert shim.mmproj_path is None
    assert shim.vision_config is None
    assert shim.image_token_id is None
    assert shim.image_processor_kwargs is None
