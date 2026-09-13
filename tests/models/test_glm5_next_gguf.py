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

import gguf
import numpy as np
import pytest
import torch

from freetoken.attention.base import AttnType
from freetoken.models.config import FullAttentionGroupConfig, LinearGatedDeltaGroupConfig
from freetoken.models.glm5_next.args import DSA_LAYER, KDA_LAYER

_NUM_LAYERS = 45  # block_count 46 - nextn_predict_layers 1: the trunk
_DSA_IDS = tuple(range(3, _NUM_LAYERS, 4))  # 3, 7, ..., 43
_KDA_IDS = tuple(i for i in range(_NUM_LAYERS) if i not in _DSA_IDS)
_VOCAB = 154880
_CONTEXT = 1_048_576

# attention.head_count_kv over block_count (46): 0 marks the KDA/recurrent layer
# class. The NextN draft block blk.45 is a DSA layer and falls outside the trunk
# slice -- same split as the real file's metadata.
_HEAD_COUNT_KV = [1 if (i % 4 == 3 or i == _NUM_LAYERS) else 0 for i in range(46)]

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
    w = gguf.GGUFWriter(str(path), arch)
    for key, val in sorted(meta.items()):
        if isinstance(val, bool):
            w.add_bool(key, val)
        elif isinstance(val, list):
            w.add_array(key, val)
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
_DOWN_BANK_BYTES = np.random.default_rng(1234).integers(0, 256, (_NE, _H, 210), dtype=np.uint8)


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
    entries = _iter_tensor_set() if tensors is None else tensors
    w = gguf.GGUFWriter(str(path), "glm5next")
    for key, val in sorted(meta.items()):
        if isinstance(val, bool):
            w.add_bool(key, val)
        elif isinstance(val, list):
            w.add_array(key, val)
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
    tmp_path, *, tokens, merges, special_ids, chat_template=None, token_type=None
):
    import gguf as gguf_mod

    path = tmp_path / "tokenizer.gguf"
    w = gguf_mod.GGUFWriter(str(path), "glm5next")
    w.add_array("tokenizer.ggml.tokens", list(tokens))
    w.add_array("tokenizer.ggml.token_type", token_type or [1] * len(tokens))
    w.add_array("tokenizer.ggml.merges", list(merges))
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
    isolates the converter path (ByteLevel-only pre-tokenization; the gguf's
    pre=glm4 scheme is ignored by GGUFGPTConverter) rather than vocab drift. Known
    special-token handling: the serve path re-encodes rendered chat text, so the
    converted tokenizer registers the gguf special tokens (token_type walk) as
    atomic AddedTokens - asserted below via the render->encode check. The
    code-pre tokenization divergence stays pinned; the boundary split is
    reported via warnings.warn, not masked.
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
    # The code sample is a PINNED known divergence: the reference pre-tokenizes with
    # the GPT-2 regex Split (grouping '(x' and '):\u010a' into single BPE pieces)
    # while GGUFGPTConverter emits ByteLevel-only pre-tokenization and ignores the
    # gguf's pre=glm4 scheme - both tokenizations decode losslessly, but they are
    # not identical. If a future converter emits the regex split, the pinned assert
    # below flips and this comment should be removed along with the mapping review.
    prose = [
        "The quick brown fox jumps over the lazy dog.",
        "Съешь ещё этих мягких французских булок, да выпей чаю.",
        "你好，世界！GLM-5.3-Flash 是一个混合专家模型。",
    ]
    code = "def f(x):\n    return x + 12345  # comment"

    for s in prose:
        ref_toks = ref.tokenize(s)
        got_toks = tok.convert_ids_to_tokens(tok.encode(s))
        assert got_toks == ref_toks, (
            f"prose sample {s[:20]!r} diverges from the reference: "
            f"gguf={got_toks[:10]} ref={ref_toks[:10]}"
        )
        assert tok.decode(tok.encode(s)) == s

    got_code = tok.convert_ids_to_tokens(tok.encode(code))
    ref_code = ref.tokenize(code)
    # pinned known divergence (reported finding): ByteLevel-only pre-tokenization
    # vs the reference's regex Split - gguf splits '(x' as '(' + 'x', keeps the
    # newline as a separate 'Ċ' token, and does not merge '):' + newline.
    assert got_code[:7] == ["def", "Ġf", "(", "x", "):", "Ċ", "ĠĠĠ"], got_code[:10]
    assert ref_code[:4] == ["def", "Ġf", "(x", "):Ċ"], ref_code[:10]
    assert tok.decode(tok.encode(code)) == code  # still lossless

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
