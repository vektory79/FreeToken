"""glm5_next (GLM-5.3-Flash) GGUF config shim: ``glm5next.*`` metadata -> ModelConfig.

Runs off a synthetic glm5next GGUF written with gguf-py: the KV section mirrors the
real checkpoint's metadata (the exact llama.cpp key spellings from
research/phase1-config-keys.md), and token_embd.weight is a stub whose ggml
shape[-1] is the vocab -- the only tensor fact build_gguf_shim consumes. The
147 GB real checkpoint is never opened.
"""

from __future__ import annotations

import numpy as np
import pytest

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


def test_iter_gguf_weights_is_a_phase2_stub():
    from freetoken.models.glm5_next import iter_gguf_weights

    with pytest.raises(NotImplementedError, match="Phase 2"):
        iter_gguf_weights("unused.gguf", None, include_moe_experts=True, include_non_moe=True)


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
