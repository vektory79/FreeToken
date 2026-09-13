"""GLM-5.3-Flash (``glm5next``) GGUF adapter: build the FreeToken ``ModelConfig``
from GGUF metadata.

The GGUF checkpoint's geometry is identical to the HF glm5_next model (hybrid
34xKDA + 11xDSA decoder, 288-expert sigmoid/noaux_tc MoE, mHC residual streams),
so this translates the ``glm5next.*`` metadata keys into a text-config view and
delegates to ``glm5_next.config.parse_config`` -- the same ``ModelConfig`` /
``Glm5NextArgs`` payload the HF path produces, sourced from GGUF KV metadata
instead of a HF config object. Weight iteration (the tensor-name translator over
packed GGUF blocks) is Phase 2 work and stubbed out here.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import TYPE_CHECKING, Iterator

from freetoken.models.glm5_next.args import DSA_LAYER, KDA_LAYER

if TYPE_CHECKING:
    import torch

    from freetoken.models.config import ModelConfig
    from freetoken.models.gguf.config import GgufConfigShim

# llama-hparams.h: LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID == 2 (the real file's value).
# FreeToken hardcodes this sigmoid/noaux_tc router, so any other value mis-serves.
_EXPERT_GATING_SIGMOID = 2


def _scalar(val, key: str):
    """Fold a llama key_or_arr value (scalar, or uniform per-layer array) to scalar."""
    if isinstance(val, (list, tuple)):
        if not val:
            raise ValueError(f"glm5next.{key} is an empty per-layer array")
        if len(set(val)) > 1:
            raise ValueError(
                f"glm5next.{key} varies per layer; FreeToken carries one scalar"
            )
        return val[0]
    return val


def parse_gguf_config(shim: "GgufConfigShim") -> "ModelConfig":
    from freetoken.models.glm5_next.config import parse_config

    m = shim.metadata

    def g(key: str):
        val = m.get(f"glm5next.{key}")
        if val is None:
            raise KeyError(f"missing GGUF metadata key glm5next.{key}")
        return val

    # Optional keys that would silently change serving semantics on a variant
    # checkpoint; every one carries the pinned value on the real file.
    gating = m.get("glm5next.expert_gating_func")
    if gating is not None and int(gating) != _EXPERT_GATING_SIGMOID:
        raise ValueError(
            f"glm5next.expert_gating_func {gating} != 2 (llama.cpp SIGMOID); "
            "FreeToken only implements the sigmoid/noaux_tc router"
        )
    rope_dim = m.get("glm5next.rope.dimension_count")
    if rope_dim is not None and int(rope_dim) != 0:
        raise ValueError(
            f"glm5next.rope.dimension_count {rope_dim} != 0; the arch is nope-only "
            "(glm5next.cpp asserts n_rot() == 0) and would be served without rope"
        )
    kv_vocab = m.get("glm5next.vocab_size")
    if kv_vocab is not None and int(kv_vocab) != int(shim.vocab_size):
        raise ValueError(
            f"glm5next.vocab_size {kv_vocab} != vocab sized from token_embd.weight "
            f"({shim.vocab_size})"
        )

    # block_count INCLUDES the NextN/MTP draft block (blk.{n_layer}, no hc_* tensors);
    # FreeToken has no MTP, so the config describes the trunk and Phase 2 skips
    # blk.>= trunk. Per-layer arrays span block_count and are sliced to trunk length.
    block_count = int(g("block_count"))
    nextn = int(m.get("glm5next.nextn_predict_layers", 0))
    if not 0 <= nextn < block_count:
        raise ValueError(
            f"glm5next.nextn_predict_layers {nextn} outside 0..{block_count - 1}"
        )
    num_layers = block_count - nextn

    # Per-layer attention.head_count_kv == 0 marks the KDA/recurrent layer class
    # (glm5next.cpp is_recr); the split is read over block_count, trunk counts.
    head_kv_raw = g("attention.head_count_kv")
    scalar_split = False
    if isinstance(head_kv_raw, (list, tuple)):
        head_kv = [int(x) for x in head_kv_raw]
        if len(head_kv) != block_count:
            raise ValueError(
                f"glm5next.attention.head_count_kv has {len(head_kv)} entries "
                f"for block_count {block_count}"
            )
    else:
        # key_or_arr: a scalar applies to every layer; the uniform split then trips
        # the strict-minority check below.
        scalar_split = True
        head_kv = [int(head_kv_raw)] * block_count
    layer_types = tuple(
        KDA_LAYER if hkv == 0 else DSA_LAYER for hkv in head_kv[:num_layers]
    )
    n_kda = layer_types.count(KDA_LAYER)
    if not 0 < n_kda < num_layers:
        # Name the scalar case explicitly: the bare count alone reads like bad data.
        hint = (
            " (a scalar head_count_kv applies to every layer; glm5next requires "
            "mixed KDA/DSA)"
            if scalar_split
            else ""
        )
        raise ValueError(
            f"glm5next GGUF: {n_kda} KDA layers of {num_layers} trunk layers "
            f"(llama.cpp requires 0 < n_recr < n_layer){hint}"
        )

    # llama.cpp feeds ONE head count to the KDA (d_inner / ssm_beta widths) and the
    # MLA (q_b / wo widths) alike (glm5next.cpp:105-106,175,182). The GGUF carries a
    # single head-count key, so the MLA==KDA equality holds structurally here; Phase 2
    # still enforces it on tensor shapes (d_inner == num_heads * kda.head_dim).
    num_heads = int(_scalar(g("attention.head_count"), "attention.head_count"))

    index_kpool = int(g("attention.indexer.kpool"))
    index_topk = int(g("attention.indexer.top_k"))
    if index_kpool <= 0 or index_topk <= 0:
        # A negative kpool would pass the modulo below and silently disable compression.
        raise ValueError(
            f"glm5next.attention.indexer: kpool {index_kpool} and top_k {index_topk} "
            "must be > 0 (glm5next.cpp asserts indexer_kpool > 0)"
        )
    if index_topk % index_kpool != 0:
        raise ValueError(
            f"indexer.top_k {index_topk} not divisible by indexer.kpool {index_kpool} "
            "(glm5next.cpp select-pool math)"
        )

    swiglu_raw = m.get("glm5next.swiglu_clamp_exp")
    swiglu_limit = (
        None if swiglu_raw is None else float(_scalar(swiglu_raw, "swiglu_clamp_exp"))
    )
    shexp_raw = m.get("glm5next.swiglu_clamp_shexp")
    if shexp_raw is not None:
        shexp_limit = float(_scalar(shexp_raw, "swiglu_clamp_shexp"))
        if shexp_limit != swiglu_limit:
            raise ValueError(
                f"glm5next.swiglu_clamp_shexp {shexp_limit} != swiglu_clamp_exp "
                f"{swiglu_limit}; FreeToken carries one clamp for both expert kinds"
            )

    first_dense = int(g("leading_dense_block_count"))
    if not 0 <= first_dense <= num_layers:
        raise ValueError(
            f"glm5next.leading_dense_block_count {first_dense} outside 0..{num_layers}"
        )

    moe_inter = int(_scalar(g("expert_feed_forward_length"), "expert_feed_forward_length"))
    n_shared = int(g("expert_shared_count"))
    shexp_ffn = m.get("glm5next.expert_shared_feed_forward_length")
    if shexp_ffn is not None and int(shexp_ffn) != moe_inter * n_shared:
        # FreeToken rejects a present-but-mismatched key; llama derives
        # n_ff_exp * max(1, n_shared), so it would clamp a 0 shared count to 1.
        raise ValueError(
            f"glm5next.expert_shared_feed_forward_length {shexp_ffn} != "
            f"expert_feed_forward_length * expert_shared_count ({moe_inter} * {n_shared})"
        )
    ln_eps = m.get("glm5next.attention.layer_norm_epsilon")
    if ln_eps is not None and not math.isclose(float(ln_eps), 1e-6, rel_tol=1e-6):
        # FreeToken hardcodes the indexer k-norm eps (attention.py _IdxLayerNorm).
        raise ValueError(
            f"glm5next.attention.layer_norm_epsilon {ln_eps} != the hardcoded 1e-6"
        )

    text = SimpleNamespace(
        num_hidden_layers=num_layers,
        layer_types=layer_types,
        mlp_layer_types=("dense",) * first_dense + ("sparse",) * (num_layers - first_dense),
        hidden_size=int(g("embedding_length")),
        num_attention_heads=num_heads,
        q_lora_rank=int(g("attention.q_lora_rank")),
        kv_lora_rank=int(g("attention.kv_lora_rank")),
        # NoPE: rope.dimension_count is UNMAPPED (llama asserts n_rot()==0); positional
        # information enters only through the indexer's pool-compression APE.
        qk_rope_head_dim=0,
        qk_nope_head_dim=int(g("attention.key_length_mla")),
        v_head_dim=int(g("attention.value_length_mla")),
        rms_norm_eps=float(g("attention.layer_norm_rms_epsilon")),
        max_position_embeddings=int(g("context_length")),
        vocab_size=int(shim.vocab_size),
        intermediate_size=int(_scalar(g("feed_forward_length"), "feed_forward_length")),
        # FreeToken flips this to "swiglu_clamp" when swiglu_limit resolves (config.py).
        hidden_act="silu",
        mla_use_nope=True,
        index_n_heads=int(g("attention.indexer.head_count")),
        index_head_dim=int(g("attention.indexer.key_length")),
        index_topk=index_topk,
        # No GGUF key (llama never reads indexer.types); every DSA layer owns a full
        # indexer, and the payload is per-layer so family code can index by layer id.
        indexer_types=("full",) * num_layers,
        index_kpool=index_kpool,
        index_kpool_compress=index_kpool > 1,
        # llama's n_select math always force-includes the in-progress tail pool.
        index_kpool_always_select_tail=True,
        # rope-free indexer in llama.cpp (n_rot()==0); no interleave key exists.
        indexer_rope_interleave=False,
        mhc=True,
        mhc_num_residual_streams=int(g("hyper_connection.count")),
        mhc_sinkhorn_iterations=int(g("hyper_connection.sinkhorn_iterations")),
        hc_eps=float(g("hyper_connection.epsilon")),
        linear_attn_config={
            "num_heads": num_heads,
            "head_dim": int(g("kda.head_dim")),
            "short_conv_kernel_size": int(g("ssm.conv_kernel")),
            "gate_lower_bound": float(g("kda.gate_lower_bound")),
        },
        swiglu_limit=swiglu_limit,
        num_experts=int(g("expert_count")),
        num_experts_per_tok=int(_scalar(g("expert_used_count"), "expert_used_count")),
        moe_intermediate_size=moe_inter,
        norm_topk_prob=bool(g("expert_weights_norm")),
        n_shared_experts=n_shared,
        routed_scaling_factor=float(g("expert_weights_scale")),
        tie_word_embeddings=bool(shim.tie_word_embeddings),
    )
    return parse_config(
        SimpleNamespace(
            text_config=text,
            architectures=list(shim.architectures),
            model_type=shim.model_type,
        )
    )


def iter_gguf_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, "torch.Tensor"]]:
    raise NotImplementedError(
        "glm5next GGUF weight iteration lands in a follow-up change (Phase 2: "
        "tensor-name translator over the packed GGUF blocks); this wave ships the "
        "config shim only."
    )


__all__ = ["parse_gguf_config", "iter_gguf_weights"]