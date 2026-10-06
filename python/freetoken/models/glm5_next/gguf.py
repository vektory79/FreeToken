"""GLM-5.3-Flash (``glm5next``) GGUF adapter: build the FreeToken ``ModelConfig``
from GGUF metadata.

The GGUF checkpoint's geometry is identical to the HF glm5_next model (hybrid
34xKDA + 11xDSA decoder, 288-expert sigmoid/noaux_tc MoE, mHC residual streams),
so this translates the ``glm5next.*`` metadata keys into a text-config view and
delegates to ``glm5_next.config.parse_config`` -- the same ``ModelConfig`` /
``Glm5NextArgs`` payload the HF path produces, sourced from GGUF KV metadata
instead of a HF config object. Weight iteration translates the gguf tensor names
to the params ``models/glm5_next/weight.py`` yields for HF checkpoints (fused
KDA in_proj / conv1d, fused DSA kv_b_proj, derived A_log); routed-expert banks
stream verbatim (ggml type intact) via ``iter_gguf_expert_sources``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Iterator

import torch

from freetoken.layers.base import BaseOP
from freetoken.models.glm5_next.args import DSA_LAYER, KDA_LAYER
from freetoken.models.gguf.dequant import (
    GGML_NAME,
    GGML_Q6_K,
    GGML_Q8_0,
    dequantize,
    row_bytes,
)

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig
    from freetoken.models.gguf.config import GgufConfigShim
    from freetoken.models.gguf.reader import GgufTensor
    from freetoken.models.glm5_next.config import VisionConfig

logger = logging.getLogger(__name__)

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


# GLM-5.3-Flash vision reference (the NVFP4 checkpoint's vision_config): every field
# the mmproj metadata does not carry is pinned to these values, and a mismatch with
# the file fails fast instead of silently mis-serving the tower.
_MMPROJ_PINNED = {
    "depth": 24,
    "hidden_size": 1024,
    "num_heads": 16,
    "intermediate_size": 4096,
    "out_hidden_size": 4096,
    "patch_size": 14,
    "spatial_merge_size": 2,
    "rms_norm_eps": 1e-5,
    "swiglu_limit": 10.0,
    "temporal_patch_size": 2,
    "in_channels": 3,
    "projection_intermediate_size": 10240,
    "attention_bias": True,
}


def parse_mmproj_vision_config(mmproj_path: str) -> "VisionConfig":
    """VisionConfig for the glm5next tower from a clip-architecture mmproj GGUF.

    Nine fields map from clip.* metadata; the four the metadata does not carry
    (temporal_patch_size, in_channels, projection_intermediate_size, attention_bias)
    come from the pinned GLM-5.3-Flash reference and are verified against tensor
    shapes whenever the mmproj tensor table exposes them.
    """
    from freetoken.models.gguf.reader import _reader, load_gguf_metadata
    from freetoken.models.glm5_next.config import VisionConfig

    m = load_gguf_metadata(mmproj_path)

    def c(key: str):
        val = m.get(f"clip.{key}")
        if val is None:
            raise ValueError(f"mmproj {mmproj_path}: missing required metadata key clip.{key}")
        return val

    if not m.get("clip.has_vision_encoder"):
        raise ValueError(
            f"mmproj {mmproj_path}: clip.has_vision_encoder is not True; the file does "
            "not carry a vision tower"
        )
    projector = c("projector_type")
    if str(projector) != "glm5next":
        raise ValueError(
            f"mmproj {mmproj_path}: clip.projector_type {projector!r} != 'glm5next'; "
            "FreeToken implements only the glm5next projector"
        )

    def pinned_int(key: str, field: str) -> int:
        val = int(c(key))
        expected = _MMPROJ_PINNED[field]
        if val != expected:
            raise ValueError(
                f"mmproj {mmproj_path}: clip.{key} {val} != the GLM-5.3-Flash vision "
                f"reference {expected} ({field})"
            )
        return val

    depth = pinned_int("vision.block_count", "depth")
    hidden = pinned_int("vision.embedding_length", "hidden_size")
    heads = pinned_int("vision.attention.head_count", "num_heads")
    inter = pinned_int("vision.feed_forward_length", "intermediate_size")
    out_hidden = pinned_int("vision.projection_dim", "out_hidden_size")
    patch = pinned_int("vision.patch_size", "patch_size")
    merge = pinned_int("vision.spatial_merge_size", "spatial_merge_size")
    eps = float(c("vision.attention.layer_norm_epsilon"))
    if not math.isclose(eps, _MMPROJ_PINNED["rms_norm_eps"], rel_tol=1e-5):
        raise ValueError(
            f"mmproj {mmproj_path}: clip.vision.attention.layer_norm_epsilon {eps} != "
            f"the GLM-5.3-Flash vision reference {_MMPROJ_PINNED['rms_norm_eps']}"
        )
    swiglu = float(c("vision.swiglu_limit"))
    if not math.isclose(swiglu, _MMPROJ_PINNED["swiglu_limit"], rel_tol=1e-6):
        raise ValueError(
            f"mmproj {mmproj_path}: clip.vision.swiglu_limit {swiglu} != the "
            f"GLM-5.3-Flash vision reference {_MMPROJ_PINNED['swiglu_limit']}"
        )

    # tensor-table cross checks: header shapes only, the weight data is never read
    temporal = _MMPROJ_PINNED["temporal_patch_size"]
    in_channels = _MMPROJ_PINNED["in_channels"]
    proj_inter = _MMPROJ_PINNED["projection_intermediate_size"]
    shapes = {
        t.name: tuple(reversed([int(d) for d in t.shape]))
        for t in _reader(mmproj_path).tensors
    }
    patch_slices = [
        n for n in shapes if n == "v.patch_embd.weight" or n.startswith("v.patch_embd.weight.")
    ]
    if patch_slices:
        # the Conv3d patch embedding ships one slice per temporal frame; the base
        # tensor must be present, it is the slice the input channels are read from
        if "v.patch_embd.weight" not in shapes:
            raise ValueError(
                f"mmproj {mmproj_path}: v.patch_embd.weight.N slices without the base "
                "v.patch_embd.weight tensor"
            )
        if len(patch_slices) != temporal:
            raise ValueError(
                f"mmproj {mmproj_path}: {len(patch_slices)} v.patch_embd.weight slices "
                f"!= temporal_patch_size {temporal}"
            )
        found_in = shapes["v.patch_embd.weight"][1]
        if found_in != in_channels:
            raise ValueError(
                f"mmproj {mmproj_path}: v.patch_embd.weight has {found_in} input "
                f"channels != the GLM-5.3-Flash vision reference {in_channels}"
            )
    gate = shapes.get("mm.gate.weight")
    if gate is not None and gate[0] != proj_inter:
        raise ValueError(
            f"mmproj {mmproj_path}: mm.gate.weight output width {gate[0]} != the "
            f"GLM-5.3-Flash vision reference {proj_inter} (projection_intermediate_size)"
        )
    if _MMPROJ_PINNED["attention_bias"] and "v.blk.0.attn_qkv.bias" not in shapes:
        raise ValueError(
            f"mmproj {mmproj_path}: no v.blk.*.attn_qkv.bias tensors; the GLM-5.3-Flash "
            "vision reference is attention_bias=True"
        )

    return VisionConfig(
        hidden_size=hidden,
        depth=depth,
        num_heads=heads,
        intermediate_size=inter,
        projection_intermediate_size=proj_inter,
        out_hidden_size=out_hidden,
        in_channels=in_channels,
        patch_size=patch,
        temporal_patch_size=temporal,
        spatial_merge_size=merge,
        rms_norm_eps=eps,
        swiglu_limit=swiglu,
        attention_bias=_MMPROJ_PINNED["attention_bias"],
    )


# mmproj block tensors -> the tail of the visual.blocks.N.<rel> param (everything
# before the dot: the tensor-name suffix llama.cpp stores under v.blk.<N>.)
_VISION_BLOCK_MAP = {
    "attn_qkv.weight": "attn.qkv.weight",
    "attn_qkv.bias": "attn.qkv.bias",
    "attn_out.weight": "attn.proj.weight",
    "attn_out.bias": "attn.proj.bias",
    "attn_q_norm.weight": "attn.q_norm.weight",
    "attn_k_norm.weight": "attn.k_norm.weight",
    "ln1.weight": "norm1.weight",
    "ln2.weight": "norm2.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_gate.bias": "mlp.gate_proj.bias",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_up.bias": "mlp.up_proj.bias",
    "ffn_down.weight": "mlp.down_proj.weight",
    "ffn_down.bias": "mlp.down_proj.bias",
}

# mmproj tower-level tensors -> the full visual.* param (verified against the
# NVFP4 reference; mm.post_norm is the merger's LayerNorm, v.post_ln the tower's
# RMSNorm - the brief's candidate table had the two swapped)
_VISION_MM_MAP = {
    "mm.patch_merger.weight": "visual.downsample.weight",
    "mm.patch_merger.bias": "visual.downsample.bias",
    "mm.model.fc.weight": "visual.merger.proj.weight",
    "mm.gate.weight": "visual.merger.gate_proj.weight",
    "mm.up.weight": "visual.merger.up_proj.weight",
    "mm.down.weight": "visual.merger.down_proj.weight",
    "mm.post_norm.weight": "visual.merger.post_projection_norm.weight",
    "mm.post_norm.bias": "visual.merger.post_projection_norm.bias",
}


def iter_gguf_vision_weights(
    mmproj_path: str, device=None
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield (visual.* param, bf16 tensor) for every mmproj tower tensor.

    The GGUF reader returns torch-order shapes (ggml ne reversed), so every linear
    form lands (out, in) directly and the q|k|v fusion sits on the output axis
    exactly as vision.py's qkv split expects. The Conv3d patch embedding ships one
    slice per temporal frame: they stack on a new dim 2 into (out, in, kt, kh, kw).
    Everything casts to bf16 like the reference _iter_vision. An unmapped name,
    a missing vision block or an incomplete patch-slice set raises ValueError
    naming the fact - no silent skips.
    """
    from freetoken.models.gguf.reader import iter_gguf_tensors, load_gguf_metadata

    # device is accepted for reader-signature parity with the HF iter_vision_weights;
    # the loader materializes onto the model params from these CPU tensors
    depth = int(load_gguf_metadata(mmproj_path)["clip.vision.block_count"])
    blocks: set[int] = set()
    patch: dict[int, torch.Tensor] = {}
    for t in iter_gguf_tensors(mmproj_path):
        name = t.name
        tensor = _cast(t, torch.bfloat16)
        if name.startswith("v.blk."):
            parts = name.split(".")
            layer, suffix = int(parts[2]), ".".join(parts[3:])
            rel = _VISION_BLOCK_MAP.get(suffix)
            if rel is None:
                raise ValueError(f"unmapped mmproj tensor: {name}")
            blocks.add(layer)
            pn = f"visual.blocks.{layer}.{rel}"
            yield pn, _logged(pn, tensor, t)
        elif name == "v.patch_embd.weight" or name.startswith("v.patch_embd.weight."):
            patch[0 if name == "v.patch_embd.weight" else int(name.rsplit(".", 1)[1])] = tensor
        elif name == "v.patch_embd.bias":
            yield "visual.patch_embed.proj.bias", _logged("visual.patch_embed.proj.bias", tensor, t)
        elif name == "v.post_ln.weight":
            yield "visual.post_layernorm.weight", _logged("visual.post_layernorm.weight", tensor, t)
        else:
            rel = _VISION_MM_MAP.get(name)
            if rel is None:
                raise ValueError(f"unmapped mmproj tensor: {name}")
            yield rel, _logged(rel, tensor, t)

    missing = sorted(set(range(depth)) - blocks)
    if missing:
        raise ValueError(
            f"mmproj {mmproj_path}: no tensors for vision blocks {missing} "
            f"(clip.vision.block_count {depth})"
        )
    temporal = int(_MMPROJ_PINNED["temporal_patch_size"])
    if sorted(patch) != list(range(temporal)):
        raise ValueError(
            f"mmproj {mmproj_path}: patch-embedding slices {sorted(patch)} != the "
            f"temporal frames 0..{temporal - 1} (the base v.patch_embd.weight is frame 0)"
        )
    # stack, not cat: each slice is (out, in, kh, kw) and the temporal axis is
    # INSERTED at dim 2 -> (out, in, kt, kh, kw); cat would stretch kh instead
    weight = torch.stack([patch[i] for i in range(temporal)], dim=2)
    pn = "visual.patch_embed.proj.weight"
    yield pn, _logged(pn, weight, note="temporal stack of patch_embd slices")


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
    # gguf marker for the model build (gemma4 precedent): is_gguf_model keys on it.
    return replace(
        parse_config(
            SimpleNamespace(
                text_config=text,
                architectures=list(shim.architectures),
                model_type=shim.model_type,
                # the --mmproj shim carries the tower config built from the mmproj
                # metadata; text-only shims read as None through getattr's default
                vision_config=getattr(shim, "vision_config", None),
            )
        ),
        moe_weight_format="gguf",
    )


# --------------------------------------------------------------------------------------
# Weight loading: GGUF tensor names -> FreeToken glm5_next module params.
# Mapping authority: research/phase2-tensor-map.md (1383 mapped + 29 skip-listed MTP).
# --------------------------------------------------------------------------------------

# Routed-expert stacks: NOT yielded by iter_gguf_weights (mirrors the HF path, where
# routed experts only serve from the offload cache); iter_gguf_expert_sources streams
# them with their ggml type intact - quant dispatch is Phase 4/5 and must not see
# dequantized copies (dequant.py has no IQ/Q3_K/Q4_K support either).
_EXPERT_BANK_ROLES = {
    "ffn_gate_exps.weight": "gate",
    "ffn_up_exps.weight": "up",
    "ffn_down_exps.weight": "down",
}

# suffix (after "blk.N.") -> (FreeToken param name, cast). "qw" yields the packed
# block bytes under the .qweight name (Phase 4 native-quant path, gemma4 convention);
# "bf16"/"fp32" dequantize in place, matching the dtype weight.py yields for HF
# checkpoints: layer norms + o_norm + indexer linears bf16, A_log / dt_bias / hc_* /
# e_score_correction_bias / ape fp32. hc_*_fn and the indexer linears are Q8_0 in the
# gguf but dense downstream, so they dequantize here.
_COMMON_LAYER_MAP = {
    "attn_norm.weight": ("input_layernorm.weight", "bf16"),
    "ffn_norm.weight": ("post_attention_layernorm.weight", "bf16"),
    "hc_attn_fn.weight": ("hc_attn_fn", "fp32"),
    "hc_attn_base.weight": ("hc_attn_base", "fp32"),
    "hc_attn_scale.weight": ("hc_attn_scale", "fp32"),
    "hc_ffn_fn.weight": ("hc_ffn_fn", "fp32"),
    "hc_ffn_base.weight": ("hc_ffn_base", "fp32"),
    "hc_ffn_scale.weight": ("hc_ffn_scale", "fp32"),
}
_KDA_LAYER_MAP = {
    "ssm_f_b.weight": ("self_attn.f_b_proj.weight", "qw"),
    "ssm_g_b.weight": ("self_attn.g_b_proj.weight", "qw"),
    "attn_output.weight": ("self_attn.o_proj.weight", "qw"),
    "ssm_norm.weight": ("self_attn.o_norm.weight", "bf16"),
}
_DSA_LAYER_MAP = {
    "attn_q_a.weight": ("self_attn.q_a_proj.weight", "qw"),
    "attn_q_a_norm.weight": ("self_attn.q_a_layernorm.weight", "bf16"),
    "attn_q_b.weight": ("self_attn.q_b_proj.weight", "qw"),
    "attn_kv_a_mqa.weight": ("self_attn.kv_a_proj_with_mqa.weight", "qw"),
    "attn_kv_a_norm.weight": ("self_attn.kv_a_layernorm.weight", "bf16"),
    "attn_output.weight": ("self_attn.o_proj.weight", "qw"),
    "indexer.attn_k.weight": ("self_attn.indexer.wk.weight", "bf16"),
    "indexer.attn_q_b.weight": ("self_attn.indexer.wq_b.weight", "bf16"),
    "indexer.proj.weight": ("self_attn.indexer.weights_proj.weight", "bf16"),
    "indexer.k_norm.weight": ("self_attn.indexer.k_norm.weight", "bf16"),
    "indexer.k_norm.bias": ("self_attn.indexer.k_norm.bias", "bf16"),
    "indexer_compressor_gate.weight": ("self_attn.indexer.index_kpool_compress_gate", "bf16"),
    "indexer_compressor_ape.weight": ("self_attn.indexer.index_kpool_compress_ape", "fp32"),
}
_DENSE_FFN_MAP = {
    "ffn_gate.weight": ("mlp.gate_proj.weight", "qw"),
    "ffn_up.weight": ("mlp.up_proj.weight", "qw"),
    "ffn_down.weight": ("mlp.down_proj.weight", "qw"),
}
_MOE_MAP = {
    "ffn_gate_inp.weight": ("mlp.gate.weight", "bf16"),
    "exp_probs_b.bias": ("mlp.e_score_correction_bias", "fp32"),
    "ffn_gate_shexp.weight": ("mlp.shared_experts.gate_proj.weight", "qw"),
    "ffn_up_shexp.weight": ("mlp.shared_experts.up_proj.weight", "qw"),
    "ffn_down_shexp.weight": ("mlp.shared_experts.down_proj.weight", "qw"),
}

# KDA in_proj fusion slots. The concat order q|k|v|b|f_a|g_a is pinned by the
# consumer: Glm5NextKDA._in_proj_split = [p, p, p, h, d, d] (kda.py:64-65) and
# weight.py:94 _KDA_IN_PROJ = (q_proj, k_proj, v_proj, b_proj, f_a_proj, g_a_proj).
_IN_PROJ_SLOTS = {
    "attn_q.weight": "q",
    "attn_k.weight": "k",
    "attn_v.weight": "v",
    "ssm_beta.weight": "b",
    "ssm_f_a.weight": "f_a",
    "ssm_g_a.weight": "g_a",
}
_IN_PROJ_SLOT_ORDER = ("q", "k", "v", "b", "f_a", "g_a")

# KDA conv1d fusion: one depthwise conv over the merged q|k|v stream; weight.py
# concatenates {q,k,v}_conv1d on the channel axis, Glm5NextKDA.conv1d is (24576, 1, 4).
_CONV_SLOTS = {
    "ssm_conv1d_q.weight": "q",
    "ssm_conv1d_k.weight": "k",
    "ssm_conv1d_v.weight": "v",
}
_CONV_SLOT_ORDER = ("q", "k", "v")

_KV_B_SLOTS = {"attn_k_b.weight": "k", "attn_v_b.weight": "v"}


def _cast(t: "GgufTensor", dtype: torch.dtype) -> torch.Tensor:
    """Dense ``dtype`` tensor from a (possibly quantized) GgufTensor: dequantize.py
    returns storage-order values, the caller reshapes to the torch shape."""
    return dequantize(t.packed().reshape(-1), t.ggml_type, dtype).reshape(t.shape)


def _logged(
    name: str, tensor: torch.Tensor, t: "GgufTensor | None" = None, note: str = ""
) -> torch.Tensor:
    """Plan acceptance 'dtype/type per tensor logged': one debug line per yield,
    silent at INFO so serving never pays for it."""
    if t is not None:
        logger.debug(
            "glm5next %s: ggml %s %dx%dB -> %s %s",
            name,
            GGML_NAME.get(t.ggml_type, t.ggml_type),
            t.rows,
            t.row_bytes,
            tuple(tensor.shape),
            tensor.dtype,
        )
    else:
        logger.debug(
            "glm5next %s: %s -> %s %s", name, note, tuple(tensor.shape), tensor.dtype
        )
    return tensor


def _a_log(t: "GgufTensor", name: str) -> torch.Tensor:
    # gguf stores -exp(A_log) per head (kimi-k3 convention, glm5next.cpp:7); invert
    # in fp32 - the recurrent kernels read A_log as fp32 (kda.py gate params).
    ssm = _cast(t, torch.float32)
    if bool((ssm >= 0).any()):
        raise ValueError(
            f"{name}: ssm_a holds -exp(A_log) but {int((ssm >= 0).sum())} entries "
            "are >= 0; cannot derive A_log"
        )
    return torch.log(-ssm)


def _fuse_kv_b(k_b: "GgufTensor", v_b: "GgufTensor") -> torch.Tensor:
    """kv_b_proj.weight from the two 3D Q8_0 tensors - the #1 silent-corruption risk.

    Axis convention (unit-verifiable against an HF checkpoint): attn_k_b reads
    torch (64, 512, 256) = [head, kv_lora, qk_head], so it needs the last-two-axes
    transpose to reach [head, qk_head, kv_lora]; attn_v_b reads (64, 256, 512) =
    [head, v_head, kv_lora] as-is. The consumer views the result as
    (64, 512, 512) and takes w[:, :256] = w_uk, w[:, 256:] = w_uv
    (attention.py:144-150), i.e. rows per head are [k(256); v(256)] over kv_lora -
    so the cat order is k then v on dim 1. The transpose crosses the Q8_0 pack axis
    (k_b packs qk_head, the target packs kv_lora), so both pieces dequantize to
    bf16; a packed fusion is impossible here.
    """
    k = _cast(k_b, torch.bfloat16).transpose(1, 2)
    v = _cast(v_b, torch.bfloat16)
    heads, k_rows, kv_lora = k.shape
    return torch.cat([k, v], dim=1).reshape(heads * (k_rows + v.shape[1]), kv_lora)


def _require_tp1(what: str) -> None:
    from freetoken.distributed import get_tp_info

    if get_tp_info().size > 1:
        # same status as the HF reader: the loader emits full fused tensors
        raise NotImplementedError(f"glm5next GGUF {what} supports TP=1 only")


def iter_gguf_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
    include_vision: bool = False,
    mmproj_path: str | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield (param_name, tensor) for every non-expert glm5next param.

    Q8_0/Q6_K projections yield packed ``.qweight`` uint8 [rows, row_bytes] for the
    Phase 4 native-quant ops (gemma4 convention); rows named "bf16"/"fp32" in the
    maps yield dense casts instead. Fused targets (in_proj, conv1d, kv_b_proj) emit
    once all their pieces are seen; anything without a map entry raises ValueError
    naming the tensor, and blk.45 (MTP/NextN) is skipped entirely.

    include_vision defaults False: a bare .gguf carries no vision tensors, and the
    text-only spec's loader call passes no include_vision at all. load_weight passes
    True together with the mmproj path once the family registers encoders and the
    boot carries --mmproj; the visual.* tensors then stream after the text trunk.
    """
    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.utils import cached_load_hf_config

    # caller contract (same as the HF reader): routed experts stream via
    # iter_gguf_expert_sources; ValueError so the guards survive python -O.
    if include_moe_experts:
        raise ValueError("glm5next GGUF: include_moe_experts=True is not supported")
    if not include_non_moe:
        raise ValueError("glm5next GGUF: include_non_moe=False is not supported")
    _require_tp1("weight loading")
    config = parse_gguf_config(cached_load_hf_config(model_path))
    num_layers = config.num_layers
    first_dense = config.first_k_dense_replace
    kda_layers = set(config.glm5_args.kda_layer_ids)

    in_proj_buf: dict[int, dict[str, torch.Tensor]] = {}
    conv_buf: dict[int, dict[str, "GgufTensor"]] = {}
    kv_b_buf: dict[int, dict[str, "GgufTensor"]] = {}

    for t in iter_gguf_tensors(model_path):
        name = t.name
        if name == "token_embd.weight":
            if t.ggml_type != GGML_Q8_0:
                # the convert swap builds the embedding as Q8_0; a variant dense type
                # would otherwise die as a late loader shape assert.
                raise ValueError(
                    f"{name} is {GGML_NAME.get(t.ggml_type, t.ggml_type)}; "
                    "the convert swap builds the embedding as Q8_0"
                )
            pn = "model.embed_tokens.qweight"
            yield pn, _logged(pn, t.packed(), t)  # Q8_0 packed table
            continue
        if name == "output.weight":
            if t.ggml_type != GGML_Q6_K:
                # the untied head swaps to a Q6_K GGUF head (convert contract).
                raise ValueError(
                    f"{name} is {GGML_NAME.get(t.ggml_type, t.ggml_type)}; "
                    "the convert swap builds the untied head as Q6_K"
                )
            pn = "lm_head.qweight"
            yield pn, _logged(pn, t.packed(), t)  # Q6_K packed table
            continue
        if name == "output_norm.weight":
            pn = "model.norm.weight"
            yield pn, _logged(pn, _cast(t, torch.bfloat16), t)
            continue
        if not name.startswith("blk."):
            raise ValueError(f"unmapped glm5next GGUF tensor: {name}")
        layer = int(name.split(".")[1])
        if layer >= num_layers:
            # blk.45 is the NextN/MTP draft block + its .nextn. glue: no MTP support,
            # same ignore verdict as the HF path (weight.py never reads that layer).
            logger.debug("glm5next GGUF: skipping MTP tensor %s", name)
            continue
        suffix = name.split(".", 2)[2]
        if suffix in _EXPERT_BANK_ROLES:
            continue  # routed banks -> iter_gguf_expert_sources
        base = f"model.layers.{layer}"

        slot = _IN_PROJ_SLOTS.get(suffix)
        if slot is not None:
            in_proj_buf.setdefault(layer, {})[slot] = t.packed()
        elif suffix in _CONV_SLOTS:
            conv_buf.setdefault(layer, {})[_CONV_SLOTS[suffix]] = t
        elif suffix in _KV_B_SLOTS:
            kv_b_buf.setdefault(layer, {})[_KV_B_SLOTS[suffix]] = t
        elif suffix == "ssm_a":
            pn = f"{base}.self_attn.A_log"
            yield pn, _logged(pn, _a_log(t, name), note="derived from ssm_a fp32")
        elif suffix == "ssm_dt.bias":
            # dt ships in a .bias tensor but is a plain fp32 param, no Linear behind it
            pn = f"{base}.self_attn.dt_bias"
            yield pn, _logged(pn, _cast(t, torch.float32), t)
        else:
            table = _KDA_LAYER_MAP if layer in kda_layers else _DSA_LAYER_MAP
            entry = table.get(suffix) or _COMMON_LAYER_MAP.get(suffix)
            if entry is None:
                entry = (
                    _DENSE_FFN_MAP.get(suffix)
                    if layer < first_dense
                    else _MOE_MAP.get(suffix)
                )
            if entry is None:
                raise ValueError(f"unmapped glm5next GGUF tensor: {name}")
            rel, cast = entry
            if cast == "qw":
                if t.ggml_type != GGML_Q8_0:
                    # every map-branch linear swaps to a Q8_0 GGUFLinear; a K/iq-quant
                    # dense row would otherwise die as a late loader shape assert.
                    raise ValueError(
                        f"{name} is {GGML_NAME.get(t.ggml_type, t.ggml_type)}; "
                        "the convert swap builds packed linears as Q8_0"
                    )
                pn = f"{base}.{rel[: -len('.weight')]}.qweight"
                yield pn, _logged(pn, t.packed(), t)
            else:
                pn = f"{base}.{rel}"
                yield pn, _logged(
                    pn,
                    _cast(t, torch.bfloat16 if cast == "bf16" else torch.float32),
                    t,
                )

        # Emit fused targets once all pieces are present (gemma4 buffer pattern).
        slots = in_proj_buf.get(layer)
        if slots is not None and len(slots) == len(_IN_PROJ_SLOT_ORDER):
            widths = {s: p.shape[1] for s, p in slots.items()}
            if len(set(widths.values())) > 1:
                # packed cat needs one row width; a mixed-type variant would otherwise
                # die inside torch.cat with a tensor-less error.
                raise ValueError(
                    f"glm5next GGUF: layer {layer} in_proj pieces mix row_bytes {widths}"
                )
            # packed concat is exact: every piece has ne0 = 4096 -> same row_bytes
            pn = f"{base}.self_attn.in_proj.qweight"
            yield pn, _logged(
                pn,
                torch.cat([slots[s] for s in _IN_PROJ_SLOT_ORDER], dim=0),
                note="fused in_proj q|k|v|b|f_a|g_a packed",
            )
            del in_proj_buf[layer]
        cbuf = conv_buf.get(layer)
        if cbuf is not None and len(cbuf) == len(_CONV_SLOT_ORDER):
            pn = f"{base}.self_attn.conv1d.weight"
            yield pn, _logged(
                pn,
                torch.cat(
                    [_cast(cbuf[s], torch.bfloat16) for s in _CONV_SLOT_ORDER], dim=0
                ),
                note="fused conv1d q|k|v channel cat, bf16",
            )
            del conv_buf[layer]
        kbuf = kv_b_buf.get(layer)
        if kbuf is not None and len(kbuf) == len(_KV_B_SLOTS):
            pn = f"{base}.self_attn.kv_b_proj.weight"
            yield pn, _logged(
                pn,
                _fuse_kv_b(kbuf["k"], kbuf["v"]),
                note="fused kv_b_proj bf16 (k last-two-axes transpose + v)",
            )
            del kv_b_buf[layer]

    for buf_name, buf in (
        ("in_proj", in_proj_buf),
        ("conv1d", conv_buf),
        ("kv_b", kv_b_buf),
    ):
        if buf:
            # ValueError, not assert: under python -O a bare assert would vanish and
            # the loader would silently consume a partial param set.
            raise ValueError(
                f"glm5next GGUF: incomplete {buf_name} groups "
                f"{sorted((l, sorted(s)) for l, s in buf.items())}"
            )

    if include_vision:
        if mmproj_path is None:
            raise ValueError(
                "glm5next GGUF: include_vision=True requires the --mmproj path; "
                "the .gguf file itself carries no vision tensors"
            )
        yield from iter_gguf_vision_weights(mmproj_path)


def iter_gguf_expert_sources(
    model_path: str, config: "ModelConfig"
) -> Iterator[tuple[int, dict[str, "GgufTensor"]]]:
    """Stream the stacked routed-expert banks for the offload loader (Phase 5 hook).

    One yield per MoE trunk layer: (layer, {"gate"/"up"/"down": GgufTensor}) with
    the packed block bytes untouched and the ggml type carried on the tensor (this
    file mixes IQ3_XXS / IQ4_XS gate+up and IQ4_XS / Q6_K down per layer; the bank
    loader must dispatch per layer). Bank geometry: gate/up torch (288, 2048, 4096),
    down (288, 4096, 2048); packed rows run ffn-major within an expert, so a dim-0
    slice IS one expert's contiguous packed rows - consumers slice, no repack.
    Yields CHECKPOINT layer ids (first_k_dense_replace .. num_layers-1), not bank
    indices - the Phase 5 bank loader offsets via bank_layer_of
    (moe/expert_pieces.py:27-31).
    """
    from freetoken.models.gguf.reader import iter_gguf_tensors

    first_dense, num_layers = config.first_k_dense_replace, config.num_layers
    bufs: dict[int, dict[str, "GgufTensor"]] = {}
    for t in iter_gguf_tensors(model_path):
        if not t.name.startswith("blk."):
            continue
        role = _EXPERT_BANK_ROLES.get(t.name.split(".", 2)[2])
        if role is None:
            continue
        layer = int(t.name.split(".")[1])
        if layer < first_dense:
            # a bank under leading_dense_block_count is a corrupt file: dropping it
            # silently would desync the bank count the offload cache allocates.
            raise ValueError(
                f"glm5next GGUF: routed-expert bank {t.name} on dense layer "
                f"{layer} (< leading_dense_block_count {first_dense})"
            )
        if layer >= num_layers:
            continue  # blk.45 MTP banks: ignored with the rest of the draft block
        slots = bufs.setdefault(layer, {})
        slots[role] = t
        if len(slots) == len(_EXPERT_BANK_ROLES):
            yield layer, slots
            del bufs[layer]
    if bufs:
        raise ValueError(
            f"glm5next GGUF: incomplete expert bank groups "
            f"{sorted((l, sorted(s)) for l, s in bufs.items())}"
        )


# --------------------------------------------------------------------------------------
# Model layer swap: packed Q8_0/Q6_K yields -> native GGUF-quant ops (gemma4 pattern).
# --------------------------------------------------------------------------------------


def is_gguf_model(config: "ModelConfig") -> bool:
    """True when the model was parsed from a GGUF checkpoint (native-quant path)."""
    return getattr(config, "moe_weight_format", None) == "gguf"


class GGUFLMHead(BaseOP):
    """Untied LM head over the native Q6_K output table.

    This checkpoint ships output.weight, so unlike gemma4 there is no tied-head
    path; the prefill last-token slicing mirrors ParallelLMHead.forward. TP=1 only.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, quant_type: int):
        self._quant_type = quant_type
        self.qweight = torch.empty(
            num_embeddings, row_bytes(embedding_dim, quant_type), dtype=torch.uint8
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.core import get_global_ctx
        from freetoken.layers.gguf import fused_mul_mat_gguf

        batch = get_global_ctx().batch
        if batch.is_prefill:
            indices = batch.attn_metadata.get_last_indices(batch.size)
            x = x[indices].contiguous()
        return fused_mul_mat_gguf(x, self.qweight, self._quant_type)


def convert_glm5_next_to_gguf(model, config: "ModelConfig") -> None:
    """In place: swap the packed Q8_0 projections, the Q8_0 embedding and the Q6_K
    untied head for the native GGUF ops.

    Only the .qweight-yielded params swap; the bf16/fp32 outputs (kv_b_proj, conv1d,
    hc_*, indexer, router, e_score_correction_bias) keep the normal path, and the
    routed experts are untouched (offload banks, Phase 5).
    """
    if config.tie_word_embeddings:
        # the iterator yields lm_head.qweight only when output.weight ships; a tied
        # variant would die late on that missing key. GGUFTiedLMHead (gemma4) is the
        # future path once a tied glm5next gguf exists.
        raise ValueError(
            "glm5next gguf serving requires the untied file (output.weight present); "
            "the tied variant has no lm_head path yet (GGUFTiedLMHead is the future "
            "analog)"
        )
    from freetoken.layers.gguf import GGUFEmbedding, GGUFLinear

    inner = model.model
    inner.embed_tokens = GGUFEmbedding(config.vocab_size, config.hidden_size, GGML_Q8_0)

    def swap(owner, attr):
        lin = getattr(owner, attr)
        out_features, in_features = lin.weight.shape
        setattr(
            owner,
            attr,
            GGUFLinear(in_features, out_features, GGML_Q8_0, has_bias=lin.bias is not None),
        )

    for layer in inner.layers.op_list:
        attn = layer.self_attn
        if hasattr(attn, "in_proj"):  # KDA layer
            for attr in ("in_proj", "f_b_proj", "g_b_proj", "o_proj"):
                swap(attn, attr)
        else:  # DSA/MLA layer: kv_b_proj stays dense (the MLA absorption needs it unquantized)
            for attr in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj"):
                swap(attn, attr)
        mlp = layer.mlp
        owner, attrs = (
            (mlp, ("gate_proj", "up_proj", "down_proj"))
            if hasattr(mlp, "gate_proj")
            else (mlp.shared_experts, ("gate_proj", "up_proj", "down_proj"))
        )
        for attr in attrs:
            swap(owner, attr)

    model.lm_head = GGUFLMHead(config.vocab_size, config.hidden_size, GGML_Q6_K)


def load_gguf_expert_sources(model_path: str, model_config, *, layer_sink=None):
    """Offload-bank hook (weight.py resolves it by name): per MoE BANK layer, the
    three stacked bank tensors as packed uint8 views ([E, rows, row_bytes], bytes
    and ggml type verbatim - no dequant, no repack; a dim-0 slice is one expert's
    contiguous rows) plus the per-layer (gate, up, down) ggml type ints, built in
    ROLE order - never the file's tensor encounter order (the offload dispatch
    unpacks (gate, up, down)). Checkpoint layer ids offset to bank indices via
    first_k_dense_replace (bank_layer_of contract).

    Banks are materialized into pinned per-layer HostBanks (q4_0 parity): a
    device_ptr on the file-backed mmap VA is an illegal access at the first copy,
    and cudaHostRegister cannot take the GGUF mmap's 32B alignment - so each bank
    is a one-time copy into its own allocation, pinned as its layer completes.
    ``layer_sink`` (converter) fires per completed layer instead; nothing is
    pinned and the sink owns the banks from then on.
    """
    from freetoken.layers.gguf import _MMVQ
    from freetoken.moe.host_banks import HostBank, LayerCompletionTracker, PinPipeline

    first = model_config.first_k_dense_replace
    num_moe = model_config.num_moe_layers
    n_expert = model_config.num_experts
    roles = ("gate", "up", "down")

    # First pass: geometry + type validation, loud, before any allocation. Rows are
    # PER ROLE - down packs H output rows while gate/up pack I, so a single
    # gate-derived row count reshapes the down bank wrong (review B1).
    specs: dict[int, dict[str, tuple[int, int]]] = {}
    for layer, slots in iter_gguf_expert_sources(model_path, model_config):
        bank = layer - first
        if not 0 <= bank < num_moe:
            raise ValueError(f"gguf expert bank for checkpoint layer {layer} outside the MoE trunk")
        if set(slots) != set(roles):
            raise ValueError(
                f"gguf expert bank for checkpoint layer {layer}: roles "
                f"{sorted(slots)} != {sorted(roles)}"
            )
        per = {}
        for role in roles:
            t = slots[role]
            gt = int(t.ggml_type)
            if gt not in _MMVQ:
                raise ValueError(
                    f"gguf expert bank {t.name!r}: ggml type {gt} has no MMVQ kernel "
                    f"(supported: {sorted(_MMVQ)}); the moe_vec switch would return zeros"
                )
            # ggml ne0 = the linear INPUT dim (drives the row width), ne1 = OUTPUT
            # rows; packed() is the raw byte view, so the width comes from ne0 +
            # BLOCK_SHAPE (ggml_type).
            per[role] = (int(t.shape[1]), row_bytes(int(t.shape[-1]), gt))
        specs[bank] = per
    missing = [i for i in range(num_moe) if i not in specs]
    if missing:
        raise ValueError(f"gguf expert banks missing for bank layers {missing}")

    hb = {
        role: [HostBank((n_expert, *specs[bank][role]), torch.uint8) for bank in range(num_moe)]
        for role in roles
    }
    banks = {role: [b.tensor for b in per] for role, per in hb.items()}
    types: list = [None] * num_moe

    def _fill(sink) -> None:
        # One pass over the file; a layer completes when all three of its banks have
        # landed, which is when the tracker hands it to the PinPipeline (serving) or
        # the converter sink.
        # ONE completion event per layer: all three of its banks are copied together
        # above, so the tracker fires the pin immediately after that single note.
        # (expected_per_layer must MATCH the note count - gemma4 notes per write, 2x
        # per layer; a mismatch means on_layer never fires and NOTHING gets pinned,
        # which surfaces as an IMA at the first decode copy, not at load.)
        tracker = LayerCompletionTracker(1, hb, sink) if sink is not None else None
        for layer, slots in iter_gguf_expert_sources(model_path, model_config):
            bank = layer - first
            for role in roles:
                t = slots[role]
                rows, rb = specs[bank][role]
                banks[role][bank].copy_(t.packed().reshape(n_expert, rows, rb))
            types[bank] = tuple(int(slots[r].ggml_type) for r in roles)
            if tracker is not None:
                tracker.note(bank)

    if layer_sink is not None:
        _fill(layer_sink)
    elif torch.cuda.is_available():
        with PinPipeline() as pins:
            _fill(pins)
    else:
        _fill(None)  # CUDA-less host: banks stay pageable (no device copies happen)
    return banks, tuple(types)


__all__ = [
    "parse_gguf_config",
    "iter_gguf_weights",
    "iter_gguf_expert_sources",
    "load_gguf_expert_sources",
    "is_gguf_model",
    "convert_glm5_next_to_gguf",
]
