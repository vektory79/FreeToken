"""GLM-5.3-Flash (``glm5next``) GGUF adapter: build the FreeToken ``ModelConfig``
from GGUF metadata.

The GGUF checkpoint's geometry is identical to the HF glm5_next model (hybrid
34xKDA + 11xDSA decoder, 288-expert sigmoid/noaux_tc MoE, mHC residual streams),
so this translates the ``glm5next.*`` metadata keys into a text-config view and
delegates to ``glm5_next.config.parse_config`` -- the same ``ModelConfig`` /
``Glm5NextArgs`` payload the HF path produces, sourced from GGUF KV metadata
instead of a HF config object. Weight iteration translates the gguf tensor names
to the params ``models/glm5_next/weight.py`` yields for HF checkpoints (fused
KDA in_proj / conv1d, fused DSA kv_b_proj, derived A_log); quantized linears
yield their packed ggml rows under the GGUF segment naming, and routed-expert
banks load separately with their ggml type intact (``moe/expert_banks`` ->
``models/gguf/experts``).
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Iterator

import numpy as np
import torch

from freetoken.layers.quantization.ggml_types import float_dtype, row_bytes, type_name
from freetoken.models.gguf.reader import GgufTensorInfo, gguf_tensor_index
from freetoken.models.gguf.tensors import float_tensor, fused, matrix
from freetoken.models.glm5_next.args import DSA_LAYER, KDA_LAYER
from freetoken.third_party import gguf as ggml

# the two block types the convert contract pins (embedding table / untied head)
_GGML_Q8_0 = int(ggml.GGMLQuantizationType.Q8_0)
_GGML_Q6_K = int(ggml.GGMLQuantizationType.Q6_K)

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig
    from freetoken.models.gguf.config import GgufConfigShim
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

# Image-serving pins (the NVFP4 checkpoint's processor_config.json + config.json):
# the token-budget defaults and the image placeholder id. The id itself is derived
# from the GGUF vocab and must land on the pin - a reordered vocab is a different
# checkpoint than the one the vision tower was verified against.
_IMAGE_TOKEN = "<|image|>"
_GLM5NEXT_IMAGE_TOKEN_ID = 154854
_IMAGE_PROCESSOR_PINNED = {"min_image_tokens": 16, "max_image_tokens": 8000}


def resolve_gguf_image_serving(model_path: str, metadata: dict) -> int:
    """image_token_id for a multimodal GGUF boot, or ValueError naming the gap.

    The GGUF tokenizer ships no template of its own beyond the metadata KV; image
    input needs a template that renders the image placeholder wrapper and a vocab
    that maps the placeholder string to the pinned token id.
    """
    template = metadata.get("tokenizer.chat_template")
    if not template:
        raise ValueError(
            f"GGUF {model_path}: no tokenizer.chat_template metadata; a chat template "
            "is required for image input"
        )
    if _IMAGE_TOKEN not in template:
        raise ValueError(
            f"GGUF {model_path}: the chat template renders no {_IMAGE_TOKEN!r} image "
            "placeholder; image prompts would silently lose the image"
        )
    tokens = metadata.get("tokenizer.ggml.tokens")
    if tokens is None:
        raise ValueError(
            f"GGUF {model_path}: no tokenizer.ggml.tokens metadata; cannot resolve "
            "the image token id"
        )
    try:
        token_id = list(tokens).index(_IMAGE_TOKEN)
    except ValueError:
        raise ValueError(
            f"GGUF {model_path}: the vocab carries no {_IMAGE_TOKEN!r} token"
        ) from None
    if token_id != _GLM5NEXT_IMAGE_TOKEN_ID:
        raise ValueError(
            f"GGUF {model_path}: {_IMAGE_TOKEN!r} sits at vocab id {token_id} != the "
            f"GLM-5.3-Flash reference {_GLM5NEXT_IMAGE_TOKEN_ID}; a different vocab "
            "layout invalidates the pinned image token id"
        )
    return token_id


def vision_config_from_tower_metadata(m: dict, shapes: dict, *, source: str) -> "VisionConfig":
    """VisionConfig for the glm5next tower from clip.* tower metadata.

    `source` names the carrier for error messages (the mmproj file, or an FTW's
    merged source_metadata.gguf). `shapes` maps tower tensor names to ggml-order
    shapes for the header-only cross checks; empty for a metadata-only carrier,
    which has no tensor table to check against.
    """
    from freetoken.models.glm5_next.config import VisionConfig

    def c(key: str):
        val = m.get(f"clip.{key}")
        if val is None:
            raise ValueError(f"mmproj {source}: missing required metadata key clip.{key}")
        return val

    if not m.get("clip.has_vision_encoder"):
        raise ValueError(
            f"mmproj {source}: clip.has_vision_encoder is not True; the file does "
            "not carry a vision tower"
        )
    projector = c("projector_type")
    if str(projector) != "glm5next":
        raise ValueError(
            f"mmproj {source}: clip.projector_type {projector!r} != 'glm5next'; "
            "FreeToken implements only the glm5next projector"
        )

    def pinned_int(key: str, field: str) -> int:
        val = int(c(key))
        expected = _MMPROJ_PINNED[field]
        if val != expected:
            raise ValueError(
                f"mmproj {source}: clip.{key} {val} != the GLM-5.3-Flash vision "
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
            f"mmproj {source}: clip.vision.attention.layer_norm_epsilon {eps} != "
            f"the GLM-5.3-Flash vision reference {_MMPROJ_PINNED['rms_norm_eps']}"
        )
    swiglu = float(c("vision.swiglu_limit"))
    if not math.isclose(swiglu, _MMPROJ_PINNED["swiglu_limit"], rel_tol=1e-6):
        raise ValueError(
            f"mmproj {source}: clip.vision.swiglu_limit {swiglu} != the "
            f"GLM-5.3-Flash vision reference {_MMPROJ_PINNED['swiglu_limit']}"
        )

    # tensor-table cross checks: header shapes only, the weight data is never read
    temporal = _MMPROJ_PINNED["temporal_patch_size"]
    in_channels = _MMPROJ_PINNED["in_channels"]
    proj_inter = _MMPROJ_PINNED["projection_intermediate_size"]
    patch_slices = [
        n for n in shapes if n == "v.patch_embd.weight" or n.startswith("v.patch_embd.weight.")
    ]
    if patch_slices:
        # the Conv3d patch embedding ships one slice per temporal frame; the base
        # tensor must be present, it is the slice the input channels are read from
        if "v.patch_embd.weight" not in shapes:
            raise ValueError(
                f"mmproj {source}: v.patch_embd.weight.N slices without the base "
                "v.patch_embd.weight tensor"
            )
        if len(patch_slices) != temporal:
            raise ValueError(
                f"mmproj {source}: {len(patch_slices)} v.patch_embd.weight slices "
                f"!= temporal_patch_size {temporal}"
            )
        found_in = shapes["v.patch_embd.weight"][1]
        if found_in != in_channels:
            raise ValueError(
                f"mmproj {source}: v.patch_embd.weight has {found_in} input "
                f"channels != the GLM-5.3-Flash vision reference {in_channels}"
            )
    gate = shapes.get("mm.gate.weight")
    if gate is not None and gate[0] != proj_inter:
        raise ValueError(
            f"mmproj {source}: mm.gate.weight output width {gate[0]} != the "
            f"GLM-5.3-Flash vision reference {proj_inter} (projection_intermediate_size)"
        )
    # a metadata-only FTW carrier has no tensor table: the numeric control is the phase verification reports
    if shapes and _MMPROJ_PINNED["attention_bias"] and "v.blk.0.attn_qkv.bias" not in shapes:
        raise ValueError(
            f"mmproj {source}: no v.blk.*.attn_qkv.bias tensors; the GLM-5.3-Flash "
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


def image_processor_config_from_tower_metadata(m: dict, shapes: dict, *, source: str) -> dict:
    """Glm5NextImageProcessor kwargs from clip.* tower metadata.

    Same carrier contract as vision_config_from_tower_metadata: `shapes` feeds the
    temporal-slice check and stays empty for a metadata-only FTW carrier.
    """

    def c(key: str):
        val = m.get(f"clip.{key}")
        if val is None:
            raise ValueError(f"mmproj {source}: missing required metadata key clip.{key}")
        return val

    patch = int(c("vision.patch_size"))
    image_size = int(c("vision.image_size"))
    merge = int(c("vision.spatial_merge_size"))
    if image_size % patch != 0:
        raise ValueError(
            f"mmproj {source}: clip.vision.image_size {image_size} is not a "
            f"multiple of clip.vision.patch_size {patch}"
        )
    mean = [float(x) for x in c("vision.image_mean")]
    std = [float(x) for x in c("vision.image_std")]
    if len(mean) != 3 or len(std) != 3:
        raise ValueError(
            f"mmproj {source}: clip.vision.image_mean/std must carry one entry "
            f"per RGB channel, got {len(mean)}/{len(std)}"
        )
    temporal = _MMPROJ_PINNED["temporal_patch_size"]
    slices = [
        n
        for n in shapes
        if n == "v.patch_embd.weight" or n.startswith("v.patch_embd.weight.")
    ]
    if slices and len(slices) != temporal:
        raise ValueError(
            f"mmproj {source}: {len(slices)} v.patch_embd.weight slices != "
            f"temporal_patch_size {temporal}"
        )
    return {
        "patch_size": patch,
        "merge_size": merge,
        "temporal_patch_size": temporal,
        "image_mean": mean,
        "image_std": std,
        **_IMAGE_PROCESSOR_PINNED,
    }


def parse_mmproj_vision_config(mmproj_path: str) -> "VisionConfig":
    """VisionConfig from a clip-architecture mmproj GGUF file (the --mmproj boot path)."""
    from freetoken.models.gguf.reader import _reader, load_gguf_metadata

    return vision_config_from_tower_metadata(
        load_gguf_metadata(mmproj_path),
        {t.name: tuple(reversed([int(d) for d in t.shape])) for t in _reader(mmproj_path).tensors},
        source=mmproj_path,
    )


def parse_mmproj_image_processor_config(mmproj_path: str) -> dict:
    """Image-processor kwargs from a clip-architecture mmproj GGUF file."""
    from freetoken.models.gguf.reader import _reader, load_gguf_metadata

    return image_processor_config_from_tower_metadata(
        load_gguf_metadata(mmproj_path),
        {t.name: tuple(reversed([int(d) for d in t.shape])) for t in _reader(mmproj_path).tensors},
        source=mmproj_path,
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
    from freetoken.models.gguf.reader import gguf_architecture, gguf_tensor_index, load_gguf_metadata

    # device is accepted for reader-signature parity with the HF iter_vision_weights;
    # the loader materializes onto the model params from these CPU tensors
    arch = gguf_architecture(mmproj_path)
    if arch != "clip":
        # fail here too: the reader is callable standalone, not only under build_gguf_shim
        raise ValueError(
            f"mmproj {mmproj_path}: architecture {arch!r} != 'clip'; an mmproj file "
            "carries the vision tower"
        )
    depth = int(load_gguf_metadata(mmproj_path)["clip.vision.block_count"])
    blocks: set[int] = set()
    patch: dict[int, tuple[str, torch.Tensor]] = {}
    for t in gguf_tensor_index(mmproj_path).values():
        name = t.name
        tensor = _cast(t, torch.bfloat16)
        if name.startswith("v.blk."):
            parts = name.split(".")
            layer, suffix = int(parts[2]), ".".join(parts[3:])
            if layer >= depth:
                raise ValueError(
                    f"mmproj {mmproj_path}: vision block {layer} outside "
                    f"clip.vision.block_count {depth}"
                )
            rel = _VISION_BLOCK_MAP.get(suffix)
            if rel is None:
                raise ValueError(f"unmapped mmproj tensor: {name}")
            blocks.add(layer)
            pn = f"visual.blocks.{layer}.{rel}"
            yield pn, _logged(pn, tensor, t)
        elif name == "v.patch_embd.weight" or name.startswith("v.patch_embd.weight."):
            idx = 0 if name == "v.patch_embd.weight" else int(name.rsplit(".", 1)[1])
            if idx in patch:
                # the base tensor and a .0 slice both claim frame 0: an ambiguity
                # stack() would resolve silently and wrongly
                raise ValueError(
                    f"mmproj {mmproj_path}: duplicate patch-embedding slice index {idx} "
                    f"({patch[idx][0]} and {name})"
                )
            patch[idx] = (name, tensor)
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
    weight = torch.stack([patch[i][1] for i in range(temporal)], dim=2)
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
    # the gguf marker the model build keys on: moe_weight_format="gguf" drives the
    # engine's GGUF strategy gates and the expert-bank format dispatch.
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
        # multimodal GGUF boot: the shim precomputed the image placeholder id
        # (fail-fast on a missing template/vocab entry) and carries the tower path
        # the image processor fallback reads its clip.* metadata from
        image_token_id=getattr(shim, "image_token_id", None),
        mmproj_path=getattr(shim, "mmproj_path", None),
    )


# --------------------------------------------------------------------------------------
# Weight loading: GGUF tensor names -> FreeToken glm5_next module params.
# Mapping authority: research/phase2-tensor-map.md (1383 mapped + 29 skip-listed MTP).
# --------------------------------------------------------------------------------------

# Routed-expert stacks: NOT yielded by iter_gguf_weights (mirrors the HF path, where
# routed experts only serve from the offload cache); they load with their ggml type
# intact through moe/expert_banks -> models/gguf/experts. A bank under
# leading_dense_block_count is a corrupt file: models/gguf/experts rejects it loudly
# instead of desyncing the bank count the offload cache allocates.
_EXPERT_BANK_ROLES = {
    "ffn_gate_exps.weight": "gate",
    "ffn_up_exps.weight": "up",
    "ffn_down_exps.weight": "down",
}

# suffix (after "blk.N.") -> (FreeToken param name, cast). "qw" yields the packed
# block bytes under the .qweight name (Phase 4 native-quant path, gemma4 convention);
# "bf16"/"fp32" dequantize in place, matching the dtype weight.py yields for HF
# checkpoints: layer norms + o_norm + indexer norms bf16, A_log / dt_bias / hc_* /
# e_score_correction_bias / ape fp32. hc_*_fn and the indexer linears are Q8_0 in the
# gguf but dense downstream, so they dequantize here. "qw" entries become packed
# GGUF linears (segment ``weight`` params, convert contract Q8_0).
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
# per layer-class rows MINUS the bespoke fusions (in_proj / conv1d / kv_b_proj) and
# the KDA derived params (ssm_a -> A_log, ssm_dt.bias -> dt_bias), which iter emits
# explicitly; the indexer's wq_b is listed bf16 here but ships as a Q8_0 GGUF linear
# on the new stack (it is built with the checkpoint's quant config).
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
    "indexer.attn_q_b.weight": ("self_attn.indexer.wq_b.weight", "qw"),
    "indexer.attn_k.weight": ("self_attn.indexer.wk.weight", "bf16"),
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


def _cuda_target(device: torch.device | str | None) -> torch.device | None:
    """The CUDA device to decode quantized rows on, or None for the numpy path."""
    if device is None or torch.device(device).type != "cuda":
        return None
    return torch.device(device)


def _cast(info: GgufTensorInfo, dtype: torch.dtype, device: torch.device | None = None) -> torch.Tensor:
    """Dense ``dtype`` tensor in torch order from a (possibly quantized) tensor.

    The GPU path decodes with the vendored ggml kernel on ``device`` (the packed
    rows move to CUDA as packed bytes - the kernel takes device tensors, a host
    dequant would not even feed it), the CPU path with the vendored gguf-py numpy
    quants; the two never mix within one tensor (their fp16 scale chains round
    differently in edge cases)."""
    native = float_dtype(info.ggml_type)
    if native is not None:
        return float_tensor(info).to(dtype)
    ne = tuple(int(d) for d in info.ne)
    k, rows = ne[0], int(math.prod(ne[1:]))
    target = _cuda_target(device)
    if target is not None:
        from freetoken.kernel import ggml as ggml_kernel

        flat = ggml_kernel.dequantize(matrix(info).to(target), info.ggml_type, k, dtype)
    else:
        from freetoken.third_party.gguf import quants

        vals = quants.dequantize(
            info.data().reshape(rows, info.nbytes // rows),
            ggml.GGMLQuantizationType(info.ggml_type),
        )
        flat = torch.from_numpy(np.ascontiguousarray(vals)).to(dtype)
    return flat.reshape(tuple(reversed(ne)))


def _logged(
    name: str, tensor: torch.Tensor, info: GgufTensorInfo | None = None, note: str = ""
) -> torch.Tensor:
    """Plan acceptance 'dtype/type per tensor logged': one debug line per yield,
    silent at INFO so serving never pays for it."""
    if info is not None:
        logger.debug(
            "glm5next %s: ggml %s %s -> %s %s",
            name,
            type_name(info.ggml_type),
            tuple(int(d) for d in info.ne),
            tuple(tensor.shape),
            tensor.dtype,
        )
    else:
        logger.debug(
            "glm5next %s: %s -> %s %s", name, note, tuple(tensor.shape), tensor.dtype
        )
    return tensor


def _a_log(info: GgufTensorInfo, name: str, device: torch.device | None = None) -> torch.Tensor:
    # gguf stores -exp(A_log) per head (kimi-k3 convention, glm5next.cpp:7); invert
    # in fp32 - the recurrent kernels read A_log as fp32 (kda.py gate params).
    ssm = _cast(info, torch.float32, device)
    if bool((ssm >= 0).any()):
        raise ValueError(
            f"{name}: ssm_a holds -exp(A_log) but {int((ssm >= 0).sum())} entries "
            "are >= 0; cannot derive A_log"
        )
    return torch.log(-ssm)


def _fuse_kv_b(k_b: GgufTensorInfo, v_b: GgufTensorInfo, device: torch.device | None = None) -> torch.Tensor:
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
    k = _cast(k_b, torch.bfloat16, device).transpose(1, 2)
    v = _cast(v_b, torch.bfloat16, device)
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

    Quantized linears yield packed ggml rows under the GGUF linear's segment
    names (``weight``, ``weight_1``, ... via ``tensors.fused``), filling exactly
    the buffers ``GGUFLinearMethod.create_weights`` allocates; the kv_b_proj /
    conv1d fusions and every bf16/fp32 row yield dense casts (the MLA absorption
    and the depthwise conv read unquantized weights). Routed-expert banks are
    not yielded here: they load through ``moe/expert_banks`` ->
    ``models/gguf/experts``. A tensor without a mapping raises ValueError naming
    the tensor, and blk.{n_layer} (MTP/NextN) is skipped entirely.

    include_vision defaults False: a bare .gguf carries no vision tensors, and the
    text-only spec's loader call passes no include_vision at all. load_weight passes
    True together with the mmproj path once the family registers encoders and the
    boot carries --mmproj; the visual.* tensors then stream after the text trunk.
    """
    from freetoken.utils import cached_load_hf_config

    # caller contract (same as the HF reader): routed experts load as GGUF expert
    # banks; ValueError so the guards survive python -O.
    if include_moe_experts:
        raise ValueError("glm5next GGUF: include_moe_experts=True is not supported")
    if not include_non_moe:
        raise ValueError("glm5next GGUF: include_non_moe=False is not supported")
    _require_tp1("weight loading")
    config = parse_gguf_config(cached_load_hf_config(model_path))
    index = gguf_tensor_index(model_path)
    hidden = config.hidden_size
    num_layers = config.num_layers
    first_dense = config.first_k_dense_replace
    kda_layers = set(config.glm5_args.kda_layer_ids)
    consumed: set[str] = set()

    def need(name: str, what: str) -> GgufTensorInfo:
        info = index.get(name)
        if info is None:
            # ValueError, not assert: under python -O a bare assert would vanish and
            # the loader would silently consume a partial param set.
            raise ValueError(f"glm5next GGUF: {what} is missing tensor {name}")
        consumed.add(name)
        return info

    def packed_linear(prefix: str, info: GgufTensorInfo):
        # every map-branch linear is a Q8_0 GGUF linear (convert contract); a
        # K/iq-quant dense row would otherwise die as a late loader shape assert
        if info.ggml_type != _GGML_Q8_0:
            raise ValueError(
                f"{info.name} is {type_name(info.ggml_type)}; "
                "the convert swap builds packed linears as Q8_0"
            )
        return fused(prefix, [info], [matrix(info)], int(info.ne[0]))

    def dense(owner: str, param: str, info: GgufTensorInfo, cast: str):
        pn = f"{owner}.{param}"
        return pn, _logged(
            pn, _cast(info, torch.bfloat16 if cast == "bf16" else torch.float32, device), info
        )

    # token_embd / output: the convert contract pins the table types; a variant
    # dense type would otherwise die as a late loader shape assert.
    emb = need("token_embd.weight", "the vocab table")
    if emb.ggml_type != _GGML_Q8_0:
        raise ValueError(
            f"token_embd.weight is {type_name(emb.ggml_type)}; "
            "the convert swap builds the embedding as Q8_0"
        )
    yield from fused("model.embed_tokens", [emb], [matrix(emb)], hidden)
    if config.tie_word_embeddings:
        # the iterator would otherwise die late on the missing lm_head tensors
        raise ValueError(
            "glm5next gguf serving requires the untied file (output.weight present); "
            "the tied variant has no lm_head path yet"
        )
    out = need("output.weight", "the untied head")
    if out.ggml_type != _GGML_Q6_K:
        raise ValueError(
            f"output.weight is {type_name(out.ggml_type)}; "
            "the convert swap builds the untied head as Q6_K"
        )
    yield from fused("lm_head", [out], [matrix(out)], hidden)
    yield dense("model.norm", "weight", need("output_norm.weight", "the final norm"), "bf16")

    for layer in range(num_layers):
        blk, pre = f"blk.{layer}", f"model.layers.{layer}"
        sa = f"{pre}.self_attn"
        for suffix, (param, cast) in _COMMON_LAYER_MAP.items():
            yield dense(pre, param, need(f"{blk}.{suffix}", f"layer {layer}"), cast)
        layer_map = _KDA_LAYER_MAP if layer in kda_layers else _DSA_LAYER_MAP
        if layer in kda_layers:
            # in_proj fusion slots: the concat order q|k|v|b|f_a|g_a is pinned by the
            # consumer, Glm5NextKDA._in_proj_split = [p, p, p, h, d, d] (kda.py) and
            # weight.py's _KDA_IN_PROJ leaf order
            pieces = [
                need(f"{blk}.{s}.weight", f"layer {layer} in_proj")
                for s in ("attn_q", "attn_k", "attn_v", "ssm_beta", "ssm_f_a", "ssm_g_a")
            ]
            for p in pieces:
                if p.ggml_type != _GGML_Q8_0:
                    raise ValueError(
                        f"{p.name} is {type_name(p.ggml_type)}; "
                        "the convert swap builds packed linears as Q8_0"
                    )
            widths = {row_bytes(p.ggml_type, int(p.ne[0])) for p in pieces}
            if len(widths) > 1:
                # the packed cat needs one row width; a mixed-type variant would
                # otherwise die inside torch.cat with a tensor-less error
                raise ValueError(
                    f"glm5next GGUF: layer {layer} in_proj pieces mix row_bytes {widths}"
                )
            # packed concat is exact: every piece has ne0 == hidden -> one row width;
            # the q|k|v|b|f_a|g_a order is pinned by Glm5NextKDA._in_proj_split
            yield from fused(f"{sa}.in_proj", pieces, [matrix(p) for p in pieces], hidden)
            conv = [
                need(f"{blk}.ssm_conv1d_{p}.weight", f"layer {layer} conv1d")
                for p in ("q", "k", "v")
            ]
            pn = f"{sa}.conv1d.weight"
            yield pn, _logged(
                pn,
                torch.cat([_cast(t, torch.bfloat16, device) for t in conv], dim=0),
                note="fused conv1d q|k|v channel cat, bf16",
            )
            pn = f"{sa}.A_log"
            yield pn, _logged(
                pn,
                _a_log(need(f"{blk}.ssm_a", f"layer {layer}"), f"{blk}.ssm_a", device),
                note="derived from ssm_a fp32",
            )
            # dt ships in a .bias tensor but is a plain fp32 param, no Linear behind it
            pn = f"{sa}.dt_bias"
            yield pn, _logged(pn, _cast(need(f"{blk}.ssm_dt.bias", f"layer {layer}"), torch.float32, device), note="plain fp32 param")
        else:
            # kv_b_proj is dense by construction (the MLA absorption reads it as a
            # bmm operand); _fuse_kv_b owns the #1 silent-corruption axis convention
            pn = f"{sa}.kv_b_proj.weight"
            yield pn, _logged(
                pn,
                _fuse_kv_b(
                    need(f"{blk}.attn_k_b.weight", f"layer {layer} kv_b"),
                    need(f"{blk}.attn_v_b.weight", f"layer {layer} kv_b"),
                    device,
                ),
                note="fused kv_b_proj bf16 (k last-two-axes transpose + v)",
            )
        for suffix, (param, cast) in layer_map.items():
            owner, leaf = param.rpartition(".")[0], param.rpartition(".")[2]
            if cast == "qw":
                yield from packed_linear(
                    f"{pre}.{owner}",
                    need(f"{blk}.{suffix}", f"layer {layer}"),
                )
            else:
                yield dense(f"{pre}.{owner}", leaf, need(f"{blk}.{suffix}", f"layer {layer}"), cast)
        ffn_map = _DENSE_FFN_MAP if layer < first_dense else _MOE_MAP
        for suffix, (param, cast) in ffn_map.items():
            owner, leaf = param.rpartition(".")[0], param.rpartition(".")[2]
            if cast == "qw":
                yield from packed_linear(
                    f"{pre}.{owner}",
                    need(f"{blk}.{suffix}", f"layer {layer}"),
                )
            else:
                yield dense(f"{pre}.{owner}", leaf, need(f"{blk}.{suffix}", f"layer {layer}"), cast)

    # anything the trunk never asked for is a corrupt or unknown file: name it
    skipped = 0
    for name in sorted(set(index) - consumed):
        if name.startswith("blk."):
            head, sep, suffix = name[len("blk."):].partition(".")
            if sep and head.isdigit():
                if int(head) >= num_layers:
                    # blk.45 is the NextN/MTP draft block + its .nextn. glue: no MTP
                    # support, same ignore verdict as the HF path (weight.py never
                    # reads that layer)
                    skipped += 1
                    continue
                if suffix in _EXPERT_BANK_ROLES:
                    continue  # routed banks -> moe/expert_banks -> models/gguf/experts
        raise ValueError(f"unmapped glm5next GGUF tensor: {name}")
    if skipped:
        logger.debug("glm5next GGUF: skipped %d MTP/draft-block tensors", skipped)

    if include_vision:
        if mmproj_path is None:
            raise ValueError(
                "glm5next GGUF: include_vision=True requires the --mmproj path; "
                "the .gguf file itself carries no vision tensors"
            )
        yield from iter_gguf_vision_weights(mmproj_path)


# --------------------------------------------------------------------------------------
# Quant-config adapter: checkpoint module names -> the GGUF tensors that store them.
# The registry resolves to_gguf per family (register.checkpoint_quant_config) and the
# GGUFConfig scheme lookup probes it for every quantized module; the dense kv_b / conv
# fusions map their multi-tensor sets too, but they are unquantized modules, so the
# one-tensor-per-part rule never fires on them.
# --------------------------------------------------------------------------------------

_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_TOP_LEAVES = {
    "model.embed_tokens": "token_embd",
    "lm_head": "output",
    "model.norm": "output_norm",
}
# module leaf (after "model.layers.N.") -> the blk.N tensor-name suffix(es) that store it
_LEAVES: dict[str, tuple[str, ...]] = {
    "input_layernorm": ("attn_norm",),
    "post_attention_layernorm": ("ffn_norm",),
    "hc_attn_fn": ("hc_attn_fn",),
    "hc_attn_base": ("hc_attn_base",),
    "hc_attn_scale": ("hc_attn_scale",),
    "hc_ffn_fn": ("hc_ffn_fn",),
    "hc_ffn_base": ("hc_ffn_base",),
    "hc_ffn_scale": ("hc_ffn_scale",),
    # KDA in_proj pieces (the packed map splits the fused module back to these)
    "self_attn.q_proj": ("attn_q",),
    "self_attn.k_proj": ("attn_k",),
    "self_attn.v_proj": ("attn_v",),
    "self_attn.b_proj": ("ssm_beta",),
    "self_attn.f_a_proj": ("ssm_f_a",),
    "self_attn.g_a_proj": ("ssm_g_a",),
    "self_attn.f_b_proj": ("ssm_f_b",),
    "self_attn.g_b_proj": ("ssm_g_b",),
    "self_attn.o_proj": ("attn_output",),
    "self_attn.o_norm": ("ssm_norm",),
    "self_attn.A_log": ("ssm_a",),
    "self_attn.dt_bias": ("ssm_dt.bias",),
    "self_attn.conv1d": ("ssm_conv1d_q", "ssm_conv1d_k", "ssm_conv1d_v"),
    # DSA / MLA
    "self_attn.q_a_proj": ("attn_q_a",),
    "self_attn.q_a_layernorm": ("attn_q_a_norm",),
    "self_attn.q_b_proj": ("attn_q_b",),
    "self_attn.kv_a_proj_with_mqa": ("attn_kv_a_mqa",),
    "self_attn.kv_a_layernorm": ("attn_kv_a_norm",),
    "self_attn.kv_b_proj": ("attn_k_b", "attn_v_b"),
    "self_attn.indexer.wk": ("indexer.attn_k",),
    "self_attn.indexer.wq_b": ("indexer.attn_q_b",),
    "self_attn.indexer.weights_proj": ("indexer.proj",),
    "self_attn.indexer.k_norm": ("indexer.k_norm",),
    "self_attn.indexer.k_norm.bias": ("indexer.k_norm.bias",),
    "self_attn.indexer.index_kpool_compress_gate": ("indexer_compressor_gate",),
    "self_attn.indexer.index_kpool_compress_ape": ("indexer_compressor_ape",),
    # FFN: dense-lead MLP
    "mlp.gate_proj": ("ffn_gate",),
    "mlp.up_proj": ("ffn_up",),
    "mlp.down_proj": ("ffn_down",),
    # MoE trunk: router, correction bias, shared expert, routed-expert stacks
    "mlp.gate": ("ffn_gate_inp",),
    "mlp.e_score_correction_bias": ("exp_probs_b.bias",),
    "mlp.shared_experts.gate_proj": ("ffn_gate_shexp",),
    "mlp.shared_experts.up_proj": ("ffn_up_shexp",),
    "mlp.shared_experts.down_proj": ("ffn_down_shexp",),
    "mlp.experts.0.gate_proj": ("ffn_gate_exps",),
    "mlp.experts.0.up_proj": ("ffn_up_exps",),
    "mlp.experts.0.down_proj": ("ffn_down_exps",),
}


def to_gguf(name: str, tensors: dict[str, int]) -> tuple[str, ...]:
    """Checkpoint module name -> the GGUF tensors (without ``.weight``) that store it; empty for modules the GGUF has no matrix for."""
    if name in _TOP_LEAVES:
        return (_TOP_LEAVES[name],)
    m = _LAYER_RE.match(name)
    if m is None:
        return ()
    mapped = _LEAVES.get(m.group(2))
    if mapped is None:
        return ()
    blk = f"blk.{m.group(1)}"
    return tuple(f"{blk}.{s}" for s in mapped)


def ftw_tensor_types(model_path: str) -> dict[str, int]:
    """The ggml-type table of an FTW converted by an older build, keyed bare.

    Those indexes kept only the per-layer (gate, up, down) expert triples under
    ``gguf_types``; the dense half needs no per-tensor record because the convert
    contract pins it: iter_gguf_weights refuses any other type on the raw path,
    so every convertible file - and therefore every FTW - packs token_embd as
    Q8_0, output as Q6_K and every mapped linear as Q8_0. Bare keys work because
    the scheme probe falls back from ``<tensor>.weight`` to ``<tensor>``."""
    from freetoken.checkpoint.ftw import FTWReader
    from freetoken.models.gguf.reader import gguf_config_source, load_gguf_metadata

    reader = FTWReader(model_path)
    try:
        triples = reader.meta("gguf_types")
    finally:
        reader.close()
    if (
        not isinstance(triples, list)
        or not triples
        or not all(
            isinstance(row, (list, tuple))
            and len(row) == 3
            and all(isinstance(t, int) and not isinstance(t, bool) for t in row)
            for row in triples
        )
    ):
        raise RuntimeError(
            f"{model_path}: FTW converted from a GGUF by an older build records no usable "
            "gguf_types expert triples; re-convert with ft checkpoint"
        )
    # the trunk geometry parse_gguf_config derives (block_count includes the NextN
    # block); read straight from the carrier metadata - the full parse would pull
    # guards (vocab consistency) this rebuild does not need
    meta = load_gguf_metadata(gguf_config_source(model_path))
    num_layers = int(meta["glm5next.block_count"]) - int(meta.get("glm5next.nextn_predict_layers", 0))
    first_dense = int(meta["glm5next.leading_dense_block_count"])
    if first_dense + len(triples) != num_layers:
        raise RuntimeError(
            f"{model_path}: {len(triples)} gguf_types expert triples for "
            f"{num_layers - first_dense} MoE layers; the checkpoint does not match its config"
        )
    # the layer maps are the single source of the per-leaf storage type: "qw"
    # rides the convert contract's Q8_0, "bf16"/"fp32" dequantize to float types
    cast_types = {
        "qw": _GGML_Q8_0,
        "bf16": int(ggml.GGMLQuantizationType.BF16),
        "fp32": int(ggml.GGMLQuantizationType.F32),
    }
    leaves: dict[str, int] = {}
    for layer_map in (_COMMON_LAYER_MAP, _KDA_LAYER_MAP, _DSA_LAYER_MAP, _DENSE_FFN_MAP, _MOE_MAP):
        for suffix, (_param, cast) in layer_map.items():
            kind = cast_types[cast]
            if leaves.setdefault(suffix, kind) != kind:
                raise RuntimeError(f"glm5next convert contract: {suffix} maps to two ggml types")
    # iter emits the KDA derived params (fp32) and the fused in_proj pieces (Q8_0)
    # explicitly, outside the maps
    leaves["ssm_a"] = cast_types["fp32"]
    leaves["ssm_dt.bias"] = cast_types["fp32"]
    # the pre-re-platform loader dequantized the indexer's wq_b (the map flipped
    # it to "qw" only after the merge), so old carriers store bf16 rows; a float
    # entry builds the module unquantized and matches the stored shape
    leaves["indexer.attn_q_b.weight"] = cast_types["bf16"]
    table = {"token_embd": _GGML_Q8_0, "output": _GGML_Q6_K, "output_norm": cast_types["bf16"]}
    for layer in range(num_layers):
        for suffix, kind in leaves.items():
            table[f"blk.{layer}.{suffix}"] = kind
        for suffix in ("attn_q", "attn_k", "attn_v", "ssm_beta", "ssm_f_a", "ssm_g_a"):
            table[f"blk.{layer}.{suffix}"] = _GGML_Q8_0
    # the triples are MoE-ordinal (load_ftw_banks keys them the same way)
    for ordinal, (gate, _up, down) in enumerate(triples):
        layer = first_dense + ordinal
        table[f"blk.{layer}.ffn_gate_exps"] = int(gate)
        table[f"blk.{layer}.ffn_up_exps"] = int(gate)
        table[f"blk.{layer}.ffn_down_exps"] = int(down)
    return table


__all__ = ["ftw_tensor_types", "iter_gguf_weights", "parse_gguf_config", "to_gguf"]
