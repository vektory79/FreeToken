"""GLM-5.3-Flash from a llama.cpp ``glm5-next`` GGUF (unsloth ``GLM-5.3-Flash-GGUF``): config from the GGUF metadata, tensor names, and the converter's transforms undone.

The GGUF geometry is identical to the HF checkpoint (34 KDA linear-attention layers,
11 NoPE-MLA/DSA layers with a kpool-compressed indexer on a 3+1 hybrid block layout,
mHC x4 residual streams, 288-expert sigmoid/noaux_tc MoE with a shared expert and
clamped SwiGLU), so ``parse_gguf_config`` derives a text-config-shaped namespace from
the GGUF KV metadata and reuses ``glm5_next.config.parse_config`` wholesale.

A ``glm5v`` mmproj beside the text file adds the vision tower: its metadata becomes
the HF ``vision_config`` keys and the image processor config, and its tensors load
under ``visual.`` as the HF checkpoints do. The llama.cpp converter splits the Conv3d
patch embedding into two temporal halves (``v.patch_embd.weight`` and ``.weight.1``),
stacked back at load; every other tower tensor is stored in HF layout.

The llama.cpp converter (unsloth's glm5-next writer; tensor names identical across
the ``glm5-next`` and the older ``glm5next`` arch spellings) stores KDA's ``A_log``
pre-transformed as ``-exp(A_log)`` -- inverted at load -- and splits the MLA
``kv_b_proj`` into per-head 3-D ``attn_k_b`` / ``attn_v_b`` tensors (k stored
transposed, [heads, kv_lora, nope] next to [heads, v, kv_lora]); the halves may be
quantized (dynamic quants store them Q8_0), so they dequantize and fuse into the
dense layout the module's MLA absorption reads. The trailing MTP block (``blk.N.nextn.*``, ``nextn_predict_layers == 1``) is
dropped exactly like the HF loader drops the never-read MTP layer; routed experts
serve from the GGUF expert banks (``freetoken.models.gguf.experts``).
"""

from __future__ import annotations

import re
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np
import torch

from freetoken.layers.quantization.ggml_types import is_quantized
from freetoken.models.gguf.mmproj import Mmproj, find_mmproj, log_mmproj, mmproj_tensor
from freetoken.models.gguf.reader import (
    GgufTensorInfo,
    gguf_tensor_index,
    gguf_tensor_names,
    load_gguf_metadata,
)
from freetoken.models.gguf.tensors import check_scales, fused, matrix

from .config import parse_config

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig
    from freetoken.models.gguf.config import GgufConfigShim

# HF config fields the GGUF metadata does not carry (GLM-5.3-Flash's config.json values).
_FAMILY_DEFAULTS = dict(
    hidden_act="silu",
    attention_bias=False,
    # NoPE means no rope on the main attention; the indexer's interleaved rope is a
    # served-model fact the GGUF KV predates, and getting it wrong scores the indexer
    # keys incorrectly (DSA retrieval degrades to garbage).
    indexer_rope_interleave=True,
)

# checkpoint module leaf (glm5_next HF naming, language-model rooted) -> GGUF tensors
_LAYER_LEAVES: dict[str, tuple[str, ...]] = {
    # KDA input projections (the six leaves of the fused in_proj)
    "self_attn.q_proj": ("attn_q",),
    "self_attn.k_proj": ("attn_k",),
    "self_attn.v_proj": ("attn_v",),
    "self_attn.b_proj": ("ssm_beta",),
    "self_attn.f_a_proj": ("ssm_f_a",),
    "self_attn.g_a_proj": ("ssm_g_a",),
    "self_attn.f_b_proj": ("ssm_f_b",),
    "self_attn.g_b_proj": ("ssm_g_b",),
    # both layer kinds name their output projection the same
    "self_attn.o_proj": ("attn_output",),
    # DSA (MLA) projections
    "self_attn.q_a_proj": ("attn_q_a",),
    "self_attn.q_b_proj": ("attn_q_b",),
    "self_attn.kv_a_proj_with_mqa": ("attn_kv_a_mqa",),
    # two tensors, dequantized and fused at load: the module keeps its dense bf16 kv_b
    # (exempted in the registry's unquantized_modules, so quantized halves are fine too)
    "self_attn.kv_b_proj": ("attn_k_b", "attn_v_b"),
    # kpool indexer
    "self_attn.indexer.wq_b": ("indexer.attn_q_b",),
    "self_attn.indexer.wk": ("indexer.attn_k",),
    "self_attn.indexer.weights_proj": ("indexer.proj",),
    # dense MLP of the leading blocks
    "mlp.gate_proj": ("ffn_gate",),
    "mlp.up_proj": ("ffn_up",),
    "mlp.down_proj": ("ffn_down",),
    # shared expert of the MoE blocks
    "mlp.shared_experts.gate_proj": ("ffn_gate_shexp",),
    "mlp.shared_experts.up_proj": ("ffn_up_shexp",),
    "mlp.shared_experts.down_proj": ("ffn_down_shexp",),
    # routed experts (GGUF expert banks)
    "mlp.experts.0.gate_proj": ("ffn_gate_exps",),
    "mlp.experts.0.up_proj": ("ffn_up_exps",),
    "mlp.experts.0.down_proj": ("ffn_down_exps",),
}
_TOP_LEAVES: dict[str, tuple[str, ...]] = {
    # checkpoint-space names: the language-model root is already applied when to_gguf runs
    "model.language_model.embed_tokens": ("token_embd",),
    "lm_head": ("output",),
}
_LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")


def to_gguf(name: str, tensors: dict[str, int]) -> tuple[str, ...]:
    """Checkpoint module name -> the GGUF tensors (without ``.weight``) that store it; empty for modules the GGUF has no matrix for."""
    if name in _TOP_LEAVES:
        return _TOP_LEAVES[name]
    m = _LAYER_RE.match(name)
    if m is None:
        return ()
    return tuple(f"blk.{m.group(1)}.{leaf}" for leaf in _LAYER_LEAVES.get(m.group(2), ()))


def _scalar(val, key: str):
    """Fold a llama key_or_arr value (scalar, or uniform per-layer array) to a scalar."""
    if isinstance(val, (list, tuple)):
        if not val:
            raise ValueError(f"glm5next.{key} is an empty per-layer array")
        if len(set(val)) > 1:
            raise ValueError(f"glm5next.{key} varies per layer; FreeToken carries one scalar")
        return val[0]
    return val


def _layer_types(shim: "GgufConfigShim", num_layers: int) -> tuple[str, ...]:
    """Per-decoder-layer kinds: DSA where the MLA projections sit, KDA elsewhere, from the
    per-block ``attention.head_count_kv`` list (0 on the KDA blocks, 1 on the MLA ones).

    Every public glm5-next GGUF carries the list (the generic scalar spelling cannot express
    the hybrid; llama.cpp broadcasts it and then fails tensor creation), so anything else is
    refused rather than guessed at."""
    from .args import DSA_LAYER, KDA_LAYER

    per_block = shim.metadata.get(f"{shim.model_type}.attention.head_count_kv")
    if not isinstance(per_block, list) or len(per_block) < num_layers:
        raise ValueError(
            f"{shim.model_path}: glm5-next GGUF lacks the per-block attention.head_count_kv list "
            "the KDA/MLA hybrid layout needs (reconvert with a current llama.cpp)"
        )
    return tuple(DSA_LAYER if int(kv) else KDA_LAYER for kv in per_block[:num_layers])


def parse_gguf_config(shim: "GgufConfigShim") -> ModelConfig:
    md = shim.metadata
    prefix = shim.model_type

    check_scales(shim.model_path)

    def kv(key: str, default=None):
        return md.get(f"{prefix}.{key}", default)

    def need(key: str):
        val = kv(key)
        if val is None:
            raise KeyError(f"missing GGUF metadata key {prefix}.{key}")
        return val

    kv_vocab = kv("vocab_size")
    if kv_vocab is not None and int(_scalar(kv_vocab, "vocab_size")) != int(shim.vocab_size):
        raise ValueError(
            f"{prefix}.vocab_size {kv_vocab} != vocab sized from token_embd.weight ({shim.vocab_size})"
        )

    # block_count also counts the MTP block a GGUF carries after the decoder
    block_count = int(need("block_count"))
    n_mtp = int(kv("nextn_predict_layers", 0) or 0)
    if not 0 <= n_mtp < block_count:
        raise ValueError(f"{prefix}.nextn_predict_layers {n_mtp} outside 0..{block_count - 1}")
    num_layers = block_count - n_mtp
    layer_types = _layer_types(shim, num_layers)
    from .args import KDA_LAYER

    n_kda = layer_types.count(KDA_LAYER)
    if not 0 < n_kda < num_layers:
        # an all-KDA (or all-DSA) trunk has no layer of one kind; llama.cpp needs
        # 0 < n_recr < n_layer the same way
        raise ValueError(
            f"{prefix}: {n_kda} KDA layers of {num_layers} trunk layers "
            "(llama.cpp requires 0 < n_recr < n_layer)"
        )
    first_dense = int(need("leading_dense_block_count"))
    if not 0 <= first_dense <= num_layers:
        raise ValueError(f"{prefix}.leading_dense_block_count {first_dense} outside 0..{num_layers}")

    heads = int(_scalar(need("attention.head_count"), "attention.head_count"))
    qk_nope, v_head = int(need("attention.key_length_mla")), int(need("attention.value_length_mla"))
    # NoPE: the main attention carries no rotary dims (rope survives only in the indexer)
    if int(kv("rope.dimension_count", 0) or 0) != 0:
        raise NotImplementedError("glm5-next GGUF: rotary dims on the main attention are not served")

    # KDA head count: one q/k/v head set of kda.head_dim each; the conv channels or the
    # beta rows count them, falling back to the shared head_count (the two coincide on
    # every release) when the tensor table is gone (metadata-only GGUF).
    kda_head_dim = int(need("kda.head_dim"))
    index = gguf_tensor_index(shim.model_path)
    conv_q = next((i for i in index.values() if i.name.endswith("ssm_conv1d_q.weight")), None)
    kda_heads = conv_q.ne[2] // kda_head_dim if conv_q is not None else heads

    kpool_raw = kv("attention.indexer.kpool")
    kpool = 1 if kpool_raw is None else int(_scalar(kpool_raw, "attention.indexer.kpool"))
    index_topk = int(need("attention.indexer.top_k"))
    if kpool <= 0 or index_topk <= 0:
        # a non-positive kpool would pass the modulo check and silently disable compression
        raise ValueError(f"{prefix}.attention.indexer: kpool {kpool} and top_k {index_topk} must be > 0")
    if index_topk % kpool != 0:
        raise ValueError(f"indexer.top_k {index_topk} not divisible by indexer.kpool {kpool}")
    # the compressor tensors exist exactly when the indexer pools; a metadata-only GGUF
    # keeps the fact only through kpool > 1
    names = gguf_tensor_names(shim.model_path)
    kpool_compress = any(n.endswith("indexer_compressor_gate.weight") for n in names) if names else kpool > 1
    indexer_types_kv = kv("attention.indexer.types")
    if indexer_types_kv is not None and not all(indexer_types_kv):
        # llama.cpp reads False as "no full indexer on this layer", which the model cannot express
        raise NotImplementedError(f"{shim.model_path}: glm5-next GGUF with layers that lack a full indexer is not served")
    indexer_types = ("full",) * num_layers

    # the clamp is per-layer in the GGUF (and the shared expert carries its own array); the model
    # takes one value, so anything non-uniform is refused instead of silently taking the first
    clamp = need("swiglu_clamp_exp")
    clamp_vals = list(clamp) if isinstance(clamp, list) else [clamp]
    clamp_shexp = kv("swiglu_clamp_shexp")
    shexp_vals = list(clamp_shexp) if isinstance(clamp_shexp, list) else [clamp_shexp] if clamp_shexp is not None else []
    if len(set(map(float, clamp_vals + shexp_vals))) != 1:
        raise NotImplementedError(f"{shim.model_path}: non-uniform swiglu_clamp_exp / swiglu_clamp_shexp is not served")
    if int(kv("expert_gating_func", 2) or 2) != 2:
        raise NotImplementedError(f"{shim.model_path}: expert_gating_func other than sigmoid (2) is not served")
    moe_inter = int(_scalar(need("expert_feed_forward_length"), "expert_feed_forward_length"))
    n_shared = int(kv("expert_shared_count", 1) or 1)
    shexp_ffn = kv("expert_shared_feed_forward_length")
    if shexp_ffn is not None and int(_scalar(shexp_ffn, "expert_shared_feed_forward_length")) != moe_inter * n_shared:
        raise ValueError(
            f"{prefix}.expert_shared_feed_forward_length {shexp_ffn} != "
            f"expert_feed_forward_length * expert_shared_count ({moe_inter} * {n_shared})"
        )
    text = SimpleNamespace(
        **_FAMILY_DEFAULTS,
        num_hidden_layers=num_layers,
        layer_types=tuple(layer_types),
        mlp_layer_types=("dense",) * first_dense + ("sparse",) * (num_layers - first_dense),
        first_k_dense_replace=first_dense,
        hidden_size=int(need("embedding_length")),
        num_attention_heads=heads,
        vocab_size=int(shim.vocab_size),
        intermediate_size=int(_scalar(need("feed_forward_length"), "feed_forward_length")),
        rms_norm_eps=float(need("attention.layer_norm_rms_epsilon")),
        max_position_embeddings=int(need("context_length")),
        q_lora_rank=int(need("attention.q_lora_rank")),
        kv_lora_rank=int(need("attention.kv_lora_rank")),
        qk_nope_head_dim=qk_nope,
        qk_rope_head_dim=0,
        v_head_dim=v_head,
        mla_use_nope=True,
        index_n_heads=int(need("attention.indexer.head_count")),
        index_head_dim=int(need("attention.indexer.key_length")),
        index_topk=index_topk,
        indexer_types=indexer_types,
        index_kpool=kpool,
        index_kpool_compress=kpool_compress,
        # old converters predate the key; both released checkpoints set it true
        index_kpool_always_select_tail=bool(kv("attention.indexer.kpool_select_tail", True)),
        linear_attn_config={
            "num_heads": kda_heads,
            "head_dim": kda_head_dim,
            "short_conv_kernel_size": int(need("ssm.conv_kernel")),
            "gate_lower_bound": float(need("kda.gate_lower_bound")),
        },
        mhc=True,
        hc_mult=int(need("hyper_connection.count")),
        hc_sinkhorn_iters=int(need("hyper_connection.sinkhorn_iterations")),
        hc_eps=float(need("hyper_connection.epsilon")),
        n_routed_experts=int(need("expert_count")),
        n_shared_experts=n_shared,
        num_experts_per_tok=int(need("expert_used_count")),
        moe_intermediate_size=moe_inter,
        norm_topk_prob=bool(need("expert_weights_norm")),
        routed_scaling_factor=float(need("expert_weights_scale")),
        n_group=int(kv("expert_group_count", 1) or 1),
        topk_group=int(kv("expert_group_used_count", 1) or 1),
        tie_word_embeddings=bool(shim.tie_word_embeddings),
        swiglu_limit=float(clamp_vals[0]),
    )
    hf_like = SimpleNamespace(
        text_config=text,
        model_type="glm5_next",
        architectures=list(shim.architectures),
        vision_config=getattr(shim, "vision_config", None),
        image_token_id=getattr(shim, "image_token_id", None),
    )
    config = parse_config(hf_like)
    return replace(config, expert_quant="gguf")


# ----------------------------------------------------------------------------------------------
# Vision tower (a glm5v mmproj)
# ----------------------------------------------------------------------------------------------

_PROJECTOR = "glm5v"
_IMAGE_TOKEN = "<|image|>"
_PATCH = ("v.patch_embd.weight", "v.patch_embd.weight.1")

# tower / merger tensors in HF layout already; only the patch embedding is split
_TOP = {
    "v.patch_embd.bias": "patch_embed.proj.bias",
    "v.post_ln.weight": "post_layernorm.weight",
    "mm.patch_merger.weight": "downsample.weight",
    "mm.patch_merger.bias": "downsample.bias",
    "mm.model.fc.weight": "merger.proj.weight",
    "mm.post_norm.weight": "merger.post_projection_norm.weight",
    "mm.post_norm.bias": "merger.post_projection_norm.bias",
    "mm.gate.weight": "merger.gate_proj.weight",
    "mm.up.weight": "merger.up_proj.weight",
    "mm.down.weight": "merger.down_proj.weight",
}
_BLOCK = {
    "attn_qkv": "attn.qkv",
    "attn_out": "attn.proj",
    "attn_q_norm": "attn.q_norm",
    "attn_k_norm": "attn.k_norm",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "ln1": "norm1",
    "ln2": "norm2",
}
_BLOCK_RE = re.compile(r"^v\.blk\.(\d+)\.(\w+)\.(weight|bias)$")


def gguf_mm_config(text_md: dict[str, Any], mmproj: Mmproj, vocab_size: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """The HF config.json keys and the preprocessor_config.json of the tower in ``mmproj``, for the text GGUF whose metadata is ``text_md``."""
    return _mm_config(text_md, load_gguf_metadata(mmproj.path), mmproj.path, gguf_tensor_index(mmproj.path), vocab_size)


# a metadata-only FTW carrier has no tensor table, so the tower facts main derives from
# mmproj tensor headers fall back to the GLM-5.3-Flash tower reference (round-5 phase
# verification pinned them tensor-exact: verify_mmproj_tensors 347/347 params)
_TOWER_REFERENCE = {"projection_intermediate_size": 10240, "in_channels": 3, "attention_bias": True}


def gguf_carrier_mm_config(text_md: dict[str, Any], vocab_size: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """The vision facts of a metadata-only FTW carrier from the clip.* KVs the converter merged.

    A carrier strips the tensor table, so the header-derived fields fall back to the
    GLM-5.3-Flash tower reference; a carrier without vision facts stays text-only, and
    a partial clip.* section fails fast in the shared checks below.
    """
    if text_md.get("clip.has_vision_encoder") is not True:
        return {}, None
    return _mm_config(text_md, text_md, "the FTW carrier", None, vocab_size)


def _mm_config(
    text_md: dict[str, Any],
    md: dict[str, Any],
    source: str,
    index: dict[str, GgufTensorInfo] | None,
    vocab_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Shared core of the mmproj-file and carrier paths: ``source`` names the tower
    origin for error messages, ``index`` is the tensor table or None on a carrier."""
    projector = md.get("clip.vision.projector_type") or md.get("clip.projector_type")
    if projector != _PROJECTOR:
        raise ValueError(f"{source} is a {projector!r} tower; GLM-5.3-Flash needs {_PROJECTOR!r}")
    # the converter writes use_silu=True only for the silu tower; a gelu tower carries
    # use_gelu or neither key, and the model has no gelu epilogue
    if md.get("clip.use_silu") is not True:
        raise NotImplementedError(f"{source}: only the silu tower is served")

    def kv(key: str):
        return md[f"clip.vision.{key}"]

    arch = text_md["general.architecture"]
    text_hidden = int(text_md[f"{arch}.embedding_length"])
    if int(kv("projection_dim")) != text_hidden:
        raise ValueError(f"{source} projects to {int(kv('projection_dim'))} but the text model is {text_hidden} wide")
    tokens = text_md["tokenizer.ggml.tokens"]
    if _IMAGE_TOKEN not in tokens:
        raise ValueError(f"the text GGUF tokenizer has no {_IMAGE_TOKEN} token")
    if index is not None:
        # the converter only splits a temporal-2 patch embedding, so a glm5v file always
        # carries both halves; one without the second is not this tower
        if _PATCH[1] not in index:
            raise ValueError(f"{source}: the second patch-embedding half {_PATCH[1]!r} is missing")
        # ggml order [in, out]: the gate's out features are the merger's intermediate width
        proj_inter = int(index["mm.gate.weight"].ne[1])
        in_channels = int(index[_PATCH[0]].ne[2])
        attention_bias = "v.blk.0.attn_qkv.bias" in index
    else:
        proj_inter = _TOWER_REFERENCE["projection_intermediate_size"]
        in_channels = _TOWER_REFERENCE["in_channels"]
        attention_bias = _TOWER_REFERENCE["attention_bias"]
    temporal = 2
    patch, merge = int(kv("patch_size")), int(kv("spatial_merge_size"))
    # llama.cpp stores the canvas budget in single-frame pixels; the processor counts
    # soft tokens, one per (patch * merge) square of pixels per frame. The released
    # mmproj files predate the budget keys (they carry only clip.vision.image_size):
    # those towers serve with the GLM-5.3-Flash reference budget the round-5 phase
    # verification pinned (min 16 / max 8000 soft tokens). A file that carries the
    # keys keeps the derived whole-token divisibility check.
    per_token = patch * patch * merge * merge
    if "clip.vision.image_min_pixels" in md or "clip.vision.image_max_pixels" in md:
        budget = {}
        for key in ("image_min_pixels", "image_max_pixels"):
            pixels = int(kv(key))
            if pixels % per_token:
                raise ValueError(f"{source}: {key} {pixels} is not a whole number of {per_token}-pixel tokens")
            budget[key] = pixels // per_token
    else:
        budget = {"image_min_pixels": 16, "image_max_pixels": 8000}
    # the released files spell the clamp fact swiglu_limit (llama.cpp's older key); the
    # newer swiglu_clamp spelling wins when present, neither is a broken tower
    clamp = md.get("clip.vision.swiglu_clamp")
    if clamp is None:
        clamp = md.get("clip.vision.swiglu_limit")
    if clamp is None:
        raise ValueError(f"{source}: neither clip.vision.swiglu_clamp nor clip.vision.swiglu_limit is present")
    vision_config = {
        "hidden_act": "silu",
        "hidden_size": int(kv("embedding_length")),
        "depth": int(kv("block_count")),
        "num_heads": int(kv("attention.head_count")),
        "intermediate_size": int(kv("feed_forward_length")),
        "projection_intermediate_size": proj_inter,
        "out_hidden_size": text_hidden,
        # ggml order [kw, kh, in, out]
        "in_channels": in_channels,
        "patch_size": patch,
        "temporal_patch_size": temporal,
        "spatial_merge_size": merge,
        "rms_norm_eps": float(kv("attention.layer_norm_epsilon")),
        "swiglu_limit": float(clamp),
        "attention_bias": attention_bias,
    }
    hf_fields = {
        "vision_config": vision_config,
        "image_token_id": tokens.index(_IMAGE_TOKEN),
    }
    preprocessor = {
        "image_processor_type": "Glm5NextImageProcessor",
        "patch_size": patch,
        "temporal_patch_size": temporal,
        "merge_size": merge,
        "image_mean": [float(v) for v in kv("image_mean")],
        "image_std": [float(v) for v in kv("image_std")],
        "min_image_tokens": budget["image_min_pixels"],
        "max_image_tokens": budget["image_max_pixels"],
    }
    return hf_fields, preprocessor


def _vision_name(gguf_name: str) -> str:
    if gguf_name in _TOP:
        return _TOP[gguf_name]
    if (m := _BLOCK_RE.match(gguf_name)) and m[2] in _BLOCK:
        return f"blocks.{m[1]}.{_BLOCK[m[2]]}.{m[3]}"
    raise ValueError(f"unexpected tensor {gguf_name!r} in a {_PROJECTOR} mmproj")


def iter_mmproj_weights(mmproj: Mmproj) -> Iterator[tuple[str, torch.Tensor]]:
    """The tower state dict under ``visual.``, named as the HF checkpoints load."""
    index = gguf_tensor_index(mmproj.path)
    blocks = {int(m[1]) for gguf_name in index if (m := _BLOCK_RE.match(gguf_name))}
    if blocks:
        # tower completeness: every block the metadata declares must be present, and
        # no block may sit beyond it
        md = load_gguf_metadata(mmproj.path)
        if "clip.vision.block_count" not in md:
            raise ValueError(f"mmproj {mmproj.path}: no clip.vision.block_count; the tower depth is unknown")
        depth = int(md["clip.vision.block_count"])
        missing = sorted(set(range(depth)) - blocks)
        if missing:
            raise ValueError(
                f"mmproj {mmproj.path}: {len(blocks)} of {depth} tower blocks present; "
                f"missing vision blocks {missing}"
            )
        beyond = sorted(b for b in blocks if b >= depth)
        if beyond:
            raise ValueError(f"mmproj {mmproj.path}: vision block(s) {beyond} outside the declared depth {depth}")
    slices = [mmproj_tensor(index[n]) for n in _PATCH]
    yield "visual.patch_embed.proj.weight", torch.stack(slices, dim=2)
    for gguf_name, info in index.items():
        if gguf_name not in _PATCH:
            yield f"visual.{_vision_name(gguf_name)}", mmproj_tensor(info)


# ----------------------------------------------------------------------------------------------
# Weights
# ----------------------------------------------------------------------------------------------

def _float(info: GgufTensorInfo) -> torch.Tensor:
    """A float tensor in torch order, dequantizing packed ggml rows: dynamic quants quantize
    tensors the module consumes as floats (the kv_b halves, the mHC mixers, the compressor gate).
    Trailing unit dims in ``ne`` are dropped -- some writers ship the conv kernels 4-D
    (llama.cpp's glm5-next loader tries that shape first) and a trailing 1 carries no elements."""
    ne = info.ne
    while len(ne) > 1 and ne[-1] == 1:
        ne = ne[:-1]
    if not is_quantized(info.ggml_type):
        from freetoken.models.gguf.tensors import float_tensor

        return float_tensor(info).reshape(tuple(reversed(ne)))
    from freetoken.third_party.gguf import quants
    from freetoken.third_party.gguf.constants import GGMLQuantizationType

    flat = quants.dequantize(np.array(info.data()), GGMLQuantizationType(info.ggml_type))
    return torch.from_numpy(flat).view(*reversed(ne))


def _kv_b(kb: GgufTensorInfo, vb: GgufTensorInfo, num_heads: int, kv_lora: int) -> torch.Tensor:
    """Per-head k [kv_lora, nope] (stored transposed) + v [v, kv_lora] -> the module's fused dense [heads * (nope + v), kv_lora]."""
    k = _float(kb).transpose(1, 2)  # [heads, nope, kv_lora]
    fused_w = torch.cat([k, _float(vb)], dim=1)
    assert fused_w.shape[0] == num_heads, (fused_w.shape, num_heads)
    return fused_w.reshape(num_heads * fused_w.shape[1], kv_lora).to(torch.bfloat16).contiguous()


def iter_gguf_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
    include_vision: bool = True,
    mmproj_path: str | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """The dense state dict from the GGUF; the routed experts load as offload banks (``freetoken.models.gguf.experts``)."""
    if not include_non_moe:
        return
    from freetoken.utils import cached_load_hf_config

    config = parse_gguf_config(cached_load_hf_config(model_path))
    # an explicit --mmproj is operator intent: stream the tower from that file even when
    # the text-only shim (no auto-discovered mmproj) is not multimodal
    vision = include_vision and (mmproj_path is not None or config.is_multimodal)
    log_mmproj(model_path, vision)
    args = config.glm5_args
    index = gguf_tensor_index(model_path)
    hidden = config.hidden_size
    from freetoken.third_party.gguf.constants import GGMLQuantizationType

    q8_0, q6_k = int(GGMLQuantizationType.Q8_0), int(GGMLQuantizationType.Q6_K)
    consumed: set[str] = set()

    def t(name: str) -> GgufTensorInfo:
        key = name if name in index else f"{name}.weight"
        consumed.add(key)
        return index[key]

    def packed(name: str) -> GgufTensorInfo:
        # the convert swap builds every packed linear as Q8_0; a deviant dense row
        # would otherwise die as a late loader shape assert
        info = t(name)
        if info.ggml_type != q8_0:
            raise ValueError(
                f"{info.name} is {GGMLQuantizationType(info.ggml_type).name}; "
                "the convert swap builds packed linears as Q8_0"
            )
        return info

    def bf16(name: str) -> torch.Tensor:
        return _float(t(name)).to(torch.bfloat16)

    emb = t("token_embd")
    if emb.ggml_type != q8_0:
        raise ValueError(
            f"{emb.name} is {GGMLQuantizationType(emb.ggml_type).name}; "
            "the convert swap builds the embedding as Q8_0"
        )
    yield from fused("model.embed_tokens", [emb], [matrix(emb)], hidden)
    out = t("output")
    if out.ggml_type != q6_k:
        raise ValueError(
            f"{out.name} is {GGMLQuantizationType(out.ggml_type).name}; "
            "the convert swap builds the untied head as Q6_K"
        )
    yield from fused("lm_head", [out], [matrix(out)], hidden)
    yield "model.norm.weight", bf16("output_norm")

    for layer in range(config.num_layers):
        blk, pre = f"blk.{layer}", f"model.layers.{layer}"
        yield f"{pre}.input_layernorm.weight", bf16(f"{blk}.attn_norm")
        yield f"{pre}.post_attention_layernorm.weight", bf16(f"{blk}.ffn_norm")
        # mHC mixing tensors, fp32 on every layer
        for sub in ("attn", "ffn"):
            for part in ("fn", "base", "scale"):
                yield f"{pre}.hc_{sub}_{part}", _float(t(f"{blk}.hc_{sub}_{part}"))

        if args.is_kda_layer(layer):
            sa = f"{pre}.self_attn"
            p, h, d = args.linear_num_heads * args.linear_head_dim, args.linear_num_heads, args.linear_head_dim
            group = [packed(f"{blk}.{n}") for n in ("attn_q", "attn_k", "attn_v", "ssm_beta", "ssm_f_a", "ssm_g_a")]
            yield from fused(f"{sa}.in_proj", group, [matrix(i) for i in group], hidden)
            # the GGUF conv kernels already carry the [channels, 1, kernel] singleton dim
            conv = torch.cat([_float(t(f"{blk}.ssm_conv1d_{q}")) for q in ("q", "k", "v")], dim=0)
            yield f"{sa}.conv1d.weight", conv.to(torch.bfloat16)
            for proj, src, k in (("f_b_proj", "ssm_f_b", d), ("g_b_proj", "ssm_g_b", d), ("o_proj", "attn_output", p)):
                info = packed(f"{blk}.{src}")
                yield from fused(f"{sa}.{proj}", [info], [matrix(info)], k)
            # the converter stores A_log pre-transformed as -exp(A_log); the kernels want the raw log
            ssm_a = _float(t(f"{blk}.ssm_a"))
            if not torch.isfinite(ssm_a).all():
                # NaN >= 0 is False, so a NaN row would slip the sign check and log()
                # would silently poison the recurrent gate
                raise ValueError(f"{blk}.ssm_a: ssm_a holds non-finite entries; cannot derive A_log")
            bad = int((ssm_a >= 0).sum())
            if bad:
                raise ValueError(
                    f"{blk}.ssm_a: ssm_a holds -exp(A_log) but {bad} entries are >= 0; cannot derive A_log"
                )
            yield f"{sa}.A_log", torch.log(-ssm_a)
            yield f"{sa}.dt_bias", _float(t(f"{blk}.ssm_dt.bias"))
            yield f"{sa}.o_norm.weight", bf16(f"{blk}.ssm_norm")
        else:
            sa = f"{pre}.self_attn"
            for proj, src, k in (
                ("q_a_proj", "attn_q_a", hidden),
                ("q_b_proj", "attn_q_b", args.q_lora_rank),
                ("kv_a_proj_with_mqa", "attn_kv_a_mqa", hidden),
                ("o_proj", "attn_output", args.num_heads * args.v_head_dim),
            ):
                info = packed(f"{blk}.{src}")
                yield from fused(f"{sa}.{proj}", [info], [matrix(info)], k)
            yield f"{sa}.kv_b_proj.weight", _kv_b(
                t(f"{blk}.attn_k_b"), t(f"{blk}.attn_v_b"), args.num_heads, args.kv_lora_rank
            )
            yield f"{sa}.q_a_layernorm.weight", bf16(f"{blk}.attn_q_a_norm")
            yield f"{sa}.kv_a_layernorm.weight", bf16(f"{blk}.attn_kv_a_norm")
            # wq_b serves its checkpoint quant (packed); wk / weights_proj are dense modules
            # (glm_moe_dsa's indexer reasoning), dequantized at load when the file quantized them
            wqb = packed(f"{blk}.indexer.attn_q_b")
            yield from fused(f"{sa}.indexer.wq_b", [wqb], [matrix(wqb)], args.q_lora_rank)
            yield f"{sa}.indexer.wk.weight", _float(t(f"{blk}.indexer.attn_k")).to(torch.bfloat16)
            yield f"{sa}.indexer.weights_proj.weight", _float(t(f"{blk}.indexer.proj")).to(torch.bfloat16)
            yield f"{sa}.indexer.k_norm.weight", bf16(f"{blk}.indexer.k_norm")
            yield f"{sa}.indexer.k_norm.bias", bf16(f"{blk}.indexer.k_norm.bias")
            yield f"{sa}.indexer.index_kpool_compress_gate", bf16(f"{blk}.indexer_compressor_gate")
            yield f"{sa}.indexer.index_kpool_compress_ape", _float(t(f"{blk}.indexer_compressor_ape"))

        mlp = f"{pre}.mlp"
        if layer < config.first_k_dense_replace:
            for proj, src, k in (
                ("gate_proj", "ffn_gate", hidden),
                ("up_proj", "ffn_up", hidden),
                ("down_proj", "ffn_down", config.intermediate_size),
            ):
                info = packed(f"{blk}.{src}")
                yield from fused(f"{mlp}.{proj}", [info], [matrix(info)], k)
        else:
            yield f"{mlp}.gate.weight", bf16(f"{blk}.ffn_gate_inp")
            yield f"{mlp}.e_score_correction_bias", _float(t(f"{blk}.exp_probs_b.bias"))
            shared_i = config.moe_intermediate_size * max(1, config.n_shared_experts)
            for proj, src, k in (
                ("gate_proj", "ffn_gate_shexp", hidden),
                ("up_proj", "ffn_up_shexp", hidden),
                ("down_proj", "ffn_down_shexp", shared_i),
            ):
                info = packed(f"{blk}.{src}")
                yield from fused(f"{mlp}.shared_experts.{proj}", [info], [matrix(info)], k)

    # anything the trunk never asked for is a corrupt or unknown file: name it
    for leftover in sorted(set(index) - consumed):
        if leftover.startswith("blk."):
            head, sep, suffix = leftover[len("blk."):].partition(".")
            if sep and head.isdigit():
                if int(head) >= config.num_layers:
                    continue  # the MTP/draft block, ignored like the HF reader
                if suffix.startswith(("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")):
                    continue  # routed banks load through models/gguf/experts
        raise ValueError(f"unmapped glm5next GGUF tensor: {leftover}")

    if vision:
        yield from iter_mmproj_weights(find_mmproj(model_path, mmproj_path))




# The gguf -> FTW convert contract of the pre-re-platform loader (round-4 era):
# which ggml type each gguf leaf packs. Main's loader builds quant configs from
# the checkpoint's own type table / index meta; these maps serve the older
# carriers that kept only the expert triples, through ftw_tensor_types.

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
    from freetoken.third_party.gguf.constants import GGMLQuantizationType

    q8_0 = int(GGMLQuantizationType.Q8_0)
    q6_k = int(GGMLQuantizationType.Q6_K)

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
        "qw": q8_0,
        "bf16": int(GGMLQuantizationType.BF16),
        "fp32": int(GGMLQuantizationType.F32),
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
    table = {"token_embd": q8_0, "output": q6_k, "output_norm": cast_types["bf16"]}
    for layer in range(num_layers):
        for suffix, kind in leaves.items():
            table[f"blk.{layer}.{suffix}"] = kind
        for suffix in ("attn_q", "attn_k", "attn_v", "ssm_beta", "ssm_f_a", "ssm_g_a"):
            table[f"blk.{layer}.{suffix}"] = q8_0
    # the triples are MoE-ordinal (load_ftw_banks keys them the same way)
    for ordinal, (gate, _up, down) in enumerate(triples):
        layer = first_dense + ordinal
        table[f"blk.{layer}.ffn_gate_exps"] = int(gate)
        table[f"blk.{layer}.ffn_up_exps"] = int(gate)
        table[f"blk.{layer}.ffn_down_exps"] = int(down)
    return table


__all__ = ["ftw_tensor_types", "gguf_mm_config", "iter_gguf_weights", "iter_mmproj_weights", "parse_gguf_config", "to_gguf"]
