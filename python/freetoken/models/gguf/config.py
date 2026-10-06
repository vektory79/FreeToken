"""GGUF config shim: the object the model registry sees for a ``.gguf`` model.

``cached_load_hf_config`` returns one of these for GGUF paths instead of a HF
``PretrainedConfig``. It carries the architecture key (so the registry can dispatch),
the raw GGUF metadata dict, and a few derived facts that need the tensor table
(``vocab_size``, ``tie_word_embeddings``). The per-arch ``parse_gguf_config`` reads
``metadata`` to build the FreeToken ``ModelConfig``.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any

from .reader import _reader, gguf_architecture, load_gguf_metadata, gguf_tensor_names

# GGUF ``general.architecture`` -> FreeToken registry key (a GGUF-specific spec that
# reuses the model classes but a GGUF parse_config / iter_weights).
GGUF_ARCH_TO_REGISTRY: dict[str, str] = {
    "gemma4": "Gemma4GGUFForCausalLM",
    "glm5next": "Glm5NextGGUFForCausalLM",
}

# Registry key when an mmproj file is supplied (--mmproj): a key here opts the arch
# into raw-gguf vision boots. Text-only dispatch stays GGUF_ARCH_TO_REGISTRY, so a
# boot without the flag resolves the exact same spec it always did.
GGUF_ARCH_TO_REGISTRY_VISION: dict[str, str] = {
    "glm5next": "Glm5NextGGUFForConditionalGeneration",
}

# arch -> "module:attr" building the tower config from the mmproj metadata. Called
# lazily: the model modules import this one, so a top-level import would cycle.
_GGUF_VISION_CONFIG_HOOK: dict[str, str] = {
    "glm5next": "freetoken.models.glm5_next.gguf:vision_config_from_tower_metadata",
}

# arch -> "module:attr" building the image-processor kwargs from the same tower
# metadata (phase-3 fallback processor)
_GGUF_VISION_PROCESSOR_HOOK: dict[str, str] = {
    "glm5next": "freetoken.models.glm5_next.gguf:image_processor_config_from_tower_metadata",
}

# arch -> "module:attr" resolving the image placeholder token id from the main
# GGUF's tokenizer metadata. Fails fast on a missing chat template / vocab entry:
# image prompts would otherwise silently lose the image.
_GGUF_VISION_SERVING_HOOK: dict[str, str] = {
    "glm5next": "freetoken.models.glm5_next.gguf:resolve_gguf_image_serving",
}


@dataclass(frozen=True)
class GgufConfigShim:
    architectures: list[str]
    model_path: str
    model_type: str
    metadata: dict[str, Any]
    vocab_size: int
    tie_word_embeddings: bool
    # raw-gguf vision boot facts: the tower file path (raw .gguf + --mmproj) or the
    # tower metadata embedded in this file's KV section (a converted FTW's merged
    # source_metadata.gguf); both absent serves the checkpoint text-only
    mmproj_path: str | None = None
    vision_config: Any = None
    image_token_id: int | None = None
    # image-processor fallback kwargs when the tower rides in the metadata (FTW boot:
    # no mmproj file exists to parse); None on the raw-gguf path, which parses lazily
    image_processor_kwargs: dict | None = None

    def to_dict(self) -> dict[str, Any]:
        """Minimal HF-config-like dict for trunk code that introspects the config
        (e.g. server arg parsing reads ``torch_dtype`` to resolve ``--dtype auto``).
        GGUF weights dequantize to a bf16 compute path."""
        return {
            "architectures": list(self.architectures),
            "model_type": self.model_type,
            "torch_dtype": "bfloat16",
            "vocab_size": self.vocab_size,
            "tie_word_embeddings": self.tie_word_embeddings,
        }


def _vocab_size(model_path: str) -> int:
    from .reader import _reader

    for t in _reader(model_path).tensors:
        if t.name == "token_embd.weight":
            return int(t.shape[-1])  # ggml [hidden, vocab] -> vocab is last
    # A metadata-only GGUF (an FTW dir's source_metadata.gguf) strips the tensor table, so
    # fall back to the tokenizer vocab. llama.cpp sizes token_embd's rows to n_vocab =
    # len(tokenizer.ggml.tokens), so this equals the tensor-derived value exactly.
    toks = load_gguf_metadata(model_path).get("tokenizer.ggml.tokens")
    if toks is not None:
        return len(toks)
    raise ValueError(f"GGUF {model_path}: no token_embd.weight to size the vocab")


def build_gguf_shim(model_path: str, mmproj_path: str | None = None) -> GgufConfigShim:
    arch = gguf_architecture(model_path)
    metadata = load_gguf_metadata(model_path)
    names = gguf_tensor_names(model_path)
    vision_config = None
    image_token_id = None
    image_processor_kwargs = None
    tower_meta = None
    tower_shapes: dict[str, tuple] = {}
    if mmproj_path is not None:
        # fail fast before anything else reads the flag: a missing or non-clip file is
        # an operator error, and the error must name the file
        if not os.path.isfile(mmproj_path):
            raise ValueError(f"--mmproj file not found: {mmproj_path}")
        mmproj_arch = gguf_architecture(mmproj_path)
        if mmproj_arch != "clip":
            raise ValueError(
                f"--mmproj {mmproj_path}: architecture {mmproj_arch!r} != 'clip'; "
                "an mmproj file carries the vision tower"
            )
        registry_key = GGUF_ARCH_TO_REGISTRY_VISION.get(arch)
        if registry_key is None:
            raise ValueError(
                f"--mmproj is only supported for {sorted(GGUF_ARCH_TO_REGISTRY_VISION)} "
                f"GGUF checkpoints, not {arch!r}"
            )
        tower_meta = load_gguf_metadata(mmproj_path)
        tower_shapes = {
            t.name: tuple(reversed([int(d) for d in t.shape])) for t in _reader(mmproj_path).tensors
        }
    elif metadata.get("clip.has_vision_encoder"):
        # tower metadata rides INSIDE this file's KV section: a converted FTW's
        # source_metadata.gguf carries the mmproj clip.* keys merged at convert time,
        # so a GGUF-FTW boot needs no --mmproj and never opens the source .gguf. A
        # raw checkpoint GGUF must not reach here with vision keys - its tensor table
        # is the LLM, not a tower, so serving would silently lack the visual weights.
        registry_key = GGUF_ARCH_TO_REGISTRY_VISION.get(arch)
        if registry_key is None:
            raise ValueError(
                f"this GGUF carries clip.* tower metadata, but {arch!r} has no vision "
                f"registry entry (known: {sorted(GGUF_ARCH_TO_REGISTRY_VISION)})"
            )
        if names:
            raise ValueError(
                f"GGUF {model_path}: carries clip.* tower metadata but is not a "
                "metadata-only FTW carrier; pass --mmproj with the tower file"
            )
        tower_meta = metadata
    else:
        registry_key = GGUF_ARCH_TO_REGISTRY.get(arch)
        if registry_key is None:
            raise ValueError(
                f"GGUF architecture {arch!r} is not supported "
                f"(known: {sorted(GGUF_ARCH_TO_REGISTRY)})"
            )
    if tower_meta is not None:
        module_name, _, attr = _GGUF_VISION_CONFIG_HOOK[arch].partition(":")
        vision_config = getattr(importlib.import_module(module_name), attr)(
            tower_meta, tower_shapes, source=str(model_path)
        )
        module_name, _, attr = _GGUF_VISION_SERVING_HOOK[arch].partition(":")
        image_token_id = getattr(importlib.import_module(module_name), attr)(
            model_path, metadata
        )
        if mmproj_path is None:
            # FTW carrier: bake the processor kwargs now (the tower file they would
            # be parsed from does not exist); the raw-gguf path parses lazily
            module_name, _, attr = _GGUF_VISION_PROCESSOR_HOOK[arch].partition(":")
            image_processor_kwargs = getattr(importlib.import_module(module_name), attr)(
                tower_meta, tower_shapes, source=str(model_path)
            )
    if names:
        # No separate output projection -> embeddings are tied.
        tie_word_embeddings = "output.weight" not in names
    else:
        # Metadata-only GGUF (an FTW dir's source_metadata.gguf): the tensor table is
        # stripped, so the fact travels as a KV written at convert time.
        from .reader import OUTPUT_WEIGHT_PRESENT_KV

        present = metadata.get(OUTPUT_WEIGHT_PRESENT_KV)
        if present is None:
            raise ValueError(
                f"{model_path}: metadata-only GGUF lacks {OUTPUT_WEIGHT_PRESENT_KV!r}; "
                "reconvert the checkpoint with the current freetoken.checkpoint.convert"
            )
        tie_word_embeddings = not present
    return GgufConfigShim(
        architectures=[registry_key],
        model_path=model_path,
        model_type=arch,
        metadata=metadata,
        vocab_size=_vocab_size(model_path),
        tie_word_embeddings=tie_word_embeddings,
        mmproj_path=mmproj_path,
        vision_config=vision_config,
        image_token_id=image_token_id,
        image_processor_kwargs=image_processor_kwargs,
    )


__all__ = [
    "GgufConfigShim",
    "GGUF_ARCH_TO_REGISTRY",
    "GGUF_ARCH_TO_REGISTRY_VISION",
    "build_gguf_shim",
]
