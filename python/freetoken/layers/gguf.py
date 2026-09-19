"""Native-GGUF quantized layers: weights stay in their packed block layout and are
dequantized *inside* the borrowed llama.cpp kernels (no bf16 copy ever materialized).

Mirrors vLLM/sglang's ``GGUFLinearMethod`` / ``GGUFEmbeddingMethod`` dispatch, ported
onto FreeToken's ``BaseOP``. FreeToken keeps fused projections (qkv, gate_up) as a
single tensor: because Q4_0/K-quants pack each *output row* independently over the
input dim, the loader can concatenate the per-shard packed rows along dim 0 (they
share an input dim, hence the same ``row_bytes``), so a fused layer is still one
``[out, row_bytes]`` qweight -- no per-shard padding bookkeeping needed.

TP is assumed to be 1 (the gemma4 GGUF path restricts to TP=1, like the HF path).
"""

from __future__ import annotations

import os

import torch

from freetoken.models.gguf.dequant import (
    BLOCK_SHAPE,
    GGML_BF16,
    GGML_F16,
    GGML_F32,
    GGML_IQ3_XXS,
    GGML_IQ4_XS,
    GGML_NAME,
    GGML_Q3_K,
    GGML_Q4_K,
    GGML_Q4_0,
    GGML_Q6_K,
    GGML_Q8_0,
    row_bytes,
)

from .base import BaseOP

# ggml type groups for kernel dispatch (subset we build kernels for).
_UNQUANTIZED = {GGML_F32, GGML_F16, GGML_BF16}
# every format here has both an MMVQ (small-batch GEMV) and an MMQ (large-batch)
# kernel; the iq entries were lifted by the grouped-MMQ work (types 18/23).
_MMVQ = {GGML_Q4_0, GGML_Q8_0, GGML_Q6_K, GGML_Q3_K, GGML_Q4_K, GGML_IQ3_XXS, GGML_IQ4_XS}
_MMQ = {GGML_Q4_0, GGML_Q8_0, GGML_Q6_K, GGML_Q3_K, GGML_Q4_K, GGML_IQ3_XXS, GGML_IQ4_XS}
_DEQUANT = {GGML_Q4_0, GGML_Q8_0, GGML_Q6_K, GGML_Q3_K, GGML_Q4_K}
# Formats with dequant + MMVQ kernels but NO MMQ. Empty today (the iq gap was
# closed in gguf_kernel.cu); kept so the dense-batch guard below stays in place
# for any future MMVQ-only format.
_IQ_ONLY: set[int] = set()

# Below this token count, the MMVQ GEMV kernel wins (matches vLLM's heuristic).
_MMVQ_SAFE = 6


def fused_mul_mat_gguf(x: torch.Tensor, qweight: torch.Tensor, qweight_type: int) -> torch.Tensor:
    """y = x @ dequant(qweight).T, dispatched by batch size and quant type."""
    if not x.is_contiguous():
        # The MMVQ/MMQ kernels read the activation buffer assuming row-major
        # strides; split views (e.g. KDA f_a/g_a) would be consumed as garbage.
        x = x.contiguous()
    from freetoken.kernel.gguf import (
        ggml_dequantize,
        ggml_mul_mat_a8,
        ggml_mul_mat_vec_a8,
    )

    out_features = qweight.shape[0]
    if x.shape[0] == 0:
        return x.new_empty((0, out_features))
    if qweight_type in _UNQUANTIZED:
        return x @ qweight.T
    if x.shape[0] <= _MMVQ_SAFE and qweight_type in _MMVQ:
        return ggml_mul_mat_vec_a8(qweight, x, qweight_type, out_features)
    if qweight_type in _MMQ:
        return ggml_mul_mat_a8(qweight, x, qweight_type, out_features)
    if qweight_type in _IQ_ONLY:
        raise NotImplementedError(
            f"{GGML_NAME.get(qweight_type, qweight_type)} has no MMQ kernel (iq formats "
            f"are MMVQ-only): dense batch {x.shape[0]} > {_MMVQ_SAFE} is unsupported; "
            "routed-expert banks route through ggml_moe_a8_vec instead"
        )
    if qweight_type in _DEQUANT:
        block, type_size = BLOCK_SHAPE[qweight_type]
        in_features = qweight.shape[1] // type_size * block
        weight = ggml_dequantize(qweight, qweight_type, out_features, in_features, x.dtype)
        return x @ weight.T
    raise NotImplementedError(f"unsupported GGUF type {GGML_NAME.get(qweight_type, qweight_type)}")


class GGUFLinear(BaseOP):
    """Linear whose weight is a native GGUF block-quantized ``[out, row_bytes]`` tensor."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        quant_type: int,
        has_bias: bool = False,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self._quant_type = quant_type
        self.qweight = torch.empty(out_features, row_bytes(in_features, quant_type), dtype=torch.uint8)
        self.bias = torch.empty(out_features) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = fused_mul_mat_gguf(x, self.qweight, self._quant_type)
        if self.bias is not None:
            out = out + self.bias
        return out


def _kda_nsplit_env() -> int:
    """FREETOKEN_GGUF_KDA_NSPLIT: unset/empty or "2" = split, "1" = single launch.
    Strict whitelist read PER CALL (the v3a ft_gguf_moe_mtile pattern): no trim,
    no numeric parsing, so " 2"/"02"/"+2" stragglers fail fast like the C knob.
    Default 2: the split measured +2.61% e2e prefill (RTX 5090, MTILE=32) with
    bitwise-identical outputs (.tasks/dense-q80-gemm/ab-nsplit.md)."""
    v = os.environ.get("FREETOKEN_GGUF_KDA_NSPLIT")
    if v is None or v == "" or v == "2":
        return 2
    if v == "1":
        return 1
    raise ValueError(f"FREETOKEN_GGUF_KDA_NSPLIT must be 1 or 2, got {v!r}")


def kda_in_proj_forward(layer, x: torch.Tensor) -> torch.Tensor:
    """Glm5NextKDA's fused ``in_proj`` forward - the ONLY call site that may
    consume ``FREETOKEN_GGUF_KDA_NSPLIT`` (one helper, no knob ifs elsewhere).

    At ``2`` the dense MMQ GEMM (batch > 6) launches as two sub-N GEMMs at a
    32-row tile boundary so each launch's weight footprint fits L2: the
    production N=24896 is 108.35 MB > 96.0 MiB L2, two 12448-row launches are
    ~54.2 MB each. Disjoint 32-row tiles, k-only per-element reduction and
    deterministic x-quant make the split bitwise identical to the single launch
    (pinned by tests/kernels/test_gguf_quant.py). Everything outside the knob's
    scope - batch <= 6 (MMVQ vec kernels), non-MMQ formats, N not a multiple of
    32, N < 64 (a chunk would be < 32 rows), non-GGUF layers - falls back to the
    plain forward, byte identical. Do not flip the knob after boot: decode CUDA
    graphs capture the dense MMQ path for bs > 6 (engine/graph.py
    GraphRunner._capture_graphs), baking whatever structure was live at capture
    time - selection happens across boots, not mid-boot.
    """
    nsplit = _kda_nsplit_env()
    if nsplit == 1 or not isinstance(layer, GGUFLinear):
        return layer.forward(x)
    qw, qt = layer.qweight, layer._quant_type
    n_out, n_tok = qw.shape[0], x.shape[0]
    # the split sits on a 32-row tile edge to keep need_check=false; that flag is a
    # PERF guard keyed on N%32, NOT an exactness condition - bitwise parity rests on
    # the disjoint 32-col tiles, the k-only per-element reduction and the
    # deterministic per-launch x-quant
    if n_tok <= _MMVQ_SAFE or qt not in _MMQ or n_out < 64 or n_out % 32 != 0:
        return layer.forward(x)
    half = (n_out // 2) // 32 * 32
    out = torch.empty((n_tok, n_out), dtype=x.dtype, device=x.device)
    # per-output-row packing makes a dim-0 slice a plain view (no repack); the
    # narrow copy_ writes land each sub-launch result in the correct columns
    out[:, :half].copy_(fused_mul_mat_gguf(x, qw.narrow(0, 0, half), qt))
    out[:, half:].copy_(fused_mul_mat_gguf(x, qw.narrow(0, half, n_out - half), qt))
    if layer.bias is not None:
        out = out + layer.bias
    return out


class GGUFEmbedding(BaseOP):
    """Vocab embedding stored as a native GGUF block-quantized table.

    The full table is never dequantized: only the looked-up rows are gathered (in
    packed form) and dequantized per lookup, matching vLLM's ``_apply_gguf_embedding``.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        quant_type: int,
        embed_scale: float | None = None,
    ):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self._quant_type = quant_type
        self.qweight = torch.empty(
            num_embeddings, row_bytes(embedding_dim, quant_type), dtype=torch.uint8
        )
        self._embed_scale = embed_scale
        self._embed_scale_t: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.gguf import ggml_dequantize

        flat = x.flatten()
        rows = self.qweight.index_select(0, flat)  # [n, row_bytes] packed
        y = ggml_dequantize(rows, self._quant_type, flat.shape[0], self.embedding_dim, torch.bfloat16)
        y = y.view(*x.shape, self.embedding_dim)
        if self._embed_scale is not None:
            if self._embed_scale_t is None:
                self._embed_scale_t = torch.tensor(self._embed_scale, dtype=y.dtype, device=y.device)
            y = y * self._embed_scale_t
        return y


__all__ = ["GGUFLinear", "GGUFEmbedding", "fused_mul_mat_gguf", "kda_in_proj_forward"]
