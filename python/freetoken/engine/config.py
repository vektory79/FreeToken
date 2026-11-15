from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace, FrozenInstanceError
from functools import cached_property
from typing import TYPE_CHECKING, Any, Iterable, List, Mapping, Sequence

import math
import torch
from freetoken.distributed import DistributedInfo
from freetoken.layers.quantization import set_quant_config
from freetoken.mm.config import ENCODER_SECTIONS, MultimodalConfig
from freetoken.models.register import EncoderSpec, ModelSpec, _load_attr, checkpoint_quant_config, get_model_spec
from freetoken.utils import cached_load_hf_config, init_logger

if TYPE_CHECKING:
    from freetoken.models import ModelConfig
    from freetoken.models.config import SlotStateSpec

logger = init_logger(__name__)


def snap_route_affinity(value: float, ladder: Sequence[float]) -> float:
    """The ladder value a request's affinity snaps to: nearest, ties -> smaller."""
    return min(ladder, key=lambda a: (abs(a - value), a))


def batch_route_affinity(
    req_values: Iterable[float | None], *, ladder: Sequence[float] | None, default: float
) -> tuple[float, bool]:
    """The single route affinity a decode batch serves at, and whether requests disagreed.

    The one snap point: a request value None resolves to the server default first, then
    snaps to the nearest ladder value (ties -> smaller). A decode batch can only serve
    one affinity, so a disagreement falls back to 0.0 (the caller warns once). Without
    a ladder nothing snaps and the batch serves the server default - exactly today's
    behavior.
    """
    if not ladder:
        return default, False
    snapped = {
        snap_route_affinity(default if value is None else value, ladder) for value in req_values
    }
    if len(snapped) == 1:
        return snapped.pop(), False
    return 0.0, True


def resolve_route_affinity_ladder(
    ladder: Sequence[float] | None, *, moe_strategy: str, has_route_affinity_setter: bool
) -> tuple[tuple[float, ...], str | None]:
    """The ladder that actually has captured decode graphs, and why it was dropped.

    The one resolution point: GraphRunner captures a decode graph per (bs, affinity),
    so a value outside the ladder means a permanent eager decode, and DecodeManager
    snaps the batch onto the same result. Both sides call this instead of gating the
    raw flag themselves.
    """
    ladder = tuple(ladder or ())
    if not ladder:
        return (), None
    if moe_strategy not in ("offload", "hybrid"):
        return (), (
            f"route affinity ladder applies to offload/hybrid MoE backends only "
            f"(got {moe_strategy!r}); the ladder is ignored"
        )
    if not has_route_affinity_setter:
        return (), "this model has no set_route_affinity; ignoring the route affinity ladder"
    return ladder, None


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    # --hf-overrides: applied to the checkpoint config the model is built from (cached_load_hf_config)
    hf_overrides: Mapping[str, Any] = field(default_factory=dict)
    max_running_req: int = 4
    attention_backend: str = "auto"
    moe_strategy: str = "auto"
    # old name of moe_strategy; __post_init__ folds it in
    moe_backend: str | None = field(default=None, repr=False)
    # --quant-backend: layer[.kind]=kernel entries, comma separated
    quant_backend: str | None = None
    # PLE table backend: "disk" (default) reads rows from the checkpoint files per fill, "pinned" preloads the table into page-locked host RAM.
    ple_backend: str = "disk"
    # --embed-weights: "gpu" (default) keeps the input embedding tables in VRAM; "host" keeps the untied ones in pinned host RAM, read zero-copy by the lookup.
    embed_weights: str = "gpu"
    # DeepSeek-V4.1 Decoder SWA Bounded Replay: "bounded" (default) runs the decoder layers on each
    # prompt's last window; "exact" runs them on every token.
    swa_decoder_replay: str = "bounded"
    # Expert-bank host load (--expert-load): auto|serial|parallel. "auto" reads scattered
    # experts in parallel but falls back to serial when free RAM can't cover the banks + the
    # parallel reader's extra (non-reclaimable) whole-shard buffer; "serial" forces the
    # low-memory reclaimable read; "parallel" forces the fast read.
    expert_load: str = "auto"
    moe_cache_size: int = 0
    moe_cache_rate: float | None = None
    moe_cache_auto: bool = False
    # --moe-cache-mem: slot-cache budget in bytes, turned into a slot count once the expert banks are loaded
    moe_cache_mem: int | None = None
    # --moe-cache-scope: unified (one LRU pool per expert quant) or per-layer (one pool per layer)
    moe_cache_scope: str = "unified"
    # --moe-cache-split: how pools share the slots; equal or profile:<path> (by expert use from a profile file)
    moe_cache_split: str = "equal"
    kv_reserve_tokens: int = 8192  # KV floor for --moe-cache-auto; small by design (MoE-priority)
    moe_cache_policy: str = "lru"
    moe_prefill_overlap: bool = True
    # Prefill hit/miss split: serve cache-resident experts D2D during prefill
    # prefetch instead of re-streaming the full layer over PCIe. Needs CUDA >= 13.0
    # (cudaMemcpyBatchAsync) or HIP (hipMemcpyBatchAsync);
    # no-op unless moe_cache_size > 2 * num_experts.
    moe_prefill_hit_d2d: bool = False
    moe_collect_stats: bool = False  # capture decode miss-rate counters into the cuda graph
    # Maya-style route affinity (--route-affinity): bonus for experts resident in the MoE
    # slot cache on the router's SELECTION score (mix weights stay unbiased). 0.0 = off;
    # effective only for sigmoid routers on the offload/hybrid backends.
    route_affinity: float = 0.0
    # --route-affinity-ladder: affinity values with captured decode graphs, for
    # per-request route_affinity. Resolved in __post_init__ to the sorted, deduplicated
    # flag values plus the server default; empty = no ladder (the default everywhere).
    route_affinity_ladder: tuple[float, ...] = ()
    # CPU decode expert threads (--moe-strategy cpu/hybrid, or offload with
    # --moe-cpu-layers): 0 = auto (one per physical core, pinned). One executor
    # pool serves every expert format, so the budget is global, not per partition.
    moe_cpu_threads: int = 0
    # Hybrid CPU/GPU decode (--moe-strategy offload only): which MoE layers decode on
    # the CPU executor instead of the GPU offload/PCIe path. Spec is an explicit id
    # list ("3,7,11"), a count ("8" -> 8 layers evenly strided across depth), or a
    # fraction ("0.5"). None/"" = all layers on GPU (plain offload). --moe-strategy cpu
    # already means all layers on CPU and ignores this.
    moe_cpu_layers: str | None = None
    # Hybrid MoE backend (--moe-strategy hybrid): max experts fetched over PCIe per
    # (layer, decode step); the rest of that step's misses are computed on the CPU.
    # -1 (default) = auto: fetch the benched pcie_bw/cpu_bw fraction of each step's
    # misses so the PCIe fetch and the CPU compute finish together (perfect overlap);
    # falls back to a fixed cap of 1 without a usable `ft bench bw` profile.
    moe_hybrid_max_fetch: int = -1
    # Hybrid with locked layers (banks the pin quota could not take, OS-locked instead): a locked layer keeps GPU
    # slots of its own pools and fetches this share of each decode step's misses through pinned staging; the rest
    # are computed on the CPU. -1 (default) = auto: per layer from its `ft bench bw` row (bench_profile.locked_fetch_fraction),
    # a fixed cap of 1 without one. 0 = locked layers decode on the CPU only. moe_locked_max_fetch caps the fetches per
    # (layer, step) and sizes the staging (that many rows of the widest expert, per bank).
    moe_locked_fetch: float = -1.0
    moe_locked_max_fetch: int = 8
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    page_size: int = 1
    # KV-cache storage quantization: "none" stores the compute dtype, "fp8" stores e4m3
    # codes plus one fp32 scale per (token, slab, layer, kv head) -- about 2x the tokens
    # per GiB, at a small accuracy cost. --kv-cache-dtype; resolved from "auto" by
    # _adjust_config, which also refuses it on a pool family or attention backend that
    # cannot read the scales.
    kv_quant: str = "none"
    memory_ratio: float = 0.9
    # Hybrid GDN models default to the HybridRadixCache (cross-request GDN-state prefix reuse);
    # `--cache-type naive` opts out. linear_state_cache_ratio sizes the GDN snapshot cache as
    # ceil(ratio * max_running_req) extra slots.
    linear_state_cache_ratio: float = 2.0
    session_tier_ram_gib: float = 0.0
    session_tier_dir: str | None = None
    session_tier_ssd_gib: float = 0.0
    # Window/full ratio for the SWA radix cache (`--cache-type radix` on SWA models) and the DSV4
    # window tier: the DEFAULT window-pool size = max(working-set floor, ratio x full-pool tokens).
    # < 1.0 trades retained window-prefix capacity for memory savings; must be in (0, 1]. It is the
    # DSV4 window/full ratio directly. Used only when swa_num_pages_override is None (a runtime
    # rebuild can pin an absolute window instead).
    swa_full_tokens_ratio: float = 0.2
    # Absolute window-pool size in the pool's own pages (usable, dummy excluded); None -> use the
    # ratio default above. A runtime cache rebuild sets this (num_swa_pages) to pin the window
    # regardless of the full anchor; the ratio is the startup default and the fallback.
    swa_num_pages_override: int | None = None
    distributed_timeout: float = 60.0
    use_dummy_weight: bool = False
    use_pynccl: bool = True
    # speculative decoding
    speculative_algorithm: str | None = None
    speculative_num_steps: int = 0
    speculative_draft_model_path: str | None = None
    # the proposer's states on the target's linear-state slots, resolved by _adjust_config
    spec_slot_states: tuple[SlotStateSpec, ...] = ()
    max_seq_len_override: int | None = None
    num_page_override: int | None = None  # if not None, will override the number of pages
    # KV capacity in tokens; resolved into num_page_override by _adjust_config once page_size
    # is final. Mutually exclusive with num_page_override.
    num_token_override: int | None = None
    # Raw-GGUF vision boot: path to a clip-architecture mmproj GGUF carrying the vision
    # tower + projector. Unset serves a GGUF checkpoint text-only. Validated where the
    # config shim is built (build_gguf_shim), not at argparse time.
    mmproj_path: str | None = None
    # Runtime knobs of the multimodal path; the architecture side (vision_config, mrope) lives in ModelConfig.
    mm: MultimodalConfig = field(default_factory=MultimodalConfig)

    def __post_init__(self):
        if not (self.moe_locked_fetch == -1 or 0.0 <= self.moe_locked_fetch <= 1.0):
            raise ValueError(f"moe_locked_fetch must be -1 (auto) or in [0, 1], got {self.moe_locked_fetch}")
        if self.moe_locked_fetch > 0 and round(self.moe_locked_fetch * (1 << 16)) == 0:
            # the ensure kernel takes the share in Q16: this one rounds to 0 and the layers would never fetch
            raise ValueError(f"moe_locked_fetch {self.moe_locked_fetch} rounds to 0 in Q16; pass 0 to keep locked layers CPU-only")
        if self.moe_locked_max_fetch < 1:
            raise ValueError(f"moe_locked_max_fetch must be at least 1, got {self.moe_locked_max_fetch}")
        if self.route_affinity < 0:
            raise ValueError(f"route_affinity must be >= 0, got {self.route_affinity}")
        if self.route_affinity_ladder:
            if any(not math.isfinite(v) or v < 0 for v in self.route_affinity_ladder):
                raise ValueError(
                    f"route_affinity_ladder must be finite values >= 0, got {self.route_affinity_ladder}"
                )
            # ladder mode: the server default stays capturable; sorted, deduplicated so
            # the graph capture order is deterministic
            object.__setattr__(
                self,
                "route_affinity_ladder",
                tuple(sorted(set(self.route_affinity_ladder) | {self.route_affinity})),
            )
        if self.moe_backend is None:
            return
        if self.moe_strategy != "auto":
            raise ValueError("moe_backend is the old name of moe_strategy; pass only moe_strategy")
        logger.warning("EngineConfig.moe_backend is deprecated; use moe_strategy")
        object.__setattr__(self, "moe_strategy", self.moe_backend)
        object.__setattr__(self, "moe_backend", None)

    @cached_property
    def hf_config(self):
        return cached_load_hf_config(
            self.model_path, self.hf_overrides, mmproj_path=self.mmproj_path
        )

    @cached_property
    def model_spec(self) -> ModelSpec:
        return get_model_spec(self.hf_config.architectures[0])

    @cached_property
    def active_encoders(self) -> tuple[EncoderSpec, ...]:
        """The encoder towers this process builds: the family registers them, the checkpoint config carries their section, --mm-disable did not name them."""
        return tuple(
            e
            for e in self.model_spec.encoders
            if getattr(self.hf_config, e.config_key, None) is not None
            and e.kind not in self.mm.disabled_encoders
        )

    @cached_property
    def served_modalities(self) -> frozenset[str]:
        """Modalities this process accepts."""
        return frozenset(m for e in self.active_encoders for m in e.modalities)

    @cached_property
    def model_config(self) -> ModelConfig:
        # the parser sees no section for a tower this process does not build (for the vision tower that also means 1-D rope)
        hf_config = copy.copy(self.hf_config)
        built = {e.config_key for e in self.active_encoders}
        for key in set(ENCODER_SECTIONS) | {e.config_key for e in self.model_spec.encoders}:
            if key not in built and getattr(hf_config, key, None) is not None:
                try:
                    setattr(hf_config, key, None)
                except FrozenInstanceError:
                    # the --mmproj GgufConfigShim carries vision_config as a real field;
                    # rebuild the frozen shim without the section this process skips
                    hf_config = replace(hf_config, **{key: None})
        spec = self.model_spec
        quant = checkpoint_quant_config(self.model_path, hf_config, spec)
        set_quant_config(quant)
        model_config = _load_attr(spec.module, spec.parse_config)(hf_config)
        return replace(model_config, quant=quant)

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.table_positions

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def decode_rows_per_req(self) -> int:
        """Query rows each running request takes per decode step; every per-request decode allowance derives from it."""
        return self.speculative_num_steps + 1 if self.speculative_algorithm is not None else 1

    @property
    def slot_states(self) -> tuple[SlotStateSpec, ...]:
        """Every per-request state the linear-state pool holds: the model's own, then the proposer's."""
        return self.model_config.slot_states + self.spec_slot_states

    @property
    def distributed_addr(self) -> str:
        return "tcp://127.0.0.1:2333"
