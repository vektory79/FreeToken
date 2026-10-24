from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import os

from freetoken.engine.cache_budget import expert_bytes_per_slot, plan_cache_budget, resolve_moe_cache_auto
from freetoken.engine.engine import _pin_budget_bytes


def test_moe_priority_fills_experts_up_to_total():
    # budget large enough to cache every expert; KV gets the remainder.
    # per_expert=100, cache_per_page=10, total=8 experts (L*E), E=4.
    size, pages, overlap = plan_cache_budget(
        budget_bytes=2000, per_expert_bytes=100, cache_per_page=10,
        num_experts=4, total_experts=8, prefill_overlap=True,
        kv_reserve_pages=5, max_slots=8,
    )
    assert size == 8  # capped at full residency
    assert pages == (2000 - 8 * 100) // 10  # == 120, remainder to KV
    assert overlap is True


def test_offload_case_experts_take_most_kv_gets_reserve_floor():
    # budget too small for full residency: experts take what they can, KV keeps its floor.
    size, pages, overlap = plan_cache_budget(
        budget_bytes=1000, per_expert_bytes=100, cache_per_page=10,
        num_experts=2, total_experts=50, prefill_overlap=True,
        kv_reserve_pages=10, max_slots=50,
    )
    # raw = (1000 - 10*10) // 100 = 9 ; clamped to [4, 50] -> 9
    assert size == 9
    assert pages == max((1000 - 9 * 100) // 10, 10)  # remainder 10 pages, == floor
    assert overlap is True


def test_marlin_cap_clamps_count_and_rolls_bytes_to_kv():
    # budget would fund 1500 experts, but marlin caps at 992; freed bytes become KV pages.
    size, pages, overlap = plan_cache_budget(
        budget_bytes=200_000, per_expert_bytes=100, cache_per_page=10,
        num_experts=128, total_experts=4000, prefill_overlap=True,
        kv_reserve_pages=0, max_slots=992,
    )
    assert size == 992
    assert pages == (200_000 - 992 * 100) // 10


def test_small_cache_disables_prefill_overlap():
    # cap below 2*num_experts -> overlap impossible, falls back to num_experts floor.
    size, pages, overlap = plan_cache_budget(
        budget_bytes=10_000, per_expert_bytes=100, cache_per_page=10,
        num_experts=8, total_experts=12, prefill_overlap=True,
        kv_reserve_pages=0, max_slots=12,
    )
    assert overlap is False
    # raw = 10000//100 = 100, clamped to hi = min(12, 12) = 12.
    assert size == 12


def test_insufficient_kv_memory_raises():
    with pytest.raises(AssertionError, match="not enough memory"):
        plan_cache_budget(
            budget_bytes=410, per_expert_bytes=100, cache_per_page=10,
            num_experts=4, total_experts=4, prefill_overlap=False,
            kv_reserve_pages=0, max_slots=4,
        )  # experts eat 400, KV gets 1 page -> not > 1


def test_budget_too_small_for_min_moe_plus_reserve_raises():
    # Budget cannot fund even the minimum MoE slots + the KV reserve, so the floored plan
    # would exceed budget_bytes. Reject in arithmetic rather than OOM in a later CUDA alloc.
    with pytest.raises(AssertionError, match="budget too small"):
        plan_cache_budget(
            budget_bytes=300, per_expert_bytes=100, cache_per_page=10,
            num_experts=4, total_experts=4, prefill_overlap=False,
            kv_reserve_pages=10, max_slots=4,
        )  # min moe = 4 slots (400 B) + reserve (10 pages = 100 B) = 500 B > 300 B budget


def test_prefill_overlap_false_is_honored():
    # Even when the cache could fit 2*num_experts, an explicit False stays False.
    size, pages, overlap = plan_cache_budget(
        budget_bytes=2000, per_expert_bytes=100, cache_per_page=10,
        num_experts=4, total_experts=8, prefill_overlap=False,
        kv_reserve_pages=0, max_slots=8,
    )
    assert size == 8
    assert overlap is False
    assert pages == (2000 - 8 * 100) // 10


def test_expert_bytes_per_slot_sums_row_bytes_over_banks():
    sources = {
        "gate_up": [torch.zeros(4, 32, 8, dtype=torch.float16)],  # row = 32*8*2 = 512
        "down": [torch.zeros(4, 8, 16, dtype=torch.float16)],     # row = 8*16*2 = 256
    }
    assert expert_bytes_per_slot(sources) == 512 + 256


def test_resolve_auto_applies_ratio_once():
    # baseline 1000, weights 100, ratio 0.9 -> budget = 900 - 100 - 0(fixed) = 800
    size, pages, overlap = resolve_moe_cache_auto(
        baseline_free=1000, weights_bytes=100, memory_ratio=0.9,
        cache_per_page=10, fixed_cache_size=0, per_expert_bytes=50,
        num_experts=4, total_experts=8, prefill_overlap=True,
        kv_reserve_tokens=0, page_size=1,
    )
    # budget 800: experts cap at 8 -> 400 bytes; KV = 400//10 = 40 pages
    assert size == 8 and pages == 40 and overlap is True


def test_resolve_auto_caps_slots_at_the_kernel_limit():
    size, _, _ = resolve_moe_cache_auto(
        baseline_free=10_000_000, weights_bytes=0, memory_ratio=1.0,
        cache_per_page=10, fixed_cache_size=0, per_expert_bytes=100,
        num_experts=128, total_experts=4000, prefill_overlap=False,
        kv_reserve_tokens=0, page_size=1, max_slots=992,
    )
    assert size == 992


def _dsv4_adjust_cfg(**over):
    # A DSV4 _adjust_config stub mirroring the real checkpoint (ds_fp4 experts, dsv4_sparse
    # attention, offload MoE backend).
    from types import SimpleNamespace

    model_config = SimpleNamespace(
        single_stream_only=False, dsv4_args=SimpleNamespace(window_size=128), is_moe=True,
        expert_quant="ds_fp4", has_swa_attention=False, has_linear_attention=False,
    )

    class Cfg:
        moe_cache_auto = True
        moe_cache_size = 0
        moe_cache_rate = None
        moe_cache_mem = None
        moe_strategy = "offload"
        max_running_req = 1
        cuda_graph_max_bs = 1
        cuda_graph_bs = [1]
        max_seq_len = 1024
        max_extend_tokens = 4096
        page_size = 1
        attention_backend = "dsv4_sparse"
        moe_cpu_layers = None
        num_page_override = None
        num_token_override = None

        @property
        def model_config(self):
            return model_config

    cfg = Cfg()
    for k, v in over.items():
        object.__setattr__(cfg, k, v)
    return cfg


def test_adjust_config_allows_auto_for_dsv4():
    # DSV4 now supports --moe-cache-auto via the affine KV cost bridge (dsv4_auto_cost_model);
    # _adjust_config must NOT reject it.
    from freetoken.engine.engine import _adjust_config

    cfg = _dsv4_adjust_cfg()
    _adjust_config(cfg)  # must not raise
    assert cfg.moe_strategy == "offload"
    assert cfg.moe_cache_auto is True  # resolved later at engine init, not here
    assert cfg.page_size == 128  # DSV4's KV page is the P-token window page


def test_adjust_config_resolves_num_tokens_for_dsv4():
    # --num-tokens resolves AFTER every page_size override, so DSV4's P=128 page divides it.
    from freetoken.engine.engine import _adjust_config

    cfg = _dsv4_adjust_cfg(num_token_override=131072)
    _adjust_config(cfg)
    assert cfg.page_size == 128
    assert cfg.num_page_override == 1024


def test_adjust_config_rejects_num_tokens_not_multiple_of_page():
    from freetoken.engine.engine import _adjust_config

    cfg = _dsv4_adjust_cfg(num_token_override=131000)  # not a multiple of 128
    with pytest.raises(ValueError, match="not a multiple"):
        _adjust_config(cfg)


def test_adjust_config_rejects_num_tokens_with_num_pages():
    from freetoken.engine.engine import _adjust_config

    cfg = _dsv4_adjust_cfg(num_token_override=131072, num_page_override=1024)
    with pytest.raises(ValueError, match="mutually exclusive"):
        _adjust_config(cfg)


def test_adjust_config_resolves_num_tokens_generic():
    # Generic model keeps its page_size (1 here): tokens map 1:1 onto pages.
    from types import SimpleNamespace

    from freetoken.engine.engine import _adjust_config

    model_config = SimpleNamespace(
        single_stream_only=False, is_moe=False, expert_quant="none",
        has_swa_attention=False, has_linear_attention=False,
    )

    class Cfg:
        moe_cache_auto = False
        moe_cache_size = 0
        moe_cache_rate = None
        moe_cache_mem = None
        moe_strategy = "auto"
        max_running_req = 4
        cuda_graph_max_bs = 2
        cuda_graph_bs = [1, 2]
        max_seq_len = 1024
        page_size = 1
        attention_backend = "triton" if torch.version.hip is not None else "fi"
        num_page_override = None
        num_token_override = 5000

        @property
        def model_config(self):
            return model_config

    cfg = Cfg()
    _adjust_config(cfg)
    assert cfg.num_page_override == 5000


def test_mha_kv_cost_simple_full_attention():
    import torch

    from freetoken.kvcache.mha_pool import MHAKVCache
    from freetoken.models.config import KVCacheGroupSpec
    from freetoken.utils import div_even

    class StubModelConfig:
        has_swa_attention = False

        def kv_cache_group_specs(self):
            return [KVCacheGroupSpec(
                name="full", layer_ids=tuple(range(3)),
                num_kv_heads=8, head_dim=64, sliding_window=None,
            )]

        def linear_attention_group(self):
            return None

    class StubConfig:
        dtype = torch.bfloat16
        page_size = 16
        max_running_req = 4
        swa_full_tokens_ratio = 0.2
        swa_num_pages_override = None
        model_config = StubModelConfig()

        class tp_info:
            size = 1

    cache_per_page, fixed, _, _ = MHAKVCache.kv_cost(StubConfig())
    per_token = 2 * 64 * div_even(8, 1, allow_replicate=True) * 2 * 3
    assert cache_per_page == per_token * 16
    assert fixed == 0


def test_engine_resolve_auto_moe_cache_size_maps_kwargs():
    import torch

    from freetoken.engine.engine import Engine
    from freetoken.models.config import KVCacheGroupSpec

    class StubModelConfig:
        has_swa_attention = False
        num_experts = 4
        num_moe_layers = 2  # total_experts = 8

        def kv_cache_group_specs(self):
            return [KVCacheGroupSpec(
                name="full", layer_ids=(0, 1, 2), num_kv_heads=8, head_dim=64, sliding_window=None,
            )]

        def linear_attention_group(self):
            return None

    class StubConfig:
        dtype = torch.float16
        page_size = 16
        max_running_req = 4
        hybrid_swa_cache_mode = "auto"
        memory_ratio = 0.9
        moe_prefill_overlap = True
        kv_reserve_tokens = 0
        swa_full_tokens_ratio = 0.2
        swa_num_pages_override = None
        model_config = StubModelConfig()

        class tp_info:
            size = 1

    class StubBanks:
        # 2 layers (num_moe_layers above) x 4 experts each -- per-layer host bank contract.
        sources = {
            "gate_up": [torch.zeros(4, 32, 8, dtype=torch.float16)] * 2,  # row = 32*8*2 = 512
            "down": [torch.zeros(4, 8, 16, dtype=torch.float16)] * 2,     # row = 8*16*2 = 256
        }

    from freetoken.kvcache.mha_pool import MHAKVCache

    engine = Engine.__new__(Engine)  # bypass __init__/GPU
    engine._baseline_free = 10_000_000
    engine._weights_bytes = 1_000_000
    engine._pool_cls = MHAKVCache  # __init__ skipped -> install the generic pool family

    size, pages, overlap = engine._resolve_auto_moe_cache_size(StubConfig(), StubBanks())

    # cross-check against the same pure functions, proving the kwarg mapping is faithful
    from freetoken.engine.cache_budget import expert_bytes_per_slot, resolve_moe_cache_auto
    from freetoken.kvcache.mha_pool import MHAKVCache

    cache_per_page, fixed, _, _ = MHAKVCache.kv_cost(StubConfig())
    expected = resolve_moe_cache_auto(
        baseline_free=10_000_000, weights_bytes=1_000_000, memory_ratio=0.9,
        cache_per_page=cache_per_page, fixed_cache_size=fixed,
        per_expert_bytes=expert_bytes_per_slot(StubBanks.sources),
        num_experts=4, total_experts=8, prefill_overlap=True,
        kv_reserve_tokens=0, page_size=16,
    )
    assert (size, pages, overlap) == expected

    class StubMethod:
        def slot_limit(self):
            return 5

    size, _, _ = engine._resolve_auto_moe_cache_size(StubConfig(), StubBanks(), StubMethod())
    assert size == 5


# ---------------------------------------------------------------------------
# offload-cache sizing guard + auto-resolution (_require_offload_cache_size / _adjust_config),
# the floor rule compute_cache_floors documents above.
# ---------------------------------------------------------------------------


def _offload_engine_config(**overrides):
    """A frozen EngineConfig for a quantized-experts MoE checkpoint in the bare-invocation state
    (moe_strategy="auto") unless overridden — the shared fixture for the _adjust_config tests."""
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    config = EngineConfig(
        model_path="/tmp/freetoken-test-model",
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16,
        attention_backend="triton" if torch.version.hip is not None else "fi",
        **overrides,
    )
    object.__setattr__(
        config,
        "model_config",
        SimpleNamespace(
            has_swa_attention=False,
            has_linear_attention=False,
            is_moe=True,
            num_layers=10,
            num_moe_layers=10,
            num_experts=8,
            expert_quant="nvfp4",
            moe_strategy="auto",
        ),
    )
    return config


def test_guard_passes_when_size_covers_one_expert_per_layer():
    from freetoken.engine.engine import _require_offload_cache_size

    _require_offload_cache_size(cache_size=128, num_experts=128)  # no raise


def test_guard_raises_actionable_error_when_too_small():
    from freetoken.engine.engine import _require_offload_cache_size

    with pytest.raises(ValueError) as exc:
        _require_offload_cache_size(cache_size=0, num_experts=128)
    msg = str(exc.value)
    assert "128" in msg and "moe-cache" in msg


def test_adjust_config_defaults_moe_cache_auto_for_auto_resolved_offload_backend(monkeypatch):
    """Bare `ft serve <FTW MoE checkpoint>`: no --moe-backend, no --moe-cache-* flags at all.

    args.py's parse-time default only fires when the backend is *already*
    offload-family at parse time -- but a bare invocation leaves moe_strategy="auto" at parse
    time, and the "auto" -> offload/cpu/hybrid resolution only happens later, in _adjust_config,
    once the model_config (and its expert_quant) is known. This proves the engine-level
    resolution: a quantized-experts model auto-resolving to an offload-family backend also gets
    moe_cache_auto=True, so _init_offload_moe_cache's _require_offload_cache_size guard is never
    reached with moe_cache_size still 0.
    """
    import freetoken.engine.engine as engine_module
    from freetoken.engine.engine import _adjust_config
    from freetoken.moe import is_offload_moe_strategy

    # a unified-memory host (GB10) resolves auto to fused instead
    monkeypatch.setattr(engine_module, "_is_unified_memory_gpu", lambda index=None: False)
    config = _offload_engine_config()
    _adjust_config(config)

    # Which member of the family gets picked is not this test's claim. Assert the family, not the member.
    assert is_offload_moe_strategy(config.moe_strategy)
    assert config.moe_cache_auto is True
    assert config.moe_cache_size == 0  # still unresolved -- the scheduler sizes it from VRAM


def test_page_table_width_covers_whole_trailing_pages():
    # _write_page_table writes WHOLE trailing pages, so the width must reach the last
    # page's end, not just the next multiple of 32 (DSV4's P=128 exposed the gap).
    from freetoken.engine.engine import _page_table_width

    assert _page_table_width(4001, 128) == 4096   # align32 alone gave 4032 -> OOB
    assert _page_table_width(4096, 128) == 4096   # page-aligned length unchanged
    assert _page_table_width(100, 1) == 128       # page_size 1 degenerates to align32
    assert _page_table_width(33, 128) == 128
    for max_seq_len in (1, 31, 33, 4001, 4095, 4096):
        for page_size in (1, 32, 64, 128):
            w = _page_table_width(max_seq_len, page_size)
            last_col = -(-max_seq_len // page_size) * page_size - 1
            assert w > last_col and w % 32 == 0


def _generic_rotary_cfg(max_position, override, scaling=None):
    from types import SimpleNamespace

    from freetoken.models.config import RotaryConfig

    model_config = SimpleNamespace(
        single_stream_only=False, is_moe=False, expert_quant="none",
        has_swa_attention=False, has_linear_attention=False,
        rotary_config=RotaryConfig(
            head_dim=128, rotary_dim=128, max_position=max_position, base=1e4, scaling=scaling
        ),
    )

    class Cfg:
        moe_cache_auto = False
        moe_cache_size = 0
        moe_cache_rate = None
        moe_cache_mem = None
        moe_strategy = "auto"
        max_running_req = 4
        cuda_graph_max_bs = 2
        cuda_graph_bs = [1, 2]
        max_seq_len = 1024
        page_size = 1
        attention_backend = "triton"
        num_page_override = None
        num_token_override = None
        max_seq_len_override = None

        @property
        def model_config(self):
            return model_config

    cfg = Cfg()
    object.__setattr__(cfg, "max_seq_len_override", override)
    return cfg


def test_adjust_config_rejects_override_past_rope_table():
    from freetoken.engine.engine import _adjust_config

    with pytest.raises(ValueError, match="rope table"):
        _adjust_config(_generic_rotary_cfg(max_position=1024, override=2048))


def test_adjust_config_allows_override_at_rope_table_boundary():
    from freetoken.engine.engine import _adjust_config

    _adjust_config(_generic_rotary_cfg(max_position=1024, override=1024))  # must not raise


YARN_4X = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 1024}


def test_adjust_config_allows_override_to_the_yarn_extended_table():
    # A YaRN override that leaves max_position_embeddings at the trained length (Qwen3.8's 1M
    # recipe) serves original * factor positions; the gate must follow the table, not the config.
    from freetoken.engine.engine import _adjust_config

    _adjust_config(_generic_rotary_cfg(max_position=1024, override=4096, scaling=YARN_4X))
    with pytest.raises(ValueError, match=r"rope table \(4096 positions\)"):
        _adjust_config(_generic_rotary_cfg(max_position=1024, override=4097, scaling=YARN_4X))


def test_adjust_config_rope_gate_exempts_dsv4():
    # DSV4 sizes its own rope table from the resolved max_seq_len (_adjust_dsv4_config),
    # so the generic gate must not fire even when the override dwarfs max_position.
    from types import SimpleNamespace

    from freetoken.engine.engine import _adjust_config

    cfg = _dsv4_adjust_cfg(max_seq_len_override=10_000_000)
    cfg.model_config.rotary_config = SimpleNamespace(max_position=1024)
    _adjust_config(cfg)  # must not raise


# ---- _pin_budget_bytes: host bytes already pinned outside the expert banks ----


def test_reserved_subtracts_from_the_cap(monkeypatch):
    monkeypatch.setenv("FREETOKEN_PIN_BUDGET_GB", "2")
    assert _pin_budget_bytes() == 2 * 2**30
    assert _pin_budget_bytes(reserved=2**30) == 2**30
    assert _pin_budget_bytes(reserved=4 * 2**30) == 0


def test_uncapped_platform_stays_uncapped(monkeypatch):
    monkeypatch.delenv("FREETOKEN_PIN_BUDGET_GB", raising=False)
    if hasattr(os, "uname") and "microsoft" in os.uname().release.lower():
        pytest.skip("WSL caps pinning")
    assert _pin_budget_bytes(reserved=2**30) is None


def test_adjust_config_rejects_gguf_experts_on_cpu_paths():
    # gguf moe_weight_format still has no CPU-resident or fused expert path: cpu/fused
    # picks must fail at config time. offload passes the gate, and hybrid is accepted for
    # a capable per-layer type set (see the hybrid gate tests below).
    from freetoken.engine.engine import _adjust_config

    model_config = SimpleNamespace(
        single_stream_only=False,
        is_moe=True,
        expert_quant="none",
        moe_weight_format="gguf",
        hidden_act="swiglu_clamp",
        has_swa_attention=False,
        has_linear_attention=False,
    )

    class Cfg:
        moe_cache_auto = False
        moe_cache_size = 0
        moe_cache_rate = None
        moe_cache_mem = None
        moe_strategy = "cpu"
        moe_cpu_layers = None
        max_running_req = 4
        cuda_graph_max_bs = 2
        cuda_graph_bs = [1, 2]
        max_seq_len = 1024
        page_size = 1
        attention_backend = "fi"
        num_page_override = None
        num_token_override = None

        @property
        def model_config(self):
            return model_config

    cfg = Cfg()
    with pytest.raises(ValueError, match="gguf moe_weight_format supports offload only"):
        _adjust_config(cfg)

    cfg.moe_strategy = "fused"
    with pytest.raises(ValueError, match="gguf moe_weight_format supports offload only"):
        _adjust_config(cfg)

    cfg.moe_strategy = "offload"
    _adjust_config(cfg)  # the only wired expert path passes the gate
    assert cfg.moe_strategy == "offload"




# ---- gguf + hybrid gate: per-layer header types vs the executor capability set ----


def _write_expert_types_gguf(path, layer_types) -> str:
    """Minimal .gguf carrying ONLY the ffn_{gate,up,down}_exps stacks, typed per layer.

    The gates scan the tensor table alone (GGUFReader mmaps; no bank data is touched),
    so one-quant-block payloads are enough: a uint8 (block * type_size)-byte payload
    decodes to a ggml shape the reader's own n_bytes math accepts (its innermost dim
    must be a whole number of quant blocks). Hand-rolled GGUF v3 bytes: the vendored
    gguf package ships a reader only, and the pip writer left main's dep set."""
    import struct

    from freetoken.third_party.gguf.constants import GGML_QUANT_SIZES

    tensors = []
    for blk, types in sorted(layer_types.items()):
        if len(types) == 2:  # (gate_up, down): the combined ffn_gate_up_exps stacking (gemma4)
            for role, tid in zip(("gate_up", "down"), types):
                block, type_size = GGML_QUANT_SIZES[tid]
                tensors.append((f"blk.{blk}.ffn_{role}_exps.weight", tid, (block * type_size, 1, 1), block * type_size))
            continue
        for role, tid in zip(("gate", "up", "down"), types):
            block, type_size = GGML_QUANT_SIZES[tid]
            tensors.append((f"blk.{blk}.ffn_{role}_exps.weight", tid, (block * type_size, 1, 1), block * type_size))
    buf = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 0)
    offsets = []
    pos = 0
    for name, tid, ne, nbytes in tensors:
        offsets.append(pos)
        pos += nbytes
        nq = name.encode()
        buf += struct.pack("<Q", len(nq)) + nq
        buf += struct.pack("<I", len(ne)) + struct.pack(f"<{len(ne)}Q", *ne)
        buf += struct.pack("<IQ", tid, offsets[-1])
    data_start = (len(buf) + 31) // 32 * 32
    with open(path, "wb") as f:
        f.write(buf)
        f.write(b"\x00" * (data_start - len(buf)))
        for (name, tid, ne, nbytes), off in zip(tensors, offsets):
            assert off == f.tell() - data_start
            f.write(b"\x00" * nbytes)
    return str(path)


def _write_gguf_tensors(path, tensors) -> str:
    """Hand-rolled GGUF v3 bytes (the vendored gguf package ships a reader only):
    ``tensors`` are (name, ggml type id, ne, payload bytes) triples, names and shapes
    unconstrained - the gates under test read the tensor table alone."""
    import struct

    buf = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 0)
    offsets = []
    pos = 0
    for name, tid, ne, nbytes in tensors:
        offsets.append(pos)
        pos += nbytes
        nq = name.encode()
        buf += struct.pack("<Q", len(nq)) + nq
        buf += struct.pack("<I", len(ne)) + struct.pack(f"<{len(ne)}Q", *ne)
        buf += struct.pack("<IQ", tid, offsets[-1])
    data_start = (len(buf) + 31) // 32 * 32
    with open(path, "wb") as f:
        f.write(buf)
        f.write(b"\x00" * (data_start - len(buf)))
        for (_, _, _, nbytes), off in zip(tensors, offsets):
            assert off == f.tell() - data_start
            f.write(b"\x00" * nbytes)
    return str(path)


def _write_3d_expert_gguf(path, layer_shapes, num_experts) -> str:
    """Expert stacks in the real packed shape, ``ne = (K, rows, E)`` in ggml order:
    Q8_0 gate/up stacks of ``rows`` rows and down stacks, one 32-element block per
    row (34 bytes). ``layer_shapes`` = {blk: (I, H)}."""
    tensors = []
    for blk, (I, H) in sorted(layer_shapes.items()):
        for role, rows in (("gate", I), ("up", I), ("down", H)):
            tensors.append((f"blk.{blk}.ffn_{role}_exps.weight", 8, (32, rows, num_experts), num_experts * rows * 34))
    return _write_gguf_tensors(path, tensors)


def _gguf_gate_model_config(**extra):
    return SimpleNamespace(
        single_stream_only=False,
        is_moe=True,
        expert_quant="none",
        moe_weight_format="gguf",
        hidden_act="swiglu_clamp",
        has_swa_attention=False,
        has_linear_attention=False,
        num_moe_layers=2,
        **extra,
    )


def _gguf_gate_cfg(model_config, strategy, model_path=None):
    # mirror the duck-typed Cfg of test_adjust_config_rejects_gguf_experts_on_cpu_paths,
    # plus model_path (the header scan is the only consumer)
    return SimpleNamespace(
        moe_cache_auto=False,
        moe_cache_size=0,
        moe_cache_rate=None,
        moe_strategy=strategy,
        moe_cpu_layers=None,
        model_path=model_path,
        max_running_req=4,
        cuda_graph_max_bs=2,
        cuda_graph_bs=[1, 2],
        max_seq_len=1024,
        page_size=1,
        attention_backend="fi",
        num_page_override=None,
        num_token_override=None,
        model_config=model_config,
    )


def test_adjust_config_accepts_gguf_hybrid_for_capable_types(tmp_path):
    # glm5next packs (18, 18, 23) / (18, 18, 14) / (23, 23, 14) bank types, all inside
    # the CPU executor's capability set (the ggml-cpu leg serves every ggml type;
    # Task 02): the explicit hybrid pick must pass the gate instead of a blanket
    # offload-only rejection.
    from freetoken.engine.engine import _adjust_config

    path = _write_expert_types_gguf(tmp_path / "capable.gguf", {0: (18, 18, 23), 1: (18, 18, 14)})
    cfg = _gguf_gate_cfg(_gguf_gate_model_config(), "hybrid", model_path=path)
    _adjust_config(cfg)  # must not raise
    assert cfg.moe_strategy == "hybrid"


def test_adjust_config_rejects_gguf_hybrid_for_incapable_types(tmp_path, monkeypatch):
    # A build whose _cpu_moe extension lacks the gguf format: the gate refuses the
    # hybrid pick at config time and names the offending layer formats instead of
    # crashing in the executor after the whole weight load. On HEAD (no gate) the
    # explicit hybrid pick sailed through to the post-load NotImplementedError.
    from freetoken.engine.engine import _adjust_config
    from freetoken.kernel import _cpu_moe

    monkeypatch.setattr(_cpu_moe, "weight_formats", lambda: ["bf16", "mxfp4_triton"])
    path = _write_expert_types_gguf(tmp_path / "incapable.gguf", {0: (18, 18, 23), 1: (8, 8, 14)})
    cfg = _gguf_gate_cfg(_gguf_gate_model_config(), "hybrid", model_path=path)
    with pytest.raises(ValueError, match=r"MoE layers \[0, 1\] need \['gguf'\]"):
        _adjust_config(cfg)


def test_adjust_config_rejects_gguf_hybrid_without_ggml_cpu(tmp_path, monkeypatch):
    # the other real incapability: no ggml-cpu kernels (non-x86-64 build) -- every
    # gguf expert type is then unservable on the CPU, whatever its type id
    import freetoken.kernel.ggml_cpu as ggml_cpu
    from freetoken.engine.engine import _adjust_config

    monkeypatch.setattr(ggml_cpu, "select", lambda: None)
    path = _write_expert_types_gguf(tmp_path / "capable.gguf", {0: (18, 18, 23), 1: (18, 18, 14)})
    cfg = _gguf_gate_cfg(_gguf_gate_model_config(), "hybrid", model_path=path)
    with pytest.raises(ValueError, match="no ggml-cpu kernels"):
        _adjust_config(cfg)


def test_adjust_config_rejects_gguf_hybrid_when_types_unverifiable():
    # Fail-safe direction: without a scannable .gguf file the gate cannot prove that
    # every bank type is CPU-executable, so hybrid stays rejected (offload keeps working).
    from freetoken.engine.engine import _adjust_config

    cfg = _gguf_gate_cfg(_gguf_gate_model_config(), "hybrid", model_path=None)
    with pytest.raises(ValueError, match="cannot be verified from the file header"):
        _adjust_config(cfg)


def test_adjust_config_rejects_gguf_hybrid_for_mismatched_gate_up_types(tmp_path):
    # the loader hard-fails on gate/up type divergence; the gate says so pre-load
    from freetoken.engine.engine import _adjust_config

    path = _write_expert_types_gguf(tmp_path / "mixed.gguf", {0: (18, 23, 14)})
    model_config = _gguf_gate_model_config()
    model_config.num_moe_layers = 1  # the scan only answers on a complete bank set
    cfg = _gguf_gate_cfg(model_config, "hybrid", model_path=path)
    with pytest.raises(ValueError, match="gate and up experts have different ggml types"):
        _adjust_config(cfg)


try:
    from freetoken.kernel import _cpu_moe as _cpu_moe_ext  # noqa: F401

    _HAVE_CPU_MOE_EXT = True
except ImportError:
    _HAVE_CPU_MOE_EXT = False


# ---- D3 gates: capability screen, header-scan rejection, pre-load auto floor gate ----


def test_gguf_expert_bank_types_scans_the_header(tmp_path):
    from freetoken.moe.expert_banks import gguf_expert_bank_types, gguf_signature_groups

    path = _write_expert_types_gguf(tmp_path / "m.gguf", {0: (18, 18, 23), 1: (18, 18, 14)})
    model_config = _gguf_gate_model_config()
    assert gguf_expert_bank_types(path, model_config) == {0: (18, 18, 23), 1: (18, 18, 14)}
    groups, slots = gguf_signature_groups(path, model_config)
    # layers of different (type, rows, row_bytes) signatures never share a pool
    assert [sorted(g) for g in sorted(groups, key=lambda g: g[0])] == [[0], [1]]
    assert all(s > 0 for s in slots)

    # incomplete bank set: unverifiable layouts must never pass a gate
    partial = _write_expert_types_gguf(tmp_path / "p.gguf", {0: (18, 18, 23)})
    assert gguf_expert_bank_types(partial, model_config) is None  # config wants 2 layers
    one = _gguf_gate_model_config()
    one.num_moe_layers = 1
    assert gguf_expert_bank_types(partial, one) == {0: (18, 18, 23)}
    assert gguf_expert_bank_types(None, one) is None
    assert gguf_expert_bank_types(str(tmp_path / "absent.gguf"), one) is None

    # combined ffn_gate_up_exps stacking scans as (t, t, down)
    combined = _write_expert_types_gguf(tmp_path / "c.gguf", {0: (23, 14)})
    assert gguf_expert_bank_types(combined, one) == {0: (23, 23, 14)}


def test_cpu_moe_format_rejection_names_the_layers(monkeypatch):
    from freetoken.kernel import _cpu_moe
    from freetoken.engine.engine import _cpu_moe_format_rejection

    real = list(_cpu_moe.weight_formats())
    monkeypatch.setattr(_cpu_moe, "weight_formats", lambda: real)
    ok = [("gguf", {"gate_up_type": 18, "down_type": 23}), ("bf16", {})]
    assert _cpu_moe_format_rejection(ok, "silu") is None
    assert _cpu_moe_format_rejection([], "silu") is None  # no layers: nothing to reject

    bad = [("gguf", {}), ("nvfp4_marlin", {}), (None, {})]
    rejection = _cpu_moe_format_rejection(bad, "silu")
    assert rejection and "layers [1, 2]" in rejection and "'nvfp4_marlin'" in rejection
    assert "--moe-strategy offload" in rejection

    assert "relu" in _cpu_moe_format_rejection(ok, "relu")


def test_cpu_moe_format_rejection_without_ggml_cpu(monkeypatch):
    import freetoken.kernel.ggml_cpu as ggml_cpu
    from freetoken.engine.engine import _cpu_moe_format_rejection

    monkeypatch.setattr(ggml_cpu, "select", lambda: None)
    rejection = _cpu_moe_format_rejection([("gguf", {})], "silu")
    assert rejection and "no ggml-cpu kernels" in rejection
    assert _cpu_moe_format_rejection([("bf16", {})], "silu") is None  # gguf-only check


def test_cpu_moe_format_rejection_mirrors_the_executor_mxfp4_act_skip(monkeypatch):
    # the executor skips its ABI probe when every layer is mxfp4_triton (the act runs
    # inside the mxfp4 kernel, which predates the max_generic_act_id marker): a screen
    # that still demands compiled_extension_supports falsely rejects a servable
    # mxfp4 + swigluoai pick on a stale prebuilt .so
    from freetoken.moe import cpu_executor
    from freetoken.engine.engine import _cpu_moe_format_rejection

    monkeypatch.setattr(cpu_executor, "compiled_extension_supports", lambda act: False)
    assert _cpu_moe_format_rejection([("mxfp4_triton", {})], "swigluoai") is None
    rejection = _cpu_moe_format_rejection([("mxfp4_triton", {}), ("bf16", {})], "swigluoai")
    assert rejection and "predates activation 'swigluoai'" in rejection


def test_cpu_moe_format_rejection_silent_for_non_gguf_expert_formats():
    # the screen rejects formats the executor cannot serve, not gguf-ness: an explicit
    # cpu/hybrid pick on bf16 or nvfp4 experts passes it untouched
    from freetoken.engine.engine import _cpu_moe_format_rejection

    assert _cpu_moe_format_rejection([("bf16", {})], "silu") is None
    assert _cpu_moe_format_rejection([("nvfp4", {})], "silu") is None
    assert _cpu_moe_format_rejection([("bf16", {}), ("nvfp4", {})], "silu") is None


def test_gate_wiring_precedes_the_bank_load():
    # fails-before on HEAD: the only CPU-executor capability checks lived in
    # CpuMoeExecutor.__init__, strictly AFTER the multi-minute bank load, and nothing
    # called the orphaned check_partition_floors before it
    import inspect

    from freetoken.engine import engine as engine_mod
    from freetoken.engine.engine import Engine

    init_src = inspect.getsource(Engine._init_offload_moe_cache)
    # main re-platform moved the CPU-executor capability screen to _adjust_config
    # (through _gguf_hybrid_rejection); the header-scan floor gate stays wired
    # immediately before the multi-minute bank load
    assert init_src.index("_gguf_auto_floor_gate") < init_src.index("load_expert_banks(")
    assert "_cpu_moe_format_rejection" in inspect.getsource(engine_mod._gguf_hybrid_rejection)
    assert "_gguf_hybrid_rejection" in inspect.getsource(engine_mod._adjust_config)


def _floor_gate_engine(envelope_holder):
    """Engine.__new__ stand-in whose auto-plan resolve is replaced by the (size, pages,
    overlap) triple stored in ``envelope_holder['envelope']`` -- the gate must feed the
    header-scan groups through check_partition_floors against it."""
    from freetoken.engine.engine import Engine

    engine = Engine.__new__(Engine)

    def fake_resolve(config, banks, method=None, max_slots=None, per_expert_bytes=None):
        envelope_holder["per_expert_bytes"] = per_expert_bytes
        return envelope_holder["envelope"], 128, False

    engine._resolve_auto_moe_cache_size = fake_resolve
    return engine


def test_gguf_auto_floor_gate_rejects_an_unfundable_multi_signature_plan(tmp_path):
    from freetoken.engine.engine import Engine
    from freetoken.moe.expert_banks import gguf_signature_groups

    path = _write_expert_types_gguf(tmp_path / "m.gguf", {0: (18, 18, 23), 1: (18, 18, 14), 2: (23, 23, 14)})
    model_config = _gguf_gate_model_config()
    model_config.num_moe_layers = 3
    model_config.num_experts = 8
    cfg = _gguf_gate_cfg(model_config, "offload", model_path=path)
    cfg.moe_cache_auto = True
    cfg.use_dummy_weight = False
    groups, slots = gguf_signature_groups(path, model_config)
    assert len(groups) == 3

    holder = {"envelope": 15}  # layer-0-width slots: below the 3 x 8-slot floors
    with pytest.raises(ValueError, match="slot floors"):
        Engine._gguf_auto_floor_gate(_floor_gate_engine(holder), cfg, method=None)


def test_gguf_auto_floor_gate_passes_a_fundable_plan_and_skips_one_pool(tmp_path):
    from freetoken.engine.engine import Engine

    path = _write_expert_types_gguf(tmp_path / "m.gguf", {0: (18, 18, 23), 1: (18, 18, 14), 2: (23, 23, 14)})
    model_config = _gguf_gate_model_config()
    model_config.num_moe_layers = 3
    model_config.num_experts = 8
    cfg = _gguf_gate_cfg(model_config, "offload", model_path=path)
    cfg.moe_cache_auto = True
    cfg.use_dummy_weight = False

    holder = {"envelope": 10**6}
    Engine._gguf_auto_floor_gate(_floor_gate_engine(holder), cfg, method=None)
    assert holder["per_expert_bytes"] > 0  # the plan was denominated in the header's mean width

    uniform = _write_expert_types_gguf(tmp_path / "u.gguf", {0: (18, 18, 23), 1: (18, 18, 23)})
    cfg = _gguf_gate_cfg(_gguf_gate_model_config(), "offload", model_path=uniform)
    cfg.moe_cache_auto = True
    cfg.use_dummy_weight = False
    holder = {"envelope": 0, "called": False}

    def no_resolve(config, banks, method=None, max_slots=None, per_expert_bytes=None):
        holder["called"] = True
        return 0, 0, False

    engine = Engine.__new__(Engine)
    engine._resolve_auto_moe_cache_size = no_resolve
    Engine._gguf_auto_floor_gate(engine, cfg, method=None)
    assert not holder["called"]  # single signature -> one pool -> the plan floor covers it


def test_late_split_checks_verify_totals_not_signature_floors():
    # the per-signature floors are pre-load-only (the gguf auto gate): the checks that
    # run after the plan is materialized into pools verify totals - split_slots' pool
    # minima and _require_offload_cache_size's num_experts floor - so an envelope that
    # funds the totals but not every signature's whole-expert floor boots with the
    # documented small-pool degradation (clamped decode batch) instead of a late raise
    from freetoken.engine.cache_budget import check_partition_floors
    from freetoken.engine.engine import _require_offload_cache_size
    from freetoken.moe.pool_policy import group_layers, make_pools, split_slots

    num_experts = 8
    pools = make_pools(group_layers("unified", ["a", "b", "c"], [1, 2, 4]), num_experts, min(num_experts, 1), None)
    total = 10  # funds the totals (3 pool minima, >= num_experts) but not the 3 x 8-slot floors
    sizes = split_slots(total, pools)
    assert sum(sizes) == total and all(1 <= s <= num_experts for s in sizes)
    _require_offload_cache_size(total, num_experts)  # no raise: the documented degradation
    # the same numbers ARE floor-checked pre-load, where the header scan can prove them
    with pytest.raises(ValueError, match="slot floors"):
        check_partition_floors([[2], [1], [0]], [4, 2, 1], num_experts, total)


def test_gguf_signature_groups_slot_bytes_match_layer_slot_bytes_on_3d_banks(tmp_path):
    # the scan's row_bytes = nbytes // (ne[1]*ne[2]) was only exercised on degenerate
    # ne=(x, 1, 1) fixtures before: on the real packed shape ne=(K, rows, E) the
    # per-group slot bytes must equal layer_slot_bytes of the banks the loader
    # materializes from the same file
    from freetoken.engine.cache_budget import layer_slot_bytes
    from freetoken.models.gguf.experts import load_gguf_expert_banks
    from freetoken.moe.expert_banks import gguf_signature_groups

    E, I, H = 4, 6, 10
    path = _write_3d_expert_gguf(tmp_path / "m.gguf", {0: (I, H), 1: (I, H), 2: (2 * I, H)}, E)
    model_config = _gguf_gate_model_config()
    model_config.num_moe_layers = 3
    groups, slots = gguf_signature_groups(path, model_config)
    banks = load_gguf_expert_banks(path, 3, device=torch.device("cpu"))
    per_layer = layer_slot_bytes(banks.sources)
    assert [sorted(g) for g in sorted(groups, key=lambda g: g[0])] == [[0, 1], [2]]
    assert slots == [per_layer[g[0]] for g in sorted(groups, key=lambda g: g[0])]
    assert slots[0] == 2 * I * 34 + H * 34  # gate|up rows then down rows, one 34-byte row each


def test_gguf_signature_groups_fail_safe_on_unparsable_layer_name(tmp_path):
    # a bank tensor named blk.<not-a-number>... must return None groups (the
    # silently-None scanner contract), not leak a raw ValueError out of the scan
    from freetoken.moe.expert_banks import gguf_signature_groups

    tensors = [
        (f"blk.0.ffn_{role}_exps.weight", 8, (32, 4, 2), 2 * 4 * 34)
        for role in ("gate", "up", "down")
    ]
    tensors.append(("blk.bad.ffn_gate_exps.weight", 8, (32, 4, 2), 2 * 4 * 34))
    path = _write_gguf_tensors(tmp_path / "bad-name.gguf", tensors)
    assert gguf_signature_groups(path, _gguf_gate_model_config()) is None


def test_recipe_gguf_786k_hybrid_0_85_passes_the_budget_assert():
    # the re-counted glm5next-GGUF hybrid recipe (spec (f)): no prefill overlap, 0.85,
    # 786k nvfp4 KV - the greedy MoE fill must fit the budget with the KV reserve.
    budget = resolve_moe_cache_auto(
        baseline_free=int(29.6 * 2**30), weights_bytes=3 * 2**30, memory_ratio=0.85,
        cache_per_page=430_000, fixed_cache_size=int(1.2 * 2**30),
        per_expert_bytes=11_400_000, num_experts=288, total_experts=8 * 288,
        prefill_overlap=False, kv_reserve_tokens=786_432, page_size=64,
    )
    size, pages, overlap = budget
    assert overlap is False and pages >= 12_288 and 1400 < size < 2304
    total = size * 11_400_000 + pages * 430_000
    assert total <= int(0.85 * int(29.6 * 2**30)) - 3 * 2**30 - int(1.2 * 2**30)


def test_recipe_nvfp4_786k_hybrid_fails_the_budget_assert():
    # the same recipe on the NVFP4-FTW model (per_slot 14.4 MB, dense 17 GiB) honestly
    # dies in the plan arithmetic - the failure that killed the pre-merge recipe
    with pytest.raises(AssertionError, match="cache budget too small: minimum plan"):
        resolve_moe_cache_auto(
            baseline_free=int(29.6 * 2**30), weights_bytes=17 * 2**30, memory_ratio=0.85,
            cache_per_page=430_000, fixed_cache_size=int(1.2 * 2**30),
            per_expert_bytes=15_100_000, num_experts=288, total_experts=8 * 288,
            prefill_overlap=True, kv_reserve_tokens=786_432, page_size=64,
        )


@pytest.mark.skipif(not _HAVE_CPU_MOE_EXT, reason="compiled _cpu_moe extension missing")
@pytest.mark.skip(reason="D3 verdict: per-type viability is obsolete on the new stack - the ggml-cpu leg serves every ggml type, so _cpu_moe_executor_viable stays format-level")
def test_cpu_moe_executor_viable_reads_gguf_types(tmp_path):
    # --moe-cpu-layers auto consults this: the "gguf" alias alone must not count as
    # viable - the file's per-layer types decide, so an incapable set fails at config
    # time instead of post-load in the executor.
    from freetoken.engine.engine import _cpu_moe_executor_viable

    def model_config(num_layers):
        # silu: no dependency on the compiled extension's act table, only its presence
        return SimpleNamespace(
            single_stream_only=False,
            is_moe=True,
            expert_quant="none",
            moe_weight_format="gguf",
            hidden_act="silu",
            has_swa_attention=False,
            has_linear_attention=False,
            num_moe_layers=num_layers,
        )

    capable = _write_expert_types_gguf(tmp_path / "capable.gguf", {0: (18, 18, 23), 1: (23, 23, 14)})
    assert _cpu_moe_executor_viable(model_config(2), capable) is True
    incapable = _write_expert_types_gguf(tmp_path / "incapable.gguf", {0: (8, 8, 8)})
    assert _cpu_moe_executor_viable(model_config(1), incapable) is False
    assert _cpu_moe_executor_viable(model_config(1), None) is False  # unscannable -> not viable
    # flat formats keep their alias behavior
    flat = SimpleNamespace(hidden_act="silu", expert_quant="nvfp4")
    assert _cpu_moe_executor_viable(flat) is True


def _write_ftw_gguf_types_dir(path, rows, *, quant_format="gguf", include_types=True) -> str:
    # meta-only FTW dir (no banks needed): the capability gates read the index json alone
    from freetoken.checkpoint.ftw import FTWWriter

    w = FTWWriter(str(path))
    w.add_tensor("token_embd.weight", torch.zeros(4, dtype=torch.bfloat16))
    meta = {"quant_format": quant_format}
    if include_types:
        meta["gguf_types"] = [list(r) for r in rows]
    w.finalize(meta)
    return str(path)


def test_adjust_config_accepts_gguf_hybrid_for_ftw_types(tmp_path):
    # the FTW analog of the capable-types gate: the persisted index meta answers the
    # same capability question for an FTW dir, where the bare-gguf scan cannot look
    from freetoken.engine.engine import _adjust_config

    path = _write_ftw_gguf_types_dir(tmp_path / "ckpt", [(18, 18, 23), (18, 18, 14)])
    cfg = _gguf_gate_cfg(_gguf_gate_model_config(), "hybrid", model_path=path)
    _adjust_config(cfg)  # must not raise
    assert cfg.moe_strategy == "hybrid"


def test_adjust_config_rejects_gguf_hybrid_for_ftw_without_types(tmp_path):
    # an FTW dir's types are not yet verifiable (the gguf_types meta rides the D4
    # loader re-land): hybrid stays rejected at config time (fail-safe), offload keeps
    # working
    from freetoken.engine.engine import _adjust_config

    path = _write_ftw_gguf_types_dir(tmp_path / "ckpt", [], include_types=False)
    cfg = _gguf_gate_cfg(_gguf_gate_model_config(), "hybrid", model_path=path)
    with pytest.raises(ValueError, match="cannot be verified from the file header"):
        _adjust_config(cfg)


@pytest.mark.skipif(not _HAVE_CPU_MOE_EXT, reason="compiled _cpu_moe extension missing")
@pytest.mark.skip(reason="D3: branch per-signature partition executor machinery dropped in round-5 merge (main per-format slot pools); deep adaptation deferred")
def test_cpu_moe_executor_viable_reads_ftw_types(tmp_path):
    # --moe-cpu-layers auto must see an FTW gguf checkpoint's persisted types too:
    # capable types make it viable where the bare-gguf-only scan said "not viable"
    from freetoken.engine.engine import _cpu_moe_executor_viable

    def model_config(num_layers):
        return SimpleNamespace(
            single_stream_only=False,
            is_moe=True,
            expert_quant="none",
            moe_weight_format="gguf",
            hidden_act="silu",
            has_swa_attention=False,
            has_linear_attention=False,
            num_moe_layers=num_layers,
        )

    capable = _write_ftw_gguf_types_dir(tmp_path / "capable", [(18, 18, 23), (23, 23, 14)])
    assert _cpu_moe_executor_viable(model_config(2), capable) is True
    incapable = _write_ftw_gguf_types_dir(tmp_path / "incapable", [(8, 8, 8)])
    assert _cpu_moe_executor_viable(model_config(1), incapable) is False


# ---- Task 07: offload + explicit --moe-cpu-layers on gguf (engine-stage failures) ----


@pytest.mark.skipif(not _HAVE_CPU_MOE_EXT, reason="compiled _cpu_moe extension missing")
def test_adjust_config_leaves_offload_explicit_cpu_layers_to_the_engine(tmp_path):
    # Unlike hybrid, offload + EXPLICIT --moe-cpu-layers ids has no config-time type
    # gate: the loud failure is the engine-stage format screen / eager executor
    # construction, both strictly before CUDA graph capture.
    from freetoken.engine.engine import _adjust_config

    path = _write_expert_types_gguf(tmp_path / "incapable.gguf", {0: (18, 18, 23), 1: (8, 8, 14)})
    cfg = _gguf_gate_cfg(_gguf_gate_model_config(), "offload", model_path=path)
    cfg.moe_cpu_layers = "1"
    _adjust_config(cfg)  # must not raise: the engine stage owns the loud failure


def test_moe_slot_pool_split(tmp_path):
    import json

    from freetoken.moe.pool_policy import group_layers, load_profile, make_pools, split_slots

    pools = make_pools(group_layers("per-layer", [0] * 4, [1] * 4), num_experts=16, min_num_slots=8, slot_limit=None)
    assert split_slots(41, pools) == [11, 10, 10, 10]
    with pytest.raises(ValueError, match="need"):
        split_slots(31, pools)
    with pytest.raises(ValueError, match="hold"):
        split_slots(65, pools)
    # one pool takes the whole cache, as before pools existed
    assert split_slots(100, make_pools(group_layers("unified", [0] * 4, [1] * 4), 16, 8, None)) == [100]
    # unified pools one bank layout each, widest slots first
    assert group_layers("unified", ["a", "b", "a", "b"], [1, 3, 1, 3]) == [[1, 3], [0, 2]]

    # layer 0 uses two experts, layer 1 has a hot half and a cold half, layers 2 and 3 use all 16 evenly
    counts = [[5] * 2 + [0] * 14, [3] * 8 + [1] * 8, [1] * 16, [1] * 16]
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"model": "toy", "expert_counts": counts}))
    assert split_slots(48, pools, load_profile(path, 4, 16)) == [8, 8, 16, 16]
    with pytest.raises(ValueError, match="needs"):
        load_profile(path, 4, 32)



def test_moe_pool_flags_and_decode_batch_clamp():
    from unittest.mock import patch

    from freetoken.engine.engine import _clamp_decode_batch_to_pools
    from freetoken.server.args import parse_args

    hf = SimpleNamespace(to_dict=lambda: {"architectures": ["Qwen3MoeForCausalLM"], "torch_dtype": "bfloat16"})
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: hf):
        args, _ = parse_args([
            "--model", "/models/anon", "--moe-cache-mem", "1.5GiB",
            "--moe-cache-scope", "per-layer", "--moe-cache-split", "profile:/p.json",
        ])
        assert (args.moe_cache_mem, args.moe_cache_scope, args.moe_cache_split) == (3 << 29, "per-layer", "profile:/p.json")
        for bad in (["--moe-cache-mem", "12"], ["--moe-cache-mem", "1GiB", "--moe-cache-size", "512"]):
            with pytest.raises(SystemExit):
                parse_args(["--model", "/models/anon", *bad])

    # GPU decode needs batch * top_k distinct experts to fit the smallest pool below num_experts
    config = SimpleNamespace(
        model_config=SimpleNamespace(num_experts=16, num_experts_per_tok=4),
        max_running_req=4, cuda_graph_max_bs=None, cuda_graph_bs=[1, 2, 4],
    )
    _clamp_decode_batch_to_pools(config, [12, 16, 9])
    assert (config.max_running_req, config.cuda_graph_max_bs, config.cuda_graph_bs) == (2, 2, [1, 2])
