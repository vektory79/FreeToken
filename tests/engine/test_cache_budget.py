from __future__ import annotations

from types import SimpleNamespace

import numpy as np
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
        moe_strategy = "auto"
        max_running_req = 4
        cuda_graph_max_bs = 2
        cuda_graph_bs = [1, 2]
        max_seq_len = 1024
        page_size = 1
        attention_backend = "fi"
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
        attention_backend="fi",
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
            expert_quant="nvfp4",  # quantized experts -> must resolve to an offload backend
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


def test_adjust_config_defaults_moe_cache_auto_for_auto_resolved_offload_backend():
    """Bare `ft serve <FTW MoE checkpoint>`: no --moe-backend, no --moe-cache-* flags at all.

    args.py's parse-time default only fires when the backend is *already*
    offload-family at parse time -- but a bare invocation leaves moe_strategy="auto" at parse
    time, and the "auto" -> offload/cpu/hybrid resolution only happens later, in _adjust_config,
    once the model_config (and its expert_quant) is known. This proves the engine-level
    resolution: a quantized-experts model auto-resolving to an offload-family backend also gets
    moe_cache_auto=True, so _init_offload_moe_cache's _require_offload_cache_size guard is never
    reached with moe_cache_size still 0.
    """
    from freetoken.engine.engine import _adjust_config
    from freetoken.moe import is_offload_moe_strategy

    config = _offload_engine_config()
    _adjust_config(config)

    # Which member of the family gets picked is not this test's claim, and is not ours to
    # decide: a bare "auto" consults ~/.cache/freetoken/benchbw.json, so a box that has run
    # `ft bench bw` resolves nvfp4 experts to hybrid instead. Assert the family, not the member.
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


def _generic_rotary_cfg(max_position, override):
    from types import SimpleNamespace

    model_config = SimpleNamespace(
        single_stream_only=False, is_moe=False, expert_quant="none",
        has_swa_attention=False, has_linear_attention=False,
        rotary_config=SimpleNamespace(max_position=max_position),
    )

    class Cfg:
        moe_cache_auto = False
        moe_cache_size = 0
        moe_cache_rate = None
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


def test_adjust_config_auto_keeps_gguf_on_offload_despite_hybrid_profile(monkeypatch):
    # The benchbw profile may carry a "gguf" row (the reader maps it through the
    # identity format mapping), but the auto->hybrid upgrade must ignore it: gguf has
    # no CPU MoE weight path, so the profile can never justify hybrid for it.
    import freetoken.moe.bench_profile as bench_profile
    from types import SimpleNamespace

    monkeypatch.setattr(bench_profile, "load_backend_recommendation", lambda *a, **k: "hybrid")

    def make_cfg():
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
            moe_strategy = "auto"
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

        return Cfg()

    from freetoken.engine.engine import _adjust_config

    cfg = make_cfg()
    _adjust_config(cfg)
    assert cfg.moe_strategy == "offload"

    # contrast: the same (monkeypatched) profile DOES upgrade a plain bf16 MoE model,
    # so the exclusion above is really the gguf format check
    cfg2 = make_cfg()
    cfg2.model_config.moe_weight_format = None
    _adjust_config(cfg2)
    assert cfg2.moe_strategy == "hybrid"


# ---- gguf + hybrid gate: per-layer header types vs the executor capability set ----


def _write_expert_types_gguf(path, layer_types) -> str:
    """Minimal .gguf carrying ONLY the ffn_{gate,up,down}_exps stacks, typed per layer.

    The hybrid gate scans the tensor table alone (GGUFReader mmaps; no bank data is
    touched), so one-quant-block payloads are enough: the byte shape (block, type_size)
    decodes to the ggml shape (block, block) iter_gguf_tensors expects."""
    import gguf

    w = gguf.GGUFWriter(str(path), "glm5next")
    for blk, types in sorted(layer_types.items()):
        for role, tid in zip(("gate", "up", "down"), types):
            qt = gguf.GGMLQuantizationType(tid)
            block, type_size = gguf.GGML_QUANT_SIZES[qt]
            w.add_tensor(
                f"blk.{blk}.ffn_{role}_exps.weight",
                np.zeros((block, type_size), np.uint8),
                raw_dtype=qt,
            )
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)


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
    # the CPU executor's capability set (Task 02): the explicit hybrid pick must pass
    # the gate instead of the old blanket offload-only rejection.
    from freetoken.engine.engine import _adjust_config

    path = _write_expert_types_gguf(tmp_path / "capable.gguf", {0: (18, 18, 23), 1: (18, 18, 14)})
    cfg = _gguf_gate_cfg(_gguf_gate_model_config(), "hybrid", model_path=path)
    _adjust_config(cfg)  # must not raise
    assert cfg.moe_strategy == "hybrid"


def test_adjust_config_rejects_gguf_hybrid_for_incapable_types(tmp_path):
    # Q8_0 (type id 8) banks have no CPU GEMV: the gate refuses the hybrid pick at
    # config time and names the offending type id instead of crashing in the executor
    # after the whole weight load.
    from freetoken.engine.engine import _adjust_config

    path = _write_expert_types_gguf(tmp_path / "incapable.gguf", {0: (18, 18, 23), 1: (8, 8, 14)})
    cfg = _gguf_gate_cfg(_gguf_gate_model_config(), "hybrid", model_path=path)
    with pytest.raises(ValueError, match=r"types \[8\] have no CPU GEMV"):
        _adjust_config(cfg)


def test_adjust_config_rejects_gguf_hybrid_when_types_unverifiable():
    # Fail-safe direction: without a scannable .gguf file the gate cannot prove that
    # every bank type is CPU-executable, so hybrid stays rejected (offload keeps working).
    from freetoken.engine.engine import _adjust_config

    cfg = _gguf_gate_cfg(_gguf_gate_model_config(), "hybrid", model_path=None)
    with pytest.raises(ValueError, match="cannot be verified from the file header"):
        _adjust_config(cfg)


try:
    from freetoken.kernel import _cpu_moe as _cpu_moe_ext  # noqa: F401

    _HAVE_CPU_MOE_EXT = True
except ImportError:
    _HAVE_CPU_MOE_EXT = False


@pytest.mark.skipif(not _HAVE_CPU_MOE_EXT, reason="compiled _cpu_moe extension missing")
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


# ---- Task 07: offload + explicit --moe-cpu-layers on gguf (engine-stage failures) ----


def _capability_stub(sig, num_layers, E=4):
    """Minimal decode_target=cpu gguf cache for the partition screen (no banks: the
    screen reads quant_format + gguf_types only)."""
    from freetoken.moe.offload_cache import OffloadMoeCache

    return OffloadMoeCache(
        num_layers=num_layers, num_experts=E, cache_size=2 * E,
        device=torch.device("cpu"), quant_format="gguf",
        gguf_types=tuple([tuple(sig)] * num_layers), decode_target="cpu",
    )


def _bank_stub_cache(sig, num_layers, E=4, H=512, I=256):
    """OffloadMoeCache + shape-correct zero banks the executor construction can
    resolve (per-role widths follow the ROLE's own gguf type, never one shared row
    width)."""
    from freetoken.models.gguf.dequant import BLOCK_SHAPE
    from freetoken.moe.offload_cache import OffloadMoeCache

    cache = OffloadMoeCache(
        num_layers=num_layers, num_experts=E, cache_size=2 * E,
        device=torch.device("cpu"), quant_format="gguf",
        gguf_types=tuple([tuple(sig)] * num_layers), decode_target="cpu",
    )
    cache.bank_sources = {
        "gate": [torch.zeros(E, I, (H // 256) * BLOCK_SHAPE[sig[0]][1], dtype=torch.uint8) for _ in range(num_layers)],
        "up": [torch.zeros(E, I, (H // 256) * BLOCK_SHAPE[sig[1]][1], dtype=torch.uint8) for _ in range(num_layers)],
        "down": [torch.zeros(E, H, (I // 256) * BLOCK_SHAPE[sig[2]][1], dtype=torch.uint8) for _ in range(num_layers)],
    }
    return cache


def test_adjust_config_leaves_offload_explicit_cpu_layers_to_the_engine(tmp_path):
    # Unlike hybrid, offload + EXPLICIT --moe-cpu-layers ids has no config-time type
    # gate: the loud failure is the engine's partition screen / eager executor
    # construction, both strictly before CUDA graph capture.
    from freetoken.engine.engine import _adjust_config

    path = _write_expert_types_gguf(tmp_path / "incapable.gguf", {0: (18, 18, 23), 1: (8, 8, 14)})
    cfg = _gguf_gate_cfg(_gguf_gate_model_config(), "offload", model_path=path)
    cfg.moe_cpu_layers = "1"
    _adjust_config(cfg)  # must not raise: the engine stage owns the loud failure


def test_multi_partition_gate_advice_follows_the_strategy():
    # When the failing boot IS offload + --moe-cpu-layers, "use --moe-strategy
    # offload instead" is stale advice: the remedy is adjusting/dropping the flag.
    # Explicit cpu/hybrid strategies keep the offload fallback advice.
    from freetoken.engine.engine import _multi_partition_gate_advice

    offload = _multi_partition_gate_advice(SimpleNamespace(moe_strategy="offload", moe_cpu_layers="1"))
    assert "moe-cpu-layers" in offload
    assert "moe-strategy offload" not in offload
    for strategy in ("cpu", "hybrid"):
        advice = _multi_partition_gate_advice(SimpleNamespace(moe_strategy=strategy, moe_cpu_layers=None))
        assert "--moe-strategy offload" in advice


def test_offload_cpu_layers_gguf_partition_screen_names_mid_partition(tmp_path):
    # Task 07 B-5(a): a real mixed-signature file -> header scan -> capability
    # stubs. The incapable partition sits MID-list and is named with its members;
    # the offload + --moe-cpu-layers boot gets the adjust/drop remedy, and the
    # raise site is wired to the strategy-aware advice.
    import inspect

    from freetoken.engine.engine import Engine, _multi_partition_gate_advice
    from freetoken.moe.expert_banks import gguf_expert_bank_types

    path = _write_expert_types_gguf(
        tmp_path / "mixed.gguf", {0: (18, 18, 23), 1: (8, 8, 14), 2: (18, 18, 14)}
    )
    model_config = _gguf_gate_model_config()
    model_config.num_moe_layers = 3  # the scan only answers on a complete bank set
    types_by_layer = gguf_expert_bank_types(path, model_config)
    assert types_by_layer is not None

    groups: dict[tuple, list[int]] = {}
    for lid, types in types_by_layer.items():
        groups.setdefault(tuple(types), []).append(lid)
    members_lists = list(groups.values())
    caches = [_capability_stub(sig, len(members)) for sig, members in groups.items()]

    rejections = Engine._partition_executor_rejections(members_lists, caches)
    assert [(m, "no CPU GEMV" in r and "8" in r) for m, r in rejections] == [([1], True)]

    advice = _multi_partition_gate_advice(SimpleNamespace(moe_strategy="offload", moe_cpu_layers="1"))
    assert "moe-cpu-layers" in advice and "moe-strategy offload" not in advice
    assert "_multi_partition_gate_advice" in inspect.getsource(Engine._init_offload_moe_cache)


def test_offload_cpu_layers_gguf_capable_ids_boot_and_incapable_die_eagerly():
    # Task 07 B-5(a): with capable explicit ids the executor stage BOOTS (one pool
    # per signature partition, disjoint cores); an incapable single-signature boot
    # dies in eager CpuMoeExecutor construction (strictly before graph capture),
    # naming the type and the adjust/drop remedy for the offload + cpu-layers boot.
    from freetoken.engine.engine import Engine

    config = SimpleNamespace(moe_cpu_threads=3, max_running_req=4, cuda_graph_max_bs=2)
    layers = [
        SimpleNamespace(
            top_k=2, activation="swiglu_clamp", apply_router_weight_on_input=False,
            quant_method=None, alpha=1.0, limit=10.0,
        )
    ]

    # capable ids: the boot gets its per-partition executors
    caches = []
    for sig in ((18, 18, 23), (18, 18, 14), (23, 23, 14)):
        cache = _bank_stub_cache(sig, 1)
        cache.cpu_layer_ids = frozenset({0})  # the engine attaches per-partition residency
        caches.append(cache)
    fake = SimpleNamespace(device=torch.device("cpu"), cpu_moe_executors=[])
    Engine._init_cpu_moe_executors(fake, config, caches, layers)
    assert len(fake.cpu_moe_executors) == 3
    assert fake.cpu_moe_executors[0].label.startswith("pool 1/3")
    flat = [c for ex in fake.cpu_moe_executors for c in ex.core_ids]
    assert len(flat) == len(set(flat)), "per-partition pools must pin disjoint cores"

    # incapable id on a single-signature file: eager construction fails first
    bad = _bank_stub_cache((18, 18, 8), 3)  # Q8_0 down rows: no CPU GEMV
    bad.cpu_layer_ids = frozenset({1})  # a strict subset: offload + --moe-cpu-layers 1
    with pytest.raises(NotImplementedError, match=r"type 8 has no CPU GEMV.*moe-cpu-layers"):
        Engine._init_cpu_moe_executors(
            SimpleNamespace(device=torch.device("cpu"), cpu_moe_executors=[]),
            config, [bad], layers,
        )
