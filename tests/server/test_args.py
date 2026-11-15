"""--linear-state-cache-ratio: parses onto ServerArgs and sizes the hybrid GDN pool.

ServerArgs inherits SchedulerConfig(EngineConfig), so the field already exists on the engine
side (engine/config.py, default 2.0); the flag only needs the parser entry.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from freetoken.core import Req
from freetoken.distributed import DistributedInfo
from freetoken.engine.config import EngineConfig
from freetoken.scheduler.decode import DecodeManager
from freetoken.kvcache.linear_state_pool import _linear_pool_num_slots
from freetoken.server.args import parse_args

ANON_PATH = "/models/anon"


class _Config:
    def to_dict(self):
        return {"architectures": ["Qwen3MoeForCausalLM"], "torch_dtype": "bfloat16"}


def _parse(extra):
    with patch("freetoken.utils.cached_load_hf_config", lambda _p: _Config()):
        args, _ = parse_args(["--model", ANON_PATH, *extra])
    return args


def test_linear_state_cache_ratio_default_is_engine_default():
    args = _parse([])
    assert args.linear_state_cache_ratio == 2.0
    assert EngineConfig.linear_state_cache_ratio == 2.0


def test_linear_state_cache_ratio_parses_float():
    assert _parse(["--linear-state-cache-ratio", "8"]).linear_state_cache_ratio == 8.0
    assert _parse(["--linear-state-cache-ratio", "0.5"]).linear_state_cache_ratio == 0.5


def test_linear_state_cache_ratio_sizes_the_pool():
    """mr=1: ratio 2.0 -> 9 slots (4 evictable), ratio 8 -> 13 slots (8 evictable)."""
    for ratio, slots in ((2.0, 9), (8.0, 13)):
        c = SimpleNamespace(max_running_req=1, cache_type="hybrid_radix",
                            linear_state_cache_ratio=ratio)
        assert _linear_pool_num_slots(c) == slots, (ratio, _linear_pool_num_slots(c))


def test_linear_state_cache_ratio_fractional_ceil():
    """2.5 * 3 -> extra = max(4, ceil(7.5)) = 8 (int() would truncate to 7):
    pool = 4*3 + 8 + 1 = 21."""
    c = SimpleNamespace(max_running_req=3, cache_type="hybrid_radix",
                        linear_state_cache_ratio=2.5)
    assert _linear_pool_num_slots(c) == 21


def test_linear_state_cache_ratio_rejects_non_positive():
    """<= 0 fails fast at engine-config adjustment (mirrors swa_full_tokens_ratio)
    instead of silently clamping to the 4-slot cache floor."""
    from freetoken.engine.engine import _adjust_config

    for bad in (0.0, -1.0):
        config = SimpleNamespace(
            model_config=SimpleNamespace(
                single_stream_only=False, dsv4_args=None, has_swa_attention=False,
                has_linear_attention=True, is_moe=True,
            ),
            max_running_req=4, cuda_graph_max_bs=None, linear_state_cache_ratio=bad,
        )
        with pytest.raises(ValueError, match="linear_state_cache_ratio must be > 0"):
            _adjust_config(config)


def test_mmproj_default_is_none_and_no_flag_keeps_field_unset():
    assert _parse([]).mmproj_path is None


def test_mmproj_flag_lands_on_engine_field(tmp_path):
    # validation (exists + arch clip) is deferred to the config-shim build
    # (build_gguf_shim), so argparse accepts a not-yet-existing path untouched
    assert _parse(["--mmproj", str(tmp_path / "absent.gguf")]).mmproj_path == str(
        tmp_path / "absent.gguf"
    )


def test_route_affinity_ladder_default_is_empty():
    """No flag: empty ladder, so the per-request field cannot change the batch value."""
    args = _parse([])
    assert args.route_affinity_ladder == ()
    assert EngineConfig.route_affinity_ladder == ()


def test_route_affinity_ladder_resolves_onto_the_server_default():
    """The server default stays capturable, values dedupe and sort for capture order."""
    args = _parse(["--route-affinity", "0.3", "--route-affinity-ladder", "0.3,0.05,0.3"])
    assert args.route_affinity_ladder == (0.05, 0.3)
    args = _parse(["--route-affinity-ladder", "0.05"])
    assert args.route_affinity_ladder == (0.0, 0.05)


def test_route_affinity_ladder_rejects_negative_or_non_finite():
    for bad in ("-1,0.3", "0.05,nan", "0.05,inf"):
        with pytest.raises(SystemExit):
            _parse(["--route-affinity-ladder", bad])


def test_engine_config_validates_the_ladder_programmatically():
    """The programmatic (non-CLI) path reaches __post_init__ unvalidated."""
    tp_info = DistributedInfo(rank=0, size=1)
    with pytest.raises(ValueError, match="route_affinity_ladder"):
        EngineConfig(
            model_path="anon", tp_info=tp_info, dtype=torch.bfloat16,
            route_affinity_ladder=(0.3, -0.1),
        )
    resolved = EngineConfig(
        model_path="anon", tp_info=tp_info, dtype=torch.bfloat16,
        route_affinity=0.3, route_affinity_ladder=(0.3, 0.05),
    )
    assert resolved.route_affinity_ladder == (0.05, 0.3)


@pytest.mark.parametrize(
    ("values", "ladder", "default", "value", "mismatch"),
    (
        ((None, None), (0.0, 0.05, 0.3), 0.0, 0.0, False),  # all default: snaps back to it
        ((None, 0.3), (0.05, 0.3), 0.3, 0.3, False),        # None -> server default, then snap
        ((0.08,), (0.0, 0.05, 0.3), 0.0, 0.05, False),      # nearest below
        ((0.29,), (0.0, 0.05, 0.3), 0.0, 0.3, False),       # nearest above
        ((0.1, 0.1), (0.1,), 0.3, 0.1, False),              # consistent after snapping
        ((0.1, 0.29), (0.0, 0.05, 0.3), 0.0, 0.0, True),    # disagreement: 0.0 fallback
        ((0.2, None), (), 0.3, 0.3, False),                 # no ladder: today's path
    ),
)
def test_batch_route_affinity_snap_rule(values, ladder, default, value, mismatch):
    from freetoken.engine.config import batch_route_affinity

    assert batch_route_affinity(values, ladder=ladder, default=default) == (value, mismatch)


def test_snap_route_affinity_ties_go_to_the_smaller_value():
    from freetoken.engine.config import snap_route_affinity

    assert snap_route_affinity(0.175, (0.05, 0.3)) == 0.05
    assert snap_route_affinity(0.175, (0.3, 0.05)) == 0.05


def test_decode_batch_route_affinity_warns_once_on_disagreement(monkeypatch):
    """The scheduler snaps at decode-batch build: consistent -> the value, mixed ->
    0.0 and a single warning for the whole server lifetime."""
    from freetoken.core import Req
    from freetoken.scheduler.decode import DecodeManager, logger as decode_logger

    warned = []
    # leave warning_once (the deduping wrapper) intact; record the logger call it
    # finally dispatches so three mismatched batches still warn exactly once
    monkeypatch.setattr(decode_logger, "warning", lambda msg: warned.append(msg))
    ladder, default = (0.05, 0.3), 0.0

    def req(uid, affinity):
        return Req(
            input_ids=torch.zeros(2, dtype=torch.int32), table_idx=0, cached_len=0,
            output_len=1, uid=uid, sampling_params=None, cache_handle=None,
            route_affinity=affinity,
        )

    dm = DecodeManager(page_size=1, route_affinity_ladder=ladder, route_affinity_default=default)
    dm.running_reqs = {req(1, 0.29), req(2, 0.26)}
    batch = dm.schedule_next_batch()
    assert batch.route_affinity == 0.3
    assert warned == []

    dm.running_reqs = {req(1, 0.29), req(2, 0.16)}
    batch = dm.schedule_next_batch()
    assert batch.route_affinity == 0.0
    dm.schedule_next_batch()
    dm.schedule_next_batch()
    assert len(warned) == 1  # once, not per step


def _affinity_req(uid: int, affinity: float) -> Req:
    return Req(
        input_ids=torch.zeros(2, dtype=torch.int32), table_idx=0, cached_len=0,
        output_len=1, uid=uid, sampling_params=None, cache_handle=None,
        route_affinity=affinity,
    )


def test_effective_ladder_resolves_once_for_runner_and_scheduler():
    """The gate lives in one resolver: the runner captures exactly what the scheduler
    snaps onto, so a request value can never key an uncaptured graph."""
    from freetoken.engine.config import resolve_route_affinity_ladder

    ladder = (0.0, 0.05, 0.3)
    assert resolve_route_affinity_ladder(
        ladder, moe_strategy="hybrid", has_route_affinity_setter=True
    ) == (ladder, None)
    # a non-offload decode captures no ladder graphs: the scheduler must see the same
    # empty result, so a request field snaps to the server default instead
    assert resolve_route_affinity_ladder(
        ladder, moe_strategy="cpu", has_route_affinity_setter=True
    ) == ((), "route affinity ladder applies to offload/hybrid MoE backends only "
            "(got 'cpu'); the ladder is ignored")
    # the same for a model without the setter (the engine captured nothing from the ladder)
    assert resolve_route_affinity_ladder(
        ladder, moe_strategy="offload", has_route_affinity_setter=False
    ) == ((), "this model has no set_route_affinity; ignoring the route affinity ladder")


def test_decode_manager_serves_the_default_when_the_ladder_dropped():
    """Empty effective ladder: a request's field must never produce an uncaptured
    (padded, value) key - the batch serves the server default the graphs key on."""
    dm = DecodeManager(page_size=1, route_affinity_default=0.3)
    dm.running_reqs = {_affinity_req(1, 0.3)}
    assert dm.schedule_next_batch().route_affinity == 0.3

    dm = DecodeManager(page_size=1, route_affinity_default=0.0)
    dm.running_reqs = {_affinity_req(1, 0.3)}
    assert dm.schedule_next_batch().route_affinity == 0.0
