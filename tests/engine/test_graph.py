"""GraphRunner capture: the forwards run on the dummy request's page, and the MoE offload cache stays warm from the eager run into the capture."""

from contextlib import contextmanager
from types import SimpleNamespace

import torch

import freetoken.core as core
from freetoken.core import Batch, Context, Req, get_global_ctx
from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.engine.graph import GraphRunner


def test_capture_runs_on_the_dummy_page_with_a_warm_offload_cache(monkeypatch):
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    ctx = Context(page_size=1)
    # row 0 is a live request holding slot 0, row 1 is the dummy request on the dummy page
    ctx.page_table = torch.tensor([[0], [9]], dtype=torch.int32)
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    events = []

    @contextmanager
    def cuda_graph(graph, pool=None, stream=None):
        events.append("capture")
        yield
        events.append("captured")

    def forward():
        batch = get_global_ctx().batch
        events.append(batch.out_loc.tolist())
        return torch.zeros(batch.size, 3)

    def noop(*args, **kwargs):
        return None

    monkeypatch.setattr("torch.cuda.CUDAGraph", lambda: SimpleNamespace(pool=lambda: "pool"))
    monkeypatch.setattr("torch.cuda.graph", cuda_graph)
    for name in ("synchronize", "empty_cache", "reset_peak_memory_stats"):
        monkeypatch.setattr(f"torch.cuda.{name}", noop)
    monkeypatch.setattr("freetoken.engine.graph.get_free_memory", lambda device: 0)
    monkeypatch.setattr("freetoken.engine.graph._upload_graph", noop)

    dummy = Req(input_ids=torch.zeros(1, dtype=torch.int32), table_idx=1, cached_len=0, output_len=1, uid=-1,
                sampling_params=None, cache_handle=None)
    GraphRunner(stream=None, device=torch.device("cpu"), forward=forward,
                attn_backend=SimpleNamespace(init_capture_graph=noop, prepare_for_capture=noop),
                cuda_graph_bs=[2], cuda_graph_max_bs=None, free_memory=0, max_seq_len=1, dummy_req=dummy,
                moe_offload_cache=SimpleNamespace(reset=lambda: events.append("reset")))

    assert events == ["reset", [9, 9], "capture", [9, 9], "captured", "reset", "reset"]


def _ladder_runner(monkeypatch, *, affinity_values, affinity_hook, bs_list=(1, 2), affinity_default=0.0):
    """The existing capture harness, parameterized for the affinity ladder."""
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    ctx = Context(page_size=1)
    ctx.page_table = torch.tensor([[0], [9]], dtype=torch.int32)
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)

    @contextmanager
    def cuda_graph(graph, pool=None, stream=None):
        yield

    def forward():
        return torch.zeros(get_global_ctx().batch.size, 3)

    noop = lambda *args, **kwargs: None  # noqa: E731
    monkeypatch.setattr("torch.cuda.CUDAGraph", lambda: SimpleNamespace(pool=lambda: "pool"))
    monkeypatch.setattr("torch.cuda.graph", cuda_graph)
    for name in ("synchronize", "empty_cache", "reset_peak_memory_stats"):
        monkeypatch.setattr(f"torch.cuda.{name}", noop)
    monkeypatch.setattr("freetoken.engine.graph.get_free_memory", lambda device: 0)
    monkeypatch.setattr("freetoken.engine.graph._upload_graph", noop)

    dummy = Req(input_ids=torch.zeros(1, dtype=torch.int32), table_idx=1, cached_len=0, output_len=1, uid=-1,
                sampling_params=None, cache_handle=None)
    runner = GraphRunner(stream=None, device=torch.device("cpu"), forward=forward,
                         attn_backend=SimpleNamespace(init_capture_graph=noop, prepare_for_capture=noop),
                         cuda_graph_bs=list(bs_list), cuda_graph_max_bs=None, free_memory=0, max_seq_len=1,
                         dummy_req=dummy, moe_offload_cache=SimpleNamespace(reset=noop),
                         affinity_values=affinity_values, affinity_hook=affinity_hook,
                         affinity_default=affinity_default)
    return runner


def test_capture_keys_a_graph_per_ladder_value_and_hooks_each(monkeypatch):
    """The ladder loops inside the bs loop: the hook bakes every value in and the map
    keys on (padded bs, affinity); the hook stays None without a ladder."""
    hooked = []
    runner = _ladder_runner(
        monkeypatch, affinity_values=(0.0, 0.3), affinity_hook=lambda v: hooked.append(v),
    )
    assert hooked == [0.0, 0.3] * 2  # one per (bs, value), bs order 2 then 1
    assert set(runner.graph_map) == {(2, 0.0), (2, 0.3), (1, 0.0), (1, 0.3)}

    batch = Batch(reqs=[], phase="decode", rows_per_req=1)
    batch.route_affinity = 0.3
    assert runner.can_use_cuda_graph(batch)
    batch.route_affinity = 0.05  # uncaptured: the replay must fall to eager
    assert not runner.can_use_cuda_graph(batch)


def test_capture_without_a_ladder_keys_the_server_default(monkeypatch):
    """No hook, one graph per bs keyed on the value replay looks up - today's path."""
    hooked = []
    runner = _ladder_runner(monkeypatch, affinity_values=(0.3,), affinity_hook=None)
    assert hooked == []
    assert set(runner.graph_map) == {(2, 0.3), (1, 0.3)}
    batch = Batch(reqs=[], phase="decode", rows_per_req=1)
    batch.route_affinity = 0.3
    assert runner.can_use_cuda_graph(batch)
    runner.apply_affinity(batch)
    assert hooked == []  # the single-value mode never hooks


def test_eager_decode_then_prefill_restores_the_server_default(monkeypatch):
    """An uncaptured decode value lands on the blocks (apply_affinity bakes it), and the
    next eager prefill must re-bake the server default - otherwise prefill would route
    through the decode's affinity and shift the resident-expert bonus."""
    class _Block:
        route_affinity = 0.3

    class _Model:
        """The Glm5Next shape: blocks read the attribute on each eager forward."""

        def __init__(self):
            self.blocks = [_Block()]

        def set_route_affinity(self, value):
            for block in self.blocks:
                block.route_affinity = value

    model = _Model()
    runner = _ladder_runner(
        monkeypatch, affinity_values=(0.0, 0.3), affinity_hook=model.set_route_affinity,
        affinity_default=0.3,
    )

    decode = Batch(reqs=[], phase="decode", rows_per_req=1)
    decode.route_affinity = 0.05  # uncaptured: falls to the eager path
    runner.apply_affinity(decode)
    assert model.blocks[0].route_affinity == 0.05

    prefill = Batch(reqs=[], phase="prefill")
    runner.apply_affinity(prefill)
    assert model.blocks[0].route_affinity == 0.3  # back to the server default
