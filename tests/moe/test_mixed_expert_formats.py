"""Experts of several formats in one model: bf16 layers next to NVFP4 layers.

A checkpoint may leave some MoE layers unquantized (Muse Spark keeps its first and last layer's
experts bf16). Each format fills its own layers' banks, the banks merge over the union of roles with
zero-byte placeholders, the slot cache keeps one pool per format, the FTW stores each layer's own
banks, and the CPU executor computes each layer in its format.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.distributed import set_tp_info, try_get_tp_info

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

E, H, I, TOPK = 8, 256, 128, 2
FORMATS = ["bf16", "nvfp4", "nvfp4", "nvfp4", "bf16"]
L = len(FORMATS)

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def _init_tp():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _layer(layer_id: int, fmt: str):
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.layers.quantization import NoQuantConfig, QuantBackend, QuantConfig, set_quant_backend

    _init_tp()
    if fmt == "bf16":
        quant = NoQuantConfig()
    else:
        set_quant_backend(QuantBackend.parse("moe.nvfp4=triton"))
        quant = QuantConfig.from_hf({"quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4", "ignore": ["lm_head"]}})
    return OffloadMoELayer(layer_id, E, TOPK, H, I, quant_config=quant, prefix=f"model.layers.{layer_id}.mlp.experts")


def _dequant(packed, scale, row_global):
    n, k2 = packed.shape
    codes = torch.stack([packed & 0xF, packed >> 4], dim=-1).view(n, 2 * k2).long()
    return _E2M1.to(packed.device)[codes] * scale.float().repeat_interleave(16, dim=1) * row_global.float().unsqueeze(1)


def _pieces(seed: int = 0):
    """The checkpoint side: per layer, bf16 or ModelOpt NVFP4 pieces in the layer's format."""
    g = torch.Generator().manual_seed(seed)
    pieces = {}
    for layer, fmt in enumerate(FORMATS):
        if fmt == "bf16":
            pieces[layer] = {
                "gate_up": (torch.randn(E, 2 * I, H, generator=g) / 16).to(torch.bfloat16),
                "down": (torch.randn(E, H, I, generator=g) / 16).to(torch.bfloat16),
            }
        else:
            pieces[layer] = {
                "gate_up": torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, generator=g),
                "gate_up_scale": (torch.rand(E, 2 * I, H // 16, generator=g) + 0.25).to(torch.float8_e4m3fn),
                "gate_up_global": torch.full((E, 2 * I), 0.02, dtype=torch.float16),
                "down": torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, generator=g),
                "down_scale": (torch.rand(E, H, I // 16, generator=g) + 0.25).to(torch.float8_e4m3fn),
                "down_global": torch.full((E, H), 0.02, dtype=torch.float16),
            }
    return pieces


def _reader(pieces, kind):
    """A family reader for one storage form: it yields only the layers stored that way."""
    from freetoken.layers.quantization import QuantKind

    want = "bf16" if kind is QuantKind.NONE else "nvfp4"
    for layer, fmt in enumerate(FORMATS):
        if fmt == want:
            yield layer, 0, E, pieces[layer]


def _weights(pieces, layer):
    """Dense fp32 (gate_up [E, 2I, H], down [E, H, I]) of one layer, the reference."""
    p = pieces[layer]
    if FORMATS[layer] == "bf16":
        return p["gate_up"].float(), p["down"].float()
    gu = torch.stack([_dequant(p["gate_up"][e], p["gate_up_scale"][e], p["gate_up_global"][e]) for e in range(E)])
    dn = torch.stack([_dequant(p["down"][e], p["down_scale"][e], p["down_global"][e]) for e in range(E)])
    return gu, dn


def _reference(pieces, layer, hidden, weights, ids):
    gu, dn = _weights(pieces, layer)
    out = torch.zeros(hidden.shape, dtype=torch.float32)
    x = hidden.float().cpu()
    for t in range(hidden.shape[0]):
        for k in range(ids.shape[1]):
            e = int(ids[t, k])
            h = gu[e] @ x[t]
            out[t] += float(weights[t, k]) * (dn[e] @ (torch.nn.functional.silu(h[:I]) * h[I:]))
    return out


def _assert_close(out, ref):
    torch.testing.assert_close(out.float().cpu(), ref, rtol=3e-2, atol=0.03 * float(ref.abs().max()))


def _mixed_banks(pieces, device, layer_sink=None):
    from freetoken.moe.expert_banks import build_expert_banks, merge_expert_banks, method_groups

    methods = [_layer(layer, fmt).quant_method for layer, fmt in enumerate(FORMATS)]
    parts = [
        build_expert_banks(method, L, _reader(pieces, method.kind), device=device, layer_sink=layer_sink, layers=layers)
        for method, layers in method_groups(methods)
    ]
    return merge_expert_banks(parts, methods, E), methods


def _mixed_cache(device, pieces, *, decode_target="gpu", prefill_overlap=False):
    from freetoken.moe.offload_cache import OffloadMoeCache, bank_layouts
    from freetoken.moe.pool_policy import group_layers, make_pools, pool_of_layer, split_slots
    from freetoken.engine.cache_budget import layer_slot_bytes

    banks, methods = _mixed_banks(pieces, device)
    slot_bytes = layer_slot_bytes(banks.sources)
    groups = group_layers("unified", bank_layouts(banks.sources, banks.layer_tags), slot_bytes)
    pools = make_pools(groups, E, TOPK, None)
    sizes = split_slots(4 * E, pools)
    cache = OffloadMoeCache(
        num_layers=L, num_experts=E, cache_size=sum(sizes), device=device, quant_format=banks.quant_format,
        decode_target=decode_target, hybrid_max_fetch=1, prefill_overlap=prefill_overlap,
        layer_layouts=[m.layout() for m in methods], pool_of_layer=pool_of_layer(pools, L), pool_sizes=sizes,
    )
    cache.set_bank_sources(banks.sources, layer_tags=banks.layer_tags)
    cache.reset()
    layers = [_layer(layer, fmt) for layer, fmt in enumerate(FORMATS)]
    for layer in layers:
        layer.offload_cache = cache
    return cache, layers, banks


def test_engine_binds_the_most_common_format_and_checks_every_layer():
    from freetoken.engine.engine import _check_layer_formats, offload_layer_methods, shared_offload_method
    from freetoken.layers.base import BaseOP
    from freetoken.layers.quantization import QuantKind
    from freetoken.moe.expert_banks import ExpertBanks

    class Model(BaseOP):
        def __init__(self):
            self.layers = [_layer(layer, fmt) for layer, fmt in reversed(list(enumerate(FORMATS)))]

    model = Model()
    methods = offload_layer_methods(model)
    assert [m.kind for m in methods] == [QuantKind.NONE if f == "bf16" else QuantKind.NVFP4 for f in FORMATS]
    assert shared_offload_method(model).kind is QuantKind.NVFP4
    _check_layer_formats(ExpertBanks("nvfp4", {}, layer_formats=list(FORMATS)), methods)
    with pytest.raises(ValueError, match=r"MoE layers \[0, 4\]"):
        _check_layer_formats(ExpertBanks("nvfp4", {}), methods)


def test_mixed_dummy_banks_merge_the_roles_with_placeholders():
    from freetoken.moe.expert_banks import load_expert_banks

    methods = [_layer(layer, fmt).quant_method for layer, fmt in enumerate(FORMATS)]
    config = SimpleNamespace(num_moe_layers=L, num_experts=E)
    banks = load_expert_banks("", config, method=methods[1], device=torch.device("cpu"), dummy=True, layer_methods=methods)
    assert banks.layer_formats == FORMATS and banks.quant_format == "nvfp4"
    assert set(banks.sources) == {"gate_up", "gate_up_scale", "gate_up_global", "down", "down_scale", "down_global"}
    for layer, fmt in enumerate(FORMATS):
        if fmt == "bf16":
            assert banks.sources["gate_up"][layer].shape == (E, 2 * I, H) and banks.sources["gate_up"][layer].dtype == torch.bfloat16
            assert banks.sources["down_scale"][layer].shape == (E, 0)
        else:
            assert banks.sources["gate_up"][layer].shape == (E, 2 * I, H // 2) and banks.sources["gate_up"][layer].dtype == torch.uint8
    with pytest.raises(ValueError, match="several formats"):
        load_expert_banks("", config, method=methods[1], device=torch.device("cpu"), dummy=True,
                          layer_methods=methods, layer_residency=["pinned"] * L)


@cuda
def test_mixed_banks_give_each_reader_its_layers_and_resolve_every_reader_first(monkeypatch):
    from freetoken.moe import expert_pieces
    from freetoken.moe.expert_banks import _mixed_expert_banks

    pieces = _pieces(seed=2)
    methods = [_layer(layer, fmt).quant_method for layer, fmt in enumerate(FORMATS)]
    events = []

    def fake_reader(model_path, config, kind, *, layers=None, parallel=False, workers=8, chunk=0):
        events.append(("resolve", sorted(layers)))

        def read():
            events.append(("read", sorted(layers)))
            for layer in layers:
                yield layer, 0, E, pieces[layer]

        return read()

    monkeypatch.setattr(expert_pieces, "iter_expert_pieces", fake_reader)
    config = SimpleNamespace(num_moe_layers=L, num_experts=E)
    banks = _mixed_expert_banks("", config, methods, torch.device("cpu"), False, False, 8, 0)
    assert sorted(layers for step, layers in events if step == "resolve") == [[0, 4], [1, 2, 3]]
    assert [step for step, _ in events[:2]] == ["resolve", "resolve"]
    assert banks.layer_formats == FORMATS


@cuda
def test_mixed_layers_decode_and_prefill_through_their_pools():
    torch.manual_seed(0)
    device = torch.device("cuda")
    pieces = _pieces()
    cache, layers, banks = _mixed_cache(device, pieces, prefill_overlap=True)
    assert len(cache.pool_sizes) == 2 and cache._raw_banks
    # a staging buffer holds one bf16 layer (the widest rows); two of them still fit the 13 + 19 slots
    assert cache.prefill_overlap
    hidden = torch.randn(1, H, dtype=torch.bfloat16, device=device) / 4
    weights = torch.rand(1, TOPK, device=device)
    for step in range(6):
        for layer in layers:
            ids = torch.randperm(E)[:TOPK].to(torch.int32).view(1, TOPK)
            out = layer._decode_routed(hidden, weights, ids.clone().to(device))
            _assert_close(out, _reference(pieces, layer.layer_id, hidden, weights, ids))

    M = 16
    hidden = torch.randn(M, H, dtype=torch.bfloat16, device=device) / 4
    ids = torch.stack([torch.randperm(E)[:TOPK] for _ in range(M)]).to(torch.int32)
    weights = torch.rand(M, TOPK, device=device)
    for layer in layers:
        # the bf16 prefill kernel overwrites its input
        out = layer._prefill_routed(hidden.clone(), weights, ids.to(device))
        _assert_close(out, _reference(pieces, layer.layer_id, hidden, weights, ids))
    # decode after the prefill stomped the staging slots must reload, not read another format's bytes
    for layer in layers:
        ids = torch.randperm(E)[:TOPK].to(torch.int32).view(1, TOPK)
        out = layer._decode_routed(hidden[:1], weights[:1], ids.clone().to(device))
        _assert_close(out, _reference(pieces, layer.layer_id, hidden[:1], weights[:1], ids))


@cuda
def test_cpu_executor_computes_each_layer_in_its_format():
    pytest.importorskip("freetoken.kernel._cpu_moe")
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    torch.manual_seed(1)
    device = torch.device("cuda")
    pieces = _pieces(seed=1)
    cache, layers, _ = _mixed_cache(device, pieces, decode_target="hybrid")
    ex = CpuMoeExecutor(
        cache, top_k=TOPK, activation="silu", apply_router_weight_on_input=False, num_threads=2,
        max_tokens=4, device=device, layer_formats=[(f, {}) for f in FORMATS],
    )
    cache.set_cpu_executor(ex)
    bs = 3
    hidden = torch.randn(bs, H, dtype=torch.bfloat16, device=device) / 4
    weights = torch.rand(bs, TOPK, device=device)
    for layer in range(L):
        ids = torch.stack([torch.randperm(E)[:TOPK] for _ in range(bs)]).to(torch.int32)
        _assert_close(ex.decode(layer, hidden, weights, ids.to(device)), _reference(pieces, layer, hidden, weights, ids))
    # hybrid: capped PCIe fetch on the GPU, the overflow misses on the CPU, per layer format
    for _ in range(3):
        for layer in layers:
            ids = torch.stack([torch.randperm(E)[:TOPK] for _ in range(bs)]).to(torch.int32)
            out = layer._decode_routed(hidden, weights, ids.clone().to(device))
            _assert_close(out, _reference(pieces, layer.layer_id, hidden, weights, ids))


@cuda
def test_ftw_stores_each_layers_own_banks(tmp_path):
    from freetoken.checkpoint.convert import _ConvertSink
    from freetoken.checkpoint.ftw import FTWWriter, load_ftw_banks

    pieces = _pieces(seed=2)
    reference, _ = _mixed_banks(pieces, torch.device("cuda"))
    writer = FTWWriter(str(tmp_path))
    sink = _ConvertSink(writer, names={"gate_up": "gate_up_packed", "down": "down_packed"})
    banks, _ = _mixed_banks(pieces, torch.device("cuda"), layer_sink=sink)
    assert banks.streamed and sink.num_layers == L
    writer.finalize({"quant_format": banks.quant_format, "layer_quant_formats": banks.layer_formats, "expert_bank_num_layers": L})

    loaded = load_ftw_banks(str(tmp_path), num_layers=L)
    assert loaded.layer_formats == FORMATS and loaded.layer_tags == FORMATS and loaded.quant_format == "nvfp4"
    assert set(loaded.sources) == set(reference.sources)
    for role, per_layer in reference.sources.items():
        for layer, tensor in enumerate(per_layer):
            got = loaded.sources[role][layer]
            assert got.shape == tensor.shape and got.dtype == tensor.dtype, (role, layer)
            assert torch.equal(got.view(torch.uint8), tensor.view(torch.uint8)), (role, layer)


@cuda
def test_two_gguf_signatures_hybrid_decode_keeps_pool_locality():
    """D3 g.3: one gguf cache carrying two expert-type signatures (the glm5next
    width mix) resolves to two pools, and the real hybrid ensure kernel keeps the
    fetch contract per pool: the cap counts the step's UNIQUE miss pairs, -1
    routes are only the overflow beyond the cap, and one pool's slot/LRU writes
    never leak into the other pool's segment."""
    from freetoken.moe.offload_cache import OffloadMoeCache

    _init_tp()
    torch.manual_seed(0)
    L, E, TOPK = 4, 8, 2
    tags = ["A", "A", "B", "B"]
    widths = {"A": (64, 80), "B": (96, 112)}  # byte-row widths of the two signatures
    sources = {"gate_up": [], "down": []}
    for layer in range(L):
        for role, w in zip(("gate_up", "down"), widths[tags[layer]]):
            sentinel = torch.arange(E, dtype=torch.uint8).view(E, 1) + layer * E + 1
            sources[role].append(sentinel.repeat(1, w).pin_memory())

    def make(fetch):
        cache = OffloadMoeCache(
            num_layers=L, num_experts=E, cache_size=24, device=torch.device("cuda"),
            quant_format="gguf", decode_target="hybrid", hybrid_max_fetch=fetch,
            pool_of_layer=[0, 0, 1, 1], pool_sizes=[10, 14],
        )
        cache.set_bank_sources(sources, layer_tags=tags)
        assert len(cache.pool_sizes) == 2 and cache._raw_banks
        return cache

    def step(cache, layer, rows):
        ids = torch.tensor(rows, dtype=torch.int32, device="cuda")
        routes = cache.ensure_experts_hybrid(layer, ids, torch.ones(len(rows), TOPK, device="cuda"))
        cache.copy_missing()
        return ids.cpu(), routes

    # uncapped: every unique miss pair of the step is fetched, no -1 lanes
    cache = make(E)
    pool_b = slice(10, 24)
    b_ids, b_usage = cache.id_of_slot[pool_b].clone(), cache.usage[pool_b].clone()
    ids, routes = step(cache, 0, [[0, 1], [0, 2], [1, 3], [2, 3]])  # 4 unique pairs on 8 lanes
    assert int(cache.num_missing_full.item()) == 4
    assert int(cache.num_indices.item()) == 4
    assert (ids >= 0).all() and (ids < 10).all()  # pool-local slots, no overflow
    assert (routes.cpu_ids == -1).all() and (routes.gpu_w == 1).all()
    views = cache.pool_bank_views(0)
    for e in range(4):
        s = int(cache.slot_for_id[0, e])
        assert 0 <= s < 10
        for view, src in zip(views, (sources["gate_up"][0], sources["down"][0])):
            assert torch.equal(view[s].cpu(), src[e])  # fetched bytes are this layer's rows
    assert torch.equal(cache.id_of_slot[pool_b], b_ids)  # pool A's writes never leak into pool B
    assert torch.equal(cache.usage[pool_b], b_usage)

    # cap 0: nothing is fetched, every lane is overflow routed to the CPU leg
    cold = make(0)
    ids, routes = step(cold, 2, [[0, 1], [0, 2], [1, 3], [2, 3]])
    assert int(cold.num_missing_full.item()) == 4 and int(cold.num_indices.item()) == 0
    assert (ids == -1).all() and (routes.gpu_w == 0).all()
    assert (routes.cpu_ids >= 0).all()  # the CPU leg still sees the expert ids
    assert (cold.id_of_slot == -1).all()  # the GPU cache stays cold

    # cap 1 with a warm pool: only the overflow pair's lane is -1, the hit pair is not
    warm = make(E)
    step(warm, 0, [[0, 1], [0, 1]])  # pairs 0 and 1 of pool A become resident
    warm.hybrid_max_fetch = 1  # the cap is read per call: warm up uncapped, then clamp
    ids, routes = step(warm, 0, [[2, 0], [3, 0]])  # pair 0 hits; unique misses {2, 3}
    assert int(warm.num_missing_full.item()) == 2 and int(warm.num_indices.item()) == 1
    assert ids.flatten().tolist().count(-1) == 1  # -1 only on the overflow pair's lane
    fetched = {e for e in (2, 3) if int(warm.slot_for_id[0, e]) >= 0}
    assert len(fetched) == 1
    cpu_expert = ({2, 3} - fetched).pop()
    lanes = [(t, k) for t in range(2) for k in range(2) if int(ids[t, k]) == -1]
    assert len(lanes) == 1 and int(routes.cpu_ids[lanes[0][0], lanes[0][1]]) == cpu_expert

    # pool B's own decode steps never touch pool A's segment either; slots are pool-local
    warm.hybrid_max_fetch = E
    a_ids, a_usage = warm.id_of_slot[:10].clone(), warm.usage[:10].clone()
    ids_b, routes_b = step(warm, 3, [[4, 5], [6, 7]])
    assert int(warm.num_missing_full.item()) == 4 and int(warm.num_indices.item()) == 4
    assert (ids_b >= 0).all() and (ids_b < 14).all()  # pool B's local slot space
    assert (routes_b.cpu_ids == -1).all()
    for e in range(4, 8):
        s = int(warm.slot_for_id[3, e])
        assert 0 <= s < 14
        for view, src in zip(warm.pool_bank_views(3), (sources["gate_up"][3], sources["down"][3])):
            assert torch.equal(view[s].cpu(), src[e])
    assert torch.equal(warm.id_of_slot[:10], a_ids) and torch.equal(warm.usage[:10], a_usage)
