import pytest
import torch


def _activation_and_mul(gate_up: torch.Tensor, activation: str) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    if activation == "silu":
        return torch.nn.functional.silu(gate) * up
    if activation == "gelu":
        return torch.nn.functional.gelu(gate) * up
    raise AssertionError(f"unsupported test activation {activation}")


def _reference_fused_experts_decode(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
    output = torch.zeros_like(hidden_states, dtype=torch.float32)
    hidden_fp32 = hidden_states.float()
    w1_fp32 = w1.float()
    w2_fp32 = w2.float()
    for token_idx in range(hidden_states.shape[0]):
        for route_idx in range(topk_ids.shape[1]):
            expert_id = int(topk_ids[token_idx, route_idx])
            route_weight = topk_weights[token_idx, route_idx].float()
            gate_up = torch.matmul(w1_fp32[expert_id], hidden_fp32[token_idx])
            if apply_router_weight_on_input:
                gate_up = gate_up * route_weight
            activated = _activation_and_mul(gate_up, activation)
            contribution = torch.matmul(w2_fp32[expert_id], activated)
            if not apply_router_weight_on_input:
                contribution = contribution * route_weight
            output[token_idx] += contribution
    return output.to(hidden_states.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_topk_accepts_triton_kernel_tuple_output():
    from freetoken.moe.fused import fused_topk

    logits = torch.tensor(
        [[4.0, 1.0, -1.0, 2.0], [0.5, 3.0, 2.0, -2.0]],
        device="cuda",
        dtype=torch.bfloat16,
    )
    hidden_states = torch.zeros((2, 8), device="cuda", dtype=torch.bfloat16)

    weights, ids = fused_topk(hidden_states, logits, topk=2, renormalize=True)

    ref_logits, ref_ids = torch.topk(logits.float(), 2, dim=-1)
    ref_weights = torch.softmax(ref_logits, dim=-1)
    torch.testing.assert_close(ids, ref_ids.to(torch.int32))
    torch.testing.assert_close(weights, ref_weights, rtol=2e-4, atol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 24])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_experts_decode_matches_reference_for_non_contiguous_slots(batch_size, dtype):
    from freetoken.moe.fused import fused_experts_decode_impl

    device = torch.device("cuda")
    cache_size = 37
    hidden_size = 32
    intermediate_size = 24
    top_k = 8
    torch.manual_seed(42 + batch_size)

    hidden_states = 0.5 * torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    gate_up = torch.randn(
        cache_size,
        intermediate_size * 2,
        hidden_size,
        device=device,
        dtype=dtype,
    ) * 0.5
    down = torch.randn(
        cache_size,
        hidden_size,
        intermediate_size,
        device=device,
        dtype=dtype,
    ) * 0.5
    weights = torch.rand(batch_size, top_k, device=device, dtype=torch.float32)
    topk_weights = weights / weights.sum(dim=-1, keepdim=True)
    slot_ids = torch.tensor([31, 4, 18, 0, 29, 7, 35, 12], device=device, dtype=torch.int32)
    topk_ids = slot_ids.repeat(batch_size, 1).contiguous()

    output = fused_experts_decode_impl(hidden_states, gate_up, down, topk_weights, topk_ids)
    expected = _reference_fused_experts_decode(hidden_states, gate_up, down, topk_weights, topk_ids)
    torch.cuda.synchronize()

    torch.testing.assert_close(output, expected, rtol=5e-2, atol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("apply_router_weight_on_input", [False, True])
def test_fused_experts_decode_matches_grouped_impl(dtype, apply_router_weight_on_input):
    from freetoken.moe.fused import fused_experts_decode_impl, fused_experts_impl

    device = torch.device("cuda")
    batch_size = 4
    num_experts = 37
    hidden_size = 32
    intermediate_size = 24
    top_k = 4
    torch.manual_seed(91)

    hidden_states = 0.25 * torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    gate_up = torch.randn(
        num_experts,
        intermediate_size * 2,
        hidden_size,
        device=device,
        dtype=dtype,
    ) * 0.25
    down = torch.randn(
        num_experts,
        hidden_size,
        intermediate_size,
        device=device,
        dtype=dtype,
    ) * 0.25
    weights = torch.rand(batch_size, top_k, device=device, dtype=torch.float32)
    topk_weights = weights / weights.sum(dim=-1, keepdim=True)
    topk_ids = torch.tensor(
        [[31, 4, 18, 0], [29, 7, 35, 12], [6, 21, 1, 33], [16, 3, 28, 9]],
        device=device,
        dtype=torch.int32,
    )

    output = fused_experts_decode_impl(
        hidden_states.clone(),
        gate_up,
        down,
        topk_weights,
        topk_ids.clone(),
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
    expected = fused_experts_impl(
        hidden_states.clone(),
        gate_up,
        down,
        topk_weights,
        topk_ids.clone(),
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_experts_grouped_impl_is_cuda_graph_capturable():
    from freetoken.moe.fused import fused_experts_impl

    device = torch.device("cuda")
    dtype = torch.float16
    batch_size = 1
    num_experts = 8
    hidden_size = 32
    intermediate_size = 24
    torch.manual_seed(17)

    hidden_states = 0.25 * torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    gate_up = torch.randn(
        num_experts,
        intermediate_size * 2,
        hidden_size,
        device=device,
        dtype=dtype,
    ) * 0.25
    down = torch.randn(
        num_experts,
        hidden_size,
        intermediate_size,
        device=device,
        dtype=dtype,
    ) * 0.25
    topk_weights = torch.tensor([[0.4, 0.3, 0.2, 0.1]], device=device, dtype=torch.float32)
    topk_ids = torch.tensor([[6, 1, 4, 7]], device=device, dtype=torch.int32)
    output = torch.empty_like(hidden_states)

    def run():
        output.copy_(
            fused_experts_impl(
                hidden_states,
                gate_up,
                down,
                topk_weights,
                topk_ids,
            )
        )

    for _ in range(3):
        run()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("activation", ["silu", "gelu"])
@pytest.mark.parametrize("apply_router_weight_on_input", [False, True])
def test_fused_experts_decode_activation_and_router_weight_modes(
    activation,
    apply_router_weight_on_input,
):
    from freetoken.moe.fused import fused_experts_decode_impl

    device = torch.device("cuda")
    batch_size = 3
    cache_size = 19
    hidden_size = 32
    intermediate_size = 24
    top_k = 4
    dtype = torch.float16
    torch.manual_seed(123)

    hidden_states = 0.5 * torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    gate_up = torch.randn(
        cache_size,
        intermediate_size * 2,
        hidden_size,
        device=device,
        dtype=dtype,
    ) * 0.5
    down = torch.randn(
        cache_size,
        hidden_size,
        intermediate_size,
        device=device,
        dtype=dtype,
    ) * 0.5
    topk_weights = torch.tensor(
        [[0.4, 0.3, 0.2, 0.1], [0.15, 0.35, 0.25, 0.25], [0.1, 0.2, 0.3, 0.4]],
        device=device,
        dtype=torch.float32,
    )
    topk_ids = torch.tensor(
        [[17, 2, 11, 5], [3, 17, 0, 14], [8, 6, 12, 1]],
        device=device,
        dtype=torch.int32,
    )

    output = fused_experts_decode_impl(
        hidden_states,
        gate_up,
        down,
        topk_weights,
        topk_ids,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
    expected = _reference_fused_experts_decode(
        hidden_states,
        gate_up,
        down,
        topk_weights,
        topk_ids,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(output, expected, rtol=5e-2, atol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_topk_handles_non_power_of_2_k():
    """A top-10 router (qwen4_exp) must route like any other k."""
    from freetoken.moe.fused import _torch_fused_topk, fused_topk

    gating = torch.randn(5, 64, device="cuda")
    hidden = torch.randn(5, 8, device="cuda")
    weights, ids = fused_topk(hidden, gating, 10, renormalize=True)
    ref_w, ref_i = _torch_fused_topk(gating, 10, True, None)
    assert torch.equal(ids, ref_i)
    torch.testing.assert_close(weights, ref_w, rtol=1e-5, atol=1e-6)


# The in-repo triton router behind fused_topk; fp32 logits keep the reference top-k tie-free.
# Ties get their own case below, because torch.topk does not break them by expert id.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize(
    "num_tokens,num_experts,topk",
    [(1, 512, 10), (7, 512, 10), (129, 512, 10), (33, 512, 6), (4, 64, 3)],
)
def test_fused_topk_softmax_matches_torch_reference(num_tokens, num_experts, topk, renormalize):
    from freetoken.kernel.triton.moe_router import fused_topk_softmax
    from freetoken.moe.fused import _torch_fused_topk

    gen = torch.Generator(device="cuda").manual_seed(num_tokens * 31 + topk)
    gating = torch.randn(num_tokens, num_experts, generator=gen, device="cuda")

    weights, ids = fused_topk_softmax(gating, topk, renormalize)
    ref_w, ref_i = _torch_fused_topk(gating, topk, renormalize, None)

    assert weights.dtype == torch.float32 and ids.dtype == torch.int32
    assert torch.equal(ids, ref_i)
    torch.testing.assert_close(weights, ref_w, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_topk_softmax_ties_pick_the_lowest_expert_id():
    from freetoken.kernel.triton.moe_router import fused_topk_softmax

    gating = torch.full((2, 8), -10.0, device="cuda")
    gating[0, [6, 2, 5]] = 1.0  # three-way tie for two slots
    gating[1] = 0.0  # whole row tied

    weights, ids = fused_topk_softmax(gating, 3, renormalize=True)

    assert ids[0].tolist() == [2, 5, 6]
    assert ids[1].tolist() == [0, 1, 2]
    torch.testing.assert_close(weights, torch.full((2, 3), 1.0 / 3.0, device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("limit_dtype", [torch.int32, torch.int64])
def test_fused_topk_softmax_masks_padded_rows(limit_dtype):
    from freetoken.kernel.triton.moe_router import fused_topk_softmax
    from freetoken.moe.fused import _torch_fused_topk

    gen = torch.Generator(device="cuda").manual_seed(5)
    gating = torch.randn(16, 512, generator=gen, device="cuda")
    limit = torch.tensor(5, dtype=limit_dtype, device="cuda")

    weights, ids = fused_topk_softmax(gating, 10, True, limit)
    ref_w, ref_i = _torch_fused_topk(gating, 10, True, limit)

    assert (ids[5:] == -1).all()
    assert torch.equal(ids, ref_i)
    torch.testing.assert_close(weights, ref_w, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_topk_softmax_is_cuda_graph_capturable():
    """The padded-row limit must come off the device tensor, not a host read baked into the graph."""
    from freetoken.kernel.triton.moe_router import fused_topk_softmax

    gen = torch.Generator(device="cuda").manual_seed(11)
    gating = torch.randn(8, 512, generator=gen, device="cuda")
    limit = torch.tensor(8, dtype=torch.int32, device="cuda")

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fused_topk_softmax(gating, 10, True, limit)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _, ids = fused_topk_softmax(gating, 10, True, limit)

    limit.fill_(3)
    graph.replay()
    torch.cuda.synchronize()
    assert (ids[:3] != -1).all()
    assert (ids[3:] == -1).all()


# moved from tests/kernels/test_gguf_quant.py + tests/models/test_gguf_expert_banks.py
# (both retired by the round-5 test-hygiene wave): the producers below are live on
# the vendored-ggml stack, only the old branch kernels they used to feed are gone.


def test_moe_vec_chunk_cap():
    from freetoken.layers.moe import _MOE_VEC_MAX_GRID, _assert_moe_vec_chunk

    _assert_moe_vec_chunk(4096, 8)  # default max-prefill-length is unaffected
    _assert_moe_vec_chunk(8191, 8)  # the cap itself
    _assert_moe_vec_chunk(65535, 1)  # the exact grid boundary passes
    with pytest.raises(ValueError, match="max-prefill-length"):
        _assert_moe_vec_chunk(8192, 8)
    # boundary-exact: 65536 = the first grid value the moe_vec kernel cannot index
    with pytest.raises(ValueError, match="65536 exceeds the ggml moe_vec grid limit 65535"):
        _assert_moe_vec_chunk(65536, 1)
    assert _MOE_VEC_MAX_GRID == 65535


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("bs", (4, 8, 16, 32))
def test_moe_align_trio_contract(bs):
    # the trio every grouped MoE consumer gets from moe_align_block_size: int32
    # buffers, sentinel fill = numel (NOT -1), numel = pairs + (E+1)*(bs-1) whenever
    # pairs >= E+1, and expert_ids inside the padded region are real expert ids (the
    # pad bin E is what kernel-side expert bounds must keep dropping). Swept over
    # the block sizes the grouped paths run at.
    from freetoken.moe.fused import moe_align_block_size

    tokens, top_k, experts = 40, 8, 288
    topk_ids = torch.randint(0, experts, (tokens, top_k), dtype=torch.int32, device="cuda")
    sorted_ids, expert_ids, npp = moe_align_block_size(topk_ids, bs, experts)

    pairs = topk_ids.numel()
    assert sorted_ids.dtype == expert_ids.dtype == npp.dtype == torch.int32
    assert sorted_ids.numel() == pairs + (experts + 1) * (bs - 1)
    assert expert_ids.numel() == (sorted_ids.numel() + bs - 1) // bs
    assert npp.shape == (1,)

    sentinel = sorted_ids == pairs  # the producer's sentinel fill is topk_ids.numel()
    assert sentinel.any(), "pad slots must carry the numel sentinel"
    assert not (sorted_ids[~sentinel] >= pairs).any(), "real entries are pair indices"
    total_padded = int(npp.item())
    assert total_padded % bs == 0 and total_padded <= pairs + experts * bs
    live = expert_ids[: total_padded // bs]
    assert (live >= 0).all() and (live < experts).all(), "pad bin ids must not reach the kernel"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("bs", (4, 8, 16, 32))
@pytest.mark.parametrize("tokens", (40, 137))  # 320 and 1096 pairs: small and large producer paths
def test_moe_align_tail_contract(tokens, bs):
    # TP2 tail pin, driven through the TRITON producer directly (the fused wrapper
    # would pick the sgl_kernel op when it is installed): grouped kernels launch
    # bins past num_tokens_post_pad (their grid covers the padded buffer), so the
    # producer must leave NO uninitialized slots. sorted_token_ids[npp:] carries
    # the numel sentinel and expert_ids[npp/bs:] carries the sentinel expert id
    # (num_experts), which a consumer's expert bound drops even before the >=
    # boundary check rejects the bin. Swept over the block-size set.
    from freetoken.kernel.triton.moe_align import moe_align_block_size as triton_align

    top_k, experts = 8, 288
    topk_ids = torch.randint(0, experts, (tokens, top_k), dtype=torch.int32, device="cuda")
    sorted_ids, expert_ids, npp = triton_align(topk_ids, bs, experts)

    pairs = topk_ids.numel()
    total = int(npp.item())
    assert total % bs == 0
    tail = sorted_ids[total:]
    assert tail.numel() > 0
    assert (tail == pairs).all(), "sorted tail must carry the numel sentinel"
    tb = total // bs
    live = expert_ids[:tb]
    assert (live >= 0).all() and (live < experts).all(), "live blocks carry real expert ids"
    pad = expert_ids[tb:]
    assert pad.numel() > 0
    assert (pad == experts).all(), "expert_ids tail must carry the sentinel expert id"
    # belt (boundary check): bins starting at/after total are rejected; the first
    # such bin is tb and the producer's sentinel values make it a safe read
    assert expert_ids.numel() > tb and tb * bs >= total
def _reference_sigmoid_topk(logits, bias, topk, renormalize, scale, n_group, topk_group, num_token_non_padded):
    """The torch routing of the GLM / MiniMax MoE blocks: sigmoid, bias for selection only, optional group-limited top-k."""
    scores = logits.float().sigmoid()
    choice = scores + bias.float()
    if n_group > 1:
        m, e = choice.shape
        group_scores = choice.view(m, n_group, e // n_group).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1.0).bool()
        choice = choice.masked_fill(~group_mask.repeat_interleave(e // n_group, dim=1), float("-inf"))
    _, ids = torch.topk(choice, topk, dim=-1)
    weights = scores.gather(-1, ids)
    if renormalize:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    ids = ids.to(torch.int32)
    ids[torch.arange(ids.shape[0], device=ids.device) >= num_token_non_padded] = -1
    return weights * scale, ids


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("num_tokens", [1, 7, 2048])
@pytest.mark.parametrize(
    "num_experts,topk,n_group,topk_group,renormalize,scale",
    [
        (128, 8, 1, 1, True, 1.0),  # GLM-4.5-Air
        (160, 8, 1, 1, True, 2.5),  # GLM-4.7
        (288, 8, 1, 1, True, 2.5),  # GLM-5.3-Flash
        (128, 4, 1, 1, True, 2.0),  # MiniMax-M3
        (256, 8, 8, 4, True, 2.5),  # DeepSeek-V3 group-limited top-k
        (96, 6, 3, 2, False, 1.0),  # odd group count, no renormalization
    ],
)
def test_fused_topk_sigmoid_matches_torch_reference(num_tokens, num_experts, topk, n_group, topk_group, renormalize, scale):
    from freetoken.kernel.triton.moe_router import fused_topk_sigmoid

    gen = torch.Generator(device="cuda").manual_seed(num_tokens * 31 + num_experts * 7 + topk)
    logits = torch.randn(num_tokens, num_experts, generator=gen, device="cuda")
    bias = torch.randn(num_experts, generator=gen, device="cuda") * 0.1
    limit = torch.tensor(max(1, num_tokens - 3), dtype=torch.int32, device="cuda")

    weights, ids = fused_topk_sigmoid(logits, bias, topk, renormalize, scale, n_group, topk_group, limit)
    ref_w, ref_i = _reference_sigmoid_topk(logits, bias, topk, renormalize, scale, n_group, topk_group, limit)

    assert weights.dtype == torch.float32 and ids.dtype == torch.int32
    assert torch.equal(ids, ref_i)
    torch.testing.assert_close(weights, ref_w, rtol=1e-5, atol=1e-6)
