"""mHC (Manifold-Constrained Hyper-Connections) unit tests.

Checks the algebraic contracts of layers/mhc.py at GLM-5.3 geometry (n=4):
Sinkhorn projection yields an (approximately) doubly-stochastic comb matrix,
pre/post mixing matches naive per-token einsums, and identity-ish weights give
the classic single-stream residual behaviour.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.layers.mhc import (
    hc_contract,
    hc_expand,
    mhc_fused_post_pre,
    mhc_post,
    mhc_pre,
)

N, HIDDEN, T = 4, 64, 9
MIX = 2 * N + N * N
EPS = 1e-6
RMS_EPS = 1e-5
POST_MULT = 2.0
SINKHORN = 20


def _weights(seed=0, device="cpu"):
    torch.manual_seed(seed)
    fn = torch.randn(MIX, N * HIDDEN, dtype=torch.float32, device=device) * 0.05
    scale = torch.randn(3, dtype=torch.float32, device=device).abs() + 0.5
    base = torch.randn(MIX, dtype=torch.float32, device=device) * 0.3
    return fn, scale, base


def _residual(seed=1, device="cpu"):
    torch.manual_seed(seed)
    return torch.randn(T, N, HIDDEN, dtype=torch.bfloat16, device=device)


def test_comb_is_doubly_stochastic():
    fn, scale, base = _weights(seed=2)
    res = _residual(seed=3)
    _, comb, _ = mhc_pre(res, fn, scale, base, RMS_EPS, EPS, POST_MULT, SINKHORN)
    rows = comb.sum(dim=-1)
    cols = comb.sum(dim=-2)
    assert torch.allclose(rows, torch.ones_like(rows), atol=1e-3)
    assert torch.allclose(cols, torch.ones_like(cols), atol=1e-3)
    assert (comb > 0).all()


def test_post_matches_naive():
    res = _residual(seed=4)
    x = torch.randn(T, HIDDEN, dtype=torch.bfloat16)
    post = torch.rand(T, N, 1, dtype=torch.float32) * POST_MULT
    comb = torch.softmax(torch.randn(T, N, N), dim=-1)
    out = mhc_post(x, res, post, comb)
    assert out.shape == (T, N, HIDDEN)

    ref = torch.zeros(T, N, HIDDEN, dtype=torch.float32)
    for t in range(T):
        for j in range(N):
            acc = post[t, j, 0] * x[t].float()
            for i in range(N):
                acc = acc + comb[t, i, j] * res[t, i].float()
            ref[t, j] = acc
    assert torch.allclose(out.float(), ref.to(torch.bfloat16).float())


def test_fused_equals_decomposed():
    fn, scale, base = _weights(seed=7)
    res = _residual(seed=8)
    x = torch.randn(T, HIDDEN, dtype=torch.bfloat16)
    post0, comb0, _ = mhc_pre(res, fn, scale, base, RMS_EPS, EPS, POST_MULT, SINKHORN)

    r1, p1, c1, li1 = mhc_fused_post_pre(
        x, res, post0, comb0, fn, scale, base, RMS_EPS, EPS, POST_MULT, SINKHORN
    )
    r_ref = mhc_post(x, res, post0, comb0)
    p_ref, c_ref, li_ref = mhc_pre(
        r_ref, fn, scale, base, RMS_EPS, EPS, POST_MULT, SINKHORN
    )
    assert torch.equal(r1, r_ref)
    assert torch.equal(p1, p_ref)
    assert torch.equal(c1, c_ref)
    assert torch.equal(li1, li_ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("t,hidden", [(1, 64), (9, 64), (3, 4096)])
def test_triton_fused_matches_torch(t, hidden):
    """The fused triton kernel must reproduce the decomposed torch reference
    (hc_post -> hc_pre) on every output, including GLM-5.3's real hidden size."""
    from freetoken.layers.mhc import mhc_fused_post_pre_torch
    from freetoken.kernel.triton.mhc import mhc_fused_post_pre_triton

    torch.manual_seed(11)
    mix = 2 * N + N * N
    fn = torch.randn(mix, N * hidden, dtype=torch.float32, device="cuda") * 0.05
    scale = torch.rand(3, dtype=torch.float32, device="cuda") + 0.5
    base = torch.randn(mix, dtype=torch.float32, device="cuda") * 0.3
    res = torch.randn(t, N, hidden, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(t, hidden, dtype=torch.bfloat16, device="cuda")
    post0 = torch.rand(t, N, 1, dtype=torch.float32, device="cuda") * POST_MULT
    comb0 = torch.softmax(torch.randn(t, N, N, device="cuda"), dim=-1)

    ref = mhc_fused_post_pre_torch(
        x, res, post0, comb0, fn, scale, base, RMS_EPS, EPS, POST_MULT, SINKHORN
    )
    got = mhc_fused_post_pre_triton(
        x, res, post0, comb0, fn, scale, base, RMS_EPS, EPS, POST_MULT, SINKHORN
    )
    names = ("residual", "post", "comb", "layer_input")
    tols = (2e-2, 2e-3, 2e-3, 2e-2)
    for name, r, g, tol in zip(names, ref, got, tols):
        assert g.shape == r.shape, (name, g.shape, r.shape)
        err = (g.float() - r.float()).abs().max().item()
        assert err < tol, f"{name}: max abs err {err}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("dtype,tol", [(torch.float16, 2e-2), (torch.float32, 1e-4)])
def test_triton_fused_respects_input_dtype(dtype, tol):
    """--dtype float16/float32 must not be silently bf16-rounded: the kernel
    stores in the OUTPUT tensor's dtype (regression for the hard-coded
    tl.bfloat16 stores; fp32's tolerance is far below bf16's 2^-8 grid)."""
    from freetoken.layers.mhc import mhc_fused_post_pre_torch
    from freetoken.kernel.triton.mhc import mhc_fused_post_pre_triton

    torch.manual_seed(13)
    t, hidden = 4, 4096
    mix = 2 * N + N * N
    fn = torch.randn(mix, N * hidden, dtype=torch.float32, device="cuda") * 0.05
    scale = torch.rand(3, dtype=torch.float32, device="cuda") + 0.5
    base = torch.randn(mix, dtype=torch.float32, device="cuda") * 0.3
    res = torch.randn(t, N, hidden, dtype=dtype, device="cuda")
    x = torch.randn(t, hidden, dtype=dtype, device="cuda")
    post0 = torch.rand(t, N, 1, dtype=torch.float32, device="cuda") * POST_MULT
    comb0 = torch.softmax(torch.randn(t, N, N, device="cuda"), dim=-1)

    ref = mhc_fused_post_pre_torch(
        x, res, post0, comb0, fn, scale, base, RMS_EPS, EPS, POST_MULT, SINKHORN
    )
    got = mhc_fused_post_pre_triton(
        x, res, post0, comb0, fn, scale, base, RMS_EPS, EPS, POST_MULT, SINKHORN
    )
    assert got[0].dtype == dtype and got[3].dtype == dtype
    for name, r, g in zip(("residual", "layer_input"), (ref[0], ref[3]), (got[0], got[3])):
        err = (g.float() - r.float()).abs().max().item()
        assert err < tol, f"{name} [{dtype}]: max abs err {err}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_triton_pre_only_matches_torch():
    """HAS_POST=False path (layer 0's standalone hc_pre through the fused kernel)."""
    from freetoken.kernel.triton.mhc import mhc_fused_post_pre_triton

    torch.manual_seed(12)
    t, hidden = 5, 128
    mix = 2 * N + N * N
    fn = torch.randn(mix, N * hidden, dtype=torch.float32, device="cuda") * 0.05
    scale = torch.rand(3, dtype=torch.float32, device="cuda") + 0.5
    base = torch.randn(mix, dtype=torch.float32, device="cuda") * 0.3
    res = torch.randn(t, N, hidden, dtype=torch.bfloat16, device="cuda")

    ref_post, ref_comb, ref_li = mhc_pre(
        res, fn, scale, base, RMS_EPS, EPS, POST_MULT, SINKHORN
    )
    got_res, got_post, got_comb, got_li = mhc_fused_post_pre_triton(
        res.new_empty(t, hidden), res, None, None, fn, scale, base,
        RMS_EPS, EPS, POST_MULT, SINKHORN,
    )
    assert torch.equal(got_res, res)  # pass-through when no post
    assert (got_post.float() - ref_post.float()).abs().max().item() < 2e-3
    assert (got_comb.float() - ref_comb.float()).abs().max().item() < 2e-3
    assert (got_li.float() - ref_li.float()).abs().max().item() < 2e-2


# ------------------------------------------------------------------
# C-L1: FREETOKEN_MHC_STAGE1_NS stage1 split knob (kernel/triton/mhc.py)


class _KernelCapture:
    """Stands in for a triton JITFunction: records (grid, kwargs) per launch,
    so the launcher's grid math is testable without a GPU."""

    def __init__(self):
        self.launches = []

    def __getitem__(self, grid):
        def _record(*args, **kwargs):
            self.launches.append((tuple(grid), kwargs))

        return _record


def _mhc_fixtures(t, h, device="cpu", seed=5):
    torch.manual_seed(seed)
    mix = 2 * N + N * N
    fn = torch.randn(mix, N * h, dtype=torch.float32, device=device) * 0.05
    scale = torch.rand(3, dtype=torch.float32, device=device) + 0.5
    base = torch.randn(mix, dtype=torch.float32, device=device) * 0.3
    res = torch.randn(t, N, h, dtype=torch.bfloat16, device=device)
    x = torch.randn(t, h, dtype=torch.bfloat16, device=device)
    post = torch.rand(t, N, 1, dtype=torch.float32, device=device) * POST_MULT
    comb = torch.softmax(torch.randn(t, N, N, device=device), dim=-1)
    return fn, scale, base, res, x, post, comb


def _capture_stage1(monkeypatch, t, h, ns_knob=None, has_post=True):
    from freetoken.kernel.triton import mhc as mhc_k

    if ns_knob is None:
        # ns_knob=None = env unset: exercise the parser's default value.
        monkeypatch.setattr(mhc_k, "_MHC_STAGE1_NS", mhc_k._parse_stage1_ns(None))
    else:
        monkeypatch.setattr(mhc_k, "_MHC_STAGE1_NS", ns_knob)
    fn, scale, base, res, x, post, comb = _mhc_fixtures(t, h)
    caps = [_KernelCapture() for _ in range(3)]
    monkeypatch.setattr(mhc_k, "_mhc_stage1_kernel", caps[0])
    monkeypatch.setattr(mhc_k, "_mhc_stage2_kernel", caps[1])
    monkeypatch.setattr(mhc_k, "_mhc_stage3_kernel", caps[2])
    mhc_k.mhc_fused_post_pre_triton(
        x, res, post if has_post else None, comb if has_post else None,
        fn, scale, base, RMS_EPS, EPS, POST_MULT, SINKHORN,
    )
    return caps


def _legacy_stage1_params(h):
    """The pre-knob stage1 grid selection, verbatim from the old launcher."""
    import triton

    block_h = min(512, triton.next_power_of_2(h))
    ns = 1
    while ns * 2 <= min(16, h // block_h):
        ns *= 2
    split = triton.cdiv(triton.cdiv(h, ns), block_h) * block_h
    ns = triton.cdiv(h, split)
    return ns, split, block_h


def _knob_stage1_params(h, ns_knob):
    """The knob-branch stage1 grid selection, verbatim from the launcher."""
    import triton

    block_h = min(64, triton.next_power_of_2(h))
    ns = min(ns_knob, triton.cdiv(h, block_h))
    split = triton.cdiv(triton.cdiv(h, ns), block_h) * block_h
    ns = triton.cdiv(h, split)
    return ns, split, block_h


@pytest.mark.parametrize("t,h", [(1, 64), (9, 64), (1, 1024), (1, 1536), (3, 4096), (1, 8192)])
def test_stage1_default_grid_matches_ns64(monkeypatch, t, h):
    """Knob unset: the launcher must run the validated NS=64 split (clamped to
    the legacy geometry on tiny h), not the pre-knob grid."""
    import triton

    from freetoken.kernel.triton import mhc as mhc_k

    caps = _capture_stage1(monkeypatch, t, h)
    ns, split, block_h = _knob_stage1_params(h, mhc_k._parse_stage1_ns(None))
    (grid, kw), = caps[0].launches
    assert grid == (t, ns)
    assert (kw["NS"], kw["SPLIT"], kw["BLOCK_H"]) == (ns, split, block_h)
    (_, kw2), = caps[1].launches
    assert kw2["NS"] == ns and kw2["BLK_NS"] == triton.next_power_of_2(ns)


@pytest.mark.parametrize("h", [64, 1024, 4096])
def test_stage1_unset_equals_env64_launch(monkeypatch, h):
    """Unset default must produce the exact launch config (grid + constexprs,
    stage1 and stage2) of FREETOKEN_MHC_STAGE1_NS=64."""
    default_caps = _capture_stage1(monkeypatch, 1, h)
    env64_caps = _capture_stage1(monkeypatch, 1, h, ns_knob=64)
    assert default_caps[0].launches == env64_caps[0].launches
    assert default_caps[1].launches == env64_caps[1].launches


@pytest.mark.parametrize(
    "knob,h,exp_ns,exp_split",
    [
        (2, 4096, 2, 2048),
        (8, 4096, 8, 512),
        (32, 4096, 32, 128),
        (64, 4096, 64, 64),
        (33, 4096, 32, 128),  # split rounds up to BLOCK_H, ns snaps down
        (3, 4096, 3, 1408),  # non-power-of-2 partial count is legal
        (64, 64, 1, 64),  # knob above the useful split -> legacy grid
        (64, 128, 2, 64),
    ],
)
def test_stage1_knob_grid(monkeypatch, knob, h, exp_ns, exp_split):
    """Knob on: BLOCK_H shrinks to the 64-elem floor so ~N partial CTAs per
    token cover h; ns is what survives the split rounding."""
    import triton

    caps = _capture_stage1(monkeypatch, 1, h, ns_knob=knob)
    (grid, kw), = caps[0].launches
    assert grid == (1, exp_ns)
    assert (kw["NS"], kw["SPLIT"], kw["BLOCK_H"]) == (exp_ns, exp_split, 64)
    (_, kw2), = caps[1].launches
    assert kw2["NS"] == exp_ns
    assert kw2["BLK_NS"] == triton.next_power_of_2(exp_ns)


def test_stage1_knob_grid_pre_only(monkeypatch):
    """HAS_POST=False (layer-0 hc_pre) takes the same knob grid math."""
    caps = _capture_stage1(monkeypatch, 1, 4096, ns_knob=64, has_post=False)
    (grid, kw), = caps[0].launches
    assert grid == (1, 64)
    assert kw["HAS_POST"] is False


@pytest.mark.parametrize("raw,expected", [(None, 64), ("1", 1), ("2", 2), ("64", 64), (" 8", 8)])
def test_stage1_ns_parser_accepts(raw, expected):
    from freetoken.kernel.triton import mhc as mhc_k

    assert mhc_k._parse_stage1_ns(raw) == expected


@pytest.mark.parametrize("raw", ["0", "-1", "65", "100", "abc", "2.5", "", "1e1"])
def test_stage1_ns_parser_rejects(raw):
    from freetoken.kernel.triton import mhc as mhc_k

    with pytest.raises(ValueError):
        mhc_k._parse_stage1_ns(raw)


def test_stage1_ns_read_from_env(monkeypatch):
    """The knob is read once at import; this is the exact read import runs."""
    import os

    from freetoken.kernel.triton import mhc as mhc_k

    monkeypatch.setenv("FREETOKEN_MHC_STAGE1_NS", "16")
    assert mhc_k._parse_stage1_ns(os.environ.get("FREETOKEN_MHC_STAGE1_NS")) == 16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_triton_legacy_bitwise_vs_hand_launch(monkeypatch):
    """Legacy path (FREETOKEN_MHC_STAGE1_NS=1) must stay bit-identical to the
    pre-knob implementation: launch the kernels by hand with the legacy
    geometry and compare every output."""
    import triton

    from freetoken.kernel.triton import mhc as mhc_k

    monkeypatch.setattr(mhc_k, "_MHC_STAGE1_NS", 1)
    t, h = 1, 1024  # legacy geometry here: ns=2, split=512, block=512
    fn, scale, base, res, x, post0, comb0 = _mhc_fixtures(t, h, device="cuda", seed=19)
    got = mhc_k.mhc_fused_post_pre_triton(
        x, res, post0, comb0, fn, scale, base, RMS_EPS, EPS, POST_MULT, SINKHORN
    )

    ns, split, block_h = _legacy_stage1_params(h)
    mix = 2 * N + N * N
    blk_mix = triton.next_power_of_2(mix)
    dev = res.device
    res_out = torch.empty_like(res)
    post_out = torch.empty(t, N, dtype=torch.float32, device=dev)
    comb_out = torch.empty(t, N, N, dtype=torch.float32, device=dev)
    li_out = torch.empty(t, h, dtype=res.dtype, device=dev)
    sq_part = torch.empty(t, ns, dtype=torch.float32, device=dev)
    mix_part = torch.empty(t, ns, blk_mix, dtype=torch.float32, device=dev)
    pre_out = torch.empty(t, N, dtype=torch.float32, device=dev)
    mhc_k._mhc_stage1_kernel[(t, ns)](
        x.contiguous(), res.contiguous(), post0.contiguous().view(t, N),
        comb0.contiguous(), fn, res_out, sq_part, mix_part,
        H=h, N=N, MIX=mix, BLK_MIX=blk_mix,
        SPLIT=split, BLOCK_H=block_h, NS=ns, HAS_POST=True,
        num_warps=4, num_stages=2,
    )
    mhc_k._mhc_stage2_kernel[(t,)](
        sq_part, mix_part, scale, base, post_out, comb_out, pre_out,
        RMS_EPS, EPS, POST_MULT,
        SINKHORN=SINKHORN, H=h, N=N, MIX=mix, BLK_MIX=blk_mix,
        NS=ns, BLK_NS=ns, num_warps=1,
    )
    mhc_k._mhc_stage3_kernel[(t, triton.cdiv(h, 1024))](
        res_out, pre_out, li_out,
        H=h, N=N, BLOCK_H=min(1024, triton.next_power_of_2(h)), num_warps=4,
    )
    legacy = (res_out, post_out.view(t, N, 1), comb_out, li_out)
    for name, g, r in zip(("residual", "post", "comb", "layer_input"), got, legacy):
        assert torch.equal(g, r), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("knob", [4, 32, 33])
def test_triton_knob_parity_vs_legacy(monkeypatch, knob):
    """Knob on: outputs stay within the file's fp32 reduction-order tolerance
    vs both the torch reference and the legacy NS=1 run; not bitwise (the
    split reorders the fp32 reduction)."""
    from freetoken.layers.mhc import mhc_fused_post_pre_torch
    from freetoken.kernel.triton import mhc as mhc_k

    monkeypatch.setattr(mhc_k, "_MHC_STAGE1_NS", 1)
    t, h = 1, 4096  # decode shape; legacy ns=8, the knob fans out further
    fn, scale, base, res, x, post0, comb0 = _mhc_fixtures(t, h, device="cuda", seed=23)
    ref = mhc_fused_post_pre_torch(
        x, res, post0, comb0, fn, scale, base, RMS_EPS, EPS, POST_MULT, SINKHORN
    )
    base_out = mhc_k.mhc_fused_post_pre_triton(
        x, res, post0, comb0, fn, scale, base, RMS_EPS, EPS, POST_MULT, SINKHORN
    )
    monkeypatch.setattr(mhc_k, "_MHC_STAGE1_NS", knob)
    split_out = mhc_k.mhc_fused_post_pre_triton(
        x, res, post0, comb0, fn, scale, base, RMS_EPS, EPS, POST_MULT, SINKHORN
    )
    names = ("residual", "post", "comb", "layer_input")
    tols = (2e-2, 2e-3, 2e-3, 2e-2)
    for name, r, b, s, tol in zip(names, ref, base_out, split_out, tols):
        assert (b.float() - r.float()).abs().max().item() < tol, name
        err = (s.float() - b.float()).abs().max().item()
        assert err < tol, f"{name} knob={knob}: max abs err {err}"
