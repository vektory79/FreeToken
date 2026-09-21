#!/usr/bin/env python3
"""Standalone dense-q8_0 regime probe (no serve boot).

Replicates the measured kda_in_proj launch exactly: q8_0 weight
[N=24896 (gridX=778), K=4096 (row_bytes=4352)], x bf16 [M, 4096],
MMQ_X_Q8_0=4 -> grid = (778, ceil(M/4)). Timing sweep over M measures the
per-row-tile slope; ncu profiles one launch for dram__bytes / L2 hit rate /
stall mix (the tile-4 traffic-vs-issue discriminator: bytes/MAC is
shape-constant for q8_0, so timing alone cannot separate the two).
"""
import sys

import torch

from freetoken.kernel.gguf import ggml_mul_mat_a8

N = 24896
K = 4096
ROW_BYTES = K // 32 * 34
Q8_0 = 8

torch.manual_seed(0)
qw = torch.zeros(N, ROW_BYTES, dtype=torch.uint8)
qw[:, 0] = 0x00
qw[:, 1] = 0x3C  # fp16 1.0 scale per 32-block; int8 payload random 0..120
qw[:, 2:] = torch.randint(0, 121, (N, ROW_BYTES - 2), dtype=torch.uint8)
qw = qw.cuda()
x = (torch.randn(8128, K, dtype=torch.bfloat16) * 0.05).cuda()

# warmup (3 launches; ncu skips these)
for _ in range(3):
    out = ggml_mul_mat_a8(qw, x, Q8_0, N)
torch.cuda.synchronize()
n_nan = torch.isnan(out.float()).sum().item()
print("warmup ok: out", tuple(out.shape), "nan:", n_nan, flush=True)


def bench(M, iters=20):
    xs = (torch.randn(M, K, dtype=torch.bfloat16) * 0.05).cuda()
    for _ in range(2):
        ggml_mul_mat_a8(qw, xs, Q8_0, N)
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(iters):
        ggml_mul_mat_a8(qw, xs, Q8_0, N)
    ev1.record()
    torch.cuda.synchronize()
    ms = ev0.elapsed_time(ev1) / iters
    rows = (M + 3) // 4
    mac = 2 * M * N * K
    print("M=%5d gridY=%5d  %8.3f ms  %7.2f us/row-tile  %6.2f TFLOP/s  %7.1f GB/s@naive"
          % (M, rows, ms, ms * 1e3 / rows, mac / ms / 1e9, rows * N * ROW_BYTES / ms / 1e6),
          flush=True)
    return ms


print("\n=== M sweep (tile 4, weight fixed 108.35 MB) ===")
for M in (8128, 4064, 2032, 1016, 508, 254, 127, 64):
    bench(M)
