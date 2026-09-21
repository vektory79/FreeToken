#!/usr/bin/env python3
"""Format discriminator: q8_0 vs q6_K dense GEMM, identical shape/MACs.

q8_0 = 1.0625 B/elem, q6_K = 210/256 = 0.8203 B/elem -> naive weight traffic
ratio 0.772 at equal N,K,M (bytes/MAC 0.1328 vs 0.1025). If the dense kernel
is weight-traffic-bound the time ratio tracks ~0.77; if it is issue/ALU-bound
(same MACs, heavier q6_K decode) the ratio is >= 1.
"""
import torch

from freetoken.kernel.gguf import ggml_mul_mat_a8

N = 24896
K = 4096
M = 8128

torch.manual_seed(0)


def make_q8_0():
    rb = K // 32 * 34
    w = torch.zeros(N, rb, dtype=torch.uint8)
    w[:, 0] = 0x00
    w[:, 1] = 0x3C
    w[:, 2:] = torch.randint(0, 121, (N, rb - 2), dtype=torch.uint8)
    return w, 8


def make_q6_K():
    rb = K // 256 * 210  # 3360
    w = torch.zeros(N, rb, dtype=torch.uint8)
    # block_q6_K: [2B d fp16][16B scales int8][128B ql][64B qh]
    w[:, 0] = 0x00
    w[:, 1] = 0x3C
    w[:, 2:18] = 4  # scales small positive
    w[:, 18:] = torch.randint(0, 121, (N, rb - 18), dtype=torch.uint8)
    return w, 14


def bench(tag, w, typ, iters=30):
    x = (torch.randn(M, K, dtype=torch.bfloat16) * 0.05).cuda()
    wc = w.cuda()
    for _ in range(3):
        out = ggml_mul_mat_a8(wc, x, typ, N)
    torch.cuda.synchronize()
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(iters):
        ggml_mul_mat_a8(wc, x, typ, N)
    ev1.record()
    torch.cuda.synchronize()
    ms = ev0.elapsed_time(ev1) / iters
    mac = 2 * M * N * K
    wmb = wc.shape[1] * N / 1e6
    print("%-6s %8.3f ms  %6.2f TFLOP/s  W=%.1f MB  naive_BW=%.0f GB/s"
          % (tag, ms, mac / ms / 1e9, wmb, (M / 4) * wc.shape[1] * N / ms / 1e6), flush=True)
    return ms


w8, t8 = make_q8_0()
w6, t6 = make_q6_K()
m8 = bench("q8_0", w8, t8)
m6 = bench("q6_K", w6, t6)
print("time ratio q6_K/q8_0 = %.3f | traffic-model ratio = %.3f (0.8203/1.0625)"
      % (m6 / m8, 0.8203 / 1.0625))
