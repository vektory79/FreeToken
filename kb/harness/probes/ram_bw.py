import threading
import time

import torch

CHUNK = 1 << 30  # 1 GiB
N = 4
REPS = 4


def run(nthreads: int):
    srcs = [torch.empty(CHUNK, dtype=torch.uint8) for _ in range(nthreads)]
    dsts = [torch.empty(CHUNK, dtype=torch.uint8) for _ in range(nthreads)]
    barrier = threading.Barrier(nthreads)
    results = [0.0] * nthreads

    def worker(i):
        dsts[i].copy_(srcs[i])  # warmup
        barrier.wait()
        t0 = time.perf_counter()
        for _ in range(REPS):
            dsts[i].copy_(srcs[i])
        results[i] = CHUNK * REPS / (time.perf_counter() - t0) / 1e9

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(nthreads)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    agg = CHUNK * nthreads * REPS / wall / 1e9
    print(
        f"threads={nthreads}: aggregate memcpy={agg:.1f} GB/s "
        f"(DRAM read+write combined; peak read ~{agg / 2:.0f} GB/s)"
    )


run(1)
run(4)
