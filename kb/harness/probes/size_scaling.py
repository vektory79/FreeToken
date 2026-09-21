import torch

assert torch.cuda.is_available()
dev = torch.device("cuda")
print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

print(f"{'size':>10} {'iters':>6} {'H2D GB/s':>10} {'D2H GB/s':>10}")


def bench(host, devb, h2d, iters):
    for _ in range(5):
        if h2d:
            devb.copy_(host, non_blocking=True)
        else:
            host.copy_(devb, non_blocking=True)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        if h2d:
            devb.copy_(host, non_blocking=True)
        else:
            host.copy_(devb, non_blocking=True)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end)
    return host.numel() * iters / (ms / 1000) / 1e9


for mib in (4, 16, 64, 256, 1024, 4096):
    nbytes = mib << 20
    iters = max(30, min(400, (8 << 30) // nbytes))
    host = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    devb = torch.empty(nbytes, dtype=torch.uint8, device=dev)
    h2d = bench(host, devb, True, iters)
    d2h = bench(host, devb, False, iters)
    print(f"{mib:>8} MiB {iters:>6} {h2d:>10.1f} {d2h:>10.1f}", flush=True)
    del host, devb
    torch.cuda.empty_cache()
