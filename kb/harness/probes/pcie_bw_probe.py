import threading
import time
from collections import Counter

GPU_ADDR = "0000:01:00.0"
RP_ADDR = "0000:00:01.0"


def sysfs(addr, name):
    try:
        with open(f"/sys/bus/pci/devices/{addr}/{name}") as f:
            return f.read().strip()
    except Exception as e:
        return f"err:{e}"


import torch

assert torch.cuda.is_available(), "CUDA not available"
print(f"torch={torch.__version__}, gpu={torch.cuda.get_device_name(0)}", flush=True)

stop = threading.Event()
observed = []


def poller():
    while not stop.is_set():
        observed.append(
            (
                sysfs(GPU_ADDR, "current_link_speed"),
                sysfs(GPU_ADDR, "current_link_width"),
                sysfs(RP_ADDR, "current_link_speed"),
                sysfs(RP_ADDR, "current_link_width"),
            )
        )
        time.sleep(0.02)


def bench(host, devb, h2d: bool, iters=10):
    for _ in range(3):
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


poll_thread = threading.Thread(target=poller)
poll_thread.start()

for nbytes in (256 << 20, 1 << 30, 2 << 30):
    host = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    devb = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    h2d = bench(host, devb, True)
    d2h = bench(host, devb, False)

    devc = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    for _ in range(3):
        devc.copy_(devb)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(10):
        devc.copy_(devb)
    e.record()
    torch.cuda.synchronize()
    d2d = nbytes * 10 / (s.elapsed_time(e) / 1000) / 1e9

    print(
        f"size={nbytes >> 20} MiB: H2D={h2d:.1f} GB/s  D2H={d2h:.1f} GB/s  D2D={d2d:.1f} GB/s",
        flush=True,
    )
    del host, devb, devc

stop.set()
poll_thread.join()

print("link under load (top 3):", Counter(observed).most_common(3))
print(
    "link idle now:",
    sysfs(GPU_ADDR, "current_link_speed"),
    sysfs(GPU_ADDR, "current_link_width"),
)
