import time

import numpy as np

GIB = 1 << 30
buf = np.zeros(GIB, dtype=np.uint8)
out = np.zeros(GIB, dtype=np.uint8)

# warmup
np.add.reduce(buf)
out[:] = buf

# pure DRAM read: reduce over 1 GiB, 4 reps
t0 = time.perf_counter()
reps = 8
for _ in range(reps):
    s = np.add.reduce(buf)
dt = time.perf_counter() - t0
print(f"pure read:  {GIB * reps / dt / 1e9:.1f} GB/s (checksum {s})")

# pure DRAM write: fill 1 GiB, 4 reps
t0 = time.perf_counter()
for _ in range(reps):
    out.fill(0x5A)
dt = time.perf_counter() - t0
print(f"pure write: {GIB * reps / dt / 1e9:.1f} GB/s")

# non-temporal-ish write via copy (still RFO on load side)
t0 = time.perf_counter()
for _ in range(reps):
    out[:] = buf
dt = time.perf_counter() - t0
print(f"copy r+w+rfo: {GIB * reps / dt / 1e9:.1f} GB/s")
