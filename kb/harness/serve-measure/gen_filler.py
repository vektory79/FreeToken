#!/usr/bin/env python3
import os, sys

DIR = "/media/ai/src/FreeToken/.tasks/ft-gguf-serve-tuning"

P1 = ("The scheduler interleaves prefill chunks across pending requests while the radix cache reuses "
      "shared prefixes to avoid redundant computation, and the budget planner reserves KV pages before "
      "expert slots are greedily filled for the serving pipeline.")
P2 = ("A quantized expert bank streams from host memory over the peripheral bus, the gather kernel reads "
      "only the selected rows, and the fused projection writes activations back through the pinned "
      "staging buffers that the allocator prepared earlier in the boot sequence.")

def build(par, tag, reps):
    lines = []
    for i in range(reps):
        lines.append(f"{tag} {i:06d}. {par}")
    return "\n".join(lines) + "\n"

def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 1373
    os.makedirs(os.path.join(DIR, "blocks"), exist_ok=True)
    with open(os.path.join(DIR, "filler.txt"), "w") as f:
        f.write(build(P1, "Record", reps))
    for j in range(8):
        with open(os.path.join(DIR, "blocks", f"block_{j}.txt"), "w") as f:
            f.write(build(P2 + (" Variant %02d." % j), "Appendix%02d" % j, reps))
    n = os.path.getsize(os.path.join(DIR, "filler.txt"))
    print("filler bytes", n, "reps", reps)

if __name__ == "__main__":
    main()
