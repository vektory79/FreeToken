#!/usr/bin/env python3
"""Decompose a chrome trace (torch.profiler export) from the decode window.

Usage: analyze_trace.py <trace.json> [tag]
Prints: window wall, GPU busy/idle, per-class kernel totals, top kernels, top cpu ops.
"""
import collections
import json
import re
import sys

CLASSES = [
    ("fetch_copy", re.compile(r"index_copy|fast_index", re.I)),
    ("moe", re.compile(r"moe|expert", re.I)),
    ("dense_quant_gemm", re.compile(r"mul_mat|ggml|mmvq|_a8|quant|gemm|nvfp4|marlin", re.I)),
    ("attention", re.compile(r"attn|dsa|flash|paged|indexer|kpool|gather_kv|score|topk_soft", re.I)),
    ("gdn", re.compile(r"gdn|delta|recurrent|conv1d|l2norm|chunk_", re.I)),
    ("sampler_head", re.compile(r"softmax|argmax|multinomial|sampling|logits|lm_head", re.I)),
    ("memcpy", re.compile(r"memcpy|memset", re.I)),
]


def classify(name, cat):
    if cat in ("gpu_memcpy", "gpu_memset"):
        return "memcpy"
    for tag, rx in CLASSES:
        if tag == "memcpy":
            continue
        if rx.search(name or ""):
            return tag
    return "other"


def main():
    path = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else "run"
    doc = json.load(open(path))
    ev = doc.get("traceEvents", [])
    gpu_cats = {"kernel", "gpu_memcpy", "gpu_memset"}
    gpus = [e for e in ev if e.get("ph") == "X" and e.get("cat") in gpu_cats]
    cpus = [e for e in ev if e.get("ph") == "X" and e.get("cat") not in gpu_cats]
    if not gpus:
        print("no GPU events found; cats seen:",
              collections.Counter(e.get("cat") for e in ev if e.get("ph") == "X").most_common(10))
        return
    t0 = min(e["ts"] for e in gpus)
    t1 = max(e["ts"] + e.get("dur", 0) for e in gpus)
    wall_ms = (t1 - t0) / 1e3
    busy_ms = sum(e.get("dur", 0) for e in gpus) / 1e3
    per_name = collections.defaultdict(lambda: [0, 0.0])
    per_class = collections.defaultdict(lambda: [0, 0.0])
    for e in gpus:
        d = e.get("dur", 0) / 1e3
        per_name[e["name"]][0] += 1
        per_name[e["name"]][1] += d
        c = classify(e["name"], e.get("cat"))
        per_class[c][0] += 1
        per_class[c][1] += d
    print("== %s (%s) ==" % (tag, path))
    print("window: %.1f ms wall over %d GPU events; busy %.1f ms (%.1f%%); idle %.1f ms (%.1f%%)"
          % (wall_ms, len(gpus), busy_ms, 100 * busy_ms / wall_ms,
             wall_ms - busy_ms, 100 * (wall_ms - busy_ms) / wall_ms))
    print("\n-- per class (GPU ms, share of busy) --")
    for c, (n, ms) in sorted(per_class.items(), key=lambda kv: -kv[1][1]):
        print("  %-16s %8.2f ms  %5.1f%%  n=%d" % (c, ms, 100 * ms / busy_ms, n))
    print("\n-- top 25 kernels --")
    for name, (n, ms) in sorted(per_name.items(), key=lambda kv: -kv[1][1])[:25]:
        print("  %8.2f ms  n=%-6d mean=%7.1f us  %s" % (ms, n, 1e3 * ms / n, name[:110]))
    cpu_tot = collections.Counter()
    for e in cpus:
        if e.get("cat") in ("cpu_op", "user_annotation"):
            cpu_tot[e["name"][:90]] += e.get("dur", 0) / 1e3
    print("\n-- top 15 cpu ops (total, may double-count children) --")
    for name, ms in cpu_tot.most_common(15):
        print("  %8.2f ms  %s" % (ms, name))
    steps = None
    try:
        import os
        stats = json.load(open(os.path.join(os.path.dirname(path), "stats_%s.json" % tag)))
        n = stats.get("n_layer_calls", 0)
        if n:
            print("\nn_layer_calls at dump: %d (~%d decode steps)" % (n, n // 42))
    except Exception:
        pass


if __name__ == "__main__":
    main()
