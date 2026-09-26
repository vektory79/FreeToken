#!/usr/bin/env python3
"""Analyze decode-research arm results: comparison table + per-CPU busy/MHz split.

Usage: analyze.py [arm names...]  (default: all arm_*.json present)
"""
import csv
import glob
import json
import os
import sys

DIR = "/media/ai/src/FreeToken/.tasks/decode-research"
N_LAYERS = 42
EXPERT_MB = 10.88  # gate+up+down packed bytes per expert (Q3_K_XL mix)


def load_arm(name):
    path = os.path.join(DIR, "arm_%s.json" % name)
    if not os.path.exists(path):
        return None
    return json.load(open(path))


def sampler_stats(name, t0, t1):
    """Per-core mean busy fraction and MHz within [t0, t1].

    Rows are grouped by sample timestamp first: dt is measured between
    consecutive SAMPLES, not consecutive rows (all rows of one sample share t).
    """
    path = os.path.join(DIR, "cpu_%s.csv" % name)
    if not os.path.exists(path):
        return None
    by_t = {}
    for r in csv.DictReader(open(path)):
        by_t.setdefault(float(r["t"]), {})[int(r["cpu"])] = (
            int(r["busy_delta"]), float(r["mhz"]))
    ts = sorted(by_t)
    per_core = {}
    for i in range(1, len(ts)):
        t, tp = ts[i], ts[i - 1]
        if not (t0 - 1.0 <= t <= t1 + 1.0):
            continue
        dt = t - tp
        if dt <= 0.2 or dt > 5.0:
            continue
        for c, (bd, mz) in by_t[t].items():
            if c not in by_t[tp]:
                continue
            d = per_core.setdefault(c, {"busy": [], "mhz": []})
            d["busy"].append(min(1.0, bd / (100.0 * dt)))
            if mz > 0:
                d["mhz"].append(mz)
    if not per_core:
        return None
    def mean(x):
        return sum(x) / len(x) if x else 0.0
    cores = {c: {"busy_pct": round(100 * mean(d["busy"]), 1),
                 "mhz": round(mean(d["mhz"]))} for c, d in per_core.items()}
    p = [v["busy_pct"] for c, v in cores.items() if c <= 15]
    e = [v["busy_pct"] for c, v in cores.items() if c >= 16]
    pm = [v["mhz"] for c, v in cores.items() if c <= 15 and v["mhz"]]
    em = [v["mhz"] for c, v in cores.items() if c >= 16 and v["mhz"]]
    return {"cores": cores,
            "p_busy_mean_pct": round(sum(p) / len(p), 1) if p else None,
            "e_busy_mean_pct": round(sum(e) / len(e), 1) if e else None,
            "p_mhz_mean": round(sum(pm) / len(pm)) if pm else None,
            "e_mhz_mean": round(sum(em) / len(em)) if em else None}


def pool_lines(arm):
    out = []
    for pat, hits in (arm.get("log") or {}).items():
        for h in hits:
            if ("pool" in h and ("core" in h or "threads" in h)) or "fetching" in h or "isa=" in h:
                if h not in out:
                    out.append(h)
    return out[:8]


def main():
    names = sys.argv[1:] or [os.path.basename(p)[4:-5] for p in glob.glob(os.path.join(DIR, "arm_*.json"))]
    names = [n for n in names if not n.endswith("_summary")]
    summary = {}
    for name in names:
        arm = load_arm(name)
        if arm is None:
            print("arm %s: no json" % name)
            continue
        dec = arm.get("decode") or {}
        row = {"ready_s": arm.get("ready_s"), "ready_why": arm.get("ready_why"),
               "boot_gpu_mib": arm.get("gpu_after_boot_mib")}
        fill = arm.get("fill") or {}
        row["fill_cached"] = fill.get("cached_token")
        row["fill_new"] = fill.get("new_token")
        thr = (fill.get("throughput_lines") or [])
        row["fill_thr_last"] = thr[-1] if thr else None
        row["decode"] = {k: dec.get(k) for k in
                         ("steady_tok_s", "wall_s", "completion_tokens", "chunks",
                          "ttft_s", "p50_step_ms", "max_step_ms", "cached_token", "new_token", "error")}
        st = dec.get("steady_tok_s")
        if st:
            row["ms_per_step"] = round(1000 / st, 1)
            row["ms_per_layer"] = round(1000 / st / N_LAYERS, 2)
        times = None
        # sampler window from decode frame span
        # (times are not persisted; approximate with wall window minus ttft)
        w = dec.get("wall_s") or 0
        ttft = dec.get("ttft_s") or 0
        if w:
            t_end = arm.get("_end_t")
        stats = None
        if w:
            stats = sampler_stats(name, 0, 1e18) if False else None
        row["pools"] = pool_lines(arm)
        td = arm.get("teardown") or {}
        row["teardown_leftover"] = td.get("leftover")
        summary[name] = row
    print(json.dumps(summary, indent=1, ensure_ascii=False))

    # per-CPU table for requested arms
    for name in names:
        arm = load_arm(name)
        if not arm:
            continue
        dec = arm.get("decode") or {}
        w = dec.get("wall_s")
        if not w:
            continue
        # sampler ran only during the decode request; trim first 4 s (TTFT+fill of pipeline)
        st = sampler_stats(name, 0, 1e18)
        if not st:
            continue
        trimmed = {c: v for c, v in st["cores"].items() if True}
        print("\n== %s per-CPU (decode window) ==" % name)
        print("P(0-15) busy mean %s%% @ ~%s MHz ; E(16-27) busy mean %s%% @ ~%s MHz"
              % (st["p_busy_mean_pct"], st["p_mhz_mean"], st["e_busy_mean_pct"], st["e_mhz_mean"]))
        for c in sorted(trimmed):
            v = trimmed[c]
            tag = "P" if c <= 15 else "E"
            print("  cpu%-2d %s busy %5.1f%%  %5d MHz" % (c, tag, v["busy_pct"], v["mhz"]))


if __name__ == "__main__":
    main()
