#!/usr/bin/env python3
"""v3a liveness + per-format analysis from the capture-from-launch probe.

Compares the tile-16 capture (v3aprobe.sqlite, exported from the newest
report*.nsys-rep in the v3 dir) against the tile-4 baseline capture
(.tasks/mmq-prefill-kernel/v2probe2.sqlite, same 65,585-token fill, same
winner flags, grouped path tile 4).

Assertions (T46 for v3a):
  A1 grouped MMQ kernel names present per format in the prefill phase
  A2 ZERO moe_vec* in the prefill phase
  A3 moe_vec* present in the decode phase (CUDA-graph replays)
  A4 launch-count math: grouped = 126/chunk-equiv, moe_align = 42/chunk-equiv
     (prefill phase = 8 full 8128 chunks + 561-tail = 9 chunk-equivalents)
  A5 per-launch CTA counts shrink ~tile x vs baseline (grid.y = numel/tile)
Per-format read: median duration per grouped kernel name, tile16 vs tile4.
Writes v3a_probe_liveness.json; exit 3 on any assertion failure.
"""
import glob
import json
import os
import sqlite3
import statistics
import subprocess
import sys

VDIR = "/media/ai/src/FreeToken/.tasks/mmq-v3-stationary-moe"
BASE_SQ = "/media/ai/src/FreeToken/.tasks/mmq-prefill-kernel/v2probe2.sqlite"
NSYS = "/usr/local/bin/nsys"
GAP_S = 1.5
GROUPED = ("moe_q8_0", "moe_q6_K", "moe_iq3_xxs", "moe_iq4_xs")
CHUNK_EQUIVS = 9  # 8 full 8128 chunks + 561-tail


def export_sqlite(rep, sq):
    if os.path.exists(sq):
        os.remove(sq)
    r = subprocess.run([NSYS, "export", "--type", "sqlite", "--force-overwrite",
                        "true", "-o", sq, rep], capture_output=True, text=True,
                       timeout=1800)
    print("export rc:", r.returncode, (r.stderr or "")[-200:])
    if r.returncode != 0:
        sys.exit(2)


def load(sq):
    con = sqlite3.connect(sq)
    rows = con.execute("""
        SELECT k.start, k.end, s.value, k.gridX, k.gridY, k.gridZ
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName = s.id
        ORDER BY k.start
    """).fetchall()
    con.close()
    return rows


def classify(name):
    if name.startswith("moe_vec"):
        return "vec"
    if name.startswith("moe_") and "align" not in name:
        return "grouped"
    if "align" in name and "moe" in name:
        return "align"
    return "other"


def phases(rows):
    starts = [r[0] for r in rows]
    cuts = [0]
    for i in range(1, len(rows)):
        if starts[i] - starts[i - 1] > GAP_S * 1e9:
            cuts.append(i)
    cuts.append(len(rows))
    return [(a, b) for a, b in zip(cuts, cuts[1:]) if b > a]


def phase_summary(rows, a, b):
    seg = rows[a:b]
    names = {}
    for st, en, nm, gx, gy, gz in seg:
        c = classify(nm)
        d = names.setdefault(nm, {"class": c, "n": 0, "durs": [], "ctas": []})
        d["n"] += 1
        d["durs"].append(en - st)
        if c != "other":
            d["ctas"].append(int(gx) * int(gy) * int(gz))
    for d in names.values():
        d["med_dur_ms"] = round(statistics.median(d["durs"]) / 1e6, 4)
        d["sum_s"] = round(sum(d["durs"]) / 1e9, 3)
        d["med_cta"] = int(statistics.median(d["ctas"])) if d["ctas"] else None
        del d["durs"], d["ctas"]
    span = (rows[b - 1][1] - rows[a][0]) / 1e9
    return {"span_s": round(span, 2), "n": b - a, "kernels": names,
            "grouped_count": sum(d["n"] for d in names.values() if d["class"] == "grouped"),
            "vec_count": sum(d["n"] for d in names.values() if d["class"] == "vec"),
            "align_count": sum(d["n"] for d in names.values() if d["class"] == "align")}


def pick_phases(rows, tag):
    phs = phases(rows)
    print("%s: %d kernel records, %d phases" % (tag, len(rows), len(phs)))
    stats = [phase_summary(rows, a, b) for a, b in phs]
    for i, ps in enumerate(stats):
        print("  phase %d: n=%6d span=%8.2f s grouped=%5d vec=%5d align=%4d"
              % (i, ps["n"], ps["span_s"], ps["grouped_count"], ps["vec_count"],
                 ps["align_count"]))
    prefill = max(stats, key=lambda ps: ps["grouped_count"])
    decode = stats[-1]
    boot = stats[0]
    pi = max(range(len(stats)), key=lambda i: stats[i]["grouped_count"])
    return boot, prefill, decode, stats, phs[pi]


MODEL_FORMATS = ("moe_iq3_xxs", "moe_iq4_xs", "moe_q6_K")  # moe_q8_0 absent in this model


def vec_in_grouped_span(rows, a, b):
    """moe_vec* launches intersecting [first grouped start, last grouped end].
    The phase split may merge boot graph replays (before) and the prefill
    request's own 3x42x3 one-token GEMV burst (after) into the same phase -
    both are outside the prefill chunk span (v2-ab.md explained non-signal)."""
    seg = rows[a:b]
    g0 = min((st for st, en, nm, *_ in seg if classify(nm) == "grouped"), default=None)
    g1 = max((en for st, en, nm, *_ in seg if classify(nm) == "grouped"), default=None)
    if g0 is None:
        return 0, None, None
    n = sum(1 for st, en, nm, *_ in seg
            if classify(nm) == "vec" and en > g0 and st < g1)
    return n, g0, g1


def main():
    reps = sorted(glob.glob(os.path.join(VDIR, "report*.nsys-rep")),
                  key=os.path.getmtime)
    if not reps:
        print("FATAL: no report*.nsys-rep in", VDIR)
        sys.exit(2)
    rep = reps[-1]
    print("report:", rep)
    sq16 = os.path.join(VDIR, "v3aprobe.sqlite")
    export_sqlite(rep, sq16)
    rows16 = load(sq16)
    rows4 = load(BASE_SQ)
    boot16, pre16, dec16, all16, rng16 = pick_phases(rows16, "tile16")
    boot4, pre4, dec4, all4, rng4 = pick_phases(rows4, "tile4(base)")

    gnames16 = sorted(n for n, d in pre16["kernels"].items() if d["class"] == "grouped")
    gnames4 = sorted(n for n, d in pre4["kernels"].items() if d["class"] == "grouped")
    a1 = (bool(gnames16) and set(gnames16) <= set(GROUPED)
          and set(MODEL_FORMATS) <= set(gnames16))
    v16, gs16, ge16 = vec_in_grouped_span(rows16, *rng16)
    v4, gs4, ge4 = vec_in_grouped_span(rows4, *rng4)
    a2 = v16 == 0 and v4 == 0
    print("vec-in-grouped-span: tile16=%d tile4=%d" % (v16, v4))
    a3 = dec16["vec_count"] > 0
    a4 = (pre16["grouped_count"] == 126 * CHUNK_EQUIVS
          and pre16["align_count"] == 42 * CHUNK_EQUIVS)
    a5 = {}
    for n in gnames16:
        c4 = pre4["kernels"].get(n, {}).get("med_cta")
        c16 = pre16["kernels"][n]["med_cta"]
        if c4:
            a5[n] = {"cta4": c4, "cta16": c16, "ratio": round(c4 / c16, 2)}
    a5_ok = all(3.0 <= v["ratio"] <= 5.0 for v in a5.values()) and len(a5) == len(gnames16)

    fmt = {}
    for n in gnames16:
        d4 = pre4["kernels"].get(n, {})
        d16 = pre16["kernels"][n]
        fmt[n] = {
            "n16": d16["n"], "med_ms16": d16["med_dur_ms"], "sum_s16": d16["sum_s"],
            "med_cta16": d16["med_cta"],
            "n4": d4.get("n"), "med_ms4": d4.get("med_dur_ms"), "sum_s4": d4.get("sum_s"),
            "med_cta4": d4.get("med_cta"),
            "med_speedup": round(d4["med_dur_ms"] / d16["med_dur_ms"], 2) if d4.get("med_dur_ms") else None,
            "sum_speedup": round(d4["sum_s"] / d16["sum_s"], 2) if d4.get("sum_s") else None,
        }
    verdict = {
        "report": rep, "sqlite16": sq16, "baseline_sqlite4": BASE_SQ,
        "chunk_equivs": CHUNK_EQUIVS,
        "tile16": {"boot": {k: boot16[k] for k in ("n", "span_s")},
                   "prefill": {k: pre16[k] for k in ("span_s", "n", "grouped_count",
                                                    "vec_count", "align_count")},
                   "decode": {k: dec16[k] for k in ("span_s", "vec_count",
                                                    "grouped_count", "align_count")}},
        "tile4": {"prefill": {k: pre4[k] for k in ("span_s", "n", "grouped_count",
                                                   "vec_count", "align_count")}},
        "grouped_names_prefill": gnames16,
        "assert": {
            "A1_grouped_names_per_format": a1,
            "A2_zero_moe_vec_in_prefill": {"vec_tile16": v16, "vec_tile4": v4, "ok": a2},
            "A3_moe_vec_present_decode": a3,
            "A4_launch_count_math_126_42": a4,
            "A5_cta_shrink_tile16_vs_tile4": {"detail": a5, "ok": a5_ok},
        },
        "per_format": fmt,
        "phases_tile16": [{k: ps[k] for k in ("n", "span_s", "grouped_count",
                                              "vec_count", "align_count")}
                          for ps in all16],
    }
    with open(os.path.join(VDIR, "v3a_probe_liveness.json"), "w") as f:
        json.dump(verdict, f, indent=1, ensure_ascii=False)
    print(json.dumps(verdict["assert"], indent=1))
    print("\nper-format (prefill phase, grouped kernels):")
    print(json.dumps(fmt, indent=1))
    ok = a1 and a2 and a3 and a4 and a5_ok
    print("\nVERDICT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
