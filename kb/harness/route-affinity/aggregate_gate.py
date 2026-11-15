#!/usr/bin/env python3
"""Aggregate route-affinity gate arms into the Ф1/Ф3 table (TASK.md).

Groups arm jsons by affinity value; per value reports the number of boots,
medians with min-max spread for speed/p50/hit/C, NIAH and verifiable-battery
pass rates, and the speed-threshold check against the base value:
  speed pass  : median tok/s >= base median * 1.10
Winner = max median tok/s among values that pass quality (judge shares are
recorded separately by score_judge.py and merged here when present).

Usage: aggregate_gate.py TAG [TAG...] [--out gate_table.md]
"""
import argparse
import json
import os
import statistics

DIR = os.path.join("/media/ai/src/FreeToken", ".tasks", "route-affinity")
JDIR = os.path.join(DIR, "judge")


def load(tag):
    return json.load(open(os.path.join(DIR, "arm_%s.json" % tag)))


def med(vals):
    return round(statistics.median(vals), 2) if vals else None


def spread(vals):
    if not vals:
        return "-"
    lo, hi = min(vals), max(vals)
    m = statistics.median(vals)
    return "%.0f%%" % (100 * (hi - lo) / m) if m else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    arms = [load(t) for t in a.tags]
    arms = [x for x in arms if x.get("ready") and not x.get("smoke")]
    groups = {}
    for arm in arms:
        groups.setdefault(round(arm["affinity"], 3), []).append(arm)

    base_med = None
    scored = {}
    for f in sorted(os.listdir(JDIR)):
        if f.startswith("scored_") and f.endswith(".json"):
            d = json.load(open(os.path.join(JDIR, f)))
            scored[d["affinity"] if "affinity" in d else f[len("scored_"):-len(".json")]] = d

    rows = []
    for aff in sorted(groups):
        gs = groups[aff]
        tok = [x["decode"].get("steady_tok_s") for x in gs if x["decode"].get("steady_tok_s")]
        p50 = [x["decode"].get("p50_step_ms") for x in gs if x["decode"].get("p50_step_ms")]
        hit = [x.get("hit_mean_pct") for x in gs if x.get("hit_mean_pct") is not None]
        cest = [x.get("c_estimate_pct") for x in gs if x.get("c_estimate_pct") is not None]
        niah_p = sum(x.get("niah_pass", 0) for x in gs)
        niah_t = sum(x.get("niah_total", 0) for x in gs)
        b24p = sum(x.get("battery24_pass", 0) for x in gs)
        b24c = sum(x.get("battery24_checked", 0) for x in gs)
        arith = med([x.get("battery_arith_pass") for x in gs]) if gs else None
        row = {"affinity": aff, "boots": len(gs), "tok_med": med(tok), "tok_spread": spread(tok),
               "p50_med": med(p50), "hit_med": med(hit), "c_med": med(cest),
               "niah": "%d/%d" % (niah_p, niah_t) if niah_t else "-",
               "b24_checked": "%d/%d" % (b24p, b24c) if b24c else "-",
               "arith_med": arith, "tags": [x["tag"] for x in gs]}
        if aff == 0:
            base_med = row["tok_med"]
        elif base_med:
            row["speed_vs_base"] = round(100 * (row["tok_med"] / base_med - 1), 1) if row["tok_med"] else None
        if scored.get(aff) or scored.get(str(aff)):
            s = scored.get(aff) or scored[str(aff)]
            row["quality_nw"] = s.get("share_not_worse_pct")
            row["quality_strict"] = s.get("share_strict_pct")
        rows.append(row)

    lines = ["| value | boots | median tok/s | spread | p50 ms | hit % | C % | NIAH | b24 checked | arith med | speed vs base | quality not-worse % | strict % |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["affinity"], r["boots"], r["tok_med"], r["tok_spread"], r["p50_med"],
            r["hit_med"], r["c_med"], r["niah"], r["b24_checked"], r["arith_med"],
            (("+%s%%" % r["speed_vs_base"]) if r.get("speed_vs_base") is not None else "base"),
            r.get("quality_nw", "-"), r.get("quality_strict", "-")))
    lines.append("")
    lines.append("arms: %s" % json.dumps({str(r["affinity"]): r["tags"] for r in rows}, ensure_ascii=False))
    text = "\n".join(lines)
    print(text)
    if a.out:
        open(a.out, "w").write(text + "\n")


if __name__ == "__main__":
    main()
