#!/usr/bin/env python3
"""Score blind judge verdicts for the route-affinity gate (TASK.md Ф2/Ф3).

Reads judge/sheet_<L>.json + judge/key_<L>.json + judge/verdict_<L>.json
({"<pair id>": "a"|"b"|"tie"}), reveals sides and prints per-group outcomes:
base_better / tie / x_better, plus both shares:
  share_not_worse = (tie + x_better) / N   - gate threshold >= 85%
  share_strict    = x_better / N           - ties are credited to base (conservative)

With --recheck sheet_<L>_recheck.json + verdict_<L>_recheck.json: computes the
self-consistency agreement between the original and swapped-side re-judge.

Usage:
  score_judge.py --label X005 [--recheck]
"""
import argparse
import json
import os
import sys

DIR = os.path.join("/media/ai/src/FreeToken", ".tasks", "route-affinity", "judge")


def jload(name):
    path = os.path.join(DIR, name)
    if not os.path.exists(path):
        print("missing %s" % path, file=sys.stderr)
        sys.exit(1)
    return json.load(open(path))


def outcomes(verdict, key):
    res = {"base_better": [], "tie": [], "x_better": []}
    for pid, v in verdict.items():
        if pid not in key:
            continue
        if v == "tie":
            res["tie"].append(pid)
        elif v == "a":
            res["x_better" if key[pid] == "x" else "base_better"].append(pid)
        elif v == "b":
            res["x_better" if key[pid] == "x" else "base_better"].append(pid)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--recheck", action="store_true")
    a = ap.parse_args()

    key = jload("key_%s.json" % a.label)["side_base"]
    verdict = jload("verdict_%s.json" % a.label)
    res = outcomes(verdict, key)
    n = len(key)
    nb, nt, nx = len(res["base_better"]), len(res["tie"]), len(res["x_better"])
    share_not_worse = 100.0 * (nt + nx) / n if n else 0.0
    share_strict = 100.0 * nx / n if n else 0.0

    print("label=%s N=%d base_better=%d tie=%d x_better=%d" % (a.label, n, nb, nt, nx))
    print("share_not_worse=%.1f%% (gate: >= 85%%)  share_strict=%.1f%% (tie=base)" % (share_not_worse, share_strict))
    by_group = {}
    sheet = {s["id"]: s.get("group") for s in jload("sheet_%s.json" % a.label)}
    for pid, g in sheet.items():
        by_group.setdefault(g, {"n": 0, "x": 0, "t": 0, "b": 0})
        by_group[g]["n"] += 1
    for pid in res["base_better"]:
        by_group[sheet[pid]]["b"] += 1
    for pid in res["tie"]:
        by_group[sheet[pid]]["t"] += 1
    for pid in res["x_better"]:
        by_group[sheet[pid]]["x"] += 1
    print("%-12s %4s %6s %5s %5s %9s" % ("group", "N", "x_wins", "tie", "base", "not_worse%"))
    for g in sorted(by_group):
        d = by_group[g]
        nw = 100.0 * (d["x"] + d["t"]) / d["n"] if d["n"] else 0
        print("%-12s %4d %6d %5d %5d %8.0f%%" % (g, d["n"], d["x"], d["t"], d["b"], nw))
    json.dump({"label": a.label, "n": n, "base_better": res["base_better"],
               "tie": res["tie"], "x_better": res["x_better"],
               "share_not_worse_pct": round(share_not_worse, 1),
               "share_strict_pct": round(share_strict, 1), "by_group": by_group},
              open(os.path.join(DIR, "scored_%s.json" % a.label), "w"), ensure_ascii=False, indent=1)

    if a.recheck:
        v2 = jload("verdict_%s_recheck.json" % a.label)
        res2 = outcomes(v2, key)
        agree, total, flips = 0, 0, []
        for pid, v in verdict.items():
            if pid not in v2:
                continue
            total += 1
            o1 = "tie" if v == "tie" else ("x" if key[pid] == v else "base")
            w = v2[pid]
            o2 = "tie" if w == "tie" else ("x" if key[pid] == w else "base")
            if o1 == o2:
                agree += 1
            else:
                flips.append("%s: %s -> %s" % (pid, o1, o2))
        print("recheck agreement: %d/%d = %.0f%%" % (agree, total, 100.0 * agree / total if total else 0))
        for f in flips:
            print("  flip: %s" % f)


if __name__ == "__main__":
    main()
