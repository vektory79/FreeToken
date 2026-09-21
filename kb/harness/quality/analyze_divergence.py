#!/usr/bin/env python3
"""Divergence analyzer for the v2 quality battery: env0 vs env1 pairwise, plus
historical context (battery-post-fix GGUF, Task-05 hybrid, Task-08 hybrid).

Metrics per prompt: combined reasoning+content string, first-divergence char
position (common-prefix length, unicode chars), length delta, task-flip suspect
flag (div < 5), digit-prompt checks (p20 80 km/h, p21 Friday, p22 6 apples,
p23 1024). Writes divergence.json; prints a markdown table with 110-char heads
for the manual on-topic read.
"""
import json
import os
import re

QDIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(QDIR))  # .tasks/mmq-prefill-kernel
STAGES = {
    "env0": os.path.join(QDIR, "env0"),
    "env1": os.path.join(QDIR, "env1"),
    "postfix": os.path.join(os.path.dirname(os.path.dirname(QDIR)), "gguf-glm5next-path-a",
                            "verification", "phase6", "battery-post-fix"),
    "t05": os.path.join(QDIR, "t05_hybrid_main_battery"),
    "t08": os.path.join(QDIR, "t08_hybrid_battery"),
}
DIGIT = {
    20: re.compile(r"80\s*(?:km/?h|kilometers per hour)", re.I),
    21: re.compile(r"friday|пятниц", re.I),
    22: re.compile(r"6 apples|six apples", re.I),
    23: re.compile(r"1024"),
}


def load(stage):
    out = {}
    for i in range(24):
        with open(os.path.join(STAGES[stage], "p%02d.json" % i)) as f:
            out[i] = json.load(f)
    return out


def combined(a):
    return (a.get("reasoning") or "") + (a.get("content") or "")


def pair_stats(a, b):
    x, y = combined(a), combined(b)
    n = min(len(x), len(y))
    i = 0
    while i < n and x[i] == y[i]:
        i += 1
    return {"div": i, "len_a": len(x), "len_b": len(y),
            "delta": len(x) - len(y), "flip_suspect": i < 5}


def head(t, n=110):
    return t[:n].replace("\n", " ")


def main():
    data = {s: load(s) for s in STAGES}
    res = {"primary": [], "context": {}, "digit": {}}

    for i in range(24):
        a, b = data["env0"][i], data["env1"][i]
        st = pair_stats(a, b)
        st.update({"idx": i, "kind": a["kind"],
                   "finish_a": a.get("finish_reason"), "finish_b": b.get("finish_reason")})
        if i in DIGIT:
            st["digit_env0"] = bool(DIGIT[i].search(combined(a)))
            st["digit_env1"] = bool(DIGIT[i].search(combined(b)))
            res["digit"][i] = [st["digit_env0"], st["digit_env1"]]
        res["primary"].append(st)

    for s2 in ("postfix", "t05", "t08"):
        rows = []
        for i in range(24):
            st = pair_stats(data["env0"][i], data[s2][i])
            st2 = pair_stats(data["env1"][i], data[s2][i])
            rows.append({"idx": i, "kind": data["env0"][i]["kind"],
                         "env0_div": st["div"], "env1_div": st2["div"]})
        res["context"][s2] = rows

    with open(os.path.join(QDIR, "divergence.json"), "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)

    divs = [r["div"] for r in res["primary"]]
    ds = sorted(divs)
    print("## env0 vs env1 (primary, %d prompts)" % len(divs))
    identical = sum(1 for r in res["primary"]
                    if r["div"] == min(r["len_a"], r["len_b"]) and r["delta"] == 0)
    print("identical: %d | div min/median/max: %d / %d / %d" %
          (identical, ds[0], ds[len(ds) // 2], ds[-1]))
    print()
    print("| idx | kind | div@char | len0 | len1 | delta | flip? | digit e0/e1 |")
    print("|-----|------|---------|------|------|-------|-------|-------------|")
    for r in res["primary"]:
        dg = ("%s/%s" % (r.get("digit_env0"), r.get("digit_env1"))) if "digit_env0" in r else "-"
        print("| %d | %s | %d | %d | %d | %+d | %s | %s |" %
              (r["idx"], r["kind"], r["div"], r["len_a"], r["len_b"], r["delta"],
               "SUSPECT" if r["flip_suspect"] else "no", dg))
    print()
    print("## heads for the manual on-topic read (first 110 chars of combined)")
    for i in range(24):
        print("--- p%02d (%s)" % (i, data["env0"][i]["kind"]))
        print("prompt : %s" % head(data["env0"][i]["prompt"], 90))
        print("env0   : %s" % head(combined(data["env0"][i])))
        print("env1   : %s" % head(combined(data["env1"][i])))
        print("postfix: %s" % head(combined(data["postfix"][i])))
    print()
    print("## context: divergence vs historical batteries")
    print("| idx | kind | env0~postfix | env1~postfix | env0~t05 | env1~t05 | env0~t08 | env1~t08 |")
    print("|-----|------|--------------|--------------|----------|----------|----------|----------|")
    by_stage = {s2: {r["idx"]: r for r in res["context"][s2]}
                for s2 in ("postfix", "t05", "t08")}
    for i in range(24):
        cells = [by_stage[s2][i][k] for s2 in ("postfix", "t05", "t08") for k in ("env0_div", "env1_div")]
        print("| %d | %s | %s |" % (i, data["env0"][i]["kind"], " | ".join(str(c) for c in cells)))
    print()
    print("context div ranges: postfix %d-%d, t05 %d-%d, t08 %d-%d" % (
        min(r["env0_div"] for r in res["context"]["postfix"]),
        max(r["env1_div"] for r in res["context"]["postfix"]),
        min(r["env0_div"] for r in res["context"]["t05"]),
        max(r["env1_div"] for r in res["context"]["t05"]),
        min(r["env0_div"] for r in res["context"]["t08"]),
        max(r["env1_div"] for r in res["context"]["t08"])))


if __name__ == "__main__":
    main()
