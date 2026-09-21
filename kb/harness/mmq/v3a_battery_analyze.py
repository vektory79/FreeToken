#!/usr/bin/env python3
"""v3a battery divergence: base (tile 4, no env) vs win (FREETOKEN_GGUF_MOE_MTILE=32).
Same pair_stats/DIGIT methodology as quality/analyze_divergence.py (v2), pointed
at the v3a quality dirs. Writes v3a_divergence.json; prints the markdown table."""
import json
import os
import re

VDIR = "/media/ai/src/FreeToken/.tasks/mmq-v3-stationary-moe"
STAGES = {"base": os.path.join(VDIR, "quality", "base"),
          "win": os.path.join(VDIR, "quality", "win")}
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
    return {"div": i, "len_base": len(x), "len_win": len(y),
            "delta": len(x) - len(y), "identical": i >= min(len(x), len(y)) and len(x) == len(y)}


def main():
    data = {s: load(s) for s in STAGES}
    res = {"primary": [], "digit": {}}
    for i in range(24):
        a, b = data["base"][i], data["win"][i]
        st = pair_stats(a, b)
        st.update({"idx": i, "kind": a["kind"],
                   "finish_base": a.get("finish_reason"), "finish_win": b.get("finish_reason")})
        if i in DIGIT:
            st["digit_base"] = bool(DIGIT[i].search(combined(a)))
            st["digit_win"] = bool(DIGIT[i].search(combined(b)))
            res["digit"][i] = [st["digit_base"], st["digit_win"]]
        res["primary"].append(st)
    with open(os.path.join(VDIR, "v3a_divergence.json"), "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    ident = sum(1 for r in res["primary"] if r["identical"])
    divs = sorted(r["div"] for r in res["primary"])
    print("identical: %d/24 | div min/median/max: %d / %d / %d" %
          (ident, divs[0], divs[len(divs) // 2], divs[-1]))
    print("| idx | kind | div@char | len_base | len_win | delta | identical | digit base/win |")
    print("|-----|------|---------|----------|---------|-------|-----------|----------------|")
    for r in res["primary"]:
        dg = ("%s/%s" % (r["digit_base"], r["digit_win"])) if "digit_base" in r else "-"
        print("| %d | %s | %d | %d | %d | %+d | %s | %s |" %
              (r["idx"], r["kind"], r["div"], r["len_base"], r["len_win"],
               r["delta"], r["identical"], dg))
    flips = [r["idx"] for r in res["primary"] if r["div"] < 5 and not r["identical"]]
    print("task-flip suspects (div<5, not identical):", flips or "none")


if __name__ == "__main__":
    main()
