#!/usr/bin/env python3
"""Build blind pairwise judge sheets for the route-affinity gate (TASK.md Ф2).

Takes a base arm json and a candidate arm json, walks every judged text
(6 arith + 4 prose + battery24 + decode-768 + NIAH + multi-turn transcript),
shuffles A/B per pair with a deterministic seed and writes:

  judge/sheet_<label>.json  - the pairs (no labels inside);
  judge/key_<label>.json    - which side is which tag (revealed only after verdicts).

Usage:
  make_judge_sheets.py --base aff000_b1 --x aff005_b1 --label X005 [--resample 10]

--resample N additionally emits sheet_<label>_recheck.json with the same pairs
and SWAPPED sides for the self-consistency re-judge (ids selected deterministically).
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_arm import BATTERY  # legacy battery prompts, verbatim

REPO = "/media/ai/src/FreeToken"
DIR = os.path.join(REPO, ".tasks", "route-affinity")
JDIR = os.path.join(DIR, "judge")


def load(tag):
    return json.load(open(os.path.join(DIR, "arm_%s.json" % tag)))


def multitranscript(arm):
    parts = []
    for t in arm.get("multiturn", []):
        parts.append("%s\nQ: %s\nA: %s" % (t["name"], t["prompt"], t["content"]))
    return "\n\n".join(parts)


def collect(arm):
    items = []
    for b in arm.get("battery", []):
        items.append({"id": b["name"], "group": "arith" if b["name"].startswith("arith") else "prose",
                      "prompt": next((p[1] for p in BATTERY if p[0] == b["name"]), ""),
                      "text": b["content"], "expected": b.get("expected"), "ok": b.get("ok")})
    for b in arm.get("battery24", []):
        items.append({"id": b["name"], "group": "b24_" + (b.get("kind") or "?"),
                      "prompt": b["prompt"], "text": b["content"],
                      "expected": b.get("expected"), "ok": b.get("ok")})
    d = arm.get("decode", {}).get("text")
    if d:
        items.append({"id": "decode768", "group": "decode", "prompt": "(continuation of the 64k filler prefix, 768 greedy tokens)", "text": d})
    for n in arm.get("niah", []):
        items.append({"id": n["name"], "group": "niah",
                      "prompt": "NIAH key %s (expected %s)" % (n["key"], n["expected"]),
                      "text": n["content"], "expected": n.get("expected"), "ok": n.get("ok")})
    items.append({"id": "multiturn5", "group": "multiturn", "prompt": "(5-turn chained arithmetic probe)", "text": multitranscript(arm)})
    return {i["id"]: i for i in items}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--x", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--resample", type=int, default=10)
    a = ap.parse_args()
    os.makedirs(JDIR, exist_ok=True)

    base, cand = load(a.base), load(a.x)
    bi, ci = collect(base), collect(cand)
    ids = [i for i in bi if i in ci]

    sheet, key = [], {}
    for pid in ids:
        seed = hash((a.base, a.x, pid)) & 0xFFFFFFFF
        rng = random.Random(seed)
        a_first = rng.random() < 0.5
        ta, tb = (bi[pid]["text"], ci[pid]["text"]) if a_first else (ci[pid]["text"], bi[pid]["text"])
        key[pid] = "base" if a_first else "x"
        sheet.append({"id": pid, "group": bi[pid]["group"], "prompt": bi[pid].get("prompt", ""),
                      "expected": bi[pid].get("expected"),
                      "text_a": ta, "text_b": tb})

    sheet_path = os.path.join(JDIR, "sheet_%s.json" % a.label)
    key_path = os.path.join(JDIR, "key_%s.json" % a.label)
    json.dump(sheet, open(sheet_path, "w"), ensure_ascii=False, indent=1)
    json.dump({"base": a.base, "x": a.x, "side_base": key}, open(key_path, "w"), ensure_ascii=False, indent=1)
    print("sheet:", sheet_path, len(sheet), "pairs")

    if a.resample > 0:
        rng = random.Random(hash((a.base, a.x, "resample")) & 0xFFFFFFFF)
        picks = sorted(rng.sample(ids, max(1, len(ids) * a.resample // 100)))
        recheck = []
        for pid in picks:
            p = next(s for s in sheet if s["id"] == pid)
            recheck.append({"id": pid, "group": p["group"], "prompt": p["prompt"],
                            "expected": p.get("expected"),
                            "text_a": p["text_b"], "text_b": p["text_a"]})
        rp = os.path.join(JDIR, "sheet_%s_recheck.json" % a.label)
        json.dump(recheck, open(rp, "w"), ensure_ascii=False, indent=1)
        print("recheck:", rp, len(recheck), "pairs (sides swapped)")


if __name__ == "__main__":
    main()
