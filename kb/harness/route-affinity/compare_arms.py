#!/usr/bin/env python3
"""Compare route-affinity arm jsons into one summary (CPU-only).

Usage: compare_arms.py aff000 aff005 aff020 [--out summary.md]
Baseline = the first tag; prose prompts are diffed against it verbatim.
"""
import argparse
import json
import os
import sys

DIR = os.path.dirname(os.path.abspath(__file__))


def load(tag):
    path = os.path.join(DIR, "arm_%s.json" % tag)
    if not os.path.exists(path):
        print("missing %s" % path, file=sys.stderr)
        sys.exit(1)
    return json.load(open(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    arms = [load(t) for t in a.tags]
    base = arms[0]
    base_texts = {b["name"]: b["content"] for b in base.get("battery", [])}

    lines = []
    lines.append("# Route affinity A/B summary")
    lines.append("")
    lines.append("| arm | affinity | tok/s | p50 ms/step | hit mean, % | battery pass | arith pass | prose texts changed vs base |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for arm in arms:
        dec = arm.get("decode", {})
        texts_changed = None
        if arm is not base:
            changed = [b["name"] for b in arm.get("battery", [])
                       if not b["name"].startswith("arith")
                       and b["content"] != base_texts.get(b["name"])]
            texts_changed = len(changed)
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            arm.get("tag"), arm.get("affinity"),
            dec.get("steady_tok_s"), dec.get("p50_step_ms"),
            arm.get("hit_mean_pct"),
            arm.get("battery_pass"), arm.get("battery_arith_pass"),
            texts_changed if texts_changed is not None else "base"))
    lines.append("")

    # per-prompt battery diff vs baseline
    lines.append("## Battery, per arm")
    lines.append("")
    names = [b["name"] for b in base.get("battery", [])]
    lines.append("| prompt | " + " | ".join(t for t in a.tags) + " |")
    lines.append("|---|" + "---|" * len(a.tags))
    for name in names:
        row = []
        for arm in arms:
            b = next((x for x in arm.get("battery", []) if x["name"] == name), None)
            if b is None:
                row.append("n/a")
            elif b["name"].startswith("arith"):
                row.append("PASS" if b["ok"] else "FAIL")
            else:
                same = arm is base or b["content"] == base_texts.get(name)
                row.append("same" if same else "CHANGED")
        lines.append("| %s | %s |" % (name, " | ".join(row)))
    lines.append("")

    lines.append("## Hit-rate lines (per arm)")
    lines.append("")
    for arm in arms:
        h = arm.get("hit_lines") or []
        lines.append("- %s: n=%d mean=%s values=%s" % (
            arm.get("tag"), len(h), arm.get("hit_mean_pct"), h[:20]))
    lines.append("")
    text = "\n".join(lines)
    if a.out:
        open(a.out, "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
