#!/usr/bin/env python3
"""Assemble REPORT.md from campaign result JSONs and server logs."""
import glob
import json
import os
import re

DIR = "/media/ai/src/FreeToken/.tasks/ft-gguf-serve-tuning"


def load(name):
    p = os.path.join(DIR, "results_%s.json" % name)
    if not os.path.exists(p):
        return {}
    try:
        return json.load(open(p))
    except Exception:
        return {}


def grep1(logname, pat):
    p = os.path.join(DIR, "logs", logname)
    try:
        txt = open(p, encoding="utf-8", errors="replace").read()
    except Exception:
        return ""
    hits = [l.strip() for l in txt.splitlines() if re.search(pat, l)]
    return hits[-1] if hits else ""


def slots_of(logname):
    hits = []
    p = os.path.join(DIR, "logs", logname)
    try:
        txt = open(p, encoding="utf-8", errors="replace").read()
    except Exception:
        return ""
    for l in txt.splitlines():
        m = re.search(r"got (\d+) slots", l)
        if m:
            hits.append(m.group(1))
    return "/".join(hits)


def cell(x):
    return str(x) if x not in (None, "") else "-"


def row(name, logname, extra_desc):
    r = load(name)
    p = r.get("prefill") or {}
    d = r.get("decode") or {}
    ts = r.get("prefill_thr_summary") or {}
    free = grep1(logname, r"Free memory after initialization")
    ready = grep1(logname, r"ready in|Ready in")
    notes = []
    if p.get("error"):
        notes.append("prefill err: %s" % p["error"])
    if p.get("watch_in_req"):
        notes.append("WATCHDOG pattern in prefill")
    if d.get("error"):
        notes.append("decode err: %s" % d["error"])
    if not r.get("ready"):
        notes.append("boot: %s" % r.get("why"))
    return "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
        extra_desc,
        cell(r.get("ready_s")),
        cell(slots_of(logname)),
        cell(free.split(":")[-1].strip() if free else ""),
        cell(p.get("new_token")),
        cell(ts.get("chunks")),
        cell(ts.get("first")),
        cell(ts.get("steady_mean")),
        cell(d.get("steady_tok_s")),
        cell(d.get("min_inst_tok_s")),
        "; ".join(notes) if notes else "ok",
    )


HDR = ("| config | ready_s | slots g1/g2/g3 | free-after-init | prefill tok (prompt) | chunks | chunk1 tok/s | "
       "prefill steady tok/s | decode steady tok/s | decode min inst | notes |\n"
       "|---|---|---|---|---|---|---|---|---|---|---|\n")

CONFIGS = [
    ("run1_base", "base.log", "BASE (user cmd)"),
    ("base_mr1", "base_mr1.log", "BASE + --max-running-requests 1"),
    ("p_6144", "p_6144.log", "mr1 + chunk 6144"),
    ("p_8191", "p_8191.log", "mr1 + chunk 8191"),
    ("p_8191_r085", "p_8191_r085.log", "mr1 + chunk 8191 + ratio 0.85"),
    ("win_p_8191", "win_p_8191.log", "winner 8191 full (decode check)"),
    ("win_p_8191_r085", "win_p_8191_r085.log", "winner 8191/0.85 full (decode check)"),
    ("win_p_6144", "win_p_6144.log", "winner 6144 full (decode check)"),
    ("load_base", "load_base.log", "load A/B: plain boot"),
    ("load_alloc", "load_alloc.log", "load A/B: FREETOKEN_BANK_CUDA_ALLOC=1"),
]


def boot_rows():
    out = []
    for name, desc in [("b_mr1_r087", "boot: mr1 + ratio 0.87"),
                       ("b_mr1_r085", "boot: mr1 + ratio 0.85"),
                       ("b_mr2", "boot: mr2 (ratio 0.89)")]:
        logname = {"b_mr1_r087": "b_mr1_r087.log", "b_mr1_r085": "b_mr1_r085.log", "b_mr2": "b_mr2.log"}[name]
        r = load(name)
        free = grep1(logname, r"Free memory after initialization")
        notes = []
        if not r.get("ready"):
            notes.append("boot FAILED: %s" % (r.get("why") or ""))
        if "cannot fund" in (grep1(logname, r"cannot fund|slot floor") or ""):
            notes.append("slot-floor gate reject")
        out.append("| %s | %s | %s | %s | - | - | - | - | - | - | %s |" % (
            desc, cell(r.get("ready_s")), cell(slots_of(logname)),
            cell(free.split(":")[-1].strip() if free else ""), "; ".join(notes) if notes else "ok"))
    return out


def ladder_rows():
    r = load("ladder")
    out = []
    for s in r.get("ladder") or []:
        out.append("| step %s | depth %s | wall %s s | new %s | cached %s | thr %s | peak VRAM %s MiB | err %s |" % (
            s.get("step"), s.get("cum_tokens"), s.get("wall_s"), s.get("new_token"),
            s.get("cached_token"), (s.get("thr_summary") or {}).get("steady_mean"),
            s.get("peak_vram_mib"), (s.get("error") or "-")[:60]))
    return out


def main():
    parts = ["# ft serve GGUF GLM-5.3-Flash: A/B campaign (RTX 5090, 2026-09-16)\n"]
    parts.append("Baseline free before load (this session, llama-swap OFF): %s\n" %
                 grep1("base.log", r"Free memory before loading model"))
    parts.append("Production delta note: llama-swap idle wrapper historically holds ~1.27 GiB VRAM -> "
                 "subtract from free-after-init when comparing with production boots.\n")
    parts.append("## Boot+prefill+decode matrix\n")
    parts.append(HDR)
    for name, logname, desc in CONFIGS:
        if load(name) or os.path.exists(os.path.join(DIR, "logs", logname)):
            parts.append(row(name, logname, desc) + "\n")
    parts.append("\n## Boot-only plan inspections\n")
    parts.append(HDR)
    parts.extend(r + "\n" for r in boot_rows())
    parts.append("\n## Deep-fill ladder (winner config)\n")
    lr = ladder_rows()
    parts.extend(r + "\n" for r in lr) if lr else parts.append("(no ladder data)\n")
    with open(os.path.join(DIR, "REPORT.md"), "w") as f:
        f.write("\n".join(parts))
    print(open(os.path.join(DIR, "REPORT.md")).read())


if __name__ == "__main__":
    main()
