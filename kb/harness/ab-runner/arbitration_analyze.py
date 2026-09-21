#!/usr/bin/env python3
"""Analyzer for the arbitration battery (dense-q80-gemm).

Reads quality-arbitration/<boot>/{p00..p23}.json + <boot>_boot.json for the 4
plain boots (t4a t64a t4b t64b) and writes:
  - .tasks/dense-q80-gemm/arbitration-results.json (full data)
  - .tasks/dense-q80-gemm/arbitration-battery.md   (human report)

Verdict logic: same-config pairs (t4a~t4b, t64a~t64b) quantify boot-to-boot
nondeterminism; cross-config pairs (t4a~t64a, t4b~t64b, t4a~t64b, t64a~t4b)
re-test the tile effect at 24-prompt granularity. If cross flips ~ same flips,
the config adds nothing beyond boot noise (tile-invariance holds e2e);
materially more cross flips = possible tile effect (BLOCKER-class for the
default-flip decision). Honest either way.
"""
import json
import os
import statistics

TASKDIR = "/media/ai/src/FreeToken/.tasks/dense-q80-gemm"
QDIR = os.path.join(TASKDIR, "quality-arbitration")
BOOTS = ["t4a", "t64a", "t4b", "t64b"]
TILE = {"t4a": "4", "t64a": "64", "t4b": "4", "t64b": "64"}
SAME_PAIRS = [("t4a", "t4b"), ("t64a", "t64b")]
CROSS_PAIRS = [("t4a", "t64a"), ("t4b", "t64b"), ("t4a", "t64b"), ("t64a", "t4b")]
PAIR_LABEL = {("t4a", "t4b"): "s1", ("t64a", "t64b"): "s2",
              ("t4a", "t64a"): "x1", ("t4b", "t64b"): "x2",
              ("t4a", "t64b"): "x3", ("t64a", "t4b"): "x4"}
PREFILL_ANCHOR = {"4": 563.90, "64": 792.08}


def lcp(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def lcs_len(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[len(a) - 1 - i] == b[len(b) - 1 - i]:
        i += 1
    return i


def load_boot(name):
    prompts = []
    for i in range(24):
        path = os.path.join(QDIR, name, "p%02d.json" % i)
        if not os.path.exists(path):
            prompts.append(None)
            continue
        art = json.load(open(path))
        prompts.append(art if art.get("sha256") else None)
    return prompts


def combined(art):
    return (art.get("reasoning") or "") + "\x00" + (art.get("content") or "")


def classify_pair(a, b):
    """Whole-answer vs trailing classification for one divergent prompt pair."""
    ta, tb = combined(a), combined(b)
    la, lb = len(ta), len(tb)
    p, s = lcp(ta, tb), lcs_len(ta, tb)
    if p == 0:
        cls = "whole-answer"
    elif p >= min(la, lb):
        cls = "trailing-only"  # one answer is a prefix of the other
    elif s >= min(la, lb) - p:
        cls = "prefix-only"    # divergence confined to the head after a shared tail
    elif s > 0:
        cls = "mid-answer"     # shared head AND shared tail, middle differs
    else:
        cls = "suffix-flip"    # shared head, tail fully differs
    first_field = "reasoning" if (p <= (ta.find("\x00") if "\x00" in ta else la)) else "content-only"
    return {"class": cls, "lcp_chars": p, "lcs_chars": s,
            "len_a": la, "len_b": lb, "first_field": first_field,
            "reasoning_lcp": lcp(a.get("reasoning") or "", b.get("reasoning") or "")}


def main():
    boots = {}
    for n in BOOTS:
        p = load_boot(n)
        if any(x is not None for x in p):
            boots[n] = p
    if not boots:
        print("no boot artifacts found under", QDIR)
        return 1

    pairs = []
    for a, b in SAME_PAIRS + CROSS_PAIRS:
        if a not in boots or b not in boots:
            continue
        same_cfg = TILE[a] == TILE[b]
        div = []
        for i in range(24):
            pa, pb = boots[a][i], boots[b][i]
            if pa is None or pb is None:
                continue
            if pa["sha256"] != pb["sha256"]:
                div.append({"i": i, **classify_pair(pa, pb)})
        pairs.append({"pair": "%s~%s" % (a, b), "label": PAIR_LABEL[(a, b)],
                      "same_config": same_cfg, "tile_a": TILE[a], "tile_b": TILE[b],
                      "n_compared": sum(1 for i in range(24) if boots[a][i] and boots[b][i]),
                      "identical": sum(1 for i in range(24)
                                       if boots[a][i] and boots[b][i]
                                       and boots[a][i]["sha256"] == boots[b][i]["sha256"]),
                      "divergent": len(div), "divergent_prompts": div})
    sp = [p for p in pairs if p["same_config"]]
    cp = [p for p in pairs if not p["same_config"]]
    k_same = statistics.mean([p["divergent"] for p in sp]) if sp else None
    k_cross = statistics.mean([p["divergent"] for p in cp]) if cp else None
    rate = None
    if k_cross is not None and k_same is not None and k_cross > 0:
        rate = k_cross / 24.0
    if k_same is not None and k_cross is not None:
        if k_cross <= k_same:
            verdict = ("tile-invariance holds e2e: cross-config flips do not exceed "
                       "same-config (boot-noise) flips; the config adds NOTHING "
                       "beyond boot-to-boot nondeterminism")
        else:
            verdict = ("cross-config flips EXCEED same-config flips - possible tile "
                       "effect at 24-prompt granularity (BLOCKER-class for the "
                       "default-flip decision); needs the numbers below")
    else:
        verdict = "incomplete battery - verdict not computable"

    per_prompt_flips = {}
    for i in range(24):
        row = {p["label"]: (None if (i >= len(boots.get(p["pair"].split("~")[0], [])) or
                                     boots.get(p["pair"].split("~")[0], [None] * 24)[i] is None or
                                     boots.get(p["pair"].split("~")[1], [None] * 24)[i] is None)
                            else int(boots[p["pair"].split("~")[0]][i]["sha256"] !=
                                     boots[p["pair"].split("~")[1]][i]["sha256"]))
               for p in pairs}
        row["flips"] = sum(v for v in row.values() if v == 1)
        per_prompt_flips[i] = row

    boot_meta = {}
    for n in boots:
        mp = os.path.join(QDIR, "%s_boot.json" % n)
        meta = {}
        if os.path.exists(mp):
            m = json.load(open(mp))
            meta = {"tile": TILE[n], "ready_s": m["boot"].get("ready_s"),
                    "median_c2_c7": m["prefill"].get("median_c2_c7"),
                    "anchor_expected": m["prefill"].get("anchor_expected"),
                    "anchor_dev_pct": m["prefill"].get("anchor_dev_pct"),
                    "battery": m["battery"], "teardown": m["teardown"],
                    "gpu_after_boot_mib": m.get("gpu_after_boot_mib"),
                    "env_pins": m.get("env_pins")}
        boot_meta[n] = meta

    divergent_pairs = []
    for p in pairs:
        for d in p["divergent_prompts"]:
            divergent_pairs.append({"pair": p["pair"], "label": p["label"],
                                    "same_config": p["same_config"], **d})

    out = {"boots": boot_meta, "pairs": pairs,
           "k_same_avg": k_same, "k_cross_avg": k_cross,
           "nondet_rate_per_pair": rate, "verdict": verdict,
           "per_prompt_flips": per_prompt_flips,
           "divergent_pairs": divergent_pairs}
    with open(os.path.join(TASKDIR, "arbitration-results.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    # ---- markdown ----
    L = []
    L.append("# Arbitration battery: boot nondeterminism vs tile-4/64 output agreement (24 prompts)")
    L.append("")
    L.append("4 plain boots (no nsys, single instrumentation mode), strictly serialized and")
    L.append("reaped, interleaved t4a/t64a/t4b/t64b. Per boot: census -> quiet-box gate -> boot")
    L.append("(FREETOKEN_GGUF_DENSE_MTILE=<tile>, FREETOKEN_GGUF_MOE_MTILE=32, NSPLIT unset ->")
    L.append("committed default 2, winner serving flags, port 18801) -> /ready -> 65,585-token")
    L.append("fill -> 24-prompt greedy battery (p00..p23, reasoning+content verbatim + sha256)")
    L.append("-> SIGTERM/SIGKILL -> census.")
    L.append("")
    L.append("## Prefill sanity anchors (median of throughput lines c2..c7)")
    L.append("")
    L.append("| boot | tile | median c2..c7 tok/s | anchor | dev % |")
    L.append("|------|------|--------------------:|-------:|------:|")
    for n in boots:
        m = boot_meta[n]
        L.append("| %s | %s | %s | %s | %s |" % (n, TILE[n], m.get("median_c2_c7"),
                                                 m.get("anchor_expected"), m.get("anchor_dev_pct")))
    L.append("")
    L.append("## Pairwise sha matrices (24 prompts per pair)")
    L.append("")
    L.append("| pair | kind | identical | divergent | divergent prompts |")
    L.append("|------|------|----------:|----------:|-------------------|")
    for p in pairs:
        L.append("| %s (%s) | %s | %d | %d | %s |" % (
            p["pair"], p["label"], "same-config" if p["same_config"] else "cross-config",
            p["identical"], p["divergent"],
            ", ".join("p%02d" % d["i"] for d in p["divergent_prompts"]) or "-"))
    L.append("")
    L.append("## Per-prompt flip matrix (1 = sha mismatch on that pair)")
    L.append("")
    hdr = ["prompt"] + [p["label"] for p in pairs] + ["flips"]
    L.append("| " + " | ".join(hdr) + " |")
    L.append("|" + "|".join(["---"] * len(hdr)) + "|")
    for i in range(24):
        row = per_prompt_flips[i]
        vals = [str(row[p["label"]]) if row[p["label"]] is not None else "." for p in pairs]
        L.append("| p%02d | %s | %d |" % (i, " | ".join(vals), row["flips"]))
    L.append("")
    k_s_txt = "%s" % k_same if k_same is not None else "n/a"
    k_c_txt = "%s" % k_cross if k_cross is not None else "n/a"
    L.append("## Verdict")
    L.append("")
    L.append("- same-config (boot noise) flips per pair, avg: **%s** (of 24)" % k_s_txt)
    L.append("- cross-config flips per pair, avg: **%s** (of 24)" % k_c_txt)
    L.append("- quantified nondeterminism rate: %s" % (
        ("%d/24 per boot pair (%.1f%%)" % (k_cross, 100 * rate)) if rate is not None else "n/a"))
    L.append("- %s" % verdict)
    L.append("")
    L.append("Note: the sweep's 6-prompt battery used a DIFFERENT prompt set (req_mtile_*.json),")
    L.append("so its prompts-2/3/4 flips cannot be mapped by index here; the comparison is the")
    L.append("flip rate and flip-position instability.")
    L.append("")
    L.append("## Divergence classification (whole-answer vs trailing)")
    L.append("")
    if divergent_pairs:
        L.append("| pair | prompt | class | lcp | lcs | len_a | len_b | first divergence |")
        L.append("|------|--------|-------|----:|----:|------:|------:|------------------|")
        for d in divergent_pairs:
            L.append("| %s | p%02d | %s | %d | %d | %d | %d | %s |" % (
                d["pair"], d["i"], d["class"], d["lcp_chars"], d["lcs_chars"],
                d["len_a"], d["len_b"], d["first_field"]))
    else:
        L.append("none - all compared pairs were bitwise identical on all prompts")
    L.append("")
    L.append("## Census + teardown")
    L.append("")
    for n in boots:
        m = boot_meta[n]
        td = m.get("teardown") or {}
        L.append("- %s: ready %ss, gpu_after_boot %s MiB, teardown rc=%s leftovers=%s"
                 % (n, m.get("ready_s"), m.get("gpu_after_boot_mib"), td.get("rc"),
                    td.get("leftovers") or "none"))
    L.append("")
    with open(os.path.join(TASKDIR, "arbitration-battery.md"), "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
