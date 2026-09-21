#!/usr/bin/env python3
"""Parse v2 A/B stage outputs (v2ab_<stage>.out from measure.py) into medians.

Prefill metric: median of full chunks excluding chunk 1 (triton autotune
warmup) and the last full chunk (tail-report artifact, see memory
ft-last-chunk-throughput-artifact). Full-chunk counts for the 65,585-token
filler: 4096 -> 16x4096 + 49-tok tail; 6144 -> 10x6144 + 4145; 8128 -> 8x8128
+ 561. Stage files: v2ab_{before,after}_{4096,6144,8191}.out (8191 names the
8128-chunk runs). Writes v2-ab-results.json, prints a comparison table.
"""
import json
import os
import re
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
NFULL = {"4096": 16, "6144": 10, "8128": 8}
POINTS = [("4096", "4096"), ("6144", "6144"), ("8128", "8191")]
CAMPAIGN_REF = {"4096": 262.0, "6144": 281.0, "8128": 292.30}
CAMPAIGN_DECODE_REF = [13.58, 14.72]


def full_count(stage):
    return NFULL[stage.rsplit("_", 1)[1].replace("8191", "8128")]


def parse_stage(stage):
    path = os.path.join(HERE, "v2ab_%s.out" % stage)
    txt = open(path, encoding="utf-8", errors="replace").read()
    i = txt.index('{\n "name"')
    js, _ = json.JSONDecoder().raw_decode(txt[i:])
    vals = []
    for l in (js.get("prefill") or {}).get("throughput_lines") or []:
        m = re.search(r"input throughput \(token/s\):\s*([0-9]+\.?[0-9]*)", l)
        if m:
            vals.append(float(m.group(1)))
    nfull = full_count(stage)
    full = vals[:nfull]
    tail = vals[nfull:]
    steady = full[1:-1] if len(full) >= 3 else full
    med = round(statistics.median(steady), 2) if steady else None
    d = js.get("decode") or {}
    return {
        "stage": stage,
        "ready": js.get("ready"),
        "ready_s": js.get("ready_s"),
        "boot_env": js.get("boot_env"),
        "gpu_after_boot_mib": js.get("gpu_after_boot_mib"),
        "peak_vram_mib": (js.get("prefill") or {}).get("peak_vram_mib"),
        "prefill_error": (js.get("prefill") or {}).get("error"),
        "prompt_tokens": (js.get("prefill") or {}).get("prompt_tokens"),
        "thr_vals_all": vals,
        "full_chunks": full,
        "warmup_chunk1": full[:1],
        "excluded_last_full": full[-1:],
        "tail_chunks": tail,
        "steady_full_used": steady,
        "prefill_median_tok_s": med,
        "decode": {k: d.get(k) for k in ("steady_tok_s", "p50_tok_s", "wall_s",
                                         "completion_tokens", "new_token",
                                         "cached_token", "error")},
        "teardown_leftover": (js.get("teardown") or {}).get("leftover"),
    }


def main():
    res = {"before": {}, "after": {}}
    for grp in ("before", "after"):
        for pt, suf in POINTS:
            res[grp][pt] = parse_stage("%s_%s" % (grp, suf))
    cmp_ = {"prefill": {}, "decode": {}}
    for pt, _ in POINTS:
        b = res["before"][pt]["prefill_median_tok_s"]
        a = res["after"][pt]["prefill_median_tok_s"]
        cmp_["prefill"][pt] = {
            "before": b, "after": a,
            "delta_pct": round(100.0 * (a - b) / b, 1) if (a and b) else None,
            "campaign_ref": CAMPAIGN_REF[pt],
        }
    bd = res["before"]["8128"]["decode"]
    ad = res["after"]["8128"]["decode"]
    cmp_["decode"] = {
        "before_steady": bd.get("steady_tok_s"),
        "after_steady": ad.get("steady_tok_s"),
        "delta_pct": (round(100.0 * (ad["steady_tok_s"] - bd["steady_tok_s"])
                            / bd["steady_tok_s"], 1)
                      if bd.get("steady_tok_s") and ad.get("steady_tok_s") else None),
        "before_cached": bd.get("cached_token"),
        "after_cached": ad.get("cached_token"),
        "campaign_ref_class": CAMPAIGN_DECODE_REF,
    }
    cmp_["boot_ready_s"] = {grp: {pt: res[grp][pt]["ready_s"] for pt, _ in POINTS}
                            for grp in ("before", "after")}
    out = {"note": "see v2ab_probe.json / v2-ab.md for liveness",
           "stages": res, "comparison": cmp_}
    with open(os.path.join(HERE, "v2-ab-results.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("point  before  after  delta_pct  (campaign ref)")
    for pt, _ in POINTS:
        c = cmp_["prefill"][pt]
        print("%s  %s  %s  %s  (%s)" % (pt, c["before"], c["after"],
                                        c["delta_pct"], c["campaign_ref"]))
    print("decode:", json.dumps(cmp_["decode"]))
    print("boot ready_s:", json.dumps(cmp_["boot_ready_s"]))


if __name__ == "__main__":
    main()
