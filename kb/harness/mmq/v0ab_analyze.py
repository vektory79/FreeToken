#!/usr/bin/env python3
"""Parse v0 A/B stage outputs (.tasks/ft-gguf-serve-tuning results/logs) into metrics.

Usage: python3 v0ab_analyze.py v0_before ab_v0_before [v0_after ab_v0_after ...]
Prints one JSON blob per stage plus a final comparison block.
"""
import json
import re
import statistics
import sys

DIR = "/media/ai/src/FreeToken/.tasks/ft-gguf-serve-tuning"
OUT = "/media/ai/src/FreeToken/.tasks/mmq-prefill-kernel"


def parse_stage(name, logname):
    # measure.py prints exactly one JSON object; take from first "{" of the
    # last block - use the file's last '{' at line start heuristic instead:
    txt = open("%s/v0ab_%s.out" % (OUT, name), encoding="utf-8", errors="replace").read()
    start = txt.index("{")
    # the JSON always opens with '{\n "name"' (measure.py json.dumps indent=1)
    i = txt.index('{\n "name"')
    dec = json.JSONDecoder()
    js, _ = dec.raw_decode(txt[i:])
    p = js.get("prefill") or {}
    d = js.get("decode") or {}
    thr_lines = p.get("throughput_lines") or []
    vals = []
    for l in thr_lines:
        m2 = re.search(r"input throughput \(token/s\):\s*([0-9]+\.?[0-9]*)", l)
        if m2:
            vals.append(float(m2.group(1)))
    # chunk structure for the 65,585-token prompt: 8 full 8128 chunks + 561 tail
    # + (decode request line is NOT in this list - captured per-request)
    full = vals[:8] if len(vals) >= 8 else vals
    tail = vals[8:9]
    # exclude chunk 1 (triton warmup) and the last full chunk (5.2-5.4 s
    # tail-report artifact, 1500-1600 tok/s class) - median of the rest
    steady = full[1:-1] if len(full) >= 3 else full
    med = statistics.median(steady) if steady else None
    out = {
        "name": name,
        "ready": js.get("ready"),
        "ready_s": js.get("ready_s"),
        "gpu_after_boot_mib": js.get("gpu_after_boot_mib"),
        "peak_vram_mib": p.get("peak_vram_mib"),
        "prefill_error": p.get("error"),
        "prompt_tokens": p.get("prompt_tokens"),
        "completion_tokens": p.get("completion_tokens"),
        "cached_token": p.get("cached_token"),
        "throughput_vals": vals,
        "full_chunks": full,
        "tail_chunk": tail,
        "steady_full_excl_first_last": steady,
        "prefill_median_tok_s": round(med, 2) if med else None,
        "decode": {
            "steady_tok_s": d.get("steady_tok_s"),
            "p50_tok_s": d.get("p50_tok_s"),
            "avg_tok_s": d.get("avg_tok_s"),
            "min_inst_tok_s": d.get("min_inst_tok_s"),
            "completion_tokens": d.get("completion_tokens"),
            "new_token": d.get("new_token"),
            "cached_token": d.get("cached_token"),
            "wall_s": d.get("wall_s"),
            "error": d.get("error"),
        },
        "teardown_leftover": (js.get("teardown") or {}).get("leftover"),
    }
    # liveness + JIT evidence from the raw server log
    logp = "%s/logs/%s.log" % (DIR, logname)
    try:
        lt = open(logp, encoding="utf-8", errors="replace").read()
    except Exception:
        lt = ""
    out["liveness_sorted_active"] = "expert-sorted pair order active" in lt
    jit = [l.strip()[-160:] for l in lt.splitlines()
           if re.search(r"clang\+\+|ninja|Building extension|extension_module|nodenamed", l)]
    out["jit_lines"] = jit[:6]
    return out


def main():
    stages = [parse_stage(sys.argv[i], sys.argv[i + 1]) for i in range(1, len(sys.argv), 2)]
    print(json.dumps(stages, indent=1, ensure_ascii=False))
    if len(stages) == 2:
        b, a = stages
        cmp_ = {"prefill_median_delta_pct": None, "decode_delta_pct": None, "boot_delta_s": None}
        if b["prefill_median_tok_s"] and a["prefill_median_tok_s"]:
            cmp_["prefill_median_delta_pct"] = round(
                100.0 * (a["prefill_median_tok_s"] - b["prefill_median_tok_s"]) / b["prefill_median_tok_s"], 1)
        if b["decode"]["steady_tok_s"] and a["decode"]["steady_tok_s"]:
            cmp_["decode_delta_pct"] = round(
                100.0 * (a["decode"]["steady_tok_s"] - b["decode"]["steady_tok_s"]) / b["decode"]["steady_tok_s"], 1)
        if b["ready_s"] is not None and a["ready_s"] is not None:
            cmp_["boot_delta_s"] = round(a["ready_s"] - b["ready_s"], 1)
        print("=== COMPARISON ===")
        print(json.dumps(cmp_, indent=1))


if __name__ == "__main__":
    main()
