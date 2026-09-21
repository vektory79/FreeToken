#!/usr/bin/env python3
"""Analyzer for the N-split A/B (dense-q80-gemm replan wave 1).

Reads .tasks/dense-q80-gemm/nsplit{1,2}.sqlite (nsys sqlite exports of the
steady 8128-chunk span) + nsplit{1,2}_run.json (boot/prefill/decode/prompts)
and writes ab-nsplit-results.json. Deterministic, CPU-only, read-only sqlite.
"""
import json
import os
import re
import sqlite3
import statistics
from collections import defaultdict

TASKDIR = os.path.dirname(os.path.abspath(__file__))
M = 8128
FLOP_FUSED = 2.0 * M * 24896 * 4096
FLOP_HALF = 2.0 * M * 12448 * 4096
ANCHOR_TOK_S = 546.5      # morning knob=1 path (step-0 boot, same method)
CLASS_TOK_S = 556.43      # committed MTILE=32 class
DECODE_CLASS = (13.5, 14.7)
NET_WINDOW = (0.22, 0.26)   # s/chunk dense-term recovery target (judge on this)
GROSS_WINDOW = (0.24, 0.28)

COPYISH_SUBSTR = ("elementwise", "copy", "catarray", "memcpy")


def load_records(name):
    path = os.path.join(TASKDIR, "%s.sqlite" % name)
    con = sqlite3.connect("file:%s?mode=ro" % path, uri=True)
    k = con.execute("""
        SELECT k.start, k.end, s.value, k.gridX, k.gridY
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName = s.id
        ORDER BY k.start
    """).fetchall()
    try:
        mc = con.execute("""
            SELECT start, end, bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start
        """).fetchall()
    except sqlite3.Error:
        mc = []
    con.close()
    recs = [{"t0": s, "t1": e, "name": n, "gx": gx, "gy": gy, "kind": "K", "dur": e - s}
            for s, e, n, gx, gy in k]
    recs += [{"t0": s, "t1": e, "name": "CUDA memcpy", "gx": -1, "gy": -1,
              "kind": "M", "dur": e - s, "bytes": b} for s, e, b in mc]
    recs.sort(key=lambda r: r["t0"])
    return recs


def bucket128(ms):
    if ms < 10:
        return "shexp_down"
    if ms < 33:
        return "kda_o_proj"
    if ms < 44:
        return "dense_ffn_down"
    return "mla_o_proj"


def mul_class(name, gx, ms):
    if gx == 778:
        return "kda_in_proj_fused"
    if gx == 389:
        return "kda_in_proj_half"
    if gx == 256:
        return "kda_ssm_f_b/g_b"
    if gx == 512:
        return "mla_q_b"
    if gx == 48:
        return "mla_q_a"
    if gx == 16:
        return "mla_kv_a_mqa"
    if gx == 384:
        return "dense_ffn_gate_up"
    if gx == 64:
        return "shexp_gate_up"
    if gx == 128:
        return bucket128(ms)
    return "gridX=%d" % gx


def is_copyish(nm):
    low = nm.lower()
    return any(s in low for s in COPYISH_SUBSTR)


def chunk_times(run):
    lines = run["prefill"]["throughput_lines"]
    vals = []
    for l in lines:
        m = re.search(r"input throughput \(token/s\):\s*([0-9]+\.?[0-9]*)", l)
        if m:
            vals.append(float(m.group(1)))
    steady = vals[1:-2]  # c2..c7: exclude c1 warmup + bogus last-full chunk (T43)
    med = statistics.median(steady)
    return {"all_lines": vals, "steady_c2_c7": steady,
            "median_tok_s": round(med, 2), "chunk_s_median": round(M / med, 3)}


def analyze_boot(name, run):
    recs = load_records(name)
    ct = chunk_times(run)
    T = M / ct["median_tok_s"]
    W = (recs[-1]["t1"] - recs[0]["t0"]) / 1e9
    chunks_eq = W / T
    busy = sum(r["dur"] for r in recs) / 1e9

    mul = [r for r in recs if "mul_mat_q8_0" in r["name"]]
    cls = defaultdict(list)
    for r in mul:
        cls[mul_class(r["name"], r["gx"], r["dur"] / 1e6)].append(r["dur"] / 1e6)
    dense_cls = {k: {"n_per_chunk": round(len(v) / chunks_eq, 2),
                     "med_ms": round(statistics.median(v), 3),
                     "s_per_chunk": round(sum(v) / 1e3 / chunks_eq, 4)}
                 for k, v in cls.items()}
    dense_total_s = sum(sum(v) for v in cls.values()) / 1e3 / chunks_eq

    # grouped MoE / fetch / align by name
    def bucket_for(r):
        nm = r["name"]
        for tag in ("moe_iq3_xxs", "moe_iq4_xs", "moe_q6_K", "moe_align"):
            if tag in nm:
                return tag
        if "fast_index_copy_multi" in nm:
            return "fetch_fast_index_copy_multi"
        return None

    named = defaultdict(list)
    for r in recs:
        b = bucket_for(r)
        if b:
            named[b].append(r["dur"] / 1e6)
    moe_names = ("moe_iq3_xxs", "moe_iq4_xs", "moe_q6_K")
    moe = {t: {"n_per_chunk": round(len(named[t]) / chunks_eq, 2),
               "s_per_chunk": round(sum(named[t]) / 1e3 / chunks_eq, 4)}
           for t in moe_names if t in named}
    moe_total_s = sum(sum(named[t]) for t in moe_names if t in named) / 1e3 / chunks_eq
    fetch_s = sum(named["fetch_fast_index_copy_multi"]) / 1e3 / chunks_eq \
        if "fetch_fast_index_copy_multi" in named else 0.0
    align = {"n_per_chunk": round(len(named["moe_align"]) / chunks_eq, 2),
             "s_per_chunk": round(sum(named["moe_align"]) / 1e3 / chunks_eq, 5)} \
        if "moe_align" in named else None

    # copyish kernels: global per-name table (for the cross-boot delta)
    copyish = defaultdict(list)
    for r in recs:
        if r["kind"] == "K" and is_copyish(r["name"]):
            copyish[r["name"]].append(r["dur"] / 1e6)
    copyish_tab = {nm: {"n_per_chunk": round(len(v) / chunks_eq, 2),
                        "med_ms": round(statistics.median(v), 4),
                        "s_per_chunk": round(sum(v) / 1e3 / chunks_eq, 5)}
                   for nm, v in sorted(copyish.items(),
                                       key=lambda kv: -sum(kv[1]))[:12]}

    # kda span analysis
    kda = {}
    if "kda_in_proj_half" in cls:
        halves = [r for r in mul if r["gx"] == 389]
        pairs = [(halves[i], halves[i + 1]) for i in range(0, len(halves) - 1, 2)]
        half_ms = [r["dur"] / 1e6 for r in halves]
        pair_kernel_ms = [(a["dur"] + b["dur"]) / 1e6 for a, b in pairs]
        span_copy = defaultdict(list)
        for a, b in pairs:
            hi = b["t1"] + 2_000_000  # 2 ms grace after the second half
            for r in recs:
                if r["t0"] >= a["t0"] and r["t1"] <= hi and r["t0"] >= a["t1"] - 1000:
                    if r["kind"] == "K" and is_copyish(r["name"]) and "mul_mat" not in r["name"]:
                        span_copy[r["name"]].append(r["dur"] / 1e6)
                    elif r["kind"] == "M":
                        span_copy["CUDA memcpy"].append(r["dur"] / 1e6)
        span_copy_tab = {nm: {"per_pair": round(len(v) / len(pairs), 3),
                              "med_ms": round(statistics.median(v), 4),
                              "s_per_chunk": round(sum(v) / 1e3 / chunks_eq, 5)}
                         for nm, v in sorted(span_copy.items(), key=lambda kv: -sum(kv[1]))}
        span_copy_total_s = sum(sum(v) for v in span_copy.values()) / 1e3 / chunks_eq
        kda = {
            "mode": "split",
            "launches_per_chunk": round(len(halves) / chunks_eq, 2),
            "expected_launches_per_chunk": 68,
            "liveness": "PASS" if abs(len(halves) / chunks_eq - 68) < 1.5 else "FAIL",
            "per_half_launch_med_ms": round(statistics.median(half_ms), 3),
            "per_pair_kernel_med_ms": round(statistics.median(pair_kernel_ms), 3),
            "tflop_per_s_per_half": round(FLOP_HALF / (statistics.median(half_ms) * 1e6), 2),
            "tflop_per_s_per_call": round(FLOP_FUSED / (statistics.median(pair_kernel_ms) * 1e6), 2),
            "s_per_chunk_kernels": round(sum(half_ms) / 1e3 / chunks_eq, 4),
            "span_copy_table": span_copy_tab,
            "span_copy_total_s_per_chunk": round(span_copy_total_s, 5),
        }
    elif "kda_in_proj_fused" in cls:
        fused = [r for r in mul if r["gx"] == 778]
        ms = [r["dur"] / 1e6 for r in fused]
        near = defaultdict(list)
        for r in fused:
            hi = r["t1"] + 2_000_000
            for q in recs:
                if q["t0"] >= r["t1"] - 1000 and q["t1"] <= hi and q["kind"] == "K" \
                        and is_copyish(q["name"]) and "mul_mat" not in q["name"]:
                    near[q["name"]].append(q["dur"] / 1e6)
        near_tab = {nm: {"per_launch": round(len(v) / len(fused), 3),
                         "s_per_chunk": round(sum(v) / 1e3 / chunks_eq, 5)}
                    for nm, v in sorted(near.items(), key=lambda kv: -sum(kv[1]))}
        kda = {
            "mode": "fused",
            "launches_per_chunk": round(len(fused) / chunks_eq, 2),
            "expected_launches_per_chunk": 34,
            "liveness": "PASS" if abs(len(fused) / chunks_eq - 34) < 1.5 else "FAIL",
            "per_launch_med_ms": round(statistics.median(ms), 3),
            "tflop_per_s": round(FLOP_FUSED / (statistics.median(ms) * 1e6), 2),
            "s_per_chunk": round(sum(ms) / 1e3 / chunks_eq, 4),
            "near_copy_table": near_tab,
        }
    else:
        kda = {"mode": "missing", "liveness": "FAIL"}

    dense_eff_s = dense_total_s + kda.get("span_copy_total_s_per_chunk", 0.0)

    return {
        "T_steady_s": round(T, 3),
        "chunk_time_s": ct["chunk_s_median"],
        "window_s": round(W, 3),
        "chunks_eq": round(chunks_eq, 3),
        "busy_s_per_chunk": round(busy / chunks_eq, 3),
        "idle_s_per_chunk": round((W - busy) / chunks_eq, 3),
        "prefill": ct,
        "dense_term": {"s_per_chunk": round(dense_total_s, 4),
                       "classes": dense_cls},
        "kda_in_proj": kda,
        "grouped_moe": {"per_name": moe, "total_s_per_chunk": round(moe_total_s, 4),
                        "align": align},
        "fetch_copies_s_per_chunk": round(fetch_s, 4),
        "rest_s_per_chunk": round((busy / chunks_eq) - dense_total_s - moe_total_s
                                  - fetch_s - kda.get("span_copy_total_s_per_chunk", 0.0), 4),
        "copyish_top_table": copyish_tab,
        "dense_term_eff_s_per_chunk": round(dense_eff_s, 4),
        "decode": run.get("decode"),
        "prompts": [{k: p[k] for k in ("i", "prompt", "output_sha256", "error",
                                       "completion_tokens", "cached_token", "wall_s")}
                    for p in run.get("prompts", [])],
        "server_env_pins": run.get("server_env_pins"),
        "graph_capture_lines": run.get("graph_capture_lines"),
        "capture": run.get("capture"),
        "teardown": run.get("teardown"),
    }


def main():
    out = {"meta": {
        "single_variable": "FREETOKEN_GGUF_KDA_NSPLIT (nsplit1 unset=1 vs nsplit2=2)",
        "anchor": "MTILE=32 (FREETOKEN_GGUF_MOE_MTILE=32 both boots), port 18801",
        "fill": "65,585 tokens (8x8128 + 561), .tasks/ft-gguf-serve-tuning/filler.txt",
        "nsys_window": "fill chunks 3..7 (start after 2nd, stop after 6th thr line)",
        "validity_refs": {"morning_knob1_anchor_tok_s": ANCHOR_TOK_S,
                          "committed_class_tok_s": CLASS_TOK_S},
        "judge": {"gross_window_s": GROSS_WINDOW, "net_window_s": NET_WINDOW},
    }}
    boots = {}
    for name in ("nsplit1", "nsplit2"):
        with open(os.path.join(TASKDIR, "%s_run.json" % name), encoding="utf-8") as f:
            run = json.load(f)
        boots[name] = analyze_boot(name, run)
    b1, b2 = boots["nsplit1"], boots["nsplit2"]

    t1 = b1["prefill"]["median_tok_s"]
    t2 = b2["prefill"]["median_tok_s"]
    d1 = b1["dense_term_eff_s_per_chunk"]
    d2 = b2["dense_term_eff_s_per_chunk"]
    recovery = d1 - d2
    chunk1 = b1["prefill"]["chunk_s_median"]
    chunk2 = b2["prefill"]["chunk_s_median"]
    kda1 = b1["kda_in_proj"]
    kda2 = b2["kda_in_proj"]
    verdict = {
        "prefill_median_tok_s": {"nsplit1": t1, "nsplit2": t2,
                                 "delta_pct": round((t2 / t1 - 1.0) * 100, 2)},
        "chunk_time_s": {"nsplit1": chunk1, "nsplit2": chunk2,
                         "delta_s": round(chunk1 - chunk2, 3)},
        "validity_gate_nsplit1": {
            "tok_s": t1,
            "delta_vs_morning_anchor_pct": round((t1 / ANCHOR_TOK_S - 1.0) * 100, 2),
            "delta_vs_class_pct": round((t1 / CLASS_TOK_S - 1.0) * 100, 2),
            "band": "~-2..+3%",
            "verdict": "PASS" if -2.0 <= (t1 / ANCHOR_TOK_S - 1.0) * 100 <= 3.0
                       else "OUT_OF_BAND_recorded_internal_AB_stays_valid",
        },
        "dense_term_eff_s": {"nsplit1": d1, "nsplit2": d2,
                             "recovery_s": round(recovery, 4)},
        "kda_per_launch": {
            "nsplit1_per_launch_ms": kda1.get("per_launch_med_ms"),
            "nsplit1_per_chunk_kernels_s": kda1.get("s_per_chunk"),
            "nsplit2_per_half_ms": kda2.get("per_half_launch_med_ms"),
            "nsplit2_per_pair_kernel_ms": kda2.get("per_pair_kernel_med_ms"),
            "nsplit2_kernels_per_chunk_s": kda2.get("s_per_chunk_kernels"),
            "nsplit1_tflop_s": kda1.get("tflop_per_s"),
            "nsplit2_tflop_s_per_half": kda2.get("tflop_per_s_per_half"),
            "nsplit2_tflop_s_per_call": kda2.get("tflop_per_s_per_call"),
        },
        "liveness": {"nsplit1": kda1.get("liveness"),
                     "nsplit1_launches": kda1.get("launches_per_chunk"),
                     "nsplit2": kda2.get("liveness"),
                     "nsplit2_launches": kda2.get("launches_per_chunk")},
        "judge_on_net": {
            "recovery_s": round(recovery, 4),
            "net_window_s": NET_WINDOW,
            "gross_window_s": GROSS_WINDOW,
            "verdict": ("CONFIRMED" if NET_WINDOW[0] <= recovery <= GROSS_WINDOW[1]
                        else ("ABOVE_UPPER_BOUND" if recovery > GROSS_WINDOW[1]
                              else "BELOW_NET_WINDOW")),
        },
        "decode": {
            "nsplit1_steady_tok_s": (b1["decode"] or {}).get("steady_tok_s"),
            "nsplit2_steady_tok_s": (b2["decode"] or {}).get("steady_tok_s"),
            "class": DECODE_CLASS,
        },
        "bitwise_diff": {
            "method": "6 greedy prompts, sha256 over the raw choices JSON",
            "identical": (b1["prompts"] and b2["prompts"]
                          and all(p["output_sha256"] and p["output_sha256"] == q["output_sha256"]
                                  for p, q in zip(b1["prompts"], b2["prompts"]))),
            "per_prompt": [{"i": p["i"],
                            "equal": p["output_sha256"] == q["output_sha256"],
                            "sha_nsplit1": p["output_sha256"],
                            "sha_nsplit2": q["output_sha256"],
                            "nsplit1_error": p["error"],
                            "nsplit2_error": q["error"]}
                           for p, q in zip(b1["prompts"], b2["prompts"])],
        },
    }
    out["nsplit1"] = b1
    out["nsplit2"] = b2
    out["comparison"] = verdict
    with open(os.path.join(TASKDIR, "ab-nsplit-results.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()
