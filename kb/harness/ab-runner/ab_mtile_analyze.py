#!/usr/bin/env python3
"""Analyzer for the dense m-tile A/B sweep (dense-q80-gemm replan wave 2).

Reads .tasks/dense-q80-gemm/mtile{4,8,16,32,64}.sqlite (nsys sqlite exports of
the steady 8128-chunk span) + mtile{4,8,16,32,64}_run.json (boot/prefill/decode/
prompts) and writes ab-mtile-results.json. Deterministic, CPU-only, read-only
sqlite.

T46 liveness anchors (review-b-candidate-a F2 / B6, CORRECTED):
  - total dense MMQ launches ~356/chunk, CONSTANT across tiles;
  - kda halves stay 68/chunk (2x34, gridX=389, NSPLIT=2 committed default);
  - per-launch grid.y = ceil(M/tile) shrinks by the tile factor:
    M=8128 -> 2032/1016/508/254/127;
  - decode replays the bs=1 MMVQ graph (neither knob enters) -> expected flat
    vs the 13.5-14.7 class.
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
CLASS_TOK_S = 561.50        # NSPLIT=2 instrumented serving class (same-mode)
PLAIN_CONTEXT = (546.5, 553.8)  # plain-boot context (CUPTI ~1.2% symmetric)
DECODE_CLASS = (13.5, 14.7)
TILES = [4, 8, 16, 32, 64]
DENSE_LAUNCH_ANCHOR = 356   # total dense MMQ launches/chunk (68 kda + ~288 other)

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
    dense_n_per_chunk = len(mul) / chunks_eq
    cls = defaultdict(list)
    for r in mul:
        cls[mul_class(r["name"], r["gx"], r["dur"] / 1e6)].append(r)
    dense_cls = {}
    for k, v in cls.items():
        dense_cls[k] = {"n_per_chunk": round(len(v) / chunks_eq, 2),
                        "med_ms": round(statistics.median([r["dur"] / 1e6 for r in v]), 3),
                        "s_per_chunk": round(sum(r["dur"] for r in v) / 1e9 / chunks_eq, 4),
                        "med_gridX": int(statistics.median([r["gx"] for r in v])),
                        "med_gridY": int(statistics.median([r["gy"] for r in v])),
                        "gridY_all": sorted(set(r["gy"] for r in v))}
    dense_total_s = sum(r["dur"] for rr in cls.values() for r in rr) / 1e9 / chunks_eq

    expected_gy = (M + _tile(name) - 1) // _tile(name)
    gy_mismatch = {k: d["gridY_all"] for k, d in dense_cls.items()
                   if d["med_gridY"] != expected_gy}

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

    # kda span analysis (NSPLIT=2 halves)
    kda = {}
    if "kda_in_proj_half" in cls:
        halves = cls["kda_in_proj_half"]
        half_ms = [r["dur"] / 1e6 for r in halves]
        kda = {
            "mode": "split",
            "launches_per_chunk": round(len(halves) / chunks_eq, 2),
            "expected_launches_per_chunk": 68,
            "liveness_count": "PASS" if abs(len(halves) / chunks_eq - 68) < 1.5 else "FAIL",
            "per_half_launch_med_ms": round(statistics.median(half_ms), 3),
            "tflop_per_s_per_half": round(FLOP_HALF / (statistics.median(half_ms) * 1e9), 2),
            "s_per_chunk_kernels": round(sum(half_ms) / 1e3 / chunks_eq, 4),
            "med_gridX": int(statistics.median([r["gx"] for r in halves])),
            "med_gridY": int(statistics.median([r["gy"] for r in halves])),
        }
    elif "kda_in_proj_fused" in cls:
        fused = cls["kda_in_proj_fused"]
        ms = [r["dur"] / 1e6 for r in fused]
        kda = {
            "mode": "fused",
            "launches_per_chunk": round(len(fused) / chunks_eq, 2),
            "expected_launches_per_chunk": 34,
            "liveness_count": "PASS" if abs(len(fused) / chunks_eq - 34) < 1.5 else "FAIL",
            "per_launch_med_ms": round(statistics.median(ms), 3),
            "tflop_per_s": round(FLOP_FUSED / (statistics.median(ms) * 1e9), 2),
            "s_per_chunk": round(sum(ms) / 1e3 / chunks_eq, 4),
        }
    else:
        kda = {"mode": "missing", "liveness_count": "FAIL"}

    kda_share_s = kda.get("s_per_chunk_kernels", kda.get("s_per_chunk", 0.0))

    return {
        "T_steady_s": round(T, 3),
        "chunk_time_s": ct["chunk_s_median"],
        "window_s": round(W, 3),
        "chunks_eq": round(chunks_eq, 3),
        "busy_s_per_chunk": round(busy / chunks_eq, 3),
        "idle_s_per_chunk": round((W - busy) / chunks_eq, 3),
        "prefill": ct,
        "dense_term": {"s_per_chunk": round(dense_total_s, 4),
                       "launches_per_chunk": round(dense_n_per_chunk, 2),
                       "launch_anchor": DENSE_LAUNCH_ANCHOR,
                       "launch_liveness": "PASS" if abs(dense_n_per_chunk - DENSE_LAUNCH_ANCHOR) <= 4
                       else "CHECK",
                       "expected_gridY": expected_gy,
                       "gridY_mismatch_classes": gy_mismatch,
                       "classes": {k: {kk: vv for kk, vv in d.items() if kk != "gridY_all"}
                                   for k, d in dense_cls.items()}},
        "kda_in_proj": kda,
        "kda_share_s_per_chunk": round(kda_share_s, 4),
        "grouped_moe": {"per_name": moe, "total_s_per_chunk": round(moe_total_s, 4),
                        "align": align},
        "fetch_copies_s_per_chunk": round(fetch_s, 4),
        "rest_s_per_chunk": round(busy / chunks_eq - dense_total_s - moe_total_s - fetch_s, 4),
        "decode": run.get("decode"),
        "prompts": [{k: p[k] for k in ("i", "prompt", "output_sha256", "error",
                                       "completion_tokens", "cached_token", "wall_s")}
                    for p in run.get("prompts", [])],
        "server_env_pins": run.get("server_env_pins"),
        "graph_capture_lines": run.get("graph_capture_lines"),
        "capture": run.get("capture"),
        "teardown": run.get("teardown"),
    }


def _tile(name):
    return int(name[len("mtile"):])


def main():
    boots = {}
    for t in TILES:
        name = "mtile%d" % t
        with open(os.path.join(TASKDIR, "%s_run.json" % name), encoding="utf-8") as f:
            run = json.load(f)
        boots[name] = analyze_boot(name, run)
    base = boots["mtile4"]
    base_med = base["prefill"]["median_tok_s"]

    med = {n: b["prefill"]["median_tok_s"] for n, b in boots.items()}
    winner = max(med, key=med.get)
    sweep_table = []
    for n, b in boots.items():
        t = _tile(n)
        sweep_table.append({
            "tile": t,
            "prefill_median_tok_s": med[n],
            "delta_pct_vs_tile4": round((med[n] / base_med - 1.0) * 100, 2),
            "chunk_time_s": b["prefill"]["chunk_s_median"],
            "dense_term_s_per_chunk": b["dense_term"]["s_per_chunk"],
            "dense_launches_per_chunk": b["dense_term"]["launches_per_chunk"],
            "kda_launches_per_chunk": b["kda_in_proj"].get("launches_per_chunk"),
            "kda_per_half_ms": b["kda_in_proj"].get("per_half_launch_med_ms"),
            "expected_gridY": b["dense_term"]["expected_gridY"],
            "gridY_mismatch_classes": b["dense_term"]["gridY_mismatch_classes"],
            "decode_steady_tok_s": (b["decode"] or {}).get("steady_tok_s"),
        })

    t44 = {
        "tok_s": base_med,
        "delta_vs_class_561_50_pct": round((base_med / CLASS_TOK_S - 1.0) * 100, 2),
        "band": "~-2..+3% same-mode",
        "verdict": "PASS" if -2.0 <= (base_med / CLASS_TOK_S - 1.0) * 100 <= 3.0
                   else "OUT_OF_BAND",
        "plain_boot_context": PLAIN_CONTEXT,
    }
    wname = winner
    win = boots[wname]
    bitwise = {
        "method": "6 greedy prompts, sha256 over the raw choices JSON, SAME-mode "
                  "(all 5 boots nsys-instrumented)",
        "baseline": "mtile4",
        "winner": wname,
        "identical": bool(base["prompts"] and win["prompts"] and len(base["prompts"]) == 6
                          and all(p["output_sha256"] and p["output_sha256"] == q["output_sha256"]
                                  for p, q in zip(base["prompts"], win["prompts"]))),
        "per_prompt": [{"i": p["i"],
                        "equal": p["output_sha256"] == q["output_sha256"],
                        "sha_mtile4": p["output_sha256"],
                        "sha_winner": q["output_sha256"],
                        "mtile4_error": p["error"],
                        "winner_error": q["error"]}
                       for p, q in zip(base["prompts"], win["prompts"])],
    }
    decode_flat = {
        n: {"steady_tok_s": (b["decode"] or {}).get("steady_tok_s"),
            "cached_token": (b["decode"] or {}).get("cached_token"),
            "error": (b["decode"] or {}).get("error")}
        for n, b in boots.items()
    }

    out = {"meta": {
        "single_variable": "FREETOKEN_GGUF_DENSE_MTILE (4/8/16/32/64; mtile4 = unset)",
        "anchor": "FREETOKEN_GGUF_MOE_MTILE=32 all boots; NSPLIT committed default 2 "
                  "(knob never set); port 18801; winner flags",
        "fill": "65,585 tokens (8x8128 + 561), .tasks/ft-gguf-serve-tuning/filler.txt",
        "nsys_window": "fill chunks 3..7 (start after 2nd, stop after 6th thr line)",
        "validity_refs": {"nsplit2_instrumented_class_tok_s": CLASS_TOK_S,
                          "plain_boot_context": PLAIN_CONTEXT},
        "t46_anchor": {"dense_launches_per_chunk": DENSE_LAUNCH_ANCHOR,
                       "kda_halves_per_chunk": 68,
                       "gridY_by_tile": {t: (M + t - 1) // t for t in TILES}},
    }}
    for n, b in boots.items():
        out[n] = b
    out["comparison"] = {
        "sweep_table": sweep_table,
        "winner": wname,
        "winner_delta_pct": round((med[wname] / base_med - 1.0) * 100, 2),
        "validity_gate_mtile4": t44,
        "judge": "prefill median c2..c7 delta vs tile-4; report ANY direction honestly",
        "decode_flat_check": decode_flat,
        "bitwise_diff": bitwise,
    }
    with open(os.path.join(TASKDIR, "ab-mtile-results.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps(out["comparison"], indent=1))


if __name__ == "__main__":
    main()
