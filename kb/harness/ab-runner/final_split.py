#!/usr/bin/env python3
"""Build the final step0-split.json: per-class attribution (medians x census),
regime model, and projections. Census = 34 KDA + 12 MLA + 43 MoE-shexp +
3 dense-FFN blocks; wq_b absent from the MMQ path (measured).

This is the SINGLE generator for the Step-0 deliverable (supersedes
step0_analyze_superseded.py, which is renamed/write-guarded). Deterministic,
CPU-only, opens denseq80.sqlite read-only:

    python3 .tasks/dense-q80-gemm/final_split.py

Regenerates step0-split.json AND step0-projection-table.md (the widening
table verbatim as embedded in step0-profile.md). Re-run only re-derives from
the recorded sqlite; it never touches the GPU.
"""
import json
import os
import sqlite3
import statistics
from collections import defaultdict

TASKDIR = os.path.dirname(os.path.abspath(__file__))
SQ = os.path.join(TASKDIR, "denseq80.sqlite")
con = sqlite3.connect("file:%s?mode=ro" % SQ, uri=True)
rows = con.execute("""
    SELECT k.start, k.end, k.gridX, k.gridY
    FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName = s.id
    WHERE s.value = 'mul_mat_q8_0' ORDER BY k.start
""").fetchall()
con.close()

T = 14.87  # steady median chunk time from the server log (c2..c7, T43)
W = 57.519
chunks_eq = W / T
M = 8128
BW = 1638e9
L2_BYTES = 100663296  # 96.0 MiB, torch.cuda.get_device_properties(0) query
                      # (Review-A, 2026-09-19): NVIDIA GeForce RTX 5090

# class -> (K, N, layers, gridX bucket)
CLASSES = {
    "kda_in_proj_fused": (4096, 24896, 34, 778),
    "kda_o_proj": (8192, 4096, 34, 128),
    "mla_o_proj": (16384, 4096, 12, 128),
    "shexp_gate": (4096, 2048, 43, 64),
    "shexp_up": (4096, 2048, 43, 64),
    "shexp_down": (2048, 4096, 43, 128),
    "mla_q_b": (1536, 16384, 12, 512),
    "mla_q_a": (4096, 1536, 12, 48),
    "mla_kv_a_mqa": (4096, 512, 12, 16),
    "dense_ffn_gate": (4096, 12288, 3, 384),
    "dense_ffn_up": (4096, 12288, 3, 384),
    "dense_ffn_down": (12288, 4096, 3, 128),
    "kda_ssm_f_b": (128, 8192, 34, 256),
    "kda_ssm_g_b": (128, 8192, 34, 256),
}

# measured per-launch medians by (gridX, duration bucket)
dur = defaultdict(list)
for st, en, gx, gy in rows:
    dur[gx].append((en - st) / 1e6)
med = {gx: statistics.median(v) for gx, v in dur.items()}


def bucket128(ms):
    if ms < 10:
        return "shexp_down"
    if ms < 33:
        return "kda_o_proj"
    if ms < 44:
        return "dense_ffn_down"
    return "mla_o_proj"


med_final = {}
for gx, v in dur.items():
    if gx == 128:
        for name in ("shexp_down", "kda_o_proj", "dense_ffn_down", "mla_o_proj"):
            sel = [x for x in v if bucket128(x) == name]
            med_final[name] = statistics.median(sel)
    else:
        for name, (K, N, n, g) in CLASSES.items():
            if g == gx:
                med_final[name] = med[gx]

cls_out = {}
tot_meas = tot_model = 0.0
tot_launch = 0
for name, (K, N, n, g) in CLASSES.items():
    Wb = N * ((K + 31) // 32) * 34
    mac = 2 * M * N * K
    ms_meas = med_final[name]
    naive_ms = ((M + 3) // 4) * Wb / BW * 1e3
    s_chunk = ms_meas * n / 1e3
    model_s = naive_ms * n / 1e3
    rate = mac / ms_meas / 1e9
    cls_out[name] = {
        "K": K, "N": N, "layers": n, "grid": [g, 2032],
        "launches_per_chunk": n,
        "median_ms_per_launch": round(ms_meas, 3),
        "seconds_per_chunk": round(s_chunk, 4),
        "weight_MB": round(Wb / 1e6, 2),
        # bytes / bytes: comparing decimal MB against the MiB figure mixed
        # units (the old value said 1.13); the ratio is 1.08 either way
        "W_over_L2": round(Wb / L2_BYTES, 2),
        "naive_traffic_model_s_per_chunk": round(model_s, 4),
        "measured_over_naive_model": round(s_chunk / model_s, 3),
        "tflop_per_s_measured": round(rate, 2),
    }
    tot_meas += s_chunk
    tot_model += model_s
    tot_launch += n

window_dense_s = sum(en - st for st, en, gx, gy in rows) / 1e9 / chunks_eq

# totals are computed from the classes, not hardcoded (N3): the old 117.8 TF
# constant omitted f_b/g_b/q_a/kv_a (2.8 TF).
tot_tflop = sum(2.0 * M * K * N * n for (K, N, n, g) in CLASSES.values()) / 1e12
RATE_CLUSTER_TF = 22.0  # per-MAC cluster rate at tile 4 (regime evidence)
RATE_CLUSTER = RATE_CLUSTER_TF * 1e12
floor_s = tot_tflop / RATE_CLUSTER_TF

# Projection model (F1/F4 fixed): the only software-discriminable recoverable
# share is the kda_in_proj L2 excess alone; it shrinks as (1 - 4/tile) while
# every other class is held at its measured tile-4 time. This is a FLOOR
# model, not a prediction: the payoff is UNMEASURED and tile-invariance of
# the 22 TF/s rate is untested (the M-sweep varies M at fixed tile 4 only;
# the v3a sibling family grew 13.7 -> 39.6 TF/s from tile 4 -> 32).
N_IN_PROJ = 24896
in_proj_launches = CLASSES["kda_in_proj_fused"][2]
in_proj_ms = med_final["kda_in_proj_fused"]
flop_per_launch = 2.0 * M * N_IN_PROJ * 4096
in_proj_at_rate_ms = flop_per_launch / RATE_CLUSTER * 1e3
excess_s4 = (in_proj_ms - in_proj_at_rate_ms) / 1e3 * in_proj_launches

base_tok = M / T
proj = {
    "model": ("floor model: only the kda_in_proj L2 excess recovers, "
              "shrinking as (1 - 4/tile); all other classes held at measured "
              "tile-4 times; NOT a prediction - payoff UNMEASURED, decide "
              "via the T52 A/B (22 TF/s tile-invariance untested; v3a "
              "sibling family grew 13.7 -> 39.6 TF/s from tile 4 -> 32)"),
    "rate_anchor_tf_s": 22.0,
    "excess_s4": round(excess_s4, 3),
    "tile4_this_run": {
        "dense_term_s": round(tot_meas, 3),
        "in_proj_excess_recovered_s": 0.0,
        "chunk_s": round(T, 2),
        "tok_s": round(base_tok, 1),
    },
}
for tile in (8, 16, 32, 64):
    rec = excess_s4 * (1.0 - 4.0 / tile)
    dense = tot_meas - rec
    chunk = T - rec
    tok = M / chunk
    proj["tile%d" % tile] = {
        "dense_term_s": round(dense, 3),
        "in_proj_excess_recovered_s": round(rec, 3),
        "chunk_s": round(chunk, 2),
        "tok_s": round(tok, 1),
        "e2e_vs_this_run_pct": round((tok / base_tok - 1.0) * 100, 1),
    }

# Conservative N-split recovery (F4): recovery to the same shape's own
# M-sweep cap (19.6-19.9 TF/s at minimal DRAM pressure), not to the 22 TF/s
# cluster; the upper bound is full recovery to 22 TF/s (= excess_s4). Which
# target applies is exactly what ncu (blocked this wave) would decide.
n_split = {}
for rate_tf in (19.6, 19.9):
    rec = (in_proj_ms - flop_per_launch / (rate_tf * 1e12) * 1e3) \
        / 1e3 * in_proj_launches
    n_split["at_%s_tf_s" % rate_tf] = round(rec, 2)
n_split["upper_full_22tf_s"] = round(excess_s4, 2)
n_split["note"] = ("kda_in_proj N-split (2 x 12448 rows, each < L2, "
                   "numerics-exact); conservative target = the same shape's "
                   "M-sweep cap 19.6-19.9 TF/s")

out = {
    "meta": {
        "report": "denseq80.nsys-rep",
        "sqlite": "denseq80.sqlite",
        "T_steady_median_s": T,
        "window_s": W,
        "chunks_eq": round(chunks_eq, 3),
        "boot_tok_s": round(M / T, 1),
        "validity_gate_T44": "-1.8% vs the 556.43 class (546.6 tok/s = 8128/14.87; "
                             "median of the per-chunk throughput lines 546.58), "
                             "PASS (~-2% band)",
        "MMQ_X_Q8_0": 4, "MMQ_Y_Q8_0": 32, "NWARPS": 4,
        "moe_mtile_env": 32,
        "L2_bytes": L2_BYTES,
        "L2_MiB": 96.0,
        "device": "NVIDIA GeForce RTX 5090",
        "device_props_source": "torch.cuda.get_device_properties(0) device query, "
                               "Review-A 2026-09-19: L2_cache_size=100663296 B "
                               "= 96.0 MiB, 170 SMs, 31.4 GiB",
        "sustained_BW_GB_s": 1638,
        "peak_BW_GB_s": 1792,
        "generated_by": "python3 .tasks/dense-q80-gemm/final_split.py "
                        "(deterministic; reads denseq80.sqlite read-only; "
                        "also regenerates step0-projection-table.md)",
    },
    "dense_term": {
        "kernel": "mul_mat_q8_0",
        "launches_per_chunk_census": tot_launch,
        "launches_per_chunk_window_normalized": round(len(rows) / chunks_eq, 1),
        "seconds_per_chunk_median_based": round(tot_meas, 3),
        "seconds_per_chunk_window_normalized": round(window_dense_s, 3),
        "pct_of_chunk": round(100 * tot_meas / T, 1),
    },
    "other_terms_per_chunk": {
        # constants from the step-0 nsys window sums, not re-derived here;
        # denseq80.sqlite is retained so they are re-derivable (Review-B N4)
        "_provenance": "constants from the step0 nsys window sums; "
                       "re-derivable from denseq80.sqlite (retained)",
        "grouped_moe_mtile32": {
            "moe_iq3_xxs": 2.749, "moe_iq4_xs": 1.531, "moe_q6_K": 0.090,
            "total_s": 4.37,
            # 137.5 TF recount (2 x 8128 x 8 pairs x 3 proj x 4096 x 2048 x 42
            # layers); the earlier 173 TF denominator was unanchored (N9)
            "routed_moe_tflop_per_chunk_recount": 137.5,
            "tflop_per_s_from_recount": 31.5,
        },
        "fetch_copies_fast_index_copy_multi": 3.106,
        "rest_attention_gdn_dsa_mhc_glue": 1.35,
    },
    "regime": {
        "verdict": "ALU/issue-bound at ~22 TFLOP/s per-MAC ceiling; NOT BW-bound",
        "evidence": [
            "measured/naive-model ratio 0.550-0.592 across classes (all classes "
            "0.550-0.566 except ssm f_b/g_b 0.592) = the ~22 TF/s ceiling "
            "expressed in traffic units (pure-DRAM ceiling at 1638 GB/s = "
            "12.34 TF/s for 0.1328 B/FLOP = 0.2656 B/MAC)",
            "M-sweep on in_proj shape: time linear in M, per-MAC rate constant "
            "18.1-19.9 TF/s from gridY=16 to 2032; naive-implied BW 2413-2660 "
            "GB/s > 1792 peak = impossible",
            "format A/B at identical shape/MACs: q6_K (23% fewer weight bytes, "
            "heavier decode) is 6.3% SLOWER (ratio 1.063 vs traffic-model 0.772)",
            "single exception: kda_in_proj_fused W=108.35 MB > L2 100.66 MB "
            "(96.0 MiB) -> 18.1 TF/s vs the 21.8-22.4 cluster (~ -17.5%); the "
            "deficit is a MIX, not L2 overflow alone: the same shape runs only "
            "19.6-19.9 TF/s at small M (minimal re-read pressure), so the "
            "DRAM/L2-dependent component is ~6-9% and the rest is shape-inherent",
        ],
        "ncu": "blocked by ERR_NVGPUCTRPERM (user permission); software "
               "discriminators used instead",
    },
    "widening_projection_issue_bound": proj,
    "n_split_kda_in_proj": n_split,
    "widening_projection_if_traffic_bound_REFUTED": {
        "note": "1/tile dividend on the whole 6.04 s is refuted by the format A/B + M-sweep",
        "tile32": {"dense_s": 0.75, "chunk_s": 9.58, "tok_s": 848},
    },
    "op_classes": cls_out,
    "totals": {
        "measured_s_per_chunk": round(tot_meas, 3),
        "naive_model_s_per_chunk": round(tot_model, 3),
        "measured_over_model": round(tot_meas / tot_model, 3),
        "dense_tflop_per_chunk": round(tot_tflop, 1),
        "dense_tflop_per_s_avg": round(tot_tflop / tot_meas, 1),
        "dense_floor_at_22tf_s": round(floor_s, 2),
    },
}
with open(os.path.join(TASKDIR, "step0-split.json"), "w") as f:
    json.dump(out, f, indent=1)

# md fragment for step0-profile.md (F2: one command regenerates both files)
frag = ["| dense tile | dense term (s) | chunk (s) | tok/s | vs this run "
        "(%.1f tok/s) |" % base_tok,
        "|---|---:|---:|---:|---:|",
        "| 4 (this run) | %.2f | %.2f | %.1f | - |" % (tot_meas, T, base_tok)]
for tile in (8, 16, 32, 64):
    p = proj["tile%d" % tile]
    frag.append("| %d | %.2f | %.2f | %.1f | %+.1f%% |"
                % (tile, p["dense_term_s"], p["chunk_s"], p["tok_s"],
                   p["e2e_vs_this_run_pct"]))
with open(os.path.join(TASKDIR, "step0-projection-table.md"), "w") as f:
    f.write("\n".join(frag) + "\n")

print(json.dumps({
    "measured": round(tot_meas, 3), "model": round(tot_model, 3),
    "ratio": round(tot_meas / tot_model, 3), "launches": tot_launch,
    "window_dense_s": round(window_dense_s, 3),
    "tflop": round(tot_tflop, 1), "floor_s": round(floor_s, 2),
    "avg_tflops": round(tot_tflop / tot_meas, 1),
    "excess_s4": round(excess_s4, 3), "n_split": n_split,
    "proj32": proj["tile32"],
}, indent=1))
print("\n--- step0-projection-table.md ---")
print("\n".join(frag))
