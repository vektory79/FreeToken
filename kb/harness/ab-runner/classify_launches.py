#!/usr/bin/env python3
"""Classify every mul_mat_q8_0 launch in the window by op class.

(128,2032) is split by per-launch duration buckets; each class gets count,
total, median per-launch ms; motif (per-block) counts are printed to compare
with the a-priori census (34 KDA + 12 MLA + 43 MoE-shexp + 3 dense-FFN).
"""
import json
import os
import sqlite3
import statistics
from collections import defaultdict

TASKDIR = os.path.dirname(os.path.abspath(__file__))
con = sqlite3.connect(os.path.join(TASKDIR, "denseq80.sqlite"))
rows = con.execute("""
    SELECT k.start, k.end, k.gridX, k.gridY
    FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName = s.id
    WHERE s.value = 'mul_mat_q8_0' ORDER BY k.start
""").fetchall()

T = json.load(open(os.path.join(TASKDIR, "step0-split.json")))["T_steady_median_s"]
W = json.load(open(os.path.join(TASKDIR, "step0-split.json")))["window_s"]
chunks_eq = W / T
print("chunks_eq = %.3f" % chunks_eq)


def classify(gx, ms):
    if gx == 778:
        return "kda_in_proj"
    if gx == 256:
        return "kda_f_b/g_b"
    if gx == 512:
        return "mla_q_b"
    if gx == 16:
        return "mla_kv_a"
    if gx == 48:
        return "mla_qa_or_wqb(48)"
    if gx == 384:
        return "dense_ffn_gate/up"
    if gx == 64:
        return "shexp_gate/up"
    if gx == 128:
        if ms < 10:
            return "shexp_down(128@~6ms)"
        if ms < 33:
            return "kda_o_proj(128@~25ms)"
        if ms < 44:
            return "dense_ffn_down(128@~39ms)"
        return "mla_o_proj(128@~49ms)"
    return "gridX=%d" % gx


cls = defaultdict(list)
for st, en, gx, gy in rows:
    cls[classify(gx, (en - st) / 1e6)].append((en - st) / 1e6)

print("\n%-28s %7s %9s %9s %9s" % ("class", "n", "n/chunk", "tot_s", "med_ms"))
for k in sorted(cls):
    v = cls[k]
    print("%-28s %7d %9.2f %9.3f %9.3f" % (k, len(v), len(v) / chunks_eq,
                                           sum(v) / 1e3, statistics.median(v)))

# motif census: scan for pattern starts
seq = [(gx, (en - st) / 1e6) for st, en, gx, gy in rows]
motifs = {"kda_block": 0, "mla_block": 0, "dense_block": 0}
i = 0
while i < len(seq):
    gx, ms = seq[i]
    if gx == 778:
        motifs["kda_block"] += 1
        i += 7
    elif gx == 48:
        motifs["mla_block"] += 1
        i += 7
    elif gx == 384:
        motifs["dense_block"] += 1
        i += 7
    else:
        i += 1
print("\nmotif starts:", motifs, "-> per chunk:",
      {k: round(v / chunks_eq, 2) for k, v in motifs.items()})
print("expected per chunk: kda 34, mla 12, dense 3")

# where do non-motif launches sit? print count of launches consumed by motifs
consumed = (motifs["kda_block"] + motifs["mla_block"] + motifs["dense_block"]) * 7
print("launches in motifs: %d of %d" % (consumed, len(seq)))
