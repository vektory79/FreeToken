#!/usr/bin/env python3
"""Step 0 analysis: GEMM/copies/rest split from the nsys sqlite export + server log.

Reads the newest report*.nsys-rep in this folder, exports sqlite, computes:
  - capture window W (kernels + memcpys span)
  - MoE GEMM kernel time (kernel names matching ggml moe / mul_mat moe)
  - H2D memcpy time and bytes (copyKind groups, all kinds reported)
  - per-chunk normalization: chunks_eq = W / T, T = median steady chunk time
    derived from the server "input throughput" lines stored in step0_run.json
  - implied v0 (GEMM/2) and v2 (single-pass expert read at measured effective
    BW) ceilings
Writes step0_split.json and prints a human table.
"""
import glob
import json
import os
import re
import sqlite3
import statistics
import subprocess
import sys

TASKDIR = os.path.dirname(os.path.abspath(__file__))
NSYS = "/usr/local/bin/nsys"
CHUNK_TOKENS = 8128
EXPERT_TOTAL_GB = 131.5          # 42 layers x 3.13 GB, read once per chunk (v2 floor)
TODAY_TRAFFIC_TB = 29.7          # moe_vec pair-wise re-read traffic per chunk (model)

reps = sorted(glob.glob(os.path.join(TASKDIR, "report*.nsys-rep")), key=os.path.getmtime)
if not reps:
    print("FATAL: no report*.nsys-rep in", TASKDIR)
    sys.exit(1)
rep = reps[-1]
print("report:", rep)

sq = os.path.join(TASKDIR, "step0.sqlite")
if not os.path.exists(sq):
    r = subprocess.run([NSYS, "export", "--type", "sqlite", "--force-overwrite", "true",
                        "-o", sq, rep], capture_output=True, text=True, timeout=1800)
    print("export rc:", r.returncode, (r.stderr or "")[-400:])

con = sqlite3.connect(sq)
cur = con.cursor()

def q(sql, args=()):
    return cur.execute(sql, args).fetchall()

# window span over all GPU activity
k_min, k_max = q("SELECT MIN(start), MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL")[0]
m = q("SELECT MIN(start), MAX(end) FROM CUPTI_ACTIVITY_KIND_MEMCPY")
m_min, m_max = (m[0] if m else (None, None))
lo = min(x for x in (k_min, m_min) if x is not None)
hi = max(x for x in (k_max, m_max) if x is not None)
W_s = (hi - lo) / 1e9

# top kernels by total time
rows = q("""
    SELECT s.value, COUNT(*), SUM(k.end - k.start)
    FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName = s.id
    GROUP BY s.value ORDER BY 3 DESC LIMIT 40
""")
print("\n=== top kernels (name, instances, total_s) ===")
moe_re = re.compile(r"moe", re.I)  # MoE GEMM kernels only; dense ggml_mul_mat_a8,
                                   # quantize/gated-act stay in "rest"
moe_total_ns = 0
moe_count = 0
for name, cnt, tot in rows:
    print("%8.3f s  x%-7d  %s" % (tot / 1e9, cnt, name[:120]))
    if moe_re.search(name):
        moe_total_ns += tot
        moe_count += cnt

# memcpys by kind
mc = q("SELECT copyKind, COUNT(*), SUM(end - start), SUM(bytes), AVG(bytes) "
       "FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY copyKind ORDER BY copyKind")
print("\n=== memcpys (kind, count, total_s, GB, avg_KiB) ===")
h2d_ns = 0
for kind, cnt, tot, byt, avg_b in mc:
    print("kind=%d  x%-7d  %8.3f s  %10.3f GB  avg=%.0f KiB"
          % (kind, cnt, tot / 1e9, (byt or 0) / 1e9, (avg_b or 0) / 1024))
    if kind == 1:
        h2d_ns = tot

# total GPU busy vs idle
allk = q("SELECT COUNT(*), SUM(end - start) FROM CUPTI_ACTIVITY_KIND_KERNEL")[0]
all_kernel_ns = allk[1] or 0

# per-chunk time from server log (stored by step0_run.py)
run = json.load(open(os.path.join(TASKDIR, "step0_run.json")))
thr_lines = run.get("prefill", {}).get("throughput_lines", [])
vals = []
for l in thr_lines:
    mm = re.search(r"input throughput \(token/s\):\s*([0-9.]+)", l)
    nt = re.search(r"#new-token:\s*(\d+)", l)
    if mm and nt and int(nt.group(1)) == CHUNK_TOKENS:
        vals.append(CHUNK_TOKENS / float(mm.group(1)))
steady = vals[1:] if len(vals) > 1 else vals
T_med = statistics.median(steady)
T_list = [round(v, 2) for v in vals]
print("\nfull-chunk times (s):", T_list)
print("steady median T = %.2f s  (n=%d, spread %.2f..%.2f)"
      % (T_med, len(steady), min(steady), max(steady)))

chunks_eq = W_s / T_med
gemm = moe_total_ns / 1e9
copies = h2d_ns / 1e9
gemm_pc = gemm / chunks_eq
copies_pc = copies / chunks_eq
all_kernel_pc = all_kernel_ns / 1e9 / chunks_eq
other_kernel_pc = all_kernel_pc - gemm_pc
idle_pc = T_med - all_kernel_pc - copies_pc
rest_pc = T_med - gemm_pc - copies_pc
launches_per_chunk = moe_count / chunks_eq

bw_eff = (TODAY_TRAFFIC_TB * 1e3) / gemm_pc           # GB/s effective, today's kernel
gemm_v2 = EXPERT_TOTAL_GB / bw_eff                    # single pass over expert weights
t_v0 = T_med - gemm_pc / 2
t_v2 = T_med - gemm_pc + gemm_v2

out = {
    "report": rep,
    "window_s": round(W_s, 2),
    "chunks_eq": round(chunks_eq, 3),
    "T_full_chunks_s": T_list,
    "T_steady_median_s": round(T_med, 3),
    "T_steady_min": round(min(steady), 3),
    "T_steady_max": round(max(steady), 3),
    "gemm_total_s": round(gemm, 3),
    "gemm_moe_launches": moe_count,
    "launches_per_chunk": round(launches_per_chunk, 1),
    "gemm_per_chunk_s": round(gemm_pc, 3),
    "copies_h2d_total_s": round(copies, 3),
    "copies_per_chunk_s": round(copies_pc, 3),
    "all_kernel_per_chunk_s": round(all_kernel_pc, 3),
    "other_kernel_per_chunk_s": round(other_kernel_pc, 3),
    "gpu_idle_per_chunk_s": round(idle_pc, 3),
    "rest_per_chunk_s": round(rest_pc, 3),
    "pct": {"gemm": round(100 * gemm_pc / T_med, 1),
            "copies": round(100 * copies_pc / T_med, 1),
            "rest": round(100 * rest_pc / T_med, 1)},
    "tok_s_now": round(CHUNK_TOKENS / T_med, 1),
    "bw_eff_GB_s": round(bw_eff, 0),
    "implied_v0": {"chunk_s": round(t_v0, 2), "tok_s": round(CHUNK_TOKENS / t_v0, 1)},
    "implied_v2": {"gemm_s": round(gemm_v2, 2), "chunk_s": round(t_v2, 2),
                   "tok_s": round(CHUNK_TOKENS / t_v2, 1)},
}
json.dump(out, open(os.path.join(TASKDIR, "step0_split.json"), "w"), indent=1)
print("\n=== SPLIT (per 8128 chunk) ===")
print(json.dumps(out, indent=1))
