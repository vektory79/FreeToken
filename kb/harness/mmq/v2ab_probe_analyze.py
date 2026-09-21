#!/usr/bin/env python3
"""Kernel-name liveness assertions from the v2 probe nsys sqlite export.

Checks (see v2ab_probe.py docstring):
  P1 grouped MMQ MoE kernels present in the capture (moe_* kernels that are
     not moe_vec* and not moe_align*)
  P2 moe_align launches present (sgl kernel or triton `_moe_align_small`)
  P3 moe_vec* present (decode) and the FIRST moe_vec instance starts after
     the LAST grouped MoE kernel ends (server-sequential phases)
Prints the full top-kernel table plus a verdict JSON; exit 3 on failure.
"""
import json
import os
import sqlite3
import subprocess
import sys

TASKDIR = os.path.dirname(os.path.abspath(__file__))
NSYS = "/usr/local/bin/nsys"
import glob
_reps = sorted(glob.glob(os.path.join(TASKDIR, "report*.nsys-rep")), key=os.path.getmtime)
REP = _reps[-1] if _reps else os.path.join(TASKDIR, "v2probe.nsys-rep")
print("report:", REP)
SQ = os.path.join(TASKDIR, "v2probe.sqlite")

if not os.path.exists(SQ):
    r = subprocess.run([NSYS, "export", "--type", "sqlite", "--force-overwrite", "true",
                        "-o", SQ, REP], capture_output=True, text=True, timeout=1800)
    print("export rc:", r.returncode, (r.stderr or "")[-400:])
    if r.returncode != 0:
        sys.exit(2)

con = sqlite3.connect(SQ)
rows = con.execute("""
    SELECT s.value, COUNT(*), SUM(k.end-k.start), MIN(k.start), MAX(k.end)
    FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName = s.id
    GROUP BY s.value ORDER BY 3 DESC
""").fetchall()

print("=== top kernels (name, instances, total_s) ===")
for name, cnt, tot, lo, hi in rows:
    print("%8.3f s  x%-7d  %s" % ((tot or 0) / 1e9, cnt, name[:120]))


def agg(names):
    cnt = sum(r[1] for r in rows if r[0] in names)
    lo = min((r[3] for r in rows if r[0] in names and r[3] is not None), default=None)
    hi = max((r[4] for r in rows if r[0] in names and r[4] is not None), default=None)
    return cnt, lo, hi


grouped_names = [r[0] for r in rows if r[0].startswith("moe_")
                 and not r[0].startswith("moe_vec") and "align" not in r[0]]
vec_names = [r[0] for r in rows if r[0].startswith("moe_vec")]
align_names = [r[0] for r in rows if "moe_align" in r[0] or "_moe_align" in r[0]]

g_cnt, g_lo, g_hi = agg(grouped_names)
v_cnt, v_lo, v_hi = agg(vec_names)
a_cnt, a_lo, a_hi = agg(align_names)

p1 = g_cnt > 0
p2 = a_cnt > 0
p3 = v_cnt > 0 and g_hi is not None and v_lo is not None and v_lo > g_hi
verdict = {
    "grouped_present": p1,
    "grouped_names": grouped_names,
    "grouped_count": g_cnt,
    "grouped_first_start": g_lo,
    "grouped_last_end": g_hi,
    "moe_align_present": p2,
    "moe_align_names": align_names,
    "moe_align_count": a_cnt,
    "moe_vec_present_decode_only": p3,
    "moe_vec_names": vec_names,
    "moe_vec_count": v_cnt,
    "moe_vec_first_start": v_lo,
    "moe_vec_last_end": v_hi,
}
print("\n" + json.dumps(verdict, indent=1))
sys.exit(0 if (p1 and p2 and p3) else 3)
