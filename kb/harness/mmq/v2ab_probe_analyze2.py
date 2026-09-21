#!/usr/bin/env python3
"""v2 liveness assertions from the capture-from-launch probe (attempt 2).

Splits the kernel timeline into phases by GPU-idle gaps (>1.5 s):
  boot (graph-capture replays) -> prefill chunks 1..8 + tail (back-to-back)
  -> decode (49-token eager grouped prefill + 63 graph-replayed steps)
Assertions:
  P1 eager grouped MMQ MoE kernels present in the prefill phase
  P2 eager moe_vec* instances in the prefill phase == 0
  P3 moe_vec* instances in the decode phase > 0 (decode stays on moe_vec)
  P4 moe_align kernels present post-boot
Prints per-phase top kernels + verdict JSON; exit 3 on failure.
"""
import glob
import json
import os
import sqlite3
import subprocess
import sys

TASKDIR = os.path.dirname(os.path.abspath(__file__))
NSYS = "/usr/local/bin/nsys"
GAP_S = 1.5

reps = sorted(glob.glob(os.path.join(TASKDIR, "report*.nsys-rep")), key=os.path.getmtime)
if not reps:
    print("FATAL: no report*.nsys-rep")
    sys.exit(2)
rep = reps[-1]
print("report:", rep)
SQ = os.path.join(TASKDIR, "v2probe2.sqlite")
if os.path.exists(SQ):
    os.remove(SQ)
r = subprocess.run([NSYS, "export", "--type", "sqlite", "--force-overwrite", "true",
                    "-o", SQ, rep], capture_output=True, text=True, timeout=1800)
print("export rc:", r.returncode, (r.stderr or "")[-200:])
if r.returncode != 0:
    sys.exit(2)

con = sqlite3.connect(SQ)
cur = con.cursor()
rows = cur.execute("""
    SELECT k.start, k.end, s.value
    FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName = s.id
    ORDER BY k.start
""").fetchall()
print("kernel records:", len(rows))
if not rows:
    sys.exit(2)

# phase split by start-time gaps
starts = [r0[0] for r0 in rows]
cuts = [0]
for i in range(1, len(rows)):
    if (starts[i] - starts[i - 1]) > GAP_S * 1e9:
        cuts.append(i)
cuts.append(len(rows))
phases = []
for a, b in zip(cuts, cuts[1:]):
    if b > a:
        phases.append((a, b))
print("phases (gap > %.1f s): %d" % (GAP_S, len(phases)))
for pi, (a, b) in enumerate(phases):
    span = (rows[b - 1][1] - rows[a][0]) / 1e9
    print("  phase %d: kernels %6d  span %8.2f s  first %s"
          % (pi, b - a, span, rows[a][2][:40]))


def classify(name):
    if name.startswith("moe_vec"):
        return "vec"
    if name.startswith("moe_") and "align" not in name:
        return "grouped"
    if "align" in name and "moe" in name:
        return "align"
    return "other"


phase_stat = []
for pi, (a, b) in enumerate(phases):
    seg = rows[a:b]
    top = {}
    vec_first = vec_last = None
    grouped_last = None
    grouped_names = set()
    for st, en, nm in seg:
        top[nm] = top.get(nm, 0) + 1
        c = classify(nm)
        if c == "vec":
            vec_first = st if vec_first is None else vec_first
            vec_last = en
        elif c == "grouped":
            grouped_names.add(nm)
            grouped_last = en
    phase_stat.append({
        "idx": pi, "n": b - a, "span_s": round(span, 2) if (b - a) else 0,
        "t0_s": round(rows[a][0] / 1e9, 2),
        "vec_count": sum(1 for _, _, nm in seg if classify(nm) == "vec"),
        "grouped_count": sum(1 for _, _, nm in seg if classify(nm) == "grouped"),
        "align_count": sum(1 for _, _, nm in seg if classify(nm) == "align"),
        "grouped_names": sorted(grouped_names),
        "top10": sorted(top.items(), key=lambda kv: -kv[1])[:10],
    })

print("\n=== per-phase summary ===")
for ps in phase_stat:
    print(json.dumps({k: v for k, v in ps.items() if k != "top10"}))
    print("   top:", [(n[:44], c) for n, c in ps["top10"]])

# identify phases: boot = the one containing graph-capture replays (before the
# big gap); prefill = the phase with the most eager grouped kernels; decode =
# the last phase
boot = phase_stat[0]
prefill = max(phase_stat[1:], key=lambda ps: ps["grouped_count"]) if len(phase_stat) > 1 else None
decode = phase_stat[-1] if len(phase_stat) > 2 else None

p1 = bool(prefill and prefill["grouped_count"] > 0)
p2 = bool(prefill and prefill["vec_count"] == 0)
p3 = bool(decode and decode["vec_count"] > 0)
p4 = any(ps["align_count"] > 0 for ps in phase_stat[1:])
verdict = {
    "report": rep,
    "n_phases": len(phase_stat),
    "boot": {k: boot[k] for k in ("idx", "n", "span_s", "vec_count", "grouped_count")},
    "prefill": ({k: prefill[k] for k in ("idx", "n", "span_s", "vec_count",
                                         "grouped_count", "align_count",
                                         "grouped_names")} if prefill else None),
    "decode": ({k: decode[k] for k in ("idx", "n", "span_s", "vec_count",
                                       "grouped_count", "align_count")} if decode else None),
    "grouped_kernels_present_prefill": p1,
    "moe_vec_absent_prefill_eager": p2,
    "moe_vec_present_decode": p3,
    "moe_align_present": p4,
}
print("\n" + json.dumps(verdict, indent=1))
ok = p1 and p2 and p3 and p4
sys.exit(0 if ok else 3)
