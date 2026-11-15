#!/usr/bin/env python3
"""Offline checks for sim_coverage.py (CPU-only, no server).

1. Analytic policy test: one layer, hand-computed LRU/cap arithmetic, compared
   against the Sim replica step by step.
2. split_equal port vs the real pool_policy.split_slots over a sweep of S.
3. Plumbing: synthetic routes file runs end to end (sweep + validation path).
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sim_coverage import Sim, split_equal  # noqa: E402


def check(cond, msg):
    if not cond:
        print("FAIL: %s" % msg)
        sys.exit(1)
    print("ok: %s" % msg)


# ---- 1. analytic policy test ----
L, E = 1, 8
sim = Sim(L, E, [4], [0], [64], [0.0], 2, 2)
ids = [3, 1, 4, 1, 5, 2, 6, 7]
sim.run([0, 0], [ids, ids])
# step 1: 7 unique active, all missing, cap 2 -> fetch 2 (lowest ids 1,2), overflow 5
check(sim.call_active[0] == 7, "step1 active=7")
check(sim.call_miss[0] == 7, "step1 missing=7")
check(sim.call_fetch[0] == 2, "step1 fetch=2 (cap 2, recency ties -> lowest ids)")
check(sim.overflow_bytes[0] == 5 * 64, "step1 overflow bytes=5*64")
# step 2: experts 1,2 hit; 5 missing; fetch 2 lowest-id misses (3,4), overflow 3
check(sim.call_active[1] == 7 and sim.call_miss[1] == 5 and sim.call_fetch[1] == 2,
      "step2 miss=5 fetch=2")
check(int(sim.miss.sum()) == 12 and int(sim.fetch.sum()) == 4, "totals miss=12 fetch=4")

# recency priority: warm 1,2 then a step that misses both -> they must be fetched first
sim2 = Sim(1, 8, [4], [0], [64], [0.0], 2, 3)
sim2.run([0, 0, 0], [[1, 2, 3, 3, 3, 3, 3, 3], [1, 2, 3, 3, 3, 3, 3, 3], [4, 5, 6, 7, 0, 3, 3, 3]])
# step3: active {0,3,4,5,6,7}=6, none resident except... 3 was fetched at step2 -> hit
check(sim2.call_miss[2] == 5, "step3 missing=5 (3 hit)")
check(sim2.call_fetch[2] == 2, "step3 fetch=2")
check(int(sim2.slot_for_id[0, 0, 3]) >= 0, "expert 3 resident after step3")

# fraction cap: frac 0.5, 6 missing -> lo=3, min-wait -> 3
sim3 = Sim(1, 8, [8], [0], [64], [0.5], 1, 1)
sim3.run([0], [[0, 1, 2, 3, 4, 5, 0, 1]])
q = int(round(0.5 * 65536))
lo = (6 * q) >> 16
cost_lo = max(lo * (65536 - q), (6 - lo) * q)
cost_hi = max((lo + 1) * (65536 - q), (6 - lo - 1) * q)
expect = lo if cost_lo <= cost_hi else lo + 1
check(sim3.call_fetch[0] == expect, "frac fetch = min-wait(%d)" % expect)

# ---- 2. split_equal vs pool_policy ----
from freetoken.moe.pool_policy import PoolSpec, split_slots  # noqa: E402

rng = np.random.default_rng(7)
for trial in range(30):
    p = int(rng.integers(2, 5))
    members = [list(range(int(x), int(x) + int(rng.integers(1, 6)))) for x in
               np.cumsum(rng.integers(1, 6, size=p)) - rng.integers(1, 6, size=p)]
    members = [m for m in members if m]
    pools = [PoolSpec(tuple(m), 8, len(m) * 288) for m in members]
    mins = [8] * len(pools)
    maxs = [p.max_num_slots for p in pools]
    weights = [len(p.layers) for p in pools]
    room = sum(maxs)
    for S in [sum(mins), sum(mins) + 13, room // 2, room]:
        if S > room or S < sum(mins):
            continue
        a = split_equal(S, mins, maxs, weights)
        b = split_slots(S, pools, None)
        check(a == b, "split_equal %d -> %s == pool_policy %s (trial %d)" % (S, a, b, trial))

# ---- 3. end-to-end plumbing on a synthetic trace ----
tmp = tempfile.mkdtemp(prefix="covsim-")
L, E = 4, 8
steps = 60
rng = np.random.default_rng(11)
routes = []
layer_ids = []
for s in range(steps):
    for lid in range(L):
        layer_ids.append(lid)
        routes.append(rng.integers(0, E, size=8).tolist())
meta = {
    "num_layers": L, "num_experts": E, "cache_size": 32,
    "decode_target": "hybrid",
    "pool_sizes": [12, 20], "pool_of_layer": [0, 0, 1, 1],
    "hybrid_max_fetch": 2, "hybrid_fetch_fraction": [0.5, 0.5, 0.0, 0.0],
    "max_slots": None, "locked_hybrid": [], "cpu_layer_ids": [], "unpinned_layers": [],
    "layer_layout": [0, 0, 1, 1], "slot_bytes_per_layer": [64, 64, 128, 128],
}
doc = {"tag": "synth", "n_calls": len(routes), "layer_ids": layer_ids,
       "routes": routes, "n_replay_steps": 0, "meta": [meta]}
rp = os.path.join(tmp, "routes_synth.json")
json.dump(doc, open(rp, "w"))

# stats built from a prefix replay of the same sim = self-consistent telemetry
v = Sim(L, E, [12, 20], [0, 0, 1, 1], meta["slot_bytes_per_layer"], meta["hybrid_fetch_fraction"], 2, 200)
n = 120
v.run(layer_ids[:n], routes[:n])
per = []
for lid in range(L):
    s = int(v.calls[lid])
    per.append({"layer": lid, "pool": lid // 2, "pool_slots": 12 if lid < 2 else 20,
                "steps": s, "active_per_step": float(v.act[lid]) / s,
                "missing_per_step": float(v.miss[lid]) / s,
                "miss_rate": float(v.miss[lid]) / max(v.act[lid], 1),
                "fetched_per_step": float(v.fetch[lid]) / s})
A, M, F = int(v.act.sum()), int(v.miss.sum()), int(v.fetch.sum())
agg = {"layer_calls": int(v.calls.sum()), "active_per_layer": A / int(v.calls.sum()),
       "missing_per_layer": M / int(v.calls.sum()), "fetched_per_layer": F / int(v.calls.sum())}
stats = {"caches": [{"aggregate": agg, "per_layer": per}]}
sp = os.path.join(tmp, "stats_synth.json")
json.dump(stats, open(sp, "w"))

out = os.path.join(tmp, "results_synth.md")
r = subprocess.run([sys.executable, os.path.join(HERE, "sim_coverage.py"),
                    "--routes", rp, "--stats", sp, "--out", out,
                    "--csv", os.path.join(tmp, "res.csv"), "--validate-calls", str(n)],
                   capture_output=True, text=True)
print(r.stdout[-1200:])
print(r.stderr[-800:])
check(r.returncode == 0, "sim exits 0 on synthetic inputs (validation self-consistent)")
check(os.path.exists(out) and "C(S) table" in open(out).read(), "results md written")
check("PASS" in r.stdout, "validation PASS on self-consistent stats")

print("ALL SYNTH CHECKS PASSED")
