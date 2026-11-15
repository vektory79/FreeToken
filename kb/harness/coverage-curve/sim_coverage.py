#!/usr/bin/env python3
"""Offline simulator of the hybrid expert-slot policy (coverage-curve F2/F3).

Host-side replica of _ensure_experts_hybrid_kernel
(python/freetoken/moe/offload_kernels.py) with the production defaults
FREETOKEN_HYBRID_FETCH=recency, FREETOKEN_HYBRID_FETCH_ROUND=min-wait.
Replays the per-step decode route trace (routes_<tag>.json, written by the
decode-research driver in eager mode) from a cold cache for every S in the
sweep; validates the S0 point against the boot's own per-layer telemetry
(stats_<tag>.json); emits the C(S) and cpu_gib_per_step(S) tables.

Policy mirrored per (layer-launch, execution order; the cache-wide step
counter ticks once per launch):
  active = unique routed expert ids
  hits (slot >= 0): usage[slot] = step
  missing = active without a slot; fetch = min(missing, cap-count, E,
            pool_size - hits); cap-count = min-wait rounding of
            fraction * missing, or hybrid_max_fetch when fraction == 0
  picks = missing sorted by (recency desc, expert id asc)
  victims = argmin(usage), ties to the lowest slot
  overflow misses stay non-resident (computed on the CPU); every active
  expert's recency = step (bumped AFTER victim selection, as in the kernel)

Metrics:
  C(S)   = 1 - (missing - fetched) / active, aggregated over all layer-calls
           (and the median of the per-step values)
  cpu_gib_per_step(S) = overflow pairs * per-layer slot bytes, per decode step

Usage:
  python sim_coverage.py --routes routes_trace.json --stats stats_trace.json \
      --out results_trace.md [--csv results_trace.csv] [--max-delta-pp 2.0]
Exit codes: 0 ok (validation within threshold); 6 validation failed;
7 input inconsistency.
"""
import argparse
import json
import sys

import numpy as np


def die(code, msg):
    print("[sim] FATAL: %s" % msg, file=sys.stderr)
    sys.exit(code)


def load_inputs(routes_path, stats_path):
    doc = json.load(open(routes_path))
    meta_all = doc.get("meta") or []
    if not meta_all:
        die(7, "routes file has no cache meta")
    meta = meta_all[0]
    layer_ids = doc.get("layer_ids") or []
    routes = doc.get("routes") or []
    if doc.get("n_replay_steps", -1) != 0:
        die(7, "routes were captured under CUDA graphs (n_replay_steps=%s); eager boot required"
            % doc.get("n_replay_steps"))
    if len(layer_ids) != len(routes) or not routes:
        die(7, "routes/layer_ids inconsistent: %d vs %d" % (len(layer_ids), len(routes)))
    stats = json.load(open(stats_path)) if stats_path else None
    return doc, meta, layer_ids, routes, stats


def check_meta(meta, layer_ids, routes):
    L, E = meta["num_layers"], meta["num_experts"]
    pools = meta["pool_sizes"]
    pol = meta["pool_of_layer"]
    sb = meta.get("slot_bytes_per_layer")
    if not pools or not pol or not sb:
        die(7, "meta missing pool_sizes/pool_of_layer/slot_bytes_per_layer")
    if len(pol) != L or len(sb) != L:
        die(7, "meta sizes wrong: pool_of_layer=%d slot_bytes=%d vs L=%d" % (len(pol), len(sb), L))
    if sum(pools) != meta["cache_size"] or min(pools) <= 0:
        die(7, "pool_sizes sum %d != cache_size %s" % (sum(pools), meta.get("cache_size")))
    if sorted(set(pol)) != list(range(len(pools))):
        die(7, "pool_of_layer does not cover every pool: %s" % sorted(set(pol)))
    if meta.get("locked_hybrid") or meta.get("cpu_layer_ids") or meta.get("unpinned_layers"):
        die(7, "boot has locked/cpu/unpinned layers %s/%s/%s - the plain policy replica does not cover it"
            % (meta.get("locked_hybrid"), meta.get("cpu_layer_ids"), meta.get("unpinned_layers")))
    w = len(routes[0])
    if any(len(r) != w for r in routes):
        die(7, "route width varies across calls")
    return L, E, pools, pol, sb, w


def split_equal(total, mins, maxs, weights):
    """Port of pool_policy._split_equal: floors first, then fewest-slots-per-layer wins."""
    if len(mins) == 1:
        return [total]
    if total < sum(mins):
        die(7, "S=%d below pool floors %d" % (total, sum(mins)))
    if total > sum(maxs):
        die(7, "S=%d above pool room %d" % (total, sum(maxs)))
    sizes = list(mins)
    left = total - sum(sizes)
    while left > 0:
        open_ = [i for i in range(len(sizes)) if sizes[i] < maxs[i]]
        if not open_:
            break
        i = min(open_, key=lambda j: (sizes[j] / weights[j], j))
        sizes[i] += 1
        left -= 1
    return sizes


class Sim:
    """One cold-start replay of the whole trace at a fixed pool-size vector."""

    def __init__(self, L, E, pool_sizes, pool_of_layer, slot_bytes, fractions, max_fetch, n_calls):
        self.L, self.E = L, E
        self.pool_sizes = list(pool_sizes)
        self.pool_of_layer = list(pool_of_layer)
        self.slot_bytes = slot_bytes
        self.fractions = fractions
        self.max_fetch = int(max_fetch) if max_fetch is not None else 1
        self.cap = E
        P = len(self.pool_sizes)
        self.usage = [np.zeros(s, dtype=np.int64) for s in self.pool_sizes]
        self.id_of_slot = [np.full(s, -1, dtype=np.int64) for s in self.pool_sizes]
        self.slot_for_id = np.full((P, L, E), -1, dtype=np.int32)
        self.recency = np.full((L, E), -1, dtype=np.int64)
        self._step = 0
        # per-layer telemetry replica of the kernel STATS block (unique-expert basis)
        self.miss = np.zeros(L, dtype=np.int64)
        self.fetch = np.zeros(L, dtype=np.int64)
        self.act = np.zeros(L, dtype=np.int64)
        self.calls = np.zeros(L, dtype=np.int64)
        self._n_calls = n_calls
        self.overflow_bytes = np.zeros(n_calls, dtype=np.int64)
        self.call_active = np.zeros(n_calls, dtype=np.int64)
        self.call_miss = np.zeros(n_calls, dtype=np.int64)
        self.call_fetch = np.zeros(n_calls, dtype=np.int64)

    def run(self, layer_ids, routes):
        for i, (lid, ids) in enumerate(zip(layer_ids, routes)):
            self._ensure(i, lid, self.pool_of_layer[lid], ids)
        # step boundaries: a new step starts when the layer id does not increase
        step_idx = np.zeros(len(layer_ids), dtype=np.int64)
        k = 0
        prev = -1
        for i, lid in enumerate(layer_ids):
            if lid <= prev:
                k += 1
            step_idx[i] = k
            prev = lid
        self.step_idx = step_idx
        self._starts = np.concatenate([[0], np.flatnonzero(np.diff(step_idx) != 0) + 1])

    def _ensure(self, i, lid, p, ids):
        self._step += 1
        step = self._step
        act = np.unique(np.asarray(ids, dtype=np.int64))
        slots = self.slot_for_id[p, lid, act]
        hit = slots >= 0
        if hit.any():
            self.usage[p][slots[hit]] = step
        miss = act[~hit]
        nm = int(miss.size)
        n_active = int(act.size)
        frac = self.fractions[lid] if self.fractions else 0.0
        if frac > 0.0:
            q = int(round(frac * 65536))
            lo = (nm * q) >> 16
            cost_lo = max(lo * (65536 - q), (nm - lo) * q)
            cost_hi = max((lo + 1) * (65536 - q), (nm - lo - 1) * q)
            mf = lo if cost_lo <= cost_hi else lo + 1
        else:
            mf = self.max_fetch
        num_fetch = min(nm, mf, self.cap, self.pool_sizes[p] - int(hit.sum()))
        if num_fetch > 0:
            rec = self.recency[lid, miss]
            order = np.lexsort((miss, -rec))  # recency desc, expert id asc
            picks = miss[order[:num_fetch]]
            usage = self.usage[p]
            idmap = self.id_of_slot[p]
            for e in picks:
                victim = int(np.argmin(usage))  # first minimum = lowest slot (kernel tie rule)
                old = int(idmap[victim])
                if old >= 0:
                    self.slot_for_id[p, old // self.E, old % self.E] = -1
                idmap[victim] = lid * self.E + int(e)
                self.slot_for_id[p, lid, e] = victim
                usage[victim] = step
        overflow = nm - num_fetch
        self.recency[lid, act] = step
        self.miss[lid] += nm
        self.fetch[lid] += num_fetch
        self.act[lid] += n_active
        self.calls[lid] += 1
        self.overflow_bytes[i] = overflow * self.slot_bytes[lid]
        self.call_active[i] = n_active
        self.call_miss[i] = nm
        self.call_fetch[i] = num_fetch

    def summarize(self):
        A, M, F = int(self.act.sum()), int(self.miss.sum()), int(self.fetch.sum())
        c_agg = 1.0 - (M - F) / A if A else 0.0
        ends = np.append(self._starts[1:], self._n_calls)
        per_step_c = []
        per_step_gib = []
        for s0, e0 in zip(self._starts, ends):
            sl = slice(int(s0), int(e0))
            a = int(self.call_active[sl].sum())
            m = int(self.call_miss[sl].sum())
            f = int(self.call_fetch[sl].sum())
            per_step_c.append(1.0 - (m - f) / a if a else 1.0)
            per_step_gib.append(int(self.overflow_bytes[sl].sum()) / 2**30)
        return {
            "C_agg": c_agg,
            "C_med": float(np.median(per_step_c)),
            "C_min": float(np.min(per_step_c)),
            "gib_step_mean": float(np.mean(per_step_gib)),
            "gib_step_med": float(np.median(per_step_gib)),
            "A": A, "M": M, "F": F,
            "per_layer_miss_rate": (self.miss / np.maximum(self.act, 1)).tolist(),
            "per_layer_fetch_per_call": (self.fetch / np.maximum(self.calls, 1)).tolist(),
            "calls_per_layer": self.calls.tolist(),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", required=True)
    ap.add_argument("--stats", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--max-delta-pp", type=float, default=2.0)
    ap.add_argument("--gate-add", type=int, default=300)
    ap.add_argument("--validate-calls", type=int, default=None,
                    help="replay only the first N layer-calls for the S0 validation "
                         "(the boot's stats dump is cumulative up to its dump point); "
                         "default: the whole trace")
    a = ap.parse_args()

    doc, meta, layer_ids, routes, stats = load_inputs(a.routes, a.stats)
    L, E, pools0, pol, sb, width = check_meta(meta, layer_ids, routes)
    S0 = int(meta["cache_size"])
    fractions = meta.get("hybrid_fetch_fraction")
    max_fetch = meta.get("hybrid_max_fetch", 1)
    max_slots = meta.get("max_slots")

    # pool construction mirrors the engine: the boot's own grouping (order kept from
    # pool indices), make_pools(min = top_k * decode-batch), then the split policy
    members = [[] for _ in pools0]
    for layer, p in enumerate(pol):
        members[p].append(layer)
    mins = [min(E, width) for _ in pools0]
    maxs = [min(len(m) * E, max_slots) if max_slots else len(m) * E for m in members]
    weights = [len(m) for m in members]
    total_room = sum(maxs)

    sweep = [S0] + [S0 + d for d in range(100, 601, 100)] + [2 * S0, L * E]
    seen = set()
    sweep = [s for s in sweep if not (s in seen or seen.add(s))]
    feasible = sorted(s for s in sweep if sum(mins) <= s <= total_room)
    skipped = [s for s in sweep if s not in feasible]

    rows = []
    by_S = {}
    for S in feasible:
        sizes = list(pools0) if S == S0 else split_equal(S, mins, maxs, weights)
        sim = Sim(L, E, sizes, pol, sb, fractions, max_fetch, len(routes))
        sim.run(layer_ids, routes)
        r = sim.summarize()
        r["S"] = S
        r["pool_sizes"] = sizes
        by_S[S] = r
        rows.append(r)
    if S0 not in by_S:
        die(7, "S0=%d itself infeasible for the pools" % S0)
    S0r = by_S[S0]

    # validation sim: a prefix replay when the telemetry covers only the calls up
    # to its dump point (--validate-calls), else the same full trace
    val_sim = S0r
    if a.validate_calls:
        n = min(a.validate_calls, len(routes))
        vsim = Sim(L, E, list(pools0), pol, sb, fractions, max_fetch, n)
        vsim.run(layer_ids[:n], routes[:n])
        val_sim = vsim.summarize()

    # validation at S0 vs the boot's own telemetry
    val = {"delta_pp": None}
    tel_per_layer = None
    if stats:
        caches = stats.get("caches") or []
        if caches:
            agg = caches[0].get("aggregate") or {}
            calls = agg.get("layer_calls", 0)
            tel_miss = agg.get("missing_per_layer", 0.0) * calls
            tel_act = agg.get("active_per_layer", 0.0) * calls
            tel_fetch = agg.get("fetched_per_layer", 0.0) * calls
            pl = caches[0].get("per_layer") or []
            if isinstance(pl, dict) and "per_layer" in pl:
                # the driver stores decode_miss_stats_per_layer() verbatim:
                # {"per_layer": [rows]}
                pl = pl["per_layer"]
            tel_per_layer = {row["layer"]: row for row in pl if isinstance(row, dict) and "layer" in row}
            if tel_act:
                c_tel = 1.0 - (tel_miss - tel_fetch) / tel_act
                c_sim = val_sim["C_agg"]
                val["delta_pp"] = abs(c_sim - c_tel) * 100.0
                val["ok"] = val["delta_pp"] <= a.max_delta_pp
                val["C_tel"] = c_tel
                val["C_sim"] = c_sim
                val["validate_calls"] = a.validate_calls or len(routes)

    Sg = by_S.get(S0 + a.gate_add)
    verdict = {"gate_add": a.gate_add, "available": Sg is not None}
    if Sg:
        d_agg = (Sg["C_agg"] - S0r["C_agg"]) * 100.0
        d_med = (Sg["C_med"] - S0r["C_med"]) * 100.0
        verdict.update({"delta_agg_pp": d_agg, "delta_med_pp": d_med,
                        "verdict": "NO-GO" if d_agg < 5.0 else ("GO" if d_agg >= 8.0 else "OWNER")})

    lines = []
    lines.append("# Coverage curve results (tag %s)" % doc.get("tag", "trace"))
    lines.append("")
    lines.append("- routes: %s (%d layer-calls, %d decode steps, width %d)"
                 % (a.routes, len(layer_ids), len(layer_ids) // L, width))
    lines.append("- S0 = %d slots, boot pools = %s" % (S0, pools0))
    lines.append("- fetch: per-layer fractions = %s, fixed max_fetch = %s"
                 % (sorted(set(f for f in (fractions or []) if f)) or None, max_fetch))
    lines.append("- sweep room: floors %d, ceiling %d pairs; skipped points: %s"
                 % (sum(mins), total_room, skipped or "none"))
    lines.append("")
    if val.get("delta_pp") is not None:
        lines.append("## Validation at S0 (sim vs boot telemetry)")
        lines.append("")
        lines.append("- C tel = %.4f, C sim = %.4f (prefix %s calls), delta = %.2f pp (threshold %.1f) -> %s"
                     % (val.get("C_tel", 0.0), val.get("C_sim", 0.0), val.get("validate_calls"), val["delta_pp"],
                        a.max_delta_pp, "PASS" if val["ok"] else "FAIL"))
        lines.append("")
        lines.append("| layer | miss_rate tel | miss_rate sim | d, pp | fetch/call tel | fetch/call sim |")
        lines.append("|---|---|---|---|---|---|")
        pl_sim = val_sim["per_layer_miss_rate"]
        pl_sim_f = val_sim["per_layer_fetch_per_call"]
        max_layer_d = 0.0
        for layer in range(L):
            trow = (tel_per_layer or {}).get(layer, {})
            tm = trow.get("miss_rate")
            tf = trow.get("fetched_per_step")
            sm = pl_sim[layer]
            sf = pl_sim_f[layer]
            d = abs((tm or 0.0) - sm) * 100.0
            max_layer_d = max(max_layer_d, d)
            lines.append("| %d | %s | %.4f | %.2f | %s | %.4f |"
                         % (layer, ("%.4f" % tm) if tm is not None else "n/a", sm, d,
                            ("%.4f" % tf) if tf is not None else "n/a", sf))
        val["max_layer_delta_pp"] = max_layer_d
        lines.append("")
        lines.append("max per-layer miss_rate delta: %.2f pp" % max_layer_d)
        lines.append("")
    lines.append("## C(S) table")
    lines.append("")
    lines.append("| S | pool sizes | C(S) agg, % | d vs S0, pp | C med/step, % | cpu GiB/step mean | cpu GiB/step p50 |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append("| %d | %s | %.2f | %+.2f | %.2f | %.3f | %.3f |"
                     % (r["S"], r["pool_sizes"], r["C_agg"] * 100.0,
                        (r["C_agg"] - S0r["C_agg"]) * 100.0, r["C_med"] * 100.0,
                        r["gib_step_mean"], r["gib_step_med"]))
    lines.append("")
    lines.append("## Gate (+%d slots)" % a.gate_add)
    lines.append("")
    if Sg:
        lines.append("- delta C agg = %+.2f pp, delta C med/step = %+.2f pp"
                     % (verdict["delta_agg_pp"], verdict["delta_med_pp"]))
        lines.append("- verdict: **%s** (rule: < +5 pp NO-GO, >= +8 pp GO-prototype, else owner)"
                     % verdict["verdict"])
    else:
        lines.append("- S0+%d not feasible in the sweep (room %d pairs)" % (a.gate_add, total_room))
    lines.append("")
    text = "\n".join(lines)
    open(a.out, "w").write(text)
    print(text)
    if a.csv:
        with open(a.csv, "w") as f:
            f.write("S,pool_sizes,C_agg,C_med,gib_step_mean,gib_step_med\n")
            for r in rows:
                f.write("%d,%s,%.6f,%.6f,%.6f,%.6f\n"
                        % (r["S"], " ".join(map(str, r["pool_sizes"])),
                           r["C_agg"], r["C_med"], r["gib_step_mean"], r["gib_step_med"]))
    if val.get("delta_pp") is not None and not val["ok"]:
        sys.exit(6)


if __name__ == "__main__":
    main()
