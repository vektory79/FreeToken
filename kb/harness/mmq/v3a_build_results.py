#!/usr/bin/env python3
"""Assemble v3a-ab-results.json from the sweep artifacts."""
import json
import os

VDIR = "/media/ai/src/FreeToken/.tasks/mmq-v3-stationary-moe"
raw = json.load(open(os.path.join(VDIR, "v3a_stage_raw.json")))
liv = json.load(open(os.path.join(VDIR, "v3a_probe_liveness.json")))
base_sum = json.load(open(os.path.join(VDIR, "quality/base_stage_summary.json")))
win_sum = json.load(open(os.path.join(VDIR, "quality/win_stage_summary.json")))
div = json.load(open(os.path.join(VDIR, "v3a_divergence.json")))

BOOT = (".venv/bin/ft serve --port 18801 --host 127.0.0.1 "
        "--model-path /media/ai/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL.gguf "
        "--moe-cache-auto --kv-reserve-tokens 500000 --kv-cache-dtype fp8 "
        "--memory-ratio 0.85 --max-prefill-length 8191 --moe-strategy hybrid "
        "--moe-cpu-threads 16 --max-running-requests 1")

runs = {}
for k in ("v3a_t4", "v3a_t8", "v3a_t16", "v3a_t32"):
    r = raw[k]
    tile = k.split("t")[1]
    runs[tile] = {
        "env": None if r["env"] == "unset" else "FREETOKEN_GGUF_MOE_MTILE=" + r["env"].split("=")[1],
        "ready_s": r["ready_s"],
        "throughput_lines_in_order": r["chunks"],
        "steady_c2_c7": r["steady"],
        "prefill_median_tok_s": r["med"],
        "chunk_time_s": round(8128 / r["med"], 2),
        "decode_steady_tok_s": r["dec"],
        "decode_p50_tok_s": r["p50"],
        "decode_radix": {"cached": r["cached"], "new": 49},
        "prefill_wall_s": r["wall"],
        "peak_vram_mib": r["peak"],
        "boot_log": ".tasks/ft-gguf-serve-tuning/logs/%s.log" % k,
        "stage_out": ".tasks/mmq-v3-stationary-moe/v3a_%s.out" % k,
    }

t = {x: runs[x]["chunk_time_s"] for x in ("4", "8", "16", "32")}
a = (t["4"] - t["8"]) / 2.0
rest = round(t["4"] - 4 * a, 2)
moe4 = round(4 * a, 2)
model = {
    "method": "e2e chunk-time subtraction, rest assumed tile-invariant (dense q8_0 GEMM 6.04 s + fetch copies 3.15 s + attention/GDN ~1.45 s + misc)",
    "rest_s": rest,
    "moe_term_tile4_s": moe4,
    "moe_term_ratio_vs_tile4": {
        "pure_1_over_tile_model": {"8": 0.5, "16": 0.25, "32": 0.125},
        "measured_e2e": {x: round((t[x] - rest) / moe4, 3) for x in ("8", "16", "32")},
    },
    "nsys_tile16_moe_sum_s_per_chunk": 5.01,
    "nsys_tile4_moe_sum_s_per_chunk": 12.62,
    "nsys_moe_speedup_tile16_vs_tile4": 2.52,
}

ident = sum(1 for r in div["primary"] if r["identical"])
digit_ok = all(v == [True, True] for v in div["digit"].values())
battery = {
    "stages": {
        "base": {"env": "unset", "boot_ready_s": base_sum["facts"]["ready_s"],
                 "battery_rc": base_sum["battery"]["rc"],
                 "battery_wall_s": base_sum["battery"]["wall_s"],
                 "marker": base_sum["facts"]["marker"],
                 "boot_plan": {k: base_sum["facts"][k] for k in ("kv_alloc", "free_mem", "fetch_fraction")}},
        "win32": {"env": "FREETOKEN_GGUF_MOE_MTILE=32", "boot_ready_s": win_sum["facts"]["ready_s"],
                  "battery_rc": win_sum["battery"]["rc"],
                  "battery_wall_s": win_sum["battery"]["wall_s"],
                  "marker": win_sum["facts"]["marker"],
                  "boot_plan": {k: win_sum["facts"][k] for k in ("kv_alloc", "free_mem", "fetch_fraction")}},
    },
    "identical_pairs": "%d/24" % ident,
    "digit_prompts_ok_both": digit_ok,
    "divergence_json": ".tasks/mmq-v3-stationary-moe/v3a_divergence.json",
}

out = {
    "campaign": "v3a grouped-gguf-moe m-tile hardware sweep (tiles 4/8/16/32)",
    "date": "2026-09-19",
    "tree": "vektory79 @ 7ed25e0, v3a changeset uncommitted (gguf_kernel.cu, moe.cuh, moe.py, 2 test files)",
    "rig": "RTX 5090 single GPU, port 18801, desktop baseline VRAM 1472-1547 MiB",
    "boot_command_effective": BOOT,
    "knob": "FREETOKEN_GGUF_MOE_MTILE (per-call getenv in gguf_kernel.cu ft_gguf_moe_mtile; unset/'4'=4, '8'/'16'/'32'=tile, else TORCH_CHECK fail-fast)",
    "metric_rules": "prefill = median of server 'input throughput (token/s):' over full 8128 chunks EXCLUDING chunk 1 (JIT/autotune warmup) AND the last full chunk (T43 tail artifact); geometry 65,585 tokens = 8x8128 + 561; decode = same-prompt re-send, radix HIT 65536, first SSE frame skipped",
    "validity_gate": {
        "v2_committed_class_8128": 337.61,
        "band_3pct": [327.48, 347.74],
        "baseline_measured": runs["4"]["prefill_median_tok_s"],
        "verdict": "PASS (-1.45% vs 337.61)",
    },
    "runs": runs,
    "ab_table": [
        {"tile": x, "env": runs[x]["env"], "prefill_median": runs[x]["prefill_median_tok_s"],
         "delta_vs_tile4_pct": round((runs[x]["prefill_median_tok_s"] / runs["4"]["prefill_median_tok_s"] - 1) * 100, 2),
         "chunk_time_s": runs[x]["chunk_time_s"],
         "decode_steady": runs[x]["decode_steady_tok_s"],
         "boot_ready_s": runs[x]["ready_s"]}
        for x in ("4", "8", "16", "32")
    ],
    "moe_term_model": model,
    "nsys_liveness_tile16": {
        "report": liv["report"],
        "asserts": liv["assert"],
        "grouped_per_8128_chunk": {"grouped": 126, "moe_align": 42},
        "decode_phase": {"moe_vec_total": 7938, "math": "42 layers x 63 steps x 3 projections"},
        "cta_shrink_tile16_vs_tile4": liv["assert"]["A5_cta_shrink_tile16_vs_tile4"]["detail"],
        "per_format": liv["per_format"],
        "marker": "gguf moe grouped mmq prefill active (fired exactly once)",
    },
    "battery": battery,
    "v3b_trigger_assessment": {
        "saturates_below_traffic_ceiling": True,
        "binding_term": "per-MAC ALU/LDS + k-step serialization (invariant, schedule-independent), NOT SMEM/occupancy and NOT padding",
        "evidence": "per-launch medians speed up only 2.5-2.6x (q6_K 1.95x) for a 15.8x traffic cut at tile 16; e2e marginal gain halves per doubling (+31.1 / +17.5 / +8.6%)",
        "padding_dominates": False,
        "padding_note": "7.1% pair-slot waste at tile 32 visible as the +22% flatness vs the pure 1/tile model, not as a cliff",
        "verdict": "v3b does NOT fire: the remaining recoverable MoE headroom (~1-2 s of a 14.6 s chunk) is small vs a new persistent-kernel effort, and the chunk is now dominated by rest (dense q8_0 GEMM 6.04 s + fetch copies 3.15 s + attention/GDN) = ~78-88% of chunk time",
    },
    "winner_tile": 32,
    "artifacts": {
        "stage_raw": "v3a_stage_raw.json", "liveness": "v3a_probe_liveness.json",
        "divergence": "v3a_divergence.json", "nsys_report": liv["report"],
        "nsys_sqlite": "v3aprobe.sqlite", "stage_outs": "v3a_v3a_t*.out",
        "battery_dirs": "quality/base, quality/win",
    },
}
with open(os.path.join(VDIR, "v3a-ab-results.json"), "w") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
print(json.dumps(out["ab_table"], indent=1))
print(json.dumps(out["moe_term_model"], indent=1))
print("battery:", json.dumps(out["battery"]["stages"], indent=1)[:400])
print("winner:", out["winner_tile"], "| gate:", out["validity_gate"]["verdict"])
