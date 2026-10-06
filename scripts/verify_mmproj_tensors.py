"""Numerical verification: every mmproj tower tensor vs the NVFP4 reference (CPU-only).

Reads the operator's mmproj GGUF through the phase-2 vision reader
(freetoken.models.glm5_next.gguf.iter_gguf_vision_weights) and the bf16 NVFP4 FTW
reference (RedHatAI/GLM-5.3-Flash-NVFP4-FTW) through iter_ftw_weights, then emits
a per-tensor max_abs_diff table as markdown. Both sides are compared in fp32;
the source ggml dtype (F32 / BF16) is carried per row so systematic casts are
visible. Nothing here touches VRAM.

Usage:
    python scripts/verify_mmproj_tensors.py \
        --mmproj /media/ai/models/unsloth/GLM-5.3-Flash-GGUF/mmproj-BF16.gguf \
        --reference /media/ai/models/RedHatAI/GLM-5.3-Flash-NVFP4-FTW \
        --out kb/cases/gguf-glm5next-vision/tensor-verification-report.md
"""

from __future__ import annotations

import argparse
from collections import Counter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mmproj", required=True)
    parser.add_argument("--reference", required=True, help="NVFP4 FTW checkpoint dir")
    parser.add_argument("--out", required=True, help="markdown report path")
    parser.add_argument(
        "--threshold", type=float, default=1e-2,
        help="bf16-noise ceiling: rows above it are flagged for mapping re-check",
    )
    args = parser.parse_args()

    from freetoken.checkpoint.ftw import iter_ftw_weights
    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.models.glm5_next.gguf import iter_gguf_vision_weights

    source_dtype = {
        t.name: ("F32" if t.ggml_type == 0 else "BF16" if t.ggml_type == 30 else str(t.ggml_type))
        for t in iter_gguf_tensors(args.mmproj)
    }

    got = dict(iter_gguf_vision_weights(args.mmproj))
    ref = {
        name: tensor.clone()
        for name, tensor in iter_ftw_weights(
            args.reference, keep=lambda n: n.startswith("visual.")
        )
    }

    rows: list[dict] = []
    for name in sorted(set(got) | set(ref)):
        row = {"name": name, "src": source_dtype.get(name, "-")}
        if name not in got or name not in ref:
            row["status"] = "UNMATCHED"
            row["diff"] = float("nan")
            row["shape"] = tuple(got[name].shape) if name in got else tuple(ref[name].shape)
            rows.append(row)
            continue
        a, b = got[name], ref[name]
        row["shape"] = tuple(a.shape)
        if a.shape != tuple(b.shape):
            row["status"] = "SHAPE-MISMATCH"
            row["diff"] = float("nan")
            rows.append(row)
            continue
        diff = (a.float() - b.float()).abs().max().item()
        row["diff"] = diff
        row["status"] = "exact" if diff == 0.0 else ("le-thr" if diff <= args.threshold else "ABOVE")
        rows.append(row)

    counts = Counter(r["status"] for r in rows)
    diffs = sorted(r["diff"] for r in rows if r["diff"] == r["diff"])

    def pct(p: float) -> float:
        return diffs[min(len(diffs) - 1, int(p * (len(diffs) - 1)))] if diffs else float("nan")

    lines = [
        "# mmproj tensor verification: GGUF reader vs NVFP4 reference",
        "",
        f"- mmproj: `{args.mmproj}`",
        f"- reference: `{args.reference}` (visual.* tensors, bf16)",
        f"- tensors: {len(rows)} (mmproj-side names {len(got)}, reference-side {len(ref)})",
        "- the table is param-level: the mmproj file holds 348 tensors, its two",
        "  v.patch_embd.weight slices stack into the single visual.patch_embed.proj.weight",
        f"- exact (max_abs_diff == 0): {counts['exact']}",
        f"- <= {args.threshold:g}: {counts['le-thr']}",
        f"- ABOVE threshold: {counts['ABOVE']}",
        f"- unmatched (no counterpart): {counts['UNMATCHED']}",
        f"- shape mismatches: {counts['SHAPE-MISMATCH']}",
        "",
        "max_abs_diff percentiles (fp32 compare of both bf16 sides):",
        "",
        "| p50 | p90 | p99 | max |",
        "|---|---|---|---|",
        (
            f"| {pct(0.5):.3e} | {pct(0.9):.3e} | {pct(0.99):.3e} | {diffs[-1]:.3e} |"
            if diffs
            else "| - | - | - | - |"
        ),
        "",
        "Threshold note: the reference tower is bf16; F32-sourced mmproj tensors cast to",
        "bf16 exactly like the reference conversion did, so anything above the threshold",
        "is a mapping suspect (axis order / transpose / segment order), not cast noise.",
        "",
        "| tensor | source dtype | shape | max_abs_diff | status |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['src']} | {r['shape']} | {r['diff']:.3e} | {r['status']} |"
        )
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}: {dict(counts)}")


if __name__ == "__main__":
    main()
