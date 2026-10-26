"""Numerical verification: every mmproj tower tensor vs the NVFP4 reference (CPU-only).

Reads the operator's mmproj GGUF through the main tower reader
(freetoken.models.glm5_next.gguf.iter_mmproj_weights) and the bf16 NVFP4 FTW
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
import json
import os
from collections import Counter


def _iter_ftw_visual_raw(reference: str, keep):
    """Raw fallback when FTWReader rejects the reference: a pre-0db5023 conversion
    stores NVFP4 global scales as fp16 and the reader fail-fasts on the index, but
    the bf16 visual.* tower bytes are unaffected by that scale-dtype flip. Reads
    each tensor straight from the index byte ranges across the raw shards, which
    for plain bf16 weight entries is exactly what iter_ftw_weights does (its
    _replay_entry / _pad_legacy_rows are no-ops for such entries)."""
    import torch

    from freetoken.checkpoint.ftw import INDEX_NAME

    with open(os.path.join(reference, INDEX_NAME)) as f:
        index = json.load(f)
    shards = sorted(index["shards"], key=lambda s: s["global_off"])
    for entry in index["tensors"]:
        if entry.get("kind") != "weight" or not keep(entry["name"]):
            continue
        start, nbytes = entry["global_off"], entry["nbytes"]
        pieces = []
        for shard in shards:
            lo = max(start, shard["global_off"])
            hi = min(start + nbytes, shard["global_off"] + shard["nbytes"])
            if lo >= hi:
                continue
            with open(os.path.join(reference, shard["file"]), "rb") as f:
                f.seek(lo - shard["global_off"])
                pieces.append(f.read(hi - lo))
        raw = b"".join(pieces)
        if len(raw) != nbytes:
            raise OSError(f"{reference}: {entry['name']}: read {len(raw)}/{nbytes} bytes")
        dt = getattr(torch, entry["dtype"])
        tensor = torch.frombuffer(bytearray(raw), dtype=dt)
        yield entry["name"], (tensor.view(*entry["shape"]) if entry["shape"] else tensor.view(())).clone()


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
    from freetoken.models.gguf.mmproj import Mmproj
    from freetoken.models.gguf.reader import gguf_tensor_index, load_gguf_metadata
    from freetoken.models.glm5_next.gguf import _BLOCK, _PATCH, _TOP, iter_mmproj_weights

    source_dtype = {
        t.name: ("F32" if t.ggml_type == 0 else "BF16" if t.ggml_type == 30 else str(t.ggml_type))
        for t in gguf_tensor_index(args.mmproj).values()
    }
    # reverse the reader's mapping (mmproj source name -> visual param): the report
    # is param-level (visual.*), so the source dtype must be looked up through it
    depth = int(load_gguf_metadata(args.mmproj)["clip.vision.block_count"])
    src_of: dict[str, list[str]] = {}
    for suffix, rel in _BLOCK.items():
        for layer in range(depth):
            src_of.setdefault(f"visual.blocks.{layer}.{rel}", []).append(f"v.blk.{layer}.{suffix}")
    for mmproj_name, visual in _TOP.items():
        src_of.setdefault(f"visual.{visual}", []).append(mmproj_name)
    # the two patch-embedding slices stack into one param: carry both source dtypes
    src_of["visual.patch_embed.proj.weight"] = list(_PATCH)
    src_of["visual.patch_embed.proj.bias"] = ["v.patch_embd.bias"]
    src_of["visual.post_layernorm.weight"] = ["v.post_ln.weight"]

    def src_column(name: str) -> str:
        dtypes = sorted({source_dtype[s] for s in src_of.get(name, []) if s in source_dtype})
        return "/".join(dtypes) if dtypes else "-"

    got = dict(iter_mmproj_weights(Mmproj(args.mmproj, "verify", "glm5v", "BF16")))
    ref_note = ""
    try:
        ref = {
            name: tensor.clone()
            for name, tensor in iter_ftw_weights(
                args.reference, keep=lambda n: n.startswith("visual.")
            )
        }
    except ValueError as exc:
        if "NVFP4 global scales as fp16" not in str(exc):
            raise
        ref_note = " (raw index+shard reads: pre-0db5023 fp16-scale carrier, tower bytes unaffected)"
        ref = {
            name: tensor.clone()
            for name, tensor in _iter_ftw_visual_raw(
                args.reference, lambda n: n.startswith("visual.")
            )
        }

    rows: list[dict] = []
    for name in sorted(set(got) | set(ref)):
        row = {"name": name, "src": src_column(name)}
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
        f"- reference: `{args.reference}` (visual.* tensors, bf16){ref_note}",
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
