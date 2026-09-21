#!/usr/bin/env python3
"""Dense-q80 step 0: real GGUF header scan (gguf-py) for the dense-side tensors.

Prints every non-routed-expert tensor grouped per pattern with dims + quant type
+ bytes, aggregates routed-expert types per block, and pulls a few header KV
pairs. Shapes here are authoritative for the traffic model; op-class mapping to
the glm5_next model definition is done in the step0 analyzer.

Output: denseq80_header.json + a printed table.
"""
import json
import os
from collections import defaultdict

import gguf

GGUF_PATH = "/media/ai/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL.gguf"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "denseq80_header.json")

r = gguf.GGUFReader(GGUF_PATH)

tensors = []
for t in r.tensors:
    tensors.append({
        "name": t.name,
        "shape_gguf": [int(d) for d in t.shape],
        "type": gguf.GGMLQuantizationType(t.tensor_type).name,
        "bytes": int(t.data.nbytes),
    })

probe = [x for x in tensors if x["name"].endswith("ffn_down.weight") or x["name"].endswith("ffn_down_shexp.weight")]
orient = "torch_rows_first" if any(x["shape_gguf"][0] == 4096 for x in probe) else "gguf_ne_first"

dense = [x for x in tensors if "_exps" not in x["name"]]
experts = [x for x in tensors if "_exps" in x["name"]]


def strip_name(n):
    parts = n.split(".", 2)
    if parts[0] == "blk" and len(parts) == 3 and parts[1].isdigit():
        return "*." + parts[2]
    return n


blocks = defaultdict(list)
other = []
for x in dense:
    if x["name"].startswith("blk."):
        blocks[int(x["name"].split(".")[1])].append(x)
    else:
        other.append(x)

expert_types_by_block = defaultdict(lambda: defaultdict(int))
expert_bytes_by_block = defaultdict(int)
for x in experts:
    parts = x["name"].split(".")
    bid = int(parts[1])
    pos = parts[2]
    expert_types_by_block[bid][pos + ":" + x["type"]] += 1
    expert_bytes_by_block[bid] += x["bytes"]

kv = {}
for field in ("general.architecture", "general.name", "general.file_type",
              "glm5.context_length"):
    for t in r.fields.values():
        if t.name == field:
            try:
                v = t.parts[-1]
                if v.size == 1:
                    iv = int(v[0])
                    kv[field] = iv
                else:
                    kv[field] = bytes(v).decode("utf-8", "replace")
            except Exception as e:
                kv[field] = "err:%s" % e

out = {
    "gguf_path": GGUF_PATH,
    "orientation": orient,
    "n_tensors": len(tensors),
    "n_expert_tensors": len(experts),
    "expert_type_counts_by_block": {str(b): dict(v) for b, v in sorted(expert_types_by_block.items())},
    "expert_bytes_by_block": {str(b): v for b, v in sorted(expert_bytes_by_block.items())},
    "non_expert_tensors": sorted(dense, key=lambda x: x["name"]),
    "other_tensors": sorted(other, key=lambda x: x["name"]),
    "header_kv": kv,
}
with open(OUT, "w") as f:
    json.dump(out, f, indent=1)

print("orientation probe:", orient, "| n_tensors:", len(tensors),
      "| n_expert_tensors:", len(experts))
print("\n=== general ===")
for k, v in kv.items():
    print(" ", k, "=", v)
print("\n=== expert (ffn_*_exps) type counts, first 3 blocks ===")
for b in sorted(expert_types_by_block)[:3]:
    print(" blk", b, dict(expert_types_by_block[b]))
b0 = min(expert_bytes_by_block)
print("expert bytes/block MB: min=%d max=%d nblocks=%d" % (
    min(expert_bytes_by_block.values()) / 1e6,
    max(expert_bytes_by_block.values()) / 1e6,
    len(expert_bytes_by_block)))

print("\n=== non-expert (dense-side) tensor patterns ===")
by_name = defaultdict(list)
for b in sorted(blocks):
    for x in blocks[b]:
        by_name[strip_name(x["name"])].append((b, x["shape_gguf"], x["type"], x["bytes"]))
for pat in sorted(by_name):
    rows = by_name[pat]
    bs = sorted(set(r[0] for r in rows))
    shapes = sorted(set((tuple(r[1]), r[2]) for r in rows))
    tot = sum(r[3] for r in rows)
    shape_str = "; ".join("%s %s" % (s, t) for s, t in shapes[:6])
    more = " (+%d more shapes)" % (len(shapes) - 6) if len(shapes) > 6 else ""
    print("%-44s n=%-5d blocks=%-9s  %s%s  sum %10.2f MB" % (
        pat, len(rows), (str(bs[0]) + "-" + str(bs[-1])) if len(bs) > 1 else str(bs[0]),
        shape_str, more, tot / 1e6))

print("\n=== other (non-block) tensors ===")
for x in sorted(other, key=lambda z: -z["bytes"])[:15]:
    print("%-52s %-10s %10.2f MB  %s" % (x["name"], x["type"], x["bytes"] / 1e6, x["shape_gguf"]))
print("\nwrote", OUT)
