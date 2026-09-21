"""Wave-1 post-verification: inspect the converted FTW dir (read-only).

Reports: file list, shard total, index meta (quant_format, gguf_types, counts,
expert_bank_num_layers), per-layer experts_bank entries, sample shapes, tmp/partial leftovers.
"""
import json
import os
import re
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else (
    "/media/ai/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/"
    "GLM-5.3-Flash-UD-Q3_K_XL-FTW"
)


def brief(x, n=300):
    s = json.dumps(x) if not isinstance(x, str) else x
    return s if len(s) <= n else s[:n] + f"...(+{len(s) - n} chars)"


with open(os.path.join(OUT, "freetoken_weight.json")) as f:
    idx = json.load(f)

print("== files in OUT ==")
for name in sorted(os.listdir(OUT)):
    p = os.path.join(OUT, name)
    sz = os.path.getsize(p) if os.path.isfile(p) else -1
    print(f"  {name}  {sz}")

shards = idx.get("shards", [])
total = sum(s["nbytes"] for s in shards)
print("== index meta ==")
print("format:", idx.get("format"), "version:", idx.get("version"), "align:", idx.get("align"))
print("shard_limit:", idx.get("shard_limit"), "shards:", len(shards), "total_bytes:", total)
print("counts:", idx.get("counts"))
print("expert_bank_num_layers:", idx.get("expert_bank_num_layers"))
print("quant_format:", idx.get("quant_format"))
print("fingerprint:", idx.get("fingerprint"))
print("source_model_path:", idx.get("source_model_path"))
print("copied_metadata:", brief(idx.get("copied_metadata")))
print("side_files:", brief(idx.get("side_files")))

gt = idx.get("gguf_types")
print("gguf_types:", "None" if gt is None else f"type={type(gt).__name__} len={len(gt)}")
if isinstance(gt, dict):
    for k, v in list(gt.items())[:4]:
        print(f"  gguf_types[{k!r}]: len={len(v) if hasattr(v, '__len__') else '?'} "
              f"sample={brief(v)}")
elif isinstance(gt, list):
    print("  gguf_types[0]:", brief(gt[0]))
    print("  gguf_types[-1]:", brief(gt[-1]))

tensors = idx.get("tensors", [])
kinds = {}
for t in tensors:
    kinds[t["kind"]] = kinds.get(t["kind"], 0) + 1
print("== tensors ==")
print("total entries:", len(tensors), "by kind:", kinds)

layer_re = re.compile(r"^(?P<base>.+)#L(?P<layer>\d{5})$")
layers_by_base = {}
sample = {}
for t in tensors:
    if t.get("kind") != "experts_bank":
        continue
    m = layer_re.match(t["name"])
    if m:
        base = m.group("base")
        layers_by_base.setdefault(base, []).append(int(m.group("layer")))
        sample.setdefault(base, t)
    else:
        print("  non-layer experts_bank entry:", t["name"], "shape:", t["shape"],
              "dtype:", t["dtype"], "nbytes:", t["nbytes"])

for base, lays in sorted(layers_by_base.items()):
    lays.sort()
    t = sample[base]
    shape = t["shape"]
    print(f"  base={base!r}: {len(lays)} layers, ids {lays[0]}..{lays[-1]}, "
          f"contiguous={lays == list(range(lays[0], lays[-1] + 1))}")
    print(f"    sample {t['name']!r}: shape={shape} dtype={t['dtype']} nbytes={t['nbytes']}")
    if len(shape) == 3:
        ne, rows, row_bytes = shape
        calc = ne * rows * row_bytes
        print(f"    [num_experts={ne}, rows={rows}, row_bytes={row_bytes}] "
              f"numel check: {calc} == {t['nbytes']} -> {calc == t['nbytes']}")

leftovers = [n for n in os.listdir(OUT) if n.endswith(".tmp") or n.endswith(".partial")]
print("== tmp/partial leftovers ==", leftovers or "none")
