#!/usr/bin/env python3
"""Wave-2 quality battery parity analysis (FTW boot vs bare-GGUF references).

Compares the 24 battery outputs produced by run-wave2-measure.py against the
bare-GGUF reference boots:
  - .tasks/gguf-glm5next-path-a/verification/phase6/battery-post-fix (path-a boot)
  - .tasks/dense-q80-gemm/quality-arbitration/{t4a,t64a,t64b} (dense-q80 boots;
    t64a/t64b = the current committed production defaults)

Metrics per prompt per reference: sha256 equality (same out_sha format as
arbitration), difflib ratio + longest common prefix/suffix over
reasoning+content, divergence class (identical / mid-answer / suffix-flip /
whole-answer). Bare-GGUF boot-to-boot output is bimodal (T54: same-config pairs
flip on most prompts), so "matches at least one bare-GGUF reference boot"
classifies as boot-noise-level parity; anything else is printed for eyeball.
"""
import difflib
import json
import os

ROOT = "/media/ai/src/FreeToken"
OUTDIR = os.path.join(ROOT, ".tasks/ftw-gguf-fastpath/run-wave2")
REFS = {
    "patha": os.path.join(ROOT, ".tasks/gguf-glm5next-path-a/verification/phase6/battery-post-fix"),
    "t4a": os.path.join(ROOT, ".tasks/dense-q80-gemm/quality-arbitration/t4a"),
    "t64a": os.path.join(ROOT, ".tasks/dense-q80-gemm/quality-arbitration/t64a"),
    "t64b": os.path.join(ROOT, ".tasks/dense-q80-gemm/quality-arbitration/t64b"),
}
FTWDIR = os.path.join(OUTDIR, "ftw")


def sha_of(art):
    import hashlib
    blob = json.dumps({"content": art.get("content") or "", "reasoning": art.get("reasoning") or "",
                       "finish_reason": art.get("finish_reason")},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def cmp_pair(a, b):
    ta = (a.get("reasoning") or "") + "\n" + (a.get("content") or "")
    tb = (b.get("reasoning") or "") + "\n" + (b.get("content") or "")
    if ta == tb:
        return {"kind": "identical", "ratio": 1.0, "lcp": len(ta), "lcs": len(ta), "len_a": len(ta), "len_b": len(tb)}
    lcp = len(os.path.commonprefix([ta, tb]))
    lcs = len(os.path.commonprefix([ta[::-1], tb[::-1]]))
    ratio = round(difflib.SequenceMatcher(None, ta, tb).ratio(), 3)
    if lcp == 0 and lcs == 0:
        kind = "whole-answer"
    elif lcs >= min(len(ta), len(tb)) // 2:
        kind = "suffix-flip"
    else:
        kind = "mid-answer"
    return {"kind": kind, "ratio": ratio, "lcp": lcp, "lcs": lcs,
            "len_a": len(ta), "len_b": len(tb)}


def main():
    out = {"per_prompt": [], "summary": {}}
    ref_shas = {}
    ref_arts = {}
    for rname, rdir in REFS.items():
        ref_shas[rname] = {}
        ref_arts[rname] = {}
        for i in range(24):
            p = os.path.join(rdir, "p%02d.json" % i)
            if os.path.exists(p):
                art = json.load(open(p))
                ref_arts[rname][i] = art
                ref_shas[rname][i] = sha_of(art)
    n_exact = n_noise = n_div = 0
    for i in range(24):
        fp = os.path.join(FTWDIR, "p%02d.json" % i)
        if not os.path.exists(fp):
            out["per_prompt"].append({"idx": i, "missing": True})
            continue
        art = json.load(open(fp))
        row = {"idx": i, "kind": art.get("kind"), "sha": (art.get("sha256") or "")[:8],
               "wall_s": art.get("wall_s"), "refs": {}}
        matches = []
        best = None
        for rname in REFS:
            if i not in ref_shas[rname]:
                continue
            c = cmp_pair(art, ref_arts[rname][i])
            c["sha_equal"] = ref_shas[rname][i] == art.get("sha256")
            c["ref_sha"] = ref_shas[rname][i][:8]
            row["refs"][rname] = c
            if c["sha_equal"]:
                matches.append(rname)
            if best is None or c["ratio"] > best[1]["ratio"]:
                best = (rname, c)
        if len(matches) == len(ref_arts):
            row["verdict"] = "identical-all-refs"
            n_exact += 1
        elif matches:
            row["verdict"] = "matches:%s" % ",".join(matches)
            n_noise += 1
        else:
            row["verdict"] = "divergent-from-all-refs"
            n_div += 1
        row["best_ref"] = {"name": best[0], **best[1]} if best else None
        out["per_prompt"].append(row)
        b = row.get("best_ref") or {}
        print("p%02d %-6s verdict=%-28s best=%s ratio=%s lcp=%s class=%s sha=%s" % (
            i, art.get("kind"), row["verdict"], b.get("name"), b.get("ratio"),
            b.get("lcp"), b.get("kind"), row["sha"]))
    out["summary"] = {"identical_all_refs": n_exact, "boot_noise_match": n_noise,
                      "divergent_from_all_refs": n_div}
    print("summary: %s" % json.dumps(out["summary"]))
    with open(os.path.join(OUTDIR, "wave2_parity.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
