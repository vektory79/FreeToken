#!/usr/bin/env python3
"""Phase 6 post-fix battery: replay the 24 NVFP4-baseline prompts on the fixed GGUF
tree. Greedy (temp0/topk1), non-stream; decoded reasoning+content saved per prompt
in the same schema as battery_nvfp4/pXX.json."""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18081)
    ap.add_argument("--baseline-dir", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    with urllib.request.urlopen(base + "/v1/models", timeout=30) as r:
        model = json.loads(r.read().decode())["data"][0]["id"]
    print(f"[battery] served model: {model}", flush=True)

    failed = 0
    for i in range(24):
        with open(f"{args.baseline_dir}/p{i:02d}.json") as f:
            base_art = json.load(f)
        prompt = base_art["prompt"]
        mt = base_art["max_tokens"]
        payload = {"model": model,
                   "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0, "top_k": 1, "max_tokens": mt, "stream": False}
        t0 = time.time()
        try:
            req = urllib.request.Request(base + "/v1/chat/completions",
                                         data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                resp = json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:400]
            print(f"[battery] p{i:02d} HTTP {e.code}: {body}", flush=True)
            failed += 1
            continue
        except Exception as e:
            print(f"[battery] p{i:02d} FAILED: {type(e).__name__}: {e}", flush=True)
            failed += 1
            continue
        wall = time.time() - t0
        ch = (resp.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        art = {
            "idx": base_art["idx"], "kind": base_art["kind"], "prompt": prompt,
            "max_tokens": mt, "greedy_mode": "temp0_topk1",
            "reasoning": msg.get("reasoning_content") or "",
            "content": msg.get("content") or "",
            "finish_reason": ch.get("finish_reason"),
            "usage": resp.get("usage"), "wall_s": round(wall, 2),
        }
        with open(f"{args.outdir}/p{i:02d}.json", "w") as f:
            json.dump(art, f, indent=1, ensure_ascii=False)
        print(f"[battery] p{i:02d} {art['kind']} wall={wall:.1f}s fin={art['finish_reason']} "
              f"reaso={len(art['reasoning'])}ch cont={len(art['content'])}ch "
              f"head={art['reasoning'][:70]!r}", flush=True)
    print(f"[battery] done, failed={failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
