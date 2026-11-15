#!/usr/bin/env python3
"""Coverage-curve trace arm (kb/cases/offline-coverage-curve, phase F1).

Boots ft serve with the CURRENT production config in EAGER mode and records
every (layer, expert) decode route via the decode-research T1 driver
(FREETOKEN_DR_ROUTES=1), alongside the usual stats dump.

Config anchor (TASK.md): GGUF GLM-5.3-Flash UD-Q3_K_XL (FTW-vision-new),
--moe-strategy hybrid, memory-ratio 0.81, kv-reserve 800000 fp8, session tier
10/50, cpu-threads 16. Chunk 4095 = the stable prefill family (the chunk does
not affect the decode trace). Eager mode (--cuda-graph-max-bs 0) is REQUIRED:
per-step routes exist only outside CUDA graphs; greedy decode (temperature 0,
top_k 1) keeps the routing sequence identical to a production boot.

Usage: run_trace.py [--tag trace] [--extra "<ft serve flags>"] [--env K=V ...]
Outputs in this dir: arm_<tag>.json, serve_<tag>.log and, from the driver,
routes_<tag>.json + stats_<tag>.json.
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.request

REPO = "/media/ai/src/FreeToken"
DIR = os.path.join(REPO, ".tasks", "offline-coverage-curve")
DRIVER_DIR = os.path.join(REPO, ".tasks", "decode-research", "driver")
FT = os.path.join(REPO, ".venv", "bin", "ft")
FILLER = os.path.join(REPO, ".tasks", "ft-gguf-serve-tuning", "filler.txt")
MODEL = "GLM-5.3-Flash-UD-Q3_K_XL.gguf"
PORT = 18801
URL = "http://127.0.0.1:%d" % PORT
READY_TIMEOUT = 420
FILL_TIMEOUT = 1200
DECODE_TIMEOUT = 1800
FILL_TOKENS = 4
DECODE_TOKENS = 768
MOE_LAYERS = 42
WATCH = re.compile(r"AssertionError|OutOfMemoryError|CUDA error|Traceback \(most recent call last\)")

BASE_ARGS = [
    "--enable-cache-report",
    "--model-path", "/media/ai/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL-FTW-vision-new/",
    "--moe-cache-auto",
    "--kv-reserve-tokens", "800000",
    "--kv-cache-dtype", "fp8",
    "--memory-ratio", "0.81",
    "--max-prefill-length", "4095",
    "--moe-strategy", "hybrid",
    "--moe-cpu-threads", "16",
    "--max-running-requests", "1",
    "--session-tier-ram-gib", "10",
    "--session-tier-dir", "/media/ai/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/cache",
    "--session-tier-ssd-gib", "50",
]

BOOT_PATTERNS = [
    ("resolved_plan", re.compile(r"--moe-cache-auto resolved moe_cache_size=(\d+)\s+num_pages=(\d+)")),
    ("kv_alloc", re.compile(r"Allocating (\d+) tokens for KV cache")),
    ("slot_pools", re.compile(r"MoE slot pools: .*")),
    ("fetch_auto", re.compile(r"--moe-hybrid-max-fetch auto.*")),
    ("fetch_warn", re.compile(r"--moe-hybrid-max-fetch auto: no `ft bench bw` row.*")),
    ("ready", re.compile(r"(?:ready in|Ready in|Uvicorn running).*")),
]


def log(msg):
    print("[trace] %s" % msg, flush=True)


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def census_serve():
    out = sh("pgrep -af '[f]t serve'").stdout.strip()
    return [line for line in out.splitlines() if line]


def gpu_used_mib():
    out = sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits").stdout.strip()
    try:
        return int(out.splitlines()[0])
    except Exception:
        return -1


def start_server(extra_args, logpath, env_pairs):
    for p in census_serve():
        log("FATAL: live ft serve census: %s" % p)
        sys.exit(1)
    cmd = [FT, "serve", "--port", str(PORT), "--host", "127.0.0.1"] + BASE_ARGS + list(extra_args)
    env = dict(os.environ)
    pp = DRIVER_DIR
    if env.get("PYTHONPATH"):
        pp = DRIVER_DIR + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pp
    for kv in env_pairs or []:
        k, _, v = kv.partition("=")
        env[k] = v
    lf = open(logpath, "w")
    proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=REPO, env=env)
    with open(logpath + ".pid", "w") as f:
        f.write(str(proc.pid))
    return proc, lf


def wait_ready(logpath, proc, timeout=READY_TIMEOUT):
    t0 = time.time()
    probe = {"model": MODEL, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1}
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False, time.time() - t0, "process exited rc=%s" % proc.returncode
        try:
            tail = open(logpath, "rb").read()[-12000:].decode("utf-8", "replace")
            m = WATCH.search(tail)
            if m:
                return False, time.time() - t0, "watchdog: %s" % m.group(0)
        except Exception:
            pass
        # true readiness = a tiny REAL chat request returning 200 (/v1/models and
        # /ready answer before the engine finishes loading)
        try:
            req = urllib.request.Request(URL + "/v1/chat/completions",
                                         data=json.dumps(probe).encode(),
                                         headers={"Content-Type": "application/json"})
            r = urllib.request.urlopen(req, timeout=5)
            r.read()
            r.close()
            return True, time.time() - t0, "ok"
        except Exception:
            pass
        time.sleep(3)
    return False, time.time() - t0, "boot timeout"


def chat_request(content, max_tokens, stream, timeout, ignore_eos=False):
    rq = {"model": MODEL, "messages": [{"role": "user", "content": content}],
          "max_tokens": max_tokens, "temperature": 0, "top_k": 1, "stream": stream,
          "ignore_eos": ignore_eos}
    if stream:
        rq["stream_options"] = {"include_usage": True}
    t0 = time.time()
    err = None
    times = []
    usage = {}
    try:
        req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(rq).encode(),
                                     headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=timeout)
        if stream:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except Exception:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                if obj.get("choices"):
                    times.append(time.time())
            resp.close()
        else:
            body = json.loads(resp.read())
            resp.close()
            usage = body.get("usage") or {}
    except Exception as e:
        err = "%s: %s" % (type(e).__name__, e)
    return {"wall_s": round(time.time() - t0, 2), "error": err, "usage": usage, "chunks": len(times)}


def boot_keys(logpath):
    keys = {}
    try:
        txt = open(logpath, encoding="utf-8", errors="replace").read()
    except Exception:
        return keys
    for name, pat in BOOT_PATTERNS:
        m = pat.search(txt)
        if not m:
            keys[name] = None
        elif name == "resolved_plan":
            keys[name] = {"moe_cache_size": int(m.group(1)), "num_pages": int(m.group(2))}
        elif name == "kv_alloc":
            keys[name] = int(m.group(1))
        else:
            keys[name] = m.group(0).strip()
    return keys


def teardown(proc, grace=45):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
    time.sleep(2)
    return {"rc": proc.returncode, "left_ft_serve": census_serve()}


def check_routes(tag, expect_layers=MOE_LAYERS, expect_calls=None):
    path = os.path.join(DIR, "routes_%s.json" % tag)
    if not os.path.exists(path):
        return {"ok": False, "why": "missing %s" % path}
    try:
        doc = json.load(open(path))
    except Exception as e:
        return {"ok": False, "why": "unreadable: %r" % (e,)}
    n = doc.get("n_calls", 0)
    meta = (doc.get("meta") or [{}])[0]
    if doc.get("n_replay_steps", -1) != 0:
        return {"ok": False, "why": "n_replay_steps=%s (boot was not eager)" % doc.get("n_replay_steps"), "n_calls": n}
    if expect_calls and n < expect_calls:
        return {"ok": False, "why": "n_calls=%d < expected %d" % (n, expect_calls), "n_calls": n}
    if not meta.get("pool_sizes") or not meta.get("slot_bytes_per_layer"):
        return {"ok": False, "why": "meta incomplete: pool_sizes/slot_bytes_per_layer missing", "n_calls": n}
    return {"ok": True, "n_calls": n, "steps": n // expect_layers,
            "pool_sizes": meta["pool_sizes"], "cache_size": meta.get("cache_size"),
            "decode_target": meta.get("decode_target"),
            "fetch_fractions": sorted(set(f for f in (meta.get("hybrid_fetch_fraction") or []) if f)),
            "max_fetch": meta.get("hybrid_max_fetch"),
            "locked": meta.get("locked_hybrid"), "unpinned": meta.get("unpinned_layers")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="trace")
    ap.add_argument("--extra", default="--cuda-graph-max-bs 0")
    ap.add_argument("--env", action="append", default=[])
    a = ap.parse_args()

    logpath = os.path.join(DIR, "serve_%s.log" % a.tag)
    content = open(FILLER, encoding="utf-8").read()
    out = {"tag": a.tag, "extra": a.extra, "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    env_pairs = a.env + [
        "FREETOKEN_DECODE_RESEARCH=%s" % DIR,
        "FREETOKEN_DR_TAG=%s" % a.tag,
        "FREETOKEN_DR_ROUTES=1",
    ]
    proc, lf = start_server(shlex.split(a.extra), logpath, env_pairs)
    ok, ready_s, why = wait_ready(logpath, proc)
    out["ready"] = ok
    out["ready_s"] = round(ready_s, 1)
    out["ready_why"] = why
    out["gpu_after_boot_mib"] = gpu_used_mib()
    if not ok:
        out["boot_keys"] = boot_keys(logpath)
        out["log_tail"] = open(logpath, "rb").read()[-4000:].decode("utf-8", "replace")
        out["teardown"] = teardown(proc)
        print(json.dumps(out, indent=1, ensure_ascii=False))
        json.dump(out, open(os.path.join(DIR, "arm_%s.json" % a.tag), "w"), indent=1, ensure_ascii=False)
        sys.exit(2)

    out["boot_keys"] = boot_keys(logpath)
    log("ready in %ss; plan=%s" % (out["ready_s"], out["boot_keys"].get("resolved_plan")))

    # warmup fill: 64k prefix, 4 tokens, radix-caches the prompt
    fill = chat_request(content, FILL_TOKENS, False, FILL_TIMEOUT)
    out["fill"] = {"wall_s": fill["wall_s"], "error": fill["error"],
                   "prompt_tokens": fill["usage"].get("prompt_tokens"),
                   "completion_tokens": fill["usage"].get("completion_tokens")}
    if fill["error"]:
        out["log_tail"] = open(logpath, "rb").read()[-4000:].decode("utf-8", "replace")
        out["teardown"] = teardown(proc)
        print(json.dumps(out, indent=1, ensure_ascii=False))
        json.dump(out, open(os.path.join(DIR, "arm_%s.json" % a.tag), "w"), indent=1, ensure_ascii=False)
        sys.exit(3)

    # decode: SAME prompt -> radix hit, greedy stream 768 tokens (routes captured by
    # the driver); ignore_eos pins the decode length (greedy EOS fired at 545 on the
    # first attempt; the harness observed 467..763 across boots on the same prompt)
    t0 = time.time()
    dec = chat_request(content, DECODE_TOKENS, True, DECODE_TIMEOUT, ignore_eos=True)
    dec["t0"] = t0
    ntok = dec["usage"].get("completion_tokens") or dec["chunks"]
    out["decode"] = {"wall_s": dec["wall_s"], "error": dec["error"],
                     "completion_tokens": ntok, "chunks": dec["chunks"]}
    if dec["error"] or ntok < DECODE_TOKENS:
        out["decode"]["suspect"] = "error or short decode"
    out["teardown"] = teardown(proc)

    out["routes"] = check_routes(a.tag, expect_calls=MOE_LAYERS * DECODE_TOKENS)
    print(json.dumps(out, indent=1, ensure_ascii=False))
    json.dump(out, open(os.path.join(DIR, "arm_%s.json" % a.tag), "w"), indent=1, ensure_ascii=False)
    if not out["routes"]["ok"]:
        sys.exit(5)
    if out["decode"].get("error") or ntok < DECODE_TOKENS:
        sys.exit(4)


if __name__ == "__main__":
    main()
