#!/usr/bin/env python3
"""One serial arm of the decode research campaign (hybrid GGUF glm5next).

Boots ft serve with the user's production config + per-arm overrides, waits for
true readiness, fills a 64k prefix, re-sends the SAME prompt to measure radix-
reused decode, samples per-CPU busy/MHz during the decode window, tears down
(SIGTERM, grace for the tier flush, then SIGKILL), verifies a clean census.

Usage: run_arm.py --name <name> --log <file.log> --out <file.json>
                  [--extra "<ft serve flags>"] [--env KEY=V ...]
"""
import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import urllib.request

REPO = "/media/ai/src/FreeToken"
DIR = os.path.join(REPO, ".tasks", "decode-research")
FT = os.path.join(REPO, ".venv", "bin", "ft")
FILLER = os.path.join(REPO, ".tasks", "ft-gguf-serve-tuning", "filler.txt")
MODEL = "GLM-5.3-Flash-UD-Q3_K_XL.gguf"
PORT = 18801
URL = "http://127.0.0.1:%d" % PORT
READY_TIMEOUT = 420
FILL_TIMEOUT = 900
DECODE_TIMEOUT = 600
# Normal teardown lines (backend worker exit, tier flush) must NOT trip the watchdog;
# only genuine failures do (arbitration_run.py lesson).
WATCH = re.compile(r"AssertionError|OutOfMemoryError|CUDA error|Traceback \(most recent call last\)")

BASE_ARGS = [
    "--enable-cache-report",
    "--model-path", "/media/ai/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL-FTW/",
    "--moe-cache-auto",
    "--kv-reserve-tokens", "400000",
    "--kv-cache-dtype", "fp8",
    "--memory-ratio", "0.82",
    "--max-prefill-length", "8191",
    "--moe-strategy", "hybrid",
    "--moe-cpu-threads", "16",
    "--max-running-requests", "1",
    "--session-tier-ram-gib", "10",
    "--session-tier-dir", "/media/ai/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/cache",
    "--session-tier-ssd-gib", "50",
]

KEY_PATTERNS = [
    r"moe_cache_size|num_pages|Allocating .* KV|Allocating .*tokens",
    r"fetching .*over PCIe|hybrid_max_fetch|fetch fraction",
    r"pool .*core|affinit|CPU MoE executor|isa=",
    r"input throughput",
    r"new-token|cached-token",
    r"ready in|Ready in|Uvicorn running",
]


def log(msg):
    print("[arm] %s" % msg, flush=True)


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def census_serve():
    out = sh("pgrep -af '[f]t serve'").stdout.strip()
    return [l for l in out.splitlines() if l]


def gpu_used_mib():
    out = sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits").stdout.strip()
    try:
        return int(out.splitlines()[0])
    except Exception:
        return -1


class CpuSampler(threading.Thread):
    """1 Hz per-CPU busy delta + current MHz, written as CSV."""

    def __init__(self, path):
        super().__init__(daemon=True)
        self.path = path
        self.stop_evt = threading.Event()

    @staticmethod
    def _read():
        busy = []
        mhz = []
        with open("/proc/stat") as f:
            for line in f:
                if not line.startswith("cpu"):
                    break
                parts = line.split()
                if parts[0] == "cpu":
                    continue
                vals = list(map(int, parts[1:]))
                # user nice system irq softirq steal = busy; iowait counted separately
                busy.append(sum(vals[:3]) + sum(vals[5:8]))
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("cpu MHz"):
                    mhz.append(float(line.split(":", 1)[1]))
        return busy, mhz

    def run(self):
        prev, _ = self._read()
        with open(self.path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "cpu", "busy_delta", "mhz"])
            while not self.stop_evt.is_set():
                time.sleep(1.0)
                cur, mhz = self._read()
                t = time.time()
                rows = []
                for c, (p, q) in enumerate(zip(prev, cur)):
                    rows.append((t, c, q - p, mhz[c] if c < len(mhz) else -1))
                prev = cur
                for r in rows:
                    w.writerow(r)
                f.flush()


def start_server(extra_args, logpath, env_pairs):
    for p in census_serve():
        log("FATAL: live ft serve census: %s" % p)
        sys.exit(1)
    cmd = [FT, "serve", "--port", str(PORT), "--host", "127.0.0.1"] + BASE_ARGS + list(extra_args)
    env = dict(os.environ)
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


def chat_request(content, max_tokens, stream, timeout, logpath):
    """Returns (dict result, new_log_lines)."""
    rq = {"model": MODEL, "messages": [{"role": "user", "content": content}],
          "max_tokens": max_tokens, "temperature": 0, "top_k": 1, "stream": stream}
    if stream:
        rq["stream_options"] = {"include_usage": True}
    off = os.path.getsize(logpath)
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
    wall = time.time() - t0
    lines = new_log_lines(logpath, off)
    return {"wall_s": round(wall, 2), "error": err, "usage": usage,
            "chunks": len(times), "times": times, "lines": lines}


def new_log_lines(logpath, offset):
    try:
        with open(logpath, "rb") as f:
            f.seek(offset)
            return f.read().decode("utf-8", "replace").splitlines()
    except Exception:
        return []


def parse_tok(lines, kind):
    vals = []
    for l in lines:
        m = re.search(r"#%s:\s*(\d+)" % kind, l)
        if m:
            vals.append(int(m.group(1)))
    return vals[-1] if vals else None


def decode_metrics(res, ntok):
    """Steady rate from SSE frame deltas, first frame excluded (pre-prefill frame)."""
    times = res["times"]
    out = {"wall_s": res["wall_s"], "error": res["error"], "completion_tokens": ntok,
           "chunks": len(times)}
    if len(times) >= 4:
        tail = times[1:]
        if tail[-1] > tail[0]:
            out["steady_tok_s"] = round((len(tail) - 1) / (tail[-1] - tail[0]), 2)
        out["ttft_s"] = round(times[0] - (res["t0"] if "t0" in res else times[0]), 2)
        out["first_frame_to_last_s"] = round(tail[-1] - tail[0], 2)
        gaps = sorted(tail[i + 1] - tail[i] for i in range(len(tail) - 1))
        if gaps:
            out["p50_step_ms"] = round(1000 * gaps[len(gaps) // 2], 1)
            out["max_step_ms"] = round(1000 * gaps[-1], 1)
    return out


def grep_log(logpath):
    try:
        txt = open(logpath, encoding="utf-8", errors="replace").read()
    except Exception:
        return {}
    res = {}
    for pat in KEY_PATTERNS:
        hits = [l.strip() for l in txt.splitlines() if re.search(pat, l)]
        res[pat] = hits[-40:]
    return res


def teardown(proc, grace=45):
    sh("pkill -TERM -f '[f]t serve'")
    deadline = time.time() + grace
    while time.time() < deadline:
        if proc.poll() is not None and not census_serve():
            break
        time.sleep(0.5)
    if proc.poll() is None or census_serve():
        sh("pkill -KILL -f '[f]t serve'")
        time.sleep(3)
    try:
        proc.wait(timeout=5)
    except Exception:
        pass
    time.sleep(2)
    return {"leftover": census_serve(), "gpu_mib": gpu_used_mib()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra", default="")
    ap.add_argument("--env", action="append", default=[])
    a = ap.parse_args()

    logpath = os.path.join(DIR, a.log)
    content = open(FILLER, encoding="utf-8").read()
    out = {"name": a.name, "extra": a.extra, "env": a.env,
           "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    proc, lf = start_server(shlex.split(a.extra), logpath, a.env)
    ok, ready_s, why = wait_ready(logpath, proc)
    out["ready"] = ok
    out["ready_s"] = round(ready_s, 1)
    out["ready_why"] = why
    out["gpu_after_boot_mib"] = gpu_used_mib()
    if not ok:
        out["log_tail"] = "\n".join(new_log_lines(logpath, max(0, os.path.getsize(logpath) - 4000)))
        out["teardown"] = teardown(proc)
        print(json.dumps(out, indent=1, ensure_ascii=False))
        sys.exit(2)

    # warmup fill: 64k prefix, 4 tokens, radix-caches the prompt
    fill = chat_request(content, 4, False, FILL_TIMEOUT, logpath)
    out["fill"] = {"wall_s": fill["wall_s"], "error": fill["error"],
                   "prompt_tokens": fill["usage"].get("prompt_tokens"),
                   "cached_token": parse_tok(fill["lines"], "cached-token"),
                   "new_token": parse_tok(fill["lines"], "new-token"),
                   "throughput_lines": [l.strip() for l in fill["lines"] if "input throughput" in l]}
    if fill["error"]:
        out["log_tail"] = "\n".join(fill["lines"][-60:])
        out["teardown"] = teardown(proc)
        print(json.dumps(out, indent=1, ensure_ascii=False))
        sys.exit(3)

    # decode: SAME prompt -> radix hit, stream 768 tokens
    sampler = CpuSampler(os.path.join(DIR, "cpu_%s.csv" % a.name))
    sampler.start()
    t0 = time.time()
    dec = chat_request(content, 768, True, DECODE_TIMEOUT, logpath)
    dec["t0"] = t0
    sampler.stop_evt.set()
    sampler.join()
    ntok = dec["usage"].get("completion_tokens") or dec["chunks"]
    out["decode"] = decode_metrics(dec, ntok)
    out["decode"]["cached_token"] = parse_tok(dec["lines"], "cached-token")
    out["decode"]["new_token"] = parse_tok(dec["lines"], "new-token")
    out["log"] = grep_log(logpath)
    out["teardown"] = teardown(proc)
    print(json.dumps(out, indent=1, ensure_ascii=False))
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
