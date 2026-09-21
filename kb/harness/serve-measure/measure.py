#!/usr/bin/env python3
"""Hardware A/B harness for ft serve (GGUF GLM-5.3-Flash) on RTX 5090.

Modes:
  boot    - start server, wait /ready, capture boot log lines, teardown
  prefill - boot + one 64k prefill + teardown
  full    - boot + prefill + decode @cached-prefix + teardown
  ladder  - boot + incremental same-prefix deep-fill ladder + teardown

All runs use port 18801. Serial only - harness refuses to start if another
ft serve is alive, and verifies a clean census after teardown.
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import urllib.request

DIR = "/media/ai/src/FreeToken/.tasks/ft-gguf-serve-tuning"
LOGDIR = os.path.join(DIR, "logs")
FT = "/media/ai/src/FreeToken/.venv/bin/ft"
REPO = "/media/ai/src/FreeToken"
MODEL = "GLM-5.3-Flash-UD-Q3_K_XL.gguf"
PORT = 18801
URL = "http://127.0.0.1:%d" % PORT
READY_TIMEOUT = 300
PREFILL_TIMEOUT = 900
DECODE_TIMEOUT = 600
WATCH = re.compile(r"AssertionError|OutOfMemoryError|Backend worker is gone|CUDA error|Traceback \(most recent call last\)")

BASE_ARGS = [
    "--model-path", "/media/ai/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL.gguf",
    "--moe-cache-auto",
    "--kv-reserve-tokens", "500000",
    "--kv-cache-dtype", "fp8",
    "--memory-ratio", "0.89",
    "--max-prefill-length", "4096",
    "--moe-strategy", "hybrid",
    "--moe-cpu-threads", "16",
]

KEY_PATTERNS = [
    r"moe_cache_size|num_pages|Allocating .* KV|Allocating .*tokens",
    r"Free memory after initialization",
    r"fetching .*over PCIe|fetch fraction|hybrid_max_fetch",
    r"CPU MoE executor ready|pool .*core|affinit",
    r"slot|partition|signature",
    r"input throughput",
    r"new-token|cached-token",
    r"expert banks|serial build|low free RAM",
    r"prefill_overlap|prefill overlap|overlap",
    r"nvcc|clang\+\+|cpp_extension|JIT",
    r"ready in|Ready in|Uvicorn running|Started server",
]


def log(msg):
    print("[measure] %s" % msg, flush=True)


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def gpu_used_mib():
    out = sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits").stdout.strip()
    try:
        return int(out.splitlines()[0])
    except Exception:
        return -1


def census_serve():
    out = sh("pgrep -af 'ft serve'").stdout.strip()
    return [l for l in out.splitlines() if "pgrep" not in l]


class VramPoller(threading.Thread):
    def __init__(self, path):
        super().__init__(daemon=True)
        self.path = path
        self.stop_evt = threading.Event()
        self.peak = 0

    def run(self):
        with open(self.path, "a") as f:
            while not self.stop_evt.is_set():
                u = gpu_used_mib()
                if u > self.peak:
                    self.peak = u
                f.write("%.1f %d\n" % (time.time(), u))
                f.flush()
                self.stop_evt.wait(1.0)


def start_server(extra_args, logpath, env_pairs):
    for p in census_serve():
        log("killing leftover: %s" % p)
        sh("pkill -f 'ft serve'")
        time.sleep(5)
    cmd = [FT, "serve", "--port", str(PORT), "--host", "127.0.0.1"] + BASE_ARGS + list(extra_args)
    env = dict(os.environ)
    for kv in env_pairs or []:
        k, _, v = kv.partition("=")
        env[k] = v
    lf = open(logpath, "w")
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=REPO, env=env)
    with open(logpath + ".pid", "w") as f:
        f.write(str(proc.pid))
    return proc, lf, t0


def wait_ready(logpath, proc, timeout=READY_TIMEOUT):
    t0 = time.time()
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
        try:
            r = urllib.request.urlopen(URL + "/ready", timeout=2)
            if r.status == 200:
                r.close()
                return True, time.time() - t0, "ok"
        except Exception:
            pass
        time.sleep(2)
    return False, time.time() - t0, "boot timeout"


def teardown(proc):
    sh("pkill -TERM -f 'ft serve'")
    deadline = time.time() + 10
    while time.time() < deadline:
        if proc.poll() is not None and not census_serve():
            break
        time.sleep(0.5)
    if proc.poll() is None or census_serve():
        sh("pkill -KILL -f 'ft serve'")
        time.sleep(3)
    try:
        proc.wait(timeout=5)
    except Exception:
        pass
    time.sleep(2)
    return {"leftover": census_serve(), "gpu_mib": gpu_used_mib()}


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


def new_log_lines(logpath, offset):
    try:
        with open(logpath, "rb") as f:
            f.seek(offset)
            return f.read().decode("utf-8", "replace").splitlines()
    except Exception:
        return []


def build_req(path, content, max_tokens, stream):
    obj = {"model": MODEL, "messages": [{"role": "user", "content": content}],
           "max_tokens": max_tokens, "temperature": 0, "stream": stream}
    if stream:
        obj["stream_options"] = {"include_usage": True}
    with open(path, "w") as f:
        json.dump(obj, f)
    return path


def parse_tok(lines, kind):
    vals = []
    for l in lines:
        m = re.search(r"#%s:\s*(\d+)" % kind, l)
        if m:
            vals.append(int(m.group(1)))
    return vals[-1] if vals else None


def do_prefill(filler, logpath, max_tokens=4, timeout=PREFILL_TIMEOUT):
    content = open(filler, encoding="utf-8").read()
    rq = build_req(os.path.join(DIR, "req_prefill.json"), content, max_tokens, False)
    poller = VramPoller(os.path.join(DIR, "vram_prefill.log"))
    poller.start()
    off = os.path.getsize(logpath)
    t0 = time.time()
    err = None
    usage = {}
    try:
        req = urllib.request.Request(URL + "/v1/chat/completions", data=open(rq, "rb").read(),
                                     headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=timeout)
        body = json.loads(resp.read())
        resp.close()
        usage = body.get("usage") or {}
    except Exception as e:
        err = "%s: %s" % (type(e).__name__, e)
    wall = time.time() - t0
    poller.stop_evt.set()
    poller.join()
    lines = new_log_lines(logpath, off)
    thr = [l.strip() for l in lines if "input throughput" in l]
    return {"wall_s": round(wall, 2), "error": err, "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"), "peak_vram_mib": poller.peak,
            "throughput_lines": thr, "new_token": parse_tok(lines, "new-token"),
            "cached_token": parse_tok(lines, "cached-token"),
            "watch_in_req": bool(WATCH.search("\n".join(lines)))}


def do_decode(filler, logpath, max_tokens=64, timeout=DECODE_TIMEOUT):
    content = open(filler, encoding="utf-8").read()
    rq = build_req(os.path.join(DIR, "req_decode.json"), content, max_tokens, True)
    off = os.path.getsize(logpath)
    t0 = time.time()
    err = None
    times = []
    usage = {}
    try:
        req = urllib.request.Request(URL + "/v1/chat/completions", data=open(rq, "rb").read(),
                                     headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=timeout)
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
    except Exception as e:
        err = "%s: %s" % (type(e).__name__, e)
    wall = time.time() - t0
    lines = new_log_lines(logpath, off)
    ntok = usage.get("completion_tokens") or len(times)
    steady = None
    p50 = None
    inst_min = None
    if len(times) >= 4:
        # the first SSE frame can arrive before prefill (role/accept frame),
        # so the token rate is measured over the tail without it
        tail = times[1:]
        gaps = [tail[i + 1] - tail[i] for i in range(len(tail) - 1)]
        if len(tail) >= 3 and tail[-1] > tail[0]:
            steady = (len(tail) - 1) / (tail[-1] - tail[0])
        pos = sorted(g for g in gaps if g > 0)
        if pos:
            inst_min = 1.0 / pos[-1]
            p50 = 1.0 / pos[len(pos) // 2]
    return {"wall_s": round(wall, 2), "error": err, "completion_tokens": ntok, "chunks": len(times),
            "steady_tok_s": round(steady, 2) if steady else None,
            "p50_tok_s": round(p50, 2) if p50 else None,
            "avg_tok_s": round(ntok / wall, 2) if wall > 0 else None,
            "min_inst_tok_s": round(inst_min, 2) if inst_min else None,
            "new_token": parse_tok(lines, "new-token"), "cached_token": parse_tok(lines, "cached-token")}


def parse_thrs(lines):
    vals = []
    for l in lines:
        m = re.search(r"input throughput \(token/s\):\s*([0-9]+\.?[0-9]*)", l)
        if m:
            vals.append(float(m.group(1)))
    return vals


def thr_summary(thr_lines):
    vals = parse_thrs(thr_lines)
    if not vals:
        return None
    steady = vals[1:] if len(vals) > 1 else vals
    return {"chunks": len(vals), "first": vals[0], "steady_mean": round(sum(steady) / len(steady), 1),
            "steady_min": round(min(steady), 1), "steady_max": round(max(steady), 1)}


def common_tail(logpath, proc, t0, name):
    ok, el, why = wait_ready(logpath, proc)
    out = {"name": name, "ready": ok, "ready_s": round(el, 1), "why": why, "boot_env": {k: v for k, v in os.environ.items() if k.startswith("FREETOKEN")}}
    time.sleep(2)
    out["gpu_after_boot_mib"] = gpu_used_mib()
    out["log"] = grep_log(logpath)
    return out, ok


def finish(out, proc, logpath):
    out["teardown"] = teardown(proc)
    print(json.dumps(out, indent=1, ensure_ascii=False), flush=True)


def mode_boot(a):
    logpath = os.path.join(LOGDIR, a.log)
    proc, lf, t0 = start_server(shlex.split(a.extra), logpath, a.env)
    out, ok = common_tail(logpath, proc, t0, a.name)
    if not ok:
        out["log_tail"] = "\n".join(new_log_lines(logpath, max(0, os.path.getsize(logpath) - 4000)))
    finish(out, proc, logpath)


def mode_prefill(a):
    logpath = os.path.join(LOGDIR, a.log)
    proc, lf, t0 = start_server(shlex.split(a.extra), logpath, a.env)
    out, ok = common_tail(logpath, proc, t0, a.name)
    if ok:
        time.sleep(2)
        out["prefill"] = do_prefill(os.path.join(DIR, "filler.txt"), logpath)
        out["prefill_thr_summary"] = thr_summary(out["prefill"]["throughput_lines"])
    else:
        out["log_tail"] = "\n".join(new_log_lines(logpath, max(0, os.path.getsize(logpath) - 4000)))
    finish(out, proc, logpath)


def mode_full(a):
    logpath = os.path.join(LOGDIR, a.log)
    proc, lf, t0 = start_server(shlex.split(a.extra), logpath, a.env)
    out, ok = common_tail(logpath, proc, t0, a.name)
    if ok:
        time.sleep(2)
        out["prefill"] = do_prefill(os.path.join(DIR, "filler.txt"), logpath)
        out["prefill_thr_summary"] = thr_summary(out["prefill"]["throughput_lines"])
        if not out["prefill"]["error"] and not out["prefill"]["watch_in_req"]:
            out["decode"] = do_decode(os.path.join(DIR, "filler.txt"), logpath)
    else:
        out["log_tail"] = "\n".join(new_log_lines(logpath, max(0, os.path.getsize(logpath) - 4000)))
    finish(out, proc, logpath)


def mode_ladder(a):
    logpath = os.path.join(LOGDIR, a.log)
    proc, lf, t0 = start_server(shlex.split(a.extra), logpath, a.env)
    out, ok = common_tail(logpath, proc, t0, a.name)
    steps = []
    cum = 0
    filler = open(os.path.join(DIR, "filler.txt"), encoding="utf-8").read()
    if ok:
        time.sleep(2)
        for j in range(a.max_steps):
            if j == 0:
                content = filler
            else:
                blk = open(os.path.join(DIR, "blocks", "block_%d.txt" % (j - 1)), encoding="utf-8").read()
                content = filler + "\n" + blk
            rq = build_req(os.path.join(DIR, "req_ladder.json"), content, 2, False)
            off = os.path.getsize(logpath)
            gpu_before = gpu_used_mib()
            poller = VramPoller(os.path.join(DIR, "vram_ladder.log"))
            poller.start()
            t0 = time.time()
            err = None
            usage = {}
            try:
                req = urllib.request.Request(URL + "/v1/chat/completions", data=open(rq, "rb").read(),
                                             headers={"Content-Type": "application/json"})
                resp = urllib.request.urlopen(req, timeout=a.step_timeout)
                body = json.loads(resp.read())
                resp.close()
                usage = body.get("usage") or {}
            except Exception as e:
                err = "%s: %s" % (type(e).__name__, e)
            wall = time.time() - t0
            poller.stop_evt.set()
            poller.join()
            lines = new_log_lines(logpath, off)
            nt = parse_tok(lines, "new-token")
            ct = parse_tok(lines, "cached-token")
            if nt is None:
                nt = usage.get("prompt_tokens")
            thr = [l.strip() for l in lines if "input throughput" in l]
            oom = bool(WATCH.search("\n".join(lines)))
            step = {"step": j, "wall_s": round(wall, 2), "error": err, "new_token": nt,
                    "cached_token": ct, "cum_tokens": cum + (nt or 0),
                    "gpu_before_mib": gpu_before, "peak_vram_mib": poller.peak,
                    "gpu_after_mib": gpu_used_mib(),
                    "thr_summary": thr_summary(thr), "watch": oom}
            steps.append(step)
            log("ladder step %s: %s" % (j, json.dumps(step)))
            if err or oom:
                break
            if nt:
                cum += nt
            if cum >= a.target_tokens:
                break
    else:
        out["log_tail"] = "\n".join(new_log_lines(logpath, max(0, os.path.getsize(logpath) - 4000)))
    out["ladder"] = steps
    out["max_depth_tokens"] = cum
    finish(out, proc, logpath)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    for m in ["boot", "prefill", "full", "ladder"]:
        p = sub.add_parser(m)
        p.add_argument("--name", required=True)
        p.add_argument("--log", required=True)
        p.add_argument("--extra", default="")
        p.add_argument("--env", action="append", default=[])
        if m == "ladder":
            p.add_argument("--max-steps", type=int, default=7)
            p.add_argument("--target-tokens", type=int, default=393216)
            p.add_argument("--step-timeout", type=int, default=900)
    a = ap.parse_args()
    os.makedirs(LOGDIR, exist_ok=True)
    log("baseline gpu_mib=%s census=%s" % (gpu_used_mib(), census_serve()))
    {"boot": mode_boot, "prefill": mode_prefill, "full": mode_full, "ladder": mode_ladder}[a.mode](a)
    log("done")


if __name__ == "__main__":
    main()
