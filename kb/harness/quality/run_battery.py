#!/usr/bin/env python3
"""Quality battery stage driver (v2 MMQ A/B): census -> boot ft serve with the
stage's FREETOKEN_GGUF_GROUPED_PREFILL value -> /ready -> 24-prompt battery ->
SIGTERM (30 s) -> SIGKILL teardown -> census. Serial only; harness-side only,
no production code touched.

Usage: run_battery.py <env0|env1>
Exit codes: 0 battery ok; 2 FAILPAT hit; 3 boot failure; 4 battery rc!=0; 5 leftover processes.
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request

STAGE = sys.argv[1] if len(sys.argv) > 1 else ""
if STAGE not in ("env0", "env1"):
    sys.exit("usage: run_battery.py <env0|env1>")
ENVVAL = "0" if STAGE == "env0" else "1"

ROOT = "/media/ai/src/FreeToken"
QDIR = os.path.join(ROOT, ".tasks/mmq-prefill-kernel/quality")
BASELINE = os.path.join(ROOT, ".tasks/gguf-glm5next-path-a/verification/phase6/battery-post-fix")
OUTDIR = os.path.join(QDIR, STAGE)
LOGPATH = os.path.join(OUTDIR, "serve.log")
PORT = 18801
URL = "http://127.0.0.1:%d" % PORT
READY_TIMEOUT = 600
BATTERY_TIMEOUT = 2700
FAILPAT = re.compile(r"AssertionError|OutOfMemoryError|Backend worker is gone")

CMD = [
    os.path.join(ROOT, ".venv/bin/ft"), "serve", "--port", str(PORT), "--host", "127.0.0.1",
    "--model-path", "/media/ai/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL.gguf",
    "--moe-cache-auto", "--kv-reserve-tokens", "500000", "--kv-cache-dtype", "fp8",
    "--memory-ratio", "0.85", "--max-prefill-length", "8191",
    "--moe-strategy", "hybrid", "--moe-cpu-threads", "16", "--max-running-requests", "1",
]
ENVP = {
    "FREETOKEN_GGUF_GROUPED_PREFILL": ENVVAL,
    # harness-side probe: surfaces the bare-logger one-shot marker (moe.py:520)
    "PYTHONPATH": os.path.join(ROOT, ".tasks/mmq-prefill-kernel/probe_sitecustomize"),
}

state = {"failpat": None, "rc": None}


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def census(tag):
    serve = [l for l in sh("pgrep -af '[f]t serve'").stdout.splitlines() if "pgrep" not in l]
    vram = sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader").stdout.strip()
    sems = sh("ls /dev/shm | grep -c '^sem\\.mp-'").stdout.strip() or "0"
    print("[%s] census ft_serve=%s vram=%s sem.mp=%s" % (tag, serve or "none", vram, sems), flush=True)
    return serve


def log(msg):
    print("[stage %s] %s" % (STAGE, msg), flush=True)


def watchdog(proc):
    t0 = time.time()
    while time.time() - t0 < BATTERY_TIMEOUT + READY_TIMEOUT:
        time.sleep(2)
        if proc.poll() is not None:
            return
        try:
            full = open(LOGPATH, "rb").read().decode("utf-8", "replace")
        except OSError:
            continue
        m = FAILPAT.search(full[-20000:])
        if m:
            state["failpat"] = m.group(0)
            log("WATCHDOG FAILPAT hit: %s" % m.group(0))
            for line in full[-20000:].splitlines():
                if FAILPAT.search(line) or "Tried to allocate" in line:
                    log("  | %s" % line[:200])
            sh("pkill -9 -f 'ft serve'")
            return


def boot():
    os.makedirs(OUTDIR, exist_ok=True)
    for p in census("pre-boot"):
        log("killing leftover %s" % p)
        sh("pkill -f 'ft serve'")
        time.sleep(5)
    env = dict(os.environ)
    env.update(ENVP)
    lf = open(LOGPATH, "w")
    t0 = time.time()
    proc = subprocess.Popen(CMD, stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
    with open(LOGPATH + ".pid", "w") as f:
        f.write(str(proc.pid))
    thr = threading.Thread(target=watchdog, args=(proc,), daemon=True)
    thr.start()
    while time.time() - t0 < READY_TIMEOUT:
        if proc.poll() is not None:
            return None, proc, "process exited rc=%s" % proc.returncode
        if state["failpat"]:
            return None, proc, "failpat: %s" % state["failpat"]
        try:
            tail = open(LOGPATH, "rb").read()[-12000:].decode("utf-8", "replace")
            m = FAILPAT.search(tail)
            if m:
                return None, proc, "failpat(boot): %s" % m.group(0)
        except OSError:
            pass
        try:
            r = urllib.request.urlopen(URL + "/ready", timeout=2)
            if r.status == 200:
                r.close()
                return time.time() - t0, proc, "ok"
        except Exception:
            pass
        time.sleep(2)
    return None, proc, "boot timeout"


def teardown(proc):
    if proc.poll() is None:
        proc.terminate()
    deadline = time.time() + 30
    while time.time() < deadline:
        if proc.poll() is not None and not census("teardown-mid"):
            break
        time.sleep(1)
    if proc.poll() is None:
        log("SIGTERM grace expired -> SIGKILL")
        sh("pkill -KILL -f 'ft serve'")
        proc.wait(timeout=30)
    sh("pkill -KILL -f 'ft serve'")
    time.sleep(3)


def boot_facts(ready_s):
    full = open(LOGPATH, "rb").read().decode("utf-8", "replace")
    facts = {"ready_s": ready_s, "failpat": state["failpat"]}
    for pat, name in [
        (r"moe_cache_size=\d+", "moe_cache"),
        (r"Allocating \d+ tokens[^\n]*", "kv_alloc"),
        (r"Free memory after initialization[^\n]*", "free_mem"),
        (r"gguf moe grouped mmq prefill active[^\n]*", "marker"),
        (r"fetching [\d.]+%[^\n]*", "fetch_fraction"),
        (r"pool \d/\d[^\n]*isa=[^\n]*", "isa"),
    ]:
        m = re.search(pat, full)
        facts[name] = m.group(0)[:160] if m else None
    return facts


def main():
    rc = 0
    summary = {"stage": STAGE, "env": ENVP["FREETOKEN_GGUF_GROUPED_PREFILL"],
               "cmd": CMD, "facts": {}, "battery": {}}
    try:
        ready_s, proc, how = boot()
        if ready_s is None:
            log("BOOT FAILED: %s" % how)
            tail = open(LOGPATH, "rb").read()[-3000:].decode("utf-8", "replace")
            log("log tail: %s" % tail[-1200:])
            summary["facts"] = {"ready_s": None, "failpat": how}
            rc = 3
        else:
            log("boot ready in %.1f s" % ready_s)
            cmd = [sys.executable, os.path.join(QDIR, "ft_phase6_battery.py"),
                   "--port", str(PORT), "--baseline-dir", BASELINE, "--outdir", OUTDIR]
            t0 = time.time()
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=BATTERY_TIMEOUT)
                bout = r.stdout + r.stderr
                brc = r.returncode
            except subprocess.TimeoutExpired as e:
                bout = (e.stdout or b"").decode("utf-8", "replace") + "TIMEOUT"
                brc = 124
            with open(os.path.join(OUTDIR, "battery_client.log"), "w") as f:
                f.write(bout)
            log(bout.strip()[-4000:])
            summary["battery"] = {"rc": brc, "wall_s": round(time.time() - t0, 1)}
            if brc != 0:
                rc = 4
            summary["facts"] = boot_facts(ready_s)
            log("facts: %s" % json.dumps(summary["facts"], ensure_ascii=False))
        teardown(proc)
        leftovers = census("post")
        if leftovers:
            log("LEFTOVER ft serve processes: %s" % leftovers)
            rc = rc or 5
    finally:
        sh("pkill -KILL -f 'ft serve'")
        with open(os.path.join(QDIR, "%s_stage_summary.json" % STAGE), "w") as f:
            json.dump(summary, f, indent=1, ensure_ascii=False)
    log("stage rc=%s" % rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
