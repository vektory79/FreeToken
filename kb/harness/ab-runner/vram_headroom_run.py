#!/usr/bin/env python3
"""VRAM-headroom verification wave (dense-q80-gemm follow-up, user request).

Question: which knobs (memory-ratio / kv-reserve-tokens / max-prefill-length)
buy >= 1.5-2 GB FREE VRAM at peak while serving big contexts, and at what
throughput cost. Pure measurements - no commits, no fabrication.

Matrix (all boots: fp8 KV, hybrid, moe-cpu-threads 16, max-running-requests 1,
port 18801, committed kernel knob defaults = NO FREETOKEN_* env set):
  a: ratio 0.85, reserve 500000, chunk 8191  (control, session anchor)
  b: ratio 0.80, reserve 500000, chunk 8191
  c: ratio 0.75, reserve 400000, chunk 8191
  d: ratio 0.85, reserve 350000, chunk 8191
  e: ratio 0.85, reserve 500000, chunk 4095  (chunk lever)

Per boot (serialized + reaped, trap-on-EXIT, FAILPAT watchdog DISARMED at
teardown, census FILE-logged pre/post): boot -> VRAM sampler (2 s, per-boot
csv, phase-tagged) -> /ready -> plan-line capture -> 65,585-token fill
(8x8128+561) -> streamed decode re-send (radix HIT expected) -> 6-prompt
greedy battery (informational shas, req_mtile methodology) -> SIGTERM 30 s ->
SIGKILL group -> census post.

Extra-boot budget: exactly ONE boot beyond the matrix, in this priority:
  1. a failed matrix boot adjusted per the rule (slot-floor -> cut reserve
     100k once, e.g. 0.75/400k -> 0.75/300k);
  2. all of a-d miss the >= 1.5 GiB free target -> fallback 0.70/300000/8191.

Usage: vram_headroom_run.py [--only a|b|c|d|e] [--no-auto-fallback]
Exit codes: 0 ok; 2 FAILPAT hit; 3 boot/prefill failure; 4 battery failure;
6 foreign-process stop.
"""
import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = "/media/ai/src/FreeToken"
TASKDIR = os.path.join(ROOT, ".tasks/dense-q80-gemm")
TUNING = os.path.join(ROOT, ".tasks/ft-gguf-serve-tuning")
LOGDIR = os.path.join(TUNING, "logs")
OUTDIR = os.path.join(TASKDIR, "vram-headroom")
MODEL = ("/media/ai/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/"
         "GLM-5.3-Flash-UD-Q3_K_XL.gguf")
GPU_TOTAL_MIB = 32607
FREE_TARGET_MIB = 1536  # >= 1.5 GiB free at peak

sys.path.insert(0, TUNING)
import measure  # noqa: E402

BOOTS = [
    ("a", "0.85", "500000", "8191"),
    ("b", "0.80", "500000", "8191"),
    ("c", "0.75", "400000", "8191"),
    ("d", "0.85", "350000", "8191"),
    ("e", "0.85", "500000", "4095"),
]
FALLBACK = ("f0", "0.70", "300000", "8191")

READY_TIMEOUT = 600
FILL_TIMEOUT = 900
DECODE_TIMEOUT = 600
PROMPT_TIMEOUT = 300
TEARDOWN_GRACE = 30
WATCHDOG_SPAN = 5400
FAILPAT = re.compile(measure.WATCH.pattern + r"|backend worker .* exited")
CENSUS_PAT = "ft serve|nsys|llama"
HARD_PAT = re.compile(r"ft serve|\bnsys\b")
BENIGN_COMPUTE_APPS = ("hoptodesk",)
MAX_VRAM_STOP_MIB = 2000
PREFILL_ANCHOR_A = 808.0  # plain-boot median c2..c7 anchor (t64a/t64b)
PLAN_PATTERNS = [
    r"moe-cache-auto resolved",
    r"Allocating .* for KV cache",
    r"Free memory (before loading|after initialization)",
    r"moe-hybrid-max-fetch auto",
    r"signature group .* slots",
    r"slot-floor|minimum plan|plan_cache_budget|budget",
    r"prefill overlap|prefill_overlap",
    r"torch intra-op threads",
]

# identical 6 greedy prompts as the mtile/nsplit/arbitration waves (sha input)
PROMPTS = [
    "What is 17 * 23? Answer with just the number.",
    "Name the capital of France and one famous landmark there. One sentence.",
    "Write a haiku about rain on a tin roof.",
    "Explain in two sentences why the sky is blue.",
    "Translate 'good morning' into German, French, and Spanish.",
    "Count from 1 to 20, comma separated.",
]
PROMPT_MAX_TOKENS = 96

state = {"failpat": None, "tearing_down": False}
phase_lock = threading.Lock()
phase = {"name": "boot"}


def log(msg):
    print("[vram %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def set_phase(name):
    with phase_lock:
        phase["name"] = name


def census(tag, boot):
    out = subprocess.run(["pgrep", "-a", "-f", CENSUS_PAT],
                         capture_output=True, text=True).stdout
    procs = [l.strip() for l in out.splitlines() if l.strip()]
    vram = sh("nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader").stdout.strip()
    apps = sh("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader").stdout.strip()
    sems = sh("ls /dev/shm | grep -c '^sem\\.mp-'").stdout.strip() or "0"
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "tag": tag, "boot": boot, "procs": procs, "gpu_vram": vram,
           "gpu_compute_apps": apps, "sem_mp": sems}
    with open(os.path.join(OUTDIR, "census.log"), "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log("[census %s] procs=%s vram=[%s] apps=[%s] sem.mp-=%s" % (
        tag, procs or "none", vram, apps, sems))
    return rec


def used_mib(vram_line):
    try:
        return int(vram_line.split(",")[0].strip().split()[0])
    except Exception:
        return -1


def quiet_gate(boot):
    rec = census("pre-boot", boot)
    hard = [p for p in rec["procs"] if HARD_PAT.search(p)]
    llama = [p for p in rec["procs"] if "llama-swap" in p]
    apps = [l for l in (rec["gpu_compute_apps"] or "").splitlines()
            if l.strip() and not any(b in l for b in BENIGN_COMPUTE_APPS)]
    if hard:
        return False, "foreign ft serve/nsys processes: %s" % hard
    if apps:
        return False, "foreign GPU compute apps: %s" % apps
    if used_mib(rec["gpu_vram"]) >= MAX_VRAM_STOP_MIB:
        return False, "VRAM holder without a matched process: %s" % rec["gpu_vram"]
    if llama:
        log("NOTE: idle llama-swap proxy present; holds no compute VRAM - "
            "proceeding (sweep precedent, never killed)")
    return True, None


class VramSampler(threading.Thread):
    """2 s loop, per-boot csv: ts_iso,epoch,phase,used_mib + per-phase peaks."""

    def __init__(self, name):
        super().__init__(daemon=True)
        self.name = name
        self.path = os.path.join(TASKDIR, "vram_%s.csv" % name)
        self.stop_evt = threading.Event()
        self.peaks = {}
        self.overall_peak = 0

    def run(self):
        with open(self.path, "w") as f:
            while not self.stop_evt.is_set():
                with phase_lock:
                    ph = phase["name"]
                u = measure.gpu_used_mib()
                if u >= 0:
                    if u > self.peaks.get(ph, 0):
                        self.peaks[ph] = u
                    if u > self.overall_peak:
                        self.overall_peak = u
                    f.write("%s,%d,%s,%d\n" % (
                        time.strftime("%H:%M:%S"), int(time.time()), ph, u))
                    f.flush()
                self.stop_evt.wait(2.0)

    def stop(self):
        self.stop_evt.set()
        self.join(timeout=10)


def boot(boot_name, ratio, reserve, chunk, logpath):
    """Plain boot, env sanitized: committed kernel-knob defaults, nothing set."""
    for k in list(os.environ):
        if k.startswith("FREETOKEN_") or k == "NSYS_SESSION":
            del os.environ[k]
    cmd = [measure.FT, "serve", "--port", str(measure.PORT), "--host", "127.0.0.1",
           "--model-path", MODEL, "--moe-cache-auto",
           "--kv-reserve-tokens", reserve, "--kv-cache-dtype", "fp8",
           "--memory-ratio", ratio, "--max-prefill-length", chunk,
           "--moe-strategy", "hybrid", "--moe-cpu-threads", "16",
           "--max-running-requests", "1"]
    lf = open(logpath, "w")
    proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT,
                            env=dict(os.environ), start_new_session=True)
    with open(logpath + ".pid", "w") as f:
        f.write(str(proc.pid))
    return proc, cmd


def killpg_quiet(proc, sig):
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except (ProcessLookupError, PermissionError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass


def watchdog(proc, logpath):
    t0 = time.time()
    while time.time() - t0 < WATCHDOG_SPAN:
        time.sleep(2)
        # tearing_down: graceful SIGTERM logs per-worker 'backend worker ...
        # exited' lines - those are normal teardown, not FAILPAT
        if state["tearing_down"] or proc.poll() is not None:
            return
        try:
            tail = open(logpath, "rb").read()[-20000:].decode("utf-8", "replace")
        except OSError:
            continue
        m = FAILPAT.search(tail)
        if m:
            state["failpat"] = m.group(0)
            log("WATCHDOG FAILPAT hit: %s" % m.group(0))
            for line in tail.splitlines():
                if FAILPAT.search(line) or "Tried to allocate" in line:
                    log("  | %s" % line[:200])
            killpg_quiet(proc, 9)
            return


def wait_ready(logpath, proc, timeout=READY_TIMEOUT):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False, time.time() - t0, "process exited rc=%s" % proc.returncode
        if state["failpat"]:
            return False, time.time() - t0, "failpat: %s" % state["failpat"]
        try:
            tail = open(logpath, "rb").read()[-12000:].decode("utf-8", "replace")
            m = FAILPAT.search(tail)
            if m:
                return False, time.time() - t0, "failpat(boot): %s" % m.group(0)
        except OSError:
            pass
        try:
            r = urllib.request.urlopen(measure.URL + "/ready", timeout=2)
            if r.status == 200:
                r.close()
                return True, time.time() - t0, "ok"
        except Exception:
            pass
        time.sleep(2)
    return False, time.time() - t0, "boot timeout"


def parse_thrs(lines):
    vals = []
    for l in lines:
        m = re.search(r"input throughput \(token/s\):\s*([0-9]+\.?[0-9]*)", l)
        if m:
            vals.append(float(m.group(1)))
    return vals


def plan_lines(logpath):
    try:
        txt = open(logpath, encoding="utf-8", errors="replace").read()
    except OSError:
        return {}
    res = {}
    for pat in PLAN_PATTERNS:
        res[pat] = [l.strip() for l in txt.splitlines() if re.search(pat, l)][-20:]
    return res


def log_failure_lines(logpath):
    try:
        txt = open(logpath, "rb").read().decode("utf-8", "replace")
    except OSError:
        return []
    lines = txt.splitlines()
    hits = [i for i, l in enumerate(lines)
            if FAILPAT.search(l) or "Tried to allocate" in l
            or "minimum plan" in l or "budget" in l.lower() or "slot-floor" in l]
    out = []
    for i in hits[:8]:
        out.extend(lines[max(0, i - 2):i + 2])
    return out


def do_fill(logpath):
    content = open(os.path.join(TUNING, "filler.txt"), encoding="utf-8").read()
    rq = os.path.join(OUTDIR, "req_fill.json")
    measure.build_req(rq, content, 4, False)
    off = os.path.getsize(logpath)
    set_phase("fill")
    t0 = time.time()
    err = None
    usage = {}
    try:
        req = urllib.request.Request(measure.URL + "/v1/chat/completions",
                                     data=open(rq, "rb").read(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=FILL_TIMEOUT) as r:
            usage = (json.loads(r.read().decode()).get("usage")) or {}
    except Exception as e:
        err = "%s: %s" % (type(e).__name__, e)
    wall = time.time() - t0
    set_phase("idle2")
    lines = measure.new_log_lines(logpath, off)
    thr = parse_thrs(lines)
    steady = thr[1:-2] if len(thr) > 3 else thr
    return {"wall_s": round(wall, 2), "error": err,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "thr_values": thr,
            "median_c2_c7": round(statistics.median(steady), 2) if steady else None,
            "chunk_new_tokens": [int(x) for x in re.findall(
                r"#new-token:\s*(\d+)", "\n".join(lines))][:30]}


def do_decode(logpath):
    content = open(os.path.join(TUNING, "filler.txt"), encoding="utf-8").read()
    rq = os.path.join(OUTDIR, "req_decode.json")
    measure.build_req(rq, content, 64, True)
    off = os.path.getsize(logpath)
    set_phase("decode")
    t0 = time.time()
    err = None
    times = []
    usage = {}
    try:
        req = urllib.request.Request(measure.URL + "/v1/chat/completions",
                                     data=open(rq, "rb").read(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=DECODE_TIMEOUT) as r:
            for raw in r:
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
    except Exception as e:
        err = "%s: %s" % (type(e).__name__, e)
    wall = time.time() - t0
    set_phase("idle3")
    lines = measure.new_log_lines(logpath, off)
    steady = None
    tail = times[1:]
    if len(tail) >= 3 and tail[-1] > tail[0]:
        steady = (len(tail) - 1) / (tail[-1] - tail[0])
    return {"wall_s": round(wall, 2), "error": err,
            "completion_tokens": usage.get("completion_tokens"),
            "steady_tok_s": round(steady, 2) if steady else None,
            "new_token": measure.parse_tok(lines, "new-token"),
            "cached_token": measure.parse_tok(lines, "cached-token")}


def do_battery(name, logpath):
    set_phase("battery")
    arts = []
    ok = 0
    for i, content in enumerate(PROMPTS):
        rq_path = os.path.join(OUTDIR, "req_vram_%s_%d.json" % (name, i))
        measure.build_req(rq_path, content, PROMPT_MAX_TOKENS, False)
        off = os.path.getsize(logpath)
        t0 = time.time()
        err = None
        choices = None
        try:
            req = urllib.request.Request(
                measure.URL + "/v1/chat/completions",
                data=open(rq_path, "rb").read(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=PROMPT_TIMEOUT) as r:
                body = json.loads(r.read().decode())
            choices = body.get("choices")
        except Exception as e:
            err = "%s: %s" % (type(e).__name__, e)
        wall = time.time() - t0
        lines = measure.new_log_lines(logpath, off)
        blob = json.dumps(choices, sort_keys=True) if choices is not None else ""
        sha = hashlib.sha256(blob.encode()).hexdigest() if blob else None
        arts.append({"i": i, "wall_s": round(wall, 2), "error": err, "sha256": sha})
        if err:
            log("[bat %s] p%d FAILED %s" % (name, i, err))
            break
        ok += 1
        log("[bat %s] p%d ok %5.1fs sha=%s" % (name, i, wall, sha[:8] if sha else "-"))
    set_phase("idle4")
    return {"ok": ok, "failed": len(arts) - ok, "arts": arts}


def teardown(proc):
    set_phase("teardown")
    killpg_quiet(proc, 15)
    deadline = time.time() + TEARDOWN_GRACE
    while time.time() < deadline:
        if proc.poll() is not None:
            break
        time.sleep(1)
    if proc.poll() is None:
        log("SIGTERM grace expired -> SIGKILL process group")
        killpg_quiet(proc, 9)
    try:
        proc.wait(timeout=30)
    except Exception:
        pass
    time.sleep(3)
    out = subprocess.run(["pgrep", "-a", "-f", CENSUS_PAT],
                         capture_output=True, text=True).stdout
    leftovers = [l.strip() for l in out.splitlines() if l.strip()]
    return {"grace_expired": proc.poll() is None, "rc": proc.returncode,
            "leftovers": leftovers, "gpu_mib": measure.gpu_used_mib()}


def repair_our_leftovers():
    out = subprocess.run(["pgrep", "-a", "-f", "ft serve --port 18801"],
                         capture_output=True, text=True).stdout
    ours = [l.strip() for l in out.splitlines() if l.strip()]
    if ours:
        log("repairing OUR leftovers from an aborted attempt: %s" % ours)
        sh("pkill -9 -f 'ft serve --port 18801'")
        time.sleep(4)
    for _ in range(10):
        if measure.gpu_used_mib() < MAX_VRAM_STOP_MIB:
            break
        time.sleep(2)
    return ours


def run_boot(rec, sampler):
    """Execute one boot record end-to-end. Returns 'ok' | 'failed'."""
    proc, cmd = boot(rec["name"], rec["ratio"], rec["reserve"], rec["chunk"],
                     rec["logpath"])
    rec["cmd"] = cmd
    wd = threading.Thread(target=watchdog, args=(proc, rec["logpath"]), daemon=True)
    wd.start()
    verdict = "failed"
    try:
        okr, el, why = wait_ready(rec["logpath"], proc)
        rec["ready"] = {"ok": okr, "ready_s": round(el, 1), "why": why}
        log("%s boot %s in %.1fs (%s)" % (rec["name"], "ready" if okr else "FAILED",
                                          el, why))
        if okr:
            rec["gpu_after_boot_mib"] = measure.gpu_used_mib()
            rec["plan"] = plan_lines(rec["logpath"])
            time.sleep(2)
            rec["fill"] = do_fill(rec["logpath"])
            vals = rec["fill"]["thr_values"]
            log("%s fill: %d thr lines, median c2..cN = %s tok/s (peak cpu-free "
                "post sampler)" % (rec["name"], len(vals),
                                   rec["fill"]["median_c2_c7"]))
            if rec["fill"]["error"] or FAILPAT.search("\n".join(
                    measure.new_log_lines(rec["logpath"], 0)[-200:])):
                rec["ready"]["ok"] = False
                rec["ready"]["why"] = "fill: %s" % rec["fill"]["error"]
                verdict = "fill_fail"
            else:
                verdict = "ok"
                rec["decode"] = do_decode(rec["logpath"])
                hit = rec["decode"]["cached_token"] or 0
                log("%s decode: steady=%s tok/s cached=%s new=%s" % (
                    rec["name"], rec["decode"]["steady_tok_s"], hit,
                    rec["decode"]["new_token"]))
                rec["battery"] = do_battery(rec["name"], rec["logpath"])
                if rec["battery"]["failed"]:
                    verdict = "battery_fail"
                if state["failpat"]:
                    rec["failpat"] = state["failpat"]
                    verdict = "failpat"
                    return verdict
        else:
            rec["failure_lines"] = log_failure_lines(rec["logpath"])
            for l in rec["failure_lines"][:12]:
                log("  | %s" % l[:220])
    finally:
        state["tearing_down"] = True
        rec["teardown"] = teardown(proc)
        wd.join(timeout=10)
        rec["post_census"] = census("postflight", rec["name"])
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, choices=[n for n, _, _, _ in BOOTS])
    ap.add_argument("--no-auto-fallback", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    repair_our_leftovers()
    rc = 0
    records = []
    extra_used = False
    results_path = os.path.join(TASKDIR, "vram-headroom-results.json")

    def save():
        with open(results_path, "w") as f:
            json.dump({"boots": records, "rc": rc, "ts": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f, indent=1,
                ensure_ascii=False)

    plan = list(BOOTS)
    if a.only:
        plan = [b for b in BOOTS if b[0] == a.only]
    pending = list(plan)
    while pending:
        name, ratio, reserve, chunk = pending.pop(0)
        state["failpat"] = None
        state["tearing_down"] = False
        okq, why = quiet_gate(name)
        if not okq:
            log("FOREIGN VRAM HOLDER -> STOP before %s: %s" % (name, why))
            rc = 6
            break
        sampler = VramSampler(name)
        sampler.start()
        rec = {"name": name, "ratio": ratio, "reserve": reserve, "chunk": chunk,
               "mode": "plain (no nsys, no knobs set)",
               "csv": os.path.join(TASKDIR, "vram_%s.csv" % name),
               "logpath": os.path.join(LOGDIR, "vram_%s.log" % name),
               "pre_census": census("preflight", name)}
        try:
            verdict = run_boot(rec, sampler)
        except Exception as e:
            log("EXCEPTION on %s: %r" % (name, e))
            rec["exception"] = repr(e)
            verdict = "failed"
        finally:
            sampler.stop()
            rec["peaks_mib"] = {k: v for k, v in sampler.peaks.items()}
            rec["peak_overall_mib"] = sampler.overall_peak
            rec["peak_free_mib"] = {k: GPU_TOTAL_MIB - v
                                    for k, v in sampler.peaks.items()}
            rec["peak_free_overall_mib"] = GPU_TOTAL_MIB - sampler.overall_peak
            records.append(rec)
            save()
        # leftover check: our boot must be gone; llama-swap proxy may remain
        left_ours = [l for l in rec.get("teardown", {}).get("leftovers", [])
                     if "llama-swap" not in l]
        if left_ours:
            log("LEFTOVER processes after %s: %s" % (name, left_ours))
            sh("pkill -9 -f 'ft serve --port 18801'")
            time.sleep(4)
            rc = max(rc, 5)
        if verdict == "failed" and rec.get("ready", {}).get("ok") is not True:
            # failure adjustment: cut reserve to match, same ratio, ONCE total
            if extra_used:
                log("boot %s failed and the one extra boot is spent - continue"
                    % name)
                rc = max(rc, 3)
                continue
            new_reserve = max(300000, int(reserve) - 100000)
            if new_reserve == int(reserve):
                log("boot %s failed at minimum reserve - continue" % name)
                rc = max(rc, 3)
                continue
            adj = ("%s2" % name, ratio, str(new_reserve), chunk)
            log("boot %s failed (slot-floor/OOM class) -> ONE adjustment boot: "
                "ratio=%s reserve=%s chunk=%s" % adj[0:4])
            extra_used = True
            pending.insert(0, adj)
            continue
        if verdict in ("failed", "fill_fail", "failpat"):
            rc = max(rc, 3 if verdict in ("failed", "fill_fail") else 2)
        if verdict == "battery_fail":
            rc = max(rc, 4)
        if name in ("a", "b", "c", "d") and not extra_used and not a.only \
                and not a.no_auto_fallback:
            # fallback only if ALL of a-d missed the >= 1.5 GiB free target
            done_abcd = [r for r in records
                         if r["name"] in ("a", "b", "c", "d")]
            if len(done_abcd) == 4 and pending and pending[0][0] == "e":
                misses = [r for r in done_abcd
                          if r["peak_free_mib"].get("fill", 0) < FREE_TARGET_MIB]
                if len(misses) == 4:
                    log("a-d ALL miss the >= 1.5 GiB free target -> ONE "
                        "fallback boot 0.70/300000/8191")
                    extra_used = True
                    pending.insert(0, FALLBACK)
    save()
    log("run rc=%s" % rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
