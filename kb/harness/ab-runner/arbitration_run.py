#!/usr/bin/env python3
"""Arbitration battery (dense-q80-gemm, user-approved): quantify boot-to-boot
output nondeterminism and re-test cross-config (dense m-tile 4 vs 64) output
agreement at 24-prompt granularity. 4 PLAIN boots, strictly serialized,
interleaved t4a/t64a/t4b/t64b so boot drift cannot masquerade as a config
effect. NO nsys anywhere (single instrumentation mode).

Per boot: timestamped census -> quiet-box gate (foreign VRAM holder => STOP,
never kill) -> boot with FREETOKEN_GGUF_DENSE_MTILE=<4|64> and
FREETOKEN_GGUF_MOE_MTILE=32, FREETOKEN_GGUF_KDA_NSPLIT unset (committed
default 2) -> /ready -> 65,585-token fill (measure.do_prefill) -> 24-prompt
greedy battery (reasoning+content verbatim + sha256 per prompt) ->
SIGTERM 30 s -> SIGKILL group -> postflight census.

The idle llama-swap proxy wrapper on 18080 is a standing user service holding
no compute VRAM (sweep precedent): it does NOT stop the wave. ft serve / nsys
/ any other VRAM holder DOES.

Usage: arbitration_run.py [--only t4a|t64a|t4b|t64b]
Exit codes: 0 all boots ok; 2 FAILPAT hit; 3 boot/prefill failure;
4 battery failure; 5 teardown leftovers; 6 foreign-process stop.
"""
import argparse
import hashlib
import json
import os
import re
import signal
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
QDIR = os.path.join(TASKDIR, "quality-arbitration")
CENSUS_LOG = os.path.join(QDIR, "census.log")
BASELINE = os.path.join(ROOT, ".tasks/gguf-glm5next-path-a/verification/phase6/battery-post-fix")

sys.path.insert(0, TUNING)
import measure  # noqa: E402

BOOTS = [("t4a", "4"), ("t64a", "64"), ("t4b", "4"), ("t64b", "64")]
READY_TIMEOUT = 600
BATTERY_TIMEOUT = 3000
PROMPT_TIMEOUT = 600
TEARDOWN_GRACE = 30
WATCHDOG_SPAN = READY_TIMEOUT + BATTERY_TIMEOUT + measure.PREFILL_TIMEOUT
# FAILPAT = measure.WATCH + the backend-worker-exited variant (process hygiene)
FAILPAT = re.compile(measure.WATCH.pattern + r"|backend worker .* exited")
CENSUS_PAT = "ft serve|nsys|llama"
HARD_PAT = re.compile(r"ft serve|\bnsys\b")
BENIGN_COMPUTE_APPS = ("hoptodesk",)
MAX_VRAM_STOP_MIB = 2000
# same-mode plain-boot prefill medians (c2..c7) from the Candidate A sweep
PREFILL_ANCHOR = {"4": 563.90, "64": 792.08}

state = {"failpat": None, "tearing_down": False}


def log(msg):
    print("[arb %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def census(tag, boot):
    """Timestamped census to FILE + stdout. Returns the census dict."""
    out = subprocess.run(["pgrep", "-a", "-f", CENSUS_PAT],
                         capture_output=True, text=True).stdout
    procs = [l.strip() for l in out.splitlines() if l.strip()]
    vram = sh("nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader").stdout.strip()
    apps = sh("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader").stdout.strip()
    sems = sh("ls /dev/shm | grep -c '^sem\\.mp-'").stdout.strip() or "0"
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "tag": tag, "boot": boot, "procs": procs, "gpu_vram": vram,
           "gpu_compute_apps": apps, "sem_mp": sems}
    with open(CENSUS_LOG, "a") as f:
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
        log("NOTE: idle llama-swap proxy present (%s); holds no compute VRAM, "
            "not a VRAM holder - proceeding (sweep precedent, never killed)" % llama)
    return True, None


def boot(name, tile, logpath):
    """Plain boot. Sanitizes the env to exactly the two knobs (NSPLIT unset)."""
    for k in list(os.environ):
        if k.startswith("FREETOKEN_") or k == "NSYS_SESSION":
            del os.environ[k]
    os.environ["FREETOKEN_GGUF_MOE_MTILE"] = "32"
    os.environ["FREETOKEN_GGUF_DENSE_MTILE"] = tile
    # FREETOKEN_GGUF_KDA_NSPLIT deliberately UNSET -> committed default 2
    cmd = [measure.FT, "serve", "--port", str(measure.PORT), "--host", "127.0.0.1"] \
        + measure.BASE_ARGS + ["--memory-ratio", "0.85", "--max-prefill-length", "8191",
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
        # tearing_down: graceful SIGTERM makes the supervisor log per-worker
        # 'backend worker ... exited' lines - those are normal, not FAILPAT
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
            killpg_quiet(proc, signal.SIGKILL)
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


def server_env_pins(pid):
    try:
        raw = open("/proc/%d/environ" % pid, "rb").read().decode("utf-8", "replace")
    except OSError:
        return {}
    return dict(x.split("=", 1) for x in raw.split("\0")
                if "=" in x and (x.startswith("FREETOKEN") or x.startswith("NSYS_")))


def out_sha(art):
    blob = json.dumps({"content": art["content"], "reasoning": art["reasoning"],
                       "finish_reason": art["finish_reason"]},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def run_battery(name, outdir):
    """24 greedy prompts, fixed order p00..p23, verbatim capture + sha256."""
    with urllib.request.urlopen(measure.URL + "/v1/models", timeout=30) as r:
        model = json.loads(r.read().decode())["data"][0]["id"]
    ok, failed = 0, 0
    aborted_at = None
    t_bat = time.time()
    for i in range(24):
        if state["failpat"] or time.time() - t_bat > BATTERY_TIMEOUT:
            aborted_at = i
            log("battery aborted at p%02d (failpat=%s, wall=%.0fs)" % (
                i, state["failpat"], time.time() - t_bat))
            break
        base_art = json.load(open(os.path.join(BASELINE, "p%02d.json" % i)))
        payload = {"model": model, "messages": [{"role": "user", "content": base_art["prompt"]}],
                   "temperature": 0, "top_k": 1, "max_tokens": base_art["max_tokens"],
                   "stream": False}
        t0 = time.time()
        err = None
        resp = None
        try:
            req = urllib.request.Request(measure.URL + "/v1/chat/completions",
                                         data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=PROMPT_TIMEOUT) as r:
                resp = json.loads(r.read().decode())
        except Exception as e:
            err = "%s: %s" % (type(e).__name__, e)
        wall = time.time() - t0
        art = {"idx": base_art["idx"], "kind": base_art["kind"], "prompt": base_art["prompt"],
               "max_tokens": base_art["max_tokens"], "greedy_mode": "temp0_topk1",
               "wall_s": round(wall, 2), "error": err}
        if resp is not None:
            ch = (resp.get("choices") or [{}])[0]
            msg = ch.get("message") or {}
            art["reasoning"] = msg.get("reasoning_content") or ""
            art["content"] = msg.get("content") or ""
            art["finish_reason"] = ch.get("finish_reason")
            art["usage"] = resp.get("usage")
            art["sha256"] = out_sha(art)
        with open(os.path.join(outdir, "p%02d.json" % i), "w") as f:
            json.dump(art, f, indent=1, ensure_ascii=False)
        if "sha256" in art:
            with open(os.path.join(outdir, "p%02d.txt" % i), "w", encoding="utf-8") as f:
                f.write("### reasoning\n%s\n\n### content\n%s\n" % (art["reasoning"], art["content"]))
            ok += 1
            log("[bat %s] p%02d ok %5.1fs sha=%s fin=%s r=%dc c=%dc" % (
                name, i, wall, art["sha256"][:8], art["finish_reason"],
                len(art["reasoning"]), len(art["content"])))
        else:
            failed += 1
            log("[bat %s] p%02d FAILED %s" % (name, i, err))
            if err and any(s in err for s in ("Connection", "Remote", "reset")):
                aborted_at = i
                break
    return {"ok": ok, "failed": failed, "aborted_at": aborted_at,
            "wall_s": round(time.time() - t_bat, 1), "model": model}


def teardown(proc):
    killpg_quiet(proc, signal.SIGTERM)
    deadline = time.time() + TEARDOWN_GRACE
    while time.time() < deadline:
        if proc.poll() is not None:
            break
        time.sleep(1)
    if proc.poll() is None:
        log("SIGTERM grace expired -> SIGKILL process group")
        killpg_quiet(proc, signal.SIGKILL)
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


def parse_thrs(lines):
    vals = []
    for l in lines:
        m = re.search(r"input throughput \(token/s\):\s*([0-9]+\.?[0-9]*)", l)
        if m:
            vals.append(float(m.group(1)))
    return vals


def repair_our_leftovers():
    """Kill only OUR abandoned boot (port marker 18801); foreign never touched."""
    out = subprocess.run(["pgrep", "-a", "-f", "ft serve --port 18801"],
                         capture_output=True, text=True).stdout
    ours = [l.strip() for l in out.splitlines() if l.strip()]
    if ours:
        log("repairing OUR leftovers from the aborted attempt: %s" % ours)
        sh("pkill -9 -f 'ft serve --port 18801'")
        time.sleep(4)
    for _ in range(10):
        if measure.gpu_used_mib() < MAX_VRAM_STOP_MIB:
            break
        time.sleep(2)
    return ours


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, choices=[n for n, _ in BOOTS])
    args = ap.parse_args()
    os.makedirs(QDIR, exist_ok=True)
    repair_our_leftovers()
    rc = 0
    summary = {}
    for name, tile in BOOTS:
        if args.only and name != args.only:
            continue
        outdir = os.path.join(QDIR, name)
        os.makedirs(outdir, exist_ok=True)
        logpath = os.path.join(LOGDIR, "bat_%s.log" % name)
        state["failpat"] = None
        okq, why = quiet_gate(name)
        if not okq:
            log("FOREIGN VRAM HOLDER -> STOP before %s: %s" % (name, why))
            rc = 6
            break
        proc, cmd = boot(name, tile, logpath)
        rec = {"name": name, "tile": tile, "mode": "plain (no nsys)", "cmd": cmd,
               "env": {"FREETOKEN_GGUF_DENSE_MTILE": tile, "FREETOKEN_GGUF_MOE_MTILE": "32",
                       "FREETOKEN_GGUF_KDA_NSPLIT": None, "NSYS_SESSION": None},
               "boot": {}, "prefill": {}, "battery": {}}
        wd = threading.Thread(target=watchdog, args=(proc, logpath), daemon=True)
        wd.start()
        boot_ok = False
        try:
            okr, el, why = wait_ready(logpath, proc)
            rec["boot"] = {"ok": okr, "ready_s": round(el, 1), "why": why}
            log("%s boot %s in %.1fs (%s)" % (name, "ready" if okr else "FAILED", el, why))
            if okr:
                boot_ok = True
                rec["env_pins"] = server_env_pins(proc.pid)
                rec["gpu_after_boot_mib"] = measure.gpu_used_mib()
                time.sleep(2)
                pf = measure.do_prefill(os.path.join(TUNING, "filler.txt"), logpath)
                vals = parse_thrs(pf["throughput_lines"])
                steady = vals[1:-2] if len(vals) > 3 else vals
                med = round(statistics.median(steady), 2) if steady else None
                dev = round((med - PREFILL_ANCHOR[tile]) / PREFILL_ANCHOR[tile] * 100, 2) if med else None
                pf["thr_values"] = vals
                pf["median_c2_c7"] = med
                pf["anchor_expected"] = PREFILL_ANCHOR[tile]
                pf["anchor_dev_pct"] = dev
                rec["prefill"] = pf
                log("%s prefill median c2..c7 = %s tok/s (anchor %s, dev %s%%)" % (
                    name, med, PREFILL_ANCHOR[tile], dev))
                if med is not None and abs(dev) > 5.0:
                    log("WARNING: prefill anchor deviation > 5%% on %s - environment drift?" % name)
                if pf["error"] or pf["watch_in_req"]:
                    log("PREFILL FAILED on %s: %s" % (name, pf["error"]))
                    rec["boot"] = {"ok": False, "why": "prefill: %s" % pf["error"]}
                    boot_ok = False
                    rc = 3
                else:
                    rec["battery"] = run_battery(name, outdir)
                    if rec["battery"]["failed"] or rec["battery"]["aborted_at"] is not None:
                        rc = max(rc, 4)
                    if state["failpat"]:
                        rc = max(rc, 2)
            else:
                try:
                    rec["log_tail"] = open(logpath, "rb").read()[-2500:].decode("utf-8", "replace")
                except OSError:
                    rec["log_tail"] = ""
                rc = 3
        finally:
            state["tearing_down"] = True
            rec["teardown"] = teardown(proc)
            for _ in range(10):
                if measure.gpu_used_mib() < MAX_VRAM_STOP_MIB:
                    break
                time.sleep(2)
            rec["post_census"] = census("postflight", name)
            wd.join(timeout=10)
            with open(os.path.join(QDIR, "%s_boot.json" % name), "w") as f:
                json.dump(rec, f, indent=1, ensure_ascii=False)
            summary[name] = {"boot": rec["boot"], "median_c2_c7": rec["prefill"].get("median_c2_c7"),
                             "battery_ok": rec["battery"].get("ok"),
                             "battery_failed": rec["battery"].get("failed"),
                             "teardown": rec["teardown"]}
            left_ours = [l for l in rec["teardown"]["leftovers"]
                         if "llama-swap" not in l]
            if left_ours:
                log("LEFTOVER processes after %s: %s" % (name, left_ours))
                rc = max(rc, 5)
            elif rec["teardown"]["leftovers"]:
                log("post teardown processes (benign standing proxy): %s"
                    % rec["teardown"]["leftovers"])
        if not boot_ok or rc:
            log("stopping the wave (rc=%s) after %s" % (rc, name))
            break
    with open(os.path.join(QDIR, "arbitration_run_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    log("run rc=%s" % rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
