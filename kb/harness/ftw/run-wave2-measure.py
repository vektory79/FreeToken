#!/usr/bin/env python3
"""Wave-2 hardware validation for .tasks/ftw-gguf-fastpath (GGUF->FTW serve fastpath).

Measurement only - NO code changes, NO commits. One serialized FTW boot with the
campaign-winner serving flags (port 18801):
  preflight census -> boot (FAILPAT watchdog, hard timeouts) -> boot-log sanity
  capture -> 65,585-token prefill (per-chunk server-log throughput) -> 3x decode
  on the radix-cached prefix -> 24-prompt greedy battery (same prompts as the
  path-a / dense-q80 arbitration batteries) -> SIGTERM/SIGKILL teardown ->
  postflight census.

Methodology mirrors .tasks/dense-q80-gemm/arbitration_run.py + measure.py so the
numbers are comparable with the bare-GGUF references (t4a/t64a/t64b boots and
battery-post-fix). Process-hygiene protocol applied: census before/after,
FAILPAT watchdog + "Tried to allocate" capture, SIGTERM 30s -> SIGKILL group,
strictly serial single run.

Exit codes: 0 ok; 2 FAILPAT; 3 boot fail; 4 prefill/decode fail; 5 battery fail;
6 foreign-process stop; 7 teardown leftovers.
"""
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
TASKDIR = os.path.join(ROOT, ".tasks/ftw-gguf-fastpath")
OUTDIR = os.path.join(TASKDIR, "run-wave2")
TUNING = os.path.join(ROOT, ".tasks/ft-gguf-serve-tuning")
LOGPATH = os.path.join(OUTDIR, "ftw_boot.log")
FTW = "/media/ai/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL-FTW"
PORT = 18801
URL = "http://127.0.0.1:%d" % PORT
FILLER = os.path.join(TUNING, "filler.txt")
BASELINE = os.path.join(ROOT, ".tasks/gguf-glm5next-path-a/verification/phase6/battery-post-fix")

BOOT_TIMEOUT = 420
PREFILL_TIMEOUT = 900
DECODE_TIMEOUT = 600
BATTERY_TIMEOUT = 3000
PROMPT_TIMEOUT = 600
TEARDOWN_GRACE = 30

FAILPAT = re.compile(r"OutOfMemoryError|AssertionError|Backend worker is gone|"
                     r"backend worker .* exited|CUDA error|Traceback \(most recent call last\)")
CENSUS_PAT = r"ft serve|benchbw"
FOREIGN_HARD_PAT = re.compile(r"\bft serve|benchbw")
BENIGN_COMPUTE_APPS = ("hoptodesk",)
MAX_VRAM_STOP_MIB = 2000
# Same env pins as the t64a/t64b reference boots (= the committed production
# defaults); NSPLIT deliberately unset -> committed default 2.
ENV_PINS = {"FREETOKEN_GGUF_MOE_MTILE": "32", "FREETOKEN_GGUF_DENSE_MTILE": "64"}

SANITY_PATTERNS = [
    r"Loading expert banks \(FTW\)",
    r"gguf expert bank ggml types",
    r"moe_cache_size=\d+|num_pages=\d+|Allocating .*tokens for KV",
    r"Free memory (before|after)",
    r"fetching .*PCIe|fetch fraction",
    r"CPU MoE executor ready",
    r"MoE signature group",
    r"torch intra-op threads",
    r"Resolved config|Auto-selected attention",
    r"floor|fail-fast|cannot fund",
    r"Started server process|Uvicorn running",
    r"expert banks: slow path|expert banks",
]

state = {"failpat": None, "tearing_down": False}


def log(msg):
    print("[wave2 %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def pgrep_census():
    # list-form: a shell-wrapped pgrep matches its own /bin/sh -c wrapper
    out = subprocess.run(["pgrep", "-a", "-f", CENSUS_PAT],
                         capture_output=True, text=True).stdout
    return [l.strip() for l in out.splitlines()
            if l.strip() and "pgrep" not in l]


def used_mib():
    out = sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits").stdout.strip()
    try:
        return int(out.splitlines()[0])
    except Exception:
        return -1


def census(tag):
    procs = pgrep_census()
    vram = sh("nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader").stdout.strip()
    apps = sh("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader").stdout.strip()
    sems = sh("ls /dev/shm | grep -c '^sem\\.mp-'").stdout.strip() or "0"
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tag": tag,
           "procs": procs, "gpu_vram": vram, "gpu_compute_apps": apps, "sem_mp": sems}
    with open(os.path.join(OUTDIR, "census.log"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    log("[census %s] procs=%s vram=[%s] apps=[%s] sem.mp-=%s" % (
        tag, procs or "none", vram, apps, sems))
    return rec


def preflight_gate():
    rec = census("preflight")
    hard = [p for p in rec["procs"] if FOREIGN_HARD_PAT.search(p)]
    if hard:
        return False, "foreign ft serve/benchbw processes: %s" % hard
    apps = [l for l in (rec["gpu_compute_apps"] or "").splitlines()
            if l.strip() and not any(b in l for b in BENIGN_COMPUTE_APPS)]
    if apps:
        return False, "foreign GPU compute apps: %s" % apps
    if used_mib() >= MAX_VRAM_STOP_MIB:
        return False, "VRAM holder without a matched process: %s" % rec["gpu_vram"]
    listeners = sh("ss -ltn | grep -E ':(18080|18081|18801)\\b' || true").stdout.strip()
    if "18801" in listeners:
        return False, "port 18801 already listening: %s" % listeners
    return True, None


def killpg_quiet(proc, sig):
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except (ProcessLookupError, PermissionError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass


def watchdog(proc):
    t0 = time.time()
    while time.time() - t0 < BOOT_TIMEOUT + BATTERY_TIMEOUT + PREFILL_TIMEOUT + 120:
        time.sleep(2)
        if state["tearing_down"] or proc.poll() is not None:
            return
        try:
            tail = open(LOGPATH, "rb").read()[-20000:].decode("utf-8", "replace")
        except OSError:
            continue
        m = FAILPAT.search(tail)
        if m:
            state["failpat"] = m.group(0)
            log("WATCHDOG FAILPAT hit: %s" % m.group(0))
            for line in tail.splitlines():
                if FAILPAT.search(line) or "Tried to allocate" in line:
                    log("  | %s" % line[:240])
            killpg_quiet(proc, signal.SIGKILL)
            return


def wait_ready(proc, timeout=BOOT_TIMEOUT):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False, time.time() - t0, "process exited rc=%s" % proc.returncode
        if state["failpat"]:
            return False, time.time() - t0, "failpat: %s" % state["failpat"]
        try:
            tail = open(LOGPATH, "rb").read()[-12000:].decode("utf-8", "replace")
            m = FAILPAT.search(tail)
            if m:
                return False, time.time() - t0, "failpat(boot): %s" % m.group(0)
        except OSError:
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


def boot():
    for k in list(os.environ):
        if k.startswith("FREETOKEN_") or k == "NSYS_SESSION":
            del os.environ[k]
    os.environ.update(ENV_PINS)
    cmd = ["uv", "run", "ft", "serve", "--port", str(PORT), "--host", "127.0.0.1",
           "--model-path", FTW,
           "--moe-cache-auto", "--kv-reserve-tokens", "500000",
           "--kv-cache-dtype", "fp8", "--memory-ratio", "0.85",
           "--max-prefill-length", "8191", "--moe-strategy", "hybrid",
           "--moe-cpu-threads", "16", "--max-running-requests", "1"]
    lf = open(LOGPATH, "w")
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT,
                            env=dict(os.environ), start_new_session=True)
    with open(LOGPATH + ".pid", "w") as f:
        f.write(str(proc.pid))
    log("boot cmd: %s" % " ".join(cmd))
    return proc, t0


def new_log_lines(offset):
    with open(LOGPATH, "rb") as f:
        f.seek(offset)
        return f.read().decode("utf-8", "replace").splitlines()


def full_log_text():
    with open(LOGPATH, "rb") as f:
        return f.read().decode("utf-8", "replace")


def served_model_name():
    try:
        with urllib.request.urlopen(URL + "/v1/models", timeout=30) as r:
            return json.loads(r.read().decode())["data"][0]["id"]
    except Exception:
        return os.path.basename(FTW)


def grep_sanity():
    txt = full_log_text()
    res = {}
    for pat in SANITY_PATTERNS:
        res[pat] = [l.strip() for l in txt.splitlines() if re.search(pat, l)][-40:]
    return res


def parse_tok(lines, kind):
    vals = [int(m.group(1)) for l in lines for m in [re.search(r"#%s:\s*(\d+)" % kind, l)] if m]
    return vals[-1] if vals else None


def parse_prefill_chunks(lines):
    chunks = []
    for l in lines:
        mt = re.search(r"#new-token:\s*(\d+)", l)
        mv = re.search(r"input throughput \(token/s\):\s*([0-9]+\.?[0-9]*)", l)
        if mt and mv:
            chunks.append({"new_token": int(mt.group(1)), "tok_s": float(mv.group(1))})
    return chunks


def do_prefill(model, max_tokens=4, timeout=PREFILL_TIMEOUT):
    content = open(FILLER, encoding="utf-8").read()
    payload = {"model": model, "messages": [{"role": "user", "content": content}],
               "max_tokens": max_tokens, "temperature": 0, "stream": False}
    off = os.path.getsize(LOGPATH)
    t0 = time.time()
    err, usage = None, {}
    try:
        req = urllib.request.Request(URL + "/v1/chat/completions",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            usage = (json.loads(r.read().decode()).get("usage")) or {}
    except Exception as e:
        err = "%s: %s" % (type(e).__name__, e)
    wall = time.time() - t0
    lines = new_log_lines(off)
    chunks = parse_prefill_chunks(lines)
    full = [c["tok_s"] for c in chunks if c["new_token"] >= 7000]
    median_full = round(statistics.median(full[1:-1]), 1) if len(full) > 3 else None
    return {"wall_s": round(wall, 2), "error": err, "usage": usage,
            "chunks": chunks, "median_full_excl_last": median_full,
            "new_token": parse_tok(lines, "new-token"), "cached_token": parse_tok(lines, "cached-token"),
            "watch_in_req": bool(FAILPAT.search("\n".join(lines)))}


def do_decode(model, max_tokens=64, timeout=DECODE_TIMEOUT):
    content = open(FILLER, encoding="utf-8").read()
    payload = {"model": model, "messages": [{"role": "user", "content": content}],
               "max_tokens": max_tokens, "temperature": 0, "stream": True,
               "stream_options": {"include_usage": True}}
    off = os.path.getsize(LOGPATH)
    t0 = time.time()
    err, times, usage = None, [], {}
    try:
        req = urllib.request.Request(URL + "/v1/chat/completions",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload_s = line[5:].strip()
                if payload_s == "[DONE]":
                    break
                try:
                    obj = json.loads(payload_s)
                except Exception:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                for ch in obj.get("choices") or []:
                    d = ch.get("delta") or {}
                    if d.get("content") or d.get("reasoning_content"):
                        times.append(time.time())
    except Exception as e:
        err = "%s: %s" % (type(e).__name__, e)
    wall = time.time() - t0
    lines = new_log_lines(off)
    ntok = usage.get("completion_tokens") or (len(times) - 1 if len(times) > 1 else len(times))
    steady = p50 = inst_min = None
    if len(times) >= 4:
        tail = times[1:]  # skip the first token-bearing frame (pre-prefill SSE frame class)
        if len(tail) >= 3 and tail[-1] > tail[0]:
            steady = round((len(tail) - 1) / (tail[-1] - tail[0]), 2)
        gaps = sorted(tail[i + 1] - tail[i] for i in range(len(tail) - 1) if tail[i + 1] > tail[i])
        if gaps:
            inst_min = round(1.0 / gaps[-1], 2)
            p50 = round(1.0 / gaps[len(gaps) // 2], 2)
    return {"wall_s": round(wall, 2), "error": err, "completion_tokens": ntok,
            "steady_tok_s": steady, "p50_tok_s": p50, "min_inst_tok_s": inst_min,
            "new_token": parse_tok(lines, "new-token"),
            "cached_token": parse_tok(lines, "cached-token"),
            "watch_in_req": bool(FAILPAT.search("\n".join(lines)))}


def out_sha(art):
    blob = json.dumps({"content": art["content"], "reasoning": art["reasoning"],
                       "finish_reason": art["finish_reason"]},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def run_battery(model, outdir):
    ok = failed = 0
    aborted_at = None
    t_bat = time.time()
    for i in range(24):
        if state["failpat"] or time.time() - t_bat > BATTERY_TIMEOUT:
            aborted_at = i
            log("battery aborted at p%02d (failpat=%s, wall=%.0fs)" % (
                i, state["failpat"], time.time() - t_bat))
            break
        ref = json.load(open(os.path.join(BASELINE, "p%02d.json" % i)))
        payload = {"model": model, "messages": [{"role": "user", "content": ref["prompt"]}],
                   "temperature": 0, "top_k": 1, "max_tokens": ref["max_tokens"],
                   "stream": False}
        t0 = time.time()
        err, resp = None, None
        try:
            req = urllib.request.Request(URL + "/v1/chat/completions",
                                         data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=PROMPT_TIMEOUT) as r:
                resp = json.loads(r.read().decode())
        except Exception as e:
            err = "%s: %s" % (type(e).__name__, e)
        wall = time.time() - t0
        art = {"idx": ref["idx"], "kind": ref["kind"], "prompt": ref["prompt"],
               "max_tokens": ref["max_tokens"], "greedy_mode": "temp0_topk1",
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
            log("[bat ftw] p%02d ok %5.1fs sha=%s fin=%s r=%dc c=%dc" % (
                i, wall, art["sha256"][:8], art["finish_reason"],
                len(art["reasoning"]), len(art["content"])))
        else:
            failed += 1
            log("[bat ftw] p%02d FAILED %s" % (i, err))
            if err and any(s in err for s in ("Connection", "Remote", "reset")):
                aborted_at = i
                break
    return {"ok": ok, "failed": failed, "aborted_at": aborted_at,
            "wall_s": round(time.time() - t_bat, 1), "model": model}


def teardown(proc):
    state["tearing_down"] = True
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
    leftovers = pgrep_census()
    return {"grace_expired": proc.poll() is None, "rc": proc.returncode,
            "leftovers": leftovers, "gpu_mib": used_mib()}


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rc = 0
    okq, why = preflight_gate()
    if not okq:
        log("FOREIGN PROCESS/VRAM HOLDER -> STOP: %s" % why)
        return 6
    proc, t0 = boot()
    wd = threading.Thread(target=watchdog, args=(proc,), daemon=True)
    wd.start()
    out = {"cmd": "ft serve --model-path %s (winner flags, port %d)" % (FTW, PORT)}
    try:
        ready, ready_s, rdy_why = wait_ready(proc)
        out["boot"] = {"ready": ready, "ready_s": round(ready_s, 1), "why": rdy_why}
        log("boot %s in %.1fs (%s)" % ("ready" if ready else "FAILED", ready_s, rdy_why))
        if ready:
            time.sleep(2)
            out["gpu_after_boot_mib"] = used_mib()
            out["sanity"] = grep_sanity()
            model = served_model_name()
            out["served_model"] = model
            log("served model: %s" % model)
            pf = do_prefill(model)
            out["prefill"] = pf
            log("prefill: wall=%.1fs median_full_excl_last=%s err=%s" % (
                pf["wall_s"], pf["median_full_excl_last"], pf["error"]))
            if pf["error"] or pf["watch_in_req"] or not pf["chunks"]:
                rc = max(rc, 4)
            else:
                decodes = []
                for rep in range(3):
                    d = do_decode(model)
                    decodes.append(d)
                    log("decode rep%d: wall=%.1fs steady=%s cached=%s new=%s err=%s" % (
                        rep, d["wall_s"], d["steady_tok_s"], d["cached_token"],
                        d["new_token"], d["error"]))
                    if d["error"] or d["watch_in_req"]:
                        rc = max(rc, 4)
                        break
                out["decode"] = decodes
                bdir = os.path.join(OUTDIR, "ftw")
                os.makedirs(bdir, exist_ok=True)
                out["battery"] = run_battery(model, bdir)
                if out["battery"]["failed"] or out["battery"]["aborted_at"] is not None:
                    rc = max(rc, 5)
        else:
            try:
                out["log_tail"] = full_log_text()[-4000:]
            except OSError:
                out["log_tail"] = ""
            rc = 3
        if state["failpat"]:
            rc = max(rc, 2)
    finally:
        out["teardown"] = teardown(proc)
        for _ in range(10):
            if used_mib() < MAX_VRAM_STOP_MIB:
                break
            time.sleep(2)
        out["post_census"] = census("postflight")
        if out["teardown"]["leftovers"]:
            log("LEFTOVER processes after teardown: %s" % out["teardown"]["leftovers"])
            rc = max(rc, 7)
        wd.join(timeout=10)
        with open(os.path.join(OUTDIR, "wave2_results.json"), "w") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        log("run rc=%s" % rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
