#!/usr/bin/env python3
"""v3a liveness probe (tile 16): capture-from-launch nsys session.

Adapted from v2ab_probe.py. One ft serve boot (winner flags,
FREETOKEN_GGUF_MOE_MTILE=16, PYTHONPATH sitecustomize for the bare-logger
marker) under `nsys launch` (--cuda-graph-trace=node). Sends the 65,585-token
prefill then one 65k radix-hit decode, then `nsys stop`. Analysis:
v3a_probe_analyze.py.
"""
import glob
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time

VDIR = "/media/ai/src/FreeToken/.tasks/mmq-v3-stationary-moe"
TUNING = "/media/ai/src/FreeToken/.tasks/ft-gguf-serve-tuning"
sys.path.insert(0, TUNING)
import measure  # noqa: E402

measure.FT = os.path.join(VDIR, "ft_nsys_v3.sh")

NSYS = "/usr/local/bin/nsys"
SESSION = "v3aprobe"
WATCH = re.compile(measure.WATCH.pattern)
HARD_DEADLINE_S = 1800

state = {"failed": None, "n_thr": 0, "start_rc": None, "stop_rc": None,
         "start_err": "", "stop_err": ""}


def nsys(*args):
    try:
        return subprocess.run([NSYS, *args], capture_output=True, text=True,
                              timeout=900, cwd=VDIR)
    except Exception as e:
        class R:
            returncode = -1
            stderr = "%s: %s" % (type(e).__name__, e)
            stdout = ""
        return R()


def watchdog(proc, logpath):
    t0 = time.time()
    while time.time() - t0 < HARD_DEADLINE_S:
        time.sleep(2)
        try:
            full = open(logpath, "rb").read().decode("utf-8", "replace")
        except OSError:
            full = ""
        state["n_thr"] = full.count("input throughput")
        m = WATCH.search(full[-20000:])
        if m:
            state["failed"] = m.group(0)
            return
        if proc.poll() is not None:
            return
    if state["stop_rc"] is None and state["failed"] is None:
        state["failed"] = "hard deadline"


def main():
    nsys("shutdown", "--session", SESSION)
    nsys("delete", "--session", SESSION)
    logpath = os.path.join(TUNING, "logs", "ab_v3a_probe.log")
    extra = "--memory-ratio 0.85 --max-prefill-length 8191 --max-running-requests 1"
    env = ["FREETOKEN_GGUF_MOE_MTILE=16",
           "PYTHONPATH=" + "/media/ai/src/FreeToken/.tasks/mmq-prefill-kernel/probe_sitecustomize"]
    proc, lf, t0 = measure.start_server(shlex.split(extra), logpath, env)
    wd = threading.Thread(target=watchdog, args=(proc, logpath), daemon=True)
    wd.start()
    out = {}
    try:
        out, ok = measure.common_tail(logpath, proc, t0, "v3aprobe")
        if ok:
            r = nsys("start", "--session", SESSION)
            state["start_rc"] = r.returncode
            state["start_err"] = (r.stderr or r.stdout)[-300:]
            time.sleep(2)
            out["prefill"] = measure.do_prefill(os.path.join(TUNING, "filler.txt"), logpath)
            out["prefill_thr_summary"] = measure.thr_summary(out["prefill"]["throughput_lines"])
            if not out["prefill"]["error"] and not out["prefill"]["watch_in_req"]:
                time.sleep(2)
                out["decode"] = measure.do_decode(os.path.join(TUNING, "filler.txt"), logpath)
                time.sleep(2)
        else:
            out["log_tail"] = "\n".join(measure.new_log_lines(
                logpath, max(0, os.path.getsize(logpath) - 4000)))
    finally:
        r = nsys("stop", "--session", SESSION)
        state["stop_rc"] = r.returncode
        state["stop_err"] = (r.stderr or r.stdout)[-300:]
        if r.returncode != 0:
            nsys("cancel", "--session", SESSION)
        nsys("shutdown", "--session", SESSION)
        out["capture"] = dict(state)
        out["teardown"] = measure.teardown(proc)
        wd.join(timeout=10)
        try:
            txt = open(logpath, encoding="utf-8", errors="replace").read()
            out["marker_lines"] = [l.strip()[-160:] for l in txt.splitlines()
                                   if "grouped mmq prefill active" in l]
            out["thr_lines"] = [l.strip()[-160:] for l in txt.splitlines()
                                if "input throughput" in l]
        except Exception:
            pass
        reps = sorted(glob.glob(os.path.join(VDIR, "report*.nsys-rep")),
                      key=os.path.getmtime)
        out["report"] = reps[-1] if reps else None
        with open(os.path.join(VDIR, "v3a_probe.json"), "w") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        print(json.dumps({"capture": {k: state[k] for k in
                                      ("failed", "n_thr", "start_rc", "stop_rc")},
                          "report": out["report"],
                          "marker": out.get("marker_lines"),
                          "prefill_ctx": out.get("prefill_thr_summary"),
                          "decode_steady": (out.get("decode") or {}).get("steady_tok_s"),
                          "decode_cached": (out.get("decode") or {}).get("cached_token"),
                          "teardown": out.get("teardown")}, indent=1))
    if state["failed"]:
        sys.exit(3)


if __name__ == "__main__":
    main()
