#!/usr/bin/env python3
"""dense-q80 step 0 driver: one ft serve boot (winner flags, MTILE=32) under an
nsys session. Pattern lifted from .tasks/mmq-prefill-kernel/step0_run.py.

measure.py reused unchanged; only FT is swapped for the nsys wrapper. The
watchdog tails the server log: FAILPAT -> hard kill; after the N-th steady
"input throughput" line -> nsys start / stop bounds the capture window to
full prefill chunks (starts inside chunk 3, stops inside chunk 7).
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

TASKDIR = "/media/ai/src/FreeToken/.tasks/dense-q80-gemm"
TUNING = "/media/ai/src/FreeToken/.tasks/ft-gguf-serve-tuning"
sys.path.insert(0, TUNING)
import measure  # noqa: E402

measure.FT = os.path.join(TASKDIR, "ft_nsys_dq80.sh")

NSYS = "/usr/local/bin/nsys"
SESSION = "denseq80cap"
REPORT = os.path.join(TASKDIR, "denseq80.nsys-rep")
# FAILPAT: measure.WATCH + the backend-worker-exited variant (process-hygiene).
WATCH = re.compile(measure.WATCH.pattern + r"|backend worker .* exited")
START_AFTER_LINES = 2  # capture begins inside chunk 3
STOP_AFTER_LINES = 6   # capture ends inside chunk 7
HARD_DEADLINE_S = 1800


def nsys(*args):
    try:
        return subprocess.run([NSYS, *args], capture_output=True, text=True, timeout=180,
                              cwd=TASKDIR)
    except Exception as e:
        class R:
            returncode = -1
            stderr = "%s: %s" % (type(e).__name__, e)
            stdout = ""
        return R()


state = {"started": False, "stopped": False, "failed": None, "n_thr": 0,
         "t_start": None, "t_stop": None}


def watchdog(proc, logpath):
    t0 = time.time()
    while time.time() - t0 < HARD_DEADLINE_S:
        time.sleep(2)
        try:
            full = open(logpath, "rb").read().decode("utf-8", "replace")
        except OSError:
            full = ""
        n = full.count("input throughput")
        state["n_thr"] = n
        m = WATCH.search(full[-20000:])
        if m:
            state["failed"] = m.group(0)
            return
        if n >= START_AFTER_LINES and not state["started"]:
            r = nsys("start", "--session", SESSION, "--sample=none")
            state["started"] = True
            state["t_start"] = time.time()
            state["start_rc"] = r.returncode
            state["start_err"] = (r.stderr or r.stdout)[-600:]
        if n >= STOP_AFTER_LINES and state["started"] and not state["stopped"]:
            time.sleep(0.3)
            r = nsys("stop", "--session", SESSION)
            state["stopped"] = True
            state["t_stop"] = time.time()
            state["stop_rc"] = r.returncode
            state["stop_err"] = (r.stderr or r.stdout)[-600:]
            return
    if not state["stopped"] and state["failed"] is None:
        state["failed"] = "hard deadline without capture stop"


def main():
    logpath = os.path.join(TUNING, "logs", "dq80_nsys.log")
    a = argparse.Namespace(
        name="step0", log=os.path.basename(logpath),
        extra="--memory-ratio 0.85 --max-prefill-length 8191 --max-running-requests 1",
        env=["FREETOKEN_GGUF_MOE_MTILE=32"])
    if os.path.exists(REPORT):
        os.remove(REPORT)
    proc, lf, t0 = measure.start_server(shlex.split(a.extra), logpath, a.env)
    wd = threading.Thread(target=watchdog, args=(proc, logpath), daemon=True)
    wd.start()
    out = {}
    try:
        out, ok = measure.common_tail(logpath, proc, t0, a.name)
        if ok:
            time.sleep(2)
            out["prefill"] = measure.do_prefill(os.path.join(TUNING, "filler.txt"), logpath)
            out["prefill_thr_summary"] = measure.thr_summary(out["prefill"]["throughput_lines"])
        else:
            out["log_tail"] = "\n".join(measure.new_log_lines(
                logpath, max(0, os.path.getsize(logpath) - 4000)))
    finally:
        out["capture"] = {k: v for k, v in state.items()}
        out["teardown"] = measure.teardown(proc)
        wd.join(timeout=10)
        if state["started"] and not state["stopped"]:
            out["capture_cancel_rc"] = nsys("cancel", "--session", SESSION).returncode
        elif state["started"]:
            out["capture_shutdown_rc"] = nsys("shutdown", "--session", SESSION).returncode
        with open(os.path.join(TASKDIR, "step0_run.json"), "w") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        print(json.dumps({"capture": state, "report_exists": os.path.exists(REPORT),
                          "teardown": out.get("teardown"),
                          "thr": out.get("prefill_thr_summary")}, indent=1))
    if state["failed"]:
        sys.exit(3)


if __name__ == "__main__":
    main()
