#!/usr/bin/env python3
"""v2 liveness probe (attempt 2): capture-from-launch nsys session.

One ft serve boot (winner flags, FREETOKEN_GGUF_GROUPED_PREFILL=1, PYTHONPATH
sitecustomize for the bare-logger marker) under `nsys launch` (collection from
process start, --cuda-graph-trace=node so decode graph kernels are recorded).
Sends the 65,585-token prefill then one 65k radix-hit decode, then `nsys stop`.
v2ab_probe_analyze2.py asserts kernel-level liveness from the sqlite export:
  - eager grouped MMQ MoE kernels present in the post-boot prefill phase
  - eager moe_vec* instances: ZERO (moe_vec must never run outside graphs)
  - graph-mode moe_vec* instances present (decode steps use moe_vec in graphs;
    boot-time graph capture also produces them, before the boot gap)
No production code is touched.
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

TASKDIR = "/media/ai/src/FreeToken/.tasks/mmq-prefill-kernel"
TUNING = "/media/ai/src/FreeToken/.tasks/ft-gguf-serve-tuning"
sys.path.insert(0, TUNING)
import measure  # noqa: E402

measure.FT = os.path.join(TASKDIR, "ft_nsys_v2.sh")

NSYS = "/usr/local/bin/nsys"
SESSION = "v2probecap"
WATCH = re.compile(measure.WATCH.pattern)
HARD_DEADLINE_S = 1800

state = {"failed": None, "n_thr": 0, "start_rc": None, "stop_rc": None,
         "start_err": "", "stop_err": ""}


def nsys(*args):
    try:
        return subprocess.run([NSYS, *args], capture_output=True, text=True,
                              timeout=900, cwd=TASKDIR)
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
    # clean any stale session from previous attempts
    nsys("shutdown", "--session", SESSION)
    nsys("delete", "--session", SESSION)
    logpath = os.path.join(TUNING, "logs", "ab_v2_probe.log")
    extra = "--memory-ratio 0.85 --max-prefill-length 8191 --max-running-requests 1"
    env = ["FREETOKEN_GGUF_GROUPED_PREFILL=1",
           "PYTHONPATH=" + os.path.join(TASKDIR, "probe_sitecustomize")]
    proc, lf, t0 = measure.start_server(shlex.split(extra), logpath, env)
    wd = threading.Thread(target=watchdog, args=(proc, logpath), daemon=True)
    wd.start()
    out = {}
    try:
        out, ok = measure.common_tail(logpath, proc, t0, "v2probe")
        if ok:
            # start collection BEFORE the first prefill kernel (attempt 1
            # started mid-prefill and eager kernel activities never appeared;
            # attempt 2 skipped start entirely and stop was rejected)
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
        out["capture"] = state
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
        reps = sorted(glob.glob(os.path.join(TASKDIR, "report*.nsys-rep")),
                      key=os.path.getmtime)
        out["report"] = reps[-1] if reps else None
        with open(os.path.join(TASKDIR, "v2ab_probe.json"), "w") as f:
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
