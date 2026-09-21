#!/usr/bin/env python3
"""Dense m-tile A/B driver (dense-q80-gemm replan wave 2, Candidate A sweep).

Single variable per boot: FREETOKEN_GGUF_DENSE_MTILE (mtile4: unset -> 4,
mtile8/16/32/64: the literal value). MTILE=32 MoE anchor fixed on every boot.
NSPLIT is NOT set anywhere - the committed default 2 applies (9f09596).
measure.py reused unchanged except FT swapped for the nsys wrapper (D09).

Per boot: /ready -> 65,585-token fill (nsys window = fill chunks 3..7 of the
steady span) -> streamed decode re-send (radix HIT expected) -> 6 greedy
prompts (cross-boot bitwise diff inputs) -> teardown -> report moved to
TASKDIR/<name>.nsys-rep.
"""
import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import urllib.request

TASKDIR = "/media/ai/src/FreeToken/.tasks/dense-q80-gemm"
TUNING = "/media/ai/src/FreeToken/.tasks/ft-gguf-serve-tuning"
REPO = "/media/ai/src/FreeToken"
LOGDIR = os.path.join(TUNING, "logs")
sys.path.insert(0, TUNING)
import measure  # noqa: E402

# the server must boot UNDER the nsys wrapper (D09); forgetting this swap
# yields a plain boot: no injection banner, start/stop silently no-op
measure.FT = os.path.join(TASKDIR, "ft_nsys_nsplit.sh")

NSYS = "/usr/local/bin/nsys"
# FAILPAT: measure.WATCH + the backend-worker-exited variant (process-hygiene).
WATCH = re.compile(measure.WATCH.pattern + r"|backend worker .* exited")
START_AFTER_LINES = 2  # capture begins inside fill chunk 3
STOP_AFTER_LINES = 6   # capture ends inside fill chunk 7
HARD_DEADLINE_S = 2400

TILES = ["4", "8", "16", "32", "64"]

# Greedy prompts for the cross-boot bitwise diff (identical to the nsplit
# wave inputs); identical order/content/settings in every boot.
PROMPTS = [
    "What is 17 * 23? Answer with just the number.",
    "Name the capital of France and one famous landmark there. One sentence.",
    "Write a haiku about rain on a tin roof.",
    "Explain in two sentences why the sky is blue.",
    "Translate 'good morning' into German, French, and Spanish.",
    "Count from 1 to 20, comma separated.",
]
PROMPT_MAX_TOKENS = 96
PROMPT_TIMEOUT = 300

state = {"started": False, "stopped": False, "failed": None, "n_thr": 0,
         "t_start": None, "t_stop": None}


def nsys(*args):
    try:
        return subprocess.run([NSYS, *args], capture_output=True, text=True,
                              timeout=180, cwd=TASKDIR)
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
        n = full.count("input throughput")
        state["n_thr"] = n
        m = WATCH.search(full[-20000:])
        if m:
            state["failed"] = m.group(0)
            return
        if n >= START_AFTER_LINES and not state["started"]:
            r = nsys("start", "--session", sys.argv[sys.argv.index("--name") + 1] + "cap",
                     "--sample=none")
            state["started"] = True
            state["t_start"] = time.time()
            state["start_rc"] = r.returncode
            state["start_err"] = (r.stderr or r.stdout)[-600:]
        if n >= STOP_AFTER_LINES and state["started"] and not state["stopped"]:
            time.sleep(0.3)
            r = nsys("stop", "--session", sys.argv[sys.argv.index("--name") + 1] + "cap")
            state["stopped"] = True
            state["t_stop"] = time.time()
            state["stop_rc"] = r.returncode
            state["stop_err"] = (r.stderr or r.stdout)[-600:]
            return
    if not state["stopped"] and state["failed"] is None:
        state["failed"] = "hard deadline without capture stop"


def server_env_pins(pid):
    """Pin the ACTUAL server-process env (validity: knob absent/present)."""
    try:
        raw = open("/proc/%d/environ" % pid, "rb").read().decode("utf-8", "replace")
    except OSError:
        return {}
    return dict(x.split("=", 1) for x in raw.split("\0")
                if "=" in x and (x.startswith("FREETOKEN") or x.startswith("NSYS_")))


def do_prompts(logpath):
    out = []
    for i, content in enumerate(PROMPTS):
        rq_path = os.path.join(TASKDIR, "req_mtile_%d.json" % i)
        measure.build_req(rq_path, content, PROMPT_MAX_TOKENS, False)
        off = os.path.getsize(logpath)
        t0 = time.time()
        err = None
        choices = None
        usage = {}
        try:
            req = urllib.request.Request(measure.URL + "/v1/chat/completions",
                                         data=open(rq_path, "rb").read(),
                                         headers={"Content-Type": "application/json"})
            resp = urllib.request.urlopen(req, timeout=PROMPT_TIMEOUT)
            body = json.loads(resp.read())
            resp.close()
            choices = body.get("choices")
            usage = body.get("usage") or {}
        except Exception as e:
            err = "%s: %s" % (type(e).__name__, e)
        wall = time.time() - t0
        lines = measure.new_log_lines(logpath, off)
        blob = json.dumps(choices, sort_keys=True) if choices is not None else ""
        out.append({
            "i": i, "prompt": content, "wall_s": round(wall, 2), "error": err,
            "output_sha256": hashlib.sha256(blob.encode()).hexdigest() if blob else None,
            "choices": choices,
            "completion_tokens": usage.get("completion_tokens"),
            "new_token": measure.parse_tok(lines, "new-token"),
            "cached_token": measure.parse_tok(lines, "cached-token"),
        })
        if err:
            break
    return out


def find_report(session):
    pats = []
    for d in (REPO, TASKDIR, os.path.expanduser("~")):
        pats += [os.path.join(d, "report1.nsys-rep"), os.path.join(d, session + ".nsys-rep")]
    cands = [p for p in pats if os.path.exists(p)]
    return max(cands, key=os.path.getmtime) if cands else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True,
                    choices=["mtile" + t for t in TILES] + ["mtile4b"])
    a = ap.parse_args()
    name = a.name
    tile = "4" if name == "mtile4b" else name[len("mtile"):]
    session = name + "cap"
    report = os.path.join(TASKDIR, name + ".nsys-rep")
    logpath = os.path.join(LOGDIR, name + ".log")

    env_pairs = ["FREETOKEN_GGUF_MOE_MTILE=32", "NSYS_SESSION=" + session]
    if tile == "4":
        if name == "mtile4b":
            env_pairs.append("FREETOKEN_GGUF_DENSE_MTILE=4")
            os.environ["FREETOKEN_GGUF_DENSE_MTILE"] = "4"
        else:
            # unset = knob value 4 (the historical baseline); never inherit
            os.environ.pop("FREETOKEN_GGUF_DENSE_MTILE", None)
    else:
        env_pairs.append("FREETOKEN_GGUF_DENSE_MTILE=" + tile)
        os.environ["FREETOKEN_GGUF_DENSE_MTILE"] = tile
    # NSPLIT default 2 is committed (9f09596); NEVER set the knob here
    os.environ.pop("FREETOKEN_GGUF_KDA_NSPLIT", None)

    if os.path.exists(report):
        os.remove(report)
    for stale in (os.path.join(TASKDIR, "report1.nsys-rep"),
                  os.path.join(REPO, "report1.nsys-rep")):
        if os.path.exists(stale):
            os.remove(stale)

    proc, lf, t0 = measure.start_server(shlex.split(
        "--memory-ratio 0.85 --max-prefill-length 8191 --max-running-requests 1"),
        logpath, env_pairs)
    wd = threading.Thread(target=watchdog, args=(proc, logpath), daemon=True)
    wd.start()
    out = {}
    try:
        out, ok = measure.common_tail(logpath, proc, t0, name)
        out["name"] = name
        out["env_pairs"] = env_pairs
        out["server_env_pins"] = server_env_pins(proc.pid)
        try:
            with open(logpath, encoding="utf-8", errors="replace") as f:
                out["graph_capture_lines"] = [l.strip() for l in f
                                              if "Start capturing CUDA graphs" in l
                                              or "Capturing graphs: bs" in l][:10]
                f.seek(0)
                out["nsys_banner_lines"] = [l.strip() for l in f
                                            if "Collecting data" in l][:3]
        except OSError:
            out["graph_capture_lines"] = []
        if ok:
            time.sleep(2)
            out["prefill"] = measure.do_prefill(os.path.join(TUNING, "filler.txt"), logpath)
            out["prefill_thr_summary"] = measure.thr_summary(out["prefill"]["throughput_lines"])
            if not out["prefill"]["error"] and not out["prefill"]["watch_in_req"]:
                out["decode"] = measure.do_decode(os.path.join(TUNING, "filler.txt"), logpath)
                out["prompts"] = do_prompts(logpath)
        else:
            out["log_tail"] = "\n".join(measure.new_log_lines(
                logpath, max(0, os.path.getsize(logpath) - 4000)))
    finally:
        out["capture"] = dict(state)
        out["teardown"] = measure.teardown(proc)
        wd.join(timeout=10)
        # stop generates the report synchronously into the stop-call cwd
        # (probe evidence); grab it before tearing the session down
        if state["started"] and state["stopped"]:
            src = find_report(session)
            if src:
                try:
                    os.replace(src, report)
                    out["report_moved_from"] = src
                    out["report_moved_size"] = os.path.getsize(report)
                except OSError as e:
                    out["report_move_error"] = str(e)
            else:
                out["report_move_error"] = "no report file found after stop"
        if state["started"] and not state["stopped"]:
            out["capture_cancel_rc"] = nsys("cancel", "--session", session).returncode
        elif state["started"]:
            out["capture_shutdown_rc"] = nsys("shutdown", "--session", session).returncode
        with open(os.path.join(TASKDIR, "%s_run.json" % name), "w") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        print(json.dumps({"name": name, "capture": state,
                          "report_exists": os.path.exists(report),
                          "teardown": out.get("teardown"),
                          "thr": out.get("prefill_thr_summary"),
                          "decode": {k: out.get("decode", {}).get(k) for k in
                                     ("steady_tok_s", "cached_token", "error")}
                          if out.get("decode") else None}, indent=1))
    if state["failed"]:
        sys.exit(3)


if __name__ == "__main__":
    main()
