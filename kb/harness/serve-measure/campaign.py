#!/usr/bin/env python3
"""Serial orchestrator for the remaining A/B campaign steps (runs 2-6)."""
import json
import os
import re
import subprocess
import sys
import time

DIR = "/media/ai/src/FreeToken/.tasks/ft-gguf-serve-tuning"
MEASURE = os.path.join(DIR, "measure.py")
PY = sys.executable or "python3"


def log(msg):
    print("[campaign] %s" % msg, flush=True)


def run_measure(mode, name, logname, extra="", env=None, extra_args=None):
    cmd = [PY, MEASURE, mode, "--name", name, "--log", logname]
    if extra:
        cmd += ["--extra", extra]
    for k, v in (env or {}).items():
        cmd += ["--env", "%s=%s" % (k, v)]
    for ea in extra_args or []:
        cmd += ea
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = r.stdout
    m = re.search(r"\{.*\}", out, re.S)
    js = {}
    if m:
        try:
            js = json.loads(m.group(0))
        except Exception as e:
            js = {"parse_error": str(e)}
    js["_rc"] = r.returncode
    js["_wall_s"] = round(time.time() - t0, 1)
    if r.returncode != 0:
        js["_stderr_tail"] = r.stderr[-1500:]
    with open(os.path.join(DIR, "results_%s.json" % name), "w") as f:
        json.dump(js, f, indent=1, ensure_ascii=False)
    log("step %s done rc=%s" % (name, r.returncode))
    return js


def wait_run1(timeout=1800):
    p1 = os.path.join(DIR, "run1_base.json")
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            txt = open(p1).read()
            if '"teardown"' in txt:
                js = json.loads(txt[txt.index("{"):])
                if not js.get("teardown", {}).get("leftover"):
                    return js
        except Exception:
            pass
        time.sleep(10)
    return {}


def filler_fix(run1):
    nt = (run1.get("prefill") or {}).get("new_token")
    if nt and not (56000 <= nt <= 72000):
        reps = max(800, min(2400, round(1373 * 65536.0 / nt)))
        log("filler off target (%s tok) -> regen reps=%s" % (nt, reps))
        subprocess.run([PY, os.path.join(DIR, "gen_filler.py"), str(reps)], check=True)
        return run_measure("full", "base", "base.log")
    return run1


def best_of(r1, r2):
    def dec(r):
        d = r.get("decode") or {}
        return d.get("steady_tok_s") or 0
    if not r1.get("ready"):
        return r2
    if not r2.get("ready"):
        return r1
    return r2 if dec(r2) >= dec(r1) * 0.995 else r1


def main():
    log("waiting for run1 (BASE full)")
    r1 = filler_fix(wait_run1())
    log("run1: ready=%s ready_s=%s" % (r1.get("ready"), r1.get("ready_s")))

    # Step 2: BASE + max-running-requests 1
    r2 = run_measure("full", "base_mr1", "base_mr1.log",
                     extra="--max-running-requests 1")

    # Step 3: boot-only plan inspections
    b_a = run_measure("boot", "b_mr1_r087", "b_mr1_r087.log",
                      extra="--max-running-requests 1 --memory-ratio 0.87")
    b_b = run_measure("boot", "b_mr1_r085", "b_mr1_r085.log",
                      extra="--max-running-requests 1 --memory-ratio 0.85")
    b_c = run_measure("boot", "b_mr2", "b_mr2.log",
                      extra="--max-running-requests 2")

    # Step 4: chunk A/B on the better full config
    best = best_of(r1, r2)
    base_extra = "--max-running-requests 1" if best is r2 else ""
    log("chunk A/B base config extra=%r (decode r1=%s r2=%s)" % (
        base_extra,
        (r1.get("decode") or {}).get("steady_tok_s"),
        (r2.get("decode") or {}).get("steady_tok_s")))
    c6144 = run_measure("prefill", "p_6144", "p_6144.log",
                        extra=(base_extra + " --max-prefill-length 6144").strip())
    c8191 = run_measure("prefill", "p_8191", "p_8191.log",
                        extra=(base_extra + " --max-prefill-length 8191").strip())
    c8191_low = {}
    if c8191.get("prefill", {}).get("error") or c8191.get("prefill", {}).get("watch_in_req") \
            or not c8191.get("ready"):
        log("8191 failed at ratio 0.89 -> retry with 0.85")
        c8191_low = run_measure("prefill", "p_8191_r085", "p_8191_r085.log",
                                extra="--max-running-requests 1 --max-prefill-length 8191 --memory-ratio 0.85")

    def prefill_ok(x):
        p = x.get("prefill") or {}
        return bool(x.get("ready")) and not p.get("error") and not p.get("watch_in_req")

    winner_chunk, winner_cfg, winner_extra = None, None, None
    if prefill_ok(c8191):
        winner_chunk, winner_cfg, winner_extra = c8191, "p_8191", base_extra
    elif prefill_ok(c8191_low):
        winner_chunk, winner_cfg, winner_extra = c8191_low, "p_8191_r085", "--max-running-requests 1 --max-prefill-length 8191 --memory-ratio 0.85"
    elif prefill_ok(c6144):
        winner_chunk, winner_cfg, winner_extra = c6144, "p_6144", (base_extra + " --max-prefill-length 6144").strip()
    else:
        winner_chunk, winner_cfg, winner_extra = {}, "p_4096", base_extra

    # decode spotcheck on winner chunk (full run = prefill + decode on same boot)
    wname = "win_" + winner_cfg
    if winner_cfg == "p_8191_r085":
        w_full = run_measure("full", wname, wname + ".log", extra=winner_extra)
    elif winner_cfg == "p_8191":
        w_full = run_measure("full", wname, wname + ".log", extra=winner_extra)
    elif winner_cfg == "p_6144":
        w_full = run_measure("full", wname, wname + ".log", extra=winner_extra)
    else:
        w_full = best

    # Step 5: load-speed A/B (boot-only, BASE + mr1)
    l_a = run_measure("boot", "load_base", "load_base.log",
                      extra="--max-running-requests 1")
    l_b = run_measure("boot", "load_alloc", "load_alloc.log",
                      extra="--max-running-requests 1",
                      env={"FREETOKEN_BANK_CUDA_ALLOC": "1"})

    # Step 6: deep-fill ladder on winner config
    r6 = run_measure("ladder", "ladder", "ladder.log", extra=winner_extra,
                     extra_args=[["--max-steps", "6"], ["--target-tokens", "393216"]])

    summary = {"run1_base": r1, "run2_base_mr1": r2, "b_mr1_r087": b_a, "b_mr1_r085": b_b,
               "b_mr2": b_c, "p_6144": c6144, "p_8191": c8191, "p_8191_r085": c8191_low,
               "winner_cfg": winner_cfg, "winner_extra": winner_extra, "win_full": w_full,
               "load_base": l_a, "load_alloc": l_b, "ladder": r6}
    with open(os.path.join(DIR, "CAMPAIGN.json"), "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    log("campaign done")


if __name__ == "__main__":
    main()
