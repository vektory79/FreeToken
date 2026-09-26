#!/usr/bin/env python3
"""Serial driver for the decode research campaign: 4 arms, one server at a time."""
import json
import subprocess
import sys
import time

PY = "/media/ai/src/FreeToken/.venv/bin/python"
DIR = "/media/ai/src/FreeToken/.tasks/decode-research"

ARMS = [
    ("base", [], []),
    ("fetch3", ["--moe-hybrid-max-fetch", "3"], []),
    ("ov0", [], ["FREETOKEN_HYBRID_OVERLAP=0"]),
    ("flagsync0", [], ["FREETOKEN_CPU_MOE_FLAG_SYNC=0"]),
]


def main():
    only = sys.argv[1:] or None
    results = {}
    for name, extra, env in ARMS:
        if only and name not in only:
            continue
        print("=== ARM %s start %s ===" % (name, time.strftime("%H:%M:%S")), flush=True)
        cmd = [PY, DIR + "/run_arm.py", "--name", name, "--log", "arm_%s.log" % name,
               "--out", DIR + "/arm_%s.json" % name]
        if extra:
            cmd += ["--extra", " ".join(extra)]
        for kv in env:
            cmd += ["--env", kv]
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        dt = time.time() - t0
        ok = r.returncode == 0
        results[name] = {"rc": r.returncode, "wall_s": round(dt, 1)}
        print("=== ARM %s done rc=%s in %.1fs ===" % (name, r.returncode, dt), flush=True)
        if not ok:
            print(r.stdout[-3000:], flush=True)
            print(r.stderr[-2000:], flush=True)
        # cooldown between arms (process reaping, VRAM settle)
        time.sleep(5)
    json.dump(results, open(DIR + "/campaign_summary.json", "w"), indent=1)
    print("CAMPAIGN DONE: %s" % json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
