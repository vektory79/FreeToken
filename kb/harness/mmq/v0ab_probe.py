#!/usr/bin/env python3
"""Liveness probe: boot ft serve (winner flags) with a root logging handler
injected via PYTHONPATH, send one small prefill, grep for the v0 marker."""
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, "/media/ai/src/FreeToken/.tasks/ft-gguf-serve-tuning")
import measure  # noqa: E402

DIR = "/media/ai/src/FreeToken/.tasks/ft-gguf-serve-tuning"
LOGPATH = os.path.join(DIR, "logs", "ab_v0_probe")
EXTRA = ["--max-running-requests", "1", "--max-prefill-length", "8191",
         "--memory-ratio", "0.85"]
PROBE_ENV = ["PYTHONPATH=/media/ai/src/FreeToken/.tasks/mmq-prefill-kernel/probe_sitecustomize"]

log = measure.log
census = measure.census_serve()
log("census before: %s" % census)
proc, lf, t0 = measure.start_server(EXTRA, LOGPATH, list(PROBE_ENV))
ok, el, why = measure.wait_ready(LOGPATH, proc)
log("ready=%s in %.1fs why=%s" % (ok, el, why))
marker = None
if ok:
    time.sleep(2)
    # one small prefill: first ~24k chars of the filler (~6k tokens, one chunk)
    content = open(os.path.join(DIR, "filler.txt"), encoding="utf-8").read()[:24000]
    rq = measure.build_req(os.path.join(DIR, "req_probe.json"), content, 1, False)
    off = os.path.getsize(LOGPATH)
    t0 = time.time()
    err = None
    usage = {}
    try:
        req = urllib.request.Request(
            measure.URL + "/v1/chat/completions", data=open(rq, "rb").read(),
            headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=300)
        body = json.loads(resp.read())
        resp.close()
        usage = body.get("usage") or {}
    except Exception as e:
        err = "%s: %s" % (type(e).__name__, e)
    log("probe prefill: %.1fs usage=%s err=%s" % (time.time() - t0, usage, err))
    txt = open(LOGPATH, encoding="utf-8", errors="replace").read()
    marker = [l for l in txt.splitlines() if "expert-sorted pair order active" in l]
    thr = [l.strip()[-120:] for l in txt.splitlines() if "input throughput" in l]
    log("marker lines: %r" % marker)
    log("throughput lines: %r" % thr)
td = measure.teardown(proc)
log("teardown leftover=%s gpu=%s" % (td["leftover"], td["gpu_mib"]))
print(json.dumps({"ready": ok, "marker_found": bool(marker),
                  "marker_lines": marker or [], "teardown": td}))
