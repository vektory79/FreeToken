#!/usr/bin/env python3
"""One route-affinity arm of the FULL GATE (kb/cases/route-affinity/TASK.md).

v2 (gate): boots ft serve with the production config in GRAPH mode with
--route-affinity <v>, fills a 64k prefix, radix-reuse greedy-decodes
768 tokens (ignore_eos) for speed metrics, then runs the quality battery on
the same boot:

  1. legacy battery: 6 arithmetic (auto-check) + 4 prose (first campaign set);
  2. campaign 24-prompt battery from kb/cases/route-affinity/battery24.json
     (4 of them with verifiable answers, word-boundary auto-check);
  3. NIAH probe: 20 key-value needles injected into the filler at depths
     40k-60k, run deepest-first so the radix cache shares the prefix;
     auto-check of the numeric value;
  4. 5-turn multi-turn probe (chained arithmetic, coherence across requests).

All texts are saved verbatim into arm_<tag>.json for the blind pairwise judge.
Hit-rate lines are parsed only up to the byte offset captured right after the
768 decode, so later battery traffic cannot pollute them.

Usage: run_arm.py --affinity 0.2 --tag aff020_b1 [--smoke]
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.request

REPO = "/media/ai/src/FreeToken"
DIR = os.path.join(REPO, ".tasks", "route-affinity")
CASE = os.path.join(REPO, "kb", "cases", "route-affinity")
FT = os.path.join(REPO, ".venv", "bin", "ft")
FILLER = os.path.join(REPO, ".tasks", "ft-gguf-serve-tuning", "filler.txt")
MODEL = "GLM-5.3-Flash-UD-Q3_K_XL.gguf"
PORT = 18801
URL = "http://127.0.0.1:%d" % PORT
READY_TIMEOUT = 420
FILL_TIMEOUT = 1200
DECODE_TIMEOUT = 1200
PROMPT_TIMEOUT = 300
NIAH_TIMEOUT = 900
FILL_TOKENS = 4
DECODE_TOKENS = 768
NIAH_COUNT = 20
WATCH = re.compile(r"AssertionError|OutOfMemoryError|CUDA error|Traceback \(most recent call last\)")

BASE_ARGS = [
    "--enable-cache-report",
    "--model-path", "/media/ai/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL-FTW-vision-new/",
    "--moe-cache-auto",
    "--kv-reserve-tokens", "800000",
    "--kv-cache-dtype", "fp8",
    "--memory-ratio", "0.81",
    "--max-prefill-length", "4095",
    "--moe-strategy", "hybrid",
    "--moe-cpu-threads", "16",
    "--max-running-requests", "1",
    "--session-tier-ram-gib", "10",
    "--session-tier-dir", "/media/ai/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/cache",
    "--session-tier-ssd-gib", "50",
    "--moe-collect-stats",
]

# (name, prompt, max_tokens, expected-substring or None) - first campaign set
BATTERY = [
    ("arith_mult", "Вычисли: 147 * 23. Назови только число.", 96, "3381"),
    ("arith_pow", "Сколько будет 2 в степени 10? Назови только число.", 96, "1024"),
    ("arith_mod", "Чему равно 100 mod 7? Назови только число.", 96, "2"),
    ("arith_eq", "Реши уравнение 3x + 9 = 30. Чему равен x? Назови только число.", 96, "7"),
    ("arith_bin", "Переведи число 156 в двоичную систему. Назови только результат.", 96, "10011100"),
    ("arith_prime", "Является ли 91 простым числом? Ответь да или нет.", 96, "нет"),
    ("prose_cache", "Кратко объясни, что такое кэш процессора.", 200, None),
    ("prose_japan", "Напиши короткий абзац о природе Японии.", 200, None),
    ("prose_smooth", "Что такое экспоненциальное сглаживание? Два предложения.", 160, None),
    ("prose_window", "Составь одно предложение со словом окно в переносном смысле.", 120, None),
]

# verifiable answers inside battery24 (word-boundary regex, case-insensitive)
EXPECTED24 = {
    "b20_reason": r"\b80\b",
    "b21_reason": r"\bfriday\b|\bпятниц",
    "b22_reason": r"\b6\b",
    "b23_reason": r"\b1024\b",
}

# 5-turn chained multi-turn probe; each turn depends on the previous answers
MULTITURN = [
    ("turn1_pick", "Загадай число от 50 до 99 и назови только его.", 96),
    ("turn2_mul", "Умножь своё число на 3 и назови только результат.", 96),
    ("turn3_sub", "Вычти из результата 50 и назови только новое число.", 96),
    ("turn4_dbl", "Удвой последнее число и назови только результат.", 96),
    ("turn5_chain", "Восстанови в одном предложении всю цепочку вычислений из нашего разговора: исходное число и каждый шаг.", 200),
]

BOOT_PATTERNS = [
    ("resolved_plan", re.compile(r"--moe-cache-auto resolved moe_cache_size=(\d+)\s+num_pages=(\d+)")),
    ("fetch_auto", re.compile(r"--moe-hybrid-max-fetch auto.*")),
    ("affinity_banner", re.compile(r"route affinity.*", re.IGNORECASE)),
]
HIT_RE = re.compile(r"MoE decode,\s*pinned[^:]*: hit ([\d.]+)%")
STEP_RE = re.compile(r"miss ([\d.]+) fetch ([\d.]+) cpu ([\d.]+)")


def log(msg):
    print("[aff] %s" % msg, flush=True)


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def census_serve():
    out = sh("pgrep -af '[f]t serve'").stdout.strip()
    return [line for line in out.splitlines() if line]


def start_server(extra_args, logpath, env_pairs):
    for p in census_serve():
        log("FATAL: live ft serve census: %s" % p)
        sys.exit(1)
    cmd = [FT, "serve", "--port", str(PORT), "--host", "127.0.0.1"] + BASE_ARGS + list(extra_args)
    env = dict(os.environ)
    for kv in env_pairs or []:
        k, _, v = kv.partition("=")
        env[k] = v
    lf = open(logpath, "w")
    proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=REPO, env=env)
    with open(logpath + ".pid", "w") as f:
        f.write(str(proc.pid))
    return proc, lf


def wait_ready(logpath, proc, timeout=READY_TIMEOUT):
    t0 = time.time()
    probe = {"model": MODEL, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1}
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False, time.time() - t0, "process exited rc=%s" % proc.returncode
        try:
            tail = open(logpath, "rb").read()[-12000:].decode("utf-8", "replace")
            m = WATCH.search(tail)
            if m:
                return False, time.time() - t0, "watchdog: %s" % m.group(0)
        except Exception:
            pass
        try:
            req = urllib.request.Request(URL + "/v1/chat/completions",
                                         data=json.dumps(probe).encode(),
                                         headers={"Content-Type": "application/json"})
            r = urllib.request.urlopen(req, timeout=5)
            r.read()
            r.close()
            return True, time.time() - t0, "ok"
        except Exception:
            pass
        time.sleep(3)
    return False, time.time() - t0, "boot timeout"


def chat(messages, max_tokens, stream=False, timeout=600):
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    rq = {"model": MODEL, "messages": messages,
          "max_tokens": max_tokens, "temperature": 0, "top_k": 1, "stream": stream,
          "ignore_eos": stream}
    times = []
    usage = {}
    err = None
    if stream:
        rq["stream_options"] = {"include_usage": True}
    rbody = ""
    t0 = time.time()
    try:
        req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(rq).encode(),
                                     headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=timeout)
        if stream:
            body = ""
            for raw in resp:
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
                    delta = obj["choices"][0].get("delta", {}) or {}
                    body += delta.get("content") or ""
                    rbody += delta.get("reasoning_content") or ""
                    times.append(time.time())
            resp.close()
            combined = (body + "\n" + rbody).strip()
            return {"wall_s": round(time.time() - t0, 2), "error": err, "times": times,
                    "usage": usage, "content": combined, "content_raw": body,
                    "reasoning": rbody}
        body = json.loads(resp.read())
        resp.close()
        msg = (body.get("choices") or [{}])[0].get("message", {})
        # the glm reasoning parser keeps the work in reasoning_content and often
        # leaves message.content empty on short budgets - compare both fields
        raw = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        content_out = (raw + "\n" + reasoning).strip()
        return {"wall_s": round(time.time() - t0, 2), "error": None, "times": [],
                "usage": body.get("usage") or {}, "content": content_out,
                "content_raw": raw, "reasoning": reasoning,
                "finish_reason": (body.get("choices") or [{}])[0].get("finish_reason")}
    except Exception as e:
        return {"wall_s": round(time.time() - t0, 2), "error": "%s: %s" % (type(e).__name__, e),
                "times": [], "usage": {}, "content": "", "content_raw": "", "reasoning": "",
                "finish_reason": None}


def decode_metrics(times, ntok):
    out = {"completion_tokens": ntok}
    if len(times) >= 4:
        tail = times[1:]
        if tail[-1] > tail[0]:
            out["steady_tok_s"] = round((len(tail) - 1) / (tail[-1] - tail[0]), 2)
        gaps = sorted(tail[i + 1] - tail[i] for i in range(len(tail) - 1))
        if gaps:
            out["p50_step_ms"] = round(1000 * gaps[len(gaps) // 2], 1)
    return out


def boot_keys(logpath):
    keys = {}
    try:
        txt = open(logpath, encoding="utf-8", errors="replace").read()
    except Exception:
        return keys
    for name, pat in BOOT_PATTERNS:
        m = pat.search(txt)
        if not m:
            keys[name] = None
        elif name == "resolved_plan":
            keys[name] = {"moe_cache_size": int(m.group(1)), "num_pages": int(m.group(2))}
        else:
            keys[name] = m.group(0).strip()
    return keys


def parse_hits(logpath, end_offset=None):
    try:
        with open(logpath, "rb") as f:
            data = f.read(end_offset)
        txt = data.decode("utf-8", "replace")
    except Exception:
        return []
    return [float(m.group(1)) for m in HIT_RE.finditer(txt)]


def parse_steps(logpath, end_offset=None):
    try:
        with open(logpath, "rb") as f:
            data = f.read(end_offset)
        txt = data.decode("utf-8", "replace")
    except Exception:
        return []
    return [[float(x) for x in m.groups()] for m in STEP_RE.finditer(txt)]


def teardown(proc, grace=45):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
    time.sleep(2)
    return {"rc": proc.returncode, "left_ft_serve": census_serve()}


def niah_items(filler, fill_prompt_tokens, count):
    """20 needles at 40k-60k token depth, deepest first for radix prefix reuse.

    Keys and values are fixed across arms so the arms are comparable; every
    query has a unique key so radix cannot serve a previous query's answer.
    """
    total_tokens = fill_prompt_tokens or int(len(filler) / 5.6)
    cpt = len(filler) / float(total_tokens)
    depths = sorted((40000 + int(i * (20000.0 / (count - 1))) for i in range(count)), reverse=True)
    items = []
    for i, d in enumerate(depths):
        nonce = 7001 + i
        value = 1000 + i * 173 + (i * i) % 89
        off = min(max(int(d * cpt), 1), len(filler) - 1)
        needle = "\nService note: the access code for key ARC-%d equals %d.\n" % (nonce, value)
        q = "\n\nQuestion: what is the access code for key ARC-%d? Answer with the number only." % nonce
        items.append({"name": "niah_d%05d" % d, "depth_tokens": d, "key": "ARC-%d" % nonce,
                      "expected": str(value), "prompt": filler[:off] + needle + filler[off:] + q})
    return items


def dump(out, tag):
    print(json.dumps({k: out[k] for k in out if k not in ("battery24", "niah", "multiturn")},
                     indent=1, ensure_ascii=False))
    json.dump(out, open(os.path.join(DIR, "arm_%s.json" % tag), "w"), indent=1, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--affinity", type=float, required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny battery/decode to validate the pipeline")
    a = ap.parse_args()
    tag = a.tag or ("aff%03d" % round(a.affinity * 100))

    dec_tokens = 64 if a.smoke else DECODE_TOKENS
    niah_count = 2 if a.smoke else NIAH_COUNT
    b24 = json.load(open(os.path.join(CASE, "battery24.json")))
    if a.smoke:
        b24 = b24[:4]

    logpath = os.path.join(DIR, "serve_%s.log" % tag)
    content = open(FILLER, encoding="utf-8").read()
    out = {"v": 2, "tag": tag, "affinity": a.affinity, "smoke": bool(a.smoke),
           "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    env_pairs = None
    proc, lf = start_server(["--route-affinity", str(a.affinity)], logpath, env_pairs)
    ok, ready_s, why = wait_ready(logpath, proc)
    out["ready"] = ok
    out["ready_s"] = round(ready_s, 1)
    out["ready_why"] = why
    out["boot_keys"] = boot_keys(logpath)
    if not ok:
        out["log_tail"] = open(logpath, "rb").read()[-4000:].decode("utf-8", "replace")
        out["teardown"] = teardown(proc)
        dump(out, tag)
        sys.exit(2)

    fill = chat(content, FILL_TOKENS, False, FILL_TIMEOUT)
    out["fill"] = {"wall_s": fill["wall_s"], "error": fill["error"],
                   "prompt_tokens": fill["usage"].get("prompt_tokens")}
    if fill["error"]:
        out["teardown"] = teardown(proc)
        dump(out, tag)
        sys.exit(3)

    t0 = time.time()
    dec = chat(content, dec_tokens, True, DECODE_TIMEOUT)
    ntok = dec["usage"].get("completion_tokens") or len(dec["times"])
    out["decode"] = decode_metrics(dec["times"], ntok)
    out["decode"]["error"] = dec["error"]
    out["decode"]["wall_s"] = dec["wall_s"]
    out["decode"]["text"] = dec["content"]
    out["decode"]["reasoning"] = dec.get("reasoning", "")
    # hit/step lines only up to this point - the battery below must not pollute them
    hit_end = os.path.getsize(logpath)
    out["hit_lines"] = parse_hits(logpath, hit_end)
    hits = out["hit_lines"]
    out["hit_mean_pct"] = round(sum(hits) / len(hits), 2) if hits else None
    steps = parse_steps(logpath, hit_end)
    if steps:
        miss = sum(s[0] for s in steps) / len(steps)
        fetch = sum(s[1] for s in steps) / len(steps)
        out["miss_mean_pct"] = round(miss, 2)
        out["fetch_mean_pct"] = round(fetch, 2)
        out["c_estimate_pct"] = round(100 * (1 - (miss - fetch) / 100.0), 1)

    battery = []
    for name, prompt, max_tokens, expected in BATTERY:
        r = chat(prompt, max_tokens, False, PROMPT_TIMEOUT)
        content_out = r["content"].strip()
        okq = bool(expected) and (expected.lower() in content_out.lower())
        battery.append({"name": name, "ok": okq, "expected": expected,
                        "content": content_out, "error": r["error"]})
    out["battery"] = battery
    out["battery_pass"] = sum(1 for b in battery if b["ok"])
    out["battery_arith_pass"] = sum(1 for b in battery[:6] if b["ok"])

    b24_res = []
    for item in b24:
        r = chat(item["prompt"], item["max_tokens"], False, PROMPT_TIMEOUT)
        content_out = r["content"].strip()
        pat = EXPECTED24.get(item["name"])
        okq = bool(pat) and bool(re.search(pat, content_out, re.IGNORECASE))
        b24_res.append({"name": item["name"], "kind": item.get("kind"), "prompt": item["prompt"],
                        "max_tokens": item["max_tokens"], "ok": okq, "expected": pat,
                        "content": content_out, "error": r["error"],
                        "finish_reason": r.get("finish_reason")})
    out["battery24"] = b24_res
    out["battery24_pass"] = sum(1 for b in b24_res if b["ok"])
    out["battery24_checked"] = sum(1 for b in b24_res if b["expected"])

    niah_res = []
    for item in niah_items(content, out["fill"].get("prompt_tokens"), niah_count):
        r = chat(item["prompt"], 160, False, NIAH_TIMEOUT)
        content_out = r["content"].strip()
        okq = item["expected"] in content_out
        niah_res.append({"name": item["name"], "depth_tokens": item["depth_tokens"],
                         "key": item["key"], "expected": item["expected"], "ok": okq,
                         "content": content_out, "finish_reason": r.get("finish_reason"),
                         "prompt_tokens": r["usage"].get("prompt_tokens"),
                         "error": r["error"]})
    out["niah"] = niah_res
    out["niah_pass"] = sum(1 for n in niah_res if n["ok"])
    out["niah_total"] = len(niah_res)

    mt_res = []
    msgs = []
    for name, prompt, budget in (MULTITURN[:2] if a.smoke else MULTITURN):
        msgs.append({"role": "user", "content": prompt})
        r = chat(msgs, budget, False, PROMPT_TIMEOUT)
        reply_raw = (r.get("content_raw") or "").strip() or r["content"].strip()
        msgs.append({"role": "assistant", "content": reply_raw})
        mt_res.append({"name": name, "prompt": prompt, "content": r["content"].strip(),
                       "error": r["error"]})
    out["multiturn"] = mt_res

    out["teardown"] = teardown(proc)
    dump(out, tag)
    if not dec["error"] and ntok >= dec_tokens and out["battery_arith_pass"] >= 4:
        sys.exit(0)
    sys.exit(4)


if __name__ == "__main__":
    main()
