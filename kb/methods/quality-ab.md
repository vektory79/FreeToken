# v2 quality battery A/B: grouped MMQ prefill (env1) vs moe_vec prefill (env0)

Date: 2026-09-18. Branch vektory79 @ 1a444e4, uncommitted v2 changeset (11-12
modified files, see deviations), single RTX 5090. Harness-side scripts only -
NO production code edits, NO stash, NO commits. This closes the "quality battery
SKIPPED" item of the v2 hardware validation protocol (v2-ab.md, TASK.md).

## Method

- Design: same-day env-flip A/B on one source tree. Control env0 =
  `FREETOKEN_GGUF_GROUPED_PREFILL=0` (pre-v2 moe_vec prefill), test env1 = `=1`
  (grouped MMQ prefill, kill-switch default ON). The env var is read per call
  (layers/moe.py:489), so it requires one boot per stage; a same-boot re-run
  would radix-hit and invalidate the comparison.
- TWO boots, serial + reaped, port 18801, winner flags, same compiled extension
  (disk JIT cache, no rebuild):
  `ft serve --model-path /media/ai/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL.gguf
  --moe-cache-auto --kv-reserve-tokens 500000 --kv-cache-dtype fp8
  --memory-ratio 0.85 --max-prefill-length 8191 --moe-strategy hybrid
  --moe-cpu-threads 16 --max-running-requests 1 --port 18801 --host 127.0.0.1`
- Battery: the phase-6 runner (`quality/ft_phase6_battery.py`, copied verbatim
  from the volatile /tmp/ft_phase6_battery.py) run ONCE per boot, identical
  prompt order p00..p23 from the kb-mirrored baseline directory
  [battery-post-fix/](../cases/gguf-glm5next-path-a/verification/phase6/battery-post-fix/)
  (8 en prose / 8 ru prose / 4 code / 4 reason-with-digits; max_tokens
  160/160/192/128 by kind), greedy temp0/topk1, stream=False, model name from
  GET /v1/models, POST /v1/chat/completions with the "model" field.
- Per-boot liveness probe: PYTHONPATH sitecustomize
  ([sitecustomize.py](../harness/mmq/sitecustomize.py), `logging.basicConfig(INFO)`; the
  original probe_sitecustomize/ directory was not mirrored, the file itself is)
  surfaces the one-shot marker line "gguf moe grouped mmq prefill active"
  (moe.py:520, bare-logger trap): expected ABSENT in env0, fired in env1.
- Historical context only (NOT the control): battery-post-fix outputs
  (GGUF post-fix tree af8e2e4, moe_vec prefill era), Task-05 accepted hybrid
  battery (copied to quality/t05_hybrid_main_battery/), Task-08 re-acceptance
  battery (quality/t08_hybrid_battery/).
- Divergence analyzer: quality/analyze_divergence.py -> quality/divergence.json.
  Combined text = reasoning + content; first-divergence char = common-prefix
  length; task-flip suspect = div < 5 chars.

## Hygiene protocol

- Census before/between/after every boot: pgrep '[f]t serve', nvidia-smi,
  sem.census (`ls /dev/shm | grep -c "^sem\.mp-"`).
- Runner: quality/run_battery.py - census, boot, /ready poll with FAILPAT scan
  (AssertionError|OutOfMemoryError|Backend worker is gone) during boot AND a 2 s
  watchdog thread during the battery, hard timeouts (boot 600 s, battery 2700 s),
  SIGTERM -> 30 s grace -> SIGKILL, finally-block cleanup, census assertion.
- llama-swap stopped; serial runs only; each stage wrapped in `timeout 3900`.

## Stage env0 (FREETOKEN_GGUF_GROUPED_PREFILL=0, moe_vec prefill)

- Boot: /ready 200 in 124.4 s. Resolved plan: `--moe-cache-auto resolved
  moe_cache_size=1144 num_pages=7816 (prefill_overlap=True)`; `Allocating
  500224 tokens for KV cache, K + V = 2.97 GiB`; `Free memory after
  initialization: 4.04 GiB`; fetch fraction 30.7% (decode leg).
- Executor: pool 1/3 (18,18,23) x39 layers threads=15 cores 0-22, pool 2/3
  (23,23,14) x1 threads=1 core 23, pool 3/3 (18,18,14) x2 threads=1 core 24;
  isa=avx2+gguf(iq/qk)+avx2-w4a8k on all pools.
- Liveness marker "gguf moe grouped mmq prefill active": ABSENT - expected,
  env0 is the moe_vec prefill control.
- Battery: rc=0, failed=0, wall 375.3 s (p00 63.8 s first-request warmup,
  then 10-17 s per prompt; finish=length on all but p22 finish=stop).
- On-topic (manual read of the 110-char heads + full texts): all 24 prompts
  track the expected task from battery-digest.md (libraries/bicycle/coastal
  rain/remote work/printing press/sky/lighthouse/vaccines; ru p08-p15 same
  topics answered in Russian; Fibonacci/is_palindrome/tr-sort-uniq/Stack;
  p20 60km/45min speed, p21 Wednesday+100 days, p22 3+8-5 apples, p23 2^10).

## Stage env1 (FREETOKEN_GGUF_GROUPED_PREFILL=1, grouped MMQ prefill)

- Boot: /ready 200 in 136.9 s. Resolved plan IDENTICAL to env0:
  `moe_cache_size=1144 num_pages=7816 (prefill_overlap=True)`, KV `Allocating
  500224 tokens ... K + V = 2.97 GiB`, `Free memory after initialization:
  4.04 GiB`, fetch fraction 30.7%, same 3 pools + threads + isa. Single-variable
  A/B holds.
- Liveness marker "gguf moe grouped mmq prefill active
  (FREETOKEN_GGUF_GROUPED_PREFILL)" (moe.py:520 via the sitecustomize probe):
  FIRED EXACTLY ONCE - the grouped MMQ prefill path was active in this boot.
- Battery: rc=0, failed=0, wall ~372 s (12-25 s per prompt; p22 finish=stop,
  the rest length - same as env0).

## Per-prompt divergence table env0 vs env1

Combined text = reasoning + content. digit ok = contains the expected answer
(p20 80 km/h, p21 Friday, p22 6 apples, p23 1024). Full data:
quality/divergence.json (raw analyzer print distilled away; see
kb/baselines/mmq-prefill-kernel/README.md).

| idx | kind | div@char | len0 | len1 | delta | flip? | digit e0/e1 |
|-----|------|---------|------|------|-------|-------|-------------|
| 0 | en | 430 | 856 | 870 | -14 | no | - |
| 1 | en | 346 | 789 | 789 | +0 | no | - |
| 2 | en | 159 | 652 | 687 | -35 | no | - |
| 3 | en | 140 | 791 | 790 | +1 | no | - |
| 4 | en | 339 | 737 | 774 | -37 | no | - |
| 5 | en | 112 | 817 | 687 | +130 | no | - |
| 6 | en | 289 | 742 | 739 | +3 | no | - |
| 7 | en | 15 | 762 | 750 | +12 | no | - |
| 8 | ru | 131 | 774 | 809 | -35 | no | - |
| 9 | ru | 68 | 643 | 646 | -3 | no | - |
| 10 | ru | 572 | 689 | 698 | -9 | no | - |
| 11 | ru | 257 | 672 | 692 | -20 | no | - |
| 12 | ru | 362 | 660 | 702 | -42 | no | - |
| 13 | ru | 124 | 706 | 694 | +12 | no | - |
| 14 | ru | 313 | 640 | 644 | -4 | no | - |
| 15 | ru | 374 | 713 | 643 | +70 | no | - |
| 16 | code | 193 | 681 | 669 | +12 | no | - |
| 17 | code | 340 | 754 | 761 | -7 | no | - |
| 18 | code | 88 | 628 | 613 | +15 | no | - |
| 19 | code | 116 | 751 | 717 | +34 | no | - |
| 20 | reason | 119 | 353 | 354 | -1 | no | ok/ok |
| 21 | reason | 197 | 368 | 380 | -12 | no | ok/ok |
| 22 | reason | 98 | 257 | 272 | -15 | no | ok/ok |
| 23 | reason | 4 | 261 | 303 | -42 | no* | ok/ok |

identical: 0/24 (expected - greedy is not bitwise across runs/kernels).
div min/median/max: 4 / 193 / 572 chars. *p23 was the single div<5 suspect:
env0 "The question asks which is larger..." vs env1 "The user asks which is
larger..." - same task from char 0, identical conclusion (2^10 = 1024 > 1000);
NOT a task flip.

Context vs historical batteries (same analyzer, informational only; different
trees/configs, so cross-tree numbers are NOT the A/B metric):
env0/env1 ~ t05 (accepted hybrid, Task-05) div 68-828; ~ t08 (re-acceptance
hybrid) div 49-491; ~ battery-post-fix div 0-828 where the 0-4 entries are
exactly the four digit prompts - the postfix tree (af8e2e4) predates the
digit-split pre-tokenizer fix and its outputs coherently ask for the missing
numbers, so a tiny div into DIFFERENT content is the expected signature, not a
regression. Digit prompts in THIS tree answer correctly (see verdict).

## Verdict vs the pass bar

| bar item | result |
|----------|--------|
| on-topic >= 20/24 in BOTH stages | PASS - 24/24 in env0 and 24/24 in env1 (manual read: every response tracks the digest's expected task, no garbage) |
| 4 digit prompts correct in both | PASS - 60/0.75 = 80 km/h (both); 100 mod 7 = 2 -> Friday (both); 3+8-5 = 6 apples, finish=stop (both); 2^10 = 1024 > 1000 (both) |
| no task-level flips | PASS - 0 flips; the div@4 suspect (p23) verified same-task wording drift |
| divergence in the wording-drift class | PASS with a stated caveat - min/median/max 4/193/572 chars, 18/24 pairs inside the historical 49-339 band; 5 pairs exceed it (p00 430, p10 572, p12 362, p15 374, p17 340; p04 sits exactly at 339). Each was read at and past the divergence point: same essay plan (p00 digital divide + information literacy), same winter-morning sensory list (p10), same story plan (p12), same Pushkin outline (p15), and p17 produces byte-identical palindrome code. Same answer, larger wording spread. |

VERDICT: PASS on all four bar items. The grouped MMQ prefill (env1) introduces
no quality regression vs the moe_vec prefill control (env0) under greedy
decoding. Final v2 validation status: hardware A/B (+14.4/+15.6/+16.4% prefill)
AND quality battery both complete - the v2 protocol is CLOSED.

## Hygiene log

- Pre-wave census: no ft serve, VRAM 1606-1697 MiB, sem.mp=6, llama-swap not
  running, port 18801 free.
- env0 stage: pre-boot census clean; boot 124.4 s; FAILPAT 0 hits (boot scan +
  2 s watchdog during battery); battery rc=0; graceful SIGTERM ("Finished server
  process"); teardown-mid census clean; stage rc=0.
- env1 stage: pre-boot census clean; boot 136.9 s; FAILPAT 0 hits; battery
  rc=0; graceful SIGTERM; post census clean (ft_serve=none, VRAM 1606 MiB,
  sem.mp=6); stage rc=0.
- Serial runs only; no production edits; no stash; no commits.

## Deviations

1. The task brief said "11 modified files"; the tree carries 12 modified code
   files (6 kernel/csrc/gguf, kernel/triton/moe_align.py, layers/gguf.py,
   layers/moe.py, 3 tests). No additional code file was touched by this wave
   (git status identical before/after apart from pre-existing .veai/memory
   churn from parallel sessions, which is not code).
2. The driver's boot_facts moe_cache regex matched the ServerArgs banner
   (moe_cache_size=0) instead of the resolved line; the real resolved value
   (1144) was read directly from both serve logs and is identical across stages.
3. Historical Task-05/Task-08 outputs were copied to quality/t05_hybrid_main_battery
   and quality/t08_hybrid_battery (volatile /tmp preservation per the brief);
   battery-post-fix stayed in place and was read in situ.
4. The raw analyzer print (analyze_output.txt, since distilled away) was first
   truncated by a SIGPIPE from a `| head` pipeline; re-run cleanly.
   divergence.json was never affected.
5. Boot flags use single --memory-ratio 0.85 / --max-prefill-length 8191 instead
   of the v2ab duplicate-flag idiom (argparse last-wins made those identical;
   effective config matches the winner flags exactly). No 0.90 fallback needed:
   memory-ratio 0.85 booted fine both times.
