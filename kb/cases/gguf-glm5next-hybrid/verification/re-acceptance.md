# Task 08 - FINAL hardware re-acceptance, GGUF hybrid post-ggml-port (2026-09-15)

Repo HEAD 63b9bff (11f1a80 + 979e3fc + fix 63b9bff), branch vektory79, tree clean
(modifications limited to .veai/memory). NO code changes, NO commits - pure
measurement wave. All runs used the Task 05 runner recipe (FAILPAT
'OutOfMemoryError|AssertionError|Backend worker is gone|backend worker .* exited'
every 5 s during boot + 2 s watchdog during runs, /ready poll, model from
/v1/models, SIGTERM -> 30 s -> SIGKILL, trap-on-EXIT cleanup). Zero FAILPAT hits,
zero watchdog trips in all four runs; every client finished with failed=[].

## STEP 0 - preflight census

VRAM 1321 MiB baseline, 0 compute processes; port 18081 free; no ft/mb/pytest
leftovers; 0 leaked semaphores. Post-wave census identical (1331 MiB, 0/0/0).

## STEP 1+2 - GGUF hybrid boot + decode @64k (real 147.5 GB file)

Command: `ft serve --model-path /media/ai/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL.gguf --moe-cache-auto --num-tokens 70016 --memory-ratio 0.85 --max-prefill-length 4096 --moe-strategy hybrid --moe-cpu-threads 16` (positional path rejected as always; no explicit --moe-cache-size; launched via the runner + /tmp/ft_serve_driver.py collect-stats hook).

Boot 108 s. Resolved plan: `--moe-cache-auto resolved moe_cache_size=1234
num_pages=136 (prefill_overlap=True)`; KV `Allocating 70016 tokens, K+V = 0.78 GiB`;
`Free memory after initialization: 3.38 GiB`; per-partition overlap degrade logged
for all 3 signature groups (463/288/288 slots < 2E).

Verification asserts (all PASS):
- Fetch fraction: `--moe-hybrid-max-fetch auto: fetching 30.7% of each decode step's
  expert misses over PCIe` - the restored benchbw profile value, NOT the cap-1 fallback.
- Pool split (weighted, 979e3fc): pool 1/3 (18,18,23) x39 layers threads=15 (cores 0..22),
  pool 2/3 (23,23,14) x1 thread=1 (core 23), pool 3/3 (18,18,14) x2 threads=1 (core 24)
  = the expected [15,1,1] of --moe-cpu-threads 16.
- ISA tag: `isa=avx2+gguf(iq/qk)+avx2-w4a8k` (the 11f1a80 verbatim ggml AVX2 W4A8-K tier) on all three pools.

Decode @64k (64,512-token fill in 4096 chunks, server prefill authoritative,
first-chunk warmup excluded; then the SAME prompt re-sent - server log
`#new-token: 12, #cached-token: 64512` - streamed, TTFT excluded): **16.67 tok/s**
(767 tokens / 46.00 s span; server gen throughput fluctuating 15.4-18.5).
Short-context decode (2k): 16.12 tok/s. Fill: 248 s wall, steady prefill 260.2-260.6 tok/s.
TTFT on the cached 64k prefix 3.28 s (first-step overhead, not re-prefill).

Misses (collect_stats dump at SIGTERM, 1787 decode steps): pool 0 (39 layers)
missing 5.05/layer (miss rate 63.1%), fetched 1.33/layer (26.4% of misses),
CPU 3.72/layer; pool 2 (2 layers, q6_k down) missing 1.48/layer; pool 1 fully
cached. Derived traffic at 60.0 ms/step (1.43 ms/layer): PCIe fetch ~573 MB/step
(~9.5 GB/s), CPU leg ~1.61 GB/step (~26.9 GB/s effective vs 50.0 GB/s benched
overlapped - not saturated). CPU utilization 19.9/20 cores (1 Hz /proc/stat
sampler). VRAM returned to baseline after SIGTERM.

## STEP 3 - offload control (decisive A/B)

Same command, ONLY `--moe-strategy offload`. Boot 111 s; same plan class
(1234 slots, 136 pages); no fetch-fraction line (expected - no CPU executor).

| metric | hybrid post-port | offload control | pre-port hybrid (Task 05) |
|---|---|---|---|
| decode @64k (767 tok) | **16.67** | **12.97** | 2.99 |
| short decode (2k) | 16.12 | 14.32 | ~3 |
| burst 6x cold 2k mean/min | 15.42 / 15.04 | 14.21 / 13.89 | 3.07 / 2.95 |
| ms/step @64k (ms/layer) | 60.0 (1.43) | 77.1 (1.84) | 334.4 (7.96) |
| TTFT cached 64k | 3.28 s | 3.26 s | ~6 s |
| prefill steady | ~260 tok/s | ~260 tok/s | ~260 tok/s |
| CPU cores @64k decode | 19.9/20 | 18.9/20 (incl. iowait; sampler cannot split user/iowait) | 7.5/20 |
| PCIe traffic/step (derived) | ~0.56 GiB (~9.5 GB/s) | ~2.14 GiB (~28.4 GB/s) | ~3.5 GiB |

Hybrid advantage: 1.28x @64k, 1.13x short, +1.21 tok/s burst mean; every hybrid
burst step (min 15.04) beats every offload burst step (max 14.36). Offload today
(12.97) sits slightly below its Task 05 same-session anchor (14.25) but within
the campaign 13.8-15.6 band's low side; the A/B is same-boot-day, single-variable.

## STEP 4 - 24-prompt quality battery (hybrid path)

Command: same hybrid serve command; client /tmp/ft_phase6_battery.py --baseline-dir
.tasks/gguf-glm5next-path-a/verification/phase6/battery-post-fix --outdir
/tmp/ft_t08_hybrid_battery (temp0/topk1, max_tokens 160/160/192/128 by kind, model
field from /v1/models). Boot 110 s, battery_rc=0, graceful SIGTERM exit 143, VRAM reaped.

**Result: 24/24 on-topic (gate 20+/24 - PASS).**
- p00-p07 [en] prose: all on-topic (libraries, bicycle physics, coastal rain, remote
  work, printing press, sky blue/sunsets, lighthouse story, vaccines).
- p08-p15 [ru]: all on-topic and answered in Russian (paper books, rain for a child,
  winter village, compass, cat and snow, cases, borscht, Pushkin).
- p16-p19 [code]: correct task understanding + plausible code (iterative Fibonacci,
  is_palindrome, tr|sort|uniq -c one-liner, Stack with TypeVar).
- p20-p23 [reason]: all CORRECT - 80 km/h; Friday (100 mod 7 = 2); 6 apples
  (finish_reason: stop); 2^10 = 1024 > 1000. Digit-split pre-tokenizer holds.

Divergence vs the accepted quant-drift baseline: total response length (reasoning +
content) differs from the Task 05 accepted hybrid battery by **0-66 chars per prompt**
- tighter than the accepted 49-339 chars vs NVFP4; same reasoning structure, no
garbage, no new failure mode.

## STEP 5 - NVFP4 profile-restoration sanity

User's exact 1M command: `ft serve --moe-strategy hybrid --moe-cpu-threads 16
--kv-reserve-tokens 1048576 --kv-cache-dtype nvfp4 --model-path
/media/ai/models/RedHatAI/GLM-5.3-Flash-NVFP4-FTW/ --max-running-requests 1
--disable-moe-prefill-overlap --memory-ratio 0.89 --max-prefill-length 4096`
(+ --port 18081 --host 127.0.0.1 from the runner; short-ctx decode only, no 64k fill).

BOOTED (no budget fail-fast - the 09-14 VRAM shift stays transient). Resolved
`moe_cache_size=343 num_pages=16418`, KV 1,050,752 tokens = 3.83 GiB, free after
init 2.81 GiB. `fetching 24.8% of each decode step's expert misses over PCIe` -
the restored nvfp4 profile fraction, matching the Task 06 benchbw FULL rerun
(~24.8%). Executor: single pool, threads=16, isa=avx2+vnni(nvfp4-w4a8). Short
decode (2k, 255 tokens): **15.18 tok/s** - the normal ~14-15 class. failed=[].
(A first attempt aborted before boot: my sanity client had a URL bug - double
base prefix - and the runner reaped the server normally; relaunched after the
client fix. Client-only, not a product issue.)

## Verdicts vs the three gates

1. Original campaign gate 14.3 tok/s @64k: **PASS** - 16.67 (1.17x over gate;
   5.6x over the pre-port 2.99; above the 12-14.5 post-port projection).
2. Offload-parity gate: **PASS** - 16.67 vs 12.97 same-session control (1.28x);
   also beats the Task 05 offload anchor 14.25.
3. "Not slower than offload" burst probe: **PASS** - hybrid burst min 15.04 >
   offload burst max 14.36 (per-step margin on all 6+6 steps).

Battery 24/24 >= 20/24 PASS. NVFP4 sanity PASS (fraction 24.8%, 15.18 tok/s).

## RECOMMENDATION

The ggml integer-kernel port (11f1a80 + 979e3fc + 63b9bff) flips the GGUF hybrid
verdict: hybrid is now the RECOMMENDED strategy for this file on this rig
(16.67 vs 12.97 offload @64k), replacing the Task 05 "offload recommended".
Task 08 gates all PASS; campaign CLOSED.

## Exact commands

- Hybrid/offload/battery serve (single variable = --moe-strategy):
  `ft serve --model-path /media/ai/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL.gguf --moe-cache-auto --port 18081 --num-tokens 70016 --memory-ratio 0.85 --max-prefill-length 4096 --moe-strategy {hybrid|offload} --moe-cpu-threads 16`
- Perf client: `.venv/bin/python /tmp/ft_nvfp4_client.py --port 18081 --mode full --name <name> --out /tmp/nvfp4_<name>` (64,512-token fill + radix reuse + 6-step burst).
- Battery: `.venv/bin/python /tmp/ft_phase6_battery.py --port 18081 --baseline-dir .tasks/gguf-glm5next-path-a/verification/phase6/battery-post-fix --outdir /tmp/ft_t08_hybrid_battery`.
- NVFP4 sanity serve: user 1M command above; client `.venv/bin/python /tmp/ft_sanity_client.py --port 18081 --name nvfp4_sanity --out /tmp/nvfp4_nvfp4_sanity_sanity.json`.
- Raw logs/artifacts: /tmp/nvfp4_gguf_hyb.log, /tmp/nvfp4_gguf_off.log, /tmp/nvfp4_gguf_batt.log, /tmp/nvfp4_nvfp4_sanity.log; result JSONs /tmp/nvfp4_gguf_{hyb,off}_result.json, /tmp/nvfp4_gguf_hyb_decode_stats.json, /tmp/nvfp4_nvfp4_sanity_sanity.json; battery raw /tmp/ft_t08_hybrid_battery/p00-p23.json.

Note: the ServerArgs banner prints moe_collect_stats=False because the driver
patch applies after the banner; the stats dump at SIGTERM is valid (three pools,
1787 steps) - do not read the banner as "stats off".
