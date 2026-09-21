# Phase 6 C2: quality comparison (GGUF vs NVFP4), decoded text only

battery prompts compared: 24/24
identical: 0
first-divergence-position histogram (chars into combined reasoning+content text):
  1-32: 3
  33-128: 0
  129-512: 0
  513-100000: 0

| idx | kind | result | note |
|-----|------|--------|------|
| 0 | en | div@0 | len 621/847 |
| 1 | en | div@21 | len 388/776 |
| 2 | en | div@0 | len 659/727 |
| 3 | en | div@0 | len 405/784 |
| 4 | en | div@15 | len 409/744 |
| 5 | en | div@0 | len 722/763 |
| 6 | en | div@0 | len 512/781 |
| 7 | en | div@0 | len 463/763 |
| 8 | ru | div@0 | len 566/758 |
| 9 | ru | div@0 | len 487/656 |
| 10 | ru | div@0 | len 631/676 |
| 11 | ru | div@0 | len 822/696 |
| 12 | ru | div@0 | len 646/721 |
| 13 | ru | div@0 | len 585/706 |
| 14 | ru | div@0 | len 656/675 |
| 15 | ru | div@0 | len 749/707 |
| 16 | code | div@0 | len 804/686 |
| 17 | code | div@0 | len 833/800 |
| 18 | code | div@15 | len 746/621 |
| 19 | code | div@0 | len 679/725 |
| 20 | reason | div@0 | len 595/336 |
| 21 | reason | div@0 | len 512/415 |
| 22 | reason | div@0 | len 558/477 |
| 23 | reason | div@0 | len 472/321 |

## Divergence example p00 (en, div@0)
prompt: Write a short essay on why public libraries still matter in the internet age.
gguf  : ...```json\n{"thought": "The user wants me to solve a math problem based on the provided image. \n\n1.  **Analyze the Request:**\n    *   Input:...
nvfp4 : ...The user wants a short essay on why public libraries still matter in the internet age. This is a straightforward writing request. Let me thi...

## Divergence example p01 (en, div@21)
prompt: Explain how a bicycle stays upright to a curious ten-year-old.
gguf  : ...The user wants me to find the area of the region bounded by the curves $y = \sin(x)$, $y = \cos(x)$, $x = 0$, and $x = \pi/2$.\n\nLet's find the intersection point...
nvfp4 : ...The user wants me to explain how a bicycle stays upright to a curious ten-year-old. This is a fun challenge because:\n\n1. The actual physics of bicycle stability ...

## Divergence example p02 (en, div@0)
prompt: Describe a rainy evening in a small coastal town using vivid sensory detail.
gguf  : ...```json\n{\n  "thinking": "The user wants me to solve a math problem based on an image. However, I don't see any image provided in the prompt....
nvfp4 : ...The user wants a descriptive piece about a rainy evening in a small coastal town, using vivid sensory detail. This is a creative writing req...

# POST-FIX closure (2026-09-14, branch vektory79: fix commit 028f2d9, HEAD af8e2e4)

Full 24-prompt greedy battery re-run on the FIXED tree (artifacts battery-post-fix/),
same prompts + max_tokens + method as the pre-fix battery (battery_nvfp4/ reused as
the same-day baseline; decoded text only, ids not comparable). Same session also
carried the perf recheck. Boot markers: histogram {(18, 18, 14): 2, (18, 18, 23): 39,
(23, 23, 14): 1}; partitions 460/288/288, prefill overlap OFF on all three;
--moe-cache-auto resolved moe_cache_size=1231, KV 70,016 tokens, max_extend_tokens=4096;
NO 8192->8191 clamp line (max_extend is 4096 with --max-prefill-length 4096, the clamp
does not trigger); boot 91 s to /ready 200.

## Quality: on-topic rate

- GGUF post-fix: 20/24
- NVFP4 baseline: 24/24
- GGUF pre-fix: 3/24

identical texts: 0/24 (expected across quantizations; greedy is not bitwise).
first-divergence histogram (chars into combined reasoning+content):
  0-32: 4   (all four are the digit-corruption prompts below)
  33-128: 10
  129-512: 10

For the 20 digit-free prompts the GGUF reasoning now tracks the NVFP4 plan for
49-339 chars before quantization-level wording drift (pre-fix: divergence at char 0
in 19/24 into a different task - image math, GUI-agent JSON, bounding boxes).

| idx | kind | div@char | on-topic | note |
|-----|------|----------|----------|------|
| 0 | en | 262 | y | same essay plan |
| 1 | en | 94 | y | |
| 2 | en | 147 | y | |
| 3 | en | 72 | y | |
| 4 | en | 339 | y | emits 1440/1455 digits correctly |
| 5 | en | 145 | y | |
| 6 | en | 175 | y | |
| 7 | en | 65 | y | |
| 8 | ru | 173 | y | |
| 9 | ru | 68 | y | |
| 10 | ru | 137 | y | |
| 11 | ru | 279 | y | |
| 12 | ru | 134 | y | |
| 13 | ru | 66 | y | |
| 14 | ru | 104 | y | |
| 15 | ru | 168 | y | |
| 16 | code | 124 | y | |
| 17 | code | 75 | y | |
| 18 | code | 88 | y | |
| 19 | code | 49 | y | |
| 20 | reason | 4 | n | 60/45 -> [blank]; asks for the numbers |
| 21 | reason | 1 | n | 100 -> [blank] |
| 22 | reason | 2 | n | 3/2/4/5 -> [blank] |
| 23 | reason | 0 | n | 2/10/3 -> [image?] |

The 4 failures are ONE residual defect, not task confusion: digits in the USER
prompt are rendered to the model as special-token placeholders ([blank]/[image?]);
the model stays on-subject and coherently asks for the missing numbers. Digits the
MODEL generates are fine (p04 outputs 1440/1455) - input-side tokenizer only. Same
defect family as the known pre=glm4 split gap (the Phase-5 post-fix 5-prompt battery
already read "What is 2+2?" as "What is +2?"). NVFP4 answers all 4 correctly
(80 km/h; Friday; 6 apples; 2^10).

## Example pairs (post-fix)

p00 (en, div@262)
nvfp4 : 1. **Digital divide** - Not everyone has internet access, computers, or digital literacy skills. Libraries provide free access...
gguf  : 1. **Digital divide / access equity**: Not everyone has internet access, computers, or devices at home. Libraries provide free access to technology, Wi-Fi...

p18 (code, div@88)
nvfp4 : The classic idiom is: grep -oE '\w+' file.txt | sort | uniq -c | sort -rn | head
gguf  : The classic solution: cat file.txt | tr -s '[:space:][:punct:]' '\n' | tr '[:upper:]' '[:lower:]' | sort | u...

p21 (reason, div@1)
nvfp4 : Today is Wednesday... 100 mod 7 = 2 ... Answer: Friday.
gguf  : The user is asking: "If today is Wednesday, what day of the week will it be in [blank] days?" The number of days is missing... I should point out...

## Perf recheck (same session, tuned cmd, watchdog harness, no FAILPAT hits)

| metric | GGUF pre-fix (f801a08) | GGUF post-fix (028f2d9) | delta |
|---|---|---|---|
| boot to /ready 200 | 115 s | 91 s | warm-cache variance |
| 64k-class fill, server steady chunks (warmup + tail excluded) | 263.9-268.1, mean 265.4 (12 chunks @ 4096) | 257.2-269.6, mean 264.1 (12 chunks @ 4096) | -0.5% = parity |
| fill prompt | 64,212 tok | 57,203 tok (PARA construction, 297,388 chars; original pre-fix prompt file unrecoverable, its text was 11% denser) | per-chunk metric unaffected |
| prefill ~4k, warm | Phase-5 comparator was COLD (4213 tok / 65 s incl JIT) - not like-for-like | 3,544 tok / 14.32 s client wall = 247.5 tok/s | - |
| decode cached 256 (@57-64k fill, prefix hit logged) | 14.67 tok/s, TTFT 3.29 s (768-run: 14.28) | 13.82 tok/s, TTFT 4.40 s | -3..-6%, single sample, inside campaign spread 13.0-15.6 |
| decode fresh short 256 | 13.00 tok/s, TTFT 3.21 s | 14.69 tok/s, TTFT 3.25 s | +13%, single sample, noise |
| battery avg wall/prompt | 14.7 s | 14.9 s | parity |
| plan | 1227 slots, KV 70,016 (0.78 GiB) | 1231 slots, KV 70,016 | +4 slots (+0.3%) |
| VRAM during serve | 28,817 MiB | 28,802 MiB | - |

Fix-cost verdict: the .contiguous() normalization of non-contiguous KDA in_proj
activations costs nothing measurable - prefill steady-state -0.5%, decode deltas
mixed-sign on single samples (historical spread 13.0-15.6 tok/s dominates).

## Campaign closure verdict

The GGUF offload serving path is PRODUCTION-USABLE for private prose chat on this
box: 20/24 on-topic (vs 3/24 pre-fix, NVFP4 24/24), prefill/decode parity with the
pre-fix tree, clean boot and shutdown (SIGTERM -> exit 143, teardown 15 s, VRAM back
to 1362 MiB desktop baseline, no compute processes, no leaked-semaphore warnings).
Working today: en/ru prose chat, code prompts without numeric literals, long-context
reading (57k cached prefix, 13.8 tok/s decode). Remaining known defect (scoped,
input-side only): numeral tokens in the user message collapse to [blank]/[image?]
placeholders, so any prompt containing digits (arithmetic, dates, quantities) cannot
be answered correctly. Fixing the gpt2-converter digit split (pre=glm4) is the one
remaining item before full parity with NVFP4.

## POST-FIX artifacts

- battery-post-fix/p00-23.json + battery-post-fix/digest-vs-nvfp4.md
- ab_postfix.json (64k prompt_tokens recovered from the server log after a perf-client
  key collision - harness-only bug, documented in the file)
- perf-server-lines-postfix.txt, boot-markers-postfix.txt, gguf-server-postfix.log
- harness (ephemeral, /tmp): ft_phase6_runner.sh + ft_phase6_battery.py +
  ft_phase6_perf.py + ft_phase6_compare.py
- anomalies: perf-client JSON key collision (server log used as authoritative); the
  4k probe's server throughput line (108 tok/s) spans the preceding decode window, so
  the client wall is quoted instead; no OOM, no FAILPAT hits, clean teardown.
