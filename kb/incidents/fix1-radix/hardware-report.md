# Hardware validation report: fix-1 radix HIT at prefill chunk 4096

Date: 2026-09-18 21:30-21:55 +03. Box: RTX 5090 (32 GiB, IOMMU=pt). Branch: vektory79
(working tree contains the fix under test; nothing modified/committed/pushed).
Model: GLM-5.3-Flash UD-Q3_K_XL GGUF (~147.5 GB), fp8 KV, moe-strategy hybrid,
kv-reserve 500000, moe-cache-auto, moe-cpu-threads 16, port 18801.
Harness: [measure.py](../../harness/serve-measure/measure.py) mode=full (both known patches present:
decode steady skips times[0]; throughput regex anchored on "input throughput (token/s):").
Runner: verification/run_one.sh (FAILPAT watchdog every 3 s incl. boot-wait, hard timeout
900 s with SIGTERM->SIGKILL after 30 s, trap-on-EXIT pkill '[f]t serve').

## 1. Boot config actually used

| Run | extra args | Result |
|---|---|---|
| A attempt 1 | `--max-prefill-length 4096 --memory-ratio 0.89` | FAIL-FAST at early slot-floor gate: "byte-weighted minimum of 1059 layer-0-width slots vs the planned 1040" (engine/cache_budget.py:77). Known desktop-VRAM-tax scenario (hoptodesk 302+126 MiB + Xorg 362 MiB live). |
| A (final) | `--max-prefill-length 4096 --memory-ratio 0.90` | Boot OK, ready_s=102.2. moe_cache_size=1077, num_pages=7837, free-after-init 2.59 GiB. Deviation: ratio 0.89 -> 0.90, mandated fallback, cause = slot-floor gate above. |
| B | `--max-prefill-length 8191 --memory-ratio 0.90` (same fallback ratio as A) | Boot OK, ready_s=104.3. moe_cache_size=1068, free-after-init 2.58 GiB. First prefill chunk OOM (see 4). |

## 2. Run A verdict (primary): HIT

Exact verdict line (server log fix4096.log, after the repeat request):

```
[2026-09-18|21:41:58|core|rank=0] INFO     Prefill batch, #new-seq: 1, #new-token: 113, #cached-token: 65472, token usage: 0.13, #mamba-slot: 4/24, mamba usage: 0.17, #running-req: 1, #queue-req: 0, input throughput (token/s): 23.35
```

#cached-token: 65472 -> HIT (expected window 65472-65536; pre-fix this was 0 = full MISS,
~251 s re-prefill per repeat). 65472 = 65536 - 64 (page-64-aligned snapshot). Repeat
#new-token 113 = 65585 - 65472. Decode JSON: cached_token=65472, new_token=113.

## 3. Run B verdict: inconclusive on this box (OOM, classified)

No "Prefill batch" line was ever emitted at 8191: the first 8128-token chunk died in
triton do_bench autotune before any batch ran, so there is no #cached-token line to quote.
The 8128-chunk HIT mechanism was already proven pre-fix (campaign PHASE2: 8128 -> HIT
65536) and the fix only ADDS the L forward (cannot remove a donation path), so the
"stays HIT" sub-check remains supported by campaign evidence, not re-proven here.

## 4. Perf check (same logs, no extra runs)

Prefill, 4096-token full chunks, server "input throughput (token/s):" lines: 16 full
chunks + 49-tok tail. Excluded: chunk 1 = 91.86 (JIT/autotune warmup) and the LAST full
chunk = 939.41 (known drain artifact). Remaining 14 chunks: 298.26..301.48, MEDIAN
300.58 tok/s (tail chunk 58.42, 49 tok). Expected ~262 class -> measured 300.6: ABOVE
class, no regression (favorable; campaign 262 was the 0.85 winner config).

Decode from the repeat (63 completion tokens, 65 SSE frames): steady_tok_s=14.7
(p50 14.82, min inst 10.11), wall 8.97 s per repeated 65k request. Expected 13.5-14.7
class -> PASS at top of class. (avg_tok_s=7.03 includes ~6 s first-decode-step TTFT.)

Run B OOM detail (fix8191.log:132): `torch.OutOfMemoryError: CUDA out of memory. Tried
to allocate 256.00 MiB. GPU 0 ... 187.00 MiB is free` in
triton/backends/nvidia/driver.py:761 get_empty_cache_for_benchmark. Peak VRAM 31962 MiB.
Exactly the documented "first prefill chunk at 8191 needs ~256 MiB triton autotune
headroom"; 0.89 cannot even boot here (floor gate) and 0.90 leaves less headroom than
the campaign's 0.85 (4.11 GiB free). Per brief: NOT chased.

## 5. Hygiene census

- Preflight: pgrep 'ft serve|pytest|benchbw|measure.py' -> nothing live; nvidia-smi
  1628 MiB = desktop GUI only (Xorg 362, kwin 232, hoptodesk 428, firefox 198, ...);
  semaphores `sem.mp-*` = 11.
- Post Run A: teardown leftover=[], VRAM 1530 MiB, no leftovers.
- Post Run B: watchdog killed the '[f]t serve' tree on FAILPAT; ONE survivor was found
  and reaped: PID 985633, `python -c from multiprocessing.spawn import spawn_main`
  (spawn children do not match the '[f]t serve' pattern), which held the 3 semaphores
  leaked by the SIGKILLed tree (11 -> 14). Killed TERM->KILL, removed exactly its 3
  /dev/shm/sem.mp-{7fsslpj_,2ufl3xaj,z32vvpju} files -> count back to 11.
- Final census: no wave processes (incl. spawn_main sweep of my tree), VRAM 1631 MiB,
  semaphores 11.
- Pre-existing orphans NOT touched (report-only, they predate this wave and hold the
  6 baseline semaphores): PID 776127 (spawn_main, started Sep 16 22:25, cwd /repo) and
  PID 2010750 (spawn_main, started Sep 17 11:34, cwd /repo) - spawn children of dead
  ft trees from earlier waves. Recommend an orchestrator sweep.

## 6. Failures classified

1. Run A attempt 1 @0.89: boot fail-fast, slot-floor gate (1059 needed vs 1040 planned)
   - Environment (desktop/remote-desktop VRAM tax); recovered per brief with 0.90.
2. Run B: CUDA OOM 256 MiB triton do_bench at the first 8191 chunk (187 MiB free)
   - Environment / VRAM-headroom constraint of 8191@0.90 on this box; not fix-related.
3. Runner bug (cosmetic): watchdog referenced unset MEASPID under `set -u`
   ("MEASPID: unbound variable"); the pkill of the ft tree still executed. Note for
   future reuse of verification/run_one.sh: export MEASPID before the watchdog block.

## 7. Verdict

PASS for the fix-1 hardware gate on the primary check: a byte-identical repeat of the
65k prompt now HITS the radix cache at prefill chunk 4096 (#cached-token 65472 vs
pre-fix 0), with decode repeat wall ~9 s instead of ~251 s re-prefill, and no
prefill/decode regression (prefill median 300.6 tok/s vs ~262 class; decode steady
14.7 tok/s in the 13.5-14.7 class). The secondary 8128-chunk re-check is inconclusive
on this box (VRAM headroom OOM at the only bootable ratio); mechanism-level risk of the
fix regressing it is nil (the change only forwards mamba_last_track_seqlen across chunk
transitions), and 8128 HIT was already hardware-proven pre-fix.

Artifacts: boot logs fix4096.log/fix8191.log not mirrored (distilled away);
runner [run_one.sh](../../harness/repro/run_one.sh).

## 8. Run B retry (0.85+mr1) - closes the 8128-chunk item: PASS (HIT)

2026-09-18 21:55-22:02 +03. Single bounded retry at the campaign winner config:
`--max-prefill-length 8191 --memory-ratio 0.85 --max-running-requests 1`
(deviation from the 0.90 attempt: 0.85+mr1 is the documented winner; runner reused with
the MEASPID watchdog bug fixed via `pkill -f 'measure\.py full'`).

Boot: ready 21:57:29 (~105 s), free-after-init 4.05 GiB (expected ~4.1), moe_cache_size
=1149, num_pages=7836, groups 378/288/288 slots (overlap disabled, same as campaign),
NO OOM at the first 8128 chunk - the 0.90 attempt's failure is gone.

Exact verdict line (fix8191_r085.log:52, after the repeat request):

```
[2026-09-18|22:01:10|core|rank=0] INFO     Prefill batch, #new-seq: 1, #new-token: 49, #cached-token: 65536, token usage: 0.13, #mamba-slot: 4/8, mamba usage: 0.50, #running-req: 1, #queue-req: 0, input throughput (token/s): 10.60
```

#cached-token: 65536 -> HIT, exactly the expected value; repeat adds only 49 new tokens
(65585 - 65536). Initial prefill: 8 full 8128 chunks + 561-token tail, wall 212.5 s,
peak VRAM 32084 MiB.

Perf: full 8128 chunks throughput 123.64 (chunk 1 warmup), 336.42, 336.40, 336.54,
336.46, 335.74, 335.25, then last full chunk 1748.80 (bogus drain artifact, same class
as the documented ~1552-1602) and 561-tok tail 558.96. Median of chunks 2-7 (excluding
warmup + bogus last full chunk) = 336.43 tok/s - ABOVE the expected ~293 class (same
nominal config measured 293 in the campaign; favorable variance, no regression).
Decode repeat: steady_tok_s=13.93 (p50 13.70, min inst 7.45), 63 completion tokens,
wall 8.99 s - within the expected 13.5-14.7 class.

Hygiene: preflight census clean (no wave processes, VRAM 1535 MiB, semaphores 11);
postflight: measure.py teardown leftover=[], VRAM back to 1535 MiB, semaphores 11
(no leak this time - graceful teardown, no watchdog kill). Only the two KNOWN pre-
existing orphans remain (PIDs 776127, 2010750, report-only, untouched).

VERDICT: PASS - the 8128-chunk acceptance item ("repeat at 8128 chunks must stay HIT")
is hardware-proven under the fix: HIT 65536 at 0.85+mr1, no prefill/decode regression.
Combined with section 2 (4096 -> HIT 65472 at 0.90), the fix-1 hardware gate is fully
PASS: radix reuse now works at BOTH chunk sizes.
