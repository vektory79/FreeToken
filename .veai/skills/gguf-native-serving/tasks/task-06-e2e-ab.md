# Task 06: E2E bring-up + A/B numbers

Type: Code (measurement) | Agent: Code | Duration: ~1 day

## Goal

Boot the real file end-to-end, measure A/B numbers vs the HF/FTW baseline,
run the quality battery, and deliver the honest verdict.

## Instructions for the subagent

1. Boot smoke (the acceptance gate for Phase 5): boot -> /ready 200 -> quote
   markers (histogram, partitions/overlap, clamp, plan) -> prefill probe ->
   decode probes (cached + fresh) -> SIGTERM -> VRAM check. Watchdog FAILPAT:
   AssertionError|OutOfMemoryError|Backend worker is gone|backend worker .* exited.
   On failure: verbatim log tail + cleanup, NO fixes (measurement only).
2. Tuning: the auto plan may give too-small KV for the desired prefill length.
   Levers: --num-tokens <N> (page-multiple), --memory-ratio <R>, explicit
   --moe-cache-size (CAREFUL: may fail-fast on per-signature floors), 
   --max-prefill-length 4096 or 8192 (8192 needs ~3.5 GiB transient headroom).
3. A/B: same prompts, same method, both configs (GGUF vs HF/FTW baseline).
   Prefill: server-log "input throughput" per chunk (exclude warmup chunk).
   Decode: same prompt for prefix-cache hit, stream, count content+reasoning.
   Fresh decode: new prompt, no cache.
4. Quality battery: ~24 greedy prompts (prose EN/RU/CJK, code, reasoning with
   digits), compare DECODED TEXT (not ids - the tokenizer may diverge on
   code/digits). On-topic rate, first-divergence-position, 3 example pairs.
5. Deliver: A/B table + quality table + honest verdict + artifacts under
   verification/phase6/ (or equivalent).

## Expected result classes (glm5next reference, post fix 5b72aba)

- Radix reuse on a repeated 65k prompt: #cached-token 65472 @4096-chunks /
  65536 @8191-chunks (5b72aba made reuse work at ANY chunk size; the old
  "0 cached @4096" class is obsolete). The deep-node snapshot-eviction gap is
  CLOSED (per-node snapshot_lru + evict_mamba FIFO ordering, 9732be0; victims
  stalest-validated first, tip never a victim).
- Prefill medians of full chunks (exclude the chunk-1 triton warmup, ~124
  tok/s on 8128-chunk runs and ~92 on 4096-chunk runs, and the bogus
  last-full-chunk line, T43): ~300 @4096 / ~336 @8128 tok/s class.
- Decode steady: 13.5-14.7 tok/s class.
- Boot fallback ladder under desktop VRAM pressure: 0.89 -> 0.90 -> 0.85+mr1
  for 8191 (8191@0.90 OOMs in the first-chunk triton do_bench, T28/T50).
- Post dense-q80-gemm campaign (2026-09-20, production defaults
  FREETOKEN_GGUF_DENSE_MTILE=64 + FREETOKEN_GGUF_MOE_MTILE=32 +
  FREETOKEN_GGUF_KDA_NSPLIT=2): prefill @8128 ~792-808 tok/s instrumented
  (T43 exclusions apply; CUPTI overhead ~1.2% symmetric - plain boots sit
  ~1.2% higher; T44 band applies vs the 792.08 anchor); decode 13.2-15.0
  tok/s (bs=1 MMVQ graph, CUPTI framing). The e2e bitwise output gate is
  NOT usable (T54: boot-to-boot bimodal nondeterminism); the arbitration
  battery is the reference methodology for output-agreement questions.
- GGUF->FTW fast path is now the RECOMMENDED way to serve large gguf
  checkpoints (boot 52.1 s vs 94-132 s bare serial; 8568807/e9ad38d/ef14efb/2e9fcf1).
- FTW classes: convert 147.5 GB bare gguf -> 134.24 GiB FTW in 268 s
  (`ft checkpoint --model <gguf> --out <dir> --moe-backend offload`, NO
  --quant-backend for gguf - repack defeats raw streaming); boot 52.1 s
  (banks 125G @ 4.12 GB/s, 30 s in-boot); prefill @8128 790.7 tok/s;
  decode @65k 14.8-15.5 tok/s, radix HIT 65536.
- Hybrid-radix per-chunk donation A/B replay class (fix3, 2026-09-21): the
  pre-change baseline is recorded FIRST, one variable changes, and the BEFORE
  stage is re-anchored against the recorded baseline (T44). Measured example
  (GLM-5.3-Flash hybrid, mr=1, 0.82/400k): turn-2 divergence-hit key
  #cached-token 48704, re-prefill 60184 -> 11480, full misses 2/12 -> 1/12,
  FIFO evictions stalest-validated-first.

## Debugging an IMA or OOM

- CUDA_LAUNCH_BLOCKING=1 makes the error synchronous (names the true launch).
- compute-sanitizer --tool memcheck pinpoints the exact address.
- The first failure may be one step earlier than it appears (async errors).
- Bisect: eager (--cuda-graph-max-bs 0) vs capture -> capture-specific or not.
- Per-layer hidden-state bisection: forward hooks on both models (sequential
  boots, never both in RAM); the first <<1 cosine layer names the broken module.

## Traps (from [TRAPS.md](../TRAPS.md))

- T25: The positional model path is REJECTED by argparse - use --model-path.
- T26: The "model" field in /v1/chat/completions is REQUIRED (422 without it) -
  read the served name from GET /v1/models.
- T27: TTFT on a cached prefix (~6s) is first-decode-step overhead, not re-prefill.
- T28: KDA autotune needs ~256 MiB do_bench scratch - if the eager tail has
  less, the first decode OOMs (a capture-config boot may not OOM - different
  VRAM profile). This is an artifact, not a regression.
- T29: max_tokens=8 with a reasoning parser returns empty content (tokens
  burned in reasoning_content) - not an error.
- T43: the last full chunk's "input throughput" line is bogus (~1552-1602 tok/s
  vs real ~290; tail-drain artifact) - compute prefill medians excluding
  chunk 1 (warmup) AND the last full chunk.
- T44: the BEFORE stage must reproduce the historical campaign baseline (within
  ~3% run variance) before the AFTER delta is trustworthy.
- T45: same-session A/B without stash: per-call env kill switch + two boots with
  an env flip; the JIT disk-cache compiles once for both boots; delete the
  switch after acceptance (FREETOKEN_GGUF_GROUPED_PREFILL precedent).
  SUPERSEDED (fix3): python-only A/B = two boots with a git-stash of the source
  files + a sha256 ledger, no env kill switch needed.
- T46: a kernel swap needs kernel-level liveness proof: nsys kernel-name contract
  + exact launch-count math; bare stdlib loggers are invisible in boot logs -
  catch one-time INFO markers via a PYTHONPATH sitecustomize probe.
- T47: before skipping a validation item for "no tooling", search ALL .tasks
  campaign folders and /tmp; copy volatile /tmp artifacts into .tasks immediately.
- T48: silent radix MISS on an identical repeat when the FINAL prefill chunk
  ends below the x64 track boundary - the pre-5b72aba scheduler dropped
  mamba_last_track_seqlen across chunk transitions (config-independent,
  chunk-size-dependent); post-fix expectations above.
  SUPERSEDED (fix3): L is now consumed at continuation creation (per-chunk
  donation, 7080824); the chunk-size-dependent MISS signature stays the
  diagnostic to watch.
- T49: a NaN/Environment failure that passes in isolation with AND without the
  diff is suite-ordering Environment, not a regression - verify isolated A/B
  before blaming a change.
- T50: the boot slot-floor gate fail-fasts when desktop VRAM tax lands
  (~492 MiB) - retry ladder in the Expected result classes above.
  EXTENDED (fix3): with per-chunk donation the pool is load-bearing
  mid-prefill - commit frequency ~8x per long prompt; watch the slot-floor
  fail-fast ladder.
- D09: nsys live-serve profiling: `nsys launch` rejects -o; launch +
  start-after-ready + --cuda-graph-trace=node works; a bare mid-run start yields
  a report without eager kernel activities; analyze via sqlite export; bracket
  the window by "input throughput" lines.
- D10: Step-0 split before a kernel-swap design: one nsys pass + sqlite export
  splits GEMM/copies/rest per chunk with kernel-name accounting; measure BEFORE
  designing or the bottleneck ranking is wrong.

## Acceptance criteria

- [ ] I've created a git commit for this task
- Runs under ORCHESTRATION.md (quality loop + hygiene + commit gate).
