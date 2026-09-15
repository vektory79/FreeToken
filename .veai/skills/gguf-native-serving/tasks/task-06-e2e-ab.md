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

## Acceptance criteria

- [ ] I've created a git commit for this task
- Runs under ORCHESTRATION.md (quality loop + hygiene + commit gate).
