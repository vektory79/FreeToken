# Task 07: CPU compute tier (hybrid/offload decode)

Type: Code (measurement) | Agent: Code | Duration: ~1-2 weeks

## Goal

Give the file's gguf bank formats a CPU GEMV tier so decode can put expert
misses on DDR5 instead of PCIe, and decide hybrid vs offload MEASURED
end-to-end - never from a benchbw verdict string and never from paper math.

Worked example (the campaign this phase was distilled from): glm5next
IQ3_XXS/IQ4_XS/Q6_K, H=4096 I=2048 E=288 top_k=8, 42 MoE layers ->
microbench 53-66 GB/s (16/20 threads) vs gate >= 40 -> hybrid 16.67 tok/s
@64k vs offload 12.97 (commits 11f1a80 + 979e3fc + 63b9bff).

## Prerequisites (from task-00)

- Path to the working reference implementation where the file's formats are
  already supported (llama.cpp or other; user-provided at discovery).
  Kernels, LUTs, activation-quant functions and block layouts are transcribed
  or verified FROM IT.
- The CPU compute-tier support matrix: per format - existing CPU tier in
  cpu_moe_ext.cpp (scalar fp32-LUT vs integer W4A8), paired activation
  format, reference kernel file:line.
- Phase 6 offload baseline (the A/B anchor) and the discovery geometry:
  H, I, E, top_k, block sizes, per-projection format ids, partition census.

## Instructions for the subagent

1. MICROBENCH GATE FIRST (before any port commitment). Build a standalone
   harness that compiles the REFERENCE'S OWN kernels with the SAME ISA flags
   as the reference build (read build/CMakeCache.txt + per-variant
   flags.make; e.g. haswell tier = -O3 -mf16c -mfma -mavx -mavx2 -fopenmp),
   links shims only for unreferenced ggml.c symbols, and measures sustained
   GB/s on synthetic byte-exact blocks at the real decode geometry (per-pass
   bytes computed from the discovery census). Integer kernels are
   data-independent in timing - random payload bytes suffice; constrain only
   the f16 d fields to finite normals. Built-in parity: reference kernel vs
   scalar reference on random rows (max rel err < 1e-3) on EVERY run, before
   timing is trusted.
   GATE: CPU-leg sustained GB/s >= the box's effective PCIe gather bandwidth
   (this box: >= 40 vs PCIe ~43; glm5next measured 53-66). FAIL -> stop, the
   port cannot pay; report the number and keep offload.
   Run discipline (BINDING): SERIALIZED measurements - one measurement
   process at a time, hard timeout, PID-reap + ps census between points,
   orphan cleanup pre-run. Concurrent runs both leak and corrupt each other's
   numbers (the campaign's first microbench round was discarded for exactly
   this). OMP pinning env must be set BEFORE process start (libgomp parses
   in a pre-main constructor; setenv from main silently collapses all
   threads onto core 0).
2. PORT (per format whose matrix row says "no CPU integer tier"):
   a. Activation-quant pairing per ggml convention - each weight format has a
      PAIRED activation format: K-quants AND IQ-quants take q8_K (256-elem
      blocks, NEGATIVE iscale -127/amax, bsums[16] per block); q4_0-class
      takes q8_0 (per-32 blocks); nvfp4 takes the pg16/VNNI path. Look it up
      in the reference (quantize_row_* in ggml-quants.c), do not guess.
   b. Verbatim kernel transcription with per-function target attributes and
      per-function provenance comments (upstream file:line + license
      header); LUT equality MACHINE-CHECKED against the reference
      (byte-compare, not spot-checks); layout static_asserts from
      ggml-common.h.
   c. ISA tier + env override that MUST cap at CPU support (CPUID pick_isa);
      an uncapped override = SIGILL on hardware without the flags - allow
      cap-down only.
   d. Scalar tier stays compiled: fallback + correctness reference for tests.
   e. Quantize-once-per-projection reuse across the top_k experts of a step
      (the x row feeds gate+up; the intermediate row is per expert); scratch
      pre-sized BEFORE graph capture; no pageable hot-path allocations.
3. WEIGHTED POOLS: weight pool thread counts by partition LAYER COUNT (an
   even split starves the dominant partition: measured 6/16 threads ->
   2.96 GB/s effective vs 11.3 benched, 7.5/20 cores busy). Preserve
   floor-at-one + disjoint core sets; guard the explicit-path wrap corner
   at the split boundary.
4. BENCHBW: adding a format to the CPU-MoE bench requires a FULL rerun across
   ALL formats - a single-dtype run CLOBBERS the per-GPU profile and silently
   degrades OTHER formats to cap-1. Re-profile after ANY kernel change (the
   profile has no kernel-tier fingerprint). A CPU/PCIe ratio < 2.0 keeps the
   verdict string "offload" - EXPECTED (hybrid parity at best); the fraction
   still flows and the win is measured end-to-end.
5. ENGINE GATE: capability must be queried from the executor capability set
   (per-partition executor rejection), never hardcoded type ids or format
   names. Env overrides cap DOWN only.
6. E2E A/B (the decision): same engine, same client, same fill, single
   variable --moe-strategy (hybrid vs offload). Assert the boot log: the
   fetch fraction line (auto, NOT the cap-1 fallback), the weighted pool
   split, the isa tag. Gates: hybrid burst-min > offload burst-max; battery
   >= 20/24 on-topic; profile sanity of a second format (its fraction line
   still present).

## Validation gates

- Parity within the dfloat tolerance contract (8*2^-11*factor*|d| + 1e-3)
  against the scalar tier AND the reference dequant (gguf-py) - bit-parity is
  impossible for half-chain kernels; document why in the docstring.
- Tier A/B test: same fixtures through scalar and SIMD tiers
  (FREETOKEN_CPU_MOE_ISA=scalar vs avx2) - equal within tolerance.
- Full-path production-combo test: tier + weighted pools + merge, with an
  ISA skipif when the box lacks the flags.
- Negative: an env override above CPU support must be clamped, not SIGILL.
- benchbw profile contains EVERY format entry after the FULL rerun.
- `uv run pytest tests/ -m "not slow"` fully green.

## Traps (from [TRAPS.md](../TRAPS.md))

- T30: uncapped ISA override = SIGILL (cap-down only).
- T31/T36: a single-dtype bench rerun clobbers other formats' fractions; the
  profile has no kernel-tier fingerprint - re-profile after kernel changes.
- T32: concurrent measurement corrupts numbers - serialized discipline (D07).
- T33: the "handshake floor" measured at high f is PCIe fetch volume, not
  sync (~0.6-1.0 ms/layer is the real fixed band); isolate sync at f->0
  (cap-0 run) before any NO-GO conclusion. Fetch volume is ENDOGENOUS to
  CPU-leg speed: fast leg -> smaller f -> fewer bytes over PCIe.
- T34: an even pool split starves the dominant partition - weight by layer
  count.
- T35: projections without a working-reference anchor produce false NO-GO
  (the campaign's Task 06 estimate said 8-11 tok/s realistic; hardware with
  the ported reference kernels measured 16.67) - anchor every projection on
  a reference A/B (D08).
- T37: JIT helper CC/CXX leak - scope to the build invocation.
- T38: engine capability gates query the capability set, never hardcode ids.

## Output

- verification/: microbench report (ISA flags, bytes/pass, per-format +
  blended GB/s, parity statement), boot-log assert excerpts, A/B table
  (hybrid vs offload, same-session, single variable), battery digest,
  process census before/after.
- The honest verdict: hybrid vs offload for THIS file on THIS box, with the
  measured numbers - it replaces the phase-5 offload default only on a
  measured end-to-end win.

## Acceptance criteria

- [ ] I've created a git commit for this task
- Runs under ORCHESTRATION.md (quality loop + hygiene + commit gate).
