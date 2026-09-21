# TASK: S - complete the GGUF->FTW serve fast path (model-load acceleration from disk)

Status: DONE - committed on vektory79 (8568807, e9ad38d, ef14efb, 2e9fcf1; no push).
Composed 2026-09-17 from a hardware-measured campaign
(artifacts: `.tasks/ft-gguf-serve-tuning/`, summary memory:
`ft-serve-gguf-tuning-campaign-2026-09`). Read CONTRIBUTING.md first - it is binding.
Scope note: GGUF glm5next serving is private-use local work on branch vektory79; the
upstream issue #34 gate was waived by the user (2026-09-13) - keep it that way, no push,
no upstream PR unless the user asks.
Line anchors verified 2026-09-17 on vektory79 @ 1a444e4; re-verify before editing.

## Goal

Make `ft checkpoint` + `ft serve` work end-to-end for a GGUF glm5next checkpoint so that
`ft serve <ftw-dir>` uses the FTW fast loader (multi-worker reads with O_DIRECT, born-pinned
allocation, pin-as-completed). Bare-GGUF boot today is ~94-132 s on the RTX 5090 rig
(serial ~134 GB copy into pinned banks + a 1-thread PinPipeline). The NVFP4 reference
ladder on the same rig was serial 245 s -> parallel 131 s -> FTW 72 s, so the expectation
for this task is boot ~50-60 s. The expectation is an EXTRAPOLATION - measure before/after
and report real numbers.

## Verified current state (do not re-derive, but re-anchor before editing)

- Converter side ALREADY works for bare .gguf: `convert_checkpoint`
  (python/freetoken/checkpoint/convert.py) accepts it and streams RAW ggml blocks without
  dequantization - per-layer entries "{bank}#L{id:05d}" of dtype uint8 shaped
  [num_experts, rows, row_bytes], ALIGN-4096 enforced by FTWWriter;
  `quant_format="gguf"` is written into the finalize index meta (the `writer.finalize({...})`
  call near line 306). gguf_types are available in memory at that point as
  `banks.gguf_types` (moe/expert_banks.py ~218) but are NOT written into the meta.
- Serve-side FTW reader is format-agnostic: `load_ftw_banks`
  (python/freetoken/checkpoint/ftw.py ~435-659) reads per-layer entries into HostBanks with
  multi-worker reads + O_DIRECT fallback (FTWReader._ensure_mode ~243-260, read_into
  ~320-358, worker pool >= 16), born-pinned backing when the env opt-in is set
  (~479-482), pin-as-completed via PinPipeline; bank names pass canonical_role unchanged
  (moe/legacy_format.py:11-19); ExpertBanks accepts alphas=None (expert_banks.py:31-58).
- Per-signature partitions for the gguf offload cache are derived from the SHAPES of loaded
  banks (`engine._group_bank_layers`, engine/engine.py:663-680) and
  `_split_moe_cache_budget` (engine.py:681-768) works on banks.sources - so partitions and
  per-group slot floors should come up unchanged for FTW-loaded gguf banks. Verify at boot.

## The three gaps (in the order to fix them)

### Gap 1 - KeyError("gguf") at load time (hard blocker, one line + test)

`ftw.py` ~620: `kind, kernel = kind_kernel_for(quant_format) if quant_format is not None
else (None, None)` -> `kind_kernel_for("gguf")` raises KeyError because LEGACY_FORMAT
(python/freetoken/moe/legacy_format.py ~26-46) has no "gguf" entry. GGUF banks carry no
QuantKind/kernel (loaded path runs with kind=None).
Fix: add a "gguf" mapping to (None, None) in legacy_format (or special-case in ftw.py).
Keep the existing contract comment: "the quant_format tag names the (kind, kernel) they
were packed for" - extend it to cover the kind=None case explicitly.

### Gap 2 - gguf_types lost between converter and serve (the functional blocker)

`load_ftw_banks` returns `ExpertBanks(quant_format, sources, **alpha_kw,
layer_residency=applied, kind=kind, kernel=kernel)` (ftw.py ~657-664) WITHOUT gguf_types.
Downstream: engine.py ~969-973 sets `cache.gguf_types=None` ->
cpu_executor.py:106-112 raises "gguf expert banks need cache.gguf_types..." at init;
without an executor fix the first decode dies in layers/moe.py:439; the benchbw gguf alias
also fails to resolve (moe/benchbw.py ~517-519).
Fix (three small pieces):
a) converter: persist `gguf_types` in the finalize index meta (nested list
   [[gate, up, down] per MoE layer], straight from `banks.gguf_types`);
b) `load_ftw_banks`: read the meta, validate length == number of MoE layers, pass into
   ExpertBanks (add a kwarg; default None so nvfp4 checkpoints are unaffected);
c) engine wiring: the existing bare-gguf path already assigns cache.gguf_types from
   ExpertBanks - confirm the same assignment runs for the FTW source (engine.py ~969-973).

### Gap 3 (optional, same PR or follow-up) - early floor-gate is silent for FTW

`engine._gguf_auto_floor_gate` (engine.py:799-818) calls `gguf_signature_groups(model_path)`
which header-scans BARE .gguf only (`is_gguf_path(ftw_dir) == False` -> returns None) ->
for an unfundable plan the reject comes only AFTER banks load (+~85 s instead of +~17 s).
Option: derive the per-group slot bytes from the FTW index per-layer entry shapes (same
signatures `_group_bank_layers` computes at engine.py:663-680). If the shape provenance is
ambiguous, skip Gap 3 and leave the late check as the only barrier - it is already honest.

## Constraints and known traps

- Do NOT generalize beyond gguf; nvfp4 FTW checkpoints must keep loading byte-identically
  (golden Recorder tests exist for the shim semantics; run them).
- Any new hf_config mutation must hasattr-guard first (frozen dataclass shim crashes
  unconditionally otherwise - regression 8ce2657). Prefer zero hf_config mutation here.
- Tracker note-count contract: for glm5next the completion hook expects exactly ONE
  tracker.note(bank) per layer (after all 3 role copies). The converter sink already does
  copy_ + note per layer; make sure the FTW load path preserves the same per-layer note
  count, else pinning silently skips entirely and the partitioned gather IMAs at the first
  decode miss (the q4_0/gemma4 lesson; see memory ft-offload-banks-pinned-host).
- The gguf CUDA kernels JIT-compile on first use and need a clang++ host (g++ hosts fail on
  ATen headers); disk JIT cache makes this a non-issue after the first build
  (memory ft-gguf-kernel-jit-toolchain).
- No tests exist today for load_ftw_banks - coverage is created from scratch; put it in
  tests/checkpoint (extend existing files where they fit).

## Tests

1. Regression: `kind_kernel_for("gguf")` does not raise; returns (None, None).
2. Roundtrip (CPU): build a SMALL synthetic GGUF fixture (respect 32-byte tensor alignment
   in the file and the FTW ALIGN-4096 entry invariant), convert via the converter sink,
   then `load_ftw_banks`: assert per-layer widths (the 3-signature heterogeneity of the
   real file is the case to cover), role order gate/up/down (explicit role order, never
   slot order), gguf_types round-trip, kind=None, alphas=None accepted, layer_residency
   values sane.
3. nvfp4 FTW loading unchanged: existing golden/checkpoint tests pass.
4. If Gap 3 done: the early gate returns groups from FTW meta and rejects unfundable plans
   BEFORE banks load (sentinel test with monkeypatched load path, mirroring the existing
   gate-before-load tests in tests/moe/test_gguf_expert_banks.py).

## Hardware validation (GPU, after tests pass)

1. Disk: converting the real file needs ~140+ GB free - check before starting
   (the source is /media/ai/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/
   GLM-5.3-Flash-UD-Q3_K_XL.gguf, 147.5 GB). Verify the exact `ft checkpoint` CLI flags
   from checkpoint/__main__ before running; expect the conversion to take minutes
   (NVFP4 reference: 160 GB in 439 s).
2. Boot the FTW dir with the winner serve flags measured in the campaign:
   `--moe-cache-auto --kv-reserve-tokens 500000 --kv-cache-dtype fp8 --memory-ratio 0.85
   --max-prefill-length 8191 --moe-strategy hybrid --moe-cpu-threads 16
   --max-running-requests 1` (port 18801 for tests; llama-swap must be stopped; the
   0.89 ratio can fail-fast on the early floor gate when the desktop eats VRAM).
3. Measure and report: boot time (start -> /ready 200) vs bare-GGUF reference 94-132 s;
   slots per partition; "Free memory after initialization"; prefill median (server
   "input throughput" on 8128-token full chunks, skip chunk 1 autotune warmup, ~293 tok/s
   expected); decode @64k (radix HIT expected on repeated identical 65k prompts);
   a short quality battery (24 prompts, compare with bare-GGUF outputs - wording-level
   parity is the bar; see the campaign notes on mHC residual-norm noise).
4. Hygiene: serial runs, watchdog patterns AssertionError|OutOfMemoryError|Backend worker
   is gone, SIGTERM->SIGKILL, pgrep '[f]t serve' census between boots (bracket pattern -
   a plain 'ft serve' pkill kills the wrapper itself).

## Commits

Conventional Commits, lowercase, imperative; one change per PR. Suggested split if wanted:
`fix(checkpoint): accept gguf quant_format in ftw bank loading` (Gap 1) +
`feat(checkpoint): persist gguf_types in ftw index and restore on serve` (Gap 2).
Commit only when the user asks. Never push.

## Out of scope

- Parallel fill of bare-gguf banks (only worth doing if the phase breakdown of boot shows
  bank reading dominates; the FTW path replaces that need).
- Larger prefill kernel work; radix fixes (see .tasks/fix1-radix-track-seqlen/TASK.md).

## Outcome (implementation + hardware validation; commits 8568807/e9ad38d/ef14efb/2e9fcf1)

Note: the regression test lives in tests/moe/test_legacy_format.py (extracted, fails-before
proven); tests/checkpoint/test_ftw_gguf_banks.py was staged in two dependency slices so each
commit is bisect-green. A 4th tiny commit 2e9fcf1 carries the unrelated offload_cache
NameError fix to keep the gap-3 commit scoped.

All three gaps implemented (Gap 3 included) plus the capability fallback it required:
- kind_kernel_for("gguf") returns (None, None) via the tag-keyed _KIND_KERNEL (LEGACY_FORMAT
  is the inverse (kind, kernel)-to-tag map; the entry must NOT go there).
- gguf_types persisted in the finalize meta and restored by load_ftw_banks (structural
  validation, reader closed on error, old-converter fail-fast with a re-convert message).
- gguf_ftw_signature_groups + gguf_expert_bank_types FTW index fallbacks make the floor
  gate, the hybrid rejection and cpu-executor viability work on FTW dirs.
- _build_fused_copy_plan rejections now carry the bank name (was NameError).

Quality loop: review x2 (PASS-with-nits / FAIL on one Major) + triage, 9 findings, all
fixed, none rejected. Tests: new tests/checkpoint/test_ftw_gguf_banks.py + extended
tests/engine/test_cache_budget.py; battery 175 passed / 7 skipped.

Hardware (winner flags from the tuning campaign):
- Conversion: 147.5 GB gguf to 134.24 GiB FTW in 268 s (ft checkpoint --model <gguf> --out
  <dir> --moe-backend offload; no --quant-backend for gguf).
- Boot 52.1 s start-to-ready vs 94-132 s bare-GGUF (banks: 125G at 4.12 GB/s in 30 s); slots
  388/288/288; free-after-init 4.06 GiB; fetch fraction 30.7%; pools [15,1,1] avx2.
- Prefill median 790.7 tok/s at 8128 chunks vs current bare anchor 792-808 (the ~293 figure
  above was the pre-dense-q80/m-tile anchor; stale). Decode at 65k 14.78/15.52/15.02 tok/s,
  radix HIT 65536 on all three repeats. Battery 24/24, wording-level parity.
- Artifacts in this dir: run-convert-wave1.sh, verify-convert-wave1.py, run-wave2-measure.py,
  wave2-parity.py, run-wave2/.
