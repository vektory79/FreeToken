# TASK: MMQ prefill for GGUF MoE (v0 sort experiment -> v2 grouped MMQ kernel)

Status: Step 0 DONE 2026-09-17 (split pinned, see below); v0 code + hardware
A/B DONE 2026-09-17 - sort ACTIVE and measured, NO prefill gain (see v0
section); v2 code DONE + hardware A/B DONE 2026-09-18 - grouped path ACTIVE
and measured, +14.4/+15.6/+16.4% prefill (4096/6144/8128), far below the
2.3-2.7x ceiling (see v2 hardware A/B section).
Composed 2026-09-17 from a code+header kernel study (roofline
reconciled with hardware measurements; summary memory:
`ft-gguf-prefill-mmq-roofline`; campaign artifacts in the git-ignored ft-gguf-serve-tuning
worktree, results mirrored under [baselines/ft-gguf-serve-tuning/](../../baselines/ft-gguf-serve-tuning/)).
Read CONTRIBUTING.md first - it is binding. Line anchors verified 2026-09-17 on
vektory79 @ 1a444e4; re-verify before editing (they drift). Private-use scope for
GGUF glm5next: no push, no upstream PR.

## Goal

Raise GGUF prefill throughput (today ~262 tok/s @4096 chunks, ~293 @8128) by removing
the 226x redundant expert-weight re-reading in the prefill MoE kernel. Staged:
v0 (sort experiment, ~2x, ~0.5-1 day) -> v2 (grouped MMQ, ~2.3-2.8x, ~3-6 days).
Decode path and CUDA-graph capture are NOT touched; the new path branches on
`is_prefill` only.

## Verified background (do not re-derive; re-anchor before editing)

Geometry (from the real GGUF header via gguf-py): H=4096, I=2048, E=288 experts,
top_k=8, 46 blocks with leading_dense=3 -> 42 routed MoE layers. Expert types:
39x (IQ3_XXS gate, IQ3_XXS up, IQ4_XS down), 2x (IQ3_XXS, IQ3_XXS, Q6_K),
1x (IQ4_XS, IQ4_XS, Q6_K); 10.88 MB/expert (group 1), 134.4 GB expert total.

Current kernel (`kernel/csrc/gguf/moe_vec.cuh`): grid = (row_blocks, 1, tokens*top_k);
each block computes ONE output row of ONE (token,expert) pair and re-reads that row's
packed blocks -> every pair streams the FULL expert weight set (10.88 MB). No
cross-token weight reuse; same-expert revisits are ~226 pairs apart and L2 (~90-130 MB)
cannot hold 3.13 GB/layer. Traffic model (reconciles with BOTH measured points):
65,024 pairs x 10.88 MB = 707 GB/layer = 29.7 TB/chunk = 16.6 s at 1.79 TB/s; chunk
8128 = 27.7 s (GEMM 16.6-20.8 + full-layer copies ~3.0 + rest 4-8); chunk 4096 = 15.5 s.
The MoE GEMM is ~100% VRAM-BW-bound, 60-75% of chunk time. Activation is already
q8_1-quantized once per token (reused across all 8 experts) - keep that.

Roofline (8128-token chunk): today 293 tok/s; v0 -> MoE 4-6 s -> ~580-680 tok/s;
v2 -> MoE 1.5-3 s -> chunk 10-12 s -> ~680-810 tok/s; after v2 the bottlenecks become
the per-chunk full-layer copies (~3.0 s = 131.5 GB at PCIe 43.6 GB/s) and the
"rest" (attention/GDN/DSA/dense, 4-10 s error bar). Hard floor without prefill overlap
~900-1500 tok/s. Prefill overlap is NOT in scope (needs 576 slots/partition; group-1
has 410 at ratio 0.85). A "dequant segment to bf16 + cuBLAS" alternative is KILLED by
numbers: 14.5 GB bf16/layer transient -> +610 GB/chunk traffic, no faster than today.
Dequant must be fused into GEMM tiles - that is exactly MMQ.

## Step 0 - pin the GEMM/copies/rest split (first, CPU-side only) - DONE 2026-09-17

One nvtx/profiler pass over an existing boot (torch profiler or nsight, no new flags)
to measure the exact GEMM vs copies vs rest split of an 8128 chunk. "Rest" (4-10 s) is
the dominant uncertainty of the v2 ceiling; reconcile against the NVFP4 510-tok/s
reference on the same geometry (it pays the same rest). Write the numbers into this
TASK.md before starting v0/v2.

Measured (nsys 2026.1.3 interactive session, zero code changes; full method, raw
tables and cross-checks in [step0-profile-mmq.md](../../methods/step0-profile-mmq.md); trace
`report1.nsys-rep`, split `step0_split.json`). Winner config, port 18801, single
serial run, boot at memory-ratio 0.85 (free-after-init 4.05 GiB, no fallback needed).
Steady full-chunk time T = 28.77 s median (28.72-28.90 spread, 282-283 tok/s this
run vs campaign 291-293; ~3% run variance); capture window W = 113.89 s = 3.958
chunk-equivalents; launch counts cross-check (moe_vec_q 126/chunk = 42 layers x 3
projections; fast_index_copy_multi 42/chunk).

Pinned split per 8128-token chunk:

- MoE GEMM `moe_vec_q`: 18.13 s = 63.0% (501 launches; effective 1638 GB/s = ~92%
  of the ~1.79 TB/s VRAM roofline -> BW-bound confirmed, traffic model stands).
- Expert copy/fetch `fast_index_copy_multi`: 3.15 s = 10.9% (42 launches/chunk;
  ~3.13 GB/layer read from host banks at ~42 GB/s, PCIe-class; it is a copy KERNEL,
  not H2D memcpys - those measure ~0).
- Rest: 7.49 s = 26.0%, of which dense quantized GEMM `mul_mat_q8_0` 6.04 s (21.0%,
  ~312 launches/chunk) and only ~1.45 s attention/GDN/DSA/router/quantize/glue;
  GPU idle 0.015 s/chunk (0.05%) - the GPU is never CPU-starved.

Verdict: the 16.6-20.8 / ~3.0 / 4-10 model HOLDS (18.1 + 3.1 + 7.5 = 28.77 s).
New findings: (a) rest is GPU-kernel-bound and dominated by the dense q8_0 GEMM,
which becomes the top bottleneck right after v2; (b) the copies term is a PCIe copy
kernel, so v2 alone cannot remove it; (c) tail anomaly: the last full 8128 chunk
always reports ~5.2-5.4 s (1552-1602 tok/s) and the 561-token tail ~1.15 s in every
run - hardware-impossible vs the measured split, a tail report/pipelining artifact;
excluded, median-of-full-chunks is robust to it, follow-up optional.

Implied ceilings from the measured numbers: v0 (sort, same kernel) ~412-566 tok/s
(L2-bound re-reads at 4-8 TB/s; naive GEMM/2 basis 412); v2 (grouped MMQ, weights
read once at the measured 1638 GB/s) MoE 0.08-0.13 s -> chunk ~10.7 s -> ~758 tok/s
(2.68x), then bottleneck ranking: dense q8_0 GEMM 6.04 s > expert copies 3.15 s >
attention etc. ~1.45 s; hard floor ~758 tok/s with copies+rest as measured.

## v0 - sort pairs by expert + perm-aware moe_vec (~0.5-1 day)

1. Python, `layers/moe.py` gguf prefill branch (~425-490): when `is_prefill`, compute
   `order = torch.argsort(topk_ids.flatten())` (stable), build the inverse perm, and
   pass permuted pair indices to the kernel call (or scatter back after). Keep
   `_assert_moe_vec_chunk` (issue #186: grid = tokens*top_k <= 65535) as is.
2. Kernel, `moe_vec.cuh` (~10-line change): accept the pair->expert-sorted order so
   consecutive blocks hit the same expert (L2 reuse of its 10.88 MB) instead of
   token-major random order.
3. Contiguity guard on the activation passed to the kernel - lesson 028f2d9:
   non-contiguous activations fed the row-major-assuming ggml kernels and produced
   garbage gates.
4. A/B on hardware (protocol below): DONE 2026-09-17, full report in
   `v0-ab.md` / `v0-ab-results.json` (this folder). Result: prefill 289.54 ->
   291.26 tok/s (+0.6%, within the ~3% run variance) - NO GAIN vs the
   expected 412-566; decode 14.92 -> 15.23 tok/s (unchanged, noise); boot
   146.6 -> 124.3 s (JIT cache state, not code). The boot-log liveness line
   can never appear (defect: layers/moe.py logs via a bare stdlib logger no
   handler covers; lastResort drops INFO); a PYTHONPATH probe boot confirmed
   the marker fires - the sorted path IS active, and the measured result
   stands: the L2-reuse model does not translate into throughput. Quality
   battery SKIPPED (no reusable tooling in the campaign folder). The tooling
   gap was later closed - the battery runner + divergence analyzer now live at
   `.tasks/mmq-prefill-kernel/quality/` (2026-09-18) - but v0 was already
   dropped by then and was never battery-tested on hardware.

v0 outcome (2026-09-17): experiment complete, negative result. The
expert-sorted pair order is a throughput no-op on RTX 5090 - the per-pair
weight re-reads are not absorbed by L2 even with expert-clustered block
scheduling. v0 should not be kept on performance merits; v2 (grouped MMQ,
weights read once per expert per layer) remains the path that changes the
traffic term. User decision 2026-09-17: DROPPED - the six files were
reverted to HEAD 1a444e4 (diff preserved at /tmp/v0-changeset-backup-1a444e4.patch,
tree verified == HEAD, tests green); v2 proceeds from HEAD.

## v2 - grouped MMQ prefill (~3-6 days)

What ALREADY exists in `kernel/csrc/gguf` (do not rewrite):
- `mmq.cuh`: tile engine for Q4_0/Q8_0/Q6_K/Q3_K/Q4_K (vendored vLLM/llama.cpp).
- `vecdotq.cuh`: `vec_dot_iq3_xxs_q8_1` (~l.1848) and `vec_dot_iq4_xs_q8_1` (~l.2017)
  - the iq scalar dot functions are ALREADY vendored; no iq dot-porting needed.
- `gguf_kernel.cu:335`: `ggml_moe_a8` - the GROUPED MoE MMQ host entry taking
  `sorted_token_ids / expert_ids / num_tokens_post_padded` (the vLLM
  `moe_align_block_size` trio) with a per-type switch that stops at Q6_K.
What to build:
1. Vendor the iq tile glue (allocate_tiles_iq3_xxs / allocate_tiles_iq4_xs,
   load_tiles, mma wrappers; ~150-250 lines per type) from upstream llama.cpp
   `mmq.cu` (exists since 2024). Vendor verbatim - iq3_xxs tiles are lop3/LUT-heavy.
   Add MIT attribution headers while vendoring (known repo gap: vendored
   kernel/csrc/gguf files lack them).
2. Add cases 18 (IQ3_XXS) and 23 (IQ4_XS) to the two switches: `ggml_mul_mat_a8`
   (gguf_kernel.cu ~192) and `ggml_moe_a8` (~335); update python `_MMQ`/`_IQ_ONLY`
   sets in layers/gguf.py:31-36 accordingly (dense-batch NotImplementedError for iq
   can then be lifted for MMQ formats).
3. Python prep for the grouped entry: `moe_align_block_size` equivalent
   (sorted_token_ids/expert_ids/num_tokens_post_padded). Check whether the NVFP4
   grouped path already ships one; otherwise port vLLM's ~30-line torch version.
4. Wire the prefill branch in layers/moe.py (~425-490) to the grouped entry for the
   three projections (gate/up with top_k grouping; down with its interleaved-row
   convention - see the existing moe_vec call, top_k=1 there), keeping the gated
   epilogue (gated_act_and_mul, alpha/limit) and the topk_weights scatter/sum.
5. Bind via kernel/gguf.py::_module (JIT; clang++ host required - g++ hosts fail on
   ATen headers; disk JIT cache keyed on source+flags makes this one-time).
6. Tests: parity vs gguf-py dequant reference per format with max-ulp threshold that
   FAILS loudly on drift; regression: decode path still uses moe_vec (is_prefill
   branch only); graph capture still passes (boot with default capture); issue #186
   chunk-cap test unchanged; extend existing test files first
   (tests/kernels/test_gguf_quant.py, tests/models/test_gguf_expert_banks.py).
7. benchbw: no changes (f comes from the CPU leg; MMQ does not affect it).

Known risk: tail experts with n_e = 1-2 tokens dilute tile efficiency (padded
segments); measure and report, do not special-case prematurely.

### v2 hardware A/B - DONE 2026-09-18 (env-flip A/B, no stash)

Full report: `v2-ab.md` / `v2-ab-results.json` / `v2ab_probe_liveness.json`
(this folder). Same-session A/B, env kill switch flipped between boots
(FREETOKEN_GGUF_GROUPED_PREFILL=0 vs 1), 3 boots per stage (4096 / 6144 /
8191; chunk size is a boot flag), winner flags, port 18801, serial+reaped.

- Prefill (median full chunks, excl. c1 + last-full artifact):
  4096: 263.82 -> 301.88 tok/s (+14.4%); 6144: 281.38 -> 325.28 (+15.6%);
  8128: 289.92 -> 337.61 (+16.4%). Campaign refs 262 / 281 / 292.30 -
  BEFORE reproduces the campaign baseline, so the A/B is valid.
- Decode @65k radix-hit: 13.94 -> 14.60 tok/s (+4.7%, noise class; code
  byte-identical). Boot ready_s unchanged (no JIT in any boot: the fix wave's
  pytest runs had already compiled the v2 extension into the disk cache).
- Liveness CONFIRMED at kernel level (nsys capture-from-launch,
  --cuda-graph-trace=node, report3.nsys-rep): prefill chunks contain the
  grouped kernels moe_iq3_xxs / moe_iq4_xs / moe_q6_K at exactly
  42 layers x 3 proj x 9 chunks = 1134 + moe_align 42 x 9 = 378 (sgl backend,
  triton `_moe_align_small` unused here) and ZERO moe_vec_q inside the chunk
  span; decode shows moe_vec_q 42 x 63 x 3 = 7938 inside the CUDA-graph
  replays. Marker line "gguf moe grouped mmq prefill active" fired once
  (sitecustomize probe; bare-logger trap).
- Verdict: REAL but small gain. Traced grouped per-call times @8128 are
  ~100 ms per projection (grouped MoE ~9.5-10.5 s/chunk vs moe_vec 18.13 s,
  ~1.8x on the MoE term) - the read-once roofline (1.5-3 s) was NOT reached;
  working hypothesis (unverified): the vendored vLLM moe.cuh schedule re-reads
  expert weights per 4-token m-block (~57x per expert), cutting traffic ~4x
  instead of to-once, so the Step-0 2.3-2.7x ceiling does not apply to this
  kernel structure. Rest of chunk (fetch copies 3.15 s + dense q8_0 GEMM
  6.04 s + attention ~1.45 s) now dominates again.
- Quality battery DONE 2026-09-18 (see `quality-ab.md` / `quality/`): env-flip
  A/B, moe_vec control (env0) vs grouped prefill (env1), 24 prompts x 2 boots,
  winner flags, port 18801. VERDICT: PASS on all bar items - 24/24 on-topic in
  both stages, all 4 digit prompts correct in both (80 km/h, Friday, 6 apples,
  1024), zero task-level flips (the single div@4 suspect is same-task wording
  drift), div min/median/max 4/193/572 chars with 5/24 pairs above the
  historical 49-339 band, all read as the same answer. No quality regression.
- FINAL VALIDATION STATUS: v2 protocol COMPLETE - perf A/B (+14.4/+15.6/+16.4%
  prefill) + kernel-level liveness + quality battery all green; ready for the
  user's commit decision.
- Kill switch FREETOKEN_GGUF_GROUPED_PREFILL REMOVED 2026-09-18 (user request,
  post-validation): grouped MMQ prefill is now unconditional for supported
  types (all declared ggml block sizes > 0); unsupported types (block size 0,
  e.g. IQ2_XS/IQ4_NL) still prefill via the moe_vec fallback with
  _assert_moe_vec_chunk. Verified: 126/126 tests green, zero env references
  left in code/tests/docs, two boot smokes pass (marker fired exactly once,
  battery prompts on-topic, decode + radix cache ok). The env-flip A/B
  methodology above is now historical; env0's path survives as the fallback.
- nsys lessons (see v2-ab.md deviations): nsys 2026.1.3 `launch` rejects -o;
  mid-run `nsys start` produced a report without eager kernel activities;
  working recipe = launch + start-after-ready + --cuda-graph-trace=node.

## Hardware validation protocol (both stages)

- Harness: [measure.py](../../harness/serve-measure/measure.py) (already patched: decode steady
  skips the pre-prefill SSE frame; throughput regex anchored on
  "input throughput (token/s):"). Port 18801; llama-swap stopped.
- Config: winner flags `--moe-cache-auto --kv-reserve-tokens 500000
  --kv-cache-dtype fp8 --memory-ratio 0.85 --max-prefill-length 8191
  --moe-strategy hybrid --moe-cpu-threads 16 --max-running-requests 1`. Caveat: at
  0.89 the boot fail-fasts on the early floor gate when the desktop eats VRAM
  (hoptodesk ~492 MiB case) - use 0.90 fallback and note the deviation.
- Metrics: prefill median of full chunks from server "input throughput" lines
  (exclude chunk 1 = triton autotune warmup), decode back-to-back @65k, quality
  battery (24 prompts vs bare-GGUF outputs; wording-level parity bar), boot time.
- Hygiene: serial runs only; watchdog patterns
  AssertionError|OutOfMemoryError|Backend worker is gone; SIGTERM then SIGKILL;
  pgrep '[f]t serve' census between boots (bracket pattern - a plain pkill kills the
  wrapper itself); final census clean.
- Report A/B numbers against main before/after (CONTRIBUTING requirement for perf
  changes): prefill tok/s per stage, decode tok/s, boot time, quality battery score.

## Commits

Conventional Commits, lowercase, imperative; one change per PR. Suggested:
`perf(kernels): sort moe pairs by expert for prefill reuse` (v0) and
`feat(kernels): grouped mmq prefill for iq3_xxs/iq4_xs gguf experts` (v2).
Commit only when the user asks. Never push.

## Out of scope (separate tasks)

- [fix1-radix-track-seqlen/TASK.md](../fix1-radix-track-seqlen/TASK.md) - radix reuse fix (independent).
- [ftw-gguf-fastpath/TASK.md](../ftw-gguf-fastpath/TASK.md) - FTW load fast path (independent).
- Prefill overlap enablement (576 slots/partition) - capacity redesign, not this task.
