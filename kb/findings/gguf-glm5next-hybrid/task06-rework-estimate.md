# Task 06 rework estimate: handshake + weighted affinities + SIMD tier

Read-only analysis, 2026-09-16. Branch vektory79 @ eb7de4c (clean). All hardware numbers
are cited to the Task 05 verification artifacts (`verification/decode-numbers.md`,
`bisect-notes.md`, `fraction-overflow.md`, `ab-table.md`); all code sites re-verified in
source this session. Question: can the re-scoped Task 06 reach the 14.3 tok/s decode
gate, and what does each workstream cost?

Anchors (measured, real 147.5 GB file, 64k fill, 42 MoE layers, 3 partitions):

| config | tok/s | ms/step | decomposition |
|---|---|---|---|
| offload control | 14.25 | 70.2 | same-session anchor |
| hybrid flag-sync auto f=0.78 | 2.99 | 334.4 | 70.2 base + 126.7 handshake + 137.5 CPU leg (exact: 126.7+137.5=264.2=334.4-70.2) |
| hybrid cap-6 (CPU leg ~0) | 5.08 | 196.9 | 70.2 + 126.7 -> handshake = 3.02 ms/layer |
| hybrid hostfunc sync | 3.48 | 287.4 | 70.2 + ~79.7 (1.90 ms/layer) + 137.5 |
| hybrid OVERLAP=0 | 2.57 | 389.1 | overlap worth ~15% |

Per-layer volume (fraction-overflow.md, pool 1/3 = 39 layers): missing 5.4044
experts/layer-step, fetched 4.5811, cpu 0.8233, miss_rate 0.6755; ~12.7 MB per
expert (407 MB / (39 x 0.8233)); realized gather 1.46 ms/layer ~= 4.58 x 12.7 MB /
40.1 GB/s. Benched legs (benchbw, Task 03/05): PCIe 40.1 GB/s, CPU 11.3 GB/s
(20 threads, scalar) -> fraction 0.780.

## 1. Handshake anatomy: where the 1.9-3.0 ms/layer goes

`_decode_hybrid` (layers/moe.py:292-339) per layer k, in stream order:

| # | step (site) | classification |
|---|---|---|
| 1 | `raw = topk_ids.clone()` (moe.py:308) | artifact. Needed only because step 2 rewrites ids in place; the kernel could emit CPU-side ids to a second output. Cost: us. |
| 2 | `ensure_experts_hybrid` (offload_cache.py:900-921 -> offload_kernels.py:43) | FORCED per layer (LRU/cap/fraction state evolves per layer) but NOT a sync: one fixed-shape Triton launch, graph-safe, us-scale. |
| 3 | `record_decode_stats_hybrid` | artifact (measurement only, <2% step, fraction-overflow.md method note). |
| 4 | `on_gpu`/`cpu_ids` split (`torch.where`, `contiguous`) | forced concept (device-side split), us-scale each. |
| 5 | `decode_submit` (cpu_executor.py:825-869): 3 small D2H copies (x bf16 [bs,H] ~8 KB, ids int32, w fp32) + 2x `cuStreamWriteValue64` (done=0, ready=1) | D2H of ROUTING is FORCED per layer (the CPU needs layer k's routes); the 3 separate copies and the 2-memop doorbell are artifacts (one packed slab, one write). |
| 6 | `copy_missing` (offload_cache.py:1056) + `_expert_gemm` (moe.py:403) | FORCED work: PCIe fetch of f x 5.40 experts (~1.45 ms/layer measured) + GPU GEMM. f is THE hybrid lever (moves misses between legs). |
| 7 | `decode_sync` (cpu_executor.py:871-887) WAIT: `cuStreamWaitValue64(done>=1)` (cpu_moe_ext.cpp:659) | the round trip is FORCED (see dependency proof below); its 1.9-3.0 ms COST is mostly artifact. |
| 8 | H2D `out.copy_(io["y"])` + `torch.empty_like` per layer (decode_sync) | artifact: persistent per-(layer,bs) graph-stable output buffers kill both. |
| 9 | `gpu_routed + cpu_routed` merge add (moe.py:339) | FORCED (feeds block k+1). |

CPU-side machinery executed inside the WAIT window (all in cpu_moe_ext.cpp):
coordinator hot-polls `ready[]` (`coordinator_loop` :2286) -> `submit(t)` :2159
(mutex + `task_cv.notify_all()` broadcast to N workers) -> `run_task_body` :2114:
pass1 GEMV -> full-pool spin `barrier` -> prep -> pass2 -> `done_count` ->
`sync_cv.notify_all()` -> coordinator `sync()` :2231 (condvar wait) -> `done[slot]=1`
-> GPU front-end observes -> stream continues.

### What is architecturally FORCED

The routing dependency is decisive: block k+1's attention consumes block k's MERGED
output (`gpu_routed + cpu_routed`). Therefore `router_{k+1}`, the D2H submit, and the
CPU GEMV of layer k+1 cannot begin until layer k's CPU result is done and merged:

```
CPU_k -> merge_k -> attn/router_{k+1} -> split -> CPU_{k+1}   (strictly serial)
CPU_k  ||  fetch+GEMM_k                                       (the only concurrent pair)
```

Consequences:

- A cross-layer pipeline ("start CPU work for k+1 while the GPU finishes k") is
  INFEASIBLE - the input routing of k+1 does not exist yet. There is no slack window
  beyond the intra-layer one the current overlap already uses.
- "Batch the whole step's CPU work into ONE submit/wait" (bisect-notes phrasing) is
  NOT literally available for the same reason: layers 2..42's CPU inputs are unknown
  until their predecessors merge. The real lever is the COST of each per-layer round
  trip, not the COUNT.
- The measured overlap win is only ~15% (2.99 vs 2.57, ab-table.md) because the
  per-layer WAIT caps the window at one layer: theoretical exposure for the 3.5
  ms/layer leg over a 1.67 ms/layer GPU window is ~1.8 ms/layer, but measured exposure
  is 137.5/42 = 3.28 ms/layer - the round-trip transport (GPU-write -> coordinator
  visibility, condvar wake, barriers, sync_cv, done -> GPU WAIT wake-up) eats the
  overlap window. Fixing transport therefore also recovers overlap efficiency.

### Cost of the round trip (measured)

- flag memop handshake: 3.02 ms/layer (cap-6 row, CPU leg ~0 -> pure transport,
  ab-table rows 2-3).
- hostfunc handshake: 1.90 ms/layer (row-5 decomposition in bisect-notes) - the
  transport choice alone is worth ~1.1 ms/layer; the memop WAIT wake-up edge is the
  slower one on this driver.
- Inside the 1.9 ms: per-task protocol fat (1 condvar broadcast wake of N workers,
  1-2 full-pool spin barriers, 1 sync_cv wake) + two host-memory visibility edges
  (~30-50 us each per the in-code estimate) + worker dispatch. For hybrid-sized tasks
  (0.82 experts over a 6-14 worker pool) the GEMV itself is far smaller than the fat.

### Levers and floor

- L1 transport: fused single host entry (one `cudaLaunchHostFunc` whose callback runs
  submit+sync and records a CUDA event; stream waits via native `cudaStreamWaitEvent`),
  or the memop path with a tighter WAIT mechanism (spin-kernel on zero-copy flags:
  trades 1 SM for ~us wake granularity). Which one wins is an empirical question - the
  1.9 vs 3.0 asymmetry proves intuition is unreliable here; A/B at cap-6 first.
- L2 protocol fat: whole-expert worker assignment for hybrid-sized tasks (each worker
  runs gate->up->swiglu->down for its own experts; no cross-worker barrier is needed
  because no worker consumes another's partial), atomic done counter polled by the
  coordinator instead of condvars on the critical path, one packed D2H slab.
- L3 H2D elimination: persistent per-(layer,bs) output buffers (mirrors `_io`
  graph-stability); `decode_sync` returns a view, no `empty_like`, no copy.

Floor quantification (h = per-layer round-trip cost; 42 layers):

- today: h = 3.02 (flag) / 1.90 (hostfunc) -> +126.7 / +79.7 ms/step.
- halved (h ~ 0.95-1.5): -63..-77 ms/step.
- transport-only floor (h ~ 0.3-0.5): +12.6..+21 ms/step - only in this regime can
  hybrid possibly tie offload (section 4).
- What NO rework removes: the serial chain CPU_k -> merge_k -> router_{k+1} ->
  submit_{k+1} and its two memory-visibility edges + one host dispatch: ~0.1-0.3
  ms/layer best case, i.e. 4-13 ms/step of irreducible tax vs offload's zero.

## 2. Weighted split + per-pool fractions

Miss volume per step (fraction-overflow.md): pool 1 = 39 x 5.4044 = 210.8
expert-misses; pool 2 = 0.0045; pool 3 = 2 x 1.0284 = 2.06. Weights
210.8 : 0.0045 : 2.06 = 99.0% : 0.0% : 1.0%.

Today's even split (`resolve_pool_affinities`, cpu_executor.py:267-318; auto
divmod(20,3) -> 7/7/6 minus one coordinator each) gives the dominant pool 6 workers;
the CPU sampler shows 7.5/20 cores busy (decode-numbers.md (e)) - starved, not
saturated, 12+ cores idle.

CPU-leg scaling: the scalar gguf dots are COMPUTE-bound, not bandwidth-bound:
benched 11.3 GB/s / 20 threads = 0.57 GB/s/thread; in-situ 2.96 GB/s / 6 = 0.49
GB/s/thread (bisect-notes (2)) - consistent per-thread rate, so worker scaling is
~linear (LUT tables are L1-resident; 407 MB/step never approaches DRAM bandwidth).
Projection 6 -> 14-15 workers: 2.96 x (14.5/6) x 0.9 = 6.5-7.0 GB/s, bounded above by
the benched ratio 11.3 x 14/20 = 7.9 GB/s (the bisect's 8.5-9 assumes slightly better
efficiency than the in-situ 0.9 factor justifies). CPU leg: 407 MB / 6.5-7.9 GB/s =
52-63 ms/step (vs 137.5 today; net -80..-90 ms/step including the small extra PCIe
fetch the higher fraction adds).

Per-pool fetch fraction: f = pcie/(pcie+cpu). Pool 1 at 6.5-7.9 GB/s -> f = 0.83-0.86
(vs global 0.78 today); with the SIMD tier (section 3) f drops to ~0.63-0.70. The
minorities should run f = 1.0 (fetch everything, ~zero misses) with 1 worker each.
Implementation needs per-pool fractions, but benchbw stores ONE overlapped pair per
dtype - options: (a) scale the profiled CPU number by thread count analytically
(cheapest, defensible given linearity), (b) a ~0.5 s init-time probe per pool,
(c) benchbw rows per thread count. Sites: engine.py `_resolve_hybrid_fetch` :944-973
(already per-partition; feed it pool-aware BW), `resolve_pool_affinities` :267
(add a weights argument), call site engine.py:1015.

## 3. SIMD tier (AVX2 LUT)

Scalar cost 2.5-4 cyc/elem (task-06 spec; kernels verified at cpu_moe_ext.cpp
~:1304-1380: `q6_k_dot_scalar`, `iq4_xs_dot_scalar`, `iq3_xxs_dot_scalar` - per-element
scalar LUT loads + bf16 converts; iq3_xxs does two grid gathers + sign masking per
4 elements). The selector `select_ggufdot` :1404-1410 is the sole dispatch site; the
ISA-tier machinery (`pick_isa` :690, FREETOKEN_CPU_MOE_ISA) and SIMD precedents
(`select_nvi8dot` :756, dot_avx2/avx512 for nvfp4) already exist.

Per-format expectations (llama.cpp-class vec_dot experience):

- iq4_xs: 16-entry LUT via shuffles (no gather), 4-bit unpack -> 3-4x.
- q6_k: nibble + 2-bit unpack is pure arithmetic, vectorizes cleanly -> 2.5-3.5x.
- iq3_xxs: 256-entry grid gather (`_mm256_i32gather_epi32`, or swizzle into two
  128 x 16 B shuffle tables) + sign XOR -> 2-3x (gather-bound).

Blended CPU-leg gain x2.5-3.5 -> per-thread 1.2-2.0 GB/s; at 14-15 workers
17-28 GB/s (central ~22). Interaction with (2): the matched fraction FALLS as the CPU
leg speeds up (f = 40.1/(40.1+cpu): 0.84 scalar -> ~0.65 SIMD), shifting ~1 more
expert/layer back to PCIe. The two workstreams partially offset in the fraction
dimension (SIMD shrinks hybrid's fetch-saving prize) but the net is strongly positive
because the leg time collapses. benchbw must re-measure r (task-06 step 3): new
r = 40.1/22 ~= 1.8 - still below the 2.0 auto-pick threshold, so the benchbw headline
verdict for gguf stays "offload" even after all three workstreams.

## 4. Combined projection

Additive model anchored on the measured single-variable deltas (which reproduce
exactly: 126.7 + 137.5 = 264.2 = 334.4 - 70.2). Measured overlap behavior says the
CPU leg is nearly fully exposed (137.5 ms exposed vs ~77 theoretical), so rows below
keep the leg exposed; the best-case row instead uses the concurrent-legs model.

Leg inputs: scalar pool-1 leg 52-63 ms (B); with C (x2.5-3.5) 15-21 ms. h = 42-layer
handshake total.

| # | combination | step model | tok/s | verdict vs gate 14.3 |
|---|---|---|---|---|
| 1 | today (flag-sync) | 70.2 + 126.7 + 137.5 = 334.4 | 2.99 | measured FAIL |
| 2 | B only | 70.2 + 126.7 + 52..63 = 249..260 | 3.9-4.0 (4.3 w/ overlap credit) | matches bisect 4-5 model |
| 3 | B + A (h -> 1.0 ms/layer) | 70.2 + 42 + 52..63 = 164..175 | 5.7-6.1 | FAIL 2.3x |
| 4 | B + A + C | 70.2 + 42 + 15..21 = 127..133 | 7.5-7.9 | FAIL |
| 5 | B + A(h=0.5) + C | 70.2 + 21 + 15..21 = 106..112 | 8.9-9.4 | FAIL |
| 6 | best case: h=0.3, matched-f concurrent legs, cpu 22 GB/s | 42 x (1.11 + 0.22 + 0.3) = 68.5 | 14.6 | PASS, zero margin |
| 6b | best case, cpu 17 GB/s | 42 x (1.21 + 0.22 + 0.3) = 72.7 | 13.8 | FAIL by hair |

Best-case model: per layer = max(f x 68.6 MB / 40.1, (1-f) x 68.6 / cpu_bw) + 0.22 ms
(GEMM + fixed GPU ops) + h, with f = 40.1/(40.1+cpu_bw) so both legs finish together
(68.6 MB = 5.4044 x 12.7 MB miss bytes per layer).

Does the shared PCIe gather cancel between hybrid and offload? NO. Offload fetches
ALL 5.40 misses/layer; hybrid fetches f x 5.40 (4.58 today, ~4.5-4.75 under B/C).
Hybrid's entire theoretical prize is the (1-f) share moved to the CPU = only ~0.65-0.85
expert/layer = ~8-11 ms/step of PCIe time, because PCIe (40.1 GB/s) delivers experts
FASTER than even a SIMD CPU leg (17-28 GB/s). That prize is what pays for the
handshake tax (42 x h) and the leg slack - and it is too small: the gate (step <= 69.9
ms = the offload step) requires h <= ~0.3-0.5 ms/layer with everything else perfect.

Honest verdict on 14.3: REACHABLE ONLY AT THE EXTREME END of all three workstreams
simultaneously - h <= 0.3-0.5 (5-6x below today's best transport, unproven on this
driver; only ~3-6x above the raw ~80 us hostfunc dispatch edges), CPU leg >= ~17-22
GB/s (C must deliver >=3x AND B must deliver 14-15 workers), matched per-pool
fractions, AND the overlap window actually hiding the leg. Realistic landing zone with
mid-range outcomes: 8-11 tok/s. And even the best case (14.4-14.6) merely TIES the
14.25 offload control: hybrid frees no VRAM (delta = noise, decode-numbers.md (f)),
prefill is unaffected (263-264 tok/s), and the burst regime is where hybrid is
worst (ab-table.md burst FAIL). benchbw's verdict rule (hybrid iff cpu > 2x pcie =
80 GB/s) can never flip on this rig. The gate is technically reachable, economically
pointless.

## 5. Rework estimate per workstream

### M-A0 (prerequisite): round-trip microbench - S, ~1-2 days

Design: cap-6-style A/B harness sweeping round-trip mechanisms (flag memop /
hostfunc pair / fused hostfunc+event / spin-kernel wait) at ~0 experts; measures the
achievable h floor on THIS driver. Files: a bench script + optionally a small
cpu_moe_ext.cpp addition (event-record callback). No product risk. DECISION GATE:
proceed with A only if the measured floor is <= ~0.5 ms/layer; otherwise the whole
A workstream cannot pay for itself and the hybrid gate is dead.

### A. Handshake cost reduction - M/L, ~600-900 LOC (py + cpp), 2-3 weeks incl. hardware A/B

Design (one paragraph): make the per-layer GPU<->CPU round trip cost ~its transport
floor instead of its current protocol fat. Replace the two-memop doorbell + condvar
chain with either (i) a single fused host entry - one cudaLaunchHostFunc whose
callback runs submit+sync and records a CUDA event that the stream waits on natively
(halves the host-func pair, replaces the second dispatch with a cheap wait) - or (ii)
the tuned memop path per M-A0. Pack x/ids/w into ONE pinned D2H slab (1 copy instead
of 3). Allocate persistent per-(layer,bs) output buffers at task creation
(graph-stable like _io) so decode_sync returns a view - no empty_like, no H2D copy.
Add a coordinator fast path for hybrid-sized tasks: whole-expert worker assignment
(gate->up->swiglu->down inside one worker per expert; no cross-worker barrier since
no partial is shared), atomic done counter polled by the coordinator, no condvars on
the critical path.

Affected sites: cpu_executor.py `decode_submit` :825-869, `decode_sync` :871-887,
`_task_for` :787-808, `__init__` :396-526 (flag/coordinator setup, buffer persistence);
cpu_moe_ext.cpp `submit` :2159, `sync` :2231, `worker_loop` :2136-2157,
`run_task_body` :2114-2133, `coordinator_loop` :2286-2360, memops :652-663;
layers/moe.py `_decode_hybrid` :292-339 (merge/output path only).

Risk: HIGH. This machinery is SHARED: nvfp4 hybrid, ds_fp4 (the `_gpu_prequant`
contract and DSV4's 75-layer decode depend on flag-sync today), q4_0,
`--moe-cpu-layers`, and CUDA-graph capture (protocol immediates baked at capture,
task pointers must stay graph-stable, the watchdog's 10 s dead-slot heuristic assumes
latency patterns). A regression here damages working strategies, not just gguf.

Gain (from measured data): -80..-100 ms/step if h lands 0.8-1.0 (hostfunc-class
1.9 -> target); -105..-120 if the M-A0 floor is 0.3-0.5. Worth +2-4 tok/s on top of
B+C; alone (with today's starved leg) +1-2.

Tests/acceptance: tests/moe/test_cpu_moe*.py + test_hybrid_fetch.py green; NEW
graph-capture replays + watchdog-timing tests; ds_fp4/nvfp4 parity batteries; full
hardware A/B vs the offload control on the real file; DSV4 spot-check (no decode
regression on the flag-sync consumer).

### B. Weighted pool split + per-pool fractions - S, ~150-250 LOC, 2-4 days

Design: extend `resolve_pool_affinities` (cpu_executor.py:267-318) with per-pool
weights; the engine passes layer-count weights (39:1:2 for this file, available at
partition time from bank group sizes) -> ~15/1/1 workers of 17-20 physical cores
(one coordinator core carved per pool). Per-pool fetch fraction: scale the profiled
cpu BW by thread count (f1 = 40.1/(40.1+11.3 x 15/20) = 0.83; minorities f = 1.0
with 1 worker each) - `_resolve_hybrid_fetch` (engine.py:944-973) is already called
per partition; hand it the pool-aware bandwidth.

Affected sites: cpu_executor.py resolve_pool_affinities :267-318; engine.py:1015
call site + _resolve_hybrid_fetch :944-973; optionally benchbw.py profile rows.

Risk: LOW. Offload-only path never constructs pools (len(caches)==1 branch
untouched); nvfp4 single-partition unchanged; affinity clamping already guards
oversubscription. Worst case is a bad split logged and visible in the pool log line.

Gain (measured-anchored): CPU leg 137.5 -> 52-63 ms/step -> 2.99 -> ~3.9-4.3 tok/s
alone. The plan explicitly bars shipping B alone (task-06 re-scope note), but it is
the risk-free first commit and the enabler for A/C projections.

Tests: tests/moe/test_cpu_moe.py (affinity/threads), tests/moe/test_hybrid_fetch.py
(per-pool fraction), tests/models/test_gguf_expert_banks.py partition suite.

### C. SIMD AVX2 LUT tier - M, ~500-800 LOC cpp, 1.5-2 weeks

Design: AVX2 dot per gguf format behind `select_ggufdot` (cpu_moe_ext.cpp:1404-1410),
following the `select_nvi8dot` :756 precedent and the existing ISA-tier gating
(`pick_isa` :690): iq4_xs shuffle-LUT, q6_k arithmetic nibble unpack, iq3_xxs
gather/swizzle-LUT + sign XOR; fp32 FMA accumulate; scalar tier stays as the
reference/fallback. No python hot-path changes (task-06 constraint).

Affected sites: cpu_moe_ext.cpp only (new kernels + selector entries).

Risk: LOW-MED and CONTAINED: numerics (fp32 reassociation vs the dfloat tolerance
8 x 2^-11 x factor x |d| + 1e-3 - the parity battery decides); zero contact with the
offload/NVFP4 GPU paths; gguf CPU executor only. Expected-value note: C is only
useful if hybrid continues (the gguf executor runs solely in hybrid/cpu strategies),
so it is contingent on M-A0 + A.

Gain (measured-anchored): leg 52-63 -> 15-21 ms/step (with B) = +2-3 tok/s on top of
B+A; also lowers the matched fraction (offsets part of its own win; net positive).

Tests: tests/moe/test_cpu_moe_gguf_iq.py (analytic uniform fixtures),
tests/kernels/test_gguf_quant.py parity, FREETOKEN_CPU_MOE_ISA=avx2/scalar A/B,
benchbw refresh -> new r and fraction recorded in PLAN.md.

### Recommended execution order and combined estimate

1. M-A0 microbench (1-2 d) - decides whether A can pay at all.
2. B weighted split + per-pool fractions (2-4 d) - safe, landed-value, needed for
   every later projection.
3. IF M-A0 floor <= ~0.5 ms/layer: A handshake rework (2-3 wk) - the decisive lever
   (-80..-120 ms/step) but the high-risk one.
4. C SIMD tier (1.5-2 wk) - mechanical, contained, amplifies B.
5. Hardware gate re-run ONLY after 3+4; expect 8-11 tok/s realistic, 12.8-14.6 best
   case.

Combined engineering estimate: ~4-5 weeks. Honest bottom line: the 14.3 gate is
reachable only in the extreme best case of A and C simultaneously, with zero margin,
and even then hybrid at best TIES the offload control it is being compared against.
Offload already delivers 14.25 tok/s on this file (campaign anchor 13.8-15.6). The
data-supported recommendation is: keep offload as the served strategy for glm5next
GGUF (benchbw's verdict is empirically correct), run M-A0 + B because they are cheap
and de-risking, and treat A+C as a research investment only if hybrid is wanted for
reasons this campaign does not currently have (it frees no VRAM, does not help
prefill, and loses the burst regime).
