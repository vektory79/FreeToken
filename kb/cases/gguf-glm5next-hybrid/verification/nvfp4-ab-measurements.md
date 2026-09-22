# NVFP4 hybrid vs offload A/B measurement + gap explanation (2026-09-15)

Hardware A/B on RTX 5090 (32 GiB), 20 physical cores, DDR5 188 GB. Repo HEAD eb7de4c
(plus a memory-only commit 1b913ba on top; code tree clean). No code was changed.
Measurement harness = Task 05 recipe reused verbatim (/tmp/ft_nvfp4_runner.sh +
/tmp/ft_nvfp4_client.py + /tmp/ft_serve_driver.py collect-stats hook): FAILPAT
('OutOfMemoryError|AssertionError|Backend worker is gone|backend worker .* exited')
grepped every 5 s during boot + 2 s watchdog during runs, /ready poll, model resolved
from /v1/models, EXIT trap kills the whole tree. Both boots started from an idle GPU
(1353/1357 MiB used, zero compute processes), port 18081 free, and exited gracefully
(stats dump on SIGTERM). Both runs: client failed=[], no watchdog hits.

## 1. Boot + resolved plans (STEPS 1 and 3)

The user's EXACT 1M command BOOTED in BOTH modes today - the 2026-09-14 fail-fast did
NOT reproduce. Single variable between the runs: --moe-strategy.

| mode | boot | moe_cache_size | KV pages | KV tokens | KV | free after init | fetch fraction | CPU executor |
|---|---|---|---|---|---|---|---|---|
| hybrid | 85 s | 341 | 16432 | 1,051,648 | 3.84 GiB | 2.81 GiB | 24.9% of expert misses | 1 pool, 16 threads, avx2+vnni(nvfp4-w4a8) |
| offload | 55 s | 341 | 16417 | 1,050,688 | 3.83 GiB | 2.80 GiB | n/a | none (target=gpu) |

Exact log lines (hybrid):
- `--moe-cache-auto resolved moe_cache_size=341 num_pages=16432 (prefill_overlap=False)`
- `Allocating 1051648 tokens for KV cache, K + V = 3.84 GiB`
- `Free memory after initialization: 2.81 GiB`
- `--moe-hybrid-max-fetch auto: fetching 24.9% of each decode step's expert misses over PCIe (benched PCIe/CPU bandwidth ratio), the rest on the CPU`
- `CPU MoE executor ready: threads=16 (pinned to cores 0..23) isa=avx2+vnni(nvfp4-w4a8) fmt=nvfp4 H=4096 I=2048 experts=288 layers=42 top_k=8 act=swiglu_clamp max_tokens=1`
- `expert banks: FTW fast path`

Exact log lines (offload):
- `--moe-cache-auto resolved moe_cache_size=341 num_pages=16417 (prefill_overlap=False)`
- `Allocating 1050688 tokens for KV cache, K + V = 3.83 GiB`
- `Free memory after initialization: 2.80 GiB`

**Finding A - the VRAM-competition hypothesis is REFUTED for this config:** both
strategies resolve to the SAME plan (341 slots). The 1M KV reserve does not starve the
offload expert cache relative to hybrid; each mode gets an identical slot budget and
 spends it identically. (341 >= 336 = the 42x8 serving working set, barely.)
The 09-14 VRAM shift is still visible vs the 09-12 baseline: 341 slots today vs 373
then (-32 slots), and the 1M command that failed once on 09-14 boots again today.

## 2. Decode measurements (STEP 2 / STEP 3, same client, same session)

Client method identical to Task 05: streamed deltas carrying content OR
reasoning_content, rate = deltas / (last-first), TTFT excluded; 64k decode re-sends the
same 64,512-token prompt (server log confirms `#new-token: 12, #cached-token: 64512`);
prefill metric = server `input throughput` per 4096-chunk (first chunk = warmup).

| metric | hybrid | offload | hybrid advantage |
|---|---|---|---|
| short-context decode (2k, 255 tokens) | **14.46 tok/s** | **10.12 tok/s** | 1.43x |
| decode @64k fill (767 tokens) | **13.77 tok/s** | **9.91 tok/s** | 1.39x |
| ms/step @64k | 72.6 | 100.9 | -28 ms |
| ms per MoE layer | 1.73 | 2.40 | - |
| burst (6x cold 2k, 128 tok) mean/min | 14.20 / 13.82 | 10.30 / 10.19 | 1.38x |
| TTFT on cached 64k prefix | 4.22 s | 4.19 s | = |
| prefill steady state (4096 chunks) | ~507 tok/s | ~507 tok/s | = (prefill unaffected by strategy) |
| fill wall (64,524 tok) | 128.35 s | 128.32 s | = |

Server `gen throughput` log lines @64k: hybrid fluctuates 12.7-16.9 around 13.8;
offload 8.97-10.91 around 10.0.

Context vs history: user's "hybrid ~15" = the 2026-09-12 number (15.13 @64k, 373
slots); today's 13.77 @ 341 slots is the same regime after the 09-14 VRAM shift
(-9% decode from -8.6% slots). User's "offload ~9" reproduces as 9.9-10.3.

## 3. Per-step decomposition (collect-stats dumps + benchbw profile)

Stats dumps (1785 steps across all decode phases per run; both modes see the SAME
routing statistics): hybrid missing_per_layer=5.76 (miss_rate 72.0%), fetched 1.06/layer
(18.4% of misses), cpu 4.70/layer; offload missing_per_layer=5.70 (miss_rate 71.2%),
all misses served by H2D fetch + GPU GEMM (target=gpu).

Expert slot = one (layer, expert) pair = 166.22 GiB / (42x288) = 14.42 MB.

| per decode step | hybrid | offload |
|---|---|---|
| experts computed on GPU (hits + fetched) | 3.30/layer (8 - 4.70) | 8/layer (all, after fetching 5.70 misses) |
| PCIe H2D traffic | 1.06/layer x 42 x 14.42 MB = **0.64 GiB/step** | 5.70/layer x 42 x 14.42 MB = **3.36 GiB/step** |
| CPU GEMV traffic (DDR5) | 4.70/layer x 42 x 14.42 MB = **2.85 GiB/step** | 0 |
| required PCIe bandwidth | 0.64 GiB / 72.6 ms = 9.0 GB/s | 3.36 GiB / 100.9 ms = 34.2 GB/s |
| required CPU-leg bandwidth | 2.85 GiB / 72.6 ms = 40.2 GB/s | - |
| CPU cores busy during 64k decode (1 Hz /proc/stat sampler) | **19.9 of 20** | 3-4 of 20 |

Cached benchbw profile (per-GPU json in the machine-local benchbw cache, not mirrored;
GPU-ec07f396-... - the two legs:

| format | cpu_moe standalone | pcie_gather standalone | cpu OVERLAP | pcie OVERLAP | planner fraction = pcie_ov/(pcie_ov+cpu_ov) |
|---|---|---|---|---|---|
| nvfp4 (avx2+vnni W4A8) | 70.97 GB/s | 42.27 GB/s | **55.62 GB/s** | 18.46 GB/s | **24.9%** |
| gguf iq/qk (scalar LUT) | 11.65 GB/s | 43.65 GB/s | **11.31 GB/s** | 40.14 GB/s | **78.0%** |

Consistency checks:
- Offload is PCIe-bound: 3.36 GiB/step at the benched 42.3 GB/s standalone gather =
  81 ms of pure copy; measured step 100.9 ms (copy + per-layer GEMM serialization).
  34.2 GB/s effective is the gather+GEMM interleaving.
- Hybrid fits everything under 72.6 ms: CPU leg 40.2 GB/s < benched overlapped 55.6
  (not saturated), PCIe 9.0 GB/s << 42.3, and the dense/GDN/attention GPU work overlaps
  both. Nothing serializes; the machine is saturated (19.9/20 cores) because the CPU
  leg is actually carrying 59% of the expert volume (2.85 of 4.84 GiB/step).

## 4. Root cause of hybrid >> offload on NVFP4 (the 15-vs-9 gap)

1. Offload has no CPU leg and no overlap: EVERY missed (layer, expert) must cross PCIe
   before the GPU can compute it. At this context length ~71% of the 336 routed pairs
   miss per step -> 3.36 GiB/step -> the step time floor is the PCIe copy (~81-100 ms)
   -> ~10 tok/s. Adding slots cannot fix this (341 slots already hold the working set;
   the misses are routing turnover, not capacity - a bigger pool would still see ~70%
   per-step miss because 336 of 12,096 pairs are requested per step).
2. Hybrid splits the misses by BANDWIDTH MATCH: fetch pcie_ov/(pcie_ov+cpu_ov) = 24.9%
   over PCIe, compute 75.1% on the CPU with the W4A8 integer kernel. The NVFP4 CPU leg
   (55.6 GB/s overlapped) is 3x FASTER than its own overlapped PCIe share (18.5), so
   the planner pushes most work onto DDR5, which is not a bottleneck at 40 GB/s.
   Step = max(GPU dense work, 0.64 GiB PCIe, 2.85 GiB CPU) + fixed sync = 72.6 ms.
3. So the entire 28 ms/step gap is "which memory carries the misses": DDR5 at 40-55
   GB/s (hybrid) vs PCIe at 34-42 GB/s (offload) for 3.36 GiB/step of traffic.
   Hybrid converts a PCIe-bound step into a DDR5-bound leg that hides under the GPU.

## 5. The handshake contradiction RESOLVED (1.59-1.73 vs claimed 1.9-3.0 ms/layer)

GGUF Task 05 claim: hybrid pays a per-layer GPU<->CPU handshake floor of ~1.9-3.0 ms
x 42 layers (cap-6 run: 196.9 ms step with CPU leg ~0; hostfunc: 287.4 ms), which
"ALONE exceeds offload's whole 70 ms step". NVFP4 today runs 72.6 ms/step = 1.73
ms/layer ALL-IN - below that claimed floor - while ALSO fetching 1.06 experts/layer
over PCIe and running a real CPU leg. Contradiction.

Resolution: **the "handshake floor" is not fixed per layer; it scales with the per-layer
fetched BYTES, and that volume is endogenous to the CPU-leg speed.**
- The per-layer hybrid choreography (layers/moe.py::_decode_hybrid, identical for both
  formats): device-side routing split -> executor.decode_submit (host node) ->
  cache.copy_missing (PCIe fetch) -> GPU GEMM on hits+fetched -> executor.decode_sync
  (host wait) -> partial merge. Fixed sync cost is small (~0.6-1.0 ms/layer).
- GGUF cap-6 fetched min(6, miss) ~ 5.4-6 experts/layer = 67-75+ MB/layer; at ~30-40
  GB/s effective H2D that is 1.7-2.5 ms/layer of copy - which the bisect counted as
  "handshake". GGUF auto f=0.78 (forced by its 11.3 GB/s scalar leg) fetches the same
  ~78% of misses -> same 3x bytes. The bisect mis-attributed PCIe fetch volume to a
  fixed sync floor.
- NVFP4 fetches only 1.06/layer = 15.3 MB/layer -> ~0.4-0.8 ms/layer of copy; add the
  ~0.6-1.0 ms fixed sync and GPU dense work and you get exactly the measured 1.73
  ms/layer. The NVFP4 CPU leg also finishes INSIDE the per-layer GPU window, so the
  WAIT node never extends the step.
- Two GGUF-only structural multipliers made its floor look even worse: (a) 3
  per-signature partitions with resolve_pool_affinities even split -> dominant pool
  gets 6/16 threads (measured 2.96 GB/s vs 11.3 benched; CPU sampler 7.5/20 cores
  busy - starved, not saturated); (b) per-partition submit/sync per step. NVFP4 has
  ONE cache -> ONE 16-thread pool (stats dump: pools=[1]) -> machine saturated.

## 6. Code comparison (STEP 4b): concrete deltas, GGUF path vs NVFP4 path

Files: kernel/csrc/cpu_moe/cpu_moe_ext.cpp, moe/cpu_executor.py,
moe/bench_profile.py, engine/engine.py, layers/moe.py.

1. CPU GEMV kernel tier (THE delta): NVFP4 takes the integer W4A8 path -
   `use_vnni = (weight_format == WF_NVFP4) && (nvi8dot != nullptr)`; activations
   quantized once per step to int8 group-16 (`quant_i8_pg16`, cpu_moe_ext.cpp:1693,
   called at :2078 for the intermediate row and :2205 for the input row) and consumed
   by AVX2+VNNI int8 dots (isa tag `avx2+vnni(nvfp4-w4a8)`); 70.97 standalone / 55.62
   overlapped GB/s. GGUF IQ3_XXS/IQ4_XS/Q6_K take the fp32 LUT path:
   `select_ggufdot` returns ONLY scalar kernels (`iq3_xxs_dot_scalar` etc., :1404) -
   the comment there says the AVX2/AVX-512 LUT-gather tiers "slot in here as a
   follow-up". 11.65/11.31 GB/s = 4.8x slower leg. This is a deliberately missing tier,
   not a hardware limit.
2. Activation pre-processing: NVFP4 deinterleaves bf16->fp32 even/odd and int8-quantizes
   ONCE per step in submit() and reuses across all experts/rows (needs_di branch);
   GGUF reads raw bf16 activations per row inside the scalar dots - no prequant to
   amortize. A vectorized GGUF tier needs the same one-shot prep added.
3. Bank materialization/layout: NVFP4 has a PACKED gate_up bank (single fused pass +
   one dot per intermediate unit). GGUF has three SEPARATE per-role banks with per-role
   tables and per-role dot fns (3x tbl_at resolution, 2 dots per unit, K-quant 256-wide
   blocks; the gguf file has no packed gate_up). Minor (~pointer chases), not the 4.8x.
4. Pool topology: NVFP4 = one uniform format -> ONE OffloadMoeCache + one executor
   pool (all 16 threads; resolve_pool_affinities trivial). GGUF = per-signature
   partitions -> N pools with an EVEN thread split (6/5/5 for 3 partitions) and
   per-pool executors. GGUF-only structural cost; not fixable by "porting" anything -
   needs weighted affinities (Task 06 re-scope).
5. Fetch-fraction INPUT: the fraction model is shared (bench_profile.
   load_hybrid_fetch_fraction: f = pcie_ov/(pcie_ov+cpu_ov), engine.py:946-974). What
   differs is the benched leg feeding it: nvfp4 55.62 -> f=24.9%; gguf 11.31 -> f=78.0%.
   The GGUF profile bakes the slow scalar leg into the operating point, which is what
   triples the per-layer fetch bytes (and with it the apparent "handshake").
6. NOT different (checked, same for both): handshake architecture and graph capture
   path (submit/sync are capture-safe host nodes in both), pinned host banks,
   FREETOKEN_HYBRID_OVERLAP knob, prefill movement; --disable-moe-prefill-overlap
   affects prefill only and prefill measured IDENTICAL (~507 tok/s) in both modes.

## 7. Synthesis: what GGUF hybrid would need to behave like NVFP4 hybrid

Ordered by leverage:
1. Vectorized IQ/QK CPU GEMV tier (AVX2/AVX-512 LUT gathers mirroring the nvfp4
   decode; the dispatch site already exists in select_ggufdot). Target >= 30-40 GB/s.
   This also FIXES THE FRACTION: a re-profiled benchbw would drop f from 78% toward
   the bandwidth match, cutting per-layer fetch bytes ~3x and with them the "handshake
   floor". (Depends on the ggml-grade CPU kernel study: research/ggml-cpu-kernel-study.md
   is NOT present in the repo at HEAD - external dependency, noted as such.)
2. Weighted pool affinities / single dominant pool + per-pool fractions (the re-scoped
   Task 06): remove the 6/16-thread starvation of the dominant partition.
3. Modeled ceiling with both (Task 05 numbers): 42 x (1.46 gather + 0.6-1.3 sync+cpu)
   ~ 87-110 ms -> 9-11.5 tok/s with the scalar leg batched; with a 40 GB/s SIMD leg
   carrying 75% of ~5.7 misses/layer (~2.3 GiB -> 56 ms) + 1.4/layer fetch (17 ms)
   -> ~70 ms -> ~14 tok/s = offload parity. Beating offload structurally the way
   NVFP4 does additionally requires the GGUF expert bytes (10.9-15.8 MB/projection
   triple) to stop costing more PCIe/CPU traffic than NVFP4's 14.4 MB pair - i.e.
   GGUF hybrid can at best reach offload parity on this rig, which is exactly what
   benchbw's "offload" verdict (ratio 0.27 < 2.0) predicted and Task 05 measured.

## 8. Artifacts (exact outputs)

- /tmp/nvfp4_hybrid.log, /tmp/nvfp4_hybrid_result.json, /tmp/nvfp4_hybrid_cpu.log,
  /tmp/nvfp4_hybrid_decode_stats.json
- /tmp/nvfp4_offload.log, /tmp/nvfp4_offload_result.json, /tmp/nvfp4_offload_cpu.log,
  /tmp/nvfp4_offload_decode_stats.json
- Harness: /tmp/ft_nvfp4_runner.sh, /tmp/ft_nvfp4_client.py (Task 05's
  /tmp/ft_serve_driver.py reused unchanged)
- Task 05 GGUF references: this folder's bisect-notes.md, decode-numbers.md, ab-table.md

Exact serve commands (identical except --moe-strategy):
```
ft serve --port 18081 --host 127.0.0.1 --moe-strategy hybrid  --moe-cpu-threads 16 --kv-reserve-tokens 1048576 --kv-cache-dtype nvfp4 --model-path /media/ai/models/RedHatAI/GLM-5.3-Flash-NVFP4-FTW/ --max-running-requests 1 --disable-moe-prefill-overlap --memory-ratio 0.89 --max-prefill-length 4096
ft serve --port 18081 --host 127.0.0.1 --moe-strategy offload --moe-cpu-threads 16 --kv-reserve-tokens 1048576 --kv-cache-dtype nvfp4 --model-path /media/ai/models/RedHatAI/GLM-5.3-Flash-NVFP4-FTW/ --max-running-requests 1 --disable-moe-prefill-overlap --memory-ratio 0.89 --max-prefill-length 4096
```
(botted via the driver for collect-stats; the driver changes no behavior except
moe_collect_stats=True and a stats dump on exit.)
