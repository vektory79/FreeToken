# v3a hardware A/B sweep: grouped MoE m-tile 4/8/16/32 (GGUF glm5next, RTX 5090)

Date: 2026-09-19 (03:17-04:08 local). Branch vektory79 @ 7ed25e0, uncommitted
v3a changeset (5 code files: gguf_kernel.cu, moe.cuh, moe.py, 2 test files).
Single RTX 5090, port 18801, winner flags from the v2/tuning campaigns. One
boot per tile, strictly serialized and reaped, trap-on-EXIT runners with
FAILLPAT watchdog. NO commits, NO push. Machine-readable twin:
`v3a-ab-results.json`.

## Method

Boot command (effective; measure.py BASE_ARGS + last-wins extras):

```
.venv/bin/ft serve --port 18801 --host 127.0.0.1 \
  --model-path /media/ai/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL.gguf \
  --moe-cache-auto --kv-reserve-tokens 500000 --kv-cache-dtype fp8 \
  --memory-ratio 0.85 --max-prefill-length 8191 --moe-strategy hybrid \
  --moe-cpu-threads 16 --max-running-requests 1
```

Sole variable: env `FREETOKEN_GGUF_MOE_MTILE` (per-call getenv in
`gguf_kernel.cu ft_gguf_mtile`; unset/"4"=4). Fill = the 65,585-token campaign
filler (8x8128 + 561 tail) then a same-prompt decode re-send (mode `full` of
the patched [measure.py](../harness/serve-measure/measure.py)). Prefill metric = median
of the full-chunk "input throughput (token/s):" lines EXCLUDING chunk 1 (warmup)
and the last full chunk (T43 tail artifact) = c2..c7. Decode = streamed re-send,
radix HIT expected (#cached-token: 65536), first SSE frame skipped.

## Validity gate (T44) - PASS

Baseline boot (no env, tile 4): **332.72 tok/s** (c2..c7: 334.86 333.35 332.86
332.58 332.21 332.15) vs the committed v2 class 337.61 = **-1.45%**, inside the
~3% band [327.48, 347.74]. Chunk-1 warmup 121.22; bogus last-full line 1733.5
and tail 554.82 excluded per T43. Decode 14.21 steady (class 13.5-14.7), radix
HIT 65536. No JIT rebuild lines (disk cache warm: freetoken_gguf_kernels.so
newer than all sources). Interpretable A/B unlocked.

## A/B table (65,585-token fill @8128 chunks)

| tile | env | prefill median | delta vs t4 | chunk time | decode steady | boot ready | peak VRAM |
|------|-----|---------------:|------------:|-----------:|--------------:|-----------:|----------:|
| 4 (base) | unset | 332.72 | +0.0% | 24.43 s | 14.21 | 114.2 s | 32095 MiB |
| 8 | MTILE=8 | 436.11 | +31.1% | 18.64 s | 14.54 | 110.3 s | 32020 MiB |
| 16 | MTILE=16 | 512.48 | +54.0% | 15.86 s | 14.74 | 142.3 s | 32080 MiB |
| 32 | MTILE=32 | **556.43** | **+67.2%** | 14.61 s | 14.07 | 117.7 s | 32080 MiB |

- vs the committed v2 class (337.61): tile 16 = +51.8%, tile 32 = +64.8%.
- Decode unchanged within the daytime noise band (14.07-14.74 vs 13.5-14.7
  class): the knob is prefill-only, as designed (decode stays moe_vec in CUDA
  graphs).
- Peak VRAM flat (32020-32095 MiB); no slot-floor / OOM events; watchdog never
  fired; every radix re-send HIT exactly 65536.

## MoE-term scaling: staging model half-confirmed, ALU/LDS floor binds

e2e subtraction model (rest tile-invariant): rest = 12.85 s (dense q8_0 GEMM
6.04 s + fetch copies 3.15 s + attention/GDN ~1.45 s + misc), MoE term at
tile 4 = 11.58 s. Measured MoE-term ratio vs the pure 1/tile traffic model:

| tile | pure 1/tile | measured e2e | nsys per-launch medians |
|------|------------:|-------------:|-------------------------|
| 8    | 0.500 | 0.500 | - |
| 16   | 0.250 | 0.260 | iq4_xs 2.64x, iq3_xxs 2.50x, q6_K 1.95x |
| 32   | 0.125 | 0.152 | (no nsys at 32; e2e only) |

nsys cross-check (tile-16 capture vs the v2 tile-4 capture `v2probe2.sqlite`,
same fill/flags, 9 chunk-equivalents each): grouped MoE sum 12.62 -> 5.01
s/chunk = 2.52x for a 15.8x traffic-model cut. The kernel is therefore already
substantially **ALU/LDS/serialization-bound** (step0-source-read.md section 3
regime: per-MAC vec_dot work + 64 k-step syncs dominate), while SMEM/occupancy
do NOT bind (CTAs shrank cleanly, no OOM, no spill-driven cliff) and padding
does NOT dominate.

Where it saturates: the marginal e2e gain halves per doubling (+31.1% ->
+17.5% -> +8.6%). At tile 32 the MoE term was estimated at ~1.8 s (e2e fit;
REFUTED 2026-09-19 by direct nsys: 4.37 s - see CORRECTION at the end of
this file); a hypothetical tile 64 would recover at most ~+3-4% e2e while padding
waste grows to ~14% of pair slots and q6_K occupancy keeps dropping
(10/10/9/7 blocks/SM at 4/8/16/32). **Practical saturation point: tile 32.**

## Liveness (T46, nsys, tile 16) - PASS on all five assertions

`report1.nsys-rep` / `v3aprobe.sqlite` (387,087 kernel records), recipe D09
(launch + start-after-ready + --cuda-graph-trace=node), marker line fired
exactly once:

1. Grouped MMQ names per format present in the prefill span: `moe_iq3_xxs`
   (738), `moe_iq4_xs` (369), `moe_q6_K` (27) over 9 chunk-equivalents - no
   foreign moe_* names (moe_q8_0 correctly absent: not in this model).
2. ZERO `moe_vec*` inside the grouped-kernel span (the 378-vec post-prefill
   burst = the prefill request's own 3 sampled tokens, v2 explained
   non-signal, sits after the last grouped launch).
3. Decode phase: 7,938 moe_vec = exactly 42 layers x 63 steps x 3 projections
   inside CUDA-graph replays (+ that phase's own 126 grouped / 42 align from
   the 49-token request prefill).
4. Launch-count math exact: 1,134 grouped = 126/chunk-eq, 378 moe_align =
   42/chunk-eq - unchanged from v2 (kernel names and counts are tile-agnostic
   by design).
5. Per-launch CTAs shrink 3.8x at tile 16 vs the tile-4 baseline capture
   (iq3_xxs 1,054,208 -> 277,376; iq4_xs/q6_K 2,108,416 -> 554,752; 4.0x minus
   the tile-padding growth of numel) - grid.y = numel/tile confirms the knob
   is LIVE in the serving path.

## Quality battery (24 prompts) - PASS, strongest possible result

Base boot (no env) vs winner boot (MTILE=32), identical prompt order, winner
flags, PYTHONPATH sitecustomize marker fired in both, boot plans identical
(kv_alloc 500160 / free 4.05 GiB / fetch 30.7% / same pools). Battery rc=0,
failed=0 in both stages.

**24/24 outputs BITWISE IDENTICAL between tile 4 and tile 32** (div min/med/max
272/694/856 = the prompt lengths themselves; delta +0 on every pair) - the
tile-invariant reduction order holds end-to-end in serving. All 4 digit prompts
correct in both stages (80 km/h, Friday, 6 apples finish=stop, 1024). 0 task
flips, 0 on-topic concerns (outputs byte-equal to the baseline-stage outputs).

## v3b trigger assessment - does NOT fire

TASK.md conditions: v3b only if v3a saturates below the ceiling (SMEM/occupancy
bound) or tail padding dominates. Measured: SMEM/occupancy did NOT bind (clean
CTA shrink, flat VRAM, no fail-fast) and padding does NOT dominate (7.1%
pair-slot waste at tile 32 shows up as the +22% flatness vs the pure 1/tile
model, not a cliff). The binding term is the schedule-independent per-MAC
ALU/LDS + k-step-serialization floor, which a v3b weight-stationary reschedule
would NOT remove (same vec_dot work). Recoverable MoE headroom at tile 32 is
~1-2 s of a 14.61 s chunk (~7-13% e2e) against a new persistent-kernel + grid
redesign; the chunk is now ~78-88% "rest" (dense q8_0 GEMM 6.04 s, fetch copies
3.15 s). **Recommendation: stop at v3a/tile 32; the next real levers per the
Step-0 ranking are the dense q8_0 GEMM and the fetch copies.**

## Verdict

- Knob FREETOKEN_GGUF_MOE_MTILE: live, correct, and worth up to **+67.2%**
  prefill @8128 (332.72 -> 556.43 tok/s) with decode and quality unchanged
  (bitwise). Winner tile **32**; tile 16 is the conservative pick (+54.0%, less
  padding/occupancy margin). 8191-chunk @0.85 boots clean at every tile.
- Suggested default for GGUF serving: export FREETOKEN_GGUF_MOE_MTILE=32
  (or flip the kernel default after user review). Commit subject stays ready:
  `feat(kernels): selectable grouped gguf moe m-tile (8/16/32) via
  FREETOKEN_GGUF_MOE_MTILE`. Commit USER-GATED; nothing committed, nothing
  pushed.

## Hygiene log

- Preflight: no wave processes, VRAM 1472-1547 MiB, 5 sem.mp-, no CC/CXX env,
  git tree = exactly the 5 v3a files (+ .veai/memory churn).
- Runs strictly serial and reaped (one measure.py/battery stage at a time);
  census clean before/between/after every boot; SIGTERM->30 s->SIGKILL armed in
  every runner; FAILPAT watchdogs never fired; no backend deaths; semaphores 5
  throughout (zero leaks); postflight: VRAM 1472 MiB, no processes, no stray
  nsys reports outside .tasks/mmq-v3-stationary-moe/.
- JIT: zero rebuild lines in all 7 boots (disk cache warm from the v3a test
  waves).

## Deviations / caveats

1. nsys liveness ran on tile 16 only (per brief). Tile-4 per-format and CTA
   baselines come from the v2 campaign capture `v2probe2.sqlite` (not mirrored;
   same 65,585 fill, same winner flags, grouped tile 4)
   instead of a fresh tile-4 nsys boot.
2. The analyzer's first A1/A2 pass was over-strict (demanded moe_q8_0, absent
   in this model; phase-split merged the post-prefill 378-vec burst into
   phase 0). Fixed to model-aware + span-based checks; all final assertions
   PASS (v3a_probe_liveness.json).
3. Optional ptxas register/spill capture per format per tile: SKIPPED
   (non-gating; reconstructing the exact torch build flag set was not worth
   the risk of a wrong-flags compile; q6_K's occupancy trend is already
   visible in the timing data).
4. Battery "win" stage = tile 32 (the prefill-median winner).
5. Baseline -1.45% vs the committed class is session noise, inside the gate.

## Artifacts (run in the git-ignored mmq-v3-stationary-moe worktree; kb mirrors linked)

- [v3a-ab-results.json](../baselines/mmq-v3/v3a-ab-results.json) - machine-readable sweep record
- `v3a_v3a_t{4,8,16,32}.out` stage outputs and boot logs: not mirrored (distilled away)
- [v3a_stage_raw.json](../baselines/mmq-v3/v3a_stage_raw.json) - parsed medians/decode per tile
- `report1.nsys-rep`, `v3aprobe.sqlite`: not mirrored; [v3a_probe.json](../baselines/mmq-v3/v3a_probe.json),
  [v3a_probe_liveness.json](../baselines/mmq-v3/v3a_probe_liveness.json),
  `v3a_probe.py` (не зеркалирован), [v3a_probe_analyze.py](../harness/mmq/v3a_probe_analyze.py),
  `ft_nsys_v3.sh` (not mirrored) - liveness + per-format analysis
- battery: [base](../baselines/mmq-v3/quality/base_stage_summary.json) / [win](../baselines/mmq-v3/quality/win_stage_summary.json)
  stage summaries, [v3a_divergence.json](../baselines/mmq-v3/v3a_divergence.json),
  [v3a_battery_analyze.py](../harness/mmq/v3a_battery_analyze.py), [run_battery_v3a.py](../harness/mmq/run_battery_v3a.py)
- [v3a_stage.sh](../harness/mmq/v3a_stage.sh) - serial stage runner (census + trap + watchdog)
- `v3a_build_results.py` - results assembler (not mirrored)

## CORRECTION (2026-09-19, Step 0 of the dense-q80-gemm campaign: [step0-profile.md](step0-profile.md))

The tile-32 MoE-term magnitude above ("~1.8 s, e2e fit") was an
e2e-subtraction estimate, not a direct measurement; it assumed tile-invariant
rest. Direct nsys at the MTILE=32 anchor (dense-q80-gemm step0,
denseq80.sqlite) measures grouped MoE = **4.37 s** of a 14.87 s chunk
(29.4%); the non-MoE share is 70.6%, not the ~78-88% claimed above.
Unaffected: the v3b no-fire conclusion and the ALU/LDS-floor finding (they
derive from the tile-16 nsys cross-check 12.62 -> 5.01 s and the CTA/liveness
data, not from the tile-32 subtraction). Correction reviewed in
review-b-step0.md (B4 CONFIRMED).
