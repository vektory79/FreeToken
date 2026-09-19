---
name: "cuda-debug-tool-strategy"
description: "CUDA debug/profiling: CUDA_LAUNCH_BLOCKING, compute-sanitizer, nsys live; ncu ERR_NVGPUCTRPERM blocked"
type: project
lastUpdated: 2026-09-19T17:59
lastRecall: 2026-09-19T18:14
---

# CUDA debugging tool strategy: CUDA_LAUNCH_BLOCKING -> compute-sanitizer -> python instrumentation

Lesson from the glm5next GGUF campaign (IMA root-cause, 2026-09-14). Priority order for
async CUDA errors (illegal memory access, IMA):

1. `CUDA_LAUNCH_BLOCKING=1` - makes launches synchronous; the visible frame then names
   the TRUE culprit kernel. Fast (~1.2-2x slowdown). Use FIRST. In the campaign this
   localized the IMA to `fast_index_copy_multi_jit` in OffloadMoeCache.copy_missing
   within ~4s (vs 100s+ async surface in triton _init_handles, which is a red herring -
   the corrupting launch precedes the visible frame).
2. `compute-sanitizer --tool memcheck --error-exitcode 9 <cmd>` - NVIDIA's CUDA debugger
   (successor to cuda-memcheck; /usr/local/cuda/bin/compute-sanitizer). Tools:
   memcheck (OOB/misaligned/invalid ptr), racecheck (shared-mem races), initcheck
   (uninit reads), synccheck (stream/event API misuse). Gives kernel name + thread/block
   + address + access type at the EXACT moment of the bad access - much more precise than
   launch blocking, but 10-50x slowdown. For a 147 GB model with a ~2 min load, budget
   10-30 min to reach the crash point.
3. Python-side instrumentation (bounds assert on every pointer in the copy plan just
   before the suspect call) - fallback when sanitizer cannot attach through the wrapper
   (e.g. tvm-ffi). Convert the debug guard into a permanent construction-time check after
   the root cause is found (done in _build_fused_copy_plan: rejects pageable CPU banks /
   per-layer width != plan feat / slot-pool geometry mismatch).

Traps:
- CUDA errors are ASYNC: the failing kernel and the kernel that CORRUPTED memory are
  different launches. Never trust the frame where the error surfaces.
- A green CPU test suite does NOT cover device-side memory validity - pageable-vs-pinned
  only exists on a CUDA device (the pin-count trap produced an IMA invisible to all CPU
  tests).
- compute-sanitizer serializes launches, so it also removes the async ambiguity on its own;
  if it is too slow to reach the crash, fall back to CUDA_LAUNCH_BLOCKING=1.

**Why:** the campaign burned multiple boot cycles (~2 min each) chasing a false frame
before switching to launch blocking.
**How to apply:** any new CUDA IMA/illegal-address during serving bring-up: bisect with
CUDA_LAUNCH_BLOCKING first, escalate to compute-sanitizer only if the named kernel is not
enough, reserve python instrumentation for wrapper-blocked cases.

## Profiling (not debugging) a live serve: nsys interactive session (2026-09-17, Step-0 wave)
- Zero-code-change profiling of `ft serve`: `nsys launch --session=NAME --trace=cuda .venv/bin/ft ...`, then `nsys start --session=NAME --sample=none` and `nsys stop --session=NAME` around a steady window anchored on server "input throughput" lines; analyze via `nsys export --type sqlite` (per-kernel sums grouped by short-name per chunk). Measured overhead was nil (chunks inside vs outside the window identical).
- CLI traps: `nsys launch` REJECTS `-o` and `--sample` (both belong on `nsys start`); failed iterations leave leftover .nsys-rep files in CWD (repo root) - clean them up.
- Launch-count cross-checks validate the capture window: gguf moe path = moe_vec_q 126/chunk (42 layers x gate/up/down) and fast_index_copy_multi 42/chunk.

## ncu is BLOCKED on this box (2026-09-19, dense-q80-gemm Step 0)
`ncu` fails with ERR_NVGPUCTRPERM (GPU perf counters not enabled for the user) - do not plan a profiling campaign around ncu; enabling counters requires the user (driver permission). Substitute software discriminators that pin a kernel's regime without counters: (1) measured-vs-model BW closure per op class - a UNIFORM measured/model ratio across different shapes = a throughput ceiling expressed in traffic units (here ~22 TF/s per-MAC), while closure near 1.0x = BW-bound (moe_vec closed at 0.92x; dense q8_0 did NOT at 0.62x); (2) M-sweep on one exact shape - time linear in M at a constant rate, with implied BW above the spec peak, proves BW-bound is physically impossible; (3) format A/B at identical shape/MACs - more bytes but slower = not BW-bound (q6_K: 23% fewer bytes, 6.3% slower); (4) per-class rate spread - e.g. kda_in_proj 18.08 TF/s at W=108.35 MB > L2 vs 20.9-22.4 TF/s for W<=71.3 MB isolates an L2-overflow mix.
