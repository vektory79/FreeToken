---
name: "cuda-debug-tool-strategy"
description: "CUDA debug strategy: CUDA_LAUNCH_BLOCKING first, compute-sanitizer second, python instrumentation fallback"
type: project
lastUpdated: 2026-09-15T19:12
lastRecall: 2026-09-16T17:52
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
