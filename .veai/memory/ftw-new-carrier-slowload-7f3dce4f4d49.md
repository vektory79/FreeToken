---
name: "ftw-new-carrier-slowload"
description: "FTW new carrier 148s boot: NOT banks (25s, faster); tier compact on idle loop + bimodal warmup; fix = background compact"
type: project
lastUpdated: 2026-10-24T15:08
---

# FTW new carrier "slow boot" 148s vs legacy 81s: carrier NOT at fault (2026-10-22 forensics)

Question: why does the NEW FTW carrier (-new, 19 shards) boot 148s while legacy (81s)? Hypothesis "148s ~= raw GGUF" was a coincidence.

**Banks phase is FASTER on the new carrier**: A 25.0s @ 5.33 GB/s (84 tasks / 64089 jobs) vs B 28.0s @ 4.95 GB/s (126 / 75663); shards identical (19 vs 18 x ~8 GiB); io_uring direct pool=8 both, zero fallback markers. The ~67s delta sits AFTER engine-ready: ready->chat-200 was 83s (A) vs 22s (B); boot->API-ready only +6s (of which +9s driver-JIT 343 MiB on first boot of the night, noisy).

**Root 1 (+26.6s, code+log-proven): session-tier compaction on the scheduler idle loop.** scheduler.py:166 run_when_idle -> maybe_compact_tier runs ON the loop: shared cache dir had 39 tombstones / 24.34 GiB dead of 57.55 GiB blob (41% > P8/P13 gate) -> compaction 13:59:07-13:59:34 blocked batch formation, aborted (payload crc mismatch @ blob+31.17 GiB, ~1.2 GB/s - one-off torn read, retry of the same bytes succeeded); first batch only at 13:59:33.6. Retry compaction (14:00:30-14:01:17) cleaned the blob -> boot B inherited a CLEAN cache and paid nothing. Cache-state effect, not carrier.

**Root 2 (+39.2s, effect proven, mechanism partly open): bimodal first forward.** Same 30-token request: 53.6s (A) vs 14.4s (B) own-window. Carrier-independent: 53.6s on raw-GGUF d4-step2 (no tier/compact), 54.2s on legacy (decode-research arm, clean tier), 90.5s NVFP4; fast mode 13-16.5s on most boots. Pattern: slow mode = first boot after a GPU idle window. NOT nvcc (no artifacts), NOT JIT volume (B logged more JIT bytes yet warmed faster), NOT page cache (O_DIRECT).

**Cheap confirm (next cloud window):** reboot the new carrier twice with controlled cache dir; expect ~60-65s API-ready both; if warmup 53->14s, root 2 closes as GPU idle-window cold start; optional nvidia-smi dmon on first forward.

**Fix recommendations (NOT implemented):** (1) move maybe_compact_tier off the scheduler idle loop to a background thread - removes the 27s first-request block; (2) runners should log t(API-ready) separately from chat-200.

Residual anomalies: one-off crc-mismatch of compaction on bytes that passed on retry (possible torn read); boot A result.env missing grp metrics while rss.csv has a valid grp column (runner artifact).

Artifacts: .tasks/round5-deferred-ports/ftw-new-carrier-slowload.md, zero-scratch/{bootA-new,bootB-legacy}/.
