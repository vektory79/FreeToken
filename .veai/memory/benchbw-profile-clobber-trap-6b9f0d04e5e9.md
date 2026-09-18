---
name: "benchbw-profile-clobber-trap"
description: "benchbw writes the FULL per-GPU profile per run: single-dtype run clobbers other formats' fractions; no TTL/fingerprint"
type: project
lastUpdated: 2026-09-15T17:28
lastRecall: 2026-09-16T20:14
---

# benchbw profile clobber: single-dtype rerun drops other formats' fractions

Gotcha (incident 2026-09-15): benchbw writes the FULL per-GPU profile file (`~/.cache/freetoken/benchbw/<gpu-uuid>.json`, `result` dict + `_atomic_write_json`, python/freetoken/moe/benchbw.py:836-861) on every run. A rerun limited to one dtype (e.g. `--dtype gguf`) therefore REPLACES the whole file: every other format's measured legs and fetch fractions are lost.

**Why it bit:** a gguf-only re-profile after the kernel port silently dropped the nvfp4 entry (f=24.9%) -> NVFP4 hybrid auto on that box fell back to cap-1 with only a rank-0 warning (the legacy benchbw.json still held nvfp4 but is shadowed by the per-GPU file). The user's production NVFP4 hybrid config regressed until a full rerun restored it.

**Also:** the profile has NO TTL and NO kernel-tier fingerprint - a profile written before a kernel change (e.g. pre-avx2-w4a8k scalar-leg f=78%) is silently consumed as if current. A fingerprint/schema bump is the known follow-up (doc note only, not implemented).

**How to apply:** after ANY kernel/bench change, always re-run the FULL `ft bench bw` across all formats (never --dtype-limited) and verify the profile file contains every format entry; when booting hybrid, check the fraction log line matches the expected profile, and treat a cap-1 fallback + warning as a missing-profile-entry symptom.
