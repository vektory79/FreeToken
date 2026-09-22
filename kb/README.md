# FreeToken knowledge base

Consolidates the durable artifacts of the 2026-09 optimization campaigns so they
survive outside the git-ignored `.tasks/` working directory. Committed on the
private branch `vektory79`.

Layering rule (one fact, one home):

- Memory Bank (`.veai/memory/`) - verdicts and search hooks, updated by the
  assistant between sessions.
- `kb/` (this tree) - self-contained documents, reusable scripts, measured
  numbers.
- `.tasks/` and `.veai/tmp*` - raw originals; heavy binaries also mirrored under
  `kb/raw/` (git-ignored). Do not edit copies here; treat `.tasks/<campaign>/`
  as the live source and re-copy when a campaign produces new artifacts.

Every document should carry a header with date, hardware, commit hash and
status (`validated` / `superseded by ...`). Numbers without conditions are
wrong numbers: pre-iommu and post-iommu baselines on this rig differ, as do
daytime-vs-night decode runs (daytime noise swamped a real -10% once).

## Index

### methods/ - how to measure and how to run a campaign

| File | Hook |
|---|---|
| orchestration-digest.md | Wave protocol, STOP GATE, trap digest (ORCHESTRATION+TRAPS condensed, replaces re-reading skill files) |
| step0-profile.md | nsys step-0 split: launch decomposition, BW vs ALU regime classification (incl. widening projections under MTILE sweep) |
| step0-profile-mmq.md | same method, mmq-prefill-kernel instance |
| ab-mtile.md / ab-nsplit.md | A/B runner usage, per-call env kill switch, two-boot flip |
| quality-ab.md / v0-ab.md / v2-ab.md | quality battery + A/B numbers for MMQ variants |
| arbitration-battery.md | quality arbitration method (env0/env1, divergence analysis) |
| vram-headroom.md | per-boot VRAM/init measurement method |
| wave1-notes.md, wave2-candidate-a.md | dense-q80 wave orchestration in practice |
| step0-source-read.md, v3a-surface-map.md, v3a-ab.md | v3 stationary-MoE surface map and A/B |

### harness/ - ready-to-run scripts (exist nowhere else)

| Dir | Contents |
|---|---|
| serve-measure/ | measure.py (patched: skip SSE times[0], anchor `input throughput (token/s):`), req_*.json payloads |
| ab-runner/ | dense q8_0 A/B machinery: mtile/nsplit sweeps, arbitration, microbenches, vram_headroom_run, ab_nsplit_stage.sh (parameterized stage runner) |
| mmq/ | MMQ A/B: v0ab/v2ab analyze, v2ab probe, step0_analyze, v3a stage/analyze + battery, sitecustomize liveness probe |
| quality/ | ft_phase6_battery.py, run_battery.py, analyze_divergence.py (quality A/B battery) |
| probes/ | pcie_bw.cu + prebuilt pcie_bw/pcie_bw2 binaries, ram_bw.py, dram_dir_bw.py, size_scaling.py, nvme_thermal_test.sh (RTX 5090 / NVMe rig probes) |
| repro/ | crash repro runners, fix1 run_one.sh, merge-wave boot-smoke |
| ftw/ | FTW fast-path conversion/parity/measure scripts |

Harness gotchas (validated 2026-09-16/17, RTX 5090): first SSE frame arrives
before prefill - skip times[0]; throughput parser must anchor the exact line
`input throughput (token/s):`; measure.py stdout needs cleaning before
json.loads; medians exclude chunk-1 warmup AND the last full chunk (the last
chunk's input-throughput line is bogus ~1552-1602 tok/s); pkill must use the
`[f]t serve` bracket pattern.

### findings/ - durable technical maps (with code anchors)

| Dir | Hook |
|---|---|
| gguf-glm5next-hybrid/ | ggml CPU kernel study, hybrid machinery mapping, gemv mapping, task06 estimate |
| gguf-glm5next-path-a/ | llamacpp tensor kinds, config keys, tensor map, anchors; tensors.txt dumps |
| llamacpp-glm5next-snapshot/ | numbered snapshot of upstream llama.cpp GLM5NEXT implementation (see its README) |
| mmq-prefill-kernel/ | upstream ggml moe-align + tile-glue research (base of v2 grouped MMQ) |
| fix1-radix/ | code anchors for the mamba_last_track_seqlen chunk-transition bug (fixed in 5b72aba) |

### incidents/ - root-cause reports

| Dir | Status |
|---|---|
| nvfp4-1m-capacity/ | CLOSED: 1M infeasible on 32 GB, 786432 verified, fill ceiling ~678k, fail-fast gate |
| crash-mr080-400k/ | OPEN (2026-09-20): winner recipe crashes at boot, >=29.28 GiB free needed; repro runner in harness/repro/ |
| fix1-radix/ | hardware verification report for the radix fix |

### baselines/ - measured numbers (machine-readable)

Per-campaign result JSON/CSV/OUT, boot inventories and quality summaries.
Anchors: prefill 792-808 tok/s bare baseline after dense-q80+m-tile (2026-09);
earlier 293 tok/s anchor is pre-dense-q80. Always re-state the config line
with a number.

### cases/ - per-campaign briefs, plans, reviews

Task briefs and plan packages double as templates for future campaign plans
(Goal / Scope / Specification / Acceptance / Checklist). merge-main-work/ holds
the discovery/adaptation/test-wave procedure for integrating origin/main.

### raw/ - heavy profiles (git-ignored)

nsys .sqlite exports and .nsys-rep files mirrored from `.tasks/`. The .sqlite
files are queryable without re-running GPU benchmarks. Original .nsys-rep and
per-run logs stay in `.tasks/`; if `.tasks/` is ever wiped, only the
un-mirrored logs are lost.