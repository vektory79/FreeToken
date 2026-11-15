---
name: "kb-read-file-virtual-fs-block"
description: "read_file on kb/ works again (2026-11-15, 4 files verified); 2026-09-26 virtual-FS block historical; fallback cat"
type: project
lastUpdated: 2026-11-15T14:22
lastRecall: 2026-11-15T14:25
---

read_file on kb/ WORKS again in this harness (2026-11-15: verified byte-perfect content on kb/harness/decode-research/README.md, kb/topics/gguf-serving-perf-matrix.md, kb/topics/strata-maya-inference-study.md, kb/topics/moe-hybrid-cost-model.md). The 2026-09-26 block (empty content / "Virtual file is not accessible for reading") no longer reproduces; kb writes always worked.

**Why:** historical virtual-FS restriction of the IDE harness; apparently lifted or fixed since 2026-09-26.

**How to apply:** read kb/ files with read_file normally; if the empty-content block ever reappears, fall back to run_command (cat) - never conclude a kb file is missing from read_file output alone.
