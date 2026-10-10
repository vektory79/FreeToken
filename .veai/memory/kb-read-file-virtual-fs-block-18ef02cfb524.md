---
name: "kb-read-file-virtual-fs-block"
description: "read_file blocked on kb/ (empty content or virtual-FS warning); write/edit work; read via shell or call_code_agent"
type: project
lastUpdated: 2026-09-26T23:51
lastRecall: 2026-10-10T13:46
---

Orchestrator read_file CANNOT read files under kb/ in this harness: existing kb files return "success" with EMPTY content, and an explicit attempt surfaced the warning "Virtual file is not accessible for reading: kb/harness/README.md". Meanwhile write_file and edit_file to kb/ WORK (the T2 brief kb/cases/decode-t2-host-delivery/TASK.md was authored and edited that way on 2026-09-26).

**Why:** virtual-FS restriction of the IDE harness (same family as the .veai write trap); discovered 2026-09-26 during kb-distill when repeated read_file attempts on kb/topics/moe-hybrid-cost-model.md and kb/harness/README.md returned nothing.

**How to apply:** to read kb/ content use the escape hatches - run_command (cat/cp) or a delegated call_code_agent (its file tools read kb/ fine). kb/ is git-tracked and present on disk, so "not accessible" is a tool limitation, NOT an absent path; never conclude a kb file is missing from read_file output, and never repeat the failing read_file call (see orchestrator-emission-loop).
