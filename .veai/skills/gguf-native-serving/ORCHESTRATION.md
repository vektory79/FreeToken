# ORCHESTRATION.md - gguf-native-serving (self-sufficient skill)

This file is the orchestration contract for the gguf-native-serving skill. It is a
specialization of [`.veai/skills/code-with-subagents/SKILL.md`](../code-with-subagents/SKILL.md) (role and boundaries,
plan discovery, PLAN_EXECUTION / DIRECT_EXECUTION / ESCALATE_TO_PLAN, quality loop,
STOP GATE, handoff, evidence rules) to gguf-format enablement work. The skill is
launched WITHOUT code-with-subagents; this file replaces it. It does not replace
the domain content: discovery + the 7 phase waves live in [SKILL.md](SKILL.md),
the trap matrix in [TRAPS.md](TRAPS.md), the per-phase briefs in [tasks/](tasks/).

Default cycle:

```
task-00 discovery (incl. upstream alignment or waive) -> phase waves 1..7 (SKILL.md order)
each phase wave: implementation -> Test -> Review -> Triage -> Fix TP -> STOP GATE -> commit
```

## 1. Role and boundaries

You are the orchestrator of a gguf-format enablement campaign. You do not do the
whole work yourself: you delegate Code / Test / Review / Ask waves, and you own
the mode selection, the quality loop and the STOP GATE.

Within this role:
- you may run compact Ask-waves for technical discovery (task-00 is one);
- you may create and update campaign artifacts under `.tasks/<campaign>/`;
- you may delegate implementation, tests, review, triage and fix-waves to subagents;
- you may continue an existing campaign plan package if one exists;
- you may NOT require from the user technical context that research can produce;
- you may NOT silently swap an existing plan package for another format;
- you may NOT treat a successful coding-role call as task completion (STOP GATE);
- you may NOT hide blockers, test failures, unconfirmed assumptions or open TP;
- the user owns hardware decisions (box, ports, VRAM budget, targets) and
  repo-level git decisions (push, PRs, committing the skill files themselves);
  agents create exactly one working commit per phase at the STOP GATE (section 8)
  and never push.

Target outcome by mode:
1. PLAN_EXECUTION - the next phase is closed through its STOP GATE with the commit gate;
2. DIRECT_EXECUTION - a single-phase delta closed with the full per-wave loop;
3. ESCALATE_TO_PLAN - a canonical plan package exists and is either executed or handed off.

## 2. Mandatory discovery inputs (task-00)

No phase may start before the discovery wave has produced its outputs and the
user has answered its questions. task-00
([tasks/task-00-discovery.md](tasks/task-00-discovery.md)) is the
mandatory discovery wave. Its inputs:

- THE WORKING-REFERENCE QUESTION - ASK THE USER EXPLICITLY for the path to the
  working reference implementation where the target gguf format is ALREADY
  supported (llama.cpp or other). If the format is absent upstream, ask for any
  existing reference: kernels, LUTs, quant/activation pairings, block layouts
  and the CPU-resident-weights scheduling pattern are TRANSCRIBED FROM THERE,
  never from paper math (T35, D08). What is extracted from it: vec_dot kernels
  (ggml-cpu/arch/<isa>/quants.c), quantize_row_* (ggml-quants.c), block layouts
  (ggml-common.h), ISA build flags (build/CMakeCache.txt + flags.make), the
  backend-scheduling pattern (ggml_backend_sched splits, --n-cpu-moe).
- the upstream-alignment-or-waive decision: the target family checked against
  the roadmap/upstream issues; any deviation from upstream expectations
  requires an explicit user waive recorded in the discovery dump - no phase
  starts before this resolves.
- the discovery dump: `.tasks/<campaign>/tensors.txt`, `metadata.txt`,
  `research/anchor-verification.md`, the CUDA-vs-file gap matrix, the CPU
  compute-tier support matrix per format.
- hardware targets: the box (GPU, CPU cores / ISA tier, DDR5 and PCIe
  bandwidth), free ports, VRAM budget for boot probes.
- quality bars: the parity tolerance contract, the A/B baseline method, the
  on-topic battery bar, the microbench gate threshold (CPU-leg GB/s >= the
  box's effective PCIe gather bandwidth).

## 3. Mode selection

### PLAN_EXECUTION (default)

Discovery (task-00, including the upstream-alignment-or-waive step) + the 7
domain phases of SKILL.md ARE the default plan; the briefs
[task-00-discovery](tasks/task-00-discovery.md),
[task-01-config-shim](tasks/task-01-config-shim.md),
[task-02-tensor-translator](tasks/task-02-tensor-translator.md),
[task-03-tokenizer](tasks/task-03-tokenizer.md),
[task-04-kernel-dispatch](tasks/task-04-kernel-dispatch.md),
[task-05-expert-banks](tasks/task-05-expert-banks.md),
[task-06-e2e-ab](tasks/task-06-e2e-ab.md),
[task-07-cpu-compute-tier](tasks/task-07-cpu-compute-tier.md)
are the task files. Read SKILL.md fully, then the current
phase's brief and its required inputs; check whether a campaign plan package
already exists (`.tasks/<campaign>-tasks/PLAN.md`) - if the user provided one,
execute it and do not swap formats. Do not skip or reorder phases without the
user: each phase builds the invariants of the next. The single sanctioned
shortcut: phase 7 is skipped iff the file will be served offload-only (user
decision) - record the decision and keep the phase-6 offload baseline as the
final verdict.

### DIRECT_EXECUTION

For single-phase deltas and small fixes: adding one format to a dispatch table,
fixing a single trap, adjusting a fixture, tuning boot flags, a doc correction.
Show the compact Proposed Execution block (Goal / Scope / Implementation Plan /
Test Plan / Success Criteria / Risks) before the wave, then run the full
per-wave quality loop. Stop and escalate the moment new subtasks, durable
handoff needs or replanning appear.

### ESCALATE_TO_PLAN

When scope grows beyond the briefs: a new model-family branch, a new subsystem
integration, a phase-order change, several dependent waves that need durable
artifacts. Create the canonical plan package `.tasks/<kebab-case-name>-tasks/`
with PLAN.md + per-phase task files in the code-with-subagents format (Goal,
Scope, Specification with Requirements / Non-Goals / Acceptance Scenarios, How
to Use This Plan, Task Checklist, Shared Context, Research Artifacts; each task
file carries the full required-fields set). Then either continue in
PLAN_EXECUTION or hand off with exact paths. When in doubt between modes,
ESCALATE_TO_PLAN rather than improvise.

### Ask-waves outside task-00

Any open question that blocks a phase (variant quant, missing HF/FTW checkpoint,
unknown hardware cap) goes through a compact Ask-wave first: a concrete
engineering question, known anchors (files, classes, commits), expected answer
format. Run parallel Ask-waves only when independent.

### Replanning check

After every wave, silently re-check: is the current mode still valid, are the
next wave's dependencies still true, did scope grow? If replanning is needed,
show the delta and get new agreement when material.

## 4. Phase = wave: the quality loop

Every phase runs its brief through the FULL loop. A successful implementation
step is the START of the closing procedure, never the end of the phase.

1. Implementation (Code): delegate with the brief path + embedded essentials
   (section 6); the brief is self-contained.
2. Test: default for every phase. Skip only when the wave changed no behavior -
   justify in one line. Phase-specific shapes: synthetic-gguf fixture tests,
   parity tests vs gguf-py within the dfloat tolerance contract, static fusion
   validation before any live boot, `uv run pytest tests/ -m "not slow"` green
   before close. A failing test is a blocker: triage it first (Implementation
   bug / Test bug / Environment / Spec mismatch) before Review.
3. Review: x2 (two parallel passes - Review-A correctness/behavior/architecture,
   Review-B edge cases/integration/test gaps/operability) for all non-trivial
   waves. Kernel, hardware and budget phases are non-trivial BY DEFAULT:
   task-02 (fusion ordering), task-04 (kernel dispatch), task-05 (banks and
   pinning), task-06 (measurement validity), task-07 (kernels/budget/pools).
   Doc-only phases may use ONE focused review - justify in the wave log.
   Narrow explicit exception for measurement waves: when the wave's
   decisive fact is a sha-equality check on recorded artifacts, the wave
   may close with a single-stage orchestrator verification that reads the
   artifact directly instead of the full Review x2, provided the waiver
   rationale is recorded in the campaign triage file (precedent: the
   dense campaign's arbitration battery).
4. Triage: every finding classified TP / FP / Need more data, grounded in code,
   contract or the reference implementation. Need more data -> extra Ask-wave
   or the user. Never start a fix-wave without confirmed TP.
5. Fix TP only (Code), then re-run Test and Review as still needed. Limit: 3
   full cycles per wave, then report to the user with the state and options.

## 5. Phase STOP GATE

Before declaring a phase closed, verify ALL of:

- [ ] the brief's validation gates pass (each brief lists its own);
- [ ] all review findings triaged, no open TP;
- [ ] Test executed or explicitly waived with justification;
- [ ] [TRAPS.md](TRAPS.md) checked for the phase's trap set (pre-close checklist);
- [ ] commit created - one per phase (section 8);
- [ ] artifacts and report recorded under `.tasks/<campaign>/` with exact paths;
- [ ] process census clean - postflight done, no survivors (section 7);
- [ ] kb-distill invoked - phase/campaign insights distilled into kb/ (skill
      [kb-distill](../kb-distill/SKILL.md)); digest of what was recorded and
      skipped attached to the phase close;
- [ ] plan/brief checkboxes updated.

Hard invariant: a successful coding-role call means the implementation step
ended, not the phase. A phase is closed only through this gate.

## 6. Handoff (lossless)

Results of previous subagents are NOT automatically available to the next ones.
The briefs and the campaign research artifacts are the handoff carriers:
downstream agents receive exact file paths (`.tasks/<campaign>/tensors.txt`,
`metadata.txt`, `research/anchor-verification.md`, `verification/...`) plus the
embedded essentials - never "use the previous agent's result" or "see the last
wave". Losing a constraint, replacing a concrete path with a general phrase, or
making the next agent guess from chat history is a violation. Names are not
data; identifiers are not payload.

Payload minimums (specializations of the contract templates):
- Ask: the concrete question, the scenario, known anchors (file paths,
  commits), what to extract and where to record it (for task-00: the five
  extraction targets in section 2), expected answer format.
- Code: the brief path (the phase's brief under [tasks/](tasks/)), the phase number, the campaign
  artifact paths, the constraints and non-goals from the brief, expected output.
- Test: what changed, which validation gates of the brief apply, which fixtures
  and parity contracts, which checks to run.
- Review: the goal, the changed files or review scope, the factual change
  summary, known risks, the explicit focus (A or B), the brief path.
- Triage: all findings from both passes with origin, change context, and the
  required TP / FP / Need more data format.

## 7. Process-hygiene protocol (binding)

Every wave that spawns processes (ft serve, pytest, benchbw, microbench
harnesses) runs under this protocol. Reason: leaked CPU-eating survivors
corrupt later measurements (a prior-campaign `ft serve` hung 14+ hours; two
concurrent microbenches both leaked and invalidated their own numbers).

- PREFLIGHT census before the wave: `nvidia-smi` (VRAM + processes) and `ps`
  for the wave's patterns.
- RUNNER: every long-running invocation goes in a runner script with
  trap-on-EXIT cleanup, a FAILPAT watchdog (e.g.
  `AssertionError|OutOfMemoryError|Backend worker is gone|backend worker .* exited`)
  and hard timeouts. Polling /ready WITHOUT a deadline is forbidden.
- STOP escalation: SIGTERM -> 30 s -> SIGKILL. Graceful uvicorn shutdown after
  backend death never completes - do not wait for it.
- POSTFLIGHT census: re-run the ps census, kill this wave's survivors, census
  /dev/shm semaphores: `ls /dev/shm | grep -c "^sem\.mp-"`.
- SWEEP verification: before killing leftovers verify cmdline + start-time
  (spawn_main children reparented to systemd = stale orphans); IDE helpers
  legitimately hold baseline semaphores (a PyCharm forkserver held 5 live
  sem.mp-*) - attribute census deltas BEFORE killing, never kill IDE processes.
- Semaphore census deltas may self-resolve: /dev/shm sem.mp-* files auto-unlink
  when the holding multiprocessing resource_tracker dies (observed 11 -> 5,
  zero manual rm) - re-census before manual cleanup.
- RUNNER PID hygiene: a PID variable the watchdog references (MEASPID) must be
  exported BEFORE the watchdog block - under `set -u` the watchdog subshell
  exits when it references the unset MEASPID (the watchdog stops), while the
  trap-on-EXIT cleanup still pkill's the ft tree.
- BETWEEN-WAVE sweeps are REPORT-ONLY: never touch an active wave's processes;
  the active patterns are named in the prompt. A force-stopped agent is assumed
  dirty - the next wave starts with a sweep of ITS patterns.
- WITHIN a wave, measurement invocations are SERIALIZED: one run at a time,
  hard timeout, PID-reap (wait + verify the PID is gone) + ps census between
  runs. Concurrent runs leak AND corrupt each other's timings (T32, D07; the
  campaign's first microbench round was discarded for exactly this).
- ATTRIBUTION instrumentation is temporary by contract: [tag]-prefixed debug
  lines at decision points via init_logger (bare stdlib getLogger is
  boot-log-invisible, T46), a sha256 ledger of touched files taken before and
  after; at the end remove the lines, verify the tree bit-for-bit and re-run
  the fast suite. Never ship instrumentation (fix3 campaign pattern).
- OMP/thread-pinning env is set BEFORE process start (libgomp parses in a
  pre-main constructor; setenv from main collapses threads onto core 0).

## 8. Commit discipline

- ONE commit per phase, created at the STOP GATE, after all gates pass.
- Conventional Commits: imperative, lowercase, no trailing period
  (`feat(gguf): wire iq kernels into moe dispatch`).
- The gate `- [ ] I've created a git commit for this task` is part of every
  phase's acceptance criteria (in each brief under [tasks/](tasks/)).
- `Assisted-by: Veai` trailer allowed; attribution follows the repo rules.
- NO `git push`, no `gh pr create` / `gh pr comment` / `gh issue create` by agents.
- Staging hygiene: stage by an EXPLICIT path list (never `git add -A` / `.` -
  the index may hold stale staged versions from parallel sessions); verify
  staged == worktree per file (`git diff --cached`); verify ZERO `.veai/memory`
  entries staged (parallel orchestrator sessions churn it); re-run the fast
  test waves immediately before committing; the commit body carries the
  measured results (the why the diff does not show).
- Bisect-green series where ONE test file spans multiple commits: stage it in
  dependency-ordered slices (commit-2 subset green, full file restored at
  commit 3); extract fails-before regression tests to their proper home
  (tests/moe/test_legacy_format.py). Tests touching convert_checkpoint need a
  CUDA skipif (it hard-inits a CUDA context; fixture-level skip preferred).
- The skill files themselves (`.veai/skills/gguf-native-serving/`) are committed
  (6e673ab); further skill edits stay uncommitted until the user asks.

## 9. Evidence rules, completion, escalation

Evidence rules:
- Anchor claims on file:line, diffs, commits, measurement artifacts or the
  reference implementation; do not invent APIs, test patterns or constraints.
- Every projection is anchored on a working reference A/B, never on paper math
  (T35, D08): a projection without a reference anchor produces false NO-GO.
- Measured end-to-end numbers beat verdict strings: a benchbw verdict string
  "offload" with CPU/PCIe ratio < 2.0 is EXPECTED - the win is measured
  end-to-end, single variable `--moe-strategy` (phase 7,
  [tasks/task-07-cpu-compute-tier.md](tasks/task-07-cpu-compute-tier.md)).

Completion conditions:
- all phases the user asked for passed their STOP GATEs (commit gate included);
- quality battery delivered with the honest verdict (A/B table + quality table);
- artifacts recorded with exact paths; no hidden blocker, test failure or open TP;
- final message: wave-by-wave narrative, artifact paths, verdict, next-step options.

Escalation points - go to the user with options, never silently rounded up:
- budget asserts: microbench gate fail (the port cannot pay - keep offload and
  report the number), per-signature KV floor fail-fasts, cache-budget
  min-plan failures;
- hardware failures: SIGILL (ISA override), IMA, OOM, VRAM shortfall, backend
  death on the box;
- quality-bar misses: battery below the agreed on-topic bar, parity outside
  the dfloat tolerance contract, A/B shows no hybrid win;
- loop limit reached (3 cycles) with open TP; missing user inputs (working
  reference, HF/FTW checkpoint, hardware targets).

## 10. Cross-references

- [SKILL.md](SKILL.md) - discovery + the 7-phase domain pipeline, mandatory
  inputs, the static-fusion-validation rule, debug tool priority.
- [TRAPS.md](TRAPS.md) - T01-T63 + recipes D01-D10; the pre-close checklist
  for every phase.
- Per-phase briefs: [task-00-discovery](tasks/task-00-discovery.md),
  [task-01-config-shim](tasks/task-01-config-shim.md),
  [task-02-tensor-translator](tasks/task-02-tensor-translator.md),
  [task-03-tokenizer](tasks/task-03-tokenizer.md),
  [task-04-kernel-dispatch](tasks/task-04-kernel-dispatch.md),
  [task-05-expert-banks](tasks/task-05-expert-banks.md),
  [task-06-e2e-ab](tasks/task-06-e2e-ab.md),
  [task-07-cpu-compute-tier](tasks/task-07-cpu-compute-tier.md) - each is
  self-contained and carries its own acceptance criteria including the
  commit gate.
- Worked example (glm5next, IQ3_XXS/IQ4_XS/Q6_K, H=4096 I=2048 E=288 top_k=8):
  microbench gate 53-66 GB/s vs the >= 40 gate; final verdict hybrid 16.67
  tok/s vs offload 12.97 @64k (commits 11f1a80 + 979e3fc + 63b9bff).
  Anti-lesson: the Task-06 "handshake floor" estimate was fetch volume
  misattributed as sync (T33) - hardware with reference kernels beat the
  projection.

## 11. Session-end self-update

At the END of every session that produced reusable knowledge, the Orchestrator
MUST run a self-update pass before closing out. Reusable knowledge: root
causes with commit ids, measured expectation classes / numbers, new traps,
protocol or hygiene refinements, corrections to now-stale facts.

- Where to integrate - update-in-place over duplication, cross-reference
  instead of repeating:
  - [TRAPS.md](TRAPS.md) - a new T-number continuing the sequence, or an
    in-place extension of an existing trap already covered;
  - task briefs - expected-result classes (task-06 for serving expectations);
  - this file - hygiene / protocol bullets (sections 7 and 8).
- Consistency gates: trap / recipe counters stay in sync across the TRAPS.md
  H1, the SKILL.md counter line and section 10; ASCII only - no em-dashes,
  ASCII arrows; no lab-diary prose; open gaps are labeled open, without a
  trap number.
- Execution: delegate the integration to a Code subagent with the
  session-fact payload; for non-trivial integrations run a review pass
  (blocking_status) and apply cheap polish for non-blocking notes.
- Boundaries: the pass NEVER commits or pushes by itself - commits remain an
  explicit user decision (section 8); the SKILL.md YAML preamble is never
  touched; no README creation.
