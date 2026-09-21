# orchestration-notes.md - dense q8_0 GEMM campaign protocol digest

Sources (read in full 2026-09-19): ORCHESTRATION.md (324 lines), TRAPS.md
(280 lines) under /media/ai/src/FreeToken/.veai/skills/gguf-native-serving/,
plus the campaign TASK.md (HEAD 1706aae). Self-sufficient digest for the
orchestrator and all wave agents - it replaces re-reading both skill files.
Section 6 keeps trap/recipe texts verbatim (original Russian is the binding
formulation; compact English gist added). Skill files are NOT copied into
.tasks (user directive 2026-09-15) - do not duplicate them.

## 1. Wave protocol

### 1.1 Role split (ORCHESTRATION section 1)
The orchestrator delegates Code / Test / Review / Triage / Fix / Ask waves; it
owns mode selection, the quality loop and the STOP GATE. It may: run compact
Ask-waves, create/update campaign artifacts under `.tasks/<campaign>/`,
delegate implementation/test/review/triage/fix waves, continue an existing
campaign plan package. It may NOT: require user-supplied technical context
that research can produce, silently swap an existing plan package, treat a
successful coding-role call as task completion, hide blockers / test failures
/ unconfirmed assumptions / open TP. The user owns hardware decisions (box,
ports, VRAM budget, targets) and repo-level git decisions (push, PRs,
committing skill files); agents create exactly one working commit per phase
at the STOP GATE and never push. THIS campaign tightens further: "Commit only
when the user asks. Never push." - the campaign wording wins for code commits
(see 4).

### 1.2 Exact per-wave sequence (ORCHESTRATION header + section 4)
```
task-00 discovery -> phase waves; each phase wave:
implementation -> Test -> Review (x2) -> Triage -> Fix TP -> STOP GATE -> commit
```

1. Implementation (Code wave): delegate with the brief path + embedded
   essentials (ORCHESTRATION section 6); the brief is self-contained.
2. Test (default for every phase): skip only when the wave changed no
   behavior, justified in ONE line. Failing test = BLOCKER: triage first
   (Implementation bug / Test bug / Environment / Spec mismatch) before Review.
3. Review x2 (parallel passes) for all non-trivial waves - Review-A
   correctness/behavior/architecture, Review-B edge cases/integration/test
   gaps/operability. Kernel, hardware and budget phases are non-trivial BY
   DEFAULT (task-02/04/05/06/07). Doc-only phases may use ONE focused
   review, justified in the wave log.
4. Triage: every finding classified TP / FP / Need more data, grounded in
   code, contract or the reference implementation. Need more data -> extra
   Ask-wave or the user. NEVER start a fix-wave without confirmed TP.
5. Fix TP only (Code wave), then re-run Test and Review as still needed.
   Limit: 3 full cycles per wave, then report state + options to the user.

After every wave the orchestrator silently re-checks: is the current mode still
valid, are the next wave's dependencies still true, did scope grow? (Replanning
check; show the delta and get new agreement when material.)

### 1.3 STOP GATE (ORCHESTRATION section 5, verbatim checklist)
Before declaring a phase closed, verify ALL of:

- [ ] the brief's validation gates pass (each brief lists its own);
- [ ] all review findings triaged, no open TP;
- [ ] Test executed or explicitly waived with justification;
- [ ] TRAPS.md checked for the phase's trap set (pre-close checklist);
- [ ] commit created - one per phase (section 8);
- [ ] artifacts and report recorded under `.tasks/<campaign>/` with exact paths;
- [ ] process census clean - postflight done, no survivors (section 7);
- [ ] plan/brief checkboxes updated.

Hard invariant: a successful coding-role call means the implementation step
ENDED, not the phase. A phase is closed only through this gate.

### 1.4 Mode descriptions relevant to this campaign
- PLAN_EXECUTION (default): discovery + the phase plan ARE the plan; briefs
  are the task files; do not skip or reorder phases without the user. For
  dense-q80-gemm the campaign TASK.md (Step 0 -> Candidate A -> optional B ->
  Tests -> Commits) plays the plan-package role (follow-up of
  .tasks/mmq-v3-stationary-moe, not of the 7-phase skill plan).
- DIRECT_EXECUTION: single-phase deltas and small fixes (one dispatch-table
  entry, one trap fix, fixture tweak, boot-flag tuning, doc correction).
  Requires a compact Proposed Execution block BEFORE the wave (Goal / Scope /
  Implementation Plan / Test Plan / Success Criteria / Risks), then the FULL
  per-wave quality loop. Escalate on new subtasks, durable handoff needs or
  replanning.
- ESCALATE_TO_PLAN: scope grows beyond the briefs (new subsystem, phase-order
  change, dependent waves needing durable artifacts) -> canonical plan package
  `.tasks/<kebab-name>-tasks/` with PLAN.md + per-phase task files (Goal,
  Scope, Specification with Requirements / Non-Goals / Acceptance Scenarios,
  How to Use, Task Checklist, Shared Context, Research Artifacts). When in
  doubt between modes, ESCALATE_TO_PLAN rather than improvise.
- Ask-waves outside task-00: any open question blocking a phase (unknown
  hardware cap, missing checkpoint, user decision) goes through a compact
  Ask-wave first: concrete question, known anchors (files, classes, commits),
  expected answer format. Parallel Ask-waves only when independent.

### 1.5 (a) Measurement-only wave closure (Step 0: no code change, no commit)
Step 0 of dense-q80-gemm (nsys split + traffic model under MTILE=32, D09/D10
recipe) changes no code. How the contract maps:

- Test: waived with the one-line justification "wave changed no behavior"
  (section 4.2 is written exactly for this case).
- Commit gate item: N/A - no diff to commit; "one commit per phase" applies
  to code phases. Code-phase commit BODIES carry measured results; for a
  measurement-only wave the measured results live as ARTIFACTS under
  `.tasks/<campaign>/` with exact paths instead (v3a precedent:
  step0-source-read.md, v3a_probe.json, nsys .sqlite/.rep in the folder).
- Review: still applies - measurement validity is default-non-trivial
  ("task-06 measurement validity"), so Step 0 conclusions (regime
  classification, per-op decomposition, BW-bound check) get the Review-A/B
  pass before the design wave consumes them; check against T52 and T35.
- STOP GATE otherwise unchanged: brief gates (nsys split, traffic model vs
  measured time, ~312 launches decomposed with M/N/K per class), findings
  triaged with no open TP, TRAPS check, artifacts recorded, census clean
  (postflight), plan/brief checkboxes updated.
- Decision handoff: Step 0 output chooses between "BW-bound -> widening
  projection legitimate" and "ALU/LDS-bound -> quantify Candidate A payoff
  first" (T52). Material replanning input - present it to the user before
  designing the kernel swap.

### 1.6 (b) Kernel-implementation wave (Candidate A / B)
Full loop with all gates. Campaign bindings from TASK.md:
- Pre-audit BEFORE templating (T40/T51): token_offs sizing, ds coverage at
  nwarps 4 vs 8, need_check boundary math, per-thread array dims - the dense
  body shares lineage with moe_q but is NOT identical.
- Separate knob FREETOKEN_GGUF_DENSE_MTILE (never overload the MoE knob;
  single source of truth in moe.py; knob unit + regression test mirroring
  test_moe_mtile_env_knob).
- A/B per T45 (per-call env kill switch + two boots with env flip; switch
  removed post-validation, 7f8c570 precedent); validity per T43/T44 (medians
  exclude chunk-1 warmup AND last full chunk; BEFORE reproduces the 556.43
  tok/s class within ~3% run-variance).
- Liveness per T46: nsys kernel-name contract + exact launch-count math;
  dense stays ~312/chunk, per-launch grid shrinks; decode graphs re-captured
  at boot - re-measure decode, do NOT assume it stays flat.
- Full gate before commit: `uv run pytest tests/ -m "not slow"
  --ignore=tests/models/test_glm5_next_kda_snapshot.py`; extend
  tests/kernels/test_gguf_quant.py (mirror, do not create).
- Candidate B: D06 microbench gate BEFORE any port (clears the in-tree MMQ
  baseline).

## 2. Review protocol

### 2.1 Payloads (ORCHESTRATION section 6, lossless handoff)
Results of previous subagents are NOT automatically available to the next
ones; briefs and campaign research artifacts are the handoff carriers.
Downstream agents get exact file paths plus embedded essentials - never "use
the previous agent's result"; losing a constraint, replacing a concrete path
with a general phrase, or making the next agent guess from chat history is a
violation. Names are not data; identifiers are not payload. Minimums:
- Review: the goal, changed files or review scope, factual change summary,
  known risks, the EXPLICIT focus (A or B), the brief path.
- Triage: ALL findings from both passes with origin, change context, and the
  required TP / FP / Need more data format.
- Test: what changed, which validation gates of the brief apply, which
  fixtures and parity contracts, which checks to run.
- Code: the brief path, phase number, campaign artifact paths, constraints
  and non-goals from the brief, expected output.
- Ask: the concrete question, the scenario, known anchors (file paths,
  commits), what to extract and where to record it, expected answer format.

Pass structure: Review-A = correctness / behavior / architecture; Review-B =
edge cases / integration / test gaps / operability. Both pass in parallel on
the same change; every non-trivial wave gets both (one focused review only
for doc-only phases, justified in the wave log).

### 2.3 Triage rules
- Every finding from either pass is classified TP / FP / Need more data,
  grounded in code, contract, or the reference implementation - not opinion.
- TP = blocking: fixed by a Fix TP wave before the STOP GATE (gate item:
  "all review findings triaged, no open TP"). FP: dismissed with grounding
  recorded. Need more data: extra Ask-wave or the user; never guessed.
- Non-blocking nits: recorded; cheap polish at session-end self-update.
- Never start a fix-wave without confirmed TP. Limit 3 full cycles per wave,
  then report to the user with state and options.

### 2.4 "Fix TP" meaning
A Code wave fixing ONLY the confirmed-TP findings (never FP, never speculative
hardening), then re-running Test/Review as still needed. A fix spawning new
subtasks or needing durable handoff escalates the mode (DIRECT_EXECUTION ->
ESCALATE_TO_PLAN / user decision).

## 3. Process hygiene rules (ORCHESTRATION section 7, binding)

Reason: leaked CPU-eating survivors corrupt later measurements (a prior
`ft serve` hung 14+ hours; two concurrent microbenches leaked and
invalidated their own numbers).

- PREFLIGHT census before the wave: `nvidia-smi` (VRAM + processes) and `ps`
  for the wave's patterns.
- RUNNER: every long-running invocation goes in a runner script with
  trap-on-EXIT cleanup, a FAILPAT watchdog (e.g.
  `AssertionError|OutOfMemoryError|Backend worker is gone|backend worker .* exited`),
  hard timeouts; polling /ready WITHOUT a deadline is forbidden.
- STOP escalation: SIGTERM -> 30 s -> SIGKILL. Graceful uvicorn shutdown after
  backend death never completes - do not wait for it.
- POSTFLIGHT census: re-run the ps census, kill this wave's survivors, census
  /dev/shm semaphores: `ls /dev/shm | grep -c "^sem\.mp-"`.
- SWEEP verification: before killing leftovers verify cmdline + start-time
  (spawn_main children reparented to systemd = stale orphans); IDE helpers
  legitimately hold baseline semaphores (a PyCharm forkserver held 5 live
  sem.mp-*) - attribute census deltas BEFORE killing, never kill IDE processes.
- Semaphore census deltas may self-resolve: /dev/shm sem.mp-* files auto-unlink
  when the holding resource_tracker dies (observed 11 -> 5, zero manual rm) -
  re-census before manual cleanup.
- RUNNER PID hygiene: the watchdog-referenced PID variable (MEASPID) must be
  exported BEFORE the watchdog block - under `set -u` the watchdog subshell
  exits on the unset MEASPID (watchdog stops) while trap-on-EXIT cleanup
  still pkill's the ft tree.
- BETWEEN-WAVE sweeps are REPORT-ONLY: never touch an active wave's processes;
  the active patterns are named in the prompt. A force-stopped agent is assumed
  dirty - the next wave starts with a sweep of ITS patterns.
- WITHIN a wave, measurement invocations are SERIALIZED: one run at a time,
  hard timeout, PID-reap (wait + verify the PID is gone) + ps census between
  runs. Concurrent runs leak AND corrupt each other's timings (T32, D07; the
  first ggml-microbench round was discarded for exactly this).
- OMP/thread-pinning env is set BEFORE process start (libgomp parses in a
  pre-main constructor; setenv from main collapses threads onto core 0).
- CROSS-WAVE GPU RULE: one GPU per box; a CPU-path wave running while a
  hardware wave holds VRAM produces false OOM failures. Check for a live
  hardware wave (`pgrep 'ft serve'` + nvidia-smi) before flagging CUDA
  failures; classify OOMs under concurrent VRAM pressure as Environment and
  re-run just those tests after the wave ends. Test-only / CPU-path fixes do
  NOT invalidate a concurrent hardware run; changes touching the
  hardware-executed path DO.

## 4. Commit discipline (ORCHESTRATION section 8)

- ONE commit per phase, created at the STOP GATE after all gates pass;
  Conventional Commits: imperative, lowercase, no trailing period; the git
  commit checkbox is part of every phase's acceptance criteria.
- Assisted-by: Veai trailer allowed; attribution follows the repo rules.
- NO git push, no gh pr create / gh pr comment / gh issue create by agents.
  dense-q80-gemm is private-use scope: no push, no upstream PR.
- Staging hygiene: stage by an EXPLICIT path list (never `git add -A` / `.` -
  the index may hold stale staged versions from parallel sessions); verify
  staged == worktree per file (`git diff --cached`); verify ZERO `.veai/memory`
  entries staged (parallel sessions churn it); re-run fast test waves right
  before committing; the commit body carries the measured results.
- Skill files (`.veai/skills/gguf-native-serving/`) are committed (6e673ab);
  further skill edits stay uncommitted until the user asks.
- Campaign override: TASK.md "Commit only when the user asks. Never push." -
  stricter than the generic one-commit-per-phase duty, so a code wave at its
  STOP GATE stops before `git commit` and surfaces the ready-to-commit state
  to the user unless the prompt already authorized the commit.

## 5. Session-end self-update (ORCHESTRATION section 11)

Trigger: at the END of every session that produced reusable knowledge, the
Orchestrator MUST run a self-update pass before closing out. Reusable
knowledge = root causes with commit ids, measured expectation classes/numbers,
new traps, protocol/hygiene refinements, stale-fact corrections.

- Where - update-in-place over duplication, cross-reference instead of
  repeating: TRAPS.md (new T-number, next free T53+, or in-place extension);
  task briefs (expected-result classes; task-06 for serving, the TASK.md
  sections here); ORCHESTRATION.md (hygiene/protocol bullets, sections 7-8).
- Consistency gates: trap / recipe counters in sync across the TRAPS.md H1,
  the SKILL.md counter line and section 10; ASCII only - no em-dashes, ASCII
  arrows; no lab-diary prose; open gaps labeled open, WITHOUT a trap number.
- Execution: delegate the integration to a Code subagent with the session-fact
  payload; for non-trivial integrations run a blocking_status review pass +
  cheap polish for non-blocking notes.
- Boundaries: the pass NEVER commits or pushes by itself (commits are an
  explicit user decision); the SKILL.md YAML preamble untouched; no README.

## 6. TRAPS digest (verbatim from TRAPS.md; English gist in brackets)

### T35 (projection without reference anchor)
> **T35** Проекшн без якоря на рабочую reference - ложный NO-GO: бумажная модель
> (research/task06-rework-estimate.md) дала "реалистично 8-11 tok/s, NO-GO",
> железо с портированными ggml-ядрами дало 16.67 (verification/re-acceptance.md).
> Каждый проекшн заякоривай на A/B рабочей реализации (рецепт D08).

[Anchor every projection on a working-reference A/B, never paper math; TASK.md:
the ~820-870 tok/s ceiling is a projection to re-anchor, never a plan.]

### T40 (vendored-code audit)
> **T40** (Phase 4, vendored-code audit) Перед переиспользованием вендоренных ядер
> проверять молчаливые bound-допущения: moe.cuh exp_idx > 255 молча дропал
> экспертов 256+ при E=288 (bound должен браться из expert-размерности
> weight-тензора); moe_q ds-load читал token_offs[threadIdx.y] OOB (фикс -> [0];
> латентный баг апстрима vLLM). Класс аудита: индексные константы, размеры
> per-thread массивов, диапазоны expert-id vs геометрия модели.

[Audit silent bound assumptions (index constants, per-thread array dims, id
ranges vs geometry) before reusing vendored kernels; campaign: pre-audit the
dense body (token_offs sizing, ds coverage, need_check math) first.]

### T43 (bogus last-chunk throughput line)
> **T43** (Phase 6, measurement) Строка "input throughput" ПОСЛЕДНЕГО полного
> чанка фиктивна (~1552-1602 tok/s при реальных ~290; tail-drain артефакт,
> воспроизводится между кампаниями): медианы префилла считать исключая chunk 1
> (warmup) И последний полный чанк.

[Medians exclude chunk 1 (warmup) AND the last full chunk - its throughput
line is a reproducible tail-drain artifact.]

### T44 (A/B validity gate)
> **T44** (Phase 6, A/B validity) BEFORE-стадия обязана воспроизвести
> историческую кампанийную базу (в пределах run-variance ~3%) - только после
> этого дельта AFTER доверийна; иначе результат неинтерпретируем.

[BEFORE must reproduce the campaign baseline within ~3% run-variance before
AFTER is trusted; campaign bar = the 556.43 tok/s @8128 MTILE=32 class.]

### T46 (kernel-level liveness proof)
> **T46** (Phase 6, liveness) Смена ядра требует kernel-level liveness proof:
> nsys kernel-name контракт + ТОЧНАЯ математика счётчиков запусков (пример:
> grouped moe ядра 42 слоя x 3 proj x 9 чанков = 1134 + moe_align 42x9 = 378;
> ноль запусков старого ядра в префилл-чанках; старое ядро присутствует в decode
> graph-replays). Bare stdlib logger'ы невидимы в boot-логах (init_logger не
> вешает хендлер; lastResort дропает INFO) - one-time INFO маркеры ловить
> PYTHONPATH sitecustomize-пробой.

[Kernel swap needs nsys kernel-name contract + exact launch-count math (zero
old-kernel launches in the tested path; decode graph-replays are legitimate);
bare stdlib loggers are boot-invisible - sitecustomize probe. Campaign: dense
~312/chunk with a shrunken per-launch grid.]

### T51 (moe.cuh x/y naming inversion + ds-fill garbage rows)
> **T51** (v3a, design) Инверсия имён в moe.cuh: x = ВЕСА (m-инвариант,
> mmq_y=32), y = АКТИВАЦИИ (y-tile = 144*mmq_x байт); бриф "tile_x растёт с
> m-tile" вполовину неверен. Гейт корректности m>4 - y-ds fill в moe_q:
> апстрим заполняет только ds-строки [0, nwarps) и читает token_offs[0],
> валидно ТОЛЬКО при mmq_x/nwarps==1; при m=8/16/32 строки >= nwarps -
> молчаливый мусор, проходящий shape-чеки. Любое расширение MOE_X требует
> порта full-coverage ds-цикла плотного mul_mat_q (mmq.cuh:77-88) с
> ids=(ids0+tid.y*QI8_1+tid.x/(32/QI8_1))%mmq_x, читающим
> sorted_token_ids[col_dst_0+ids]. Эвиденс фикса:
> .tasks/mmq-v3-stationary-moe/ (v3a-surface-map.md, аналитическая
> ds-регрессия, поймавшая revert).

[x = WEIGHTS, y = ACTIVATIONS in moe.cuh; the m>4 gate is y-ds fill coverage:
upstream fills rows [0, nwarps) and reads token_offs[0], valid ONLY at
mmq_x/nwarps==1 - rows >= nwarps are silent garbage passing shape checks; any
MOE_X widening needs the full-coverage ds-cycle (mmq.cuh:77-88). Dense side:
that loop IS the dense full-coverage ds-fill (gate inherited), but the naming
inversion applies to the lineage and the T40 audit is still required.]

### T52 (BW-bound vs ALU/LDS-bound payoff)
> **T52** (v3a, measurement) Срез трафика платит, только пока ядро не
> покинуло BW-bound режим; дальше связывает per-MAC ALU/LDS-пол, и
> дальнейшие срезы трафика платят суб-линейно. Замер на v3a m-tile sweep
> (tiles 4/8/16/32): отношения MoE-члена 0.500/0.260/0.152 против чистого
> 1/tile; маржинальный e2e-выигрыш уполовинивается на каждое удвоение
> (+31.1/+17.5/+8.6%); nsys grouped MoE - 2.52x срез времени при 15.8x
> срезе трафика на tile 16 -> вес-стационарная (v3b) трафик-редукционная
> предпосылка мертва; expert battery BITWISE идентична между тайлами
> (per-element порядок редукции tile-инвариантен). Привязывай проекции
> MoE-ядер к измеренному режиму, а не к трафик-моделям. Эвиденс:
> .tasks/mmq-v3-stationary-moe/v3a-ab.md.

[Traffic cuts pay sub-linearly once the kernel leaves BW-bound (per-MAC
ALU/LDS floor; marginal e2e gain halved per tile doubling; battery BITWISE
tile-invariant). Pin projections to the MEASURED regime - hence Step 0 first.]

### D06 (standalone CPU microbench reference gate)
> **D06** Standalone CPU-микробенч ядер reference: компилируй СОБСТВЕННЫЕ .c файлы
> reference standalone (include paths only) с ТЕМИ ЖЕ ISA-флагами, что и
> референс-сборка (build/CMakeCache.txt + per-variant flags.make; haswell-ярус =
> -O3 -mf16c -mfma -mavx -mavx2 -fopenmp); линк-шимы только для недостижимых
> символов ggml.c; блоки - синтетические byte-exact, f16 d ограничен finite
> normals; парити SIMD-vs-scalar на случайных строках (max rel err < 1e-3) на
> КАЖДОМ прогоне ДО таймингов; blended + per-format прогоны, свип тредов;
> OMP_PLACES/OMP_PROC_BIND выставлять ДО старта процесса (libgomp читает их в
> pre-main конструкторе; setenv из main молча сваливает все треды на ядро 0);
> warmup + автокалибровка >= 3 s на точку. Gate: sustained GB/s >= эффективной
> PCIe gather полосы. Эвиденс: .tasks/gguf-glm5next-hybrid/verification/
> ggml-microbench.md + ggml-microbench-harness.cpp.

[Same-ISA-flags standalone reference build; byte-exact synthetic blocks; parity
max rel err < 1e-3 on random rows EVERY run BEFORE timings; OMP env pre-start;
warmup + calibration >= 3 s/point. Gate: sustained GB/s >= effective PCIe
gather. Campaign: BEFORE any Candidate B port, clearing the MMQ baseline.]

### D09 (nsys live-serve profiling recipe)
> **D09** nsys live-serve профилирование: nsys launch отвергает -o; рабочая
> схема = launch + start-after-ready + --cuda-graph-trace=node; голый mid-run
> start даёт отчёт с API/graph-записями, но БЕЗ eager kernel activities;
> анализ через sqlite export; окно брэкетится по строкам "input throughput";
> overhead ~nil (чанки внутри/вне окна идентичны).

[nsys launch rejects -o; scheme = launch + start-after-ready +
--cuda-graph-trace=node; bare mid-run start lacks eager kernel activities;
sqlite export; window bracketed by "input throughput" lines; overhead ~nil.]

### D10 (Step-0 split before kernel-swap design)
> **D10** Step-0 split перед design kernel-swap: один nsys-проход + sqlite
> export фиксирует раскол GEMM/copies/rest с kernel-name учётом на чанк
> (пример: moe_vec_q 126/чанк = 42 слоя x 3 proj; fast_index_copy_multi
> 42/чанк); GPU idle ~0.05% = нет CPU-starvation; замер ДО проектирования
> перекроил список бутылочных горлышек (dense q8_0 GEMM стал top "rest",
> а не fetch copies).

[Step-0 nsys + sqlite split with per-kernel launch counts pins the regime
BEFORE design; it re-ranked bottlenecks and spawned this campaign; re-run
under MTILE=32 on the 14.61 s chunk.]

### Cross-references kept from TRAPS.md (abbreviated, still binding)
- T32/D07: serialized measurement runs - one process at a time, hard timeout,
  PID-reap + ps census between points (binding via section 3).
- T31/T36: benchbw full-profile clobber, no TTL/kernel-tier fingerprint - any
  bench-adjacent change needs a FULL rerun of all formats + boot-fraction check.
- T39: expert-sort reorder = throughput no-op for streaming GEMV/MoE; do not
  chase cache-locality reorderings in this campaign.
- T49: full-gate NaN/Environment failure passing isolated with and without
  the diff = suite-ordering Environment, not a regression (isolated A/B).
- T50: slot-floor fail-fast under desktop VRAM tax (~492 MiB); ladder
  0.89 -> 0.90 -> 0.85+mr1 for 8191 - applies to boot probes.
- T45/T28: env kill switch removed post-validation (7f8c570); KDA autotune
  ~256 MiB do_bench scratch can OOM the eager tail (8191 needs 0.85+mr1) -
  cited by TASK.md.
