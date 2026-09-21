# Task 04: Engine gate relaxation (gguf + hybrid) - W3

Type: implementation (engine + tests)
Suggested agent: Code (balanced)
Spec coverage: R4, S4. Depends on Task 01 (executor capability set) and
Task 02 (shares engine.py with the partition wiring - run AFTER Task 02 to
avoid conflicts).

## Goal

_adjust_config accepts gguf + hybrid when ALL the file's per-layer bank types
are CPU-executable; still rejects gguf + cpu and gguf + fused; the #186 clamp
(max_extend_tokens 8191 at top_k=8) remains untouched and strategy-independent.

## Why This Task Exists

The offload-first campaign added a feature gate (not a bug) rejecting
moe_weight_format == "gguf" with strategy cpu/hybrid/fused. With Tasks 01-02
landed the rejection is stale for hybrid and blocks every real-file boot.

## Required Inputs (read first)

1. .tasks/gguf-glm5next-hybrid/research/hybrid-machinery-mapping.md - section 2
   (gate map incl. what exists at gate time) and section 5 (tests to flip).
2. .tasks/gguf-glm5next-hybrid/TASK.md - W3 section.
3. The landed executor capability set from Task 01 (_WFMT_IDS).

## Files/Areas (verified line refs)

- python/freetoken/engine.py: _adjust_config :1594 (called at :362); gguf
  rejection ~:1762-1777 (message anchor :1775, strategy set cpu/hybrid/fused +
  moe_cpu_layers). At gate time banks/gguf_types are NOT loaded yet
  (load_expert_banks runs later ~:791); available: config (incl. model_path,
  config.py:21) + model_config. #186 clamp ~:1779-1796 (guard is only
  moe_weight_format=="gguf"; 65536//top_k -> 8191 at top_k 8) - DO NOT TOUCH.
- Header-scan precedent: bank_bytes_estimate_gguf header-only scan
  (_gguf_bank_bytes) - reuse the pattern to read expert tensor gguf types from
  the file header without loading banks.

## Mechanism (decided by discovery)

Scan the gguf header for the per-layer expert tensor types (gate/up/down per
layer) and accept hybrid iff EVERY type across all layers is in the executor
capability set (static query on the Task-01 _WFMT_IDS). Keep cpu and fused
rejections: cpu has no CPU-resident gguf banks (the offload cache owns them);
fused needs packed resident weights, gguf banks are not resident bf16.

Triage delta (2026-09-15): _cpu_moe_executor_viable (engine.py:1435-1450) already returns True for the "gguf" alias and must become gguf-TYPE-aware (per-layer header types vs executor capability) when the gate lifts - fold this into the viability rework.

Task 03 review delta (2026-09-15): there is a SECOND gate site - the auto->hybrid backend upgrade in engine.py (~:1817) excludes gguf by name ("bench_fmt != 'gguf'"). When the gate lifts, decide explicitly: minimal path = keep that exclusion (the sanctioned entry for gguf is the explicit --moe-strategy hybrid flag, not the bench auto-upgrade) and document it; do not silently leave it half-wired.

## Constraints / Non-Goals

- No behavior change for non-gguf models; no change to offload-only gguf path.
- Do not regress the #186 clamp; do not touch prefill-overlap degrade.
- Message quality matters: the incapable-type rejection must name the type(s).
- One change per commit; conventional commits; Assisted-by: Veai. No push.

## What to Do

1. Read the inputs; verify cited lines (drift possible after Tasks 01-02).
2. Implement the capability check + header scan; relax hybrid only.
3. Tests (tests/engine/test_cache_budget.py):
   - :508 test_adjust_config_rejects_gguf_experts_on_cpu_paths (raises :544/
     :548): flip the hybrid case to acceptance-for-capable-types; keep cpu and
     fused rejection cases.
   - :556 test_adjust_config_auto_keeps_gguf_on_offload_despite_hybrid_profile:
     verify still green (offload auto-keep must survive).
   - New: hybrid accepted when the header types are all CPU-capable; hybrid
     still rejected when at least one type is incapable (message names it).
4. Run tests/engine (record exact output).
5. Commit (conventional, Assisted-by: Veai).

## Expected Output (report_result)

- Exact changed files; test commands + exact outputs; the new gate semantics
  in one paragraph; blockers.

## Acceptance Criteria

- [ ] Gate accepts gguf+hybrid for an all-capable type set (test green)
- [ ] Gate still rejects gguf+cpu and gguf+fused (tests green)
- [ ] Gate rejects hybrid when any type is incapable, naming the type
- [ ] tests/engine green; #186 clamp untouched
- [ ] I've created a git commit for this task
