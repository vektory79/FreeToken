# Task 07: Post-Task-02 hardening ride-along (watchdog attribution, affinity clamp, CLI help, gap tests)

Type: implementation (small hardening + tests)
Suggested agent: Code (balanced)
Trigger: run AFTER Task 04 (shares engine.py) and BEFORE Task 05. One commit.

## Goal

Close the Task 02 review findings (2x GO passes, 2026-09-15) that are real but
were deferred to avoid engine.py conflicts with Task 04:
- B-1 WeakWarning: watchdog error/log lines lack pool/partition attribution
  (cpu_executor.py:888-896, engine.py:1261 walk) - (layer_id, bs) slot ids are
  ambiguous with 3 pools; add pool/partition identity to the surfaced error.
- B-2 WeakWarning: explicit --moe-cpu-threads > total cores wraps modulo the
  WHOLE core order in resolve_pool_affinities (cpu_executor.py:256-261), so
  later pools overlap earlier pools' cores, violating the documented DISJOINT
  invariant (perf-only, extreme misconfig) - clamp/resolve deterministically
  and keep the documented overspend note truthful.
- A-Info defensive clamp: resolve_threads_and_affinity(0, subset) with a
  physical-core-free subset returns nthreads=0 with non-empty core ids
  (cpu_executor.py:229-231) - unreachable via engine; add the cheap clamp.
- B-3 WeakWarning: CLI help text (args.py:649-655, config.py:49-51) does not
  reflect total-budget-split-across-pools semantics; the pre-existing
  "Ignored by other backends" note is now more wrong than before - make the
  help truthful (one or two lines, no comment blocks).
- B-5 test gaps: (a) incapable partition positioned mid/last in the group
  list; (b) multi-pool AUTO-mode executor on CUDA (per-pool coordinator
  carve-out with allow_cores is unexercised); (c) _HYBRID_OVERLAP=0
  multi-partition case. Extend tests/moe/test_hybrid_fetch.py and/or
  tests/models/test_gguf_expert_banks.py following existing patterns.
- B-6 Info (optional, cosmetic): duplicate sys.path.insert/imports in the
  test wrapper (tests/moe/test_hybrid_fetch.py:363-367).

## Additions from the Task 04 review (2026-09-15, both GO-pass triaged TP-minor)

- Issue 1 (Test): no engine test pins the lifted half "offload + explicit
  --moe-cpu-layers <ids> on gguf" - capable ids must boot, incapable ids must
  die at the partition screen (multi-group) or eagerly in CpuMoeExecutor
  construction (_split_gguf_formats), both strictly before CUDA graph capture.
  Add to tests/engine/test_cache_budget.py (lines ~610-745 area), mirroring
  the real mini-gguf fixture pattern the Task 04 tests use.
- Issue 2 (Maintainability): stale loud-failure advice in the newly reachable
  flow - "use --moe-strategy offload instead" is printed when the strategy
  already IS offload (cpu_executor.py:119-123, engine.py:907-909; pre-existing
  Task 02 text). Fix the advice: when strategy is offload, advise adjusting or
  dropping --moe-cpu-layers instead.

## Constraints / Non-Goals

- No behavior change for default/sane configs; offload-only and single-cache
  paths byte-identical.
- No gate logic changes (Task 04 owns them).
- One commit bundling this hardening wave (conventional, Assisted-by: Veai,
  no push). Stage only own files; never git add -A; never stash/checkout.

## Acceptance Criteria

- [ ] Watchdog errors carry pool/partition attribution
- [ ] Affinity disjointness holds for all requested thread counts (clamped)
- [ ] CLI help text truthful for the split semantics
- [ ] The three gap tests added and green
- [ ] tests/moe + tests/models/test_gguf_expert_banks.py + tests/engine green
      (known pre-existing test_offload graph-capture failure not counted)
- [ ] I've created a git commit for this task
