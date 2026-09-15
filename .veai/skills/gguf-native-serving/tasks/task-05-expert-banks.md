# Task 05: Expert banks -> offload integration

Type: Code | Agent: Code | Duration: ~1-2 days

## Goal

Route stacked expert banks through the offload path: pinned HostBanks, per-signature
partitions, prefill-overlap degrade, #186 clamp, benchbw profile.

## Instructions for the subagent

1. Bank loading: load_gguf_expert_sources resolver (weight.py pattern) ->
   per-layer HostBanks ((E, rows, row_bytes) per role) + LayerCompletionTracker +
   PinPipeline. Per-role geometry (down rows = H, gate/up rows = I).
2. PIN-COUNT TRAP: LayerCompletionTracker.expected_per_layer MUST equal the
   per-layer NOTE count (the gguf fill loop calls tracker.note() ONCE per layer
   when all three banks land together; gemma4 calls it 2 per layer, one per write).
   A mismatch silently skips ALL pinning -> CUDA IMA at the first decode copy.
   Add a construction-time guard in _build_fused_copy_plan rejecting pageable
   CPU banks.
3. Per-signature partitions: group layers by (shape, dtype) role-ordered signature;
   one OffloadMoeCache per group; byte-proportional budget split with E floors;
   local layer-id remap (proven by id()-identity test); graph.py resets every
   partition. A uniform file degenerates to one group = today's behavior.
4. Prefill-overlap degrade: overlap only when the group's slots >= 2*num_experts;
   minorities get a loud info log + synchronous materialized prefill.
   Do NOT floor at 2E (too expensive for wide minority signatures).
5. #186 clamp: engine auto-clamps max_extend_tokens to 65535 // top_k.
6. benchbw: add the profile row (geometry + activation) + _offload_bank_specs
   widths; exclude from the auto->hybrid upgrade.
7. Load-time validation: bank ggml types against the kernel-supported MMVQ set;
   loud ValueError naming tensor + type + supported set.
8. Aggregated readouts: scheduler/cache_status over the partition list (dominant-
   only fallback ONLY where a single value is structurally required).

## Traps (from [TRAPS.md](../TRAPS.md))

- T21: The pin-count trap is the WORST debugging profile - the load succeeds
  (minutes), then IMA at the first decode copy. CPU tests CANNOT catch it
  (pageable-vs-pinned only exists on a CUDA device). The construction-time
  guard is the only effective defense.
- T22: The cross-group prefetch hop may be a no-op (all partitions degraded ->
  no boundary choreography runs). Evaluate honestly; a characterization test
  pinning the contract is sufficient if the hop is dead code.
- T23: The per-role EXTREME sizing overcounts the aggregate (+24.7 GiB for
  3 signatures). The exact sizing (header-only scan) eliminates this.
- T24: set_bank_sources previously asserted uniform shapes; with partitions,
  the per-signature grouping replaces that assert - keep a loud error only for
  genuinely broken shapes.

## Acceptance criteria

- [ ] I've created a git commit for this task
- Runs under ORCHESTRATION.md (quality loop + hygiene + commit gate).
