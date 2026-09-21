# Research: hybrid MoE machinery mapping for glm5next GGUF (W2/W3/W4)

Read-only code survey, 2026-09-14. Branch vektory79, offload-only GGUF campaign committed
(028f2d9 + 78115df). All line numbers verified this session via IDE search/reads.

## 1. Hybrid machinery (nvfp4 today)

Decode dispatch - `python/freetoken/layers/moe.py`, class `OffloadMoELayer`:

- `_decode_routed` (~:263): decision tree -
  `if cache.is_cpu_layer(self.layer_id): return executor.decode(...)` ->
  `if cache.decode_target == "hybrid": return self._decode_hybrid(cache, ...)` ->
  else plain offload: `cache.ensure_experts(layer_id, topk_ids)` + `cache.copy_missing()` +
  `self._expert_gemm(...)`.
- `_decode_hybrid` (:292, search-verified): overlap choreography, in order:
  1. `raw = topk_ids.clone()` (raw expert ids for the CPU partial)
  2. `cache.ensure_experts_hybrid(self.layer_id, topk_ids)` - capped LRU fetch; rewrites
     ids to slot ids for hits/fetches, `-1` for overflow misses (CPU-owned)
  3. `pending = executor.decode_submit(self.layer_id, hidden_states, topk_weights, cpu_ids)`
     - CPU GEMV is kicked off BEFORE the PCIe fetch + GPU GEMM
  4. `_HYBRID_OVERLAP` toggle: `_HYBRID_OVERLAP = os.getenv("FREETOKEN_HYBRID_OVERLAP", "1") != "0"`
     (:26) - with `=0` the CPU pool is synced early (serialized, A/B measurement mode)
  5. `cache.copy_missing()` (capped PCIe copies) -> `gpu_routed = self._expert_gemm(...)` on the
     GPU partial (zero weights on CPU routes, slots clamped)
  6. `cpu_routed = executor.decode_sync(pending)` -> `return gpu_routed + cpu_routed`
  Capture-safety contract: routing split is device-side elementwise; submit/sync are host nodes.

LRU residency + capped fetch - `python/freetoken/moe/offload_cache.py`, `OffloadMoeCache`:

- `ensure_experts` :888 (plain LRU ensure, offload path), delegates to
  `freetoken.moe.offload_kernels.ensure_experts`.
- `ensure_experts_hybrid` :900-920: "Capped-fetch LRU for the hybrid backend ... assigns slots
  to (and schedules copies for) at most `hybrid_max_fetch` -- or
  `~hybrid_fetch_fraction * misses` when the fraction is set"; delegates to
  `offload_kernels.ensure_experts_hybrid(self, layer_id, expert_ids, self.hybrid_max_fetch,
  self.hybrid_fetch_fraction)`. `num_indices` = capped fetch count (drives `copy_missing`),
  `num_missing_full` = pre-cap miss count (drives stats).
- `copy_missing` :1056 executes the scheduled PCIe copies (fast_index_copy_multi_jit
  zero-copy over pinned banks).
- Per-cache hybrid fields (:138-157): `decode_target` ("gpu"/"cpu"/"hybrid"),
  `hybrid_max_fetch: int = 1` (:148), `hybrid_fetch_fraction: float = 0.0`,
  plus `gguf_types: tuple | None` (:159-161): "per-layer (gate, up, down) ggml type ints -
  the kernel dispatch reads the layer's own types".
- Executor slot: `set_cpu_executor(executor)` :600.

CPU executor - `python/freetoken/moe/cpu_executor.py`, `CpuMoeExecutor`:

- OWNERSHIP: one executor per ENGINE, bound to ONE cache. `engine.py:897-899`:
  `if caches[0].decode_target in ("cpu", "hybrid"): self._init_cpu_moe_executor(config,
  caches[0], layers)` - i.e. only the FIRST cache gets one today, and `engine.py:881-885`
  rejects the combination outright for partitions:
  `"multi-signature expert banks serve the gpu decode target only; cpu/hybrid executors are
  per-format and cannot span partitions"` (anchor :884). This ValueError is the structural
  blocker W4 must lift/replace.
- `_init_cpu_moe_executor` (engine.py:941-976) constructs:
  `CpuMoeExecutor(cache, top_k=sample.top_k, activation=sample.activation,
  apply_router_weight_on_input=..., num_threads=config.moe_cpu_threads, max_tokens=max(...),
  device=self.device, swiglu_alpha=..., swiglu_limit=..., fmt=sample.quant_method.cpu_format
  if sample.quant_method is not None else None)` then `cache.set_cpu_executor(executor)` and
  `self.cpu_moe_executor = executor`. `num_threads` comes straight from `--moe-cpu-threads`
  (0 = auto: `resolve_threads_and_affinity` :119-141, one worker per physical core + pinning;
  under auto sizing the flag-handshake coordinator reserves the last physical core :219-224).
- Format support: `_WFMT_IDS` :72 = `{"bf16": 0, "nvfp4": 1, "mxfp4_triton": 2, "ds_fp4": 3,
  "q4_0": 4}` ("Weight-format ids must match WFmt in csrc/cpu_moe/cpu_moe_ext.cpp"). Unknown
  fmt hard-fails in `__init__` (:165-170): "...supports [...] formats, but this checkpoint's
  experts are {fmt!r}; use --moe-strategy offload (GPU-side dequant) instead."
  `fmt = fmt or cache.quant_format` (:163); banks resolved via
  `_resolve_banks({canonical_role(name): per_layer for name, per_layer in
  cache.bank_sources.items()}, fmt)` (:200-203) - a SINGLE `fmt` for the whole cache, no
  per-layer type awareness (per-projection gguf mix is W1; W4 must carry per-partition types).

## 2. Engine gate

`python/freetoken/engine/engine.py`, module function `_adjust_config` :1594 (call site :362).

- The GGUF rejection (~:1762-1777; message text anchor :1775):
  ```python
  if (
      is_moe
      and getattr(model_config, "moe_weight_format", None) == "gguf"
      and (
          config.moe_strategy in ("cpu", "hybrid", "fused") or config.moe_cpu_layers
      )
  ):
      asked = (...)
      raise ValueError(
          f"{asked}: gguf moe_weight_format supports offload only; drop the flag "
          "and let every layer decode on the GPU offload path instead."
      )
  ```
  (Immediately above it sits the activation gate for cpu/hybrid, `_cpu_moe_act_ok`,
  :1743-1758 - a second gate glm5next's `swiglu_clamp` already passes since act id 4 exists.)
- The #186 clamp (~:1779-1796) is STRATEGY-INDEPENDENT - verified: its guard is only
  `if is_moe and getattr(model_config, "moe_weight_format", None) == "gguf":` with no
  strategy term; `top_k = num_experts_per_tok`; `if top_k > 0 and config.max_extend_tokens >
  _MOE_VEC_MAX_GRID // top_k:` clamps `max_extend_tokens` to 65535//8 = 8191 and logs.
  W3 keeps it untouched.
- Data available at the gate: `config` (EngineConfig - includes `model_path`, engine/config.py:21)
  and `model_config` (moe_weight_format, hidden_act, is_moe, expert_quant, num_experts_per_tok).
  NOT available: the expert banks / per-layer gguf_types - those are materialized later in
  `_init_offload_moe_cache` via `load_expert_banks(config.model_path, ...)` (engine.py ~:791).
  So the gate cannot see per-layer bank types today. Options for the W3 relaxation:
  (a) query a static capability surface on the CPU executor (e.g. `_WFMT_IDS` / a new
  `supported_formats()` / resolver `can_handle(types)` that W1 adds) - sufficient for the
  file-level "gguf supported / not" decision;
  (b) scan the GGUF header at `config.model_path` for the real per-layer types - precedent
  exists (`bank_bytes_estimate_gguf` exact-header-scan path in expert_banks.py; test
  `test_bank_bytes_estimate_gguf_exact_header_scan`). Recommend (a); keep rejecting
  `cpu` / `fused` / `moe_cpu_layers` for gguf regardless.

## 3. Partition wiring

All in `engine.py` `_init_offload_moe_cache` (~:775-912):

- Grouping/sizing: `groups = self._group_bank_layers(banks, num_moe_layers)` (:826); single
  group degenerates to one cache with the full `moe_cache_size`; multi-group splits via
  `_split_moe_cache_budget` (byte-cap aware, :927-955 for the rebuild variant).
- Creation loop :829-879: `for members, size_g in zip(groups, sizes):` builds one
  `OffloadMoeCache` per signature group with `num_layers=len(members)`, `cache_size=size_g`,
  `prefill_overlap=overlap_g` where `overlap_g = self._partition_prefill_overlap(
  config.moe_prefill_overlap, size_g, num_experts)` (:833-834) -
  `_partition_prefill_overlap` :540-548: `return bool(wants_overlap and size_g >= 2 *
  num_experts)` (the degrade rule; "Mirrors rebuild()'s degrade at offload_cache"). Also per
  cache: `quant_format=banks.quant_format`, `gguf_types=tuple(banks.gguf_types[l] for l in
  members)` (LOCAL layer order), `decode_target`, `hybrid_max_fetch`, `layout`, `max_slots`;
  then `cache.cpu_layer_ids = {members.index(g) ...}` (local remap),
  `cache.set_bank_sources(sub_sources, layer_residency=...)`, `cache.set_alphas(...)` via
  `_slice_alphas` :559-570, and - KEY for hybrid - `if decode_target == "hybrid":
  self._resolve_hybrid_fetch(config, cache)` :873-874, i.e. the auto fetch fraction is
  already resolved PER PARTITION.
- Attach: `_route_offload_layers` :550-557 - VERIFIED:
  `for layer in layers: cache, local = routing[layer.layer_id]; layer.offload_cache = cache;
  layer.layer_id = local` ("behavioral contract pinned by
  test_engine_partitions_attach_routing").
- Legacy readers: dominant partition = `max(caches, key=lambda c: c.cache_size *
  expert_bytes_per_slot(c.bank_sources))` (:908-910) backs `ctx.moe_offload_cache` /
  `self.moe_offload_cache`; the full list is `self.moe_offload_caches`.
- Per-partition state that exists: slot pool + LRU (`slot_for_id`, `id_of_slot`,
  `expert_recency`), prefill double buffers gated by the per-partition `prefill_overlap`,
  stats counters, `hybrid_max_fetch` + `hybrid_fetch_fraction`, `cpu_layer_ids`, sliced
  alphas, per-projection `gguf_types`.
- What a per-cache CPU executor needs (W4): exactly today's `_init_cpu_moe_executor` argument
  set, instantiated per cache - `cache` itself (bank_sources + quant_format/fmt + gguf_types
  for the per-layer type table), `top_k`/`activation`/`apply_router_weight_on_input`/`alpha`/
  `limit` (identical across layers - any member works), `num_threads=config.moe_cpu_threads`,
  `max_tokens`, `device`. `_decode_hybrid` already resolves the executor via
  `cache.cpu_executor`, so per-cache attach falls out naturally once `set_cpu_executor` is
  called per cache. Open engineering points: (1) thread budget must be partitioned across 3
  pools (3 x N workers + coordinator cores must not collide; `resolve_threads_and_affinity`
  pins globally today); (2) `self.cpu_moe_executor` single-attribute consumers (status views,
  watchdog) need a list or a dominant-cache pointer; (3) CUDA-graph capture binds host-func
  task pointers per executor - the per-cache instances must exist before capture (same
  constraint that forces `_init_cpu_moe_executor` to run pre-capture today).
- Real-file signature groups (ground truth from the campaign): (18,18,23)x39,
  (18,18,14)x2, (23,23,14)x1 -> 3 caches; the minorities' 2E prefill-overlap floor is
  infeasible at big KV, so their `prefill_overlap` degrades to False via the rule above.

## 4. Benchbw

Module: `python/freetoken/moe/benchbw.py` (1009 lines; invoked via `ft bench bw`).

- `_CPU_MOE_FORMATS` :66 = `frozenset({"bf16", "nvfp4", "mxfp4_triton", "ds_fp4"})`
  (note: `q4_0` is absent too - the executor serves it, the bench does not).
- The "gguf" profile row: `DTYPE_WORKLOADS["gguf"]` :146-150 (anchor :148):
  `"gguf": Workload("dtype:gguf", 4096, 2048, 288, 8, ("gguf",), activation="swiglu_clamp",
  swiglu_alpha=1.0, swiglu_limit=10.0)`; comment :146-147 states the current contract:
  "CPU MoE has no gguf weight path, so _bench_format notes it and the verdict is always
  offload".
- `_offload_bank_specs` :289-332; the gguf branch :316-327: gate/up =
  `I * (H // 256) * 98` u8, down = `H * (I // 256) * 210` u8, with the comment tying the
  widths to `offload_cache._BANK_BYTES_PER_EXPERT["gguf"]` ("the per-layer type mix does not
  change the gathered bytes").
- CPU-MoE auto-skip + verdict: `_bench_format` :622-676. :648-650:
  `if fmt not in _CPU_MOE_FORMATS: _note(entry, f"CPU MoE has no {fmt} weight path; hybrid
  unavailable")` -> `cpu_moe_gbs` stays None -> the else branch :672-675 fires:
  `entry["recommended"] = "offload"` ("No CPU-vs-PCIe pair to compare ... offload is the
  safe call absent evidence for hybrid"). That is the entire gguf verdict story today.
- CPU timing leg -> fetch fraction: when both legs measured (:653-668):
  `ratio = cpu_g / pcie_g`, `recommended = recommend(cpu_g, pcie_g, threshold)` (:610,
  threshold default 2.0 - hybrid iff CPU bandwidth clears PCIe by the threshold; verify the
  exact comparison in the body at implementation time), then the contended pair via
  `measure_overlap_bw` (:550) fills `cpu_moe_overlap_gbs` / `pcie_gather_overlap_gbs` with
  the comment "This pair sets the hybrid backend's fetch split (load_hybrid_fetch_fraction)".
- Consumption chain: `bench_profile.load_hybrid_fetch_fraction` :159 prefers the
  `dtype_kernels[fmt]` overlapped pair (`pcie_ov / (pcie_ov + cpu_ov)`), falls back to
  standalone `pcie/cpu`, then per-model workloads; None -> fixed cap 1 + warning. Engine
  `_resolve_hybrid_fetch` (engine.py ~:920-939): explicit `--moe-hybrid-max-fetch >= 0`
  wins; auto sets `cache.hybrid_max_fetch = cache.num_experts` (inert) and
  `cache.hybrid_fetch_fraction = fraction` - and it is called per partition (:873-874), so
  every partition picks up the gguf fraction automatically once the profile row has it.
- To emit "hybrid" for gguf (W2): add the gguf types to `_CPU_MOE_FORMATS` (or a per-type
  capability mechanism) AND give the CPU bench leg a gguf bank source
  (`_cpu_moe_bank_sources` :334+ builds executor-layout synthetic banks; needs a gguf
  layout) - after that, `measure_cpu_moe_bw` / ratio / overlap legs run unchanged.

## 5. Tests

- (a) Engine gate: `tests/engine/test_cache_budget.py:508
test_adjust_config_rejects_gguf_experts_on_cpu_paths` - `pytest.raises(ValueError, match=
  "gguf moe_weight_format supports offload only")` for `moe_strategy="cpu"` (:544) and
  `"fused"` (:548), offload passes (:550-552). This is the test to flip for W3: hybrid must
  become accepted (when the executor supports the file's types) while cpu/fused keep
  failing. Also `:556 test_adjust_config_auto_keeps_gguf_on_offload_despite_hybrid_profile`
  - pins that the auto->hybrid profile upgrade IGNORES gguf rows; revisit if auto may
  upgrade gguf after W2 (decide explicitly). Related: `tests/moe/test_offload.py:449`
  ("Family, not member: a box with a benchbw profile resolves bf16 experts to hybrid").
- (b) Benchbw / hybrid fetch: `tests/moe/test_hybrid_fetch.py` (182 lines, 6 tests):
  `test_balanced_fetch_tracks_fraction` :28, `test_load_hybrid_fetch_fraction` :43,
  `test_profile_lookup_prefers_the_gpu_uuid_file` :69,
  `test_hybrid_fraction_gpu_matches_cpu_reference` :89 (CUDA skipif),
  `test_hybrid_fixed_cap_unchanged` :119 (CUDA skipif),
  `test_benchbw_gguf_profile_sanity` :131 - HARD-PINS the offload verdict
  (`assert "gguf" not in benchbw._CPU_MOE_FORMATS` :170,
  `entry["recommended"] == "offload"` :178, note text :181): this is the W2 test to flip.
- (c) Offload partitions (NOTE: MoE partition tests live in tests/models + tests/moe, NOT
  tests/kvcache - that directory is the KV pools): `tests/models/test_gguf_expert_banks.py`
  (1261 lines): `test_engine_partitions_degenerate_single_signature` :529,
  `test_engine_partitions_three_groups_end_to_end` :546,
  `test_partition_prefill_overlap_degrade_rule` :645,
  `test_prefill_choreography_group_boundary_no_hop_contract` :660,
  `test_engine_partitions_attach_routing` :744, `test_engine_partitions_real_model_routing`
  :799, plus `test_moe_cache_budget_split_rule` :481 and
  `test_set_bank_sources_rejects_heterogeneous_widths` :458.
- (d) Hybrid nvfp4 patterns to mirror: `tests/moe/test_hybrid_fetch.py` (split math +
  profile reader + GPU-kernel-vs-CPU-reference parity via a CPU ids tensor driving
  `ensure_experts_hybrid` on a reference cache); `tests/moe/test_cpu_moe.py` +
  `tests/moe/test_cpu_moe_q4_0.py` (executor construction + the q4_0 resolver precedent W1
  mirrors); `tests/moe/test_offload.py` (offload family gating). GGUF dequant parity
  tolerance model: `tests/kernels/test_gguf_quant.py` (per-element bound
  8*2^-11*factor*|d| + 1e-3).
- Suites/markers (tests/README.md): `uv run pytest tests/` full ~2-4 min;
  `uv run pytest tests/ -m "not slow"` "skip[s] the handful of tens-of-seconds tests"
  (README:43-45); GPU tests self-skip via skipif when CUDA is absent; marlin nvfp4 tests
  need `vllm` importable. The task-spec suite sizes (tests/engine 146,
  test_gguf_expert_banks.py 25, tests/kvcache 19, tests/kernels 21-23,
  test_glm5_next_gguf.py 58-59) match the prior campaign's collection counts;
  test_gguf_expert_banks.py now carries 30 `def test_` (grew during the offload campaign) -
  re-check with `pytest --collect-only` at run time.

## Delta map (minimal touch points per work item)

### W2 - benchbw CPU-MoE leg for gguf
- `python/freetoken/moe/benchbw.py:66` - extend `_CPU_MOE_FORMATS` (or add a per-type
  capability hook) so `_bench_format` stops auto-skipping (:648-650).
- `python/freetoken/moe/benchbw.py:334+` - gguf layout in `_cpu_moe_bank_sources` +
  `measure_cpu_moe_bw` support (synthetic gguf banks) - feeds ratio/verdict/overlap legs.
- Profile row :146-150 and `_offload_bank_specs` :316-327 stay as-is (geometry already
  correct).
- Flip `tests/moe/test_hybrid_fetch.py:131 test_benchbw_gguf_profile_sanity`: verdict
  "hybrid", sane fraction (0 < frac < 1), cpu_moe_gbs populated; keep the geometry asserts.
- No engine change needed: `load_hybrid_fetch_fraction` :159 + per-partition
  `_resolve_hybrid_fetch` pick the new row up automatically.

### W3 - engine gate relaxation
- `python/freetoken/engine/engine.py` ~:1762-1777 - split the rejection: hybrid allowed for
  gguf when the CPU-executor capability query (added by W1, e.g. over `_WFMT_IDS` /
  resolvers) supports all the file's bank types; keep rejecting `cpu`, `fused`, and
  `moe_cpu_layers`. `config.model_path` is available if a header scan is preferred.
- Keep the #186 clamp ~:1779-1796 untouched (verified strategy-independent).
- Update `tests/engine/test_cache_budget.py:508` (hybrid now passes for gguf + executor
  support; cpu/fused still raise); decide the auto-upgrade question at :556.

### W4 - partition-aware hybrid
- `python/freetoken/engine/engine.py:881-885` - lift/replace the multi-partition
  cpu/hybrid `ValueError` once per-cache executors exist.
- `engine.py:897-899` + `_init_cpu_moe_executor` :941-976 - construct one
  `CpuMoeExecutor` PER CACHE inside the partition loop (:829-879): per-partition thread
  split of `--moe-cpu-threads`, per-cache `fmt`/`gguf_types` (already stored locally per
  cache), `cache.set_cpu_executor` per cache; keep `self.cpu_moe_executor` semantics
  (dominant or list) for status/watchdog readers.
- `python/freetoken/layers/moe.py` `_decode_hybrid` :292 - NO change (resolves via
  `cache.cpu_executor`); re-audit the LayerCompletionTracker note-count contract if hybrid
  adds a new bank source path (campaign lesson 1).
- Per-partition `_resolve_hybrid_fetch` (:873-874) already runs per cache - no change.
- New risk surface: thread/affinity budgeting across 3 pools (coordinator cores),
  pre-capture construction ordering, watchdog per executor.
