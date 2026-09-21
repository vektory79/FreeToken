# Phase 5 acceptance boot smoke - REAL file - 2026-09-14 00:05-00:08 - FAIL

Verdict: FAIL at boot (step 2/8). Prefill and decode probes never ran. No code was
changed; no retry with alternate flags (per acceptance rules). Evidence preserved.

## Serve command (exact, second and only load attempt)

    ft serve --model-path /media/ai/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL.gguf --moe-strategy offload --moe-cache-auto --port 18081

(.venv/bin/ft, freetoken 0.1.2. NOTE: positional model path is NOT accepted by
`ft serve` - first attempt died at argparse in 5s: "the following arguments are
required: --model-path/--model". Harness-only issue.)

Runner: /tmp/ft_boot_smoke.sh + /tmp/ft_smoke_client.py (ephemeral watchdog recipe:
FAILPAT grep every 5s, kill -9 tree on hit, trap on EXIT). Server log:
/tmp/ft_smoke_server.log, archived at verification/boot-smoke-server-log-2026-09-14.txt.

## Step verdicts

1. Preflight: PASS - file 147,535,921,568 B present; GPU idle (1948 MiB desktop
   baseline, no compute procs); no ft processes.
2. Boot + watchdog: FAIL - AssertionError killed the scheduler at 131 s wall
   (00:05:37 start -> 00:07:35 death). /ready stayed 503. Watchdog fired correctly,
   printed FAILPAT lines, kill -9 tree, non-zero exit.
3. Boot markers / /v1/models: PARTIAL (markers below appeared; /v1/models never
   reached). Partition/slot-floor lines never emitted (cache init died first).
4. Prefill probe: NOT RUN. 5. Decode probe (cached): NOT RUN.
6. Decode probe (fresh): NOT RUN. 7. Shutdown: N/A - watchdog kill -9 (asserted
   process was already dead). 8. Cleanup verified: NO_FT_PROCESSES; nvidia-smi
   compute apps back to desktop-only (hoptodesk C+G), VRAM 1948/32607 MiB; RAM
   freed (11 GiB used of 188); sem.mp count 18 (no pre-run baseline taken).

## Boot-log markers that DID appear (verbatim, timestamps stripped)

    gguf expert bank ggml types, per-layer (gate, up, down) histogram: {(18, 18, 14): 2, (18, 18, 23): 39, (23, 23, 14): 1}

    --max-prefill-length 8192 -> 8191: the ggml moe_vec grid is tokens*top_k <= 65535 (issue #186, top_k 8); clamped for the gguf expert path

    --moe-cache-auto resolved moe_cache_size=1326 num_pages=133 (prefill_overlap=True)

Also: "Auto-selected attention backend: dsa"; "Page size 1 is auto-adjusted to 64
for latent-KV attention"; "Resolved config: moe_strategy='offload',
attention_backend='dsa', cache_type='hybrid_radix', page_size=64"; "Free memory
before loading model: 28.94 GiB"; "expert banks: slow path (serial build)"
(00:05:52 -> histogram 00:07:35, ~1m43s serial bank build over the 147 GB file).

## Verbatim failure

    [FrontendAPI] ERROR Backend supervisor: AssertionError: Prefill overlap borrows two full expert-layer buffers from the unified MoE cache, so cache_size must be at least 2 * num_experts (raise moe_cache_size or disable moe_prefill_overlap)

Traceback: launch.py:91 _run_scheduler -> scheduler.py:65 -> engine.py:405 __init__
-> engine.py:783 _init_offload_moe_cache -> OffloadMoeCache(...) ->
offload_cache.py:176 __post_init__:

    assert not self.prefill_overlap or self.cache_size >= 2 * self.num_experts

No "Tried to allocate" line exists in the log (Python assert, not an allocator OOM).

## Factual anchors (read-only, no fixes made)

- GGUF metadata (metadata.txt): glm5next.expert_count = 288, expert_used_count = 8.
  So the assert demands cache_size >= 2*288 = 576 PER OffloadMoeCache partition
  under the default moe_prefill_overlap=True.
- Engine splits the auto-planned 1326-slot byte envelope across the three width
  signatures (engine.py _split_moe_cache_budget, byte_cap_slots=1326); at least one
  partition received < 576 slots and its OffloadMoeCache construction asserted.
- The anticipated "minority groups clamped at the 288-slot floor" (== num_experts)
  is below the 576 prefill-overlap requirement by design - floor and invariant
  conflict under prefill_overlap=True.
- served_model_name from ServerArgs (never served): 'GLM-5.3-Flash-UD-Q3_K_XL.gguf'.

## Anomalies

- First runner attempt used a positional model path (argparse death at 5 s, no
  load); corrected to --model-path and relaunched - only the second attempt loaded.
- Runner stderr line "строка 94: 1537822 Убито setsid ..." is the harness noticing
  the watchdog kill -9, not a server-side event.

# Re-run after F1 fix - 2026-09-14 00:30-00:32 - FAIL (new bug, one step further)

Same command, same protocol, same watchdog runner (FAILPAT unchanged). Verdict: FAIL
at boot; prefill/decode probes never ran. F1 (prefill_overlap budget split) is
CONFIRMED FIXED; boot died at GraphRunner construction on a NEW missing-import bug.
No code changed by the smoke; no retry (deterministic NameError).

## New markers that appeared this run (verbatim)

    --moe-cache-auto resolved moe_cache_size=1382 num_pages=132 (prefill_overlap=True)

    MoE signature group (layers [8]) got 288 slots < 2*288: prefill overlap disabled for this partition (synchronous materialized prefill instead)

    MoE signature group (layers [9, 41]) got 288 slots < 2*288: prefill overlap disabled for this partition (synchronous materialized prefill instead)

Both minority groups: 288 slots each, overlap OFF (synchronous materialized prefill).
Dominant 39-layer group: no degradation line (keeps overlap ON; its slot count is
not logged in the partition lines). Then: "Allocating 8448 tokens for KV cache,
K + V = 0.09 GiB"; "Free memory after initialization: 2.82 GiB". Histogram + clamp
lines identical to run 1. Serial bank build 00:30:24 -> 00:31:47 (~1m23s).

## Verbatim failure (new)

    [FrontendAPI] ERROR Backend supervisor: NameError: name 'OffloadMoeCache' is not defined

Traceback: engine.py:477 __init__ -> GraphRunner(..., moe_offload_cache=self.moe_offload_caches
or self.moe_offload_cache) -> graph.py:122 __init__: `if isinstance(moe_offload_cache,
OffloadMoeCache)` -> NameError: OffloadMoeCache is not defined (graph.py does not import
it for the new plural-caches path). Then "Backend worker is gone and cannot be
restarted; stopping the API server"; graceful stop reaped 2 backend workers
(pids=[1575393, 1575394]). Watchdog FAILPAT hit on "Backend worker is gone"; runner
detected death at 117s wall (last /ready code: 503).

Cleanup verified after run 2: NO_FT_PROCESSES; zero GPU compute apps; VRAM 1303 MiB
baseline; RAM 9 GiB used of 188. Full log archived:
verification/boot-smoke-server-log-2026-09-14-run2.txt.

# Run 3 after graph.py import fix - 2026-09-14 00:37-00:39 - FAIL (CUDA IMA in capture)

Fix applied before this run (minimal diff, only permitted change): graph.py promoted
`from freetoken.moe.offload_cache import OffloadMoeCache` out of the TYPE_CHECKING
guard to module level (matching engine.py style, one why-comment added); removed the
redundant TYPE_CHECKING line. The B3 plural handling was already present and correct
(single/list normalization in __init__, per-partition reset at capture boundaries).
Import sanity: `uv run python -c "import freetoken.engine.graph"` -> IMPORT_OK.
Regression: `uv run --extra dev pytest tests/engine -q` -> 146 passed in 1.88s.

The fix worked: GraphRunner constructed, capture began. Boot died in CUDA graph
capture on the FIRST batch (bs=4, "avail_mem = 2.84 GiB", 0/3) - NOT an OOM:

    RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered

Call chain (verbatim frames): graph.py:182 _capture_graphs `self.buffer.logits[:bs] =
model.forward()` -> glm5_next/model.py:188 -> :153 -> :130 -> glm5_next/moe.py:87
routed_forward -> layers/moe.py:218 _decode_routed -> :279 `cache.ensure_experts(self.layer_id,
topk_ids)` -> offload_cache.py:876 -> offload_kernels.py:28 `lru_ensure(... id_base=layer_id
* cache.num_experts ...)` -> flashlib/kernels/slot_cache/triton/lru_ensure.py:335 _seq ->
lru_ensure_kernel -> triton compiler.py:465 _init_handles -> RuntimeError (async CUDA
error surfaced at kernel load). Then c10::AcceleratorError cudaErrorIllegalAddress at
tensor destruction; watchdog FAILPAT hit (RuntimeError), kill -9 tree, 97s wall, /ready
stayed 503.

Markers this run (same as run 2 except plan): auto plan moe_cache_size=1381
num_pages=141; groups [8] and [9,41] 288 slots each, overlap OFF; dominant group no
degradation line (overlap ON); KV 9024 tokens / 0.10 GiB; free after init 2.84 GiB.

Factual anchor (arithmetic, not a diagnosis): ensure_experts is called with the GLOBAL
layer_id and id_base=layer_id*num_experts, while each partition cache carries
slot/id arrays sized num_layers_in_group * 288 (e.g. 576 for the 2-layer group);
288*41 exceeds that. Root-causing this is OUTSIDE the permitted fix scope - STOP per
protocol. Cleanup verified: NO_FT_PROCESSES, no compute apps, VRAM 1414 MiB baseline,
RAM 8 GiB used. Full log: verification/boot-smoke-server-log-2026-09-14-run3.txt.

# Run 4 on current tree - 2026-09-14 01:08-01:11 - FAIL (IMA REPRODUCES)

Measurement-only run, no files modified. Tree sanity before boot: test_gguf_expert_banks
25 passed (incl. test_engine_partitions_real_model_routing), tests/engine 146 passed.
The run-3 IMA REPRODUCED byte-for-byte on the current tree - the stale-hybrid-tree-artifact
hypothesis is FALSIFIED; this is a real, deterministic boot blocker.

Verbatim failure: identical chain to run 3 - graph.py:182 `self.buffer.logits[:bs] =
model.forward()` -> glm5_next/moe.py:87 routed_forward -> layers/moe.py:279
`cache.ensure_experts(self.layer_id, topk_ids)` -> offload_kernels.py:28 lru_ensure
(id_base=layer_id * cache.num_experts) -> flashlib lru_ensure.py:335 _seq ->
_lru_ensure_kernel[(1,)] -> RuntimeError: Triton Error [CUDA]: an illegal memory access
was encountered (surfacing inside triton compiler.py:465 _init_handles - CUDA errors are
async, so the corrupting launch may precede this frame). c10::AcceleratorError at tensor
destruction afterwards. No "Tried to allocate" lines. Watchdog FAILPAT hit (RuntimeError);
101s wall; /ready stayed 503.

Markers this run: clamp 8192->8191 (01:09:31); histogram identical; auto plan
moe_cache_size=1368 num_pages=129 (prefill_overlap=True) [plan varies per run:
1326/1382/1381/1368 - free-memory dependent]; groups [8] and [9,41] at 288 slots overlap
OFF; dominant group silent (overlap ON); serial bank build 01:09:39->01:10:49 (~70s);
KV 8256 tokens / 0.09 GiB; free after init 2.83 GiB; capture died on FIRST batch bs=4
(avail 2.83 GiB, not OOM).

Implication for the wave: the identity-proof/routing test passes while the real boot
crashes, so the test's assumptions do not cover the real failing condition. Probes never
ran; /v1/models never reached. Cleanup verified: NO_FT_PROCESSES, no compute apps, VRAM
1330 MiB baseline, RAM 8 GiB used. Full log: verification/boot-smoke-server-log-2026-09-14-run4.txt.
