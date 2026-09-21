# Git discovery report: rebase `vektory79` onto `main`

Repo: /media/ai/src/FreeToken (origin = https://github.com/FlashML-org/FreeToken.git)
Mode: strictly read-only (status / log / rev-list / rev-parse / merge-base / diff / stash list). No fetch, no rebase, no push performed.

## Current state

- Checked-out branch: `vektory79`, HEAD = 6e673abfe5a92eef11668af7d43d77092042df21 ("Model implementation skill")
- Upstream for the branch: NOT configured (`branch.vektory79.remote` / `branch.vektory79.merge` unset; `@{upstream}` fails with "вышестоящая ветка не настроена")
- Working tree: NOT clean - 2 tracked, unstaged modified files (porcelain v2 status `.M`):
  - `.veai/memory/lean-subagent-context-b42997304646.md`
  - `.veai/memory/process-hygiene-protocol-acfeb33263ee.md`
- No untracked files. `git stash list`: empty.
- No in-progress operations: `.git/rebase-merge` absent, `.git/rebase-apply` absent, `.git/MERGE_HEAD` absent.

## Remotes & main refs

- Remotes (`git remote -v`): single remote `origin` -> `https://github.com/FlashML-org/FreeToken.git` (fetch/push)
- Local `main`: exists, afd99cb3b7c118dce4a9835359bcbc33b70c3cb8
- `origin/main`: exists, afd99cb3b7c118dce4a9835359bcbc33b70c3cb8 (same oid)
- Tip of both: `afd99cb chore(docs): update wechat group qr code link to #482 (#485)`
- Divergence local main vs origin/main (`rev-list --left-right --count main...origin/main`): `0  0` - identical.
- Caveat: fetching was forbidden in this session, so "origin/main" reflects the last fetch stored locally.

## Divergence

- Merge base of HEAD and main: `0ffd5c8b2941974ed64dec09170b19259e2ba5aa`
- `rev-list --left-right --count main...HEAD` = `12  50`
  - 12 commits on main that are not on HEAD
  - 50 commits on HEAD that are not on main (8 true merge commits + 42 non-merge commits; one non-merge commit `deb5c10` carries a merge-style subject - see Branch commits)
- Note: history contains `3e5bbdd "Merge branch 'FlashML-org:main' into feat/fp8-quantization"` whose second parent `af71ba4` is NOT reachable from current main. This means main history was rewritten at some point (or the merge brought in a pre-rewrite snapshot). The authoritative common ancestor for the rebase is 0ffd5c8.

## Main commits (12 commits after merge-base)

`git log --oneline HEAD..main` (newest first):

```
afd99cb chore(docs): update wechat group qr code link to #482 (#485)
68a81ff feat(minimax_m3): serve image input on MiniMax-M3 (#480)
63d6471 feat(muse_glimmer): serve image input on Muse-Glimmer-30B (#481)
bea8d06 fix(engine): resolve moe-strategy auto to fused on unified-memory GPUs (GB10) (#445)
3faef36 feat(glm5_next): serve image input on GLM-5.3-Flash (#479)
84d236c feat(gemma4): serve image input on the Gemma-4 releases (#467)
e0886cc fix(models): compute shared experts before in-place routed experts (#463)
db64879 feat(server): carry tool-result images on the following user turn
8ff0cce feat(launch): declare image input to the agents when the server serves it
f7dbab7 fix(quant): load ModelOpt exports that ship no input_scale (#462)
08d728d feat(mm): serve image input on the Qwen families (#454)
9535656 fix(server): honor configured output default across APIs (#411)
```

`git diff --stat 0ffd5c8 main`: 151 files changed, 7191 insertions(+), 615 deletions(-).
Subsystems touched on main side:
- NEW multimodal pipeline: `python/freetoken/mm/` (config, encoder_cache, media, processor, processors/gemma4|glm5_next|minimax_m3|muse_glimmer|qwen_vl), `scheduler/mm.py`
- Vision for models: new `qwen3_vl/` model dir, `glm5_next/vision.py`, `minimax_m3/vision.py`, `muse_glimmer/vision.py`, gemma4 vision updates
- Engine: `engine/config.py` (+39), `engine/engine.py` (+147), `engine/graph.py`, `kernel/triton/rope.py` (+127 new), `layers/rotary.py` (+138)
- Server: `server/args.py` (+107), openai/anthropic/responses APIs, `tokenizer/server.py`
- Attention/kvcache: `attention/qsa_sparse.py`, `attention/triton.py`, `kernel/triton/attention.py`, `kvcache/qsa_pool.py`, `layers/moe.py`, `models/register.py`, `models/weight.py`, `scheduler/scheduler.py`
- Docs: `docs/cli.md` (+28), `docs/models.md` (+17); housekeeping: README/CONTRIBUTING/.github config, removed `assets/freetoken-wechatgroup.png`, `pyproject.toml`
- Tests: large mm/vision test wave (`tests/mm/`, `tests/tokenizer/test_mm_tokenize.py`, `tests/models/test_*_vision.py`, `tests/kernels/test_mrope.py`, `tests/kernels/test_triton_attention.py` +103, `tests/models/test_quant_config.py` +45, etc.)

## Branch commits (50 commits in `main..HEAD`, with files)

Order: newest first (as printed by `git log --format='--- %h %s' --name-only main..HEAD`). 8 true merges + 42 single-parent commits. `deb5c10` has a merge-style subject but only ONE parent - it is a merge commit that was linearized by a past rebase, so its 13-file diff replays as a regular commit during rebase.

```
--- 6e673ab Model implementation skill
.veai/memory/MEMORY.md
.veai/memory/benchbw-profile-clobber-trap-6b9f0d04e5e9.md
.veai/memory/cuda-debug-tool-strategy-dec30f16a02e.md
.veai/memory/ft-gguf-glm5next-private-scope-d788b6c0ee9e.md
.veai/memory/ft-gguf-kernel-jit-toolchain-67df3b73da55.md
.veai/memory/ft-gguf-native-serving-skill-e4232bb48315.md
.veai/memory/ft-gguf-test-fixture-crafting-7a85eb5d2682.md
.veai/memory/ft-offload-banks-pinned-host-2039b53cb2e9.md
.veai/memory/ft-offload-banks-pinned-host-90784dc7d0a9.md
.veai/memory/ft-pr-relevance-glm-hybrid-742559ca5630.md
.veai/memory/ft-serve-gguf-glm5next-unsupported-a97d176f3b2e.md
.veai/memory/ft-serve-moe-flags-semantics-35ff08215256.md
.veai/memory/ft-serve-prefill-overlap-512k-infeasible-e4766181008c.md
.veai/memory/ft-serve-test-and-e2e-gotchas-b9def4525192.md
.veai/memory/gguf-hybrid-decode-handshake-floor-d98055871b80.md
.veai/memory/gguf-hybrid-reacceptance-final-39843d88ac42.md
.veai/memory/glm53-flash-nvfp4-cache-budget-e072d0274c3d.md
.veai/memory/glm53-post-iommu-baseline-fb58a59761d2.md
.veai/memory/lean-subagent-context-b42997304646.md
.veai/memory/nvme-990evo-plus-iommu-fio-gotchas-0b993bdb2c17.md
.veai/memory/pr408-kv-nvfp4-1m-port-029031181f9f.md
.veai/memory/process-hygiene-protocol-acfeb33263ee.md
.veai/memory/rtx5090-pcie-gen5-bw-cap-30a075828b64.md
.veai/skills/code-with-subagents/SKILL.md
.veai/skills/gguf-native-serving/ORCHESTRATION.md
.veai/skills/gguf-native-serving/SKILL.md
.veai/skills/gguf-native-serving/TRAPS.md
.veai/skills/gguf-native-serving/tasks/task-00-discovery.md
.veai/skills/gguf-native-serving/tasks/task-01-config-shim.md
.veai/skills/gguf-native-serving/tasks/task-02-tensor-translator.md
.veai/skills/gguf-native-serving/tasks/task-03-tokenizer.md
.veai/skills/gguf-native-serving/tasks/task-04-kernel-dispatch.md
.veai/skills/gguf-native-serving/tasks/task-05-expert-banks.md
.veai/skills/gguf-native-serving/tasks/task-06-e2e-ab.md
.veai/skills/gguf-native-serving/tasks/task-07-cpu-compute-tier.md
--- d015514 Memory
.veai/memory/MEMORY.md
.veai/memory/benchbw-profile-clobber-trap-6b9f0d04e5e9.md
.veai/memory/ft-gguf-glm5next-private-scope-d788b6c0ee9e.md
.veai/memory/ft-gguf-kernel-jit-toolchain-67df3b73da55.md
.veai/memory/ft-gguf-test-fixture-crafting-7a85eb5d2682.md
.veai/memory/ft-pr-relevance-glm-hybrid-742559ca5630.md
.veai/memory/ft-serve-gguf-glm5next-unsupported-a97d176f3b2e.md
.veai/memory/ft-serve-moe-flags-semantics-35ff08215256.md
.veai/memory/ft-serve-prefill-overlap-512k-infeasible-e4766181008c.md
.veai/memory/ft-serve-test-and-e2e-gotchas-b9def4525192.md
.veai/memory/gguf-hybrid-decode-handshake-floor-321fcf0b0112.md
.veai/memory/gguf-hybrid-decode-handshake-floor-d98055871b80.md
.veai/memory/gguf-hybrid-reacceptance-final-39843d88ac42.md
.veai/memory/glm53-flash-nvfp4-cache-budget-e072d0274c3d.md
.veai/memory/glm53-post-iommu-baseline-fb58a59761d2.md
.veai/memory/lean-subagent-context-b42997304646.md
.veai/memory/pr408-kv-nvfp4-1m-port-0c2bf3951cd7.md
.veai/memory/process-hygiene-protocol-acfeb33263ee.md
--- 63b9bff fix(moe): cap gguf dot tier at cpu support, trim weighted pool splits
python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp
python/freetoken/moe/cpu_executor.py
tests/moe/test_cpu_moe_gguf_iq.py
tests/moe/test_hybrid_fetch.py
--- 979e3fc feat(moe): weight the per-partition cpu pool split by partition layer count
python/freetoken/engine/engine.py
python/freetoken/moe/cpu_executor.py
tests/moe/test_hybrid_fetch.py
--- 11f1a80 feat(moe): verbatim ggml AVX2 W4A8-K integer tier for the gguf CPU GEMV
python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp
tests/moe/test_cpu_moe_gguf_iq.py
tests/moe/test_hybrid_fetch.py
--- 1b913ba Memory
.veai/memory/MEMORY.md
.veai/memory/ft-gguf-glm5next-phase5-acceptance-975a360a0e4a.md
.veai/memory/ft-gguf-glm5next-private-scope-d788b6c0ee9e.md
.veai/memory/ft-gguf-kernel-jit-toolchain-124193259365.md
.veai/memory/ft-gguf-kernel-jit-toolchain-67df3b73da55.md
.veai/memory/ft-gguf-test-fixture-crafting-7a85eb5d2682.md
.veai/memory/ft-offload-banks-pinned-host-2039b53cb2e9.md
.veai/memory/ft-pr-relevance-glm-hybrid-742559ca5630.md
.veai/memory/ft-serve-gguf-glm5next-unsupported-504d543c618e.md
.veai/memory/ft-serve-gguf-glm5next-unsupported-590e12c7424d.md
.veai/memory/ft-serve-moe-flags-semantics-35ff08215256.md
.veai/memory/ft-serve-prefill-overlap-512k-infeasible-e4766181008c.md
.veai/memory/ft-serve-test-and-e2e-gotchas-b9def4525192.md
.veai/memory/gguf-hybrid-decode-handshake-floor-321fcf0b0112.md
.veai/memory/glm53-flash-nvfp4-cache-budget-8ce65b535fed.md
.veai/memory/glm53-post-iommu-baseline-fb58a59761d2.md
.veai/memory/lean-subagent-context-b42997304646.md
.veai/memory/nvme-990evo-plus-iommu-fio-gotchas-0b993bdb2c17.md
.veai/memory/pr408-kv-nvfp4-1m-port-0c2bf3951cd7.md
.veai/memory/rtx5090-pcie-gen5-bw-cap-30a075828b64.md
--- eb7de4c fix(moe): pool attribution, affinity clamp, truthful help and advice for per-partition cpu executors
python/freetoken/engine/config.py
python/freetoken/engine/engine.py
python/freetoken/moe/cpu_executor.py
python/freetoken/server/args.py
tests/engine/test_cache_budget.py
tests/models/test_gguf_expert_banks.py
tests/moe/test_hybrid_fetch.py
--- ef83ee8 docs(engine): note the cpu moe executor sample borrows across partitions
python/freetoken/engine/engine.py
--- f6b94ad feat(engine): accept gguf + hybrid when every bank type is CPU-executable
python/freetoken/engine/engine.py
python/freetoken/moe/expert_banks.py
tests/engine/test_cache_budget.py
--- 1635ecd fix(kernel): scope gguf jit cc/cxx override to the build invocation
python/freetoken/kernel/gguf.py
tests/kernels/test_gguf_quant.py
tests/moe/test_hybrid_fetch.py
--- aad5d3a feat(engine): per-cache cpu moe executors, lift multi-partition hybrid block
python/freetoken/engine/engine.py
python/freetoken/moe/cpu_executor.py
tests/models/test_gguf_expert_banks.py
tests/moe/test_hybrid_fetch.py
--- 2169baa feat(moe): bench the cpu moe leg for gguf expert formats
python/freetoken/moe/bench_profile.py
python/freetoken/moe/benchbw.py
tests/moe/test_hybrid_fetch.py
--- 427a431 test(moe): negative-path coverage for gguf executor resolution guards
tests/moe/test_cpu_moe_gguf_iq.py
--- bd02232 fix(moe): harden gguf cpu executor bank resolution and pin skip invariants
python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp
python/freetoken/moe/cpu_executor.py
tests/moe/test_cpu_moe_gguf_iq.py
--- d845981 Memory
.veai/memory/MEMORY.md
.veai/memory/ft-gguf-glm5next-phase5-acceptance-975a360a0e4a.md
.veai/memory/ft-gguf-kernel-jit-toolchain-124193259365.md
.veai/memory/ft-gguf-test-fixture-crafting-7a85eb5d2682.md
.veai/memory/ft-offload-banks-pinned-host-2039b53cb2e9.md
.veai/memory/ft-serve-gguf-glm5next-unsupported-590e12c7424d.md
.veai/memory/ft-serve-moe-flags-semantics-ac92781688a8.md
.veai/memory/ft-serve-prefill-overlap-512k-infeasible-2fbd9f4276bf.md
.veai/memory/lean-subagent-context-b42997304646.md
--- 5a04423 feat(moe): add cpu gemv for gguf iq3_xxs/iq4_xs/q6_k banks
python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp
python/freetoken/moe/cpu_executor.py
tests/models/test_gguf_expert_banks.py
tests/moe/test_cpu_moe_gguf_iq.py
--- c9f5f45 Memory
.veai/memory/MEMORY.md
.veai/memory/ft-gguf-glm5next-phase5-acceptance-76ec539bb748.md
.veai/memory/ft-gguf-glm5next-phase5-acceptance-975a360a0e4a.md
.veai/memory/ft-gguf-glm5next-private-scope-d788b6c0ee9e.md
.veai/memory/ft-gguf-kernel-jit-toolchain-124193259365.md
.veai/memory/ft-gguf-test-fixture-crafting-efa42fbec77a.md
.veai/memory/ft-offload-banks-pinned-host-2039b53cb2e9.md
.veai/memory/ft-serve-gguf-glm5next-unsupported-579af61234a1.md
.veai/memory/ft-serve-gguf-glm5next-unsupported-590e12c7424d.md
.veai/memory/ft-serve-prefill-overlap-512k-infeasible-2fbd9f4276bf.md
.veai/memory/ft-serve-test-and-e2e-gotchas-b9def4525192.md
--- 78115df fix(gguf): glm4 digit-split pre-tokenizer, exact bank sizing, cache-status tests, docs
docs/install.md
python/freetoken/engine/engine.py
python/freetoken/kvcache/cache_status.py
python/freetoken/models/gguf/tokenizer.py
python/freetoken/moe/expert_banks.py
python/freetoken/moe/offload_cache.py
tests/kvcache/test_cache_unit_bytes.py
tests/models/test_gguf_expert_banks.py
tests/models/test_glm5_next_gguf.py
--- af8e2e4 Memory
.veai/memory/MEMORY.md
.veai/memory/ft-gguf-glm5next-phase5-acceptance-76ec539bb748.md
.veai/memory/ft-gguf-glm5next-private-scope-d788b6c0ee9e.md
.veai/memory/ft-gguf-kernel-jit-toolchain-124193259365.md
.veai/memory/ft-gguf-test-fixture-crafting-efa42fbec77a.md
.veai/memory/ft-offload-banks-pinned-host-2039b53cb2e9.md
.veai/memory/ft-pr-relevance-glm-hybrid-742559ca5630.md
.veai/memory/ft-serve-gguf-glm5next-unsupported-23b45ac011ac.md
.veai/memory/ft-serve-gguf-glm5next-unsupported-579af61234a1.md
.veai/memory/ft-serve-moe-flags-semantics-ac92781688a8.md
.veai/memory/ft-serve-prefill-overlap-512k-infeasible-2fbd9f4276bf.md
.veai/memory/ft-serve-test-and-e2e-gotchas-b9def4525192.md
.veai/memory/glm53-flash-nvfp4-cache-budget-8ce65b535fed.md
.veai/memory/glm53-post-iommu-baseline-fb58a59761d2.md
.veai/memory/nvme-990evo-plus-iommu-fio-gotchas-0b993bdb2c17.md
.veai/memory/pr408-kv-nvfp4-1m-port-13edf6413fdc.md
.veai/memory/rtx5090-pcie-gen5-bw-cap-30a075828b64.md
--- 028f2d9 fix(gguf): normalize non-contiguous activations in fused_mul_mat_gguf
python/freetoken/layers/gguf.py
tests/kernels/test_gguf_quant.py
--- f801a08 feat(moe): route glm5next gguf expert banks through offload
python/freetoken/engine/engine.py
python/freetoken/engine/graph.py
python/freetoken/kvcache/cache_status.py
python/freetoken/layers/moe.py
python/freetoken/models/glm5_next/__init__.py
python/freetoken/models/glm5_next/gguf.py
python/freetoken/models/weight.py
python/freetoken/moe/bench_profile.py
python/freetoken/moe/benchbw.py
python/freetoken/moe/expert_banks.py
python/freetoken/moe/offload_cache.py
python/freetoken/scheduler/scheduler.py
tests/engine/test_cache_budget.py
tests/models/test_gguf_expert_banks.py
tests/models/test_glm5_next_gguf.py
tests/moe/test_hybrid_fetch.py
--- e2fb847 Memory
.veai/memory/MEMORY.md
.veai/memory/ft-gguf-glm5next-private-scope-d788b6c0ee9e.md
.veai/memory/ft-gguf-kernel-jit-toolchain-124193259365.md
.veai/memory/ft-serve-gguf-glm5next-unsupported-23b45ac011ac.md
.veai/memory/ft-serve-gguf-glm5next-unsupported-fce4c477fa0d.md
.veai/memory/ft-serve-test-and-e2e-gotchas-b9def4525192.md
--- 2637714 feat(models): wire glm5next gguf dispatch and tokenizer
python/freetoken/engine/engine.py
python/freetoken/layers/gguf.py
python/freetoken/models/gguf/dequant.py
python/freetoken/models/gguf/tokenizer.py
python/freetoken/models/glm5_next/gguf.py
python/freetoken/models/glm5_next/model.py
tests/engine/test_cache_budget.py
tests/kernels/test_gguf_quant.py
tests/models/test_glm5_next_gguf.py
```

```
--- 4443d7e feat(models): add glm5next gguf weight iterator and expert sources
python/freetoken/models/gguf/dequant.py
python/freetoken/models/glm5_next/__init__.py
python/freetoken/models/glm5_next/gguf.py
tests/models/test_glm5_next_gguf.py
--- ad13127 Memory
.veai/memory/MEMORY.md
.veai/memory/ft-pr-relevance-glm-hybrid-742559ca5630.md
.veai/memory/ft-serve-gguf-glm5next-unsupported-48dd4f1019a0.md
.veai/memory/ft-serve-gguf-glm5next-unsupported-fce4c477fa0d.md
.veai/memory/ft-serve-moe-flags-semantics-ac92781688a8.md
.veai/memory/ft-serve-prefill-overlap-512k-infeasible-6a5d5c0d1fd0.md
.veai/memory/ft-serve-test-and-e2e-gotchas-b9def4525192.md
.veai/memory/glm53-flash-nvfp4-cache-budget-9ea1520673f8.md
.veai/memory/glm53-post-iommu-baseline-95f74e785908.md
.veai/memory/nvme-990evo-plus-iommu-fio-gotchas-0b993bdb2c17.md
.veai/memory/pr408-kv-nvfp4-1m-port-13edf6413fdc.md
.veai/memory/rtx5090-pcie-gen5-bw-cap-33ae77e9943f.md
--- 4162f7c feat(models): add glm5next gguf config shim and registry spec
python/freetoken/kernel/aot_models.py
python/freetoken/models/gguf/config.py
python/freetoken/models/glm5_next/__init__.py
python/freetoken/models/glm5_next/gguf.py
python/freetoken/models/register.py
tests/models/test_glm5_next_gguf.py
--- 18ef07e Memory
.veai/memory/MEMORY.md
.veai/memory/ft-pr-relevance-glm-hybrid-e86db741a45c.md
.veai/memory/ft-serve-gguf-glm5next-unsupported-48dd4f1019a0.md
.veai/memory/ft-serve-moe-flags-semantics-ac92781688a8.md
.veai/memory/ft-serve-moe-flags-semantics-ad7d6de9a43d.md
.veai/memory/ft-serve-prefill-overlap-512k-infeasible-6a5d5c0d1fd0.md
.veai/memory/ft-serve-test-and-e2e-gotchas-b9def4525192.md
.veai/memory/glm53-flash-nvfp4-cache-budget-360195b98396.md
.veai/memory/glm53-post-iommu-baseline-95f74e785908.md
.veai/memory/nvme-990evo-plus-iommu-fio-gotchas-0b993bdb2c17.md
.veai/memory/pr408-kv-nvfp4-1m-port-13edf6413fdc.md
.veai/memory/rtx5090-pcie-gen5-bw-cap-5dd13615ef79.md
--- f06347b OS fixes notes
.veai/docs/rtx5090-pcie-gen5-iommu-bandwidth.md
.veai/memory/MEMORY.md
.veai/memory/ft-serve-moe-flags-semantics-ad7d6de9a43d.md
.veai/memory/ft-serve-test-and-e2e-gotchas-7375616a617c.md
.veai/memory/glm53-flash-nvfp4-cache-budget-360195b98396.md
.veai/memory/nvme-990evo-plus-iommu-fio-gotchas-0b993bdb2c17.md
.veai/memory/pr408-kv-nvfp4-1m-port-13edf6413fdc.md
.veai/memory/rtx5090-pcie-gen5-bw-cap-5dd13615ef79.md
--- f7ff3e2 More engine compatibility errors
.veai/memory/MEMORY.md
.veai/memory/ft-serve-test-and-e2e-gotchas-7375616a617c.md
.veai/memory/glm53-flash-nvfp4-cache-budget-d28de3176510.md
.veai/memory/pr408-kv-nvfp4-1m-port-13edf6413fdc.md
python/freetoken/layers/quantization/method.py
python/freetoken/server/args.py
tests/models/test_quant_config.py
tests/server/test_deprecated_flag_hints.py
--- ba0f20f Merge branch 'pr-408' into vektory79
(true merge, parents: deb5c10 + 04d4621)
--- deb5c10 Merge remote-tracking branch 'origin/perf/default-vendored-topk-router' into HEAD
SINGLE-PARENT commit (linearized merge); its own diff:
.gitignore
.veai/mcp_servers.json
.veai/memory/MEMORY.md
.veai/memory/ft-serve-test-and-e2e-gotchas-ca05468edd30.md
.veai/memory/glm53-flash-nvfp4-cache-budget-403468cb69b2.md
docs/cli.md
python/freetoken/server/access_log_filter.py
python/freetoken/server/api_server.py
python/freetoken/server/control_api.py
python/freetoken/server/launch.py
python/freetoken/server/supervisor.py
python/freetoken/tokenizer/server.py
tests/server/test_lifespan_shutdown.py
tests/server/test_rebuild_maintenance.py
tests/server/test_supervisor.py
--- 04d4621 feat(kvcache): support nvfp4 latent dsa kv
docs/cli.md
python/freetoken/attention/__init__.py
python/freetoken/attention/dsa.py
python/freetoken/engine/engine.py
python/freetoken/kernel/triton/glm_dsa_sparse.py
python/freetoken/kernel/triton/kv_nvfp4.py
python/freetoken/kvcache/__init__.py
python/freetoken/kvcache/dsa_pool.py
tests/attention/test_dsa_kpool.py
tests/engine/test_kv_quant_config.py
tests/kernels/test_kv_nvfp4.py
tests/kvcache/test_dsa_pool.py
tests/models/test_glm5_next_config.py
tests/models/test_glm_dsa.py
--- 7f4d788 Merge commit '9b103b04f9c8a1544dbe857013dd129170defbd7' into feat/nvfp4-kv-quantization
(true merge, parents: 3b84b80 + 9b103b0)
--- 9b103b0 feat(kvcache): add fp8 support for dsa kv cache
python/freetoken/attention/__init__.py
python/freetoken/attention/dsa.py
python/freetoken/engine/engine.py
python/freetoken/kernel/triton/glm_dsa_sparse.py
python/freetoken/kernel/triton/kv_quant.py
python/freetoken/kvcache/__init__.py
python/freetoken/kvcache/dsa_pool.py
tests/attention/test_dsa_kpool.py
tests/engine/test_kv_quant_config.py
tests/kvcache/test_dsa_pool.py
tests/kvcache/test_kv_cache_rebuild.py
tests/kvcache/test_mha_pool_fp8.py
--- 3b84b80 Merge commit 'cfe82df02b1d8999d86609aa44bf600ade2665d6' into feat/nvfp4-kv-quantization
(true merge, parents: eae141d + cfe82df)
--- cfe82df Merge pull request #3 from naerymdan/perf/kv-fp8-read-path
(true merge, parents: ca3675e + 73ca76c)
--- eae141d Merge commit 'ca3675ecde8d53385ddb32cf4d611c7230d0b897' into feat/nvfp4-kv-quantization
(true merge, parents: 2f554c9 + ca3675e)
--- ca3675e test(kernels): stabilize fp8 extend attention regression
tests/kernels/test_triton_attention.py
--- 2f554c9 feat(kvcache): add nvfp4 kv quantization
benchmarks/README.md
benchmarks/bench_kv_quant.py
docs/cli.md
python/freetoken/attention/__init__.py
python/freetoken/attention/qsa_sparse.py
python/freetoken/attention/triton.py
python/freetoken/engine/engine.py
python/freetoken/kernel/triton/attention.py
python/freetoken/kernel/triton/kv_nvfp4.py
python/freetoken/kernel/triton/qsa/attend.py
python/freetoken/kvcache/__init__.py
python/freetoken/kvcache/base.py
python/freetoken/kvcache/hybrid_swa_pool.py
python/freetoken/kvcache/mha_pool.py
python/freetoken/kvcache/qsa_pool.py
python/freetoken/server/args.py
tests/engine/test_kv_quant_config.py
tests/kernels/test_kv_nvfp4.py
tests/kernels/test_qsa_nvfp4.py
tests/kvcache/test_qsa_pool_fp8.py
--- 33872fd Merge pull request #2 from MT-z/fix/fp8-tests-single-process
(true merge, parents: 3e5bbdd + 0820ff4)
--- 73ca76c perf(kernel): size the extend tile from the KV cache element size
python/freetoken/kernel/triton/attention.py
tests/kernels/test_triton_attention.py
--- 05861fb perf(kernel): apply the fp8 KV dequant scale after the dot, not to the tile
python/freetoken/kernel/triton/attention.py
python/freetoken/kernel/triton/e4m3_compat.py
tests/kernels/test_e4m3_compat.py
--- 0820ff4 test(kernels): give the triton-attention doubles the scale accessors the backend now reads
tests/kernels/test_triton_attention.py
--- 5febeee test(kvcache): give the layer-ids remap test a model deep enough for its ids
tests/kvcache/test_mha_pool_fp8.py
--- dabe93c test(kernels): put the scale-one encoder's V tensor on the device
tests/kernels/test_kv_fp8.py
--- 7f9a05a test(kvcache): size the fp8 slot round-trip to the rows it indexes
tests/kvcache/test_qsa_pool_fp8.py
--- 3e5bbdd Merge branch 'FlashML-org:main' into feat/fp8-quantization
(true merge, parents: 5b9efc5 + af71ba4; second parent af71ba4 is NOT reachable from current main - sign of main history rewrite)
--- 5b9efc5 Merge pull request #1 from MT-z/fix/kv-fp8-vstore-pitch
(true merge, parents: 03fb043 + 811ccee)
--- 811ccee fix(kernels): give the V tensor its own row pitch in the fp8 KV store
python/freetoken/kernel/triton/kv_quant.py
--- 03fb043 feat(kvcache): store the KV cache as fp8 e4m3 codes (--kv-cache-dtype fp8)
docs/cli.md
docs/models.md
python/freetoken/attention/__init__.py
python/freetoken/attention/qsa_sparse.py
python/freetoken/attention/triton.py
python/freetoken/engine/config.py
python/freetoken/engine/engine.py
python/freetoken/kernel/aot_models.py
python/freetoken/kernel/triton/attention.py
python/freetoken/kernel/triton/e4m3_compat.py
python/freetoken/kernel/triton/kv_quant.py
python/freetoken/kernel/triton/qsa/attend.py
python/freetoken/kernel/triton/qsa/score.py
python/freetoken/kvcache/__init__.py
python/freetoken/kvcache/base.py
python/freetoken/kvcache/hybrid_swa_pool.py
python/freetoken/kvcache/mha_pool.py
python/freetoken/kvcache/qsa_pool.py
python/freetoken/server/args.py
tests/engine/test_kv_quant_config.py
tests/kernels/test_e4m3_compat.py
tests/kernels/test_kv_fp8.py
tests/kernels/test_qsa_fp8.py
tests/kernels/test_triton_attention.py
tests/kvcache/test_mha_pool_fp8.py
tests/kvcache/test_qsa_pool_fp8.py
tests/models/qwen4_exp/common.py
tests/models/qwen4_exp/test_qsa_backend.py
```

Note: per-file commit attribution in the next section lists both real changes and merge commits; merge commits (except the linearized `deb5c10`) usually carry no own hunks - the actual conflict-driving changes come from the non-merge commits listed.

## Overlapping files (conflict candidates)

Method: intersection of `git diff --name-only 0ffd5c8 HEAD` (118 branch-side files) with `git diff --name-only 0ffd5c8 main` (151 main-side files). OVERLAP: 26 files.

| # | File | Branch commits that touched it (m = true merge) |
|---|------|--------------------------------------------------|
| 1 | docs/cli.md | 03fb043, 2f554c9, 04d4621; m: ba0f20f, deb5c10 |
| 2 | docs/models.md | 03fb043; m: ba0f20f, 3e5bbdd |
| 3 | python/freetoken/attention/qsa_sparse.py | 03fb043, 2f554c9 |
| 4 | python/freetoken/attention/triton.py | 03fb043, 2f554c9 |
| 5 | python/freetoken/engine/config.py | 03fb043, eb7de4c; m: ba0f20f, 3e5bbdd |
| 6 | python/freetoken/engine/engine.py | 03fb043, 2f554c9, 9b103b0, 04d4621, 2637714, f801a08, 78115df, aad5d3a, f6b94ad, ef83ee8, eb7de4c, 979e3fc; m: ba0f20f, 3e5bbdd, 7f4d788 - BIGGEST collision zone (main side +147 lines: mm/image-input wiring) |
| 7 | python/freetoken/engine/graph.py | f801a08 |
| 8 | python/freetoken/kernel/aot_models.py | 03fb043, 4162f7c; m: ba0f20f, 3e5bbdd |
| 9 | python/freetoken/kernel/triton/attention.py | 03fb043, 2f554c9, 73ca76c, 05861fb; m: 3b84b80 |
| 10 | python/freetoken/kvcache/__init__.py | 03fb043, 2f554c9, 9b103b0, 04d4621; m: 3e5bbdd, 7f4d788 |
| 11 | python/freetoken/kvcache/qsa_pool.py | 03fb043, 2f554c9 |
| 12 | python/freetoken/layers/moe.py | f801a08 |
| 13 | python/freetoken/models/glm5_next/__init__.py | 4162f7c, 4443d7e, f801a08 |
| 14 | python/freetoken/models/glm5_next/model.py | 2637714 |
| 15 | python/freetoken/models/register.py | 4162f7c (main side rewrote register.py heavily: +127) |
| 16 | python/freetoken/models/weight.py | f801a08 |
| 17 | python/freetoken/scheduler/scheduler.py | f801a08 |
| 18 | python/freetoken/server/api_server.py | only linearized merge deb5c10 |
| 19 | python/freetoken/server/args.py | 03fb043, 2f554c9, f7ff3e2, eb7de4c; m: ba0f20f, 3e5bbdd - main side +107 lines |
| 20 | python/freetoken/server/launch.py | only linearized merge deb5c10 |
| 21 | python/freetoken/tokenizer/server.py | only linearized merge deb5c10 |
| 22 | tests/kernels/test_triton_attention.py | 03fb043, 0820ff4, 73ca76c, ca3675e; m: cfe82df - main side +103 lines |
| 23 | tests/models/qwen4_exp/common.py | 03fb043; m: ba0f20f |
| 24 | tests/models/qwen4_exp/test_qsa_backend.py | 03fb043 |
| 25 | tests/models/test_glm5_next_config.py | 04d4621; m: ba0f20f, 3e5bbdd |
| 26 | tests/models/test_quant_config.py | f7ff3e2 (main side +45) |

Expected conflict clusters:
1. fp8/nvfp4 KV wave (03fb043, 2f554c9, 9b103b0, 04d4621...) vs main's multimodal/image-input wave: `engine/engine.py`, `server/args.py`, `engine/config.py`, `kvcache/__init__.py`, `kernel/triton/attention.py`, `tests/kernels/test_triton_attention.py`, docs.
2. glm5next gguf wave (4162f7c, 4443d7e, 2637714, f801a08...) vs main's glm5_next vision changes: `models/glm5_next/__init__.py`, `models/glm5_next/model.py`, `models/register.py`, `models/weight.py`, `scheduler/scheduler.py`, `engine/graph.py`, `layers/moe.py`.
3. Server files touched only by the linearized merge `deb5c10` (api_server.py, server/launch.py, tokenizer/server.py) - moderate risk, depends on how main evolved those areas.

## State flags

- (a) Working tree: DIRTY - 2 tracked modified files, both under `.veai/memory/` (unstaged `.M`): `lean-subagent-context-b42997304646.md`, `process-hygiene-protocol-acfeb33263ee.md`. Plain `git rebase` will refuse until they are committed or stashed.
- (b) Upstream: branch `vektory79` has NO upstream configured. Ahead/behind vs upstream: N/A. Consequence: no force-push is required by current setup; after rebase the first push would be a normal `git push -u origin vektory79`. Fact recorded only - per CONTRIBUTING.md agents never push.
- (c) Local main vs origin/main: identical (`0 0`), so the actual "last commit of main" = afd99cb both locally and on origin (as of the last fetch; no fetch was allowed in this session).
- (d) No unfinished operations: no `.git/rebase-merge`, no `.git/rebase-apply`, no `.git/MERGE_HEAD`; stash is empty.
- (e) History hygiene: branch contains 8 true merges (incl. GitHub PR merges #1/#2/#3 already integrated) and one linearized merge-titled commit `deb5c10`; second parent `af71ba4` of `3e5bbdd` is not in current main - main history was rewritten at some point.

## Recommendations

1. Target for rebase: `main` = `origin/main` = `afd99cb` (merge-base 0ffd5c8). Suggested: `git rebase main` (optionally `git fetch origin` first to refresh origin/main - not done here, read-only session).
2. Before starting, clear the tree: commit the two `.veai/memory/*` modifications on the branch (pattern of "Memory" commits already exists on this branch) or stash them. `.veai/` files are NOT touched by main's 12 commits, so either way they replay cleanly.
3. Decide merge handling: default rebase flattens the 8 true merges into ~42 replayed non-merge commits (this also drops the already-integrated PR merges #1/#2/#3). If the PR-merge structure should survive, use `--rebase-merges`; otherwise expect a linearized branch.
4. `deb5c10` (merge-titled single-parent commit with a 13-file diff) will replay as a regular commit - do not be surprised by its subject.
5. Highest-conflict order of work: resolve `engine/engine.py` and `server/args.py` first (both waves touch them), then the glm5_next cluster (`register.py`, `glm5_next/__init__.py`, `glm5_next/model.py`, `weight.py`), then attention/kvcache kernel files, then tests/docs.
6. After a successful rebase: run the test suite per repo rules; pushing is a human action (CONTRIBUTING.md forbids agents from `git push`).

