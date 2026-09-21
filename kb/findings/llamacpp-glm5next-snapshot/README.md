# Upstream llama.cpp GLM5NEXT snapshot (2026-09-13)

Numbered copies of upstream llama.cpp sources captured 2026-09-13 00:35-01:07
during the gguf-glm5next-path-a reverse-engineering pass. Raw inputs for
`kb/findings/gguf-glm5next-path-a/` (tensor kinds, config keys, tensor map).

Files:
- glm5next_model_full_numbered.txt (814 lines): full upstream GLM5NEXT model
  implementation - glm5next_n_select, indexer_top_k / indexer_kpool hparams,
  hybrid memory (recurrent + kv-cache-kpool), ssm_a sign note (holds
  -exp(A_log) for kimi-k3 lineage, +exp for bailingmoe3; converter checks).
- llama-arch.cpp (1158 lines) with extracted slices:
  kv_names_L161_L424.txt (LLM_KV_NAMES), tensor_names_L426_L699.txt
  (LLM_TENSOR_NAMES), glm5next_cases_L1075_L1160.txt (arch case list).
- kpool.h.txt / kpool_grep.txt: llama-kv-cache-kpool indexer pool mechanics
  (pool_cells I32 / pool_bias F32 / pool_reps, n_pools = ne[0]/r).
- lm_hparams_generic.txt: generic hparams loading.
- lm_create_tensor_qkv.txt / lmh_create_qkv.txt: QKV tensor creation in
  llama-model / llama-memory-hybrid.

WARNING: the upstream git revision these were captured from is NOT recorded.
Before reusing for a new architecture port, verify names against current
upstream; tensor/KV names may have drifted since 2026-09-13. Fix this for
future snapshots: always record repo URL + commit hash at capture time.