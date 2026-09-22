# Merge round 3 test results (verify-before-finalization)

Date: 2026-09-20. Merge state under test: in-progress merge of origin/main (6410c97) into
vektory79 @ 1445fc1, staged tree d1f2313d == fresh `git merge-tree --write-tree HEAD 6410c97`
(blob-identical for both files); worktree == index. Provenance: the merge was started 18:41 by an
interrupted prior run of this same round; this session re-verified the state bit-for-bit and
continued it instead of abort+redo. No textual conflicts existed (ort auto-union; adaptation-log.md).

All pytest runs: SERIALIZED (one at a time), each under a hard `timeout`, census
(pgrep 'pytest|ft serve|benchbw|ft_run' + `ls /dev/shm | grep -c '^sem\.mp-'`) before/after each
run. Pre-test GPU check: no foreign wave (30.7 GiB free; only two 302 MiB desktop compute apps;
no ft serve/pytest processes).

## Results

| # | Command (strict `uv run --extra dev`) | Result |
|---|---|---|
| 1 | `python -c "import freetoken; import freetoken.engine.engine; import freetoken.kernel.utils"` | OK, rc=0 |
| 2 | `python -m pytest tests/engine -q --tb=short -rf` | 161 passed, 2 skipped, 0 failed (2.1s) |
| 3 | `python -m pytest tests/models/test_gguf_expert_banks.py tests/models/test_glm5_next_config.py tests/models/test_glm5_next_gguf.py tests/models/test_glm5_next_kda_op.py tests/models/test_glm5_next_model.py tests/models/test_glm5_next_vision.py tests/models/test_qwen3_vl.py tests/models/test_qwen3_vl_vision.py tests/checkpoint/test_ftw_gguf_banks.py --ignore=tests/models/test_glm5_next_kda_snapshot.py -q --tb=short -rf` | 155 passed, 1 skipped, 0 failed (20.8s) |
| 4 | `python -m pytest tests/kernels/test_qsa_fp8.py tests/kernels/test_qsa_nvfp4.py tests/kvcache/test_qsa_pool.py tests/kvcache/test_qsa_pool_fp8.py tests/models/qwen4_exp/test_qsa_backend.py tests/models/qwen4_exp/test_qsa_hf.py tests/models/qwen4_exp/test_qsa_kernels.py -q --tb=short -rf` | 67 passed, 1 skipped, 0 failed (5.2s) |

TOTAL: 383 passed / 0 failed / 4 skipped. Classification: no failures -> nothing to triage.
Notes: the baseline qsa_fp8 CUDA-invalid-argument class was quiet this run; the task brief's
"tests/moe/test_gguf_expert_banks.py" actually lives at tests/models/test_gguf_expert_banks.py
(verified by file search). Full logs (ephemeral): /tmp/r3-t-engine.log, /tmp/r3-t-models.log,
/tmp/r3-t-qsa.log.

Census: pre/post every run -> no pytest/ft serve/benchbw/ft_run processes (the "1 run" count is
the census wrapper's own shell line); sem.mp- stable at 18 throughout (known standing baseline).

## Boot-smoke (required: merge touches engine.py weight-load path used by the GGUF fast path)

EXECUTED 18:49:11-18:51:46 BY THE INTERRUPTED PRIOR RUN OF THIS SAME MERGE STATE, re-verified by
this session: worktree engine.py/kernel/utils.py mtimes = 18:41:00 (merge checkout moment) and
blob OIDs identical to the staged merge tree -> the smoke ran against exactly the code being
committed. Runner: boot-smoke-r3.sh (trap-on-EXIT cleanup + FAILPAT watchdog every 3s + 250x3s
/ready poll). Findings from boot-smoke-r3.log (60 lines) + boot-smoke-r3-client.out:

- Primary gguf-786k recipe: bare GGUF UD-Q3_K_XL on :18081, --moe-cache-auto
  --kv-reserve-tokens 786432 --kv-cache-dtype nvfp4 --memory-ratio 0.89 --max-prefill-length 4096
  --moe-strategy hybrid --moe-cpu-threads 16. No fallback needed.
- /ready == 200 at 18:51:09 -> ~118 s boot (inside the 94-132 s bare-GGUF band).
- Slot-gate SILENT (no fail-fast); per-group slots 296/288/288 >= working set 288; the
  "296 < 2*288: prefill overlap disabled" INFO is the known benign overlap disable, not the gate.
  KV pool 788992 tokens. CPU MoE executor pools [39,1,2] carry gguf signatures
  (avx2, iq3_xxs/iq4_xs) -> our gguf paths intact in the merged tree.
- POST /v1/chat/completions with "model" field, max_tokens 8: HTTP 200 + usage present
  (client.out); empty content from the reasoning parser = expected, not an error.
- SIGTERM shutdown: graceful reap of 2 backend workers, "Application shutdown complete",
  "Finished server process [606801]" (exit 143 = by design).
- FAILPAT ('AssertionError|OutOfMemoryError|Backend worker is gone|backend worker .* exited|
  Traceback|CUDA error|ValueError'): zero hits in the log.
- Post-run state: no ft serve process remains; sem.mp- = 18 (standing baseline); GPU idle.

VERDICT: PASS. Invariants hold (slot-gate silent, total slots >= working set, boot time within
band, clean shutdown, census clean); exact VRAM-dependent numbers not compared per protocol
(they depend on starting free VRAM).
