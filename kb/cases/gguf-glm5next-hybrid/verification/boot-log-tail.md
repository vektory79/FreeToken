# Task 05 boot log tail - hybrid_main (REAL file, 2026-09-15)

Command (runner /tmp/ft_task05_runner.sh, driver forces moe_collect_stats=True):

    .venv/bin/python /tmp/ft_serve_driver.py serve \
      --model-path /media/ai/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL.gguf \
      --moe-cache-auto --num-tokens 70016 --memory-ratio 0.85 --max-prefill-length 4096 \
      --moe-strategy hybrid --moe-cpu-threads 16

Log: /tmp/ft_t05_hybrid_main.log

- VRAM before boot: 1348 MiB (idle gate passed at 2000 MiB threshold)
- /ready: 503 during load -> 200; **ready in 136s** (watchdog FAILPAT clean)
- served model (/v1/models): `GLM-5.3-Flash-UD-Q3_K_XL.gguf`

Key boot lines:

    --moe-cache-auto resolved moe_cache_size=1232 num_pages=133 (prefill_overlap=True)
    --moe-hybrid-max-fetch auto: fetching 78.0% of each decode step's expert misses over
    PCIe (benched PCIe/CPU bandwidth ratio), the rest on the CPU      [x3 partitions]
    CPU MoE executor ready [pool 1/3: gguf (18, 18, 23) x39 layers]: threads=6
    (pinned to cores 0..10) isa=avx2+gguf(iq/qk) fmt=iq3_xxs (up=iq3_xxs, down=iq4_xs)
    H=4096 I=2048 experts=288 layers=39 top_k=8 act=swiglu_clamp max_tokens=4
    Free memory after initialization: 3.37 GiB
    Free GPU memory after capturing CUDA graphs: 3.30 GiB   [graphs bs 1,2,4]

- pools 2/3 (gguf (23,23,14) x1, (18,18,14) x2): threads 5 each (even 16-thread split).
- No "fixed fetch cap of 1" fallback line -> auto fraction ACTIVE (assert passed).
- Teardown: SIGTERM -> graceful reap ("Graceful stop: reaping 2 backend worker(s)"),
  scheduler exit clean; VRAM back to 1357 MiB; driver dumped
  /tmp/hybrid_decode_stats.json on exit.
- Same boots for the A/B probes (hybrid_cap6 / hybrid_ov0 / hybrid_hostfunc) and the
  offload control (plain ft, USE_DRIVER=0) all passed the identical smoke recipe.

## benchbw profile refresh (pre-flight, required for the auto fraction)

The cached profile (2026-09-12, v4) predated the Task 03 gguf leg (no "gguf" key ->
the engine would have hit the cap-1 fallback). `ft bench bw` full refresh, 03:24:

    gguf   12.69 MB   CPU-MoE 11.7 GB/s   PCIe-gather 43.6 GB/s   0.27x  offload
       overlapped: CPU-MoE 11.3 + PCIe 40.1 GB/s -> hybrid fetches 78.0% of misses

Reproduces Task 03's measured leg (11.71 / 43.45 / 0.781); headline verdict "offload"
is the conservative 2.0-threshold rule, expected and NOT a blocker. Saved to the
machine-local benchbw profile cache (GPU-ec07f396-8269-f150-1f86-0c511e4972fb.json,
not mirrored).

## Full regression (STEP 1)

    uv run --extra dev pytest tests/ -m "not slow" \
      --ignore=tests/models/test_glm5_next_kda_snapshot.py \
      --ignore=tests/server/test_muse_glimmer_parsers.py -q
    => 16 failed, 1867 passed, 171 skipped, 13 deselected in 111.16s (0:01:51)

All 16 map 1:1 onto the known baseline classes (no new failures):
- 7x tests/attention/test_dsa_kpool.py[nvfp4] + 2x tests/kvcache/test_dsa_pool.py[nvfp4]
  + 1x tests/models/test_glm_dsa.py::test_backend_ragged_prefill_identity_and_selection
  = the "No module named 'tests'" import class (10x)
- 2x tests/models/test_glm_dsa.py (sparse_kernel / splitk) = triton OutOfResources class
- 1x tests/kernels/test_pinned_tensor.py UVA (cudaHostGetDevicePointer)
- 1x tests/kernels/test_qsa_fp8.py CUDA invalid argument
- 1x tests/models/qwen4_exp/test_ple.py flaky bitwise
- 1x tests/moe/test_offload.py graph-capture FakeOffloadCache rot (graph.py:124,
  "'FakeOffloadCache' object is not iterable")
Known classes that did NOT fire this run: test_prefill_hit_d2d cudaMemcpyBatchAsync (3x),
nvfp4_backends test_b12x order-flake.
