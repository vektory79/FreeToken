# Wave 1 notes: N-split of kda_in_proj (+ fix wave) - anchor for the A/B wave

Branch vektory79. Wave 1 nominal HEAD 1706aae; actual HEAD cb8db23 (two .veai-only
commits - the python diff vs 1706aae is identical, review-b B1). All wave-1 + fix-wave
changes are UNCOMMITTED by design: the commit is user-gated and expected only after
the A/B wave validates. No csrc/, no setup.py, no kernel JIT touched (python-only).

## 1. What wave 1 changed (4 files, +241/-2)

- python/freetoken/layers/gguf.py (53/1): `_kda_nsplit_env()` (~:112) - strict
  whitelist env read, PER CALL; `kda_in_proj_forward()` (~:124) - the split helper;
  `__all__` adds `kda_in_proj_forward` only; the env name is read at exactly one
  site and consumed at exactly one site (single source of truth).
- python/freetoken/models/glm5_next/kda.py (6/1): the fused in_proj call site
  routes through `kda_in_proj_forward` (~:111-116). Production in_proj has_bias=False.
- tests/kernels/test_gguf_quant.py (143/0): helpers `_spy_kda_a8` / `_kda_nsplit_layer`
  + 4 knob tests (launch-structure spy via lazy module-attr monkeypatch).
- tests/models/test_glm5_next_kda_op.py (39/0): real-op non-GGUF passthrough test.

### Fix-wave delta (this file's companion; review TPs NA-F1/NA-F2, NB-F1/NB-F3)

- gguf.py comment reword (NA-F1): the old comment implied 32-alignment drives
  exactness. Correct statement now in code: the split sits on a 32-row tile edge to
  keep need_check=false; that flag is a PERF guard keyed on N%32, NOT an exactness
  condition - bitwise parity rests on the disjoint 32-col tiles, the k-only
  per-element reduction and the deterministic per-launch x-quant. No code change.
- 3 test additions (NA-F2): `test_kda_nsplit_parity_m_tail_batches`,
  `test_kda_nsplit_parity_noncontig_x`, and the "" env pin inside
  `test_kda_nsplit_env_knob`. No new files (mirror rule).
- Durability (NB-F1/NB-F3): this file + gate-nsplit-run2.log (see 6).
- SKIPPED with justification (NA-F2 partial): q6_K-through-split case (dense linears
  are Q8_0-only in production, glm5_next/gguf.py:693 - unreachable) and an op-level
  GGUF integration test (covered on hardware by the A/B wave's T46 liveness
  assertion; NB-F4 defers the wiring pin to that assertion).

## 2. Knob contract (FREETOKEN_GGUF_KDA_NSPLIT)

- Per-call `os.environ.get` inside `_kda_nsplit_env` - no module state, no caching;
  flipping the env between calls flips the launch structure (test-pinned).
- Values: unset / "" / "1" -> 1 (single launch); "2" -> split; anything else ->
  ValueError BEFORE any launch. Strict whitelist: no trim, no numeric parsing -
  " 2", "02", "+2", "12", "0", "-2", "abc" all fail fast (v3a ft_gguf_moe_mtile
  pattern). "" pinned explicitly: behaves exactly like unset.
- Scope gates (helper): isinstance GGUFLinear AND GEMM batch n_tok > 6
  (n_tok IS the GEMM M = hidden_states.shape[0]; 2D [total, H]; decode bs<=6 stays
  on MMVQ vec kernels) AND quant type in _MMQ AND N >= 64 AND N % 32 == 0.
  Anything outside -> `layer.forward` fallback = byte-identical pre-change behavior.
  The gate matches fused_mul_mat_gguf's MMQ dispatch exactly (no divergence).
- Split geometry: half = (N//2)//32*32; production N=24896 -> two 12448-row
  launches (12448 = 389 x 32; 2 x 54.2 MB weights < 96.0 MiB L2). Assembly:
  preallocated [M, N] out + two narrow `copy_` writes (the extension API has no
  out= param). Bias applied exactly once, after assembly.
- EXACTNESS (why torch.equal, no tolerance): dim-0 narrow of the [out, row_bytes]
  qweight is a pure byte view (per-output-row packing, no repack; chunk 2's base
  stays 256-byte aligned at production row_bytes=4352); the MMQ kernel reduces
  k-only per element (no cross-row accumulation; the launcher splits the grid along
  the X/N dimension only); x is quantized to q8_1 internally PER LAUNCH
  deterministically (same x tensor, same quant bytes in both launches). need_check
  is keyed on N%32 (perf guard), M-tails are handled identically in both launches.
- Do NOT flip the knob after boot: decode CUDA graphs (engine/graph.py
  GraphRunner._capture_graphs) capture the model forward for bs>6 and bake whatever
  launch structure was live at capture time; replay never re-reads the env.
  Selection happens ACROSS boots, not mid-boot. The A/B wave uses one env value
  per boot.
- Production reach: glm5next KDA in_proj only (kda.py:116). Dense linears are
  Q8_0-only in the loader (glm5_next/gguf.py:693); q6_K appears only on GGUFLMHead
  which never routes through the helper. Other in_proj owners
  (qwen4_exp/gdn.py:149, qwen3_5_moe/gdn.py:147) use LinearColParallelMerged, not
  GGUFLinear - no other consumer can reach the helper.
- Env-only plumbing is sufficient for the A/B: the knob is consumed inside the
  model forward; the ft serve worker inherits the launcher env; benchbw/e2e
  harnesses need nothing.

## 3. Parity / boundary test inventory (tests/kernels/test_gguf_quant.py)

All use torch.equal / launch-structure asserts, not tolerances:

1. test_kda_nsplit_env_knob: unset -> single [96]; "" -> single (NEW: exactly like
   unset, default 1); "2" -> [32, 64]; "1" flips back per call; 7 invalid values
   ("12","0","-2","abc"," 2","02","+2") raise ValueError pre-launch.
2. test_kda_nsplit_bitwise_parity: production N=24896/K=4096/M=256; knob=1 ==
   plain GGUFLinear.forward; knob=2 [12448,12448] == knob=1 BITWISE.
3. test_kda_nsplit_parity_m_tail_batches (NEW, NA-F2a): M=7/13 - batch > 6 (MMQ
   dispatch) but not a multiple of mmq_x=4, so both launches ride the m-tile tail
   path (clamped y-loads, guarded dst writes); production N=24896; split == single
   BITWISE.
4. test_kda_nsplit_parity_noncontig_x (NEW, NA-F2b): strided x = base[::2]
   (13x512 view, row stride 2x512), production N=24896/K=512; the per-launch
   .contiguous() inside fused_mul_mat_gguf is deterministic, split == single
   BITWISE; asserts not x.is_contiguous() first.
5. test_kda_nsplit_boundary_math: N=48/32 ragged -> fallback single launch
   (byte-identical); N=64 -> [32,32]; N=96 -> [32,64]; all bitwise; non-GGUF
   passthrough (SimpleNamespace forward).
6. test_kda_nsplit_decode_batch_stays_mmvq: M=4 with knob=2 stays on the vec
   kernel, byte-identical vs fused_mul_mat_gguf.
7. (tests/models/test_glm5_next_kda_op.py) test_nsplit_knob_non_gguf_passthrough:
   real Glm5NextKDA op with a non-GGUF in_proj, knob=2 -> torch.equal passthrough
   vs a re-instantiated op.

Known deferred pins (NB-F4, triage): no end-to-end GGUF wiring test in the tree -
the A/B wave's T46 nsys liveness assertion (kda launches 34 -> 68/chunk with
knob=2) is the on-hardware wiring pin. Carry it in the A/B brief.

## 4. Copy-overhead arithmetic (wave-1 assembly cost, NB-F1/NB-F3)

Production chunk: n_tok=8128, N=24896, bf16, 34 kda_in_proj calls/chunk.

- Per call: 2 copies, each 8128 x 12448 x 2 B = 202.35 MB -> 404.7 MB payload/call.
- Per chunk: 34 calls x 2 = 68 copies -> 68 x 404.7 MB = 27.5 GB traffic.
- At copy-class BW: 1638 GB/s (step0 effective) -> 16.8 ms; ~1.8 TB/s -> 15.3 ms.
- 15.3-16.8 ms on a 14.87 s chunk = 0.10-0.12%. Net > 14x positive vs the 0.24-0.28 s
  dense-term target. dst is a strided view (out[:, :half]) - elementwise strided
  copy, ~BW-bound; efficiency loss only makes the estimate conservative.

## 5. GROSS vs NET A/B target framing (NB-F3 - read before judging the A/B)

- GROSS target: dense-term recovery 0.24-0.28 s/chunk. CONSERVATIVE: the ~0.5 s
  figure was ~2x optimistic (the same shape's own M-sweep caps at 19.6-19.9 TF/s;
  review-a F4). Anchors: TASK.md:106, step0-profile.md:23/:226,
  review-a-step0.md:159/:189, triage-step0.md:17/:58.
- NET target: ~0.22-0.26 s/chunk after subtracting the verified 15.3-16.8 ms
  assembly overhead.
- Judgement rule: compare the MEASURED dense-term delta against the NET window.
  A recovery between 0.22 and 0.28 s with launches present at 68/chunk is the
  expected outcome; do not fail the wave merely because measured recovery sits
  below GROSS. The upside may be larger (tile-invariance of the 22 TF/s rate is
  untested; the v3a sibling family grew 13.7 -> 39.6 TF/s from tile 4 -> 32).
- Method guard (campaign lesson): derive floors/totals from measured op sums with
  per-launch x launch-count; a projection below its own floor = broken analysis.
  Note the final full prefill chunk's input-throughput line is bogus (~1552-1602
  tok/s) - use the median of full chunks minus the last.

## 6. Evidence durability + hygiene (NB-F1)

- gate-nsplit-run2.log (this folder): durable copy of /tmp/gate-nsplit.log made
  2026-09-19 before TTL expiry. 21-line invocation header (source stats, exact
  command reconstruction with its caveat, date, HEAD cb8db23, numstat +241/-2,
  env) + the 775-line log. Gate result: 16 failed / 2173 passed / 206 skipped /
  15 deselected in 175.63s - all 16 Environment baseline (11x import class,
  pinned_tensor UVA, qsa_fp8 [1-1-64], qwen4_ple :421, 2x glm_dsa OOR); zero
  non-baseline failures (review-b B3, verified per-test from the tracebacks).
  The log starts mid-run (no pytest header): the exact command is reconstructed
  from TASK.md:156 and flagged as such in the header - do not treat it as
  verbatim provenance.
- Census (fix wave, 2026-09-19): PRE - no ft serve / nsys / pytest processes
  (only the census shell itself), VRAM 1432 MiB used / 32607 MiB; POST - clean,
  zero survivors, VRAM 1451 MiB / 32607 MiB (documented IDE baseline ~1.4-1.5 GB).
- Fix-wave test verdict (2026-09-19, GPU, serialized): tests/kernels/test_gguf_quant.py
  76 passed / 0 failed, exit 0 (74 nodeids pre-fix + 2 new tests; file 799 -> 850
  lines); tests/models/test_glm5_next_kda_op.py 4 passed / 0 failed, exit 0.
  Full gate NOT re-run for a comment+tests change - the A/B wave's boot
  exercises the serving path.

## 7. A/B wave brief (must-carry list from triage-nsplit.md Status)

- Two boots: env unset vs FREETOKEN_GGUF_KDA_NSPLIT=2; MTILE=32 anchor both sides;
  port 18801; T43/T44/T46 discipline; T44 validity gate vs the 546.5 tok/s class
  (MTILE=32 anchor; 556.43 is the MTILE=4 reference).
- T46 liveness assertion: kda launches 34 -> 68/chunk with knob=2 (this is also
  the NB-F4 wiring pin).
- Decode graphs: re-captured + decode re-measured (dense MMQ rides bs>6 graphs;
  capture bakes the launch structure). Production A/B at max-running-requests=1
  replays the bs=1 graph (MMVQ vec path) so decode should stay flat in practice -
  measure it anyway.
- Cross-boot bitwise output diff on a few prompts (parity contract beyond unit
  tests).
- Judge against the NET target (5), not GROSS. One env value per boot - no
  mid-boot flips.

## 8. Open at save time

- Commit remains user-gated, only after the A/B wave validates (campaign protocol).
- Candidate A (dense m-tile widening) A/B follows this wave; Candidate B (int8 mma,
  D06 gate) stays gated behind Candidate A's outcome.
