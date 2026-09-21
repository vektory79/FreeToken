# W1 code-site map: CPU GEMV for GGUF IQ3_XXS / IQ4_XS / Q6_K in the CPU MoE executor

Read-only research, 2026-09-14. All paths relative to repo root `/media/ai/src/FreeToken`.
Goal: exact touch points to add CPU GEMV for gguf types IQ3_XXS(18), IQ4_XS(23), Q6_K(14)
so `--moe-strategy hybrid` serves glm5next GGUF (H=4096, I=2048, E=288, top_k=8;
per-layer gate/up IQ3_XXS, down IQ4_XS; ckpt 11/12/44 variants use IQ4_XS/Q6_K).

Two task-spec premises corrected by this pass:
- The "q4_0 resolver" does NOT receive a GgufTensor; it receives the offload cache's
  materialized pinned host banks (`cache.bank_sources`), one `[E, rows, row_bytes]`
  uint8 tensor per (bank, layer), and returns C++ pointer tables + geometry.
- The real blocker for heterogeneous per-projection types is not inside the resolver:
  the executor takes ONE `weight_format` for both projections, AND the engine refuses
  multi-signature partitions for cpu/hybrid (engine.py:946-949). Both need W1/W4 work.

## 1. python/freetoken/moe/cpu_executor.py (675 lines)

- `_WFMT_IDS` (line 72): `{"bf16": 0, "nvfp4": 1, "mxfp4_triton": 2, "ds_fp4": 3, "q4_0": 4}`.
  Comment (71): "Weight-format ids must match WFmt in csrc/cpu_moe/cpu_moe_ext.cpp".
- `_ACT_IDS` (60-69): silu/swish=0, gelu=1, gelu_tanh=2, gpt_oss_swiglu/swigluoai=3,
  swiglu_clamp=4. glm5next uses silu (act id 0) + swiglu clamp via swiglu_alpha/limit.
- `CpuMoeExecutor.__init__` (168-319): `fmt = fmt or cache.quant_format` (175);
  raises `NotImplementedError` for formats outside `_WFMT_IDS` (176-181, message points
  to --moe-strategy offload); ABI probe `max_generic_act_id` (186-199); bank resolution
  at 199:
  `self._resolve_banks({canonical_role(name): per_layer for name, per_layer in cache.bank_sources.items()}, fmt)`
  - `canonical_role` (moe/legacy_format.py:17-20) is IDENTITY for "gate"/"up"/"down"
    (it only aliases gate_up_packed/gate_up_blocks -> "gate_up", *_scales -> *_scale).
    So a gguf partition's banks arrive as keys {"gate","up","down"} while every existing
    resolver reads `banks["gate_up"]`/`banks["down"]` - the gguf resolver must handle
    the three separate banks itself.
- `_make_table` (321-333): builds the per-layer int64 base-address table the C++ side
  indexes as `tbl[layer_id]`; keeps tables + layer tensors on `self._banks` as GC guard.
- `_resolve_banks` (336-405): per-fmt dispatch -> returns (ptrs dict, (H, I)); unused
  pointer kwargs are 0. Signature: `(self, banks: dict, fmt: str) -> tuple[dict, tuple[int, int]]`.
- **q4_0 resolver `_resolve_q4_0_banks` (407-432)** - the pattern to mirror. Receives
  `banks: dict` (role -> list of num_layers `[E, ...]` uint8 tensors); reads shapes from
  `banks[0]` only. Asserts dtype uint8, `H % 32 == 0 and I % 32 == 0`,
  `gate_up.shape[2] == (H//32)*18`, `down.shape[2] == (I//32)*18`; returns
  `dict(gate_up_ptr=table, down_ptr=table, *_scale/_global/_bias ptrs = 0), (H, I)`.
  Docstring: "the C++ W4A16 GEMV reads a row in place (18 bytes / 32 K)".
- Python-vs-C++ dispatch: there is NO python GEMV fallback. All formats go to the C++
  extension (`from freetoken.kernel import _cpu_moe`, 170); "fallback" exists only
  inside C++ as the scalar ISA tier. Resolvers are pure geometry validation.
- Decode plumbing (537-606): `decode` = `decode_submit` + `decode_sync`.
  `decode_submit(layer_id, hidden_states, topk_weights, topk_ids)` (543-588): D2H into
  pinned `x` bf16 [bs,H], `ids` int32 [bs,top_k] (**may carry -1; C++ skips id<0**),
  `w` f32; `create_task(layer_id, bs, ptrs...)` cached per (layer, bs); flag-slot memop
  submit or `submit_with_cuda_stream`. `decode_sync` (589-606) waits + H2D `y`.
- Second construction site: moe/benchbw.py:471 (`_build_cpu_moe_executor`) and 570;
  `_cpu_moe_bank_sources` at benchbw.py:339 synthesizes banks per format (W2 touch).

## 2. python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp (2160 lines)

- Overview header (1-19). Includes torch/extension.h + cuda_runtime_api.h (18-19).
- **Binding = pybind11 torch extension** (`PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)`,
  2114-2160; module name `_cpu_moe`; built by root setup.py alongside _pinned_tensor,
  NOT JIT). Data crossing the boundary is RAW POINTERS ONLY, never tensors:
  - ctor: 10 ints + 8 `uintptr_t` args (each the address of a per-layer int64 pointer
    table from `_make_table`, resolved via `tbl_at(tbl, layer_id)` 1236-1239; null
    table = unused bank) + 2 doubles + `std::vector<int> core_ids` (2145-2158).
  - `create_task(layer_id, num_tokens, x_ptr, ids_ptr, w_ptr, y_ptr)` (2159-2163 area;
    impl ~1546-1556) stores raw pinned IO pointers in a heap `MoeTask`; GIL released on
    submit/sync/run_task.
  - Free functions `memops_probe/memop_submit/memop_sync` (2168-2174) for the flag
    handshake, `max_generic_act_id` ABI marker (2176-2180).
- **Per-format organization**: function POINTER per format + ISA-tier selector, not a
  dispatch table: `dot_fn` (bf16), `nvdot_fn`/`nvi8dot_fn` (nvfp4 fp32 / W4A8),
  `dsdot_fn`, `mxgemv_fn`, `q4dot_fn`; selectors `select_dot/select_nvdot/
  select_nvi8dot(756)/select_dsdot/select_mxgemv/select_q4dot(1216-1221)` called once
  in the ctor (1391-1397). `enum WFmt { WF_BF16=0, WF_NVFP4=1, WF_MXFP4=2, WF_DSFP4=3,
  WF_Q4_0=4 };` (1223). Row-level dispatch is `fmt`-branched inline fns:
  `gemm1_dot` (1484-1503) and `gemm2_dot` (1506-1523). Whole-pass overrides:
  `do_pass1` (1582) routes to `do_pass1_mxfp4` / `do_pass1_dsfp4` (1784-1817) else the
  generic bf16/nvfp4/q4_0 pass; `do_pass2` likewise.
- **nvfp4 W4A8 path**: `select_nvi8dot` (756) + `use_vnni = (fmt==WF_NVFP4) && nvi8dot`
  (ctor 1395-1397); activations pre-quantized per-16-block to int8 [even(8),odd(8)] by
  `quant_i8_pg16` (1462-1478) into `xi8/gi8 + xas/gas` scratch (ctor 1424-1443);
  VPDPBUSD-based `nvi8dot` takes packed weights + fp8-e4m3 block scales + fp16 global.
- **q4_0 CPU path end-to-end** (template for a new block format):
  1. Block dot `q4_0_dot_i8_scalar` (1123-1146), `q4_unpack32` + `q4_0_dot_i8_avx2`
     (1164-1186), `q4_0_dot_i8_vnni` (1187-1214); fp16 scale via F16C `q4_scale`
     (1150-1153); ISA selection `select_q4dot` (1216-1221).
  2. Ctor: stride fields `q4_gu_row_bytes=(H/32)*18, q4_dn_row_bytes=(I/32)*18`
     (1404-1411), `use_q4a8=true` (1419), Q8_0 activation scratch (1424-1443).
  3. Activation prequant once per token/route: input in `submit()` (1941-1947) via
     `quant_q8_0` (1463-1481); intermediate in `prep_g_row` (1826-1831). The
     prepare phase runs between pass1 and pass2 in `run_task_body` (1901-1923) behind
     `needs_di || use_q4a8`.
  4. `gemm1_dot` WF_Q4_0 branch (1494-1498): `w = gu_packed + (e*2I+row)*q4_gu_row_bytes;
     return q4dot(w, xi8, xas, H)`; `gemm2_dot` (1516-1520) symmetric over I.
  - W4A8 is a performance choice, not a contract: ds_fp4 runs a plain fp32
    deinterleave dot (`do_pass1_dsfp4`, 1784-1817) with bf16 rounding to match the
    reference (`f32_to_bf16` before swiglu, 1801-1811; down: round each weighted
    route before summing, 1855-1858) - the IQ formats can do the same.
- **Adding a new block format** = (a) WFmt enum value; (b) `_WFMT_IDS` entry; (c)
  resolver in cpu_executor.py (geometry asserts + pointer tables); (d) scalar + AVX2
  dot fn(s) + selector (LUTs as host arrays - see section 3 note on vendored files);
  (e) ctor init: strides + scratch (none needed if no activation prequant); (f)
  `gemm1_dot`/`gemm2_dot` branches (+ `do_pass1/do_pass2`/`prep_g_row` hooks only if
  a prepare phase is wanted); (g) H/I divisibility check (K-quant blocks are 256 wide
  -> require H % 256 == 0 && I % 256 == 0; true for glm5next 4096/2048).

## 3. kernel/csrc/gguf/dequantize.cuh + ggml-common.h (vendored source of truth)

`dfloat` = **half** (ggml-common.h:930-931) - the fp16-chain precision contract.
Block structs all in ggml-common.h; kernels in dequantize.cuh; type-id dispatch
`ggml_get_to_cuda` (dequantize.cuh:540-583): case 14 -> q6_K, 18 -> iq3_xxs, 23 -> iq4_xs.

- **Q6_K (type 14, 256 elems, 210 B)**: struct (ggml-common.h:104-110)
  `uint8 ql[128]; uint8 qh[64]; int8 scales[16]; half d;` - **d is LAST (bytes 208:210)**.
  Kernel `dequantize_block_q6_K` (dequantize.cuh:321-345): 64 threads, ip=tid/32,
  il=tid%32, is=8*ip+il/16; base `y + i*QK_K + 128*ip + il`; for l in {0,32,64,96}:
  `y[l] = d * scales[is + l/32... (sc[0],sc[2],sc[4],sc[6])] * ((int8)((ql[l/32*32... ] & 0xF) | (((qh>>(l/32))&3)<<4)) - 32)`
  precisely: `ql = x.ql + 64*ip + il`, `qh = x.qh[32*ip+il]`, `sc = scales + is`;
  y[0] uses ql[0]&0xF | ((qh>>0)&3)<<4 with sc[0]; y[32] uses ql[32] with sc[2];
  y[64] uses ql[0]>>4 with sc[4]; y[96] uses ql[32]>>4 with sc[6]. Value = d*sc*(q-32),
  q in [-32,31].
- **IQ3_XXS (type 18, 256 elems, 98 B)**: struct (ggml-common.h:137-141)
  `half d; uint8 qs[3*QK_K/8 = 96];` - **d FIRST (bytes 0:2)**; the 8 per-32 scale
  words live INSIDE qs at bytes 64:96 (kernel reads `gas = (uint16*)(qs + QK_K/4) + 2*ib`).
  Kernel `dequantize_block_iq3_xxs` (dequantize.cuh:477-496): 32 threads, il=tid/8,
  ib=tid%8; `q3 = qs + 8*ib`; `aux32 = gas[0] | gas[1]<<16`;
  `d = half2float(d) * (0.5f + (aux32>>28)) * 0.5f`;
  `signs = ksigns_iq2xs[(aux32 >> 7*il) & 127]`;
  `grid1/2 = (uint8*)(iq3xxs_grid + q3[2*il+0/1])` (grid entries are uint32, 4 bytes);
  `y[32*ib + 8*il + j]   = d * grid1[j] * (signs & kmask_iq2xs[j] ? -1 : 1)`  j=0..3
  `y[... + j + 4]        = d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1 : 1)`
  LUTs (host copies needed for CPU; all `static const __device__` in ggml-common.h,
  verbatim-vendored - DO NOT edit, add C++ copies in cpu_moe_ext.cpp):
  `iq3xxs_grid[256]` uint32 (ggml-common.h:563), `ksigns_iq2xs[128]` uint8 (:888),
  `kmask_iq2xs` (nearby), `kvalues_iq4nl[16]` int8 (:927), `ksigns64[128]` uint64 (:897).
- **IQ4_XS (type 23, 256 elems, 136 B)**: struct (ggml-common.h:181-187)
  `half d; uint16 scales_h; uint8 scales_l[QK_K/64 = 4]; uint8 qs[128];` - d FIRST.
  Kernel `dequantize_block_iq4_xs` (dequantize.cuh:545-560): 32 threads, il=tid/8,
  ib=tid%8; `q4 = qs + 16*ib + 4*il`;
  `d = half2float(d) * ((((scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((scales_h >> 2*ib) & 3) << 4)) - 32)`;
  `y[32*ib + 4*il + j]  = d * kvalues_iq4nl[q4[j] & 0xf]`
  `y[32*ib + 4*il + 16 + j] = d * kvalues_iq4nl[q4[j] >> 4]`   j=0..3.
- W4A16 dot references (pair with Q8_1 activations, sign trick): vecdotq.cuh
  `vec_dot_q6_K_q8_1` 1625 (impl_mmvq 485), `vec_dot_iq3_xxs_q8_1` 1848 (uses ksigns64),
  `vec_dot_iq4_xs_q8_1` 2017. These are the *dot* form; the dequantize.cuh formulas
  above are the *value* contract (use them + the dfloat bound below for parity tests).

## 4. How the hybrid path calls the CPU executor today

- Entry: `layers/moe.py` `OffloadMoELayer._decode_routed` (271-297). Decision tree:
  `cache.is_cpu_layer(layer_id)` -> `cache.cpu_executor.decode(layer_id, hidden,
  topk_weights, topk_ids)` (284-289); `cache.decode_target == "hybrid"` ->
  `_decode_hybrid` (290-292); else pure GPU offload (`ensure_experts` + `copy_missing`
  + `_expert_gemm`, 293-297).
- `_decode_hybrid` (299-339) - the function that computes expert rows on CPU on a miss:
  `raw = topk_ids.clone()`; `cache.ensure_experts_hybrid(layer_id, topk_ids)` rewrites
  ids in place to GPU slot ids or -1 (miss); `cpu_ids = where(on_gpu, -1, raw)`
  (316-317); **`pending = executor.decode_submit(layer_id, hidden_states, topk_weights,
  cpu_ids)` (317)** kicks the CPU pool BEFORE `cache.copy_missing()` (323) + the GPU
  `_expert_gemm` over hits/fetched (323-331, GPU weights zeroed on CPU routes); merge
  `gpu_routed + cpu_routed` (339). Overlap A/B knob `FREETOKEN_HYBRID_OVERLAP` (321-325).
  The CPU side gets only (layer_id, hidden [bs,H] bf16, topk_weights f32, cpu_ids int32
  with -1 for GPU routes) - no bank knowledge, it reads `cache.bank_sources` via the
  pointer tables built at construction.
- Executor construction: `engine/engine.py::_init_cpu_moe_executor` (939-975) builds
  ONE `CpuMoeExecutor(cache=caches[0], top_k=..., activation=..., num_threads=
  config.moe_cpu_threads, max_tokens=max(max_running_req, cuda_graph_max_bs),
  swiglu_alpha/limit=sample.alpha/.limit, fmt=sample.quant_method.cpu_format or None
  -> cache.quant_format)` and `cache.set_cpu_executor(executor)` (974). Called only
  when `caches[0].decode_target in ("cpu","hybrid")` (940).
- **Hard blocker for the real file: engine.py:946-949** - `if len(caches) > 1 and
  decode_target in ("cpu", "hybrid"): raise ValueError("multi-signature expert banks
  serve the gpu decode target only; cpu/hybrid executors are per-format and cannot
  span partitions")`. glm5next has 3 signature groups -> W1/W4 must make executors
  per-partition and lift this.
- Config gate (W3 touch point): `engine.py:1764-1778` rejects `moe_weight_format ==
  "gguf"` with strategy cpu/hybrid/fused or --moe-cpu-layers ("gguf moe_weight_format
  supports offload only"). The #186 clamp follows at 1779-1790 (keep).
- GPU counterpart of the same layer (hybrid hits): `_expert_gemm` gguf branch,
  layers/moe.py:429-459 - per-projection `ggml_moe_a8_vec(hidden, bank, ids, top_k,
  TYPE, N, tokens)` with `gate_t, up_t, down_t = cache.gguf_types[self.layer_id]`
  (433), gated epilogue, #186 `_assert_moe_vec_chunk` (476-483). Prefill stays on this
  GPU path; W1 is decode-only.

## 5. Bank bytes exposure to the CPU consumer

- The CPU executor reads the **pinned HostBanks** (`cache.bank_sources`), NOT an mmap
  of the gguf file. HostBank = lazy anonymous mmap filled via chunked O_DIRECT then
  `cudaHostRegister` (pin-after-fill; moe/host_banks.py:1-45 docstring; PinFailed at 40).
  Pinned-ness is enforced at `set_bank_sources` (offload_cache.py:446: "pin the bank
  before set_bank_sources") because the fused gather needs device aliases (UVA);
  the CPU executor additionally requires them for its raw pointer tables.
- `OffloadMoeCache.bank_sources: dict[str, list[torch.Tensor]]` (offload_cache.py:224):
  role -> one `[E, rows, row_bytes]` uint8 tensor PER LAYER (layer-resident attributes);
  registered by `set_bank_sources` (306-386) against `_BANK_SCHEMAS["gguf"] =
  ("gate", "up", "down")` (offload_cache.py:77-80: "the type rides the cache's
  gguf_types, not the schema").
- **Per-projection gguf_types**: stored as `OffloadMoeCache.gguf_types: tuple | None`
  (offload_cache.py:156-157), one (gate_t, up_t, down_t) int triple PER LOCAL layer,
  role-ordered. Provenance: ExpertBanks.gguf_types (moe/expert_banks.py:53) filled by
  `_gguf_banks` <- `load_gguf_moe_expert_sources` (models/weight.py:266-278 returns
  `(banks, gguf_types)`); engine partitioning slices it per group: `gguf_types=(tuple(
  banks.gguf_types[l] for l in members) ...)` (engine.py:853-856). Reachable from the
  executor's cache object TODAY, but the executor never reads it - it only takes the
  single `cache.quant_format`. Within one signature partition all layers share the
  triple, so per-projection types are constant per executor instance; gate/up always
  share a type in every real signature (18,18,23 / 18,18,14 / 23,23,14).
- Bank geometry (expert_banks.py:185-210 dummy path documents it): rows: gate=I, up=I,
  down=H; ne0 (row width, blocks pack over it): gate=H, up=H, down=I; row_bytes =
  ne0//256 * block_bytes (98/136/210). Conservative byte estimate `_BANK_BYTES_PER_
  EXPERT["gguf"]` uses exactly these extremes (offload_cache.py:98-111).

## 6. Existing tests

- tests/moe/test_cpu_moe_q4_0.py (210 lines): THE format-template. `_pack_q4_0`
  (31-45) packs synthetic banks; `_make_q4_0_cache` (63-79) builds a SimpleNamespace
  cache with pinned `bank_sources`; `test_cpu_decode_q4_0_matches_dequant_then_gpu`
  (83+) compares CPU GEMV vs reference dequant + production bf16 GPU decode on
  byte-identical banks; part 2 covers graph capture/replay. Clone this shape for IQ.
- tests/moe/test_cpu_moe.py: bf16/nvfp4 executor coverage; test_hybrid_fetch.py:
  hybrid ensure/fetch split semantics; test_offload.py: cache machinery.
- tests/models/test_gguf_expert_banks.py (1261 lines): `_craft_uniform_gguf_banks`
  (204-234, analytic fixtures: Q8_0 -> +0.25, Q6_K -> -16, d-LAST at 208:210);
  `test_cpu_executor_does_not_claim_gguf` (127) asserts gguf NOT in _WFMT_IDS -
  **must be flipped** when W1 lands; `test_expert_gemm_gguf_per_projection_heterogeneity`
  (957) proves per-projection type dispatch; partitions: 529/546/744/799;
  `test_adjust_config_clamps_gguf_prefill_chunk` (406) guards the #186 clamp;
  bank-bytes tests 1018-1206. `_FP16_SCALE_FIELDS` lives in
  **tests/kernels/test_gguf_quant.py:53-59** (not tests/models/: Q8_0 [0:2], Q6_K
  [208:210], Q3_K [108:110], Q4_K [0:4], IQ3_XXS [0:2], IQ4_XS [0:2]).
- tests/kernels/test_gguf_quant.py (283 lines, CUDA-gated, clang++ host per
  kernel/gguf.py:27-39): `_DEQUANT_FACTOR` (65-72: Q6_K 127*31, IQ3_XXS 8*118,
  IQ4_XS 32*127); `_finite_fp16_chain_reference` (96-120) draws half scales from
  [0.25, 2] so the fp16 chain stays finite BY CONSTRUCTION (reuse for CPU fixtures);
  tolerance test `test_dequant_matches_gguf_py_within_fp16_rounding` (141-162):
  per-element bound `8 * 2**-11 * _DEQUANT_FACTOR[qtype] * |d| + 1e-3` vs the fp32
  gguf-py reference (`gguf.dequantize`) - THE accepted contract for any new dequant;
  `test_moe_vec_parity_vs_torch_reference` (165+) covers `ggml_moe_a8_vec` per type
  (kernel/gguf.py:79 ggml_dequantize, 118 ggml_moe_a8_vec - the parity harness to
  reuse for CPU-vs-CUDA row comparisons).
- **models/gguf/dequant.py claim VERIFIED**: GGML_IQ3_XXS(18)/GGML_IQ4_XS(23) have
  BLOCK_SHAPE entries only, comment (39-43): "geometry only (row_bytes + log spelling).
  There is deliberately NO python reference dequant for these" and dequantize() raises;
  Q6_K HAS a full reference (docstring 1-4). The IQ3_XXS/IQ4_XS python reference lives
  in the venv gguf-py `quants.py` (proven vs CUDA, cos 0.99994-0.99997, Phase 6 bisect).

## Wire-up plan sketch (minimal W1 touch points)

1. `cpu_moe_ext.cpp` (the only place CPU dequant may live; vendored .cu untouched):
   WFmt += WF_IQ3_XXS=5, WF_IQ4_XS=6, WF_Q6_K=7; host LUT copies (iq3xxs_grid,
   ksigns_iq2xs, kvalues_iq4nl - source values from ggml-common.h, copied verbatim);
   per-format dot fn(s) (scalar + AVX2 tier + selector) doing fp32 block dots over
   bf16 activations (no activation prequant -> no prt phase, no new scratch, no
   W4A8); ctor row-stride fields (K/256 * {98,136,210}) + H/I %256 checks;
   `gemm1_dot`/`gemm2_dot` branches.
2. Gate/up separation (design decision): C++ assumes one gate_up table with up at
   row I+i (do_pass1 1600-1640), but gguf banks are SEPARATE gate/up tensors.
   Minimal option: add `gate_ptr`/`up_ptr` ctor args + branch in the generic pass1
   for the IQ fmts (down machinery reusable as-is). Alternative: stack gate+up into
   one [E,2I,rowbytes] pinned tensor per layer in the resolver (extra host copy,
   doubles gate/up host RAM - worse).
3. `cpu_executor.py`: `_WFMT_IDS` += the 3 ids; gguf resolver `_resolve_gguf_banks`
   (mirror `_resolve_q4_0_banks` 407-432) keyed on "gate"/"up"/"down", reading the
   per-projection types from `cache.gguf_types[0]` (uniform per signature partition)
   and validating `row_bytes == ne0//256 * block_bytes`; ctor gains a
   (gate_fmt, down_fmt) pair or per-projection weight_format since gate/up != down.
4. `engine.py`: loop `_init_cpu_moe_executor` over ALL caches (not caches[0], 940)
   and lift the multi-signature ValueError (946-949); relax the config gate
   (1764-1778) to consult the resolver (W3).
5. `layers/moe.py`: no change - `_decode_hybrid` already goes through
   `cache.cpu_executor` per layer and each OffloadMoELayer is routed to its own
   signature cache with local layer ids (engine.py:887-902).
6. `benchbw.py`: `_CPU_MOE_FORMATS` (line 66) += gguf ids (W2, feeds the auto fetch
   fraction via load_hybrid_fetch_fraction).
7. Tests: new tests/moe/test_cpu_moe_gguf_iq.py cloning the q4_0 template with
   `_craft_uniform_gguf_banks`-style analytic fixtures (scales in [0.25,2]); flip
   test_gguf_expert_banks.py:127; extend the engine-gate test (406/546 region);
   parity vs gguf-py under the 8*2^-11*factor*|d| + 1e-3 bound.
