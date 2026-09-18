---
name: "ft-gguf-kernel-jit-toolchain"
description: "gguf CUDA kernel JIT needs clang++ host; nvcc 13.3; CC/CXX scoped; pybind optional<Tensor> for None args"
type: project
lastUpdated: 2026-09-17T18:21
lastRecall: 2026-09-17T22:09
---

# gguf CUDA kernel JIT toolchain on vektory79: clang++ host required

Verified 2026-09-13 (Phase 4 of the glm5next GGUF plan); UPDATED 2026-09-15 (hybrid campaign, commit 1635ecd).

- python/freetoken/kernel/gguf.py `_host_compiler()` picks the FIRST of ("clang++", "g++-13", "g++-14", "g++-15") found via PATH; the ONLY override is env `FREETOKEN_GGUF_HOST_CXX`. On this box g++-13/14/15 are all present, so without clang the JIT trips a non-conformant `typename decltype` in torch's ATen `core/List_inl.h` with ANY gcc host. `-allow-unsupported-compiler` does NOT help (only silences nvcc version checks). nvcc + clang++ compiles cleanly.
- nvcc: /usr/local/cuda-13.3/bin/nvcc; torch (2.11.0+cu130) resolves CUDA_HOME=/usr/local/cuda via /etc/alternatives because CUDA_HOME/CUDA_PATH unset and nvcc not on PATH. 13.3 vs cu130 = warning only; gcc bounds for CUDA 13.0 are ((6,0,0),(16,0)) so g++-15 passes the version gate (failure is conformance, not version).
- FIX 1635ecd (2026-09-15): `_module()` no longer permanently force-overwrites process-global `os.environ["CC"]/["CXX"]` - the clang host override is now set around the `torch.utils.cpp_extension.load(...)` call and restored in a `finally` (exact prior state incl. removal). Regression tests in tests/kernels/test_gguf_quant.py: `test_module_load_scopes_cc_cxx_to_the_build` (failing-before verified: on HEAD it leaked {'CXX': '/usr/bin/clang++', 'CC': '/usr/bin/clang'}) and `test_module_load_without_a_host_compiler_sets_no_cc_cxx`. User-exported CXX is no longer clobbered.
- WHY IT MATTERED (silent cross-module JIT pollution): the permanent CC/CXX=clang leak made any LATER flashinfer JIT build in the same process regenerate build.ninja with the clang host and fail (`alignas(64)` below CUtensorMap's default under CUDA 13.3) - reproducible on HEAD by running any gguf-kernel test before tests/moe/test_nvfp4_backends.py. Lesson: any JIT helper that mutates process env must scope the mutation to its own build call; test-ordering flakes that only appear when a gguf test runs earlier in the process are a red flag for exactly this class.
- No non-JIT path for the gguf extension: root setup.py builds only _pinned_tensor/_cpu_moe/_ple_store; the torch JIT cache is keyed on source+flags hash; the freetoken-kernel-cache wheel is TVM-FFI-only with ZERO gguf coverage.
- The clang++ requirement is documented in docs/install.md (prior campaign); the flashinfer interplay was undocumented until the 1635ecd fix.

- v0 sort change (2026-09-17, uncommitted on vektory79): adding an OPTIONAL tensor arg to the JIT extension requires `std::optional<torch::Tensor>` in the pybind signature - torch's Tensor caster REJECTS None (verified on torch 2.11.0+cu130). Keep the python wrapper default (`order: torch.Tensor | None = None`) so older positional callers (fused_q4_0's 7-arg call) keep working. ggml_moe_a8_vec threads the param through all 19 moe_vec launcher templates - count them on any signature change (mechanical edits are where copy-paste errors hide).
- moe_vec perm design: moe_vec_q takes `const int* order` (pair = order ? order[blockIdx.z] : blockIdx.z) and writes dst[pair*nrows+row], so the kernel itself inverts the expert-sorted order and outputs stay pair-major with NO python scatter; the same order works unchanged for the down call (top_k=1 -> token==pair indexes the interleaved rows). Lifetime nuance the review caught: a local named tensor in the host fn (order_buf) outlives the LAUNCH (enqueue), not the EXECUTION - safety past the async launch comes from torch's same-stream caching allocator (freed block reusable only on its allocation stream). Never generalize the pattern to cross-stream use without record_stream; word the comment accordingly.
