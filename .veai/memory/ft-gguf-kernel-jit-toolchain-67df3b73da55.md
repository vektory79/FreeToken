---
name: "ft-gguf-kernel-jit-toolchain"
description: gguf CUDA kernel JIT needs clang++ host (kernel/gguf.py); nvcc 13.3; CC/CXX scoped to build since 1635ecd
type: project
lastUpdated: 2026-09-15T01:48
lastRecall: 2026-09-15T19:57
---

# gguf CUDA kernel JIT toolchain on vektory79: clang++ host required

Verified 2026-09-13 (Phase 4 of the glm5next GGUF plan); UPDATED 2026-09-15 (hybrid campaign, commit 1635ecd).

- python/freetoken/kernel/gguf.py `_host_compiler()` picks the FIRST of ("clang++", "g++-13", "g++-14", "g++-15") found via PATH; the ONLY override is env `FREETOKEN_GGUF_HOST_CXX`. On this box g++-13/14/15 are all present, so without clang the JIT trips a non-conformant `typename decltype` in torch's ATen `core/List_inl.h` with ANY gcc host. `-allow-unsupported-compiler` does NOT help (only silences nvcc version checks). nvcc + clang++ compiles cleanly.
- nvcc: /usr/local/cuda-13.3/bin/nvcc; torch (2.11.0+cu130) resolves CUDA_HOME=/usr/local/cuda via /etc/alternatives because CUDA_HOME/CUDA_PATH unset and nvcc not on PATH. 13.3 vs cu130 = warning only; gcc bounds for CUDA 13.0 are ((6,0,0),(16,0)) so g++-15 passes the version gate (failure is conformance, not version).
- FIX 1635ecd (2026-09-15): `_module()` no longer permanently force-overwrites process-global `os.environ["CC"]/["CXX"]` - the clang host override is now set around the `torch.utils.cpp_extension.load(...)` call and restored in a `finally` (exact prior state incl. removal). Regression tests in tests/kernels/test_gguf_quant.py: `test_module_load_scopes_cc_cxx_to_the_build` (failing-before verified: on HEAD it leaked {'CXX': '/usr/bin/clang++', 'CC': '/usr/bin/clang'}) and `test_module_load_without_a_host_compiler_sets_no_cc_cxx`. User-exported CXX is no longer clobbered.
- WHY IT MATTERED (silent cross-module JIT pollution): the permanent CC/CXX=clang leak made any LATER flashinfer JIT build in the same process regenerate build.ninja with the clang host and fail (`alignas(64)` below CUtensorMap's default under CUDA 13.3) - reproducible on HEAD by running any gguf-kernel test before tests/moe/test_nvfp4_backends.py. Lesson: any JIT helper that mutates process env must scope the mutation to its own build call; test-ordering flakes that only appear when a gguf test runs earlier in the process are a red flag for exactly this class.
- No non-JIT path for the gguf extension: root setup.py builds only _pinned_tensor/_cpu_moe/_ple_store; the torch JIT cache is keyed on source+flags hash; the freetoken-kernel-cache wheel is TVM-FFI-only with ZERO gguf coverage.
- The clang++ requirement is documented in docs/install.md (prior campaign); the flashinfer interplay was undocumented until the 1635ecd fix.
