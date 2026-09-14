---
name: "ft-gguf-kernel-jit-toolchain"
description: gguf CUDA kernel JIT needs clang++ host (kernel/gguf.py); nvcc 13.3 OK; gcc trips ATen List_inl.h
type: project
lastUpdated: 2026-09-13T16:18
lastRecall: 2026-09-13T23:39
---

# gguf CUDA kernel JIT toolchain on vektory79: clang++ host required

Verified 2026-09-13 while bringing up tests/kernels/test_gguf_quant.py (Phase 4 of the glm5next GGUF plan).

- python/freetoken/kernel/gguf.py:25-40 `_host_compiler()` picks the FIRST of ("clang++", "g++-13", "g++-14", "g++-15") found via PATH; the ONLY override is env `FREETOKEN_GGUF_HOST_CXX` (gguf.py:34-36). `_module()` FORCE-overwrites `os.environ["CXX"]/["CC"]` (gguf.py:62-65), so a user-exported CXX is clobbered whenever a fallback compiler exists.
- On this box g++-13/14/15 are all present, so the JIT runs nvcc with `-ccbin g++-13` and trips a non-conformant `typename decltype` in torch's ATen `core/List_inl.h:202` with ANY gcc host (g++-13 and g++-15 both verified failing). `-allow-unsupported-compiler` does NOT help - it only silences nvcc's host-compiler VERSION checks. nvcc + clang++ compiles cleanly (documented in gguf.py:27-33; clang is not installed anywhere on the box).
- nvcc: /usr/local/cuda-13.3/bin/nvcc. torch (2.11.0+cu130) resolves CUDA_HOME=/usr/local/cuda via the /etc/alternatives symlink because CUDA_HOME/CUDA_PATH are unset and nvcc is NOT on PATH -> the CUDA 13.3 nvcc IS what the JIT uses. 13.3 vs torch cu130 = warning only (major-version gate); gcc bounds for CUDA 13.0 are ((6,0,0),(16,0)) so g++-15 passes the version gate (failure is conformance, not version); clang bound CUDA_CLANG_VERSIONS['13.0'] = (7,0)..(21,0).
- The user's llama.cpp checkout /media/ai/src/llama-cpp-glm5next builds GGML_CUDA=ON with this exact nvcc via `cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc` - the CUDA toolchain itself is proven working; llama.cpp never compiles ATen headers, which is why gcc works there and not in the torch extension.
- No non-JIT path for the gguf extension: root setup.py:45-78 builds only _pinned_tensor/_cpu_moe/_ple_store; the torch JIT cache (~/.cache/torch_extensions/py314_cu130) is keyed on source+flags hash; the freetoken-kernel-cache wheel is TVM-FFI-only (its README:1-4) with ZERO gguf coverage.
- Unblock action: `sudo apt install clang` (any clang 7..21; Ubuntu 24.04 distro clang 18 is fine), verify `which clang++` -> /usr/bin/clang++, then `uv run --extra dev pytest tests/kernels/test_gguf_quant.py` - first run JIT-compiles (~32 s gate; the CUDA-gated tests then execute with zero changes).
- The clang++ requirement is documented NOWHERE in the repo (not docs/, not CONTRIBUTING, not freetoken-kernel-cache/README) - only in gguf.py:27-39 and the test docstrings. Candidate one-line docs/CONTRIBUTING follow-up (separate change per one-change-per-PR).
