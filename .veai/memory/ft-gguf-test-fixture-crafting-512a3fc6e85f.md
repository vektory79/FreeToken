---
name: "ft-gguf-test-fixture-crafting"
description: "gguf test methodology: _FP16_SCALE_FIELDS, analytic fixtures, seeded tolerance bounds (unseeded = latent flakes)"
type: project
lastUpdated: 2026-09-19T02:54
lastRecall: 2026-09-20T16:44
---

# Crafting synthetic gguf quant tensors in FreeToken tests

Gotchas collected while building the glm5next GGUF test fixtures (tests/models/test_glm5_next_gguf.py, tests/models/test_gguf_expert_banks.py; 2026-09-13). Apply whenever a test needs hand-packed quantized gguf tensors.

- Scale-field byte offsets per format (module table _FP16_SCALE_FIELDS in tests/models/test_gguf_quant.py): Q8_0 [0:2], Q6_K [208:210], Q3_K [108:110], Q4_K [0:4] (d and dmin are adjacent half2 fields), IQ3_XXS [0:2], IQ4_XS [0:2]. Verified against vendored gguf-py quants.py dequantize_blocks hsplit order.
- Draw scales from [0.25, 2] so the kernel's fp16 chain (ggml-common.h:930 typedef half dfloat) stays finite BY CONSTRUCTION. Blind random bytes + resampling never converges for 64-row moe blocks (accept probability ~0 for multi-block rows) and spams numpy "overflow encountered in cast" RuntimeWarnings.
- The vendored block_q6_K layout is ql, qh, scales, d-LAST (d at bytes 208:210; ggml-common.h:104-110). A test comment claiming "llama.cpp d-first" was wrong - do not repeat it.
- Handy exact constants for analytic parity: fp16(0.25) = 0x2800; IQ3_XXS qs-code 71 -> 0.75; IQ4_XS scale-byte 33 / nibble 8 -> 0.25; Q6_K with d chosen so a full block decodes to -16.
- gguf-py writer pitfalls: it DROPS empty arrays on write (merges must be non-empty; empty-array handling is then only testable at the helper level); Q6_K tensors require ne0 % 256 == 0 (fixture hidden dim had to go 64 -> 256); the BPE tokenizer initializer requires every merge's result to be in the vocab; byte-level fixture alphabets come from transformers.convert_slow_tokenizer.bytes_to_unicode.
- LESSON (cost two Error-class review findings): square fixtures (hidden == expert_ffn) mask every real-geometry asymmetry. The glm5next file's down bank has 4096 rows vs gate/up 2048 and three width signatures across layers - a square fixture let a wrong row count and a cross-layer shape assertion survive 14 green tests. MoE bank tests need non-square dims, multiple width signatures, and a down-first tensor insertion-order variant.

Why: these fixtures gate the CUDA kernel parity battery and the offload bank path; a fixture that hides geometry produces green tests over broken code.
How to apply: any new test crafting quantized gguf tensors or expert banks - reuse the existing helpers (_write_tokenizer_gguf / iter-fixture patterns) and keep the non-square + multi-signature rule.

## Analytic-uniform fixtures beat parity (2026-09-15, hybrid campaign Task 01)
Analytic uniform fixtures (exact expected values computed independently) catch fixture/packer-semantics bugs that parity-vs-reference CANNOT: same wrong bytes on both sides cancel out. In tests/moe/test_cpu_moe_gguf_iq.py the analytic test caught the IQ3_XXS scale nibble broadcast into the summed sign lanes (4x nibble) which the parity test missed by construction. Always pair parity-vs-gguf-py with an analytic uniform test.

## Tolerance pinning near the noise floor (2026-09-18, v3a Review-A)
Unseeded tolerance bounds sized at the noise floor are LATENT FLAKES: v2's unseeded grouped-vs-moe_vec 1e-3 for iq passed at commit time but fails 1/20 seeds for IQ4_XS (margin +2.0e-4, worst |a-b| ~1.2e-3). Method: seed the draw (torch.manual_seed), noise-scan ~20 seeds, size the bound >= ~2-5x the worst band (v3a landed 2e-2/1e-2 for iq; a 4e-3/2e-3 option keeps a 5x tighter tripwire). Systematic bug classes (ds-fill row garbage) shift whole pairs O(1) - 100x above the band - so loosening masks nothing real. Related gotcha: torch.empty bf16 inputs in guard-only tests yield ~1e38 garbage that overflows the q8_1 fp16 scale -> inf; use torch.zeros when the test only needs finite math.
