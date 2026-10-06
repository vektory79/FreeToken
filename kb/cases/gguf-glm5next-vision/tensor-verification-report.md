# mmproj tensor verification: GGUF reader vs NVFP4 reference

- mmproj: `/media/ai/models/unsloth/GLM-5.3-Flash-GGUF/mmproj-BF16.gguf`
- reference: `/media/ai/models/RedHatAI/GLM-5.3-Flash-NVFP4-FTW` (visual.* tensors, bf16)
- tensors: 347 (mmproj-side names 347, reference-side 347)
- the table is param-level: the mmproj file holds 348 tensors, its two
  v.patch_embd.weight slices stack into the single visual.patch_embed.proj.weight
- exact (max_abs_diff == 0): 347
- <= 0.01: 0
- ABOVE threshold: 0
- unmatched (no counterpart): 0
- shape mismatches: 0

max_abs_diff percentiles (fp32 compare of both bf16 sides):

| p50 | p90 | p99 | max |
|---|---|---|---|
| 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |

Threshold note: the reference tower is bf16; F32-sourced mmproj tensors cast to
bf16 exactly like the reference conversion did, so anything above the threshold
is a mapping suspect (axis order / transpose / segment order), not cast noise.

| tensor | source dtype | shape | max_abs_diff | status |
|---|---|---|---|---|
| visual.blocks.0.attn.k_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.0.attn.proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.0.attn.proj.weight | BF16 | (1024, 1024) | 0.000e+00 | exact |
| visual.blocks.0.attn.q_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.0.attn.qkv.bias | F32 | (3072,) | 0.000e+00 | exact |
| visual.blocks.0.attn.qkv.weight | BF16 | (3072, 1024) | 0.000e+00 | exact |
| visual.blocks.0.mlp.down_proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.0.mlp.down_proj.weight | BF16 | (1024, 4096) | 0.000e+00 | exact |
| visual.blocks.0.mlp.gate_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.0.mlp.gate_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.0.mlp.up_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.0.mlp.up_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.0.norm1.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.0.norm2.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.1.attn.k_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.1.attn.proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.1.attn.proj.weight | BF16 | (1024, 1024) | 0.000e+00 | exact |
| visual.blocks.1.attn.q_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.1.attn.qkv.bias | F32 | (3072,) | 0.000e+00 | exact |
| visual.blocks.1.attn.qkv.weight | BF16 | (3072, 1024) | 0.000e+00 | exact |
| visual.blocks.1.mlp.down_proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.1.mlp.down_proj.weight | BF16 | (1024, 4096) | 0.000e+00 | exact |
| visual.blocks.1.mlp.gate_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.1.mlp.gate_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.1.mlp.up_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.1.mlp.up_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.1.norm1.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.1.norm2.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.10.attn.k_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.10.attn.proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.10.attn.proj.weight | BF16 | (1024, 1024) | 0.000e+00 | exact |
| visual.blocks.10.attn.q_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.10.attn.qkv.bias | F32 | (3072,) | 0.000e+00 | exact |
| visual.blocks.10.attn.qkv.weight | BF16 | (3072, 1024) | 0.000e+00 | exact |
| visual.blocks.10.mlp.down_proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.10.mlp.down_proj.weight | BF16 | (1024, 4096) | 0.000e+00 | exact |
| visual.blocks.10.mlp.gate_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.10.mlp.gate_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.10.mlp.up_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.10.mlp.up_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.10.norm1.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.10.norm2.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.11.attn.k_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.11.attn.proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.11.attn.proj.weight | BF16 | (1024, 1024) | 0.000e+00 | exact |
| visual.blocks.11.attn.q_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.11.attn.qkv.bias | F32 | (3072,) | 0.000e+00 | exact |
| visual.blocks.11.attn.qkv.weight | BF16 | (3072, 1024) | 0.000e+00 | exact |
| visual.blocks.11.mlp.down_proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.11.mlp.down_proj.weight | BF16 | (1024, 4096) | 0.000e+00 | exact |
| visual.blocks.11.mlp.gate_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.11.mlp.gate_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.11.mlp.up_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.11.mlp.up_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.11.norm1.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.11.norm2.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.12.attn.k_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.12.attn.proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.12.attn.proj.weight | BF16 | (1024, 1024) | 0.000e+00 | exact |
| visual.blocks.12.attn.q_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.12.attn.qkv.bias | F32 | (3072,) | 0.000e+00 | exact |
| visual.blocks.12.attn.qkv.weight | BF16 | (3072, 1024) | 0.000e+00 | exact |
| visual.blocks.12.mlp.down_proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.12.mlp.down_proj.weight | BF16 | (1024, 4096) | 0.000e+00 | exact |
| visual.blocks.12.mlp.gate_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.12.mlp.gate_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.12.mlp.up_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.12.mlp.up_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.12.norm1.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.12.norm2.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.13.attn.k_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.13.attn.proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.13.attn.proj.weight | BF16 | (1024, 1024) | 0.000e+00 | exact |
| visual.blocks.13.attn.q_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.13.attn.qkv.bias | F32 | (3072,) | 0.000e+00 | exact |
| visual.blocks.13.attn.qkv.weight | BF16 | (3072, 1024) | 0.000e+00 | exact |
| visual.blocks.13.mlp.down_proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.13.mlp.down_proj.weight | BF16 | (1024, 4096) | 0.000e+00 | exact |
| visual.blocks.13.mlp.gate_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.13.mlp.gate_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.13.mlp.up_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.13.mlp.up_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.13.norm1.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.13.norm2.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.14.attn.k_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.14.attn.proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.14.attn.proj.weight | BF16 | (1024, 1024) | 0.000e+00 | exact |
| visual.blocks.14.attn.q_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.14.attn.qkv.bias | F32 | (3072,) | 0.000e+00 | exact |
| visual.blocks.14.attn.qkv.weight | BF16 | (3072, 1024) | 0.000e+00 | exact |
| visual.blocks.14.mlp.down_proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.14.mlp.down_proj.weight | BF16 | (1024, 4096) | 0.000e+00 | exact |
| visual.blocks.14.mlp.gate_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.14.mlp.gate_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.14.mlp.up_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.14.mlp.up_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.14.norm1.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.14.norm2.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.15.attn.k_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.15.attn.proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.15.attn.proj.weight | BF16 | (1024, 1024) | 0.000e+00 | exact |
| visual.blocks.15.attn.q_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.15.attn.qkv.bias | F32 | (3072,) | 0.000e+00 | exact |
| visual.blocks.15.attn.qkv.weight | BF16 | (3072, 1024) | 0.000e+00 | exact |
| visual.blocks.15.mlp.down_proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.15.mlp.down_proj.weight | BF16 | (1024, 4096) | 0.000e+00 | exact |
| visual.blocks.15.mlp.gate_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.15.mlp.gate_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.15.mlp.up_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.15.mlp.up_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.15.norm1.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.15.norm2.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.16.attn.k_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.16.attn.proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.16.attn.proj.weight | BF16 | (1024, 1024) | 0.000e+00 | exact |
| visual.blocks.16.attn.q_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.16.attn.qkv.bias | F32 | (3072,) | 0.000e+00 | exact |
| visual.blocks.16.attn.qkv.weight | BF16 | (3072, 1024) | 0.000e+00 | exact |
| visual.blocks.16.mlp.down_proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.16.mlp.down_proj.weight | BF16 | (1024, 4096) | 0.000e+00 | exact |
| visual.blocks.16.mlp.gate_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.16.mlp.gate_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.16.mlp.up_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.16.mlp.up_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.16.norm1.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.16.norm2.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.17.attn.k_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.17.attn.proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.17.attn.proj.weight | BF16 | (1024, 1024) | 0.000e+00 | exact |
| visual.blocks.17.attn.q_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.17.attn.qkv.bias | F32 | (3072,) | 0.000e+00 | exact |
| visual.blocks.17.attn.qkv.weight | BF16 | (3072, 1024) | 0.000e+00 | exact |
| visual.blocks.17.mlp.down_proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.17.mlp.down_proj.weight | BF16 | (1024, 4096) | 0.000e+00 | exact |
| visual.blocks.17.mlp.gate_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.17.mlp.gate_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.17.mlp.up_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.17.mlp.up_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.17.norm1.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.17.norm2.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.18.attn.k_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.18.attn.proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.18.attn.proj.weight | BF16 | (1024, 1024) | 0.000e+00 | exact |
| visual.blocks.18.attn.q_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.18.attn.qkv.bias | F32 | (3072,) | 0.000e+00 | exact |
| visual.blocks.18.attn.qkv.weight | BF16 | (3072, 1024) | 0.000e+00 | exact |
| visual.blocks.18.mlp.down_proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.18.mlp.down_proj.weight | BF16 | (1024, 4096) | 0.000e+00 | exact |
| visual.blocks.18.mlp.gate_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.18.mlp.gate_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.18.mlp.up_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.18.mlp.up_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.18.norm1.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.18.norm2.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.19.attn.k_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.19.attn.proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.19.attn.proj.weight | BF16 | (1024, 1024) | 0.000e+00 | exact |
| visual.blocks.19.attn.q_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.19.attn.qkv.bias | F32 | (3072,) | 0.000e+00 | exact |
| visual.blocks.19.attn.qkv.weight | BF16 | (3072, 1024) | 0.000e+00 | exact |
| visual.blocks.19.mlp.down_proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.19.mlp.down_proj.weight | BF16 | (1024, 4096) | 0.000e+00 | exact |
| visual.blocks.19.mlp.gate_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.19.mlp.gate_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.19.mlp.up_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.19.mlp.up_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.19.norm1.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.19.norm2.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.2.attn.k_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.2.attn.proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.2.attn.proj.weight | BF16 | (1024, 1024) | 0.000e+00 | exact |
| visual.blocks.2.attn.q_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.2.attn.qkv.bias | F32 | (3072,) | 0.000e+00 | exact |
| visual.blocks.2.attn.qkv.weight | BF16 | (3072, 1024) | 0.000e+00 | exact |
| visual.blocks.2.mlp.down_proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.2.mlp.down_proj.weight | BF16 | (1024, 4096) | 0.000e+00 | exact |
| visual.blocks.2.mlp.gate_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.2.mlp.gate_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.2.mlp.up_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.2.mlp.up_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.2.norm1.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.2.norm2.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.20.attn.k_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.20.attn.proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.20.attn.proj.weight | BF16 | (1024, 1024) | 0.000e+00 | exact |
| visual.blocks.20.attn.q_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.20.attn.qkv.bias | F32 | (3072,) | 0.000e+00 | exact |
| visual.blocks.20.attn.qkv.weight | BF16 | (3072, 1024) | 0.000e+00 | exact |
| visual.blocks.20.mlp.down_proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.20.mlp.down_proj.weight | BF16 | (1024, 4096) | 0.000e+00 | exact |
| visual.blocks.20.mlp.gate_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.20.mlp.gate_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.20.mlp.up_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.20.mlp.up_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.20.norm1.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.20.norm2.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.21.attn.k_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.21.attn.proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.21.attn.proj.weight | BF16 | (1024, 1024) | 0.000e+00 | exact |
| visual.blocks.21.attn.q_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.21.attn.qkv.bias | F32 | (3072,) | 0.000e+00 | exact |
| visual.blocks.21.attn.qkv.weight | BF16 | (3072, 1024) | 0.000e+00 | exact |
| visual.blocks.21.mlp.down_proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.21.mlp.down_proj.weight | BF16 | (1024, 4096) | 0.000e+00 | exact |
| visual.blocks.21.mlp.gate_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.21.mlp.gate_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.21.mlp.up_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.21.mlp.up_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.21.norm1.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.21.norm2.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.22.attn.k_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.22.attn.proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.22.attn.proj.weight | BF16 | (1024, 1024) | 0.000e+00 | exact |
| visual.blocks.22.attn.q_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.22.attn.qkv.bias | F32 | (3072,) | 0.000e+00 | exact |
| visual.blocks.22.attn.qkv.weight | BF16 | (3072, 1024) | 0.000e+00 | exact |
| visual.blocks.22.mlp.down_proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.22.mlp.down_proj.weight | BF16 | (1024, 4096) | 0.000e+00 | exact |
| visual.blocks.22.mlp.gate_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.22.mlp.gate_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.22.mlp.up_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.22.mlp.up_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.22.norm1.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.22.norm2.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.23.attn.k_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.23.attn.proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.23.attn.proj.weight | BF16 | (1024, 1024) | 0.000e+00 | exact |
| visual.blocks.23.attn.q_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.23.attn.qkv.bias | F32 | (3072,) | 0.000e+00 | exact |
| visual.blocks.23.attn.qkv.weight | BF16 | (3072, 1024) | 0.000e+00 | exact |
| visual.blocks.23.mlp.down_proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.23.mlp.down_proj.weight | BF16 | (1024, 4096) | 0.000e+00 | exact |
| visual.blocks.23.mlp.gate_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.23.mlp.gate_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.23.mlp.up_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.23.mlp.up_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.23.norm1.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.23.norm2.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.3.attn.k_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.3.attn.proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.3.attn.proj.weight | BF16 | (1024, 1024) | 0.000e+00 | exact |
| visual.blocks.3.attn.q_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.3.attn.qkv.bias | F32 | (3072,) | 0.000e+00 | exact |
| visual.blocks.3.attn.qkv.weight | BF16 | (3072, 1024) | 0.000e+00 | exact |
| visual.blocks.3.mlp.down_proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.3.mlp.down_proj.weight | BF16 | (1024, 4096) | 0.000e+00 | exact |
| visual.blocks.3.mlp.gate_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.3.mlp.gate_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.3.mlp.up_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.3.mlp.up_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.3.norm1.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.3.norm2.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.4.attn.k_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.4.attn.proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.4.attn.proj.weight | BF16 | (1024, 1024) | 0.000e+00 | exact |
| visual.blocks.4.attn.q_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.4.attn.qkv.bias | F32 | (3072,) | 0.000e+00 | exact |
| visual.blocks.4.attn.qkv.weight | BF16 | (3072, 1024) | 0.000e+00 | exact |
| visual.blocks.4.mlp.down_proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.4.mlp.down_proj.weight | BF16 | (1024, 4096) | 0.000e+00 | exact |
| visual.blocks.4.mlp.gate_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.4.mlp.gate_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.4.mlp.up_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.4.mlp.up_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.4.norm1.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.4.norm2.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.5.attn.k_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.5.attn.proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.5.attn.proj.weight | BF16 | (1024, 1024) | 0.000e+00 | exact |
| visual.blocks.5.attn.q_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.5.attn.qkv.bias | F32 | (3072,) | 0.000e+00 | exact |
| visual.blocks.5.attn.qkv.weight | BF16 | (3072, 1024) | 0.000e+00 | exact |
| visual.blocks.5.mlp.down_proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.5.mlp.down_proj.weight | BF16 | (1024, 4096) | 0.000e+00 | exact |
| visual.blocks.5.mlp.gate_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.5.mlp.gate_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.5.mlp.up_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.5.mlp.up_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.5.norm1.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.5.norm2.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.6.attn.k_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.6.attn.proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.6.attn.proj.weight | BF16 | (1024, 1024) | 0.000e+00 | exact |
| visual.blocks.6.attn.q_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.6.attn.qkv.bias | F32 | (3072,) | 0.000e+00 | exact |
| visual.blocks.6.attn.qkv.weight | BF16 | (3072, 1024) | 0.000e+00 | exact |
| visual.blocks.6.mlp.down_proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.6.mlp.down_proj.weight | BF16 | (1024, 4096) | 0.000e+00 | exact |
| visual.blocks.6.mlp.gate_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.6.mlp.gate_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.6.mlp.up_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.6.mlp.up_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.6.norm1.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.6.norm2.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.7.attn.k_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.7.attn.proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.7.attn.proj.weight | BF16 | (1024, 1024) | 0.000e+00 | exact |
| visual.blocks.7.attn.q_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.7.attn.qkv.bias | F32 | (3072,) | 0.000e+00 | exact |
| visual.blocks.7.attn.qkv.weight | BF16 | (3072, 1024) | 0.000e+00 | exact |
| visual.blocks.7.mlp.down_proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.7.mlp.down_proj.weight | BF16 | (1024, 4096) | 0.000e+00 | exact |
| visual.blocks.7.mlp.gate_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.7.mlp.gate_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.7.mlp.up_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.7.mlp.up_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.7.norm1.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.7.norm2.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.8.attn.k_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.8.attn.proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.8.attn.proj.weight | BF16 | (1024, 1024) | 0.000e+00 | exact |
| visual.blocks.8.attn.q_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.8.attn.qkv.bias | F32 | (3072,) | 0.000e+00 | exact |
| visual.blocks.8.attn.qkv.weight | BF16 | (3072, 1024) | 0.000e+00 | exact |
| visual.blocks.8.mlp.down_proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.8.mlp.down_proj.weight | BF16 | (1024, 4096) | 0.000e+00 | exact |
| visual.blocks.8.mlp.gate_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.8.mlp.gate_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.8.mlp.up_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.8.mlp.up_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.8.norm1.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.8.norm2.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.9.attn.k_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.9.attn.proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.9.attn.proj.weight | BF16 | (1024, 1024) | 0.000e+00 | exact |
| visual.blocks.9.attn.q_norm.weight | F32 | (64,) | 0.000e+00 | exact |
| visual.blocks.9.attn.qkv.bias | F32 | (3072,) | 0.000e+00 | exact |
| visual.blocks.9.attn.qkv.weight | BF16 | (3072, 1024) | 0.000e+00 | exact |
| visual.blocks.9.mlp.down_proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.9.mlp.down_proj.weight | BF16 | (1024, 4096) | 0.000e+00 | exact |
| visual.blocks.9.mlp.gate_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.9.mlp.gate_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.9.mlp.up_proj.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.blocks.9.mlp.up_proj.weight | BF16 | (4096, 1024) | 0.000e+00 | exact |
| visual.blocks.9.norm1.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.blocks.9.norm2.weight | F32 | (1024,) | 0.000e+00 | exact |
| visual.downsample.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.downsample.weight | F32 | (4096, 1024, 2, 2) | 0.000e+00 | exact |
| visual.merger.down_proj.weight | BF16 | (4096, 10240) | 0.000e+00 | exact |
| visual.merger.gate_proj.weight | BF16 | (10240, 4096) | 0.000e+00 | exact |
| visual.merger.post_projection_norm.bias | F32 | (4096,) | 0.000e+00 | exact |
| visual.merger.post_projection_norm.weight | F32 | (4096,) | 0.000e+00 | exact |
| visual.merger.proj.weight | BF16 | (4096, 4096) | 0.000e+00 | exact |
| visual.merger.up_proj.weight | BF16 | (10240, 4096) | 0.000e+00 | exact |
| visual.patch_embed.proj.bias | F32 | (1024,) | 0.000e+00 | exact |
| visual.patch_embed.proj.weight | F32 | (1024, 3, 2, 14, 14) | 0.000e+00 | exact |
| visual.post_layernorm.weight | F32 | (1024,) | 0.000e+00 | exact |
