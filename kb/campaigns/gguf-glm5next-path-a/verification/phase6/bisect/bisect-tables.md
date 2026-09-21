# Bisect tables (from /tmp/ft6_bisect captures; see ROOT-CAUSE.md)

## 1. Prompt-sensitivity of the last-token hidden (cos between prompt A and B)

cos_last = cos(hidden[row -1] at layer L output, prompt A vs prompt B), same boot.
Healthy model: sensitivity grows with depth (cos drops). GGUF: frozen.

L   GGUF-pair  NVFP4-pair       L    GGUF-pair  NVFP4-pair
0    0.982      0.983           23    0.886      0.588
1    0.950      0.976           27    0.923      0.377
2    0.981      0.992           31    0.826      0.292
3    0.864      0.953           33    0.875      0.206
4    0.810      0.746           35    0.948      0.122
5    0.773      0.507           37    0.885      0.105
6    0.238      0.597  <- anom  39    0.865      0.151
7    0.865      0.908           41    0.901      0.264
9    0.773      0.636           43    0.906      0.648
11   0.893      0.811           44    0.963      0.435
13   0.909      0.764
17   0.946      0.593   (full table: rerun analysis.py gguf:0 gguf:1 / nvfp4:0 nvfp4:1)

## 2. Top-10 first-token logits across 3 prompts (same boot)

GGUF   run0/1/2 IDENTICAL: [198, 2, 785, 10056/27, ...] (input-independent)
NVFP4  run0: [785, 334, 154842, 1986, 32, ...]
       run1: [785, 1986, 16360, 154842, ...]   (varies per prompt: healthy)

## 3. MoE stage (exoneration)

moe_rows.py (GGUF, 27-token prefill, rows 0/5/13/26):
  layers with row-varying topk ids: 42/42; routed norms vary per row: 42/42.
recompute_routed.py (layer 3, file bytes via gguf-py reference dequant vs kernel):
  row0 cos 0.999971 norm_ratio 0.9996 ; row1 cos 0.999940 norm_ratio 1.0009
replay through NVFP4 experts on identical captured hidden+topk:
  row0 cos(routed)=0.9959 cos(shared)=1.0000
  row1 cos(routed)=0.9730 cos(shared)=0.9999

## 4. KDA L0 stage diff (kda_stage_diff.py) - THE SMOKING GUN

stage      cos(GGUF,NVFP4)  normG    normN    rows(0,1,13,26)
in         0.999981         28.31    28.31    .99998 .99998 .99998 .99998
proj       0.999985         195.90   195.84
conv_in    0.999984         178.51   178.44
mixed      0.999990         6.38     6.38
b          0.999995         43.40    43.39
f_a        0.999990         36.83    36.83
g_a        0.999993         57.18    57.20
g1 = f_b_proj(f_a)   0.082763  32.40 / 98.66   <- BROKEN
g2 = g_b_proj(g_a)   0.052562  42.91 / 257.12  <- BROKEN
core_out   0.998314         0.125    0.122
onorm_out  0.905788         1.778    1.500
final      0.935661         2.458    2.376

## 5. Kernel isolation (fused_mul_mat_gguf, real f_b packed bytes, K=128)

x contiguous,  T=27 (MMQ): cos(kernel, torch-ref) = 0.99998
x contiguous,  T=4  (MMVQ): cos(kernel, torch-ref) = 0.99998
x non-contig (stride (24896,1) slice), T=27: cos = 0.03316, norm_ratio 0.983  <- BUG
x same but .contiguous():                    cos = 0.99998

## 6. Offline fp64 KDA replication (kda_sim.py, identical input)

cos(simGGUFweights, simREFweights) block = 0.99997  -> weights equal in effect;
the engine divergence is NOT weight noise -> runtime path bug (section 4/5).
