# Root-cause wave: GGUF chat-rendering defect - 2026-09-14 (budget-limited, diagnosis-complete)

Verdict: the tokenizer / chat template / vocab / embeddings / weight FILE and the
serve-side render+encode are all PROVEN CORRECT against the reference checkpoint.
The defect is in the RUNTIME forward path of the gguf-loaded model (post-tokenize:
loader assembly or engine execution), and remains UNFIXED - prescribed next steps below.

## Evidence chain

D1 (metadata dump, phase6/gguf-tokenizer-dump/): vocab 154,880 tokens (154,856 + 24
tail [PAD1548xx] type-5 UNUSED); token_type histogram {1:154820, 3:25, 4:11, 5:24};
ggml scalars: bos 154822 [gMASK], eos 154820 <|endoftext|>, eot 154827 <|user|>,
eom 154829 <|observation|>, model gpt2, pre glm4; embedded chat_template 10,648 chars.

D2/D3: reference (RedHatAI GLM-5.3-Flash-NVFP4) = BPE, vocab 154,820 + 36 added =
154,856 ids; templates differ by EXACTLY 4 benign chars (4 lines: Jinja `m.content.0`
-> `m.content[0]` subscript fix in the gguf copy). D3a index-by-index vocab diff:
0 mismatches over 154,856. D3b special ids align. Reference is a VLM
(model.visual.patch_embed.* present) - the off-task "image/GUI" answers fit a
GLM-V prior being sampled WITHOUT working context attention.

D4 (server-faithful render via models/gguf/tokenizer.py load_gguf_tokenizer):
rendered p00 = "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>Write a short
essay...<|assistant|><think>", 27 ids, round-trip True, cross-decode via ref vocab
correct (EN exact; RU byte-level pieces correct). No loss at render/encode.

D5 (live probe, FREETOKEN_DEBUG_RENDER=1, since reverted): server logged the SAME
render + SAME 27 ids `[154822, 154824, 154826, 25062, 287, 29905, 371, 25, 7487,
154827, 7984, ...]`. Model answered "For zero-target outputs, there are often cases
where the target list is empty..." - unrelated. Input-independence confirmed live:
"Reply with exactly BANANA" -> D&D narration JSON; "What is 2+2?" -> peer-review JSON.

Weight correlation (gguf dequant vs reference bf16, /tmp/ft6_wcorr.py): layer 0
(KDA+dense: q/k/v/o_proj, ssm_beta/f_a/g_a/f_b/g_b, conv1d, dt_bias, o_norm, A_log
derived, norms, dense FFN, hc_attn_fn) ALL 1.0000; layer 3 (DSA: q_a/q_b/kv_a_mqa/
o_proj, fused kv_b k^T|v 1.0000, router, shared experts, indexer wk/wq_b/weights_proj/
k_norm +-ape/gate 0.99998+) ALL 1.0000; embeddings 1.0000 (incl. every role special);
layer ORDER exact (blk.5.attn_q vs ref L5 1.0000, L4/L6 ~0); output.weight Q6_K vs
lm_head 0.99983. The FILE is a faithful conversion.

## Root cause (narrowed)

NOT: tokenizer, template, vocab alignment, embeddings, layer order, output head,
serve-side render/encode. The engine receives the correct 27 ids with correctly
aligned embeddings and still generates prompt-independent text => the forward pass
of the gguf-loaded model does not propagate context. Remaining suspects (ranked):
1. runtime assembly/consumption of the FUSED tensors (in_proj cat order
   q|k|v|b|f_a|g_a vs Glm5NextKDA._in_proj_split [p,p,p,h,d,d]; conv1d q|k|v;
   kv_b [k^T|v] per-head) - static order comments match, but the runtime split/
   reshape path (gguf.py _cast storage-order reshape for 3D tensors, kda.py:64-65)
   is unverified by any test on the real file;
2. offload expert-bank materialization (dim-0 slices of stacked (288,2048,4096)
   IQ3/IQ4, per-projection role order) and router bias e_score_correction_bias
   mapping (ref tensor name differs in the VLM index - unchecked);
3. per-layer config-array slicing (swiglu_clamp_exp/shexp len 46 -> 45) and
   hyper-connection runtime wiring.

## Prescription for the next wave

Offline bisection, no serving needed: load the SAME 27-token input through the
gguf loader path and through the HF/FTW loader path (same engine), compare hidden
states per layer (cos/max-abs) - the first diverging layer-local module names the
broken link. Then unit-test: Glm5NextKDA._in_proj_split against the fused gguf
in_proj on REAL tensors; the conv1d channel order; kv_b per-head [k;v] row order
as the KDA module consumes it; offload bank slice order; e_score_correction_bias
name mapping in gguf.py (ref: e_score_correction_bias not found under
model.language_model.layers.3.mlp. - check actual VLM index name).

No git mutations: the D5 debug patch was reverted (git status python/ clean);
probe server killed, VRAM reclaimed (1324 MiB baseline).
