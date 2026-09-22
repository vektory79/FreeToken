# 5-prompt live battery, GGUF model WITH the fused_mul_mat_gguf contiguity fix
# (greedy temp0/topk1, 160 tokens, run 2026-09-14 ~15:00, tag gguf_fixed)
# Compare with the pre-fix battery: 0/5 on-topic (image-math JSON, empty,
# <user>-tag confusion, nezha JSON, pyannote diarization).

p0 "Write a short essay on why public libraries still matter in the internet age."
   -> reasoning: "The user wants a short essay on why public libraries still matter
   in the internet age... Key arguments: 1. Digital divide / access equity: Not
   everyone has i..."                     ON-TOPIC

p1 "What is 2+2? Answer briefly."
   -> reasoning: "The user is asking 'What is +2?' - this seems like an incomplete
   question. They wrote 'What is' followed by '+2'. This could be: 1. A formatting
   issue where they meant to ask 'What is 2+2?'..."   ON-TOPIC (reads the actual
   question; note the known pre=glm4 arithmetic-tokenization nuance, separate defect)

p2 "Reply with exactly BANANA."
   -> reasoning: "The user is asking me to reply with exactly 'BANANA'..."
      content: "BANANA"                    ON-TOPIC, exact compliance

p3 "Name three primary colors."
   -> reasoning: "The user is asking a simple factual question: name three primary
   colors. Primary colors depend on the color system: 1. Traditional/artistic
   (RYB - Red, Yellow, Blue)..."          ON-TOPIC

p4 "Write a Python function that returns the first n Fibonacci numbers using
   iteration, with a docstring."
   -> reasoning: "The user wants a Python function that returns the first n
   Fibonacci numbers using iteration, with a docstring. **Understanding the
   requirements:** 1. A Python function 2. Returns the first n Fibonacci numbers"  ON-TOPIC

VERDICT: 5/5 on-topic vs 0/5 before the fix and 3/24 in the original Phase 6
battery. The NVFP4 baseline answers the same 5 prompts on-topic (24/24 battery).
