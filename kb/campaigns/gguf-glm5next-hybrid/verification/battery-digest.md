# Task 05 battery digest - hybrid on the REAL file (2026-09-15)

24-prompt battery (template: .tasks/gguf-glm5next-path-a/verification/phase6/
battery-post-fix/), served via /v1/chat/completions with the "model" field from
/v1/models (`GLM-5.3-Flash-UD-Q3_K_XL.gguf`), temp0/topk1, max_tokens 160/160/192/128
by kind. Raw responses: /tmp/ft_t05_hybrid_main_battery/p00-p23.json.

## Result: 24/24 on-topic (gate 20+/24) - PASS

- p00-p07 [en] prose: all on-topic (libraries, bicycle physics, coastal rain, remote
  work, printing press, sky blue/sunsets red, lighthouse story, vaccines).
- p08-p15 [ru]: all on-topic and answered in Russian (paper books, rain for a child,
  winter village, compass, cat and snow, cases, borscht, Pushkin).
- p16-p19 [code]: correct task understanding + plausible code (iterative Fibonacci with
  docstring; is_palindrome ignoring case/non-alnum; tr | sort | uniq -c bash one-liner;
  Stack class with TypeVar + docstring).
- p20-p23 [reason]: all CORRECT - 80 km/h; Friday (100 mod 7 = 2); 6 apples (finish
  reason: stop); 2^10 = 1024 > 1000. The digit-split pre-tokenizer fix holds under the
  hybrid path (these were the 4 off-topic failures of the pre-fix offload campaign).

## Divergence vs the offload-only baseline

The hybrid responses track the offload-only battery texts closely (same reasoning
structure, same conclusions); wording drift is the same quant-level kind the campaign
accepted (chars 49-339 divergence vs NVFP4 on the identical prompts). No new failure
mode, no garbage/regression text anywhere in the 24 responses.

Wall clock per prompt 42-73 s (decode-bound at ~3 tok/s) - quality is independent of
the decode-speed failure recorded in decode-numbers.md / ab-table.md.
