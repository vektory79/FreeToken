# Task 03: Tokenizer - embedded vocab + pre-tokenizer + specials

Type: Code | Agent: Code | Duration: ~2-4 hours

## Goal

Токенизатор из embedded gguf vocab (НЕ sibling tokenizer.json - embedded-vocab
является конвенцией FreeToken для .gguf путей, reader.py:31-33).

## Instructions for the subagent

1. _TOKENIZER_ARCH[arch] = converter-key (transformers ggml.py GGUF_TO_FAST_CONVERTERS).
   Choose by tokenizer.ggml.model + pre field. "gpt2" = plain BPE (structural match
   for gpt2-style vocabs). "qwen2" hardcodes qwen AddedTokens - usually wrong.
2. After convert_gguf_tokenizer: register special tokens from token_type
   (CONTROL=3/USER_DEFINED=4) via fast.add_special_tokens([AddedToken(t, special=True,
   normalized=False)]). Without this, specials byte-split at serve time (the serve
   path re-encodes rendered template TEXT, not ids).
3. EOS resolution: union the gguf-declared end ids (eos + eot + eom) - the model
   may end turns with <|user|> (reference eos = [endoftext, user, observation]).
4. Pre-tokenizer: if the reference pre_tokenizer differs from GPT-2 ByteLevel
   (e.g. glm4 splits digits 1-3 chars), PORT the reference pre_tokenizer rules
   natively (tokenizers.Sequence/Split/Regex/ByteLevel - no runtime file dependency).
   Read the rules from the reference tokenizer.json pre_tokenizer section.
5. Chat template: the embedded tokenizer.ggml.chat_template passes through
   convert_gguf_tokenizer - verify it round-trips.
6. Arch gate: apply changes only for the new arch mapping; existing mappings
   (gemma4) must be untouched.

## Validation (MANDATORY)

- Env-gated round-trip vs the HF/FTW reference tokenizer: prose EN/RU/CJK identity
  (strict), code identity (may need the pre-tokenizer fix), digit survival
  ("in 100 days" -> distinct digit tokens, no [blank]), lossless decode.
- Live battery: re-ask the previously failing prompts through ft serve.
- Fixture test: no-reference CPU test with a synthetic vocab pinning the
  pre-tokenizer structure + specials registration + eos union.

## Traps (from [TRAPS.md](../TRAPS.md))

- T13: Sibling tokenizer.json is NOT read for .gguf paths (reader.py:31-33 states
  embedded-vocab is THE convention) - do not build a --tokenizer-path flag.
- T14: The gguf-py converter drops empty arrays on write - merges must be non-empty
  in fixtures; the BPE initializer requires each merge's result in the vocab.
- T15: The serve path re-encodes rendered template TEXT (apply_chat_template(
  tokenize=False) then encode(add_special_tokens=False)) - if specials are not
  registered as AddedTokens, they byte-split into garbage.
- T16: The reference pre_tokenizer digits rule is \p{N}{1,3} (1-3 chars per
  pre-token), NOT one-per-token - port the EXACT regex, not a guess.

## Acceptance criteria

- [ ] I've created a git commit for this task
- Runs under ORCHESTRATION.md (quality loop + hygiene + commit gate).
