"""Build a HF fast tokenizer from a GGUF file's embedded tokenizer metadata.

transformers' ``AutoTokenizer.from_pretrained(gguf_file=...)`` first builds the HF
config, which the gemma4 strict dataclass rejects (per-layer ``num_key_value_heads``
array). So we call the GGUF->fast tokenizer converter directly on the
``tokenizer.ggml.*`` metadata, bypassing config entirely.
"""

from __future__ import annotations

from typing import Any

from tokenizers import Regex
from tokenizers.pre_tokenizers import ByteLevel, Sequence, Split
from transformers import AddedToken

from .reader import gguf_architecture, load_gguf_metadata

# GGUF architecture -> transformers GGUF tokenizer-converter key.
# glm5next: the gguf declares tokenizer.ggml.model=gpt2; GGUFGPTConverter is the
# plain-BPE structural match - qwen2's converter hardcodes qwen AddedTokens. The
# converter attaches GPT-2 regex pre-tokenization, so load_gguf_tokenizer swaps in
# the glm4 scheme (tokenizer.ggml.pre=glm4) for glm5next below; the env-gated
# round-trip test (FREETOKEN_GLM5NEXT_TOKENIZER_REF) judges the fidelity.
_TOKENIZER_ARCH = {"gemma4": "gemma4_text", "glm5next": "gpt2"}

# glm4 pre-tokenization, ported verbatim from the reference pre_tokenizer
# (RedHatAI GLM-5.3-Flash-NVFP4 tokenizer.json; the gguf declares pre=glm4).
_GLM4_SPLIT_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


def _glm4_pre_tokenizer():
    """The reference glm4 Sequence pre-tokenizer rebuilt natively (no ref file)."""
    return Sequence(
        [
            Split(Regex(_GLM4_SPLIT_REGEX), behavior="isolated", invert=False),
            ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False),
        ]
    )


def load_gguf_tokenizer(model_path: str):
    from transformers import PreTrainedTokenizerFast
    from transformers.integrations.ggml import convert_gguf_tokenizer

    meta = load_gguf_metadata(model_path)
    arch = gguf_architecture(model_path)
    conv_arch = _TOKENIZER_ARCH.get(arch, arch)
    tok_dict: dict[str, Any] = {
        k[len("tokenizer.ggml.") :]: v
        for k, v in meta.items()
        if k.startswith("tokenizer.ggml.")
    }
    fast, _extra = convert_gguf_tokenizer(conv_arch, tok_dict)

    # glm5next only: the converter's GPT-2 regex glues the space onto digit runs
    # ("Ġ100"); swap in the reference glm4 scheme so numerals survive encoding.
    if arch == "glm5next":
        fast.pre_tokenizer = _glm4_pre_tokenizer()

    tokens = tok_dict["tokens"]

    # Register the gguf special tokens (token_type CONTROL/USER_DEFINED) as
    # AddedTokens on the converted backend: serving re-encodes rendered chat text
    # (apply_chat_template(tokenize=False) -> encode), and without registration the
    # special strings split byte-wise instead of mapping to their vocab ids. The
    # ids do not change (the strings already sit in the vocab). Arch-agnostic:
    # gemma4's <turn|> gains atomicity too. See test_glm5next_gguf_tokenizer_matches_reference.
    token_types = meta.get("tokenizer.ggml.token_type") or []
    special_names = [
        tok
        for tok, tt in zip(tokens, token_types)
        if int(tt) in (3, 4)  # GGMLTokenTyPE CONTROL / USER_DEFINED
    ]
    if special_names:
        fast.add_special_tokens(
            [AddedToken(t, special=True, normalized=False) for t in special_names]
        )

    def tok_for(id_key: str, default: str) -> str:
        tid = meta.get(f"tokenizer.ggml.{id_key}")
        return tokens[int(tid)] if tid is not None and int(tid) < len(tokens) else default

    # gemma4 chat turns end with <turn|>; prefer it as eos so chat generation halts
    # (the formal <eos> is also a stop id, see gguf_eos_token_ids).
    turn_end = "<turn|>" if "<turn|>" in tokens else None
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=fast,
        bos_token=tok_for("bos_token_id", "<bos>"),
        eos_token=turn_end or tok_for("eos_token_id", "<eos>"),
        unk_token=tok_for("unknown_token_id", "<unk>"),
        pad_token=tok_for("padding_token_id", "<pad>"),
    )
    chat_template = meta.get("tokenizer.chat_template")
    if chat_template:
        tokenizer.chat_template = chat_template
    return tokenizer


def gguf_eos_token_ids(model_path: str, tokenizer) -> set[int]:
    """Stop ids for GGUF generation: the formal <eos> plus the chat turn end <turn|>."""
    meta = load_gguf_metadata(model_path)
    tokens = meta["tokenizer.ggml.tokens"]
    ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    eid = meta.get("tokenizer.ggml.eos_token_id")
    if eid is not None:
        ids.add(int(eid))
    # glm5next ends turns with <eot>/<eom>, not just <eos>: the gguf carries
    # eot_token_id/eom_token_id next to eos (reference stop set {154820, 154827,
    # 154829}); eos-only would run past the turn end.
    if gguf_architecture(model_path) == "glm5next":
        for key in ("eom_token_id", "eot_token_id"):
            tid = meta.get(f"tokenizer.ggml.{key}")
            if tid is not None:
                ids.add(int(tid))
    # Look the stop tokens up in the vocab directly (convert_tokens_to_ids would map an
    # absent name to <unk>, wrongly adding it as a stop id).
    for name in ("<eos>", "<turn|>"):
        try:
            ids.add(tokens.index(name))
        except ValueError:
            pass
    return ids


__all__ = ["load_gguf_tokenizer", "gguf_eos_token_ids"]
