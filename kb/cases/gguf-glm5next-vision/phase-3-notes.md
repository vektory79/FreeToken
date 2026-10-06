# Phase 3 notes: image processor from mmproj metadata

Companion note to [phases-1-2.md](phases-1-2.md); covers the deliberate
deviation from the [TASK.md](TASK.md) phase-3 brief plus the two facts the
brief got wrong about the files on disk.

## Deviation: missing clip.* fields fail fast instead of pinning to NVFP4 values

The brief suggested filling any missing clip.* field with the NVFP4 reference
value pinned in code. Implemented the opposite: every clip.* key the image
processor consumes (vision.patch_size, vision.image_size,
vision.spatial_merge_size, vision.image_mean, vision.image_std) is required,
and absence is a boot-time ValueError.

Why:

- The real mmproj (unsloth mmproj-BF16.gguf) carries every one of these keys,
  so the pin would never fire on a good file - its only possible effect is to
  mask drift in a future converter that drops a key. Fail-fast turns that
  drift into a visible boot error instead of a silently wrong processor
  (wrong mean/std silently poisons pixel normalization and content hashes).
- A pinned default contradicts the no-silent-default rule already established
  for VisionConfig in phase 1: a field the metadata does not describe is a
  fact about a DIFFERENT checkpoint, not a safe substitute.

The only still-pinned processor value is the token budget (min_image_tokens=16,
max_image_tokens=8000) - the clip.* KV set genuinely does not carry it, and it
matches the NVFP4 processor_config.json.

## Facts corrected against the files

- The chat template lives in the MAIN GGUF (tokenizer.chat_template KV, present
  in the unsloth UD-Q3_K_XL file, renders the begin/end image wrappers via an
  emit_image macro) - NOT in the mmproj, which carries no tokenizer KVs at all.
  The research note claiming the GGUF tokenizer ships no template was wrong for
  this checkpoint.
- The image placeholder token string is the ASCII 9-char image-placeholder
  glyph at vocab id 154854, matching the NVFP4 config.json pin; the resolver
  derives the id from the vocab array and fails fast on a mismatch.
