---
name: "ft-gguf-mmproj-vision-brief"
description: "GGUF glm5next vision brief: ft checkpoint packs vision into FTW; flag only for raw .gguf"
type: project
lastUpdated: 2026-10-05T23:45
lastRecall: 2026-10-05T23:33
---

# GGUF glm5next vision (mmproj) support - research done, brief ready

Research-only session 2026-09-27 (vektory79, HEAD edd6309). Deliverable:
`kb/cases/gguf-glm5next-vision/TASK.md` (NOT STARTED; moved to kb/cases on
user request 2026-09-27 - NOT in .tasks/, indexed in kb/cases/README.md).
Implementation is a separate session.

USER DECISIONS (do not reopen):
1. Cover BOTH boot paths: raw .gguf and GGUF-FTW.
2. mmproj path via explicit CLI flag (e.g. --mmproj); no auto-discovery
   (mmproj sits one dir above the main gguf in the unsloth layout).
3. Image-processor config derived from mmproj GGUF metadata (clip.* keys)
   in code; no external preprocessor_config.json.
4. Brief lives in kb/cases (versioned), not .tasks/.

Key findings (verified vs code):
- NVFP4 reference path is complete: Glm5NextVisionModel (models/glm5_next/
  vision.py), conditional-generation class builds self.visual on
  is_multimodal, Glm5NextMMProcessor needs AutoImageProcessor files.
- GGUF path is text-only: spec Glm5NextGGUFForCausalLM maps to
  Glm5NextForCausalLM; parse_gguf_config never sets vision_config;
  iter_gguf_weights has no include_vision kwarg (load_weight passes it only
  when spec.encoders - adding encoders to the GGUF spec requires the kwarg).
- mmproj-BF16.gguf: arch clip, 348 tensors (224 F32 / 124 BF16), llama.cpp
  names v.blk.N.* + mm.*; mapping candidate table in the TASK.md.
- Puzzle flagged in brief: mm.post_norm.bias exists in mmproj but FreeToken
  post_layernorm is bias-free RMSNorm - resolve before mapping.

Why: kb had the mmproj facts (phase2-tensor-map.md SS11) but nothing on the
FreeToken-side wiring; the brief bridges them with verified anchors.

How to apply: the implementation session starts from
kb/cases/gguf-glm5next-vision/TASK.md checklist (CONTRIBUTING first, re-verify
anchors, resolve Risks 1/3/4 before coding); HW e2e wave is separate per
VRAM protocol.

UPDATE (2026-09-27, same session, user follow-up): USER DECISION 5 added -
`ft checkpoint` MUST pack vision weights INTO the FTW (visual.* tensors +
tower metadata from clip.* keys); GGUF-FTW boot is tower-self-contained and
does NOT need --mmproj. The flag is only for raw .gguf boot. Brief amended:
decisions item 4, Phase 4 spec (rewritten from "FTW stores no vision" to
conversion packs it), acceptance CPU items, HW item. Stale prior design
("FTW has no vision tensors, mmproj always via flag") is superseded.
