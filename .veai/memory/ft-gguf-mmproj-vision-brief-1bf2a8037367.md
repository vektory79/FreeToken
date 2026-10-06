---
name: "ft-gguf-mmproj-vision-brief"
description: "mmproj GGUF vision: кейс ЗАКРЫТ (Фазы 1-5, e35577d..f67449b), 347/347 exact, HW PASS, TRAPS T68-T75"
type: project
lastUpdated: 2026-10-06T21:52
lastRecall: 2026-10-06T21:47
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

UPDATE (2026-10-06, реализация-сессия): Фазы 1-2 ЗАКРЫТЫ, e35577d+fae88c8+747f658 (vektory79). Vision-спека выбирается в build_gguf_shim (mmproj_path в EngineConfig; выбор спеки в parse_gguf_config невозможен - реестр диспетчеризован раньше; GgufConfigShim frozen -> VisionConfig через хук, сброс секции dataclasses.replace). Кандидатский маппинг брифа имел 2 ошибки: mm.post_norm -> visual.merger.post_projection_norm (LayerNorm с bias), v.post_ln -> visual.post_layernorm; v.patch_embd два temporal-среза -> torch.stack(dim=2). Верификация: 348 тензоров -> 347 параметров, ВСЕ exact max_abs_diff==0 против NVFP4-FTW (scripts/verify_mmproj_tensors.py, CPU-mmap); отчёт kb/cases/gguf-glm5next-vision/tensor-verification-report.md. Дистиллят kb/cases/ggug-glm5next-vision/phases-1-2.md; TRAPS T68-T71 (леджер фактически был T01-T67). kb/skills правки staged без коммита (kb-distill коммитить не может). Фазы 3-4 впереди.

UPDATE (2026-10-06, финал CPU-фаз): ФАЗЫ 1-4 ЗАКРЫТЫ полностью: e35577d (--mmproj флаг+спека), fae88c8+747f658 (ридер+верификация 347/347 exact), d864578 (image-процессор из clip.*; chat-шаблон и image_token_id=154854 из ОСНОВНОГО GGUF, в mmproj их нет), 2b91236 (vision в FTW: carrier source_metadata.gguf extra_kvs, бут без --mmproj, старые FTW бит-в-бит), d8dee13 (kb-дистиллят phases-1-2.md+phases-3-4.md+phase-3-notes.md, TRAPS T72-T75, статус кейса). Каждая фаза: Test->Review->Triage->Fix->STOP GATE->commit. HW-волна (фаза 5): boot+e2e raw .gguf (--mmproj) и GGUF-FTW, отдельный заход по VRAM-протоколу - сценариев пока НЕ упаковано.

FINAL (2026-10-06): Фаза 5 HW ПРОЙДЕНА, кейс ЗАКРЫТ (kb f67449b, phase-5-hw.md, README "ЗАКРЫТ ПОЛНОСТЬЮ", все 7 чекбоксов Acceptance [x]). Прогоны: raw+--mmproj TTFT 4.70s / 19.62 tok/s; FTW-vision без флага 3.22s / 20.29; NVFP4 4.21s / 11.64; text-only бут 105s в коридоре. Ответы на тестовую картинку совпали на всех трёх путях. Новый чекпойнт GLM-5.3-Flash-UD-Q3_K_XL-FTW-vision (136G, 18 шардов, 347 visual.*) рядом со старым FTW. Наблюдения: think-блок не убирается enable_thinking=false на всех путях (не vision); usage нет в streaming; +62s boot за башню на host.
