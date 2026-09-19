---
name: "ft-gguf-nvfp4-1m-capacity-frozen-shim"
description: "GGUF glm5next: 1M nvfp4 infeasible on 32GB (786432 verified, fill ceiling ~678k); frozen-shim crash; slot-floor gate"
type: project
lastUpdated: 2026-09-16T18:14
lastRecall: 2026-09-19T03:10
---

# GGUF glm5next serve: 1M nvfp4 KV capacity + frozen-shim crash lesson (2026-09-15)

Debugging the user's 1M command on vektory79 post-merge f786d7c. Artifacts: .tasks/ft-serve-gguf-nvfp4-1m-crash-work/ (repro-report.md, fix-report.md, verification-786k.md, failfast-report.md).

## Frozen shim crash (merge regression, FIXED)
- Merge f786d7c (main afd99cb, commit 08d728d "image input on Qwen families") added setattr-nulling of unbuilt encoder sections in EngineConfig.model_config (engine/config.py:141, ENCODER_SECTIONS = vision_config/audio_config in mm/config.py:11). GgufConfigShim (models/gguf/config.py:25) is a @dataclass(frozen=True) and does NOT define those fields; copy.copy of a frozen instance stays frozen -> dataclass __setattr__ raises FrozenInstanceError unconditionally. Crash at Engine.__init__ -> _adjust_config, ~7s after start, BEFORE weights/KV/MoE: broke EVERY `ft serve *.gguf` regardless of flags.
- Fix: `if key not in built and hasattr(hf_config, key):` guard (HF semantics byte-preserved, verified by golden Recorder test; GgufConfigShim verified to lack the fields). Regression test: tests/models/test_glm5_next_gguf.py::test_engine_model_config_resolves_over_the_frozen_shim; mm golden: tests/models/test_qwen3_vl.py::test_unbuilt_encoder_sections_are_nulled_only_where_they_exist.
- LESSON: any new code that mutates hf_config must hasattr-guard first - the frozen shim makes mutations flag-independent instant crashes. Any future merge touching hf_config mutation needs a GGUF boot smoke.

## 1M nvfp4 KV + gguf hybrid: INFEASIBLE on RTX 5090 32GB
- Global auto plan solves (974 slots + 1M nvfp4 KV pages), but the 3 gguf signature groups (39/1/2 layers) have per-partition floors of 288 slots each = ~1058 slot-0-equivalents (byte-weighted, binding group = layer 8) > 974 envelope -> ValueError in _split_moe_cache_budget (engine.py:732): "budget cannot fund the per-signature slot floors ... below the partitioned minimum of ~8510 slots". Same class as the 2E-overlap lesson in ft-serve-prefill-overlap-512k-infeasible: the greedy auto plan only knows the pre-split total.
- An explicit --kv-reserve-tokens is a hard floor; the planner must not silently shrink it, so no valid plan exists at 1M.
- VERIFIED working alternative (single variable changed, rest = user's command): --kv-reserve-tokens 786432 -> slots [298,288,288] = 1069 (prediction 974+95 matched exactly), KV 788,736 tokens / 2.88 GiB, free after init 2.86 GiB (same profile as the historically-working 1M/0.89/4096 A1), ready 132s, smoke HTTP 200, clean SIGTERM shutdown (rc 143), census clean. All three groups < 2E -> per-partition prefill overlap degrades to synchronous (Phase 5 semantics, logged). Deep fills NOT tested in that run.

## Early slot-floor gate (UX gap CLOSED 2026-09-15)
- cache_budget.py: partition_floor_slots (byte-weighted group floors in slot-0-equivalents) + check_partition_floors (shared honest reject: groups/layers, 288-slot floor each, weighted need, envelope, remedy hint). moe/expert_banks.py: gguf_signature_groups - header-only scan returning (groups, per-group slot bytes); returns None when groups are unprovable, and the gate MUST stay silent on None (late check remains the only barrier). engine.py: Engine._gguf_auto_floor_gate called in _init_offload_moe_cache BEFORE load_expert_banks; the late raise in _split_moe_cache_budget reuses the same helper as a safety net.
- Payoff: real-file reject at ~+17s instead of +85s (skips the 68s serial bank build). Real-file need = ceil(288*38.125/10.375) = 1059 slot-0-eq vs envelope 974.
- INVARIANT (review-proven, integer-exact): late check fires iff byte_cap < ceil(E*SUM(per_group_slot_bytes)/per_group_slot_bytes[0]) = partition_floor_slots; boundary envelope == need stays silent; non-gguf models untouched (fmt-resolve mirrors _legacy_expert_banks, per_expert_bytes override neutral). Gate sits after the split-residency flip where overlap is final - floors are 1E; per-partition DEGRADE keeps 2E out of the floor math.
- Tests pin: gate-before-load order (sentinel with monkeypatched load_expert_banks), silence for non-gguf format and gguf-format-on-non-gguf-path, boundary cases (need 15 > envelope 14 rejects; 15 == 15 silent), reject message prints per-group bytes/slot widths (numbers must stay explainable). Review verdict: commit-ready, no blocker/major.

## Commit state (2026-09-15, user-authorized)
- vektory79: 8ce2657 "fix(engine): guard hf_config setattr for frozen gguf config shim" (config.py + 2 test files) and 8bcca7b "fix(engine): fail fast on unfundable gguf slot floors before banks load" (cache_budget.py, expert_banks.py, engine.py, test_gguf_expert_banks.py), both on top of merge f786d7c. No push (user rule). test_gguf_expert_banks.py = 40 passed incl. the 2 CUDA tests after the deep-fill wave freed VRAM (they OOM'd earlier only from cross-wave GPU contention = Environment).

## Deep fills on the 786k config (measured 2026-09-15)
- Prefix reuse WORKS on gguf (probe 2: "#new-token: 14, #cached-token: 226176") - incremental ladder is valid.
- PASS: 226k and 517k prompt_tokens fills (HTTP 200, ~251-254 tok/s prefill). FAIL: OOM at ~678k filled (86% of the 788,160-token pool): torch.OutOfMemoryError "Tried to allocate 336.00 MiB ... 371.12 MiB is free" in attention/dsv4_indexer.py:72 (kpool-select scores [1,4096,k_sel] grow linearly with depth) via dsa_indexer_kpool.py:276. Free-VRAM trajectory: 2845 -> 681 (226k) -> 99 (517k) -> OOM MiB. None of our diffs are in that path - capacity property, not a regression.
- Full-pool fill does NOT fit at reserve 786432; 517k+ leaves only 99 MiB (decode-time transients razor-thin). Practical deep-fill lever: a SMALLER reserve frees headroom the indexer consumes (e.g. 524288 -> ~0.87 GiB more headroom; proposed, NOT hardware-verified). Gate stayed silent on hardware at boot (slots 1071 vs 1069 - benign ~10 MiB baseline drift).
- Artifacts: verification-deepfills.md in .tasks/ft-serve-gguf-nvfp4-1m-crash-work/ (full traceback, depth table).

## Merge round 2 (2026-09-15, user request "origin/master" = actually origin/main cac247a)
- origin/master does NOT exist; the real target was origin/main. cade1a9 (FTW vision-encoder #486, 23 files) + cac247a (release 0.1.3). Merged as 5d93a30 "Merge branch 'main' into vektory79" (parents 8c0f1e7 [user's .veai Memory commit], cac247a). ZERO conflicts (ort auto-merged engine.py, glm5_next/__init__.py, weight.py); 0 duplicates, "main wins" not needed. Semantic check: main's load_weight keep-filter rewrite stays inside load_weight (FTW/per-family); load_gguf_moe_expert_sources untouched and structurally independent. 346 passed / 0 failed targeted + boot smoke /ready 91s, clean shutdown. Artifacts: .tasks/merge-main-round2-work/.
- BOOT-NUMBERS ARE ENVELOPE-DEPENDENT: post-merge boot gave moe_cache_size=1091, slots [320,288,288], KV 789120 tok / 2.88 GiB, free-after-init 2.90 GiB vs the older reference 1071/[298,288,288]/788160/2.86 - all drifted from ~250 MiB more baseline-free VRAM at start. When comparing boots, the invariants are: slot gate SILENT on feasible configs, slots >= working set 336 total, more free = safer; absolute slot/page numbers move with start-free VRAM and version.
