---
name: "ft-gguf-native-serving-skill"
description: "gguf-native-serving skill ledger: TRAPS T01-T95 committed (73cceb0); run-garbage rule added uncommitted to sec.11"
type: project
lastUpdated: 2026-10-26T14:28
lastRecall: 2026-10-26T14:25
---

# GGUF native serving skill: ledger (/media/ai/src/FreeToken/.veai/skills/gguf-native-serving/)

Git-tracked, single location, no .tasks copies (user directive 2026-09-15). Skill-file commits are made only on user command; self-update passes themselves never commit.

## Structure
- SKILL.md: YAML preamble (never touch), 7-phase pipeline, counter line (currently "95 ловушек").
- TRAPS.md: T01-T95 + recipes D01-D11, dated campaign sections.
- ORCHESTRATION.md (11 sections): mode selection, phase=wave quality loop, STOP GATE, handoff, process-hygiene, sec.8 commit discipline (explicit-path staging, ZERO .veai/memory staged, no push), sec.9 evidence rules, sec.10 counter sync (T01-T95 + D01-D11), sec.11 session-end self-update.
- tasks/task-00..07 briefs: task-00 carries the working-reference question + upstream-alignment-or-waive; task-06 the expected-result classes (incl. hybrid-radix per-chunk donation A/B replay).

## Counter sync rule (sec.10/11)
After every self-update sync trap counters across TRAPS.md H1 / SKILL.md counter line / ORCHESTRATION sec.10. Known past miss: a8fb231 added T64-T66 without syncing (63 stale), fixed at T67 pass.

## Committed state
- All TRAPS entries T01-T95 are COMMITTED on vektory79 as of 2026-10-26: latest skill commits a20519e (fix3 T59-T63), a8fb231 (T64-T66), 5b64d6c (T76-T77), 542ece4 (T78-T79), 73cceb0 docs(skill): add d2 mmproj traps t92-t95, sync trap counters (3 files, 34+/3-). Working tree fully clean after cf6a5ed chore(memory).

## Campaign knowledge in TRAPS (ranges)
T30-T38+D06-D08 hybrid/CPU-kernel; T39-T47+D09/D10 MMQ prefill; T48-T50 fix-1 radix; T51-T52 v3a m-tile; T53-T56 dense-q80; T59-T63 fix3 hybrid-radix; T64-T66 decode-chain microfusion; T67 boot/shutdown limiter-attribution; T68-T79 mmproj + tier waves; T80-T91 mmproj vision/GGUF serving campaigns; T92-T95 round6 D2 (T92 fixture stand-ins mask real-file KeyError - diff fixture key shape vs real artifact, tolerance in main primitive with named error + real-shape fixture; T93 vacuous pytest.raises match hitting tmp-dir name - match strings are distinctive fragments of real errors; T94 skip-marker classification via `pytest -p no:skipping` probe + semantic dedup, grep counts re-measured never recalled; T95 metadata-only FTW carrier -> family hook gguf_carrier_mm_config from build_gguf_shim, encoder-families only, pinned-reference fallback).

## sec.11 self-update protocol (user directive 2026-09-18)
At session end, delegate integration of new session knowledge to a Code subagent: TRAPS new-T-or-extension, task-06 expected classes, sec.7/8 bullets; counters synced; YAML untouched; no README; pass never commits (user gates the commit separately).

## Self-update review practice
Each self-update pass gets a review (anchor correctness: cite the real gate/span, not a neighboring one - precedent T60 wrong-gate fix). Reviews history: .tasks/skill-review-gguf-native-serving/review-1..4.md.

## Обновление 2026-10-26 (правка после чистки .tasks)
По запросу пользователя в скилл добавлено правило end-of-campaign run-garbage assessment: ORCHESTRATION.md sec.11 новый bullet "Run-garbage assessment:" (между Execution и Boundaries; inventory du, классификация регенерируемых прогонов vs session knowledge (.md/.py/.sh/.json всегда keep), порог heavy = total >1 GiB или любой артефакт >100 MiB, itemized-предложение пользователю (item + size + как регенерировать), удаление ТОЛЬКО после явного подтверждения, freed-факт в handoff/progress) + одно предложение в session-end абзаце SKILL.md. Ревью PASS (2 неблокирующих замечания: "present the user an itemized" -> "present the user with"; мелочь). Правки НЕЗАКОММИЧЕНЫ (user-gated).
