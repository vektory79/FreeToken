---
name: "ftw-index-scans-silent-none"
description: "GGUF/FTW signature scans degrade to None silently (expert_banks.py:285/:406/:517), not raw errors"
type: project
lastUpdated: 2026-09-20T19:45
lastRecall: 2026-09-22T12:00
---

# GGUF/FTW signature-group scans degrade to None silently (by contract)

moe/expert_banks.py: `gguf_signature_groups` -> `_gguf_bank_role_stacks` (:285), `gguf_ftw_signature_groups` (:517), `_ftw_gguf_index_meta` (:406) each catch Exception and return None. An unreadable ftw index therefore does NOT raise - the floor gate (`_gguf_auto_floor_gate`) falls through to the late split check (`_split_moe_cache_budget` / check_partition_floors).

Why: even the round-3 merge wave and its discovery framed the accepted gap as "unreadable ftw index raises a raw error"; the naive read misses the silent-None contract, and review pass A disproved it (2026-09-20).

How to apply: when reasoning about failure modes of the gguf floor gate / auto moe cache sizing, treat these scans as best-effort (None = skip), never as raising paths; do not wrap them in WeightLoadError - that would contradict main's #518 semantics (config failures keep their own type).
