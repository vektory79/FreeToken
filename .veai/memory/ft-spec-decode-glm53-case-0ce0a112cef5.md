---
name: "ft-spec-decode-glm53-case"
description: "Spec-decode GLM-5.3-Flash: case kb/cases/glm53flash-spec-decode (W0 gate), MTP blk.45 in GGUF, FreeToken-only scope"
type: project
lastUpdated: 2026-10-10T14:59
---

# Спекулятивный декод GLM-5.3-Flash: кейс создан, реализация не начата (2026-10-09)

Research-сессия (cloud-LLM): бриф задач в `kb/cases/glm53flash-spec-decode/TASK.md`,
индекс в `kb/cases/README.md`. Не коммитить без запроса.

Ключевые вердикты:
- GGUF UD-Q3_K_XL СОДЕРЖИТ полный MTP/NextN блок blk.45 (DSA+MoE, банки Q3_K/Q4_K
  ~3.6 GB, nextn-клей, embed/lm_head shared) - FreeToken пропускает его целиком
  (models/glm5_next/gguf.py ~:863/:1011). Draft-инфраструктуры в FreeToken нет;
  рудименты: intermediate_states_buffer + EAGLE-маска в GDN-ядре fla,
  num_speculative_tokens в qsa_pool (qwen4_exp).
- Scope-решение пользователя: ТОЛЬКО ft serve, llama.cpp-путь не трогать (хотя
  его билд /media/ai/src/llama-cpp-glm5next build 10836 уже имеет
  --spec-type draft-mtp и graph_mtp для glm5next). Целевой профиль: 64k+
  radix-reuse, якорь 18.04-18.65 tok/s.
- Экономика: verify-батч дорожает по УНИКАЛЬНЫМ парам (слой,эксперт) в CPU-ноге
  гибрида - выигрыш != acceptance. Экосистема: ik_llama.cpp #1513 MTP на
  CPU-оффлоад-риге = 2x МЕДЛЕННЕЕ при acceptance 0.57; qwen35 GPU-full +1.7x;
  Gemma4 MoE - нет выигрыша. W0-гейт кейса: зонд step_cost(M) через N
  параллельных потоков без кода движка.
- Варианты: MTP self-draft из blk.45 (основной), n-gram drafter (первая волна,
  ноль весов), отдельная draft-модель (отклонена - кандидата нет), sidegrade
  llama.cpp (вне scope).
