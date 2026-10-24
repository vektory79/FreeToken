---
name: "hybrid-moe-pair-split-mechanics"
description: "Hybrid MoE split per (layer,expert) pair via -1 marker not per layer; LRU pool ~3 steps; CPU GEMV reads RAM weights"
type: project
lastUpdated: 2026-10-11T18:56
lastRecall: 2026-10-23T23:12
---

# Hybrid MoE: сплит CPU/GPU по парам (слой,эксперт), не по слоям (код-верификация 2026-10-10)

В ft serve --moe-strategy hybrid деление CPU/GPU идёт ПО ПАРАМ (слой,эксперт) динамически
каждый шаг, а не по слоям. `_decode_hybrid` (layers/moe.py:305-352) переписывает topk_ids
на месте: хит -> slot id (GPU GEMM), мисс-перелив -> маркер -1 (layers/moe.py:324); итог
`return gpu_routed + cpu_routed` (:352). Послойный флаг --moe-cpu-layers
(offload_cache.py:157-162) существует, но по умолчанию пуст и в hybrid не участвует.

Why: это определяет цену verify любого M-позиционного батча (в т.ч. будущего драфта).
Токен всегда проходит ВСЕ 42 MoE-слоя (glm5_next/moe.py:85-94; частичного прохода транка
нет), и на каждом слое его топ-8 делится по одной и той же таблице резидентности.

Механика (file:line, проверено research-агентом против кода):
- Мисс имеет ДВЕ судьбы: кап-доля фетчится по PCIe в VRAM-слот и считается на GPU
  (cache.copy_missing, layers/moe.py:340; кап num_fetch=min(num_missing,max_fetch) или
  ~frac*num_missing на УНИКАЛЬНЫХ мисс-экспертах слоя-шага, offload_kernels.py:342-351);
  перелив сверх капа уходит на CPU-ногу, куда едут ТОЛЬКО активации+ids в pinned буферах
  (cpu_executor.py:966-984), C++ GEMV читает веса ПРЯМО из host-RAM по сырым указателям
  (_resolve_banks, :476-480), результат H2D назад (decode_sync, :1006-1015). Веса
  CPU-миссов по PCIe не ходят никогда.
- Резидентность: timestamp-LRU на эксперте, пересчёт каждый шаг
  (_ensure_experts_hybrid_kernel, offload_kernels.py:291-407, BY_RECENCY, env
  FREETOKEN_HYBRID_FETCH). Доля fetch f = pcie_bw/cpu_bw статична, резолвится на буте
  (engine.py:1214-1244; семантика benchbw-профиля - в памяти ft-serve-moe-flags-semantics).
- Политика сверена с kb/topics/moe-hybrid-cost-model.md - совпадает. Ценовая модель шага
  (CPU-leg bandwidth x fetch volume, не фиксированный floor) - в памяти
  gguf-hybrid-decode-handshake-floor.

Следствие (опирается на W0/W0.5 кейса glm53flash-spec-decode): слот-пул ~1059-1167 слотов
на 12096 возможных пар (42x288) = глубина ~3 шага. Поэтому у любого токена на 64k покрытие
горячего окна ~c_W med 0.50 (W0.5), ~168 мисс из 336 пар, и P(чисто-GPU пути) ~ 0
(нужно 8/8 попаданий на каждом из 42 слоёв одновременно). На 2k рабочий набор влезает
в пул - CPU-нога почти простаивает. Verify M позиций: expert_ids = M*top_k, кап/доля
применяются к уникальным мисс-парам слоя-шага -> CPU-нога поглощает перелив
пропорционально с ростом M (отсюда почти линейный R(M) в W0).

How to apply: вопросы "какой слой где считается" для гибрида решаются парами, а не
слоями; цену draft/verify оценивать от уникальных пар с капом f, не от слоёв.
