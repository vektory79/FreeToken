---
title: "Ярусный кеш сессий: якорные числа железных рук по моделям"
date: 2026-09-23
hardware: RTX 5090 32GB; NVMe Samsung 990 EVO Plus
branch-commits: "vektory79 @ d77d15e (фаза 1), 4b6aacb (закалка железа), 2348da5 (фаза-2 S-wave), 88102dd (кодеки QSA/KpoolDSA, KV-only tier, async prefetch)"
status: validated (железо 2026-09-22/23; машина/описание в [../topics/session-cache-tiering.md](../topics/session-cache-tiering.md))
tags: [tiering, session-cache, baselines, gguf, ftw, qwen, glm]
---

# Якорные числа железных рук tiering-кампании

Все руки выполнены серийно на RTX 5090, по одной модели за раз. Общий tier-хвост
конфигурации для всех рук: `--session-tier-ram-gib 10 --session-tier-ssd-gib 100
--session-tier-dir <dir>`; для больших flush'ей задаётся
`FREETOKEN_WORKER_SIGINT_GRACE_S=120`. L1 ограничен 10 GiB операторским решением
(жёсткий MEMLOCK машины 23.56 GiB, без подъёма ulimit). Готовность сервера перед
рукой проверяется настоящим chat-запросом, не дешёвым `/v1/models`.

## GLM-5.3-Flash-UD-Q3_K_XL (GGUF)

- Serve-конфиг: memory-ratio 0.82, KV-пул 400k (0.82/400k); tier 10/100.
- Первый ход после рестарта = HIT, синхронное восстановление 4.19-4.32 с
  (14 сегментов / 4.94 GB blob на диске).
- A/B асинхронной предвыборки на wash-resume нагрузке: медианы хода 3.9 с ON
  против 4.0 с OFF (синхронный restore) - дельта в шуме.
- Интерлив-промывка устранена tier-ом: 0 полных промахов против 11/11 в
  базовой руке без tier.

## Qwen3.8-Flash-Next-NVFP4-FTW

- Serve-конфиг: kv-reserve 262144, mr=1, moe.nvfp4=triton; tier 10/100.
- Первый ход после рестарта = HIT: cached 48256 / 38016, wall 3.3-4.0 с.
- Flush при выключении: 16 сегментов / 11.94 GB.
- Восстановленные resume, пересекающие границу страниц: 0 нарушений целостности;
  бут сервера 31 с.

## Qwen3.8-27B-NVFP4-FTW (dense-hybrid)

- Модель: dense-MLP, но ГИБРИДНОЕ внимание (48 GDN + 16 full-attention слоёв из
  64); KV 64 KiB/token; конвертирована из raw HF modelopt NVFP4 командой
  `ft checkpoint` (pass-through 5.6 с, движок грузит NVFP4 нативно).
- Serve-конфиг: `--memory-ratio 0.85 --max-prefill-length 4096
  --max-running-requests 1` (без MoE-флагов); tier 10/100.
- Бут 21 с; заполнение 2x40k: 8/8 HIT, медиана хода 4.3 с; пик L1 6.23 GiB
  (промежуточной демоции не было).
- Flush при выключении: 21 сегмент -> 23.8 GiB в L2 (избыточность см. ниже);
  на буте после рестарта сработал запланированный компакт: 23.8 -> 14.75 GiB.
- Resume формы BUG-1 (ps=1): все HIT, 0 нарушений целостности, ответы вменяемы.

## Каверния кодеков

- Qwen3.8-27B - dense-MLP с гибридным (GDN) вниманием: он НЕ нагружает
  KV-only путь tier. Железная рука универсального KV-only tier остаётся
  открытой: на станке нет модели с обычным radix-менеджером (только гибриды
  qwen4_exp и glm5_next).
- KpoolDSA-кодек (GLM-5.3) подтверждён железом: GLM tier снова активен.
- Plain DSAKVCache: индексный slab не понижается (pre-existing поведение,
  вне кампании) - нужен кодек-бранч или явный отказ.

## Избыточность L2 на 27B-arm

Flush при выключении кладёт cumulative-границу каждого хода, поэтому L2 хранит
вложенные same-path сегменты: измерено 23.8 GiB L2 на ~63k живых токенов при
вышеописанном 27B-конфиге. Кандидат оптимизации - supersede вложенных сегментов
на offer; запланированный компакт уже возвращает часть объёма на буте
(23.8 -> 14.75 GiB).