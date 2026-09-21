# Глубокие филлы 786k на железе (лестница 256k -> 512k -> 786k)

Дата: 2026-09-16. КОД НЕ ПРАВИЛСЯ. Один прогон верифицированной 786k-команды
(та же, что в verification-786k.md), инкрементальные филлы с radix-переиспользованием
префикса. Client-лестница: /tmp/ft_deepfills/client.py, runner:
/tmp/ft_deepfills/runner.sh, полный лог: /tmp/ft_deepfills/server.log.

## Вердикт: PASS до 512k, FAIL на ~678k (86% пула) - CUDA OOM в DSA indexer

Полный пул (786k токенов) НЕ влезает: на глубине ~678k токенов (usage 0.86
пула) прегилл падает с torch.OutOfMemoryError в DSA indexer kpool-select
(dsv4_indexer.py:72) - workspace растёт с глубиной контекста и исчерпывает
post-init headroom 2.86 GiB. Decode-смоук на полной глубине не достигнут
(OOM раньше).

## Boot-числа (сверка с verification-786k.md)

Совпало: free-before-load 29.47 GiB (было 29.46), free-after-init 2.86 GiB
(идентично), CPU MoE executor 3 пула [15,1,1] (строки дословно те же), гейт
_gguf_auto_floor_gate МОЛЧИТ (ни одного ValueError/Traceback в boot) -
regression false-positive НЕ случился.

Бенигн-дрейф (red flag зафиксирован, объяснён): moe_cache_size=1071 (было
1069), num_pages=12315 (было 12324), "Allocating 788160 tokens for KV cache,
K + V = 2.87 GiB" (было 788736 / 2.88). Причина: baseline-free чтение
отличается на ~10 MiB между прогонами (29.47 vs 29.46 GiB), greedy fill
распределяет дельту в 2 слота; KV по-прежнему >= резерва 786,432
(788,160 = 12315 x 64). Слоты по группам в лог на этот раз не печатались
(строки групп появляются только при сплите с деградацией overlap - обе
конфигурации идентичны по составу групп [39/1/2]).

## Лестница (детерминированный префикс-монотонный filler; usage от сервера)

| probe | глубина факт (usage.prompt_tokens) | цель | ok >=0.97 | время | VRAM после (used/free MiB) |
|---|---|---|---|---|---|
| 1 | 226,190 | 256k | НЕТ (86%) | 912.1s | 31,468 / 681 |
| reuse | 226,176 cached / 14 new | - | 100% reuse, 3.6s | - | - |
| 2 | 517,488 | 512k | ДА (98.7%) | 1160.1s | 32,050 / 99 |
| 3 | УПАЛ на ~678k (usage 0.86) | 786k | - | ~19 мин до OOM | OOM |

- Prefix reuse ПОДТВЕРЖДЁН: reuse-проб того же текста вернулся за 3.6s с
  "#new-token: 14, #cached-token: 226176" - лестница валидна, дельты
  корректны.
- probe1 недобор 226,190 < 0.97x262,144 - ошибка МОЕЙ токен-оценки (клиент
  предполагал 1.3 tok/word, фактический коэффициент filler-текста 1.101);
  клиент измерил коэффициент с пробы 1 и дельты 2/3 считал по нему - probe2
  попал в цель (98.7%). Это артефакт оценки, не сбой сервера.
- Prefill throughput стабилен до смерти: 248-255 tok/s (последняя строка
  248.71 на 17:01:47, usage 0.86). Деградации скорости перед OOM нет.
- Серверные "input throughput" строки: сотни чанков; дельта probe1 912s,
  дельта probe2 1160s (291,298 токенов => ~251 tok/s эффективно).

## OOM (полный traceback)

Точная глубина провала: total_4096_chunks=165 => 165 x 4096 = 675,840 новых
токенов (+14 reuse) => ~675,854 токенов в дереве (usage 0.86 x 788,160 =
677,818, последний чанк мог быть неполным). До полной цели 786k не хватило
~108k токенов.

```
  File "python/freetoken/models/glm5_next/model.py", line 125, in forward
    x = self.self_attn.forward(x)
  File "python/freetoken/models/glm5_next/attention.py", line 182, in forward
    o_latent = ctx.attn_backend.mla_forward(
        q_absorbed.contiguous(), q_pe, c_kv.contiguous(), k_rope,
        self.layer_id, ctx.batch, indexer_inputs=indexer_inputs,
    )  # [T, H, kv_lora_rank]
  File "python/freetoken/attention/dsa.py", line 235, in mla_forward
    return self._prefill(md, layer_id, q_nope, q_pe, batch, indexer_inputs)
  File "python/freetoken/attention/dsa.py", line 303, in _prefill
    self._select_prefill(
        self._idx_slot[layer_id],
        ...
        kv_lens[i] - (qo[i + 1] - qo[i]),  # cached_len == first position
        ...
    )
  File "python/freetoken/attention/dsa_indexer_kpool.py", line 276, in _select_prefill
    picks = self.indexer_select_prefill(
        scores.unsqueeze(0), start_pos=start_pos + s0, seqlen=s1 - s0,
        ratio=kp, topk=k_sel, offset=0,
    )  # [1, s1-s0, k_sel] pool ids
  File "python/freetoken/attention/dsv4_indexer.py", line 72, in indexer_select_prefill
    scores = scores + torch.where(blk >= live, float("-inf"), 0)
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 336.00 MiB.
GPU 0 has a total capacity of 31.40 GiB of which 371.12 MiB is free.
Including non-PyTorch memory, this process has 29.79 GiB memory in use.
Of the allocated memory 28.95 GiB is allocated by PyTorch, with 22.00 MiB
allocated in private pools (e.g., CUDA Graphs), and 138.45 MiB is reserved
by PyTorch but unallocated.
```

Механика: kpool-select матрица scores [1, chunk 4096, k_sel] растёт линейно с
глубиной контекста (k_sel ~ контекст/kpool-страниц); на ~678k контекста один
оператор маскирования просит 336 MiB при 371 MiB свободных. Траектория free
VRAM: 2,845 MiB (boot) -> 681 (после 226k) -> 99 (после 517k) -> OOM. Аллокатор
удерживает трансзиенты прегилла, indexer-delta съедает остаток; к 517k глубине
запас был 99 MiB - OOM был делом времени, случился на первой крупной indexer
аллокации глубже (~678k). Подтверждает известный механизм "DSA indexer
workspace ~0.62 KB/context-token растёт в тот же headroom".

## Shutdown и census

- Watchdog поймал OutOfMemoryError (FATALPAT), убил клиента (rc=143) и сервер:
  backend умер ("Backend worker is gone"), uvicorn встал в shutdown-wait ->
  SIGKILL эскалация -> сервер rc=137. VRAM вернулась к baseline.
- Postflight: pgrep - чужих процессов нет; VRAM 1010 MiB used / 31,139 MiB free;
  sem.mp- = 3 СРАЗУ после смерти (известная утечка tqdm-семфоров при SIGKILL
  убитого дерева) - мои утечки удалены руками, контрольный census: sem.mp- = 0.
- Preflight был чист: sem.mp- = 0, VRAM 1386/30,763, чужих процессов нет.
- Общее время прогона: 2853s (boot 116s + клиент ~2737s) - лимит 5400s не
  достигнут, прогон завершён отказом раньше.

## Итог

- 512k глубина на этом конфиге стабильна (probe2 200 OK, 517,488 токенов).
- Полный пул 786k НЕ достижим на этой конфигурации: OOM в DSA indexer на
  ~678k (86% пула). Практический потолок контекста между 517k и 678k
  (ближайшая граница не уточнялась - вторым прогоном запрещено условием).
- Decode-смоук на полной глубине не выполнен (OOM раньше); decode на 517k не
  запрашивался по плану (только prefill-probes max_tokens=1).
