# Верификация конфига 786k на железе (786432 вместо 1048576)

Дата: 2026-09-16. ОДИН контрольный прогон, код не менялся (рабочий фикс
engine/config.py из fix-report.md незакоммичен и присутствовал в дереве).

## Вердикт: BOOT-OK. Расчёт подтверждён слот-в-слот.

Расчёт из fix-report.md: конверт при 1M = 974 слот-0-эквивалента, минус 262144
токенов KV-резерва = -4096 страниц nvfp4 ~ -0.96 GiB ~ +95 слота -> 1069 >=
~1058 (минимум этажей групп) -> проходит. Железо выдало moe_cache_size=1069.

## Команда (единственное отличие - kv-reserve-tokens 786432)

```bash
/media/ai/src/FreeToken/.venv/bin/ft serve --port 18081 --host 127.0.0.1 \
  --model-path /media/ai/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL.gguf \
  --moe-cache-auto --kv-reserve-tokens 786432 --kv-cache-dtype nvfp4 \
  --memory-ratio 0.89 --max-prefill-length 4096 --moe-strategy hybrid \
  --moe-cpu-threads 16
```

## Тайминги

- start 14:17:01 (pid 384690), resolved config +7s, bank scan +17s.
- 14:18:59 (+~111s): moe-план resolved; 14:19:00: KV alloc + CPU executors.
- /ready == 200: READY_AFTER=132s.
- smoke ~14:19:5x - 14:20:07, SIGTERM, exited rc=143 (штатный uvicorn-реап).
- Полный лог: /tmp/ft_verify_786k/server.log.

## Дословные ключевые строки лога

```
[2026-09-16|14:18:59] INFO  --moe-cache-auto resolved moe_cache_size=1069 num_pages=12324 (prefill_overlap=True)
[2026-09-16|14:18:59] INFO  MoE signature group (layers [0, 1, 2, 3, 4, 5, 6, 7, 10, ..., 40]) got 298 slots < 2*288: prefill overlap disabled for this partition (synchronous materialized prefill instead)
[2026-09-16|14:18:59] INFO  MoE signature group (layers [8]) got 288 slots < 2*288: prefill overlap disabled for this partition (synchronous materialized prefill instead)
[2026-09-16|14:18:59] INFO  MoE signature group (layers [9, 41]) got 288 slots < 2*288: prefill overlap disabled for this partition (synchronous materialized prefill instead)
[2026-09-16|14:19:00] INFO  CPU MoE executor ready [pool 1/3: gguf (18, 18, 23) x39 layers]: threads=15 (pinned to cores 0..22) isa=avx2+gguf(iq/qk)+avx2-w4a8k fmt=iq3_xxs (up=iq3_xxs, down=iq4_xs) H=4096 I=2048 experts=288 layers=39 top_k=8 act=swiglu_clamp max_tokens=4
[2026-09-16|14:19:00] INFO  CPU MoE executor ready [pool 2/3: gguf (23, 23, 14) x1 layers]: threads=1 (pinned to cores 23..23) ... layers=1 top_k=8 ...
[2026-09-16|14:19:00] INFO  CPU MoE executor ready [pool 3/3: gguf (18, 18, 14) x2 layers]: threads=1 (pinned to cores 24..24) ... layers=2 top_k=8 ...
[2026-09-16|14:19:00] INFO  Allocating 788736 tokens for KV cache, K + V = 2.88 GiB
[2026-09-16|14:19:00] INFO  Free memory after initialization: 2.86 GiB
[2026-09-16|14:17:15] INFO  Free memory before loading model: 29.46 GiB
```

## Verify-лист

- KV reserve: 788,736 токенов = 12324 страниц x 64. Равен резерву с округлением
  вверх до кратного страницы (786,432 -> 788,736), НЕ вырос. 2.88 GiB
  (против 3.83 GiB при 1M - освободилось ~0.95 GiB, что и дало +95 слота).
- Слоты по группам: [298, 288, 288] (сумма байт-эквивалентов <= конверт 1069;
  группа 0 урезана пропорциональным правилом до 298, группы 1/2 на этаже 288).
  Все три группы < 2E=576 -> per-partition prefill overlap ДЕГРАДИРОВАЛ в
  synchronous materialized prefill - ровно семантика Phase 5
  (Engine._partition_prefill_overlap), известное ожидаемое поведение.
- CPU MoE executor ready: 3 пула [15, 1, 1] потоков, cores 0..22/23/24,
  isa avx2+gguf(iq/qk)+avx2-w4a8k - совпадает с кампанией (pools [15,1,1]).
- Free memory after initialization: 2.86 GiB. Порог для 4096-чанков по
  verify-листу ~2.9 GiB - на границе, но это ТОЧНО профиль исторически
  рабочего A1 (1M/0.89/4096 на FTW: free after init 2.86 GiB, 4096-чанки
  проходили). Smoke-prefill прошёл; глубокие филлы не тестировались (вне
  скоупа одного прогона).
- Fetch fraction: в boot-логе строки нет (материализуется на decode-интервалах
  при moe_hybrid_max_fetch=-1 auto) - не зафиксирована, отсутствие строки не
  признак проблемы.
- Smoke: HTTP 200, тело:
  {"id":"chatcmpl-0","object":"chat.completion","created":1789557607,
  "model":"GLM-5.3-Flash-UD-Q3_K_XL.gguf","choices":[{"index":0,"message":
  {"role":"assistant","content":"","reasoning_content":"The user just said
  \"Hi\""},"finish_reason":"length"}],"usage":{"prompt_tokens":13,
  "completion_tokens":7,"total_tokens":20}}
  content='' + reasoning_content при max_tokens=8 - известное поведение glm
  reasoning-парсера, НЕ ошибка; HTTP 200 + usage = успех.

## Shutdown и census

- SIGTERM -> "Scheduler exiting gracefully..." -> rc=143 (штатный), без hang,
  SIGKILL не понадобился.
- Postflight: pgrep - чужих процессов нет; sem.mp- = 0; VRAM 1406 MiB used /
  30743 MiB free = baseline.
- Preflight (до прогона): sem.mp- = 0, VRAM 1406 MiB / 30743 free, чужих
  процессов нет.
- Harness: /tmp/ft_verify_786k/runner.sh (trap-on-EXIT, FAILPAT с ValueError,
  /ready-полл, hard timeout 900s, SIGTERM->30s->SIGKILL). Ни одного срабатывания
  FAILPAT за весь прогон.

## Итог для пользователя

Конфиг 786432 (единственная правка --kv-reserve-tokens) BOOT-OK на железе:
гибрид поднимается, KV 788k токенов nvfp4 (2.88 GiB), слоты [298/288/288],
smoke 200. Издержка против 1M: KV-ёмкость 788k вместо 1M токенов и overlap-
prefill деградировал во всех партициях (он и при 1M деградировал бы - там
boot вообще не доходил до этого места). Free-after-init 2.86 GiB - тот же
профиль, что у исторически рабочего 4096-конфига A1.
