---
title: "Legacy FTW fold: RAM-пик 198 GiB и zero-scratch - готово"
date: 2026-10-24
hardware: "RTX 5090 rig, 188 GiB RAM; буты GLM-5.3-Flash UD-Q3_K_XL FTW-носителей (136G, 42 MoE-слоя), NVMe 990 EVO Plus"
branch-commits: "vektory79: f12e554 поверх 8bba8e5; волна 40 кампании round5-deferred-ports"
status: done
tags: [ftw, gguf, checkpoint, legacy, fold, ram-peak, boot, moe, round5, deferred-ports]
---

# Legacy FTW fold: RAM-пик 198 GiB и zero-scratch

**Вердикт (читайте это первым).** Легаси FTW-носитель (в индексе нет
`gguf_tensor_types`) до коммита f12e554 бутался через ветку fold в
`load_ftw_banks`: merged-банки `gate_up` аллоцировались upfront, плюс 84
scratch-банка (вторые копии gate+up), а fold-копирование гоналось наперегонки
с 8-worker O_DIRECT пулом чтения - пик 198.2 GiB на машине со 188 GiB,
OOM системы. Zero-scratch affine remap пишет джобы чтения сразу в координаты
merged-банка; на реальном glm5next-носителе проходят все 42/42 слоя (fallback
не нужен). A/B на железе, один и тот же рецепт: пик суммы группы 142.47 GiB
(легаси) против 143.21 GiB (новый носитель) - профиль идентичен, chat 200,
decode 24.64 tok/s, rc 143 graceful. Карта раунда -
[round5-ggml-replatform-deferred-ports.md](round5-ggml-replatform-deferred-ports.md).

## Root cause: порядок аллокаций, а не объём данных

Банки легаси-носителя весят 125.17 GiB (gate 36.51 + up 36.51 + down 52.15;
формы gate/up `[288,2048,1568]`, down `[288,4096,1088]`, E=288). Старый fold:

- merged `gate_up` 42x[288,3616,1568] = 73.02 GiB аллоцировался upfront;
- плюс scratch 84 банка - вторые копии gate+up, ещё 73.02 GiB;
- down 52.15 GiB читался сразу в финальные банки.

Если fold отставал от читающего пула, scratch жил одновременно с merged,
и пик достигал 198.2 GiB. Пик определялся не суммой данных (125 GiB), а тем,
какие аллокации пересекаются во времени: scratch, живущий до конца фазы
чтения, складывается с финальным размещением.

## Фикс: affine dest remap (f12e554)

Джобы чтения половины пары нарезаются по границам экспертов И кратностям
chunk, внутри джоба действует смещение `dest_off = e*stride + base + off`
(`base` 0 для gate, gC для up). Условие-предикат на слой: равенство E обеих
полов, `g_cx % 4096 == 0`, `u_cx % 4096 == 0` и `nbytes == E*cx` (без ALIGN
пэд-хвоста) - иначе разрезы ломают O_DIRECT-выравнивание, и слой падает в
старый путь mmap-scratch + fold (остался как fallback для экзотических форм).
CPU-проба предиката против индекса носителя: 42/42 слоя zero-scratch,
fallback 0. Сопутствующее: контракт задач `_run_shared_reads`/`run_ring_reads`
стал 4-элементным `(dest, entry, on_done, jobs)` - правка затронула
ftw_uring.py и два теста io_backend (см. ниже). Пин-контракт не изменён:
merged-банк сабмитится в PinPipeline один раз, когда ОБЕ половины дошли
(счётчик под Lock), что эквивалентно старому fold-then-submit.

## Метод A/B (реннер с предохранителями)

Рунер `.tasks/round5-deferred-ports/runners/zero-scratch-boot.sh`:
1 Гц сэмплер VmRSS/VmHWM -> CSV; SIGKILL всей группы при >155 GiB; tail-сторож
лога с FAILPAT (`OOM|out of memory|Traceback|CUDA error|Aborted`); готовность -
РЕАЛЬНЫМ chat-запросом (не /v1/models); census до/после; SIGTERM-шатдаун с
эскалацией. Ловушки, которые вскрылись:

- Банки живут в worker-процессе: ft-main VmRSS во время загрузки ~1.5 GiB,
  сумма группы ~142 GiB. Фьюз по VmRSS главного процесса не сработал бы
  никогда - предохранитель обязан смотреть сумму группы процессов.
- Сравнивать надо с бутом НОВОГО носителя тем же рецептом, а не с байтами
  банков: ожидание "пик 125-135 GiB" из спеки было калибровано по байтам
  банков (125.17), реальный пол рецепта (банки + dense + session tier +
  CUDA-контексты) - 142.5-143.2 GiB, одинаковый для обоих носителей.

## Числа (рецепт пользователя: hybrid, kv fp8, reserve 400k, memory-ratio 0.82,
max-prefill 8191, moe-cpu-threads 16, tier 10/50 GiB, порт 18081)

| бут | носитель | пик группы | банк-фаза | ready(chat 200) | decode | rc |
|---|---|---|---|---|---|---|
| A | GLM-5.3-Flash-UD-Q3_K_XL-FTW-vision-new (merged, без fold-ветки) | 143.21 GiB | 142.45 GiB | 148 с | 23.58 tok/s | 143 |
| B | GLM-5.3-Flash-UD-Q3_K_XL-FTW-vision (легаси, zero-scratch) | 142.47 GiB | 142.47 GiB | 81 с | 24.64 tok/s | 143 |

Decode - медиана 3 стримов по 256 токенов того же промпта (базлайны стека
21.00/20.86). Пик B совпадает с плато steady (банк-фаза = пик = steady) -
scratch-горба нет; старый стек на том же носителе: 198.2 GiB (-55.7 GiB).
Фьюз 155 GiB не сработал ни разу; digit-split (17+25 -> 42) PASS на обоих
бутах; census после - чист. Артефакты: `.tasks/round5-deferred-ports/
zero-scratch/{bootA-new,bootB-legacy}/` (rss.csv, result.env, decode.json,
server.log, runner.log).

## Батарея STOP GATE

`uv run --extra dev python -m pytest tests/checkpoint tests/moe -q`:
1 failed (известная базлайн-пара offload FakeGraph) / 252 passed / 21 skipped;
smoke `tests/kvcache -q`: 307 passed / 1 skipped. Первая попытка поймала 2
реальных фейла `test_ftw_io_backend.py` - тесты конструировали задачи
старым 3-tuple контрактом; fails-before проверен stash-ем (на HEAD 16 passed),
после перевода тестов на 4-tuple - 16 passed. Урок: контрактные правки
`_run_shared_reads` обязаны прогонять io_backend-тесты, а не только
gguf_banks (byte-identical срез их не задевает).

## Урок о порядке аллокаций

Пиковая RAM загрузки - это максимум пересекающихся во времени аллокаций,
а не сумма данных. Чтение сразу в конечное размещение (affine remap) убирает
пик целиком, а не сглаживает его; scratch-буферы, живущие до конца фазы,
удваивают residency. Измеряйте сумму группы процессов: реальный владелец
памяти может быть не тем процессом, чей PID вы запускали.

См. также: [ftw-load-path.md](ftw-load-path.md) (конвертация и быстрый путь
загрузки, silent-None контракт), [../cases/round5-deferred-ports/TASK.md](../cases/round5-deferred-ports/TASK.md).
