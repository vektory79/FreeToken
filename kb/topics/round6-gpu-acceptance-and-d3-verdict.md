---
title: "Round 6 GPU-приёмка порт-долга: D2/D4/D5 PASS на железе, вердикт D3 - ре-порт hybrid-механики NO-GO"
date: 2026-10-26
hardware: "RTX 5090 32GB (cloud-окно); волны сериализованы - один ft serve за раз, гейт готовности - реальный chat-запрос"
branch-commits: "vektory79: 2e82d81 (projector alias) + 82956fb (verify raw-index fallback) поверх 398487d; база кейса - d66f6e3 (round-6 merge, parents 23c1f02 + 32056cf)"
status: "closed - все четыре GPU-волны PASS; D3: NO-GO на ре-порт 393b682 как средства закрытия decode-гэпа (-15..-18% к эталону 27.71); гэп и Fix-3 переоткрыты отдельными follow-ups"
tags: [round6, deferred-ports, gpu-acceptance, mmproj, vision, ftw-carrier, gguf-loader, nvfp4, kv-quant, benchbw, d3, ab-test, decode, no-go]
---

# Round 6 GPU-приёмка порт-долга: D2/D4/D5 PASS, вердикт D3 - ре-порт NO-GO

**Вердикт (читайте это первым).** Все четыре GPU-волны кейса
[round 6 порт-долг](../cases/round6-deferred-ports/TASK.md) прошли в
cloud-окне 2026-10-25/26 (дерево vektory79 82956fb, сериализованно):

- **D2 - зрение**: verify 347/347 exact против legacy NVFP4-FTW; boot raw
  GGUF с `--mmproj` READY 180 с, image e2e 23.95 tok/s; FTW-vision carrier
  (347 visual.* + 15 clip.*) конвертируется за 266 с и бутится БЕЗ
  `--mmproj` READY 71 с, image e2e 24.86 tok/s.
- **D4 - контракт gguf-лоадера**: кросс-валидация carrier против raw GGUF -
  38/38 round-5 пар без перестановок, 23/23 ggml-payload пары
  байт-идентичны, плотные corr 1.000000; boot raw GGUF READY 138 с,
  гварды молчат.
- **D5 - nvfp4-KV**: GPU-батареи 170 passed / 0 failed / 0 skipped; бут
  786k READY 109 с (KV 2.87 GiB), decode 20.25 tok/s при fetch-капе 1.
- **D3 - честный A/B**: ре-порт hybrid-механики ветки (393b682) как
  средства закрытия decode-гэпа - **NO-GO**: картография порта пуста
  (слот-пулы и fail-fast гейты уже на main-примитивах, benchbw-gguf -
  преимущество round 6), а гэп -15..-18% к эталону 27.71 tok/s - свойство
  стека round 6, не порт-долг.

Побочный исход волны D3: benchbw-профиль устаревшей версии отвергается
движком молча (один warning в бут-логе), и весь стек до того жил с
fetch-капом 1 - включая базлайн 22.21 fp8. Свежий FULL-профиль обязателен
после каждого мерджа и стоит +2-3 tok/s (+10..+16%) дефолтному гибриду.

## Волна D2 GPU: зрение на raw GGUF и FTW-carrier

Приёмки кейса (smoke #810 и verify 347/347) плюс остаток фазы (бут carrier
без `--mmproj`), все - на реальных артефактах unsloth GLM-5.3-Flash-GGUF
(mmproj-BF16.gguf, UD-Q3_K_XL).

**verify 347/347 exact.** Скрипт scripts/verify_mmproj_tensors.py сверил
все 347 тензора визуальной башни mmproj против legacy-носителя
RedHatAI/GLM-5.3-Flash-NVFP4-FTW: max_abs_diff = 0 у всех, unmatched 0,
shape mismatches 0, персентили p50-p99-max нулевые - вердикт round-5
воспроизведён. Первый прогон упал ДО сравнения: FTWReader отказывает
носителям с fp16 NVFP4 global scales ("reconvert it") - требование fp32
пришло с round-6 merge (upstream 0db5023, #739), а носитель на диске
конвертирован раньше; HF-источника для реконверсии на диске нет. Скрипт
получил raw-index fallback: при ровно этой ошибке визуальные записи
читаются сырыми чтениями индекса и шардов - fp16-гейт касается NVFP4-шкал
экспертов, байты башни не тронуты. Коммит 82956fb "fix(scripts): read
legacy fp16-scale ftw references via raw index in mmproj verify".

**Projector-гейт против реального артефакта.** Первый бут упал на разборе
аргументов: ValueError "... is a 'glm5next' tower; GLM-5.3-Flash needs
'glm5v'" - main-гейт пиннит _PROJECTOR = "glm5v", а реальный unsloth файл
несёт clip.projector_type = "glm5next" (орфография llama.cpp,
PROJECTOR_TYPE_GLM5NEXT). Фикс - алиасы: _PROJECTOR_ALIASES =
("glm5v", "glm5next"); glm5next-башня тензорно-точна HF-референсу
(доказано verify 347/347), поэтому обе орфографии пускаются, всё прочее
fail-fast. Коммит 2e82d81 "fix(models): accept the glm5next projector
alias in gguf mm config" с регресс-тестом
test_mmproj_glm5next_projector_alias_gates_in (failing-before /
passing-after через git stash).

| прогон | READY (реальный chat) | image e2e | decode |
|---|---|---|---|
| raw GGUF + `--mmproj` | 180 с | TTFT 4.44 с | 23.95 tok/s |
| FTW-vision carrier, без `--mmproj` | 71 с | TTFT 3.20 с | 24.86 tok/s |
| round-5 референс (raw + `--mmproj`) | 167 с | TTFT 4.70 с | 19.62 tok/s |

Оба прогона дали осмысленный ответ ("FREE TOKEN 42", красный фон) -
семейное совпадение с round-5; decode внутри послеслияной полосы
(22.21 fp8 матричный медиан / 27.71 nvfp4 round-5 референс). Буты
медленнее старого стека (180 против 167 с; carrier 71 против 47 с) -
свойство стека round 6. Carrier собран convert_checkpoint-ом с kwarg
mmproj_path (266 с): 347 visual.* тензоров (1.08 GiB) + 15 clip.* KVs
через существующую collision-гвардию write_metadata_gguf, коллизий 0;
бут идёт из метаданных носителя - mmproj_path=None в ServerArgs, ни
одного упоминания `--mmproj` в логе. Follow-up: у CLI
`ft checkpoint convert` флага `--mmproj` пока нет - паковка доступна
только через python API.

## Волна D4 GPU: кросс-валидация весов и boot-smoke

**Кросс-валидация carrier против raw GGUF** (методика round-5 D4,
адаптированная: носитель конвертирован из ЭТОГО GGUF, поэтому проверяется
точность конверсии gguf->FTW - repack без перестановок, а не допуск двух
независимых квантизаций; round-5 пороги применены как приёмочный пол):

- 41 строка отчёта = 38 round-5 пар; shape mismatches 0, перестановок 0
  (ошибка оси/ниббла/роли схлопнула бы corr в ~0);
- 23/23 ggml-payload пары (Q8_0 dense, Q6_K lm_head, IQ3_XXS/IQ4_XS байты
  банков) БАЙТ-ИДЕНТИЧНЫ - конвертер перепаковывает payload вербатим,
  риск ниббла/оси/роли исключён по построению;
- плотные пары corr 1.000000 / maxabs 0; единственные суб-единичные
  значения 0.999999 (maxabs <= 0.0019) - bf16-repack пары (kv_b_proj
  fold, indexer wk, kpool gate, weights_proj-семейство): один лишний
  bf16-раундинг деquantизованных Q8_0; всё выше round-5 пола 0.9998;
- банки экспертов corr 1.000000 / maxabs 0 на всех 9 срезах: round-5
  бэнд 0.98-0.993 мерил две НЕзависимые квантизации одного источника,
  same-source repack точен по определению.

**Boot-smoke raw GGUF (text-only, offload, kv_quant=none)**: READY 138 с
по реальному chat-запросу (round-5 бэнд 133-145), короткое пополнение
HTTP 200 с корректным usage, SIGTERM штатный, VRAM вернулась в
десктоп-базлайн. Все D4-гварды контракта (2276069 "feat(models): harden
the glm5next gguf loader contract guards": скаляр-фолдинг per-layer
массивов, vocab/coverage/A_log/тайп-пины) молчат на реальном
сервинг-файле; traceback в логе 0. Пустой content при probe без
enable_thinking=false - известный think-block-квирк GLM, виден на всех
путях round-5 тоже, не регрессия.

## Волна D5 GPU: kernel-батареи и бут nvfp4-KV 786k

**Батареи**: `uv run --extra dev python -m pytest tests/kernels/
test_kv_nvfp4.py tests/kernels/test_triton_attention.py
tests/kernels/test_kv_fp8.py -q` - **170 passed / 0 failed / 0 skipped**
(24.8 с). Известные 2x glm_dsa OutOfResources (baseline с 2026-09-12)
живут в tests/models/test_glm_dsa.py и в батарею не входят; новых
отказов нет.

**Бут 786k на raw GGUF** (рецепт: --kv-reserve-tokens 786432
--kv-cache-dtype nvfp4 --memory-ratio 0.85 --max-prefill-length 4096
--moe-strategy hybrid --moe-cpu-threads 16): READY 109 с, авто-план
1181 слот-эквивалентов / 12299 страниц, слот-гейт молчит (фолбэк на
memory-ratio 0.89 не понадобился), квант-отпечаток "Allocating 787136
tokens for KV cache, K + V = 2.87 GiB" (fp8 при том же числе страниц
дал бы ~4.8 GiB), пик занятой VRAM 29949 MiB, shutdown штатный.
Реальный chat HTTP 200; decode-медиана 20.25 tok/s (клиент, три
32-токенные стриминговые пробы) - при fetch-капе 1, сравнимо с 22.21
fp8 (тоже кап-1); честный A/B - фаза D3 ниже. Легаси-носитель
NVFP4-FTW нечитаем на текущем дереве (тот же fp16-gate, что в D2) -
бут шёл на raw GGUF, реконверсия не импровизировалась.

## Фаза D3: benchbw-ловушка, честный A/B и вердикт

### Ловушка: профиль не читается молча

Профиль benchbw на диске имел версию 5 (запись 2026-10-24, старая схема
без layer_kinds), движок round 6 читает версию 6 (bench_profile.py:
PROFILE_VERSION = 6): _usable_profile отбрасывает файл с одним warning
("has version 5, this engine reads 6; ignoring it") - и каждый гибридный
бут дерева round 6 жил с фиксированным fetch-капом 1: и базлайн
22.21 fp8, и замер D5 20.25, и вся полоса матрицы 14.3-21.2. Это то же
семейство ловушек, что однодатипный перезаписывающий прогон benchbw
(clobber); признак - строка "using a fixed fetch cap of 1" в бут-логе.

### Свежий профиль

Полный перепрогон `ft bench bw --dtype all --model <raw GGUF>
--cpu-threads 16` (~40 с) записал профиль v6: 8 строк layer_kinds при
16 потоках (5 канонических форматов + 3 gguf-вида слоёв чекпоинта).
Fetch-фракции gguf-видов: 29.9% (39 слоёв IQ3_XXS/IQ4_XS), 26.1%
(1 слой IQ4_XS/Q6_K), 30.2% (2 слоя IQ3_XXS/Q6_K); потолки CPU STREAM
71.7 GB/s, PCIe H2D 45.6 / D2H 57.2 GB/s.

### 2x2: фракции x overlap (все - nvfp4-KV 786k, клиент-медиана)

| прогон | fetch-фракции | overlap | декод (клиент) |
|---|---|---|---|
| D5, до перепрогона benchbw | кап 1 | on | 20.25 tok/s |
| повтор D5-конфига | кап 1 | on | 21.77 tok/s |
| D3-бут | resolved | on | **23.60 tok/s** |
| контроль A | resolved | off | 23.75 tok/s |
| контроль B (matched-пара к 27.71) | кап 1 | off | 23.40 tok/s |

- кап-1 вредит ТОЛЬКО при overlap-on: -12..-16% (20.25-21.77 против
  23.4-23.75 в остальных ячейках); при overlap-off фракции неотличимы
  от капа (23.40 против 23.75 - шум);
- флаг --disable-moe-prefill-overlap на декод нейтрален (+0.6% - шум);
- межбутной шум конфига ~1.5 tok/s (~7%).

### Сравнение с эталонами

- **Против 22.21 fp8 (round 6)**: +6.3% клиент (23.60 против 22.21), но
  22.21 тоже был кап-1 - дельта nvfp4-против-fp8 при одинаковых фракциях
  остаётся НЕизмеренной (fp8 со свежим профилем - отдельный замер).
- **Против 27.71 (round-5 эталон)**: прямое сравнение было бы XOR-ошибкой
  дважды: по kb ([hybrid-moe-slot-pools-port](hybrid-moe-slot-pools-port.md),
  урок 5) эталон снят при кап-1 (профиля benchbw для gguf в round-5 не
  было вовсе) и при overlap-off, плюс другая методология ("медиана полных
  decode-интервалов", не клиент-медиана). Честная matched-пара - контроль
  B: 23.40 клиент / 22.79 сервер-медиана против 27.71 = **гэп -15..-18%**
  (-4.3 tok/s клиент / -4.9 сервер), устойчиво против шума ~1.5 tok/s;
  методологическая поправка (клиент выше сервера на ~0.6-1.3 tok/s) гэп
  не закрывает. Per-token overhead: ~36.1 мс/токен (round-5) против
  ~42.7-43.9 (round-6, клиентская и серверная оценки matched-пары).

### Картография ре-порта 393b682

| элемент 393b682 | статус на дереве round 6 |
|---|---|
| per-format слот-пулы | уже есть: лог "MoE slot pools: unified, 3 pools of 29-1095 slots (equal)" ([engine.py:943](../../python/freetoken/engine/engine.py)) - структура round-5 |
| fail-fast гейты (format screen / slot-floor) | уже есть: "cannot fund the per-signature slot floors" ([cache_budget.py:87](../../python/freetoken/engine/cache_budget.py)) |
| benchbw-профиль gguf | преимущество round 6: в round-5 профиля для gguf не было (кап-1 принудительно); сейчас `ft bench bw --model <gguf>` снимает все 3 вида слоёв |
| Fix-3 межпулевое донесение | нет (grep 'donat' по python/freetoken/moe пуст); симптом - периодические полные промахи слотов на интерливе сессий; декод-проба одной сессии его принципиально не видит |

### Вердикт

**NO-GO на ре-порт 393b682 как средства закрытия decode-гэпа.** Портить
нечего: слот-пулы и гейты уже сидят на main-примитивах, benchbw-gguf -
обратное преимущество round 6, а единственный отсутствующий элемент
(Fix-3 донесение) на декод одной сессии повлиять не может по механике
(он про вытеснение МЕЖДУ сессиями). Формальный критерий раздела 7 кейса
("main-механика на nvfp4-KV >= эталона") не выполнен - серая зона
зафиксирована честно: бинарный вердикт заменён парой "NO-GO на ре-порт +
новый кейс на гэп". Гэп -15..-18% - свойство стека round 6; подозреваемые
по анатомии lever2 - chain-kernels/handshake плюс GPU-нога _cast
nvfp4-KV, заново поднятая в D5. Fix-3 донесение - отдельный мелкий
кандидат: решать A/B на fix3-харнессе (интерлив-нагрузка), не
декод-пробой.

## Уроки

1. **Профиль benchbw умирает молча.** Смена схемы под тем же номером у
   дерева-писца (v5-старая-схема записана до merge, v6-движок пришёл
   мерджем) проходит version-гейт так же беззвучно, как однодатипный
   clobber. Признак в бут-логе - "using a fixed fetch cap of 1". Правило:
   после КАЖДОГО мерджа - полный `ft bench bw --dtype all --model
   <чекпоинт>` (никогда --dtype-ограниченный) и проверка fetch-строки
   бут-лога.
2. **Свежий профиль - рычаг, а не формальность**: +2-3 tok/s (+10..+16%)
   дефолтному overlap-on гибриду. Матрице и fp8-базлайну желателен
   повтор со свежим профилем до выводов nvfp4-vs-fp8.
3. **A/B эталоны переносятся только с их условиями** (фракции x overlap x
   методология замера). XOR-сравнение resolved-против-кап-1 (23.60 против
   27.71) дало бы ложный вердикт в обе стороны; matched-пара собрана
   контрольными бутами 2x2.
4. **Гейты на имена проверять против реальных артефактов**: main пиннил
   "glm5v", реальный unsloth файл несёт llama.cpp-овское "glm5next"
   (инверсия round-5 урока: тогда фикстуры подгоняли под константу
   main). Регресс-тест должен доказывать failing-before на реальном
   значении.

## Что осталось открыто

1. **Decode-гэп -15..-18%** (per-token ~42.7-43.9 против ~36.1 мс/токен
   при bs=1): отдельный кейс расследования overhead стека round 6.
2. **Fix-3 межпулевое донесение**: отдельный кандидат под fix3-харнесс
   (интерлив); декод-проба одной сессии для него сигнала не даёт.
3. **CLI-флаг `--mmproj`** у `ft checkpoint convert` (паковка vision в
   carrier пока только через python API).
4. **fp8 против nvfp4 при одинаковых фракциях** (fp8 со свежим профилем
   v6) - не измерено.

Смежные статьи: [round6-d2-mmproj-vision-port](round6-d2-mmproj-vision-port.md),
[round6-d4-gguf-loader-contract](round6-d4-gguf-loader-contract.md),
[round6-d5-nvfp4-kv-dequant](round6-d5-nvfp4-kv-dequant.md); полоса чисел -
[gguf-serving-perf-matrix](gguf-serving-perf-matrix.md).
