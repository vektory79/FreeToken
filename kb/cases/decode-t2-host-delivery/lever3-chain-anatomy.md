# Lever 3: анатомия GPU-цепочки декода (kernel-level)

Status: RESEARCH 2026-09-27; код не менялся, коммитов нет. Источник - СУЩЕСТВУЮЩИЙ
экспорт lever-2 пробы (новая nsys-проба НЕ запускалась):
`instr_t2/nsys_gap/nsys_gap.sqlite` (утрачен)
(240 MB, 1.84M ядер), 468 декод-шагов, бёрст по `_ensure_experts_hybrid_kernel`
(42/шаг). Скрипты волны: `analyze_nsys_chain_anatomy.py`
(пер-кernel статистика), `analyze_nsys_chain_sites.py`
(пер-сайт разброс + гэп-банды), `gguf_meta.py`
(чтение GGUF-заголовка для форматов/форм). Начальная точка:
[lever2-gap-anatomy.md](lever2-gap-anatomy.md).

## TL;DR (вердикт)

- **dense_gemv (6.67 ms/шаг) - это ДВЕ разные вещи**: 6.32 ms - 316 пусков
  q8_0 MMVQ `mul_mat_vec_q` (проекции attention/KDA/shexp/ffn + lm_head),
  0.34 ms - 86 мелких bf16 `at::gemv` (router-gate, indexer wk/weights_proj).
- **Крупные MMVQ уже у DRAM-стены**: in_proj 1531 GB/s (85% spec-пика 1792),
  lm_head 1633 (91%), dense-ffn 1500 (84%), q_b 1402 (78%), o_proj 1304-1316
  (73%). Малые (f_b/g_b 252 GB/s, kv_a 501) съедаются launch-латентностью, но
  они мелкие. Агрегат: 8.35 GB (7.8 GiB) весов/шаг за 6.32 ms = 1320 GB/s
  (74% spec). Потолок идеальных ядер: 5.22 ms при 1.6 TB/s -> из dense_gemv
  физически недостижимо больше ~1.1 ms/шаг.
- **Один настоящий провал occupancy нашёлся вне dense**: `_mhc_stage1_kernel`
  (hyper-connections, 90 пусков/шаг x 13.35 us = 1.20 ms/шаг) работает на
  grid=(1,8) x 128 нитей = 8 CTA на 170 SM, ~0.9 MB трафика за 13.35 us
  (67 GB/s). Разделение сплита (NS 8 -> 32-64) - кандидат на ~-0.8-0.9 ms.
- **Launch-оверхед НЕ материален**: 3556 ядер/шаг на главном стриме, 96%
  межkernel-гэпов < 0.5 us (mean 0.21 us - CUDA-graph replay back-to-back);
  весь бюджет мелких гэпов ~0.7-0.8 ms/шаг. Остальные 11.3 ms гэпов - истинные
  ожидания (wait-slack пулов + doze, см. lever2).
- **Суммарный реалистичный резерв цепочки ~1.5-2.0 ms/шаг (+2.6-3.4%)**,
  размазанный по 4-5 независимым мелким пунктам; ни один не даёт +5% в одиночку.
  Абсолютный потолок (все ядра идеальны) ~3.3 ms (+5.7%). Вердикт: "GPU math"
  как единый рычаг НЕ воскресает; это бандл микро-рычагов.

## 1. Состав цепочки по классам (nsys, mean по 468 шагам)

| класс | ms/шаг | главные ядра | пусков/шаг | источник |
|---|---|---|---|---|
| dense_gemv | 6.67 | `mul_mat_vec_q` 6.32 + `at::gemv` (bf16) 0.34 | 316 + 86 | [layers/gguf.py](../../../python/freetoken/layers/gguf.py) `fused_mul_mat_gguf` -> [mmvq.cuh](../../../python/freetoken/kernel/csrc/gguf/mmvq.cuh) |
| gdn/mhc | 1.91 | `_mhc_stage1` 1.20, `_mhc_stage2` 0.25, `_mhc_stage3` 0.07, fla `fused_recurrent_gated_delta_rule` 0.19, RMSNorm 0.14, `causal_conv1d_update` 0.085, layernorm 0.036 | 90+90+90+34+113+34+34 | [kernel/triton/mhc.py](../../../python/freetoken/kernel/triton/mhc.py), fla, flashinfer-norm |
| add/copy | 1.27-1.36 | at:: `vectorized/unrolled/elementwise_kernel` 812+148+269, `scatter_gather` 53 | ~1280 | torch eager: mhc-стримы, резудуалы, add партиалов [moe.py:352](../../../python/freetoken/layers/moe.py) |
| topk/router | 0.76 | `sbtopk::gatherTopK` 42x4.65us=0.195, `bitonicSortKVInPlace` 0.116, DSA-индексер (radix+mbtopk suite) ~0.38 | 42 + ~190 | router top-8 из 288 ([moe.py](../../../python/freetoken/layers/moe.py)), select в [dsa_indexer_kpool.py](../../../python/freetoken/attention/dsa_indexer_kpool.py) |
| quant | 0.32 | `quantize_q8_1` 442x0.71us | 442 = 316 dense + 126 moe | вход C++-опа MMVQ/MoE ([gguf_kernel.cu](../../../python/freetoken/kernel/csrc/gguf)) |
| dsa | 0.30 (+0.33 bmm) | `_glm_dsa_splitk` 0.198, `_decode_logits` 0.054, `_merge` 0.044; отдельно cutlass bf16 `s161616gemm` 22x14.5us=0.33 (MLA-absorb bmm) | 11+11+11 (+22) | [dsa_indexer_kpool.py](../../../python/freetoken/attention/dsa_indexer_kpool.py), [attention.py](../../../python/freetoken/models/glm5_next/attention.py) |
| ensure | 0.16 | `_ensure_experts_hybrid` 42x3.74us | 42 | роутинг-буккипинг MoE |

Pool-window классы (вне цепочки, спрятаны под окна CPU-пула): `fast_index_copy_multi`
28.66 ms/шаг (42 пуска, ~24 GB/s = bench), `moe_vec_q` 2.35 ms/шаг (126 = 42x3
gate/up/down, z=top_k=8; формат экспертов - родные q3_K/q4_K банков). Не цели.

## 2. DENSE_GEMV deep-dive

Ядро: `mul_mat_vec_q` (mmvq.cuh, порт llama.cpp/sgl-kernel). Launch-геометрия:
grid=(nrows,1,1), block=(32,1,1) - ОДИН варп на строку весов, grid.x = число
выходных фичей. Это дало разложение по проекциям без инструментов: каждый сайт
= своя nrows. Регистры 40, nvecs=1 (decode bs=1, порог MMVQ x<=6 в
[layers/gguf.py](../../../python/freetoken/layers/gguf.py)). Активация x[4096]
квантуется в q8_1 перед каждым пуском (класс quant, 442/шаг по 0.71 us - чистый
launch-флор). Вес/инстанс доминирует: активация 4.4 KB, выход 2-8 KB.

Модель: GLM-5.3-Flash-UD-Q3_K_XL (FTW). Геометрия из GGUF-метаданных: hidden
4096, 46 блоков = 45 активных (34 KDA + 11 DSA, DSA = блоки 3,7,...,43) + 1
MTP-draft (blk.45 не исполняется: q_a 11/шаг, а не 12); dense-префикс MLP =
блоки 0-2; MoE = 42 блока; shexp inter 2048, expert inter 2048, 288 экспертов,
top-8; indexer 32x128, top 2048, kpool 4; mhc (hyper-connections) N=4 на каждом
блоке. Все dense-тензоры Q8_0 (1.0625 B/elem), `output.weight` = Q6_K
(0.8203 B/elem - иначе 2116 GB/s у lm_head, что выше физического пика).

Пер-сайтная таблица (p50 us; MB = вес инстанса; GB/s = MB/p50; % от 1792 spec):

| nrows | проекция (формат) | шт/шаг | p50 us | MB | GB/s | %spec |
|---|---|---|---|---|---|---|
| 24896 | KDA in_proj (слитые q\|k\|v\|b\|f_a\|g_a, q8_0) | 34 | 70.8 | 108.38 | 1531 | 85% |
| 154880 | lm_head (Q6_K!) | 1 | 318.6 | 520.3 | 1633 | 91% |
| 12288 | dense-ffn gate+up, блоки 0-2 (q8_0) | 6 | 35.7 | 53.47 | 1500 | 84% |
| 16384 | DSA q_b [16384,1536] | 11 | 19.1 | 26.74 | 1402 | 78% |
| 4096a | KDA o_proj [4096,8192] | 34 | 27.1 | 35.66 | 1316 | 73% |
| 4096b | DSA o_proj [4096,16384] | 11 | 54.7 | 71.31 | 1304 | 73% |
| 4096c | shexp_down [4096,2048] | 42 | 7.3 | 8.91 | 1221 | 68% |
| 4096d | dense ffn_down [4096,12288] | 3 | ~43 | 53.47 | ~1244 | 69% |
| 4096e | indexer wq_b [4096,1536] | 11 | ~7.3 | 6.68 | ~915 | 51% |
| 2048 | shexp gate+up [2048,4096] | 84 | 7.30 | 8.91 | 1221 | 68% |
| 1536 | DSA q_a [1536,4096] | 11 | 6.21 | 6.68 | 1075 | 60% |
| 512 | DSA kv_a [512,4096] | 11 | 4.45 | 2.23 | 501 | 28% |
| 8192 | KDA f_b/g_b [8192,128] | 68 | 4.42 | 1.11 | 252 | 14% |

(сайт 4096 - бимодальный: p10 7.3 / p50 27.1 / p90 54.7 ровно по этим
подпопуляциям 42+34 / 11+3 / 11; сумма 90/шаг сходится)

Распределение длительностей `mul_mat_vec_q`: mean 20.0 us, p50 7.4, p90 67.0 -
бимодальность от сайтов, не хвосты. in_proj крайне стабилен (p10 66.8, p90 71.3).

Арифметика границы: суммарный вес dense-трафика 8347 MB/шаг (in_proj 3685 +
o_projs 1997 + shexp 1123 + lm_head 520 + dense-ffn 481 + q_b 294 + прочее 197).
6.32 ms -> 1320 GB/s агрегатно. Полы: при spec 1792 -> 4.66 ms (максимальный
теоретический выигрыш 1.66 ms); при реалистичном stream-потолке 1.6 TB/s ->
5.22 ms (реалистичный 1.11 ms). Крупные сайты (85% ступень) уже почти там;
остаток концентрируется в мелких сайтах и launch-флоре.

## 3. Bound-вердикты по классам

| класс | трафик/шаг | факт ms | GB/s факт | % spec | вердикт |
|---|---|---|---|---|---|
| dense_gemv (крупные сайты: in_proj, o_proj, lm_head, ffn, q_b) | ~7.0 GB | ~4.85 | 1300-1633 | 73-91% | DRAM-bound у потолка - НЕТ ГОЛОВЫ |
| dense_gemv (мелкие: shexp, f_b/g_b, kv_a, wq_b, q_a) | ~1.37 GB | ~1.42 | 250-1220 | 14-68% | launch/latency-флор; лечится только СЛИЯНИЕМ пусков, не скоростью ядра |
| at::gemv bf16 мелочь (router gate [288,4096] и indexer wk/weights_proj) | ~0.4 GB | 0.34 | ~600 | 33% | launch-флор; кандидат на слияние/квант |
| mhc stage1 | ~0.08 GB (0.9 MB x 90) | 1.20 | 67 | 4% | occupancy-провал (8 CTA) - ГОЛОВА ЕСТЬ |
| mhc stage2/3 + gdn recurrent + conv + norms | ~0.1 GB | 0.71 | - | - | мелочь, latency-bound, головы нет |
| add/copy | <0.05 GB | 1.27 | - | - | 1280 пусков по 0.5-3 us, одно-CTA; сливаемо в соседей |
| topk/router | <0.05 GB | 0.76 | - | - | top-8 из 288 за 7.4 us (gatherTopK+sort) = чистый оверхед |
| quant | ~0.02 GB | 0.32 | - | - | 442 x 0.71 us launch-флор |
| dsa-attention | ~0.02 GB | 0.30 | gather-bound | - | мелочь, не трогать |
| MLA-absorb bmm (cutlass bf16, 22 пуска x 16.8 MB) | ~0.37 GB | 0.33 | 1158 | 65% | вес-доминирован; кастомному GEMV потолок ~0.05-0.08 ms - НЕ рычаг |
| ensure | <0.01 | 0.16 | - | - | нужен, не трогать |

## 4. Launch-структура

Ядер в регионе бёрста: 3566/шаг (стрим 141 главный: 3557; стримы 13/17 - по
нескольку мемопов пула). Межkernel-гэпы главного стрима:

| банд | шт/шаг | сумма ms/шаг | mean |
|---|---|---|---|
| 0-0.5 us | 3409 | 0.70 | 0.21 us |
| 0.5-20 us | 115 | 0.65 | ~6 us |
| 20-100 us | 5.9 | 0.36 | 61 us |
| 0.1-1 ms | 23.8 | 6.49 | 272 us |
| >1 ms | 2.4 | 4.79 | 2.0 ms |

96% гэпов - нулевые (граф раздаёт ядра back-to-back). Большие банды = ожидания
флага пула (wait-slack ~10-12 ms по lever2) + doze 1.31 ms - всё уже
атрибутировано lever2; нового launch-бюджета нет: сливательная экономия сверху
0.3-0.5 ms/шаг невозможна, потому что нечего экономить - гэпы и так ~0.

## 5. Ранжированные кандидаты (потолки из bound-математики)

1. **C-L1: mhc stage1 occupancy (NS-split)**. Что: поднять число сплитов
   stage1 (grid (1,8) -> (1,32/64)) и/или слить stage2. Файлы:
   [kernel/triton/mhc.py](../../../python/freetoken/kernel/triton/mhc.py),
   [layers/mhc.py](../../../python/freetoken/layers/mhc.py). Потолок:
   13.35 -> 3-5 us x 90 = **-0.75..-0.93 ms/шаг (+1.3-1.6%)**. Риск: низкий -
   порядок fp32-редукции меняется (parity-тест tests/layers/test_mhc.py),
   пересборка графа на буте. Ручка: FREETOKEN_MHC_STAGE1_NS (одна переменная,
   дефолт = текущее 8).
2. **C-L2: мелкие bf16 GEMV -> слияние/квант**. Что: router gate [288,4096]
   bf16 (42 x ~4 us) и indexer wk/weights_proj/gate (33 пуска) - это 86 пусков
   at::gemv на 0.34 ms; слить пер-слойную группу в один пуск или перевести в
   q8_0 MMVQ-путь. Файлы: роутер-сайт [moe.py](../../../python/freetoken/layers/moe.py),
   [attention.py](../../../python/freetoken/models/glm5_next/attention.py),
   weight.py. Потолок: **-0.10..-0.15 ms/шаг (+0.2-0.25%)**. Риск: низкий
   (незначимые тензоры; у router-gate контроль качества маршрутизации).
   ПРИМЕЧАНИЕ: MLA-absorb bmm (0.33 ms) изначально казался провалом, но 22
   пуска читают по 16.8 MB каждый -> 1158 GB/s (65% spec); кастомному ядру
   остаётся ~0.05-0.08 ms - в кандидаты НЕ берётся.
3. **C-L3: topk/router слияние**. Что: router top-8 из 288 (gatherTopK 4.65 +
   bitonicSort 2.77 us на слой) -> одно CTA-ядро sigmoid+top8; DSA-индексер
   (radix-sort 16K -> select 512) -> select-ядро. Файлы: роутер-сайт
   [moe.py](../../../python/freetoken/layers/moe.py),
   [dsa_indexer_kpool.py](../../../python/freetoken/attention/dsa_indexer_kpool.py).
   Потолок: **-0.30..-0.40 ms/шаг (+0.5-0.7%)**. Риск: средний - семантика
   ничьих/порядка выбора должна совпасть бит-в-бит (маршрутизация!).
4. **C-L4: quantize_q8_1 внутрь MMVQ/MoE-ядра**. Что: считать y-квант на лету
   в ядре (x[4096] = 128 блоков - дешево в L1), убрав 442 отдельных пуска.
   Файлы: [kernel/csrc/gguf](../../../python/freetoken/kernel/csrc/gguf)
   (gguf_kernel.cu + mmvq.cuh). Потолок: **-0.20..-0.30 ms/шаг (+0.35-0.5%)**.
   Риск: низкий по числам (те же q8_1-значения -> бит-идентичный результат),
   средний по C++/CUDA работе.
5. **C-L5: слитие f_b/g_b при загрузке (row-concat)**. Что: f_b_proj+g_b_proj
   KDA [8192,128]x2 -> один qweight [16384,128] (тот же приём, что in_proj в
   weight.py), выход разрезается. 68 пусков -> 34. Файлы:
   [models/glm5_next/weight.py](../../../python/freetoken/models/glm5_next/weight.py)
   + [kda.py](../../../python/freetoken/models/glm5_next/kda.py). Потолок:
   **-0.15..-0.20 ms/шаг (+0.25-0.35%)**. Риск: почти нулевой (математика
   идентична; прецедент in_proj). Туда же опционально q_a+kv_a (23 пуска -> 12).
6. **C-L6 (не рекомендую первым): консолидация add/copy**. ~1280 мелких
   elementwise (mhc-стримы, резудуалы). Теоретически -0.3..-0.5 ms, но это
   реструктуризация forward под графом - высокий effort/риск при малом выигрыше.

### No-op вердикты (за битыми ядрами не ходить)

- Крупные MMVQ (in_proj, o_proj, lm_head, dense-ffn, q_b, shexp): 73-91% spec -
  лучше не станет; выигрывает только перенос формата (уже опровергнуто
  dense-q80 кампанией) - НЕ РЫЧАГ.
- `fast_index_copy` (28.66 ms) - у bench-потолка PCIe и спрятан под пулом.
- `moe_vec_q` (2.35 ms) - под окнами пула, вне критического пути.
- dsa splitk/logits/merge (0.30 ms), gdn recurrent/conv (0.31 ms), ensure
  (0.16 ms) - latency-мелочь, потолок <=0.1 ms суммарно.
- MLA-absorb bmm: 1158 GB/s (65% spec), вес-доминирован - потолок кастома
  ~0.05-0.08 ms, НЕ РЫЧАГ.

## 6. Рекомендация

Цепочка НЕ содержит одиночного рычага +5%: реалистичный бандл C-L1..C-L5 =
**+2.6-3.4%** (1.5-2.0 ms/шаг), абсолютный потолок цепочки +5.7% при
идеальных всех ядрах. Если после C1 (doze-fix, +1.6-1.8%) нужен следующий
шаг - брать **C-L1** (самый большой одиночный, чистая env-ручка, дешёвый
дискриминатор: duration `_mhc_stage1_kernel` в трейсе 13.35 -> <=6 us ещё до
e2e), затем при желании докатывать C-L3+C-L4+C-L5 как микрокоммиты. Дизайн
A/B у каждого: одна env-переменная, серийные руки run_arm.py, контроль в одном
окне; дискриминаторы - per-kernel duration в nsys/torch-trace до e2e.

## Артефакты

- Экспорт: instr_t2/nsys_gap/nsys_gap.sqlite (240 MB) + nsys_gap.nsys-rep
  (77.5 MB), проба lever-2 (probe_nsys_gap.py, 468 шагов, 17.76 tok/s boot).
- Скрипты волны: analyze_nsys_chain_anatomy.py, analyze_nsys_chain_sites.py,
  gguf_meta.py (.tasks/decode-research/).
- Геометрия/форматы: GGUF-метаданные
  /media/ai/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL.gguf
  (1412 тензора; все dense Q8_0, output.weight Q6_K).

## Вердикт C-L1 (2026-09-27)

C-L1 (mhc stage1 NS-split, ручка FREETOKEN_MHC_STAGE1_NS) ПРИНЯТ на железе
2026-09-27: механизм PASS, e2e (сквозной прогон) PASS, батарея качества 24/24.
Код существует только в рабочем дереве - коммита нет, решение о коммите за
пользователем. Ручка: рекомендуемое значение 64; переменная не задана -
бит-в-бит легаси-путь (NS=8); "1" - сплит выключен; 0, отрицательное,
не-целое или >64 - fail-fast на импорте. Этот раздел обновляет статус
RESEARCH из шапки файла.

### Механизм (nsys-проба, NS=64)

Рецепт: --delay=60 без --duration, коллекция верифицирована, 510 шагов x 42
ядра/шаг. `_mhc_stage1_kernel`: grid (1,8) x 128 нитей -> (1,64) x 128
(8 -> 64 CTA); duration p50 13.35 -> 4.26 us (-68%; mean 4.34, p90 4.70;
первый пуск - warmup-выброс 1223 us, задокументирован). На шаг: stage1
1.20 -> 0.391 ms, stage2 0.25 -> 0.275 (редукция теперь по 64 партиалам),
stage3 0.07 -> 0.070; суммарно mhc-класс 1.52 -> 0.736 ms/шаг (-0.78 ms,
внутри спрогнозированных -0.75..-0.93). NS=32 в nsys не снимался: NS=64 уже
достиг целевого потолка.

### e2e A/B

Серийные руки в одном окне, run_arm.py, контроль = текущие дефолты:

| рука | tok/s | p50 ms | к контролю |
|---|---|---|---|
| ctl_r2 (контроль) | 17.26 | 56.4 | - |
| mhc64 | 18.28 | 53.4 | +5.9% |
| mhc32 | 17.83 | 55.1 | +3.3% |
| mhc64_r2 | 18.05 | 54.3 | +4.6% |

Референс flip_check 17.36 tok/s; контроль в -0.6% от него. Честная оговорка:
e2e-выигрыш p50 (-2.1..-3.0 ms) больше nsys-дельты механизма (-0.78 ms);
направление совпадает в обоих ботах NS=64, величина не объяснена - возможен
вторичный эффект occupancy/цепочки или дисперсия step-time; оба бота NS=64
бьют контроль, снятый в тот же час.

### Качество и prefill

run_battery.py env1 = 24/24 PASS при экспортированной
FREETOKEN_MHC_STAGE1_NS=64 (проброс в батарею через наследование os.environ,
[run_battery.py:94-98](../../harness/quality/run_battery.py)); пост-цензус
чистый. Prefill: хвостовые 49-токенные строки и fill wall (4.26-4.36 s) в
шуме; knob-руки не медленнее контроля.

### Файлы и артефакты (не закоммичены)

- Код: [kernel/triton/mhc.py](../../../python/freetoken/kernel/triton/mhc.py)
  (+44/-12); тесты
  [tests/layers/test_mhc.py](../../../tests/layers/test_mhc.py) (+211;
  42 passed, включая bitwise legacy-parity и knob-parity 4/32/33).
- e2e-руки (в git не попадают, живут в рабочем дереве):
  .tasks/decode-research/arm_ctl_r2.{json,log}, arm_mhc32.*,
  arm_mhc64.*, arm_mhc64_r2.*, cpu_*.csv.
- nsys-проба NS=64: .tasks/decode-research/instr_t2/nsys_gap/ -
  nsys_gap.nsys-rep, nsys_mhc64.sqlite, analyze_mhc64.txt,
  probe_summary.json; артефакты lever-2 пробы забэкаплены рядом как
  nsys_gap_l2.*.

### Что осталось открытым из бандла цепочки

C-L2 (мелкие bf16 GEMV: слияние/квант, +0.2-0.25%), C-L3 (слитый router
topk, +0.5-0.7%, средний риск - семантика ничьих), C-L4 (quantize_q8_1
внутрь MMVQ, +0.35-0.5%), C-L5 (f_b/g_b row-concat, +0.25-0.35%) - каждый
берётся отдельным микрокоммитом, если решено продолжать.
