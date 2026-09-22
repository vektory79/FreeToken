---
title: Профили MMQ-префилла (mmq-prefill-kernel)
date: 2026-09-17
hardware: RTX 5090 (одна карта), GGUF GLM-5.3-Flash-UD-Q3_K_XL
status: validated (дистилляция бинарных профилей, оригиналы удалены)
---

# Профили MMQ-префилла: step0, v2probe2, report2/v2probe, report1/report3

Методика снятия и разбора - [../methods/step0-profile-mmq.md](../methods/step0-profile-mmq.md),
v2 A/B - [../methods/v2-ab.md](../methods/v2-ab.md). Захваты делались на
winner-флагах кампании (до v2, т.е. MoE - построчное ядро `moe_vec_q`, dense -
`mul_mat_q8_0`); в отличие от dense-семейства, здесь MoE-терм меняется между
профилями: `step0` - базлайн `moe_vec`, `v2probe2` - grouped MMQ tile 4.

## Вердикты по семейству

- `step0` (экспорт захвата report1): базлайн `moe_vec_q` - 18.13 с/чанк (63.0%
  чанка 28.77 с), эффективная полоса 1638 ГБ/с = ~92% roofline VRAM - MoE GEMM
  полностью BW-зависим, 226x пере-чтение весов экспертов. Разбивка чанка и
  потолки - в [../methods/step0-profile-mmq.md](../methods/step0-profile-mmq.md).
- `v2probe2` (grouped MMQ, tile 4): те же 8128-чанки, MoE-ядра теперь
  `moe_iq3_xxs/moe_iq4_xs/moe_q6_K`; суммарный grouped-терм окна 113.72 с против
  45.22 с у tile-16 пробы v3a (см. [mmq-v3-profiles.md](mmq-v3-profiles.md)) -
  базлайна для cross-check тайлов из [../methods/v3a-ab.md](../methods/v3a-ab.md).
- `report2` и `v2probe` - битые захваты: записи ядер отсутствуют (kernels
  collapsed в CUDA graphs), выжимка по ядрам невозможна из обоих файлов.
- `report3` - сырой профиль без парного sqlite: выжимка невозможна без
  `nsys export` (nsys CLI недоступен при работе с базой); его числовые выводы
  (liveness v2-пробы) уже дистиллированы в
  [../baselines/mmq-prefill-kernel/v2ab_probe_liveness.json](../baselines/mmq-prefill-kernel/v2ab_probe_liveness.json).

## Профиль step0 (базлайн moe_vec; экспорт захвата report1)

Топ-5 ядер окна (55,514 записей, окно 113.89 с):

| ядро | запусков | total с | mean мс |
|---|---:|---:|---:|
| moe_vec_q | 501 | 71.75 | 143.21 |
| mul_mat_q8_0 | 1236 | 23.91 | 19.35 |
| fast_index_copy_multi | 168 | 12.46 | 74.15 |
| _glm_dsa_sparse_kernel | 44 | 2.16 | 49.01 |
| elementwise_kernel | 9391 | 0.83 | 0.09 |

Интерпретация: 501 запуск / 3.958 чанк-эквивалента = 126.6 запуска `moe_vec_q`
на чанк (42 слоя x 3 проекции) - liveness-кроссчек методики; per-чанк разбивка
(18.13 / 3.15 / 6.04 / ~1.45 с) зафиксирована в
[../methods/step0-profile-mmq.md](../methods/step0-profile-mmq.md) и лежит в
[../baselines/mmq-prefill-kernel/step0_split.json](../baselines/mmq-prefill-kernel/step0_split.json).

## Профиль v2probe2 (grouped MMQ, tile 4)

Топ-5 ядер окна (386,783 записи, окно 227.52 с; захват длиннее - включает и
decode-графы, отсюда много мелких запусков):

| ядро | запусков | total с | mean мс |
|---|---:|---:|---:|
| moe_iq3_xxs | 820 | 68.04 | 82.97 |
| mul_mat_q8_0 | 3150 | 49.74 | 15.79 |
| moe_iq4_xs | 410 | 44.06 | 107.46 |
| fast_index_copy_multi | 3192 | 33.16 | 10.39 |
| _glm_dsa_sparse_kernel | 110 | 4.25 | 38.65 |

Интерпретация: grouped-терм окна 113.72 с (iq3 68.04 + iq4 44.06 + q6_K 1.62 с);
`moe_vec_q` остался только в decode-графах (8,316 запусков, 0.153 с суммарно) -
префилл-ассерт liveness «zero moe_vec в префилле» подтверждён данными.

## Профили report2 и v2probe (битые захваты)

Оба файла - экспорты одной и той же неудачной сессии (одинаковые счётчики
записей и метки времени, содержимое graph-трейса совпадает дословно): таблица
`CUPTI_ACTIVITY_KIND_KERNEL` отсутствует, ядра свёрнуты в CUDA graphs.
Измеримое из них: graph-level трейс - 66 запусков графов, суммарно 4.73 с
GPU-времени внутри графов при окне 12.01 с. Per-kernel выжимка невозможна;
причина и рабочий рецепт захватов - в разделе «Deviations / caveats»
[../methods/v2-ab.md](../methods/v2-ab.md) (попытка 1: mid-run `nsys start`
после /ready оставил отчёт без eager-записей ядер).

## Сырьё и файлы без пары

- `report1` - сырой профиль базлайн-захвата; парный sqlite существует
  (это `step0` выше, экспорт делался штатно) - дистилляция полная, потери нет.
- `report3` - сырой профиль, парный sqlite отсутствует, выжимка невозможна
  без nsys export - файл удалён вместе с сырым каталогом; числовые выводы
  захвата заранее дистиллированы в
  [../baselines/mmq-prefill-kernel/v2ab_probe_liveness.json](../baselines/mmq-prefill-kernel/v2ab_probe_liveness.json)
  и тексте [../methods/v2-ab.md](../methods/v2-ab.md).
- `report2` (в паре с одноимённым sqlite), `v2probe` (sqlite без пары), `v2probe2`
  (sqlite без пары), `step0` (sqlite без пары) - удалены после дистилляции;
  для report2/v2probe потеря нулевая (полезных записей ядер в них нет),
  для остальных полное содержимое таблиц ядер не сохранено (решение
  пользователя, потеря принята; оригиналы были в рабочих каталогах кампаний
  на рабочей машине, вне репозитория). Машиночитаемые результаты кампании -
  [../baselines/mmq-prefill-kernel/](../baselines/mmq-prefill-kernel/README.md).