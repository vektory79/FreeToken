---
title: "Порт D5 round 6: nvfp4-KV - премиса ампутации опровергнута, GPU-деquant-ветка _cast возвращена - CPU-часть готова"
date: 2026-10-25
hardware: "RTX 5090 32GB (GPU-приёмка - в cloud-окне); CPU - контрольные батареи pytest"
branch-commits: "vektory79: ee961f0 + 1eb556d (HEAD) поверх d66f6e3 (round-6 merge, parents 23c1f02 + 32056cf); референс-спека - веточный 660c814"
status: "cpu-done - бут-смоук raw GGUF с device=cuda, GPU-батареи tests/kernels и бут nvfp4-KV 786k ждут cloud-окно; качественный A/B - фаза D3"
tags: [nvfp4, kv-quant, gguf-loader, dequant, quant-config, stale-pin, round6, deferred-ports, premise-check, no-dual-logic]
---

# Порт D5 round 6: nvfp4-KV - премиса ампутации опровергнута, GPU-деquant-ветка _cast возвращена - CPU-часть готова

**Вердикт (читайте это первым).** Фаза D5 кампании [round 6 порт-долг](../cases/round6-deferred-ports/TASK.md)
закрыта в CPU-части 2026-10-25 двумя коммитами: ee961f0 ("feat(models):
re-land the glm5next gguf loader gpu dequant branch";
[models/glm5_next/gguf.py](../../python/freetoken/models/glm5_next/gguf.py),
+50/-21) и 1eb556d ("test(quant): recalibrate nvfp4 quant config pins for
w4a4"; un-skip маркера 1433 в
[tests/models/test_glm5_next_gguf.py](../../tests/models/test_glm5_next_gguf.py)
+ перекалибровка пинов в
[tests/models/test_quant_config.py](../../tests/models/test_quant_config.py)).
Главный итог - не код, а премиса: **read-path ампутирован НЕ был**. На
upstream main 32056cf KV-quant механики нет вовсе (ни
kernel/triton/kv_nvfp4.py, ни kv_quant.py, ни supports_nvfp4_kv), веточный
стек (660c814 + round-5 store-стек) пережил merge целым, dsa-аргументы
восстановлены amend-ом d66f6e3 ещё в волне слияния -
[tests/engine/test_kv_quant_config.py](../../tests/engine/test_kv_quant_config.py)
36 passed. Единственная ампутированная деталь - GPU-деquant-ветка _cast
glm5next gguf-лоадера с осиротевшим _cuda_target; она и возвращена. Фаза
сжалась с полной ре-имплементации read-path до ~3 стыков. Прецедент round-5 -
[round5-d5-nvfp4-kv-generic.md](round5-d5-nvfp4-kv-generic.md); смежные фазы -
[round6-d2-mmproj-vision-port.md](round6-d2-mmproj-vision-port.md) и
[round6-d4-gguf-loader-contract.md](round6-d4-gguf-loader-contract.md).

## Что произошло (премиса кейса опровергнута)

Раздел 6 кейса порт-долга исходил из того, что read-path nvfp4-KV ампутирован
merge-ом: "read-path аргументы на glm_dsa_sparse_attn / dsa.py ампутированы
(kv-механика и e4m3-хелперы живы, wiring - нет)". Изучение дерева опровергло
это утверждение по всем трём составляющим:

- upstream main 32056cf не содержит KV-quant механики вообще: ни
  kernel/triton/kv_nvfp4.py, ни kv_quant.py, ни флага supports_nvfp4_kv -
  re-платформа не "жива с мёртвым wiring", её просто нет;
- веточный стек пережил merge целым: 660c814 (generic read-path) + round-5
  store-стек остались на дереве;
- dsa-аргументы не требовали восстановления в D5: квант-прелюд и аргументы в
  dsa.py::_attend возвращены amend-ом d66f6e3 ещё в волне слияния (после
  NameError has_nvfp4 на boot-smoke, см.
  [кейс round 6 merge](../cases/merge-round6-upstream-main/TASK.md));
  has_nvfp4 жив на
  [kernel/triton/glm_dsa_sparse.py:444](../../python/freetoken/kernel/triton/glm_dsa_sparse.py).

Реальная ампутация оказалась точечной: в main-версии
[glm5_next/gguf.py](../../python/freetoken/models/glm5_next/gguf.py) выпала
GPU-ветка каста _cast, а её осиротевший _cuda_target всегда возвращал None -
при device=cuda деquant молча уходил на CPU (тихая деградация бута, не
ошибка). На этой детали висел единственный skip-маркер D-PORT-D5 (тест 1433).
supports_nvfp4_kv точечно уже стоит на main-бэкендах triton / dsa /
qsa_sparse ([attention/__init__.py](../../python/freetoken/attention/__init__.py))
и закреплён существующими тестами; fa / fi / trtllm остаются False - это
external-kernel PR-ы, не D5.

## Что сделано: _cast на main-примитивах (ee961f0)

Ветка _cast ре-ленднута не verbatim, а на main-примитивах: матрица из
[models/gguf/tensors.py](../../python/freetoken/models/gguf/tensors.py),
ядро [kernel/ggml.dequantize](../../python/freetoken/kernel/ggml.py) (CUDA
контракт op_dequantize в
[kernel/csrc/ggml/freetoken/ggml_ops.cu:406](../../python/freetoken/kernel/csrc/ggml/freetoken/ggml_ops.cu)),
CPU-нога - main-овский _float:

- GPU-нога: packed-строки едут на CUDA как packed-байты, деquant - на
  устройстве; CPU-нога: _float. Ноги не смешиваются в пределах одного
  тензора.
- _cuda_target восстановлен как честный device-гейт (старый осиротевший
  всегда отдавал None = молчаливый CPU-деquant под device=cuda).
- device протянут через ВСЕ packed-row cast-места: bf16-хелпер, hc_* (6
  штук), conv1d, ssm_a, dt_bias, _kv_b device-kwargs, indexer wk /
  weights_proj, compressor ape, e_score_correction_bias.
- Осознанная дельта против ветки: squeeze ne при len(ne) > 2 сохраняет ggml
  2-D форму матрицы; docstring документирует асимметрию ног (n,) против (1,n)
  на дегенеративных [n, 1] тензорах.

Ревью сверялось против референса 660c814 (~строки 717-756) и CUDA-контракта
op_dequantize.

## Калибровка пинов quant-конфига (1eb556d)

- test_hybrid_explicit_kernel_error_names_the_usable_alternative теперь пинит
  "(usable here: w4a4, triton)". Пин host-независим: _usable_here зондирует
  только unusable_reason, без CUDA.
- test_nvfp4_auto_selection_still_lands_on_triton переписан и переименован в
  test_nvfp4_auto_selection_lands_on_w4a4_where_available: auto выбирает
  w4a4, когда sm_120-проба проходит, и triton, когда нет; явный запрос
  triton исполняется как задан; пробы monkeypatch-ены (тест остался
  CPU-чистым).
- Маркер 1433 un-skip-нут (D-PORT-D5 1 -> 0), тело теста не тронуто.
- Вне скоупа D5 (помечено отдельным follow-up, не атрибутировать фазе): 2
  других stale-pin теста test_quant_config.py (~строки 410/422) - тритонов
  fp8-block-кернел обзавёлся cpu_format='fp8_block', пины отстали.

## Батареи и счётчики

- Консолидированный прогон: tests/checkpoint + tests/engine/test_cache_budget
  + tests/models/test_glm5_next_gguf + tests/checkpoint/test_ftw_gguf_banks =
  243 passed / 12 skipped / 0 failed.
- Сюита четырёх файлов (включая test_quant_config + test_kv_quant_config):
  194 passed / 46 skipped / 0 failed; test_kv_quant_config отдельно 36 passed.

## Уроки

1. **Премису ампутации проверять против дерева до планирования.** Премиса
   кейса, записанная по памяти merge-волны ("ампутирован read-path"),
   дожила до TASK.md непроверенной; учебная волна (git ls-tree / git show
   против upstream main + grep) стоила одну волну, но сжала фазу с полной
   ре-имплементации read-path до ~3 стыков. Правило: заявка "merge отрезал
   X" проверяется ls-tree/show против апстрима, не по памяти волны.
2. **Осиротевший гейт = тихая деградация.** Упавшая ветка оставила живой
   хелпер _cuda_target с всегда-None ответом: CPU-деquant под device=cuda не
   ошибается, а замедляет бут. Ампутации надо искать не по ошибкам, а по
   always-dead веткам.
3. **Host-независимость пинов.** Перекалиброванный pin авто-выбора сделан
   от monkeypatch-проб (sm_120), а не от реального железа прогоняющего -
   тест остаётся зелёным и на машине без GPU.

## Осталось (cloud-окно)

- Бут-смоук raw GGUF glm5next с device=cuda, гоняющий GPU-ногу _cast
  (CPU-часть сюит её не трогает).
- GPU-батареи tests/kernels: test_kv_nvfp4, test_triton_attention,
  test_kv_fp8.
- Бут nvfp4-KV 786k-рецепта (эталон 27.71 tok/s) и качественный A/B - это
  фаза [D3](../cases/round6-deferred-ports/TASK.md), отдельно после D5.

Статус, шаги фазы и исправленный раздел 6 - в
[кейсе round 6 порт-долг](../cases/round6-deferred-ports/TASK.md). Смежные
фазы раунда: D2 - [round6-d2-mmproj-vision-port.md](round6-d2-mmproj-vision-port.md),
D4 - [round6-d4-gguf-loader-contract.md](round6-d4-gguf-loader-contract.md);
round-5 прецедент - [round5-d5-nvfp4-kv-generic.md](round5-d5-nvfp4-kv-generic.md).