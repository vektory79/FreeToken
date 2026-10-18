---
title: "Порт D5: ре-ленд nvfp4/fp8-KV read-пути generic-triton после ре-платформы"
date: 2026-10-21
hardware: "RTX 5090 32GB (GPU-тесты, boot A/B Qwen3.8); CPU-волна (pytest)"
branch-commits: "vektory79 660c814 поверх merged-дерева f0eed55; backup-эталон backup/vektory79-pre-merge5-b8c3794"
status: validated
tags: [round5, deferred-ports, nvfp4-kv, fp8-kv, triton, attention, kv-quant]
---

# Порт D5: ре-ленд nvfp4/fp8-KV read-пути generic-triton

**Вердикт.** Ре-платформа раунда 5 убила только чтение: store-стек упакованного
NVFP4-KV пережил merge байт-в-байт, а потерянные ядра чтения ре-лендятся НА
текущую структуру (не откатом на старую). Порт D5 (коммит 660c814) вернул
triton'у чтение nvfp4 и fp8 KV-пулов (decode/extend/split/paged); метрика спеки -
[test_kv_quant_config.py](../../tests/engine/test_kv_quant_config.py)
`-p no:skipping` 8 failed -> 36 passed; качественный A/B на Qwen3.8-27B-NVFP4-FTW:
3.46x ужатие KV без деградации (greedy 11/12 byte-identical). Карта кейса и статусы
портов - [round5-ggml-replatform-deferred-ports.md](round5-ggml-replatform-deferred-ports.md).

## Store/read-асимметрия ре-платформы

- Дельта-матрица против backup-ветки: [kernel/triton/kv_nvfp4.py](../../python/freetoken/kernel/triton/kv_nvfp4.py)
  (quantize-примитивы + load_nvfp4), [kernel/triton/kv_quant.py](../../python/freetoken/kernel/triton/kv_quant.py)
  (fp8 store) и [kvcache/mha_pool.py](../../python/freetoken/kvcache/mha_pool.py)
  (uint8-пул, codes + row-scale + block-scale) доехали до merged-дерева без единой
  строки дельты; потерян был ТОЛЬКО read-путь: ядра
  [kernel/triton/attention.py](../../python/freetoken/kernel/triton/attention.py)
  (-521 строка) и scale-пламбинг обёртки
  [attention/triton.py](../../python/freetoken/attention/triton.py). После
  ре-платформы асимметрия store/read - норма; дельту считать файл за файлом, а не
  вердиктом "фича сломана целиком".
- Ре-ленд выполнялся НА переписанные ядра: за время раунда в них появились sinks,
  sliding-window, split-caps и smem-optin, которых в backup не было; scale-пламбинг
  вплетён в новые тела (constexpr KV_NVFP4 / HAS_KV_SCALE, гейт до горячего цикла).
  Приёмка диффа: с None-шкалами арифметика ядер прежняя (kv_quant=None =>
  16-битный путь байт-в-байт).
- Флаги включаются ПАРОЙ: supports_fp8_kv + supports_nvfp4_kv у triton flipped
  вместе ([attention/__init__.py](../../python/freetoken/attention/__init__.py)) -
  plumbing общий, включать только nvfp4 значило бы оставить полу-фичу без покрытия.

## Качественный A/B для квант-KV

- Критерий - E2M1/e4m3 parity против dequant-референса пула, НЕ bf16-эквивалентность
  (урок PR #408): юнит-тесты сверяют attention-выход с dense-референсом, собранным из
  декодированных кодов пула (test_attention_reads_packed_cache в
  [test_kv_nvfp4.py](../../tests/kernels/test_kv_nvfp4.py); отдельные row- и
  block-gain'ы ловят случайный fp8-рескейлинг nvfp4).
- Сквозной критерий - "без деградации": boot A/B на RadixArk/Qwen3.8-27B-NVFP4-FTW
  (dense GDN+FULL, pool FULL, head_dim 256, --attention-backend triton,
  --kv-cache-dtype nvfp4): KV 231954 ток (4.09 GiB) против bf16 67049 ток = 3.46x;
  greedy-батарея 12 задач (temp 0, max_tokens 768): 11/12 byte-identical, 1
  argmax-flip на ОТКРЫТОЙ перефразе (оба ответа корректны) - норма, не регресс;
  needle-in-haystack @30% глубины найден идентично (ZX-7412-KM).
- Dense-кандидат для A/B: Qwen3.8-27B-NVFP4-FTW. GLM-5.3 не годится - там dsa-бэкенд,
  у которого nvfp4-KV и так был; нужен FULL/GDN-чекпойнт, чтобы трогать именно
  generic-triton путь.

## Ловушки и API-гэпы

- Ловушка e4m3_compat ("cannot cast int32 to fp8e4nv" при расхождении host-проба /
  constexpr под CUDA-graph) к KV-пути структурно неприменима: коды идут uint8-
  указателем, dtype фиксирован пулом - риск-пункт спеки закрылся сам.
- bench_kv_quant ([bench_kv_quant.py](../../benchmarks/bench_kv_quant.py), переведён
  на текущий API): 3.37x ужатие @1024 ток; decode nvfp4 МЕДЛЕННЕЕ bf16 - ужатие
  капасити не равно ускорению декода; nvfp4-KV продавать как capacity, не как speed.
- Ревью нашло API-гэп, унаследованный из backup: _validate_kv_format проверял строку
  kv_quant, но не dtype кодов - 16-битный тензор со шкалами проходил как "fp8" и
  декодировал мусор. Фикс 660c814: ValueError на не-uint8 кодах + негативные тесты
  (16-битный пул под kv_quant="fp8"; неизвестный kv_quant). Урок: валидатор формата
  обязан проверять dtype кодов, а не только дискриминатор.

## Приёмка (числа с конфигами)

| Прогон | Конфиг | Результат |
|---|---|---|
| [test_kv_quant_config.py](../../tests/engine/test_kv_quant_config.py) `-p no:skipping` | CPU, дерево 660c814 | 8 failed -> 36 passed |
| [test_kv_nvfp4.py](../../tests/kernels/test_kv_nvfp4.py) | RTX 5090, CUDA | 115 passed (81 волны 22 + 32 sliding-window + 2 валидационных) |
| [test_triton_attention.py](../../tests/kernels/test_triton_attention.py) | RTX 5090, CUDA | 49 passed |
| контрольная батарея `tests/ -m "not slow"`, CUDA_VISIBLE_DEVICES="" | CPU, дерево 660c814 | 2 failed (базлайн-пара: qwen4_ple bitwise, offload FakeGraph) / 2079 passed / 850 skipped |

Ограничение: fa / fi / trtllm остаются с supports_\*_kv=False - их внешним kernel'ам
нужна наша layout-схема (row fp32 + block e4m3-u8/16 + low-nibble first); каждый -
отдельный PR с тестом и A/B.
