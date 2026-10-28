---
name: "round6-deferred-ports-execution"
description: "round6-deferred-ports: CPU+GPU port-debt closed; D3 verdict NO-GO; decode-gap and Fix-3 split into follow-up cases"
type: project
lastUpdated: 2026-10-26T13:36
lastRecall: 2026-10-26T23:42
---

# round6-deferred-ports: исполнение порт-долга (сессия 2026-10-25)

Кейс: kb/cases/round6-deferred-ports/TASK.md (база d66f6e3, ветка vektory79, push не делался).
Директива no-dual-logic: ветка = референс-спека, всё на main-примитивах, дедуп тестов семантический.
Метод классификации маркеров: пробник `pytest -p no:skipping` + grep-счётчики на актуальном дереве.
Рабочие артефакты: .tasks/round6-deferred-ports/ (progress.md, d2/d4-marker-map.md, d5-impl-spec.md).

## Итог: CPU-объём кейса исчерпан
- D2 mmproj-зрение: 48 маркеров (a=0/b=0/c=38/d=2/e=8) все сняты. Коммиты 523452b (vision через
  main iter_mmproj_weights + carrier hook gguf_carrier_mm_config), e1de1df (gguf->FTW packing:
  convert_checkpoint mmproj_path, extra_kvs=clip.* через существующий collision-гвард reader.py),
  e1f4166 docs kb/topics/round6-d2-mmproj-vision-port.md. Ревью-блокер: реальный mmproj-BF16.gguf
  несёт swiglu_limit/image_size без бюджетных ключей -> fallback в main-примитиве + референсный
  бюджет 16/8000 (обе key-shape формы в фикстурах).
- D4 gguf-loader contract: 37 маркеров (a=0/b=1/c=17/d=18/e=1) все учтены. Коммиты 2276069
  (гварды glm5_next/gguf.py: _scalar folding per-layer массивов, parse_gguf_config, coverage-check
  итератора, A_log non-finite/NaN, type pins token_embd Q8_0/output Q6_K/packed Q8_0), 771b5d1 docs
  kb/topics/round6-d4-gguf-loader-contract.md. (b)-дубль test_ftw_text_only_carrier_unchanged удалён.
- D5 nvfp4-KV: ПОСЫЛКА КЕЙСА ОПРОВЕРГНУТА - read-path пережил merge (upstream main 32056cf вообще
  без KV-квант-механики; args восстановлены amend d66f6e3); ампутирован только GPU-деквант _cast
  glm5next-лоадера (+ мёртвый _cuda_target = тихий CPU-деквант при cuda). Коммиты ee961f0 (_cast на
  main-примитивах matrix/kernel.ggml.dequantize/_float, device через все packed-row cast sites),
  1eb556d (пины "usable here: w4a4, triton"; авто-тест -> test_nvfp4_auto_selection_lands_on_
  w4a4_where_available), 3e94da1 docs kb/topics/round6-d5-nvfp4-kv-dequant.md + коррекция посылки.
- Follow-up fp8_block: 75866d7 (старые пины ждали недостижимые reject-пути - TritonFp8BlockMoEKernel
  получил cpu_format='fp8_block'; переписаны на успешную селекцию), 398487d docs закрытие в case.
- Маркер 1433 переназначен D-PORT-D4 -> D-PORT-D5 и снят D5; все D-PORT-* = 0.
- Батареи: консолидированная `pytest tests/checkpoint tests/engine/test_cache_budget.py
  tests/models/test_glm5_next_gguf.py tests/checkpoint/test_ftw_gguf_banks.py -q` = 243/12/0;
  с test_quant_config = 260/53/0; FAILURE SET чист.

## Уроки
- Посылка "ампутирован" из merge-волны пережила в кейс-файл непроверенной - верифицировать
  git ls-tree/show против дерева ДО планирования фазы (сжало D5 с полного ре-ленда до 3 суставов).
- Вакуумный pytest.raises match по имени tmp-каталога (T93); fixture stand-ins маскируют
  real-file KeyError (T92); metadata-only FTW carrier hook (T95). TRAPS T92-T95 в скилле,
  НЕЗАКОММИЧЕНЫ (user-gated).
- Гигиена: делегированные агенты дважды клали карты маркеров в корень проекта и оставляли
  stale-версии в индексе (AM) - re-stage явных путей и проверять локацию артефактов перед коммитом.

## Осталось (только GPU, cloud-окно)
- D3: честный A/B nvfp4-KV 786k против 22.21 (fp8-KV) и 27.71 (nvfp4 round-5 эталон) -> вердикт
  ре-порт/NO-GO hybrid-механики.
- D2: MMProj smoke #810 (boot --mmproj + e2e-картинка) + verify 347/347.
- D4: boot-smoke raw GGUF + кросс-валидация 38/38.
- D5: boot nvfp4-KV 786k + GPU kernel-батареи (test_kv_nvfp4, test_triton_attention, test_kv_fp8).

## GPU-волны (cloud-окно 2026-10-26): кейс ЗАКРЫТ полностью
- D2: verify 347/347 exact (legacy NVFP4-FTW читается raw-index fallback, 82956fb); boot --mmproj READY 180s, image e2e 23.95 tok/s; реальный unsloth mmproj несёт projector_type="glm5next" -> _PROJECTOR_ALIASES=(glm5v,glm5next) (2e82d81); FTW-vision carrier (347 visual.*, 15 clip.*) конвертируется 266s и БУТИТСЯ без --mmproj READY 71s, image e2e 24.86 tok/s.
- D4: кросс-валидация carrier vs raw GGUF: 38/38 non-permuted, 23 ggml-пары байт-в-байт, dense 1.000000 (3 bf16-repack 0.999999), банки 1.000000; boot raw GGUF READY 138s, ноль D4-гвард предупреждений.
- D5: kernel-батареи 170/0/0 (test_kv_nvfp4, test_triton_attention, test_kv_fp8); boot nvfp4-KV 786k на raw GGUF READY 109s (nvfp4-фингерпринт K+V 2.87 GiB).
- D3 A/B: benchbw-профиль v5 МОЛЧА игнорировался движком (читает v6) -> кап-1 на всех бутах, включая 22.21 fp8-базлайн. FULL `ft bench bw --dtype all --cpu-threads 16` -> v6, gguf-фракции 26-30%. nvfp4-KV 786k с resolved fractions: медиана 23.60 (клиент); matched-контроль к эталону (кап-1+overlap-off, как снимался 27.71 - kb Урок 5): 23.40/22.79 сервер = гэп -15..-18%. КАРТОГРАФИЯ РЕ-ПОРТА 393b682 ПУСТА: slot-пулы уже в main (engine.py:943), fail-fast гейты уже в дереве (cache_budget.py:87). ВЕРДИКТ: NO-GO на ре-порт как средства закрытия decode-гэпа; Fix-3 донесение - отдельный кандидат под fix3-харнесс; гэп - отдельный кейс расследования per-token overhead (~42.7-43.9 vs ~36.1 мс/токен); fp8-vs-nvfp4 при matched fractions не измерено. Свежий FULL-профиль даёт +2-3 tok/s - обязателен после каждого мерджа.
- Коммит закрытия: 1a4c8d1 docs(kb) kb/topics/round6-gpu-acceptance-and-d3-verdict.md + perf-matrix appendix + case TASK.md CLOSED (follow-ups: decode-gap case, Fix-3, CLI convert --mmproj, fp8-vs-nvfp4 matched).
- Гигиена-повтор: task-инструменты делегата снова записали отчёт в stray-каталог <repo>/round6-deferred-ports/ и застейджили; перенесено в .tasks, staging снят.
