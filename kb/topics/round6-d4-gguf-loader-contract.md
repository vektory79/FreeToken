---
title: "Порт D4 round 6: контракт glm5next gguf-лоадера укреплён поверх main - CPU-часть готова"
date: 2026-10-25
hardware: "RTX 5090 32GB (GPU-приёмка - в cloud-окне); CPU - контрольная батарея и проверка гардов на реальном артефакте"
branch-commits: "vektory79: 2276069 (HEAD) поверх d66f6e3 (round-6 merge, parents 23c1f02 + 32056cf); референс-спека - веточный 9314c9b"
status: "cpu-done - boot-smoke raw GGUF и кросс-валидация 38/38 ждут cloud-окно"
tags: [gguf, loader, contract, guards, glm5next, round6, deferred-ports, main-wins, no-dual-logic]
---

# Порт D4 round 6: контракт gguf-лоадера укреплён поверх main - CPU-часть готова

**Вердикт (читайте это первым).** Фаза D4 кампании [round 6 порт-долг](../cases/round6-deferred-ports/TASK.md)
закрыта в CPU-части 2026-10-25 одним коммитом: 2276069 ("feat(models): harden
the glm5next gguf loader contract guards"; 2 файла, +299/-199 - гарды в
[models/glm5_next/gguf.py](../../python/freetoken/models/glm5_next/gguf.py) и
переписанные тесты в
[tests/models/test_glm5_next_gguf.py](../../tests/models/test_glm5_next_gguf.py)).
Все 37 skip-маркеров D-PORT-D4 учтены: 18 переписаны под main, 17
доимплементированы, 1 удалён как семантический дубль main-теста, 1 переназначен
D-PORT-D5. Контрольная батарея двух файлов сюит: 142 passed / 4 skipped /
0 failed; ревью-вердикт READY FOR STOP GATE, NIT-ы применены. Прецедент
round-5 - [glm5next-gguf-loader-port.md](glm5next-gguf-loader-port.md);
mmproj-фаза этого раунда -
[round6-d2-mmproj-vision-port.md](round6-d2-mmproj-vision-port.md).

## Что произошло

Round-6 merge заменил веточный glm5next GGUF-лоадер (9314c9b) main-овским
(#721); в tests/models/test_glm5_next_gguf.py осталось 37 skip-маркеров
D-PORT-D4. Фрагмент, уже возвращённый при merge фиксом F1 (семейный контракт
ftw_tensor_types + 5 layer maps - см.
[кейс round 6](../cases/merge-round6-upstream-main/TASK.md)), оказался покрыт
зелёными немаркированными тестами - кандидатов на разморозку из него не было.

## Что сделано поверх main (гарды сверх main-гардов)

Вся продукция фазы - на main-примитивах
[glm5_next/gguf.py](../../python/freetoken/models/glm5_next/gguf.py); всё, что
семантически покрывается main-гардами и main-тестами, возвращено не было:

- **Хелпер _scalar.** Ловушки int(list) TypeError закрыты: однородные
  per-layer массивы (attention.head_count, feed_forward_length,
  expert_feed_forward_length) фолдятся в скаляр, расхождение массива даёт
  именованный ValueError.
- **Гварды parse_gguf_config:** vocab_size сверяется с тензорно-производным;
  диапазон nextn_predict_layers; KDA-транк 0 < n_recr < n_layer; диапазон
  leading_dense_block_count; положительность kpool/top_k и top_k % kpool == 0;
  expert_shared_feed_forward_length == moe_inter * n_shared.
- **Проверка полноты consumed-set в iter_gguf_weights.** Молчаливая потеря
  тензора закрыта: непокрытые ключи дают ошибку. Исключения - MTP-блок
  (blk.N >= num_layers) и банки экспертов (ffn_{gate,up,down}_exps).
- **Гвард A_log:** неположительный или не-finite ssm_a даёт именованную
  ошибку (ловушка NaN в torch log закрыта).
- **Тайп-пины convert-контракта:** token_embd Q8_0, output Q6_K, Q8_0 на
  packed-линейных через packed().

## Карта маркеров и метод классификации

Метод шага 0 - пробный прогон "pytest -p no:skipping" над маркированными
тестами; зелёных как есть нет (a=0). Итог шага 0: b=1 / c=17 / d=18 / e=1;
на доказательствах маркер 1568 переклассифицирован (d) -> (c), финальная карта
c=18 / d=17.

- (b) 1: веточный test_ftw_text_only_carrier_unchanged удалён - семантику
  несёт main-тест test_ftw_gguf_text_only_boot_unchanged
  ([test_ftw_gguf_banks.py:542](../../tests/checkpoint/test_ftw_gguf_banks.py)).
- (c) 18: переписаны под main-семантику - поверхность ModelConfig, spec
  model_cls Glm5NextForConditionalGeneration + checkpoint_roots
  model/model.language_model, двунаправленный to_gguf round-trip против
  ftw-карт, main-типы и тексты ошибок, pre-keyed токенизация.
- (d) 17: кейсы, которых main не покрывает, доимплементированы на
  main-примитивах - гварды из предыдущего раздела.
- (e) 1: маркер 1433 переназначен D-PORT-D5 - GPU-деquant-ветка возвращается
  вместе с портом nvfp4-KV read-path
  ([round5-d5-nvfp4-kv-generic.md](round5-d5-nvfp4-kv-generic.md)).

Счётчики сверены grep-ом на актуальном дереве: D-PORT-D4 37 -> 0 по всему
репо, D-PORT-D5 = 1. Батареи: два файла сюит 142/4/0; консолидированный
прогон 241/13/0 (замер до NIT-правок).

## Уроки

1. **Over-pinning проверяется против реального артефакта.** Риск, что
   тайп-пины слишком жёсткие, снят прогоном против реального сервинг-файла
   GLM-5.3-Flash-UD-Q3_K_XL.gguf: все гварды проходят, пины держатся на
   фактическом файле; нижние UD-тиры, квантующие attention ниже Q8_0, фейлятся
   громко именованной ошибкой, а не молча.
2. **Ревью-проход окупился снова.** Вердикт READY FOR STOP GATE; NIT-ы
   применены (явный NaN-гвард с новым тестом, роутинг _scalar для чтений
   vocab_size / kpool / expert_shared, единый префикс текстов гвардов).
   Урок round-6 D2 ("один проход = блокер + миноры") повторился.
3. **Переклассификация по доказательствам, не по гипотезе шага 0.** Маркер
   1568 сдвинут (d) -> (c), когда тест прочитан по существу; стартовая карта -
   гипотеза, финальная - по факту чтения.
4. **Артефакты фаз и чистота индекса.** Карта маркеров фазы дважды была
   создана делегированными агентами в корне репо (round6-deferred-ports/)
   вместо рабочего каталога кейса и однажды осталась stale-staged в git-индексе;
   оркестратор обязан перед коммитом проверять и местоположение артефактов, и
   git status - карты маркеров в kb не зеркалируются.

## Осталось (cloud-окно)

GPU-часть фазы, по VRAM-протоколу: boot-smoke raw GGUF glm5next +
кросс-валидация весов по методике round-5 D4
([glm5next-gguf-loader-port.md](glm5next-gguf-loader-port.md)): 38/38
non-permuted, плотные corr >= 0.9998, банки экспертов в квант-допуске.

Статус, карта маркеров и шаги фазы - в разделе 5
[кейса round 6 порт-долг](../cases/round6-deferred-ports/TASK.md). Смежные
фазы: D2 - [round6-d2-mmproj-vision-port.md](round6-d2-mmproj-vision-port.md),
D5 - [round5-d5-nvfp4-kv-generic.md](round5-d5-nvfp4-kv-generic.md).