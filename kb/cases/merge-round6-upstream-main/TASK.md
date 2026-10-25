---
title: "Round 6: merge upstream/main 32056cf в vektory79 - минимальный зелёный merge + порт-долг D2/D4/D5"
date: 2026-10-25
hardware: "CPU-only окно (pytest под hard-timeout); boot-smoke и GPU-порции отложены в cloud-окно"
commits: "merge 594934d -> amend 220b79e (родители 23c1f02 + 32056cf = upstream/main); бэкап backup/vektory79-pre-merge6-23c1f02"
status: "CPU-зелёный (0 failed); boot-smoke + GPU в cloud-окне"
tags: [merge, upstream, round6, main-wins, deferred-ports, skip-маркеры, register, ftw, nvfp4, mmproj]
---

# TASK: round 6 - merge upstream/main 32056cf (38 коммитов) в vektory79

Status: **CPU-зелёный, NOT fully validated** - merge-коммит 220b79e даёт 0 failed
на CPU-поверхности, но boot-smoke и GPU-порции отложены в cloud-окно. Процедура
волны и накопленные уроки раундов - [merge-wave-procedure](../../topics/merge-wave-procedure.md);
карта порт-долга round 5 - [round5-ggml-replatform-deferred-ports](../../topics/round5-ggml-replatform-deferred-ports.md).
Полные артефакты волны: discovery / adaptation-log / test-results / review-report
в рабочей папке `.tasks/merge-main-round6-work/` (в kb не зеркалированы; суть
дистиллирована здесь и в процедурной статье).

## 0. Контекст и цель

- 2026-10-24 discovery (READ-ONLY): upstream/main ушёл вперёд на **38 коммитов**
  (11c1adb..32056cf, tip #776, 2026-10-23); merge-base = 11c1adb (родитель
  round-5 merge f0eed55). origin/main (6410c97) полностью содержится в HEAD -
  дельта только по upstream. Конфликт-кандидаты: **52 файла** (из 202 изменённых
  main-ом, +11997/-1528), из них 11 тронуты недавними веточными волнами
  починки round-5 (039bbf9, 739336f, e596bba, f12e554).
- Раунд - **re-platform-подобный** (как round 5): main пере-платформировал
  NVFP4-квант-механику (#732/#739 BREAKING, #737 w4a4 sm_120), hybrid fetch
  (#742/#778/#734/#804/#665) и принёс собственные mmproj-пути (#803/#810),
  дублирующие веточные D2/D4 (порты round-5 из 7bc68b0/9314c9b).
- Режим по директиве пользователя ДО волны (binding-урок round 5):
  **минимальный зелёный merge + deferred-ports**. Merge-коммит чинит сборку и
  тесты на механиках main; глубокая адаптация - в порт-долг.
- Backup: `backup/vektory79-pre-merge6-23c1f02`. HEAD на старте = 23c1f02
  (memory/kb-коммиты пользователя поверх 8bba8e5).

## 1. Merge и amend

- Merge 594934d: **27 конфликтных файлов** (резолвы в adaptation-log), плюс
  пост-конфликтные адаптации не-конфликтных файлов (авто-юнион argparse
  server/args.py с двумя --mmproj определениями - снято; duplicate
  e4m3_f32_to_u8 в тритон-юнионе - снято).
- Ревью (проход A/B, CPU) вернуло **MERGE НЕ ЗЕЛЁНЫЙ**: findings F1-F5,
  все закрыты амендом **594934d -> 220b79e** (ровно 9 файлов в диффе amend).
- Инварианты amend: parents 23c1f02+32056cf, `rev-list --count` = 1, 0 .veai
  файлов, конфликт-маркеров нет, провенанс 594934d побайтово объяснён 32
  файлами adaptation-log.

## 2. Findings F1-F5 (причины аменда)

- **F1 [blocker] register.py - main-wins вернул round-5 регрессию.** Резолв
  "theirs" молча потерял веточный фикс `_gguf_tensor_types` (48915b4): для
  FTW-конвертов main-версия безусловно звала `gguf_tensor_index` (metadata-only
  carrier -> пустая таблица типов) - баг "gguf->FTW не бутится" вернулся.
  Поймал `test_ftw_quant_config_resolves_schemes` (AssertionError: gguf_tensor_index
  consulted on the FTW config path). Фикс: `_gguf_tensor_types` verbatim из
  23c1f02 + точечный вызов в `checkpoint_quant_config`; сверх минимума портирован
  семейный контракт `ftw_tensor_types` (5 layer maps) в glm5_next/gguf.py -
  без него old_carrier_contract_rebuild оставался красным.
- **F2 [major] engine.py - осиротелый хелпер.** После main-wins rewrite
  `_init_offload_moe_cache` (probe/settle_banks) вызов `_gguf_auto_floor_gate`
  ампутирован: в проде ни одного call-site, но юнит-тесты звали def напрямую -
  зелёные тесты маскировали мёртвую гвардию fail-fast unfundable slot floors.
  Фикс: вызов вживлён в `_init_offload_moe_cache` сразу перед `load_expert_banks(`
  (гейт зависит только от живых символов: gguf_signature_groups,
  check_partition_floors, _resolve_auto_moe_cache_size); wiring-тест адаптирован
  под main-структуру (capability-screen переехал в `_gguf_hybrid_rejection`).
- **F3/F4 [major] skip-дисциплина.** Выведенные D-PORT-тесты не были помечены
  skip - сюиты громко падали, merge не был зелёным. Помечены с reason= именованный
  порт: test_ftw_gguf_banks (4 x D-PORT-1) и test_glm5_next_gguf (итог 35 passed /
  91 skipped; замер grep-ом маркеров: D-PORT-D2 x44, D-PORT-D4 x37, 1 прочий -
  изначальная заявка волны 30/51 была неточной).
- **F5 [minor] weight.py:263** - висячий `from .gguf import iter_gguf_vision_weights`
  ампутирован; путь `iter_vision_weights(gguf + mmproj_path)` теперь громкий
  RuntimeError с именем D-PORT-D2 (не hot: в проде недостижим при main #810/#803).
- **Вне-скоуп находка** (не из F1-F5, падение предсуществовало на 594934d):
  2 stale-pin теста test_quant_config ("usable here: triton" / auto->triton) -
  main #737 добавил nvfp4 w4a4 на sm_120; помечены skip "recalibrate with the
  D5 nvfp4 port" по прецеденту соседних stale-pin skip.

## 3. Порт-долг round 6 (branch-функциональность, ампутированная merge)

- **D2 mmproj vision - withdrawn, main-wins** (#803/#810): main принёс
  собственный mmproj-путь (Mmproj/hf_fields/env, find_mmproj override),
  веточная конверсия gguf->FTW (--mmproj -> visual.* tower + clip.* carrier)
  выведена. Оставшийся долг: 44 skip-маркера D-PORT-D2 в
  tests/models/test_gguf... (test_glm5_next_gguf.py) + 4 теста vision-блока
  tests/checkpoint/test_ftw_gguf_banks.py + RuntimeError weight.py:263.
  ИСКЛЮЧЕНИЕ: веточный стриминг-sink gguf->FTW (layer_sink, гетерогенные
  ширины слоёв не torch.cat) ПЕРЕПОРТИРОВАН на per-layer цикл main.
- **D4 loader contract** (main-wins #721): веточный glm5next GGUF-лоадер
  заменён main-овским; 37 skip-маркеров D-PORT-D4 в test_glm5_next_gguf.py.
  Контракт ftw_tensor_types уже возвращён F1, остальные гарды - с портом.
- **D5 nvfp4-KV** (ветка 660c814): read-path args (pool_scale/kv_quant/
  pool_block_scale) на glm_dsa_sparse_attn/dsa.py ампутированы (kv-механика и
  e4m3-хелперы живы, wiring - нет); + 2 stale-pin теста test_quant_config
  перекалибровать с портом.
- Прочее: D-PORT-3 (benchbw auto->hybrid upgrade) - main вернул дефолт offload,
  hybrid только через явный `--moe-strategy hybrid`; auto-hybrid при нужде
  ре-портить на per-layer-kind профиль #734. D-PORT-5 (tier eviction offers)
  ЗАКРЫТ В ВОЛНЕ: хуки пересажены на main `_evict` (derive_paths +
  `_tier_offer` перед free), поймано тестом test_kvonly_adaptive_set_degenerates_to_tip.

## 4. Сохранённое веточное (проверено ревью)

- zero-pad фиксы embedding/gguf.py:48 и linear/gguf.py:57 (torch.zeros - наша
  сторона: main всё ещё torch.empty); untile() в tensors.py юнионом с
  check_scales/weight_scale main; layer_sink + first_layer-гвардия в experts/
  expert_banks сплайсе; tier-хуки на main `_evict`; слот-гейт и
  `layer_slot_bytes` (engine/cache_budget.py:17 - `_split_moe_cache_budget`
  не существует с round-5 ре-платформы); capabilty-screen `_gguf_hybrid_rejection`.

## 5. Статус верификации и что отложено

Замеры wave-4/ревью на дереве 220b79e, все `timeout N uv run --extra dev
python -m pytest ...`, сериализованно, CPU-only (VRAM занята локальным LLM):

- tests/checkpoint: **69 passed / 13 skipped**; test_ftw_gguf_banks.py
  20 passed / 6 skipped (3 квант-config теста ЗЕЛЁНЫЕ после F1, не skip).
- tests/engine/test_cache_budget.py: **51 passed / 2 skipped** (заявленные
  wave-4 "53" - bookkeeping-ошибка; wiring-тест зелёный после F2).
- tests/models/test_glm5_next_gguf.py: **35 passed / 91 skipped** (13.8s).
- Консолидированный прогон: **155 passed / 16 skipped, 0 failed**.
- Два nit-расхождения счётчиков в test-results.md (53 vs 51; сплит D2/D4
  30/51 vs факт 44/37) - не корректности, но урок о замерах см. в
  [merge-wave-procedure](../../topics/merge-wave-procedure.md), раунд 6.

Отложено в cloud-окно: boot-smoke `ft serve` GGUF glm5next (обязателен);
GPU-порции (nvfp4 #732-#739, DSA prefill #795, pinned staging #742, vendor
get_rows #801); A/B decode после замены D3-механики main-ом; mmproj vision
smoke + порт D2; порт D4; порт D5 + recalibrate stale-pin. GPU-only тесты
предклассифицированы Environment и не запускались.

## 6. Правила для future-сессий (дистиллят)

1. Main-wins на файле с веточными точечными фиксами - НЕ доверять механике
   резолва: перед фиксацией прогнать grep веточных фиксов по списку недавних
   коммитов ветки (F1: потерянный 48915b4 поймал ровно один тест).
2. Счётчики тестов в test-results.md - только замеренные на финальном дереве,
   не пересказанные по ходу волны (ревью нашло 2 расхождения).
3. После main-wins rewrite - grep call-sites каждого веточного хелпера,
   попавшего в main-wins файл: def без прод-call-site = осиротелый
   (юнит-тесты, вызывающие def напрямую, маскируют мёртвый прод-путь).
4. Skip-маркеры с именованным D-PORT по round-5 дисциплине; сплит причин
   (сколько D2, сколько D4) - замерять grep-ом, не считать по памяти.