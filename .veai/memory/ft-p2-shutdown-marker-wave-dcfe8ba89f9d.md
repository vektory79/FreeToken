---
name: "ft-p2-shutdown-marker-wave"
description: "P2 clean-shutdown marker fast-path in replay_journal; committed 621d100, kb 9efdfbf; iron accepted (replay 22s->0s)"
type: project
lastUpdated: 2026-09-30T22:41
lastRecall: 2026-10-01T12:11
---

# P2 shutdown-marker wave (boot-shutdown-io, 2026-09-27/28, COMMITTED 621d100)

## Механика
- Sidecar <tier-dir>/shutdown.marker, JSON {journal_bytes, blob_eof, generation} в
  crc-обрамлении <II len, crc32>, tmp+fsync+rename+dir-fsync. Fast-path в
  replay_journal (session_tier.py): not torn + валидный маркер -> пропуск per-record
  zlib.crc32(pread); полный путь байт-в-байт прежний.
- Generation монотонный, сидируется из ЛЮБОГО crc-валидного маркера ДО size-match
  (иначе компакт, уменьшивший журнал, делает stale-маркер неотличимым от валидного).
- Инвалидация: shutdown_tier (в начале), compact, _discard; discard не меняет дисковые
  размеры -> явный durable unlink (dir-fsync) только при флаге _marker_present
  (5 точек обновления флага).
- Устойчивость: isinstance(mk, dict) guard (иначе AttributeError крошит бут); лог
  fast-path + "present but rejected", тишина при отсутствии маркера.
- Тесты: test_session_tier 46 passed (8 новых), test_hybrid_cache_manager 70;
  fails-before эмпирический (stash+md5, single-revert). Ревью x2 + триаж + верификация:
  STOP GATE READY.
- Коммиты: **621d100** perf(scheduler): skip l2 blob crc re-read on boot via
  clean-shutdown marker (10 файлов, 853+/25-, вкл. kb-дистилляцию; TASK.md ушёл как
  новый файл); **9efdfbf** docs(kb): record p2 iron acceptance and p3 wave notes.
  Stale-index ловушка подтверждалась: чужой мусор в индексе, снято git reset +
  перестейдж по явному списку.

## Железная приёмка - ВЫПОЛНЕНА и ПРИНЯТА (9 бутов, .tasks/boot-shutdown-io/p2-iron/)
- ARM1 graceful PASS: полный путь 22 с (112 записей / 80.23 GiB, база совпала) ->
  fast-path 0 с после каждого graceful-стопа (цель <=2 с), resume HIT 9536/9582,
  маркеры VALID gen 1->2->3.
- ARM2 №1 промах: на переполненном тире (80.61 GiB vs кэп 50) demote отказан ->
  shutdown выбросил ВСЕ записи и обтрунцировал блоб 86.55 GB->0 за ~1-2 с; SIGKILL
  +2.5 с опоздал.
- ARM2-RETRY PASS: evidence-trigger kill t=+0.7 s mid-flush -> маркер ABSENT, Boot#8
  полный путь (нежурналированный хвост блоба корректно игнорирован), конвергенция
  8<->9. Наблюдение: "replay dropped" при mid-flush недостижима (blob fsync ДО
  journal append).
- ИНЦИДЕНТ (мотивировал fix :329): прод-контент L2 оператора (80+ GiB) СТЁРТ
  shutdown-кэпом - позже объяснён как вырождение бага :329 (watermark vs live),
  закрыт ab3a117 (ft-fix-329-cap-accounting).

## Уроки
- Тест-хелпер обязан зеркалить прод-дисциплину shutdown_tier (invalidate -> flush ->
  compact -> write), иначе тестирует не тот порядок.
- caplog не видит non-propagating module logger -> собственный capture в тестах.
- nvidia-smi верификация обязательна перед железной рукой: первая попытка абортирована
  на префлайте - VRAM была занята прод-сервером (порт 18081) при заявленном
  "переключился на облако". Census по СВОЕЙ форме (свой порт), прод не трогать.
- Технота "реплей = дельта tier_on -> replay_done" НЕВАЛИДНА с fix :329 (строка
  "session tier on:" переехала в хвост replay_journal) - см. ft-fix-329-cap-accounting.
