---
name: "ft-fix-329-cap-accounting"
description: "Fix :329 boot _ssd_used reseed from live journal, commit ab3a117; replay-timing tech-note invalid"
type: project
lastUpdated: 2026-10-04T19:44
lastRecall: 2026-10-04T19:42
---

# Fix :329 - boot cap-accounting reseed (COMMITTED ab3a117, 2026-09-29)

Баг из железной приёмки P3 (.tasks/boot-shutdown-io/p3-iron/results.md): на буте ни
один путь не пересеивал _ssd_used из живых записей - она оставалась
os.path.getsize(blob) (водяной знак с дырами, session_tier.py:329); :1397 был
compact(), не полный путь. Следствия (HW): пере-эвикция на следующем стопе (116 vs 44
на идентичном стейте); watermark 87.8 GiB vs live 2.8 -> ВСЕ demote REFUSED -> стоп
вырождается в discard+truncate+маркер = стирание тира. Исторические "wipe" (ARM2
P2-iron, стирание прод-L2 ~47-80 GiB) - вырождения этого бага (механизм закрыт в
P5-исследовании, ft-p5-lru-research).

## Фикс
- Общий хвост replay_journal (~:1316): live = сумма padded (l2_n -> _BLK) по in_l2
  сегментам; _ssd_used = live; _dead_bytes = max(0, self._blob_eof - live)
  (хардненинг от верификации: эквивалентно на буте, точная семантика dead при
  повторном вызове).
- _blob_eof НЕ тронут (never-decreased водяной знак, дыры безопасны, резерв за ним
  пинит test_failed_write_hole_skipped_by_next_group_no_dropped). P2-маркер не тронут.
- Лог "session tier on: used=" перенесён из __init__ в хвост replay_journal: оба пути
  печатают LIVE; при codec-refusal (cache.py) строка больше не печатается.

## Приёмка и коммиты
- test_fast_path_boot_seeds_ssd_used_from_live_records (fails-before эмпирический:
  сингл-реверт строки пересева -> assert на watermark); test_session_tier 63/63,
  test_hybrid_cache_manager 70/70. Верификационное ревью: STOP GATE READY.
- **ab3a117** fix(scheduler): seed l2 cap accounting from live journal records on boot
  (session_tier.py +27/-9, тесты +47; тело фиксирует HW-эффект: watermark 87.8 vs live
  2.8 -> отказ всего флаша и стирание тира); **23b18a4** docs(kb): record p3 iron
  acceptance results. На момент фикса ветка vektory79 опережала origin на 5 коммитов
  (621d100, 9efdfbf, de26ba1, ab3a117, 23b18a4); push не делался.

## СЛЕДСТВИЕ ДЛЯ ЗАМЕРОВ (навсегда)
Старая технота "длительность реплея = дельта tier_on -> replay_done" НЕВАЛИДНА: строка
"session tier on:" теперь в хвосте replay_journal (после crc-верификации), обе строки
печатаются подряд на обоих путях. Мерить iostat / blob-кривую / собственные метки фаз.

## Брифы P4/P5/P6
Были зафиксированы в kb/cases/boot-shutdown-io/TASK.md (порядок P6 -> P4/P5, P5 после
ask) - ВСЕ исполнены и закоммичены позже; актуальные статусы в wave-памятиях
ft-p4-flush-pipeline-wave, ft-p5-lru-research, ft-p8-record-compact-gate,
ft-p9-two-phase-compact-wave.
