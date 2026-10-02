---
name: "ft-p3-parallel-flush-wave"
description: "P3 parallel pwrite + group fdatasync + crc32c; stale journal-fd HEAD bug; committed de26ba1; iron producer-bound"
type: project
lastUpdated: 2026-09-30T22:40
lastRecall: 2026-10-01T11:36
---

# P3 parallel-flush wave (boot-shutdown-io, 2026-09-29, COMMITTED de26ba1)

## Что сделано
- Параллельный флаш L2: резерв офсетов (_reserve_blob_span, pending-сумма группы в
  cap-чеке), pwrite 8 воркеров (O_WRONLY без O_APPEND), групповой fdatasync ~1 GiB
  ПЕРЕД журнальными записями группы. Инвариант "журнал только после durable блоба
  группы" сохранён; окно потери расширено с сегмента до группы (docstring _flush_group).
- Watermark-вместо-lseek: провал записи оставляет дыру за never-decreased водяным
  знаком; P2-маркер пишет реальный getsize -> fast-path корректно принимает (дыры не
  референсируются); подрезка watermark отвергнута триажем (гонки с writer'ами).
- CRC32C: isal НЕ экспонирует CRC32C (только IEEE zlib-equal; скрытый C-символ
  crc32_iscsi segfault через ctypes - путь закрыт измеренно) -> пакет crc32c>=2.9,<3.
  Per-record "crc_alg":"crc32c", отсутствие = zlib; downgrade: старый бинарник
  отбрасывает crc32c-записи как dead, журнал well-formed.

## Баги, пойманные в fix-волне
1. PRE-EXISTING stale journal fd после компакции (внесён 2348da5): compact()
   переименовывал journal.log, _journal_fd оставался на отлинкованном inode -> ВСЕ
   журнальные аппенды после записедропающей компакции терялись при ребуте (молчаливая
   потеря всего пост-компактного флаша). Фикс: reopen fd после _rewrite_journal;
   атрибутировано как HEAD-баг в теле коммита.
2. dict-mutation mid-iteration в flush_live: _evict_oldest_l2 -> _discard del-ит из
   словаря mid-loop -> RuntimeError; фикс list()-снапшот.
3. cap-регрессия группового резерва: группа резервировала поверх ssd-кэпа до 1 GiB;
   фикс pending-сумма в _reserve_blob_span/_evict_oldest_l2 (fails-before assert 8==4).

## Коммиты
- **de26ba1** perf(scheduler): parallel l2 flush with group fdatasync and crc32c
  (session_tier.py +249, pyproject crc32c>=2.9,<3, тесты +615); **9efdfbf** docs(kb):
  record p2 iron acceptance and p3 wave notes. Тесты на волне: test_session_tier 62,
  test_hybrid_cache_manager 70. Push не делался.

## Железная приёмка (9.83 GiB / 52 сегмента, .tasks/boot-shutdown-io/p3-iron/)
- de26ba1 = 0.93 GB/s iostat / 1.16 blob-кривая vs база 621d100 = 0.65/0.63 ->
  1.4-1.8x; ЦЕЛЬ >=1.5-2 НЕ ДОСТИГНУТА: util 20-40% - узкое место продюсер (сборка
  payload + сериализованный групповой fdatasync), не NVMe -> мотивировал P4-конвейер.
- Гейты PASS: resume HIT 9280/9325 (99.5%); SIGKILL между группами -> маркер ABSENT,
  полный путь, 10 записей живы, конвергенция. Downgrade-путь (crc32c -> dead)
  подтверждён на железе.
- ГЛАВНАЯ НАХОДКА - баг :329 (fast-path бут сеет _ssd_used из getsize(blob) =
  watermark с дырами; полный реплей - live-сумму): пере-эвикция 116 vs 44; watermark
  87.8 GiB vs live 2.8 -> тир стёрт. Закрыт ab3a117 - ft-fix-329-cap-accounting.
- Прод-L2 снова стёрт кэп-политикой (~47 GiB на A). Мусорная директория
  boot-shutdown-io/p3-plan/ в корне проекта от чужой сессии (task-tools
  relative-path баг) - кандидат на удаление.

## Уроки
- flush_live обязан итерировать снапшот list(): любая мутация словаря сегментов
  (evict/discard из резервации) роняет флаш RuntimeError'ом.
- Тест "аппенд ПОСЛЕ записедропающей компакции через границу бута" поймал stale-fd
  баг, живший с 2348da5 - gap был в отсутствии такого теста.
- isal/crc32c: не гадать про API - dir() и вектор; скрытые C-символы расширений
  небезопасны (ctypes segfault).
