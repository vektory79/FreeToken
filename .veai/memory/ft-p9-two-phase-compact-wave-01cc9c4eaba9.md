---
name: "ft-p9-two-phase-compact-wave"
description: "P9 two-phase blob compact: COMMITTED ab52a33, docs e45c592 + P11 brief; tempo PARTIAL"
type: project
lastUpdated: 2026-10-01T13:05
lastRecall: 2026-10-04T22:23
---

# P9 two-phase blob compact wave (boot-shutdown-io, 2026-10-01, UNCOMMITTED @ 1de8069)

Компакт теперь физически сжимает блоб (файл == live), крэш-безопасен; незакоммичен,
коммит и default - решение пользователя.

## Дизайн
- Two-phase commit: blob.bin.new (плотная переписка live, 8 воркеров pread->pwritev,
  чанк 32 МиБ, пос-записная crc-верификация прочитанного) -> journal.log.new (новые
  офсеты) -> swap.token {old/new sizes, sha256 журнала, generation} = АРМИРОВАНИЕ
  (fsync+rename+dir-fsync) -> коммит: hard-link .old якоря -> 2 rename -> unlink
  СТРОГО после финального fsync_dir.
- Бут-рекавери в replay_journal ДО чтения журнала/маркера: детерминированный предикат
  rollback-vs-complete; ЯКОРЯ .old = индикатор фазы (сильнее размеров при совпадении
  объёмов); old sizes в токене для до-якорной фазы + диагностики; O(журнала) не
  O(блоба); один retry ветки.
- refs-гейт: компакт откладывается при restore в полёте (офсеты меняются).
- FREETOKEN_COMPACT_WRITERS (clamp <=64, warning).

## Error-path уроки (MAJOR триажа)
- W10: runtime-OSError после армирования: except НЕ чистит evidence - armed ->
  синхронный откат через .old, невозможен -> оставить ВСЁ бут-рекавери; unlink
  якорей только после финального fsync_dir. Cleanup после arm = полный лосс live.
- W11: rollback-vs-completed неоднозначность при совпадении размеров -> якоря как
  фаза + old sizes в токене.
- W14: второй compact при висящем токене сносил evidence -> guard на входе compact
  (warning + return 0). Принятое следствие: стор до рестарта на mismatched in-memory
  offsets (restores читают новые байты по старым офсетам), бут сходится.
- MINOR: маркер НЕ пишется при _dead_records>0 (зомби); fd reopen open-before-close
  атомарно; evidence-preserving error paths - урок в
  kb/methods/deterministic-concurrency-tests.md.

## Железо (5 бутов, A/B)
- Файл == live: PASS (153.19 -> 31.95 GiB, дважды; база не сжимала 143.36).
- Темп: 1.40 ГБ/с device (write 3.10) = x2.3 от базы 0.60; гейт 1.5 НЕ ВЗЯТ ПО БУКВЕ -
  стена = конкурентный teardown движка при шатдауне (вне сервера тот же паттерн
  3.21 ГБ/с; P9 vs P9B паритет - не код). Следующий рычаг (не брифицирован): разнос
  компакта и teardown (runtime-компакт на idle-точках).
- SIGKILL mid-copy (t=+7.9 GB .new) -> полный реплей 221 rec, НОЛЬ потерь; resume HIT
  9280/9323 трижды; P2 fast-path после сжатия 0 с, маркер VALID с новым getsize.
- Тир оставлен оператору сжатым: 169 rec / 31.95 GiB / маркер present.

## Тесты/статус
- tests/scheduler/test_session_tier.py 113 (24 новых кейса P9, 3 на новую семантику);
  183/183 + 70/70 = 253 passed. Fails-before: 22/24 на HEAD, fix-волна 4/5.
- Ревью x2 + fix + верификация (нашла W14-MAJOR) + финал: STOP GATE READY.
- Незакоммичено: session_tier.py + tests + kb (P9-NOTES + 7 правок).
- Открыто: teardown-разнос (кандидат-рычаг, не брифицирован); token-окно на железе
  не демонстрировалось (мс, CPU-матрица); libaio не делался.

## P9 закоммичен (2026-10-01, вечер)
- **ab52a33** perf(scheduler): compact shrinks the l2 blob file via two-phase swap
  (session_tier.py +518, тесты +520; 183 passed перед коммитом).
- **e45c592** docs(kb): record p9 two-phase compaction wave and p11 teardown-separation
  brief (8 kb-файлов).
- Статус "UNCOMMITTED" в этой записи устарел. P11-бриф (разнос компакта и teardown)
  зафиксирован в TASK.md - отдельная сессия, старт по явному запросу. Некоммиченный
  хвост: микро-правка устаревших статусов P7/P8 в 3 kb-файлах (замечен и хвост
  P8-NOTES.md frontmatter).
