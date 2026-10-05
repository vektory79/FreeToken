---
name: "ft-serve-boot-shutdown-io-research"
description: "HW-verified boot/shutdown I/O: bank cap = multi-stream shape, born-pinned refuted; replay 22 s; flush QD1; shard restore"
type: project
lastUpdated: 2026-10-04T20:57
lastRecall: 2026-10-04T22:52
---

# ft serve boot/shutdown I/O research (2026-09-26, research-only, NO fixes implemented)

User task: explain slow boot (MoE load all-cores; L2 load many reads no restore) and slow
shutdown flush 200-400 MB/s (GLM-5.3 GGUF FTW, 0.82/400k, tier 10/50 GiB, /media/ai on
nvme0n1 990 EVO Plus ~6.4 GB/s sustained). Deliverable = chat report; fixes deferred.

## Mechanisms (code anchors, vektory79 working tree)
- Boot order is SEQUENTIAL: scheduler.py:60 Engine (banks load) -> scheduler.py:86
  CacheManager -> cache.py:293 replay_journal. No tier-vs-banks SSD overlap.
- P1 MoE load all-cores = BY DESIGN: ftw.py:56-57 _DEFAULT_CHUNK=8MiB,
  _BANK_CONCURRENCY=4 -> outer pool min(max(4,16),n_jobs)=16 threads, each read_into
  spawns nested 8-worker pool -> up to 128 concurrent 8 MiB O_DIRECT preadv. 125G @
  4.12 GB/s vs 6.4 sustained: gap ~= 990 EVO Plus 12 s ramp-up (1.8-3.4 GB/s HMB warm-up)
  + born-pinned cudaHostAlloc sections. SSD-contention hypothesis REFUTED.
- P2 slow L2 boot load = replay_journal (session_tier.py:949) verifies EVERY record's
  FULL payload: :984 zlib.crc32(os.pread(blob_fd, n, off)) - single thread, buffered QD1
  (~1.5 GB/s cap, own comment :697), zlib.crc32 holds the GIL (~1-1.5 GB/s/core).
  9.38 GiB blob -> ~10-20 s. Purpose: crash-hole convergence; REDUNDANT after graceful
  shutdown (blob+fsync before journal record). No clean-shutdown fast path exists.
- P3 flush 200-400 MB/s (kb anchor ~600): offers+flush_live -> _demote/_write_blob
  (session_tier.py:530-585): per-page pool.read bytearray copies, single os.write loop to
  BUFFERED O_APPEND fd, os.fsync(blob) PER SEGMENT (:568), then journal json+crc+write+
  fsync PER RECORD (:926-945); flush_live holds global RLock across loop (:916) -> QD1,
  zero parallelism. THEN cache.py:924 shutdown_tier calls compact() UNCONDITIONALLY
  (no-op only if zero dead records; dead bytes visible in "session tier final:" line):
  compact rewrites live spans per-record os.pread(full span)->os.pwrite (GiB python allocs,
  QD1 buffered) + fsync + rename (session_tier.py:1043-1080).

## Candidate levers (for future fix sessions, unpicked)
- P2: clean-shutdown marker (fsync'ed barrier record/file after shutdown_tier) -> boot
  skips payload CRC entirely (journal parse only, <1 s). Fallbacks: sampled CRC (first
  block/record), FTW-style parallel O_DIRECT verify + hw CRC32C (isal/crc32c), or
  verify-at-restore. Biggest boot win.
- P3: parallel segment writes (reserve _blob_eof offsets, pwrite via O_WRONLY not
  O_APPEND, 4-8 writers / io_uring), group fsync (per 1-2 GiB) instead of per segment,
  pwritev from L1 pool regions to skip payload assembly copies, defer/shrink shutdown
  compact (or move to boot-idle), byte reduction via existing L2 snapshot-dedup brief
  (.tasks/session-cache-tiering/briefs/l2-snapshot-dedup.md, 23.8 -> ~7 GiB expected),
  optional zstd on payloads (measure ratio). hw CRC32C helps both P2+P3.
- Instrument FIRST (measured-advice-over-paper): phase timers in shutdown_tier
  (offers/flush write/fsync/compact) + iostat; log anchors: "session tier: replayed N
  journal records" boot time, "replay dropped N dead/torn records", "session tier final:"
  dead=MiB -> predicts shutdown-compact rewrite volume.

## HARDWARE-VERIFIED (2026-09-27, probes in .tasks/boot-shutdown-io/, dev venv @ 655da16)
- INCIDENT: freetoken-00017.ftw (2.9 GB tail shard) deleted by UNKNOWN actor at 22:52 during
  probes (not our probes - read-only). Restored via re-conversion into -FTW-restore (CONVERT_EXIT=0),
  verified (18/18 shards vs index, old/new shard lists + gguf_types IDENTICAL, md5 spot-checks
  shard0/16 @1GiB equal, smoke read down#L00008 1.85 GiB), swapped: prod path restored,
  broken dir kept at GLM-5.3-Flash-UD-Q3_K_XL-FTW-broken-20260927. Also: two LEAKED multiprocessing
  spawn workers from a day-old ft checkpoint (PIDs 2331577/2343203, parent systemd --user, 0% CPU,
  hold source_metadata.gguf open) - hygiene finding, not killed.
- P1 VERDICT: ramp-up hypothesis REFUTED (user was right); pin pipeline REFUTED by A/B:
  Boot A default (mmap + pin-after-fill) banks 125G in 28-29 s ~4.3 GB/s avg, ready +85 s;
  Boot B FREETOKEN_BANK_CUDA_ALLOC=1 (born-pinned) banks 32 s, ready +122 s (+37 s: upfront
  cudaHostAlloc serialization) - born-pinned is NOT a lever, default stays. Real cause =
  device multi-stream ceiling at loader shape: fio 1xQD32x1M = 6.45, 8xQD8x8M = 5.27,
  16xQD8x8M = 4.70 GB/s; replica (no CUDA) seq8 4.05, seq32 3.51, par4x8 4.89, par16x8 4.83.
  Loader instantaneous bar rates oscillate 2.7-12.8 GB/s; flat ~4.1-4.3 is the average.
  Lever: fewer streams + deeper per-stream queues (io_uring), est ~1.4-1.5x on the ~30 s phase.
- P2 MEASURED: replay_journal = 22 s (Boot A) / 24 s (Boot B) EVERY boot for 99 records /
  41.41 GiB live L2 (log anchor "session tier on: ... used=41.41 GiB"). ~1.9 GB/s effective
  (buffered QD1 pread + single-core zlib.crc32). Clean-shutdown marker would save ~22 s/boot.
- P3 MEASURED: idle stop = 12 s, exit 143, dead=0.0 MiB -> user's cache has ZERO dead bytes;
  at their real stop compact() no-ops, so the 200-400 MB/s is the pure flush write path
  (QD1 + per-segment fsync) as agreed; compact is NOT part of their shutdown cost.
- Production tier dir untouched by probes: 99 records intact, no drops, l2 used unchanged;
  journal backup in .tasks/boot-shutdown-io/.

## Обновление 2026-10-04 (P12 wave-1): прогнозы P1-левера закрыты цепочкой волн
Прогноз этой записи "fewer streams + deeper per-stream queues (io_uring), est ~1.4-1.5x" закрыт: P1 pool = 1.12x (свип пула плоский), P7 io_uring QD опровергнут (5.16 ГБ/с), P12 zero-fill NO-GO (доля 8.6% @ дефолт pool=8 < гейта 10%; warm-потолок 22.85 с нереализуем; кандидаты ~0/отрицательны). Плато 5.2-5.4 ГБ/с = диск (98-99% busy) + форма пути; гейт P1 (23 с / 5.5 ГБ/с) на default-пути недостижим известными рычагами. Гейт берут только born-pinned руки ценой общего бута (не рекомендация). kb: TASK.md P12 блок закрытия, P1-NOTES коррекция, kb/methods/limiter-attribution.md.

## Alloc-bench (2026-10-04, follow-up вопроса про ~50 с RAM-аллокации born-pinned)
Замер 6 рук по 125.2 GiB (freetoken.kernel.pinned, сериализовано, .tasks/boot-shutdown-io/alloc-bench/): cudaHostAlloc серийный 21.5 с (6.25 ГБ/с); +zero_() = 25.2 с (zero_ = 3.7 с, на FTW избыточен); параллель 8/16 тредов = 0 выигрыша (21.7); ОДИН большой блок = 0 выигрыша (22.2) - драйвер сериализует изнутри, потолок ~6 ГБ/с неускоряем; cudaHostRegister = 19 ГБ/с (7 с на 125 GiB, дешёвый); python-touch parallel(8) = 2 ГБ/с (GIL). Large pages для host-pinned CUDA API нет. Структурный вердикт: born-pinned платит 21-25 с неускоряемой аллокации за ~3 с выигрыша бара - структурно проигрывает дефолту (там touch спрятан под IO, +8.6%); unlock-рычага аллокации для P10/P12 нет. Сходится с P10-iron бутами (+21..26 с wall).
