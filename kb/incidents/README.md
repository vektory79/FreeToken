# incidents/ — разборы отказов

Каждый инцидент — подкаталог с отчётом о корневой причине, артефактами
воспроизведения и статусом OPEN/CLOSED. Ключевой урок вынесен в строку таблицы;
входной документ — с чего начинать чтение.

## Инциденты

- [crash-mr080-400k/](crash-mr080-400k/) — **OPEN**. Рецепт-победитель тюнинга
  (0.80/400k, fp8 KV, 8191) требует >=29.28 GiB свободной VRAM на старте и
  падает при меньшем headroom. Вход: [README.md](crash-mr080-400k/README.md);
  артефакты: [error-report.md](crash-mr080-400k/error-report.md),
  [serve.log](crash-mr080-400k/serve.log), repro-раннер
  [repro-runner.sh](crash-mr080-400k/repro-runner.sh) (дубликат в
  [../harness/repro/crash-mr080-repro.sh](../harness/repro/crash-mr080-repro.sh)).
- [nvfp4-1m-capacity/](nvfp4-1m-capacity/) — **CLOSED**. 1M контекст в NVFP4
  не помещается в 32 ГБ: верифицированный максимум 786432 токена, потолок
  заполнения ~678k; защита — fail-fast gate на старте. Вход:
  [failfast-report.md](nvfp4-1m-capacity/failfast-report.md); далее
  [fix-report.md](nvfp4-1m-capacity/fix-report.md),
  [repro-report.md](nvfp4-1m-capacity/repro-report.md),
  [verification-786k.md](nvfp4-1m-capacity/verification-786k.md),
  [verification-deepfills.md](nvfp4-1m-capacity/verification-deepfills.md).
- [fix1-radix/](fix1-radix/) — **CLOSED**. Hardware-отчёт верификации
  radix-фикса fix-1: повтор 65k стал HIT (#cached-token 65472/65536),
  регрессии декода нет. Вход: [hardware-report.md](fix1-radix/hardware-report.md).

## Навигация

- «Сервер падает на старте с большим KV» — [crash-mr080-400k/](crash-mr080-400k/)
  плюс матрица knob'ов в [../methods/vram-headroom.md](../methods/vram-headroom.md).
- «OOM/обрыв на 1M контексте» — [nvfp4-1m-capacity/](nvfp4-1m-capacity/).
- «Radix-кэш не переиспользуется» — [fix1-radix/](fix1-radix/) и открытый бриф
  [../cases/fix3-snapshot-lru-refresh/TASK.md](../cases/fix3-snapshot-lru-refresh/TASK.md).