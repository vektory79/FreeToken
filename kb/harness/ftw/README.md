# ftw/ — GGUF -> FTW конвертация и parity-проверки

Карточки использования. Контекст: [../../topics/ftw-load-path.md](../../topics/ftw-load-path.md),
[../../methods/wave1-notes.md](../../methods/wave1-notes.md), [../../methods/wave2-candidate-a.md](../../methods/wave2-candidate-a.md).
Числа и итоги: [../../baselines/ftw/](../../baselines/ftw/).

### run-convert-wave1.sh
[run-convert-wave1.sh](run-convert-wave1.sh)
- Назначение: wave-1 hardware-validation раннер конвертации bare GGUF -> FTW (`uv run ft checkpoint --model <SRC> --out <OUT> --moe-backend offload` с `FREETOKEN_CONVERT_PROGRESS=1`).
- Запуск: `./run-convert-wave1.sh` (без аргументов; SRC/OUT/лог захардкожены); hygiene: trap-on-EXIT, timeout 1800 s, FAILPAT tail-watchdog (5 s poll), SIGTERM->SIGKILL по process-group (setsid).
- Готчи / требования: отказывается писать в непустой OUT-каталог; python-зависимости через `uv run` из корня репозитория; конвертация ~пара ГБ/мин — контролировать диск по прогрессу в логе.

### verify-convert-wave1.py
[verify-convert-wave1.py](verify-convert-wave1.py)
- Назначение: read-only послепроверка конвертации: файлы OUT, meta индекса `freetoken_weight.json` (quant_format, gguf_types, shards, counts, expert_bank_num_layers), per-layer `experts_bank` записи (непрерывность id слоёв, numel-чек), tmp/partial-остатки.
- Запуск: `verify-convert-wave1.py [<OUT-каталог>]` (по умолчанию — FTW-дерево GLM-5.3-Flash UD-Q3_K_XL).
- Готчи / требования: только инспекция — ничего не пишет; numel-чек `[num_experts, rows, row_bytes] * nbytes` — главный structural-контроль; non-layer experts_bank-записи печатаются как аномалии.

### run-wave2-measure.py
[run-wave2-measure.py](run-wave2-measure.py)
- Назначение: wave-2 измерение FTW boot: одна сериализованная прогонка (preflight census -> boot с winner-флагами -> boot-log sanity -> 65,585-токеновый префилл -> 3x decode на radix-cached prefix -> 24-промптовая greedy-батарея -> teardown) с env-pins committed defaults (`FREETOKEN_GGUF_MOE_MTILE=32`, `FREETOKEN_GGUF_DENSE_MTILE=64`).
- Запуск: `run-wave2-measure.py` (без аргументов; порт 18801; пишет `wave2_results.json` в out-каталог); коды выхода 0/2/3/4/5/6/7.
- Готчи / требования: measure.py не импортирует — методология зеркалится (см. [serve-measure/README.md](../serve-measure/README.md)), чтобы числа были сопоставимы с bare-GGUF референсами t4a/t64a/t64b и battery-post-fix; strict serial, quiet-box gate по чужим VRAM-держателям (ft serve / benchbw).

### wave2-parity.py
[wave2-parity.py](wave2-parity.py)
- Назначение: parity-анализ батареи FTW boot против bare-GGUF референсов: sha256 равенство, difflib ratio + LCP/LCS по reasoning+content, класс дивергенции (identical / mid-answer / suffix-flip / whole-answer).
- Запуск: `wave2-parity.py` (без аргументов); читает battery-артефакты FTW-прогона и референсные каталоги по захардкоженным путям; печатает per-prompt вердикт + summary.
- Готчи / требования: bare-GGUF output бимодален по boot'ам (same-config пары флипают на большинстве промптов), поэтому «matches at least one bare-GGUF reference boot» классифицируется как boot-noise-level parity — это не «identical»; полный identical-all-refs — отдельный, более сильный вердикт.