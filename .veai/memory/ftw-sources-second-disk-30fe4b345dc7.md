---
name: "ftw-sources-second-disk"
description: "Missing FTW/HF sources: search /media/data/ai/models/; 3 legacy NVFP4 carriers reconverted to -FTW-new, GPU boot pending"
type: reference
lastUpdated: 2026-10-26T17:56
lastRecall: 2026-10-26T17:05
---

# Where to look for missing FTW/HF sources

When a checkpoint conversion (ft checkpoint convert / convert_checkpoint) needs an HF source that is absent under /media/ai/models/, SEARCH THE SECOND DISK FIRST: /media/data/ai/models/ (user-provided, 2026-10-26). Both disks mirror model catalogs (unsloth/, RedHatAI/, RadixArk/ and others).

**Why:** during the round6-deferred-ports D2 wave the NVFP4-FTW reconversion was declared blocked with "no HF source exists on disk to reconvert" - the search had only covered /media/ai/models/. The user pointed out /media/data/ai/models/ as an additional source catalog.

**How to apply:** before reporting an FTW as unreconvertible or a source as missing, ls/glob both /media/ai/models/ and /media/data/ai/models/ for the HF checkpoint (safetensors) of the same model.

## Статус переконвертации (2026-10-26)
Все три legacy NVFP4-носителя переконвертированы на CPU из HF-исходников второго диска (args: moe_backend="offload", quant_backend="moe.nvfp4=triton", device="cpu"; triton - единственный NVFP4-пак с чистыми tensor-копиями, marlin/b12x/w4a4 требуют CUDA): Qwen3.8-27B-NVFP4-FTW-new 21.1G/104s, GLM-5.3-Flash-NVFP4-FTW-new 190.7G/962s, Qwen3.8-Flash-Next-NVFP4-FTW-new 130.4G/628s - все в /media/ai/models/ рядом со старыми. Проверка check_ftw (.tasks/ftw-recheck/): новые OK (fp32-глобалы), legacy FAIL-INIT. Прирост байт = ровно fp16->fp32 расширение глобалов. Legacy-носители НЕ удалены (решение пользователя). НЕ ПРОВЕРЕНО: GPU-boot новых носителей - следующее cloud-окно; бонус: new-writer layout грузится мимо legacy fold-пути.
