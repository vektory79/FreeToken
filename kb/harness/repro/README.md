# repro/ — воспроизводители и верификаторы

Карточки использования. Протокол мердж-волн: [../../topics/merge-wave-procedure.md](../../topics/merge-wave-procedure.md).

### boot-smoke-r3.sh
[boot-smoke-r3.sh](boot-smoke-r3.sh)
- Назначение: boot-smoke мердж-волны (round 3): один boot GGUF glm5next на :18081 с рецептом gguf-786k (`--kv-reserve-tokens 786432 --kv-cache-dtype nvfp4 --memory-ratio 0.89 --max-prefill-length 4096`), затем один greedy chat-запрос; печатает BOOT-SMOKE-VERDICT / CHAT-VERDICT.
- Запуск: `./boot-smoke-r3.sh` (без аргументов); slot-floor fail-fast неявно поддержан: при провале рецепта перезапускают с фолбэком 0.80/400k вручную.
- Готчи / требования: свободный порт 18081 и отсутствие живого `ft serve`; watchdog с FAILPAT (включая Traceback/CUDA error) выходит с rc 42; cleanup при зависании uvicorn делает kill -9 по дереву. Используется в процедуре: [../../topics/merge-wave-procedure.md](../../topics/merge-wave-procedure.md).

### crash-mr080-repro.sh
[crash-mr080-repro.sh](crash-mr080-repro.sh)
- Назначение: воспроизведение пользовательского краша точной командой-рецептом (`--memory-ratio 0.80 --kv-reserve-tokens 400000 --kv-cache-dtype fp8 --max-prefill-length 8191 --moe-strategy hybrid`, порт 18081) с VRAM-снапшотами до/на пике.
- Запуск: `./crash-mr080-repro.sh` (без аргументов); модель — FTW-дерево, путь захардкожен.
- Готчи / требования: инцидент ОТКРЫТ — детали, статус и логи: [../../incidents/crash-mr080-400k/README.md](../../incidents/crash-mr080-400k/README.md); fail-fast boot может не воспроизвести краш (краш случается под нагрузкой, а не на boot) — прогон с реальным префиллом отдельно не автоматизирован.

### run_one.sh
[run_one.sh](run_one.sh)
- Назначение: hygiene-раннер одной measure.py-прогонки: FAILPAT-watchdog (включая `Tried to allocate` с контекстом), hard timeout 900 s, trap-on-EXIT cleanup с SIGTERM->30 s->SIGKILL.
- Запуск: `run_one.sh <name> <logname> "<extra args>"` — прокидывает в `measure.py full --name --log --extra` из рабочего каталога кампании.
- Готчи / требования: журнал сервера обязан соответствовать `<logname>`; на FAILPAT убивает и сервер, и сам measure.py; rc наследуется от measure.py. Методология прогонов — [../../README.md](../../README.md).