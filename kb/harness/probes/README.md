# probes/ — зонды железа RTX 5090 / NVMe

Карточки использования. Контекст: [../../topics/prefill-throughput.md](../../topics/prefill-throughput.md),
[../../topics/vram-budget.md](../../topics/vram-budget.md).

### pcie_bw.cu
[pcie_bw.cu](pcie_bw.cu)
- Назначение: CUDA-источник зонда SM-копирования через mapped pinned host memory: H2D (SM читает host mem) и D2H (SM пишет host mem) на unrolled float4-копии, свип по grid size.
- Запуск: сборка `nvcc -O3 pcie_bw.cu -o pcie_bw`, затем запуск собранного бинарника (без аргументов; 1 GiB буфер, 20 итераций); CUDA-устройство обязательно.
- Готчи / требования: точные флаги сборки исторических бинарников не сохранены — приведённая команда минимально достаточная; измеряется именно SM-side DMA, а не cudaMemcpy — числа несравнимы с `cudaMemcpy`-бенчами.

### pcie_bw / pcie_bw2 (пребилты)
[pcie_bw](pcie_bw), [pcie_bw2](pcie_bw2)
- Назначение: пре-компилированные бинарники зонда PCIe/хост-памяти; кампании использовали их без пересборки.
- Запуск: прямым запуском бинарника; пересборка — из [pcie_bw.cu](pcie_bw.cu) (см. команду выше).
- Готчи / требования: точное происхождение pcie_bw2 (чем отличается от pcie_bw) из кода не восстановимо — в сомнении пересобрать из .cu; бинарники платформо-специфичны (CUDA arch текущей карты).

### pcie_bw_probe.py
[pcie_bw_probe.py](pcie_bw_probe.py)
- Назначение: torch-зонд шины: H2D / D2H / D2D на pinned памяти (256 MiB / 1 GiB / 2 GiB) с параллельным поллером `current_link_speed` / `current_link_width` GPU и root port через sysfs — ловит деградацию линка во время бенча.
- Запуск: `pcie_bw_probe.py` (без аргументов); адреса GPU `0000:01:00.0` и root port `0000:00:01.0` захардкожены.
- Готчи / требования: pinned-хост обязателен (pageable даст ложные полосы); sysfs-пути валидны для конкретного топологического слота — на другой карте поправить адреса.

### size_scaling.py
[size_scaling.py](size_scaling.py)
- Назначение: H2D/D2H bandwidth vs размер буфера (4/16/64/256/1024/4096 MiB) — виден выход на плато Gen5 и overhead мелких передач.
- Запуск: `size_scaling.py` (без аргументов); pinned host + non_blocking копи, CUDA events.
- Готчи / требования: число итераций адаптивно (до ~8 GiB суммарного трафика); на больших буферах нужен запас host RAM ~4 GiB.

### ram_bw.py
[ram_bw.py](ram_bw.py)
- Назначение: зонд host DRAM через torch memcpy: aggregate bandwidth при 1 и 4 потоках (read+write; пиковый read ~ половина aggregate).
- Запуск: `ram_bw.py` (без аргументов); чисто CPU.
- Готчи / требования: GIL ограничивает масштабирование потоков — 4 потока дают нижнюю оценку; память 8 GiB суммарно.

### dram_dir_bw.py
[dram_dir_bw.py](dram_dir_bw.py)
- Назначение: раздельные направления DRAM через numpy: pure read (`np.add.reduce`), pure write (fill), copy (read+write+RFO) по 1 GiB-буферам.
- Запуск: `dram_dir_bw.py` (без аргументов); чисто CPU.
- Готчи / требования: numpy-копия несёт read-for-ownership на load-стороне — «copy» не эквивалентна non-temporal записи; однопоточный, REP=8.

### nvme_thermal_test.sh
[nvme_thermal_test.sh](nvme_thermal_test.sh)
- Назначение: диагностика sustained-read bandwidth + термического троттлинга NVMe: SMART-снапшот до/после, 60 s sequential read QD32 1M-блоки через fio libaio, замер температуры каждые 2 s (hwmon с fallback на smart-log), diff-вердикт.
- Запуск: `./nvme_thermal_test.sh [runtime_sec] [/dev/nvmeXnY ...]` (по умолчанию 60 s и все NVMe-девайсы); сам пере-exec'ается через sudo; сырые логи в /tmp/nvme_thermal_test.*.
- Готчи / требования: требует `fio` и `nvme`-cli; read-only direct I/O — безопасно на смонтированных ФС; термический вывод зависит от hwmon-имён конкретного контроллера. Контекст: [../../topics/prefill-throughput.md](../../topics/prefill-throughput.md).