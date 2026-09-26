# decode-research/ - машиночитаемые результаты декод-кампании гибридного MoE

Кампания 2026-09-26, ветка vektory79 (HEAD 0ff8495), риг i7-14700KF (8P=cpu0-15
SMT, 12E=cpu16-27, только AVX2) + RTX 5090, модель GLM-5.3-Flash-UD-Q3_K_XL
(GGUF, FTW-директория). Конфиг каждой руки - точная production-команда
пользователя плюс одна дельта (см. [RESULTS.md](RESULTS.md)); метод: boot
`ft serve` (порт 18801) -> fill 65536-токенного префикса -> повторная отправка
того же промпта (radix-reuse) -> декод 768 токенов, temp 0, steady tok/s из
дельт SSE-фреймов.

## Содержимое

- `arm_{base,fetch3,ov0,flagsync0,fetch0,hyb_instr,off_instr,eager}.json` -
  полные итоги руки: ready/fill/decode/teardown, grep ключевых строк лога;
- `stats_{hyb,off,eager}.json` - дампы T1-драйвера инструментированных
  прогонов (aggregate + per-layer miss stats, routing-оракул для eager);
- `sample_per_cpu_base.csv` - 1 Гц per-CPU busy/MHz окна декода базовой руки;
- [CHECKSUMS.md5](CHECKSUMS.md5) - md5 всех файлов выше; копии байт-в-байт
  из рабочего каталога кампании, не редактируются (правило kb/harness).

Сводная таблица - [RESULTS.md](RESULTS.md); харнесс и T1-драйвер -
[../../harness/decode-research/README.md](../../harness/decode-research/README.md);
модель стоимости шага - [../../topics/moe-hybrid-cost-model.md](../../topics/moe-hybrid-cost-model.md).
