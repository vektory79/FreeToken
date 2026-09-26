# Результаты: 5-рукавный A/B + T1-декомпозиция декода гибридного MoE (2026-09-26)

Измерено 2026-09-26, ветка vektory79 (HEAD 0ff8495), риг i7-14700KF (8P=cpu0-15 SMT,
12E=cpu16-27, только AVX2) + RTX 5090, модель GLM-5.3-Flash-UD-Q3_K_XL (GGUF,
FTW-директория). Конфиг - точная production-команда пользователя (FTW, 0.82/400k,
kv fp8, tier 10/50, --max-prefill-length 8191, --moe-cpu-threads 16, hybrid);
декод: 65536-токенный radix-reuse (повторная отправка того же промпта), 768 max
tokens, temp 0; steady tok/s из дельт SSE-фреймов.

| arm | config delta | tok/s @64k | ms/step |
|---|---|---|---|
| base | (user config) | 16.29 | 61.4 |
| fetch3 | --moe-hybrid-max-fetch 3 | 16.14 | 62.0 |
| ov0 | FREETOKEN_HYBRID_OVERLAP=0 | 12.46 | 80.3 |
| flagsync0 | FREETOKEN_CPU_MOE_FLAG_SYNC=0 | 12.77 | 78.3 |
| fetch0 | --moe-hybrid-max-fetch 0 | 9.87 | ~101 |
| offload | --moe-strategy offload | 13.72 | 72.9 |
| eager | --cuda-graph-max-bs 0 | 15.84 | 61.7 |
| hyb_instr | base + T1 driver | 15.82 | (span 59.1 ms) |
| off_instr | offload + T1 driver | (13.5-13.7 band) | (span 73.6 ms) |

stats_{hyb,off,eager}.json: aggregate/per-layer miss stats, routing oracle for
eager; sample_per_cpu_base.csv: 1 Hz per-CPU busy/MHz of base decode window.

Интерпретация, аддитивная модель шага и вердикты -
[../../topics/moe-hybrid-cost-model.md](../../topics/moe-hybrid-cost-model.md);
харнесс - [../../harness/decode-research/README.md](../../harness/decode-research/README.md).
