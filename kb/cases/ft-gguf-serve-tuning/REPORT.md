# ft serve GGUF GLM-5.3-Flash: A/B campaign (RTX 5090, 2026-09-16)

Baseline free before load (this session, llama-swap OFF): [1m[2026-09-16|21:28:15|core|rank=0][0m [32mINFO    [0m Free memory before loading model: 29.54 GiB

Production delta note: llama-swap idle wrapper historically holds ~1.27 GiB VRAM -> subtract from free-after-init when comparing with production boots.

## Boot+prefill+decode matrix

| config | ready_s | slots g1/g2/g3 | free-after-init | prefill tok (prompt) | chunks | chunk1 tok/s | prefill steady tok/s | decode steady tok/s | decode min inst | notes |
|---|---|---|---|---|---|---|---|---|---|---|

| BASE (user cmd) | - | 295/288/288 | 2.89 GiB | - | - | - | - | - | - | boot: None |

| BASE + --max-running-requests 1 | 94.1 | 521/288/288 | 2.90 GiB | 49 | 17 | 0.02 | 0.1 | 0.21 | 0.0 | ok |

| mr1 + chunk 6144 | 108.4 | 557/288/288 | 2.96 GiB | 4145 | 11 | 98.23 | 402.1 | - | - | ok |

| mr1 + chunk 8191 | 98.2 | 557/288/288 | 2.96 GiB | - | - | - | - | - | - | prefill err: TimeoutError: timed out; WATCHDOG pattern in prefill |

| mr1 + chunk 8191 + ratio 0.85 | 102.2 | 410/288/288 | 4.11 GiB | 561 | 9 | 110.41 | 487.1 | - | - | ok |

| winner 8191/0.85 full (decode check) | 99.1 | 410/288/288 | 4.11 GiB | 561 | 9 | 110.48 | 486.6 | 13.58 | 7.04 | ok |

| load A/B: plain boot | 104.2 | 527/288/288 | 2.92 GiB | - | - | - | - | - | - | ok |

| load A/B: FREETOKEN_BANK_CUDA_ALLOC=1 | 132.5 | 527/288/288 | 2.92 GiB | - | - | - | - | - | - | ok |


## Boot-only plan inspections

| config | ready_s | slots g1/g2/g3 | free-after-init | prefill tok (prompt) | chunks | chunk1 tok/s | prefill steady tok/s | decode steady tok/s | decode min inst | notes |
|---|---|---|---|---|---|---|---|---|---|---|

| boot: mr1 + ratio 0.87 | 108.2 | 497/288/288 | 3.54 GiB | - | - | - | - | - | - | ok |

| boot: mr1 + ratio 0.85 | 114.2 | 438/288/288 | 4.15 GiB | - | - | - | - | - | - | ok |

| boot: mr2 (ratio 0.89) | 112.8 | 502/288/288 | 2.96 GiB | - | - | - | - | - | - | ok |


## Deep-fill ladder (winner config)

(no ladder data)
