# Arbitration battery: boot nondeterminism vs tile-4/64 output agreement (24 prompts)

4 plain boots (no nsys, single instrumentation mode), strictly serialized and
reaped, interleaved t4a/t64a/t4b/t64b. Per boot: census -> quiet-box gate -> boot
(FREETOKEN_GGUF_DENSE_MTILE=<tile>, FREETOKEN_GGUF_MOE_MTILE=32, NSPLIT unset ->
committed default 2, winner serving flags, port 18801) -> /ready -> 65,585-token
fill -> 24-prompt greedy battery (p00..p23, reasoning+content verbatim + sha256)
-> SIGTERM/SIGKILL -> census.

## Prefill sanity anchors (median of throughput lines c2..c7)

| boot | tile | median c2..c7 tok/s | anchor | dev % |
|------|------|--------------------:|-------:|------:|
| t4a | 4 | 578.42 | 563.9 | 2.57 |
| t64a | 64 | 808.15 | 792.08 | 2.03 |
| t4b | 4 | 578.83 | 563.9 | 2.65 |
| t64b | 64 | 808.12 | 792.08 | 2.03 |

## Pairwise sha matrices (24 prompts per pair)

| pair | kind | identical | divergent | divergent prompts |
|------|------|----------:|----------:|-------------------|
| t4a~t4b (s1) | same-config | 2 | 22 | p00, p01, p02, p03, p05, p06, p07, p08, p09, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p21, p22, p23 |
| t64a~t64b (s2) | same-config | 24 | 0 | - |
| t4a~t64a (x1) | cross-config | 2 | 22 | p00, p01, p02, p03, p05, p06, p07, p08, p09, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p21, p22, p23 |
| t4b~t64b (x2) | cross-config | 24 | 0 | - |
| t4a~t64b (x3) | cross-config | 2 | 22 | p00, p01, p02, p03, p05, p06, p07, p08, p09, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p21, p22, p23 |
| t64a~t4b (x4) | cross-config | 24 | 0 | - |

## Per-prompt flip matrix (1 = sha mismatch on that pair)

| prompt | s1 | s2 | x1 | x2 | x3 | x4 | flips |
|---|---|---|---|---|---|---|---|
| p00 | 1 | 0 | 1 | 0 | 1 | 0 | 3 |
| p01 | 1 | 0 | 1 | 0 | 1 | 0 | 3 |
| p02 | 1 | 0 | 1 | 0 | 1 | 0 | 3 |
| p03 | 1 | 0 | 1 | 0 | 1 | 0 | 3 |
| p04 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| p05 | 1 | 0 | 1 | 0 | 1 | 0 | 3 |
| p06 | 1 | 0 | 1 | 0 | 1 | 0 | 3 |
| p07 | 1 | 0 | 1 | 0 | 1 | 0 | 3 |
| p08 | 1 | 0 | 1 | 0 | 1 | 0 | 3 |
| p09 | 1 | 0 | 1 | 0 | 1 | 0 | 3 |
| p10 | 1 | 0 | 1 | 0 | 1 | 0 | 3 |
| p11 | 1 | 0 | 1 | 0 | 1 | 0 | 3 |
| p12 | 1 | 0 | 1 | 0 | 1 | 0 | 3 |
| p13 | 1 | 0 | 1 | 0 | 1 | 0 | 3 |
| p14 | 1 | 0 | 1 | 0 | 1 | 0 | 3 |
| p15 | 1 | 0 | 1 | 0 | 1 | 0 | 3 |
| p16 | 1 | 0 | 1 | 0 | 1 | 0 | 3 |
| p17 | 1 | 0 | 1 | 0 | 1 | 0 | 3 |
| p18 | 1 | 0 | 1 | 0 | 1 | 0 | 3 |
| p19 | 1 | 0 | 1 | 0 | 1 | 0 | 3 |
| p20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| p21 | 1 | 0 | 1 | 0 | 1 | 0 | 3 |
| p22 | 1 | 0 | 1 | 0 | 1 | 0 | 3 |
| p23 | 1 | 0 | 1 | 0 | 1 | 0 | 3 |

## Verdict

- same-config (boot noise) flips per pair, avg: **11** (of 24)
- cross-config flips per pair, avg: **11** (of 24)
- quantified nondeterminism rate: 11/24 per boot pair (45.8%)
- tile-invariance holds e2e: cross-config flips do not exceed same-config (boot-noise) flips; the config adds NOTHING beyond boot-to-boot nondeterminism

Note: the sweep's 6-prompt battery used a DIFFERENT prompt set (req_mtile_*.json),
so its prompts-2/3/4 flips cannot be mapped by index here; the comparison is the
flip rate and flip-position instability.

## Divergence classification (whole-answer vs trailing)

| pair | prompt | class | lcp | lcs | len_a | len_b | first divergence |
|------|--------|-------|----:|----:|------:|------:|------------------|
| t4a~t4b | p00 | mid-answer | 430 | 1 | 862 | 834 | reasoning |
| t4a~t4b | p01 | mid-answer | 226 | 1 | 790 | 775 | reasoning |
| t4a~t4b | p02 | mid-answer | 363 | 1 | 665 | 678 | reasoning |
| t4a~t4b | p03 | mid-answer | 140 | 1 | 781 | 764 | reasoning |
| t4a~t4b | p05 | mid-answer | 112 | 1 | 789 | 787 | reasoning |
| t4a~t4b | p06 | mid-answer | 281 | 1 | 741 | 730 | reasoning |
| t4a~t4b | p07 | suffix-flip | 291 | 0 | 769 | 751 | reasoning |
| t4a~t4b | p08 | mid-answer | 131 | 1 | 799 | 787 | reasoning |
| t4a~t4b | p09 | mid-answer | 68 | 1 | 695 | 639 | reasoning |
| t4a~t4b | p10 | mid-answer | 429 | 1 | 691 | 694 | reasoning |
| t4a~t4b | p11 | mid-answer | 173 | 1 | 682 | 667 | reasoning |
| t4a~t4b | p12 | mid-answer | 159 | 1 | 706 | 718 | reasoning |
| t4a~t4b | p13 | mid-answer | 124 | 1 | 708 | 663 | reasoning |
| t4a~t4b | p14 | mid-answer | 311 | 1 | 590 | 589 | reasoning |
| t4a~t4b | p15 | mid-answer | 99 | 1 | 708 | 724 | reasoning |
| t4a~t4b | p16 | mid-answer | 193 | 1 | 680 | 661 | reasoning |
| t4a~t4b | p17 | mid-answer | 75 | 1 | 747 | 766 | reasoning |
| t4a~t4b | p18 | mid-answer | 88 | 1 | 643 | 615 | reasoning |
| t4a~t4b | p19 | mid-answer | 199 | 1 | 741 | 734 | reasoning |
| t4a~t4b | p21 | suffix-flip | 126 | 0 | 359 | 378 | reasoning |
| t4a~t4b | p22 | mid-answer | 118 | 15 | 273 | 230 | reasoning |
| t4a~t4b | p23 | mid-answer | 107 | 1 | 310 | 298 | reasoning |
| t4a~t64a | p00 | mid-answer | 430 | 1 | 862 | 834 | reasoning |
| t4a~t64a | p01 | mid-answer | 226 | 1 | 790 | 775 | reasoning |
| t4a~t64a | p02 | mid-answer | 363 | 1 | 665 | 678 | reasoning |
| t4a~t64a | p03 | mid-answer | 140 | 1 | 781 | 764 | reasoning |
| t4a~t64a | p05 | mid-answer | 112 | 1 | 789 | 787 | reasoning |
| t4a~t64a | p06 | mid-answer | 281 | 1 | 741 | 730 | reasoning |
| t4a~t64a | p07 | suffix-flip | 291 | 0 | 769 | 751 | reasoning |
| t4a~t64a | p08 | mid-answer | 131 | 1 | 799 | 787 | reasoning |
| t4a~t64a | p09 | mid-answer | 68 | 1 | 695 | 639 | reasoning |
| t4a~t64a | p10 | mid-answer | 429 | 1 | 691 | 694 | reasoning |
| t4a~t64a | p11 | mid-answer | 173 | 1 | 682 | 667 | reasoning |
| t4a~t64a | p12 | mid-answer | 159 | 1 | 706 | 718 | reasoning |
| t4a~t64a | p13 | mid-answer | 124 | 1 | 708 | 663 | reasoning |
| t4a~t64a | p14 | mid-answer | 311 | 1 | 590 | 589 | reasoning |
| t4a~t64a | p15 | mid-answer | 99 | 1 | 708 | 724 | reasoning |
| t4a~t64a | p16 | mid-answer | 193 | 1 | 680 | 661 | reasoning |
| t4a~t64a | p17 | mid-answer | 75 | 1 | 747 | 766 | reasoning |
| t4a~t64a | p18 | mid-answer | 88 | 1 | 643 | 615 | reasoning |
| t4a~t64a | p19 | mid-answer | 199 | 1 | 741 | 734 | reasoning |
| t4a~t64a | p21 | suffix-flip | 126 | 0 | 359 | 378 | reasoning |
| t4a~t64a | p22 | mid-answer | 118 | 15 | 273 | 230 | reasoning |
| t4a~t64a | p23 | mid-answer | 107 | 1 | 310 | 298 | reasoning |
| t4a~t64b | p00 | mid-answer | 430 | 1 | 862 | 834 | reasoning |
| t4a~t64b | p01 | mid-answer | 226 | 1 | 790 | 775 | reasoning |
| t4a~t64b | p02 | mid-answer | 363 | 1 | 665 | 678 | reasoning |
| t4a~t64b | p03 | mid-answer | 140 | 1 | 781 | 764 | reasoning |
| t4a~t64b | p05 | mid-answer | 112 | 1 | 789 | 787 | reasoning |
| t4a~t64b | p06 | mid-answer | 281 | 1 | 741 | 730 | reasoning |
| t4a~t64b | p07 | suffix-flip | 291 | 0 | 769 | 751 | reasoning |
| t4a~t64b | p08 | mid-answer | 131 | 1 | 799 | 787 | reasoning |
| t4a~t64b | p09 | mid-answer | 68 | 1 | 695 | 639 | reasoning |
| t4a~t64b | p10 | mid-answer | 429 | 1 | 691 | 694 | reasoning |
| t4a~t64b | p11 | mid-answer | 173 | 1 | 682 | 667 | reasoning |
| t4a~t64b | p12 | mid-answer | 159 | 1 | 706 | 718 | reasoning |
| t4a~t64b | p13 | mid-answer | 124 | 1 | 708 | 663 | reasoning |
| t4a~t64b | p14 | mid-answer | 311 | 1 | 590 | 589 | reasoning |
| t4a~t64b | p15 | mid-answer | 99 | 1 | 708 | 724 | reasoning |
| t4a~t64b | p16 | mid-answer | 193 | 1 | 680 | 661 | reasoning |
| t4a~t64b | p17 | mid-answer | 75 | 1 | 747 | 766 | reasoning |
| t4a~t64b | p18 | mid-answer | 88 | 1 | 643 | 615 | reasoning |
| t4a~t64b | p19 | mid-answer | 199 | 1 | 741 | 734 | reasoning |
| t4a~t64b | p21 | suffix-flip | 126 | 0 | 359 | 378 | reasoning |
| t4a~t64b | p22 | mid-answer | 118 | 15 | 273 | 230 | reasoning |
| t4a~t64b | p23 | mid-answer | 107 | 1 | 310 | 298 | reasoning |

## Census + teardown

- t4a: ready 131.8s, gpu_after_boot 28076 MiB, teardown rc=-15 (graceful SIGTERM by design), leftovers=none (standing llama-swap proxy only)
- t64a: ready 120.2s, gpu_after_boot 28044 MiB, teardown rc=-15, leftovers=none
- t4b: ready 144.3s, gpu_after_boot 28044 MiB, teardown rc=-15, leftovers=none
- t64b: ready 117.1s, gpu_after_boot 28044 MiB, teardown rc=-15, leftovers=none

## Two-mode clustering (the real structure behind the matrices)

The 6-pair matrix is not random: the four boots form TWO output modes.

- Mode B = {t4b, t64a, t64b}: ALL intra-mode pairs bitwise identical 24/24,
  including both cross-config pairs (t4b~t64b, t64a~t4b) and the same-config
  pair t64a~t64b.
- Mode A = {t4a} alone: diverges from every Mode-B boot on the SAME 22 prompts
  (all prompts except p04 and p20).

So the flip matrix {s1,x1,x3 = 22/24; s2,x2,x4 = 0/24} is exactly a two-mode
boot signature. The tile does not select the mode (Mode A and Mode B both
occur under tile 4: t4a=A, t4b=B; both under tile 64: t64a=B, t64b=B), and the
cross-config agreement in Mode B is perfect. This is boot-to-boot
nondeterminism, bimodal, not a tile effect.

- Quantified: per-pair flip count is 0/24 or 22/24 depending on whether the
  two boots landed in the same mode (3/6 pairs same-mode -> 0; 3/6 pairs
  cross-mode -> 22). Per-prompt flip frequency between modes: 22/24 prompts
  (p04 and p20 bitwise stable in all four boots).
- Classification (per divergent pair, identical table above): 20x mid-answer
  (shared head 68-430 chars into reasoning, then greedy-cascade divergence,
  tail lengths differ by ~1-9%), 2x suffix-flip (p07, p21). First divergence
  is ALWAYS inside reasoning_content - never whole-answer-from-first-char,
  never a pure trailing-token truncation. Consistent with one divergent
  sampling decision mid-reasoning cascading under greedy decoding.
- Attempt-1 (aborted wave) note: an earlier t4a boot completed its full 24/24
  battery but the watchdog false-positived on the NORMAL teardown line
  'backend worker freetoken-detokenizer-0 exited' and SIGKILLed the group
  during graceful shutdown (4 leaked sem.mp-, 8 -> 12). Its battery outputs
  already showed the same phenomenon (p21/p22 differ from attempt-2 t4a) and
  were excluded; the wave was relaunched with the watchdog disarmed during
  teardown. census.log keeps attempt-1 lines.
