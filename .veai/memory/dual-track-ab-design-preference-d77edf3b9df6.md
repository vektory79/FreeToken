---
name: "dual-track-ab-design-preference"
description: "User wants A/B matrices in both modes: equalized fair A/B AND as-configured auto hardware-capability numbers"
type: feedback
lastUpdated: 2026-10-30T18:02
---

# A/B design: always give both the equalized and the as-configured track

Rule: when designing a comparison campaign (e.g. KV-quant bf16/fp8/nvfp4, 2026-11), produce
TWO result matrices: (1) fair A/B with confound knobs equalized across legs (e.g. explicit
--moe-cache-size = min of auto plans), and (2) as-configured/auto runs showing what the
hardware delivers per leg, including capacity ceilings (max tokens that boot per leg).

**Why:** kv-quant-ab case - I proposed only the equalized design; user accepted it as
"interesting and valuable" but immediately added: "кроме этого хочется иметь цифры того, на
что способно моё оборудование... без фиксации этого параметра тоже интересно". Equalized
isolates the mechanism; auto shows the practical payoff. Neither alone answers the user.

**How to apply:** any A/B plan where the varied flag also shifts a resource budget (dtype,
quant, cache size, threads): present both tracks and label which numbers come from which.
Capacity ladders (rising reserve until fail-fast) belong to the auto track.
