# Review-B - Candidate A m-tile sweep: hygiene / foreign-serve / bitwise matrix / artifacts

Reviewed 2026-09-20, protocol Review-B (edge cases / integration / evidence integrity /
operability). Read-only; this file is the only artifact written.

Evidence base: orchestration-notes.md, TASK.md, ab-mtile.md, ab-mtile-results.json,
mtile{4,8,16,32,64,4b}_run.json, req_mtile_0.json, ab_mtile_{stage.sh,run.py,analyze.py},
gate-candidate-a.log (header + 16-nodeid list), wave2-candidate-a.md, ab-nsplit.md,
review-b-candidate-a.md (previous pass), boot-log inventory under
.tasks/ft-gguf-serve-tuning/logs/, plus an independent read-only sqlite recount of all
6 nsys exports (mul_mat_q8_0 groups by gridX/gridY) run by this review.

## B1 hygiene compliance - CONFIRMED (one NIT, one hazard note)

- Runner mechanics verified in code: ab_mtile_stage.sh has census() before the run,
  after kill_tree, and inside the trap-on-EXIT cleanup (kill_tree, nsys session
  shutdown, stray report rm, census); hard timeout 2400 s; FAILPAT lives in
  ab_mtile_run.py WATCH = measure.WATCH + "backend worker .* exited"; watchdog
  deadline 2400 s; SIGTERM -> 10 s -> SIGKILL (inside the protocol's 30 s bound).
- Serialized + reaped: one boot per stage invocation; capture timestamps are strictly
  sequential (mtile4 22:26:59, mtile8 22:37:56, [403 kill], mtile16 01:31:46,
  mtile32 01:39:03, mtile64 01:47:22, mtile4b 02:00:53; 7-9 min spacing = boot
  duration, zero overlap); teardown.leftover [] in all 6 run jsons.
- 403 recovery: tiles 4/8 artifacts reused after re-verification (env pins, grid.y,
  sqlite counts all check out in this review); recovery census found the foreign
  llama-swap ft serve (18084, 30 GB, started 23:31) plus its 7 sem.mp- files, not
  signalled per the report; sem.mp- back to the 5 baseline at postflight (transients
  self-resolved, consistent with the auto-unlink precedent).
- NIT: NO census output is archived anywhere - the logs dir holds only server logs
  (+.pid); preflight 22:20, per-boot, recovery 00:40 and postflight 02:0x census
  results exist as prose only. Third recurrence of the wave-1 F1 / wave-2 F3 gap.
- Hazard note: kill_tree's `pkill -f 'ft serve'` is foreign-unsafe in principle - it
  would TERM/KILL any cmdline containing "ft serve", including a foreign server. The
  foreign tree survived per D1/D2 (so its cmdline evidently did not match), but the
  script has no explicit foreign-PID guard taken from the preflight census.

## B2 foreign llama-swap (18084, 30 GB claim) - PARTIALLY REFUTED (MAJOR)

- WHICH boots overlapped (from run.json capture t_start): mtile4 22:26:59 and
  mtile8 22:37:56 completed BEFORE the foreign boot (23:31); the pre-step (gate
  22:19, ptxas re-read + preflight census 22:20) also predates it. Process-level
  overlap: tiles 16/32/64 + discriminator 4b only (01:19-02:03).
- VRAM co-residency REFUTED by the sweep's own artifacts: device-wide free memory is
  4.03-4.15 GiB at graph capture in EVERY boot, pre- and post-foreign alike;
  mtile4b gpu_after_boot_mib 28069 (mine alone; 32607-28069 = 4.5 GB free matches
  the 4.07 GiB init line); mtile4b fill peak 32079/32607 MiB with zero OOM;
  teardown readings 1397-1635 MiB (desktop baseline) after every boot. 30 GB foreign
  + ~28 GB mine cannot fit 32 GB; identical free-memory readings pre/post foreign
  prove the foreign server held NO significant VRAM during any measurement window
  (llama-swap idle swap-out explains the 00:40 30 GB reading vs the 01:19+ boots).
- "persisted through the sweep" is inconsistent with the report's own postflight
  line ("No ft serve / nsys / pytest / ab_mtile survivors"). Reconcilable only if
  the foreign child's cmdline never matched the 'ft serve' census/kill pattern and
  llama-swap unloaded it before 01:19 - not determinable from archived artifacts
  (census outputs unarchived) -> NEED-MORE-DATA. No evidence the sweep signalled it.
- OOM/throttle: none (zero watchdog/FAILPAT hits, zero capture failures, clean
  teardowns, c2..c7 spreads < 1% on every boot).
- Timing anomaly: exactly one - mtile16's graph capture ran 58.30 s/batch with an
  in-line nvcc rebuild of gguf_kernel.cu; every other boot shows "ninja: no work to
  do" at 2.85-4.08 s/batch. CPU-side, finished before the nsys window (capture start
  01:31:46); chunk medians unaffected. Cause undocumented and NOT tile-driven
  (tile8 booted a new tile value with no rebuild); not attributable to the foreign
  serve. Comparability is bounded: the post-rebuild binary reproduces the tile-4
  class (mtile4b median 565.7 vs 563.9, +0.3%) and identical launch geometry (see B5).
- Net: the interaction is SMALLER than documented; the conservative-bias argument
  survives a fortiori. D2(a)'s "~30 GB VRAM co-residency" and the persistence
  phrasing need correction in ab-mtile.md.

## B3 bitwise matrix - reading CONFIRMED; "not a blocker" framing JUSTIFIED

Exact sha table (sha256 over json.dumps(choices, sort_keys=True); req_mtile_0.json
confirms temperature 0, max_tokens 96, stream false):

| prompt | tile4 | tile8 | tile16 | tile32 | tile64 | tile4b |
|--------|-------|-------|--------|--------|--------|--------|
| p0 17*23        | A | A | A | A | A | A |
| p1 Paris        | B | B | B | B | B | B |
| p2 haiku        | C | D | C | E | F | F |
| p3 sky blue     | G | H | I | J | K | K |
| p4 translate    | L | M | N | O | P | P |
| p5 count 1..20  | Q | Q | Q | Q | Q | Q |

(p2: tile4 == tile16; p3/p4: all 5 sweep boots distinct, 4b == 64; verified from the
run jsons + results JSON.)

- vs tile4 identical counts: tile8 3/6, tile16 4/6, tile32 3/6, tile64 3/6 - the
  md's "3-4 of 6" is verified. 4b vs 4 diverges on the same {2,3,4}; 4b vs 64 is
  6/6 IDENTICAL (mtile4b_run.json vs results JSON, byte-equal shas).
- Alternative readings: tile-determinism refuted by 4b vs 4 (same tile, diverges);
  foreign-serve refuted by 4 vs 8 (pre-foreign pair diverges) AND by the absence of
  post-foreign clustering (16/32 differ from 64/4b); radix-state refuted by fresh
  server per boot + identical fill + cached_token 0 on every prompt.
- "six-way disagreement" is overstated (NIT): p2 has 4 distinct values across 6
  boots (4 == 16), p3/p4 have 5 (4b == 64). No clustering of any kind - the honest
  statement the md makes elsewhere.
- N-split precedent (ab-nsplit.md): same-mode cross-boot 6/6 (NSPLIT 1 vs 2, both
  instrumented); cross-mode 4/6 with divergent i = {2,4} - the same borderline
  ~95-token generations that diverge now. v3a's 24/24 was unit-level torch.equal.
- What is DIFFERENT now: (a) the campaign's first SAME-CONFIG cross-boot pair
  (4b vs 4) - the nsplit pair differed by a knob that is bitwise-exact by
  construction, so it never tested same-config reproducibility; (b) committed
  NSPLIT default (mechanism identical to the nsplit2 boot -> not implicated);
  (c) foreign serve (refuted above); (d) the 6-prompt sample is identical to the
  nsplit wave. The nondeterminism SOURCE is not identified by the artifacts; the
  md's pointer (scheduler/atomics in a non-dense path; the hybrid CPU MoE executor
  with 15-17 threads and GPU atomic reductions are the natural suspects) is
  plausible but unproven. Residual: a changeset-occupancy perturbation of an
  order-sensitive non-dense kernel cannot be excluded from this sweep alone -
  bounded by the unit torch.equal tile-invariance battery (dense path) and by
  4b-vs-4 (flips exist at fixed tile, so no tile-causal reading survives).
- Verdict on framing: "open quality question, NOT a blocker" is JUSTIFIED for
  Candidate A: the change's quality contract is the unit bitwise battery (stands),
  the e2e divergence carries no tile signal and appears at fixed tile, and the perf
  conclusions are bitwise-independent. But it devalues single-boot e2e bitwise as a
  gate for ANY future change on this tree (finding F2).

## B4 artifact completeness / machine-readability - CONFIRMED (2 NITs)

- ab-mtile-results.json == ab-mtile.md after the regeneration: all 5 sweep rows
  cross-checked (medians 563.90/658.45/726.37/769.95/792.08; deltas 0/+16.77/+28.81/
  +36.54/+40.46%; chunk times; dense terms 5.5148/3.3714/2.2230/1.6402/1.3855 s;
  kda 67.88/67.31/67.70/67.78/68.12 per chunk at 37.823/23.549/15.453/11.163/9.394 ms
  per half; gridY 2032/1016/508/254/127; decode 13.79/14.36/13.21/14.62/13.53;
  bitwise block 3/6 with p2/p3/p4 divergent). Consistent.
- mtile4b artifacts present (.nsys-rep/.sqlite/_run.json + boot log). Gap: the
  analyzer covers only the 5 sweep boots, so the discriminator had no machine-readable
  liveness - closed by this review's sqlite recount (mtile4b: 1412 dense launches,
  gy = 2032 only, gx = 389 x 276, no gx = 778).
- env pins: mtile4 pins {MOE_MTILE: 32, NSYS_SESSION} with DENSE_MTILE absent (= unset,
  correct); mtile8/16/32/64 add DENSE_MTILE = 8/16/32/64; mtile4b pins DENSE_MTILE = 4
  explicit; NO FREETOKEN_GGUF_KDA_NSPLIT key in any server_env_pins (the /proc filter
  captures every FREETOKEN_* var) -> NSPLIT default 2 everywhere; sqlite confirms zero
  fused gx = 778 launches in all 6 boots.
- gate-candidate-a.log: header complete (command, 2026-09-19T22:19:02+03:00, HEAD
  4505572, branch, tree description incl. the :505 de-indent, FREETOKEN env none);
  2416 collected / 15 deselected / 2401 selected; 16-nodeid list present. It RESOLVES
  the previous Review-B F4: actual composition = import-class (ModuleNotFoundError
  'tests') x11 (dsa_kpool x7 + dsa_pool x2 + glm_dsa ragged[nvfp4] x1 failing at
  test_glm_dsa.py:316 + test_muse_glimmer_parsers x1) + glm_dsa OutOfResources x2 +
  pinned_tensor UVA + qsa_fp8 [1-1-64] + ple boundary = 16. The wave2-candidate-a.md
  attribution "glm_dsa Out-of-Resources = 3 (... [nvfp4])" is WRONG on ragged[nvfp4]:
  the log traceback shows the import failure at :316, the same site wave-1 F4 cited.
  So there was NO class shift: 11 import / 2 OOR exactly as wave 1; failing FILE set
  identical. NIT: correct the wave2 notes.
- NIT (machine-readability): the analyzer's per-class "s_per_chunk" fields hold
  MICROSECONDS (sum(ns)/1e3; e.g. mla_o_proj 559379 = 0.559 s) while top-level
  dense_term.s_per_chunk and the grouped_moe/kda fields are true seconds. Label/units
  mismatch; the md's tables use correct seconds (per-class sums cross-check to
  5.5148 s at tile 4), so no reported number is wrong - but a JSON consumer reading
  classes[*].s_per_chunk as seconds over-reads by 1e6.

## B5 T46 evidence - CONFIRMED (independent sqlite recount by this review)

Raw CUPTI_ACTIVITY_KIND_KERNEL counts of mul_mat_q8_0 by (gridX, gridY), all 6 boots:

| boot | kernel records | gy values present | kda gx=389 | dense total | per chunk |
|------|---------------|-------------------|------------|-------------|-----------|
| mtile4  | 57136 | {2032} only | 274 | 1408 | 348.8 |
| mtile8  | 54644 | {1016} only | 260 | 1338 | 346.4 |
| mtile16 | 52226 | {508} only  | 250 | 1278 | 346.1 |
| mtile32 | 56001 | {254} only  | 266 | 1369 | 348.8 |
| mtile64 | 54785 | {127} only  | 262 | 1346 | 349.9 |
| mtile4b | 57194 | {2032} only | 276 | 1412 | ~348.6 |

- "zero mismatch classes" verified STRONGER than the analyzer's median check:
  DISTINCT gridY per boot is exactly one value (2032/1016/508/254/127; 4b = 2032),
  across every dense class in every boot.
- kda gridX = 389 constant; fused gx = 778 absent everywhere -> the NSPLIT = 2 path
  is live in all 6 boots; halves = 67.3-68.1/chunk vs expected 68.
- Launch counts per chunk constant across tiles (346.1-349.9) - the tile changes
  grid dims, not launch count, as anchored.
- NIT: the absolute anchor is miscalibrated - measured 346-350/chunk vs the ~356
  estimate; the analyzer honestly prints launch_liveness CHECK (outside its +/-4
  window) while the md rounds to "PASS on every boot". The tile-CONSTANCY pin T46
  needs holds exactly; only the 356 constant was an estimate.

## B6 tooling - CONFIRMED

- ab_mtile_stage.sh / ab_mtile_run.py follow the campaign runner patterns: census
  pre/post, trap-on-EXIT, FAILPAT watchdog, hard timeout 2400 s (stage timeout +
  watchdog deadline), SIGTERM -> 10 s -> SIGKILL, one boot per stage invocation with
  kill_tree + census between boots (serialization by construction). The watchdog is
  an in-process thread on the Popen handle - it sidesteps the MEASPID-export trap
  entirely.
- The nsys --force -> --force-overwrite=true fix did not invalidate boot 1:
  mtile4.sqlite holds 57,136 kernel records and the analyzer's full tile-4 class
  table derives from it; capture started/stopped rc = 0, report generated and moved.
- NIT: stage.sh's usage comment lists mtile4|8|16|32|64 only (run.py accepts mtile4b).

## B7 scope - CONFIRMED clean

- Sweep-wave modifications, exhaustively: (1) layers/moe.py :505 docstring de-indent
  - docstring-only fix of the previous Review-B F5 NIT, inside the already-reviewed
  changeset; (2) ab_mtile_analyze.py dense-term units fix - .tasks script; (3)
  ab_mtile_stage.sh nsys flag fix - .tasks script; (4) the mtile4b extra boot -
  documented deviation with a stated purpose. Nothing else evidenced.
- Out-of-scope list untouched: no fetch-copy work (fast_index_copy_multi only
  measured), no prefill-overlap redesign (boot logs show the known per-partition
  disable warnings, config as-is), no MoE re-tuning, no KV/pool changes; all boots
  ran fixed winner flags (memory-ratio 0.85, max-prefill-length 8191,
  max-running-requests 1, port 18801).

## Findings

- F1 MAJOR (doc correction + NEED-MORE-DATA residue): the foreign-serve narrative in
  ab-mtile.md D1/D2 overstates the interaction - "~30 GB VRAM co-residency" is
  refuted by the sweep's own device-wide free-memory/peak evidence (identical
  4.03-4.15 GiB free pre/post foreign; 28.07 GB after boot = mine alone; fill peak
  32079/32607 MiB with zero OOM), and "persisted through the sweep" contradicts the
  postflight's own "no ft serve survivors". The conservative-bias conclusion is
  unaffected (a fortiori). Correct D1/D2; the child's actual fate stays
  NEED-MORE-DATA until census raw outputs exist somewhere.
- F2 MAJOR (methodology, record): single-boot e2e greedy bitwise is proven
  boot-to-boot flaky on this tree (3/6 prompts diverge at FIXED tile, pre-foreign);
  it must not serve as a pass/fail gate for future waves. Quality gates stay at
  unit level (torch.equal batteries), or e2e gates get redefined as N boots per leg
  with majority hashes.
- F3 NEED-MORE-DATA: one-off gguf_kernel.cu JIT rebuild at the mtile16 boot (58 s
  capture; cause undocumented, not tile-driven, no source edit evidenced);
  binary-comparability bounded by mtile4b's perf (+0.3% vs tile-4 class) and
  identical launch geometry.
- F4 NIT: analyzer per-class s_per_chunk holds microseconds under a seconds label.
- F5 NIT: "six-way disagreement" overstated (p2 is 4-way distinct incl. 4b; p3/p4
  5-way; tile4 == tile16 on p2).
- F6 NIT: wave2-candidate-a.md misclassifies glm_dsa ragged[nvfp4] as OOR (log:
  import-class at :316); previous-Review-B F4 resolves as NO class shift, same file
  set as wave 1.
- F7 NIT: dense-launch anchor ~356 vs measured 346-350 (analyzer prints CHECK; md
  rounds to PASS).
- F8 NIT: census outputs / stage stdout / recovery census unarchived (third
  recurrence of the artifact-recording gap); stage.sh usage omits mtile4b;
  kill_tree pattern is foreign-unsafe in principle.

## Overall verdict

Sweep VALID. The tile-64 winner (+40.46% same-mode prefill) stands on clean,
serialized, machine-verified evidence; hygiene was substantially compliant; the
foreign-serve interaction was smaller than documented; the bitwise finding is
correctly read as boot-to-boot nondeterminism, not a tile effect.

## Recommendation on the quiet-box battery

Ride as a RECORDED OPEN QUESTION - do not gate the user decision (commit Candidate
A, flip the production default) on a quiet-box battery. Grounds: the change's
quality contract is the unit-level torch.equal tile-invariance battery, which
stands; the e2e divergence exists at fixed tile (4b vs 4) and pre-foreign (4 vs 8),
so it is neither tile- nor foreign-caused; the divergent set {2,3,4} is the
borderline-generation set that also diverged cross-mode in the nsplit wave.
OPTIONAL cheap arbitration (~4 boots, ~40 min, zero code change) IF the user wants
the e2e gate rehabilitated or the nondeterminism dated: 2x tile-64 + 2x tile-4 on a
quiet box (foreign absent), census logged to a file this time - it measures the
same-config flip rate and whether tile-64 vs tile-4 diverge beyond the borderline
set. Regardless: fix F1/F5/F6 wording in ab-mtile.md / wave2-candidate-a.md before
those artifacts are cited further, and record F2 in the campaign notes.

