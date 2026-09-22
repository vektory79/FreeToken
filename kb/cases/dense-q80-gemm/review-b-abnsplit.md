# Review-B: dense-q80-gemm A/B wave (NSPLIT 1 vs 2) - hygiene / deviations / artifacts / operability

Read-only review. Sources: orchestration-notes.md (sections 1, 2, 3, 6), TASK.md
(Replanning + Replan wave 1), wave1-notes.md (sections 5, 7 judge-on-NET rule),
ab-nsplit.md, ab-nsplit-results.json, nsplit1_run.json, nsplit2_run.json,
nsplit1_plain_boot_run.json, req_ns_{0,4,5}.json, ab_nsplit_stage.sh,
ab_nsplit_run.py, ab_nsplit_analyze.py, ft_nsys_nsplit.sh, ft_nsys_dq80.sh,
boot logs .tasks/ft-gguf-serve-tuning/logs/nsplit{1,2}.log(.pid), nsplit1_plain.log,
step-0 logs step0_nsys.log / dq80_nsys.log. Binary sizes verified by direct ls.

## Per-item verdicts

### B1. Hygiene compliance - CONFIRMED
- stage.sh: preflight `census()` (pgrep '[f]t serve|[n]sys' + nvidia-smi + sem.mp- count),
  trap-on-EXIT cleanup (kill_tree + nsys session shutdown + stray report rm + re-census),
  `timeout 2400` around the driver, SIGTERM -> 10 s -> SIGKILL. FAILPAT lives in
  ab_nsplit_run.py watchdog (declared split in the stage.sh header): WATCH =
  measure.WATCH.pattern + "|backend worker .* exited", 2 s poll, exit rc 3 on hit,
  separate hard deadline 2400 s. MEASPID export trap not applicable (python driver
  passes the proc handle directly).
- Serialization/reaping: one boot per stage invocation; timestamps strictly ordered
  (plain ready 18:37:09 -> boot-1 ready 18:59:17 -> boot-2 ready 19:09:39);
  teardown.leftover = [] in all three run jsons; teardown.gpu_mib back to baseline.
- Census: VRAM pre 1369-1399 MiB, post 1394 MiB (run jsons + md agree); sem.mp- = 5
  throughout (the documented IDE baseline). No strays; probe leftover
  REPO/report1.nsys-rep removed (stage.sh cleanup + run.py stale-report purge).
- Boot logs + .pid: nsplit1.log/.pid, nsplit2.log/.pid present. nsplit1_plain.log
  present but has NO .pid sidecar (N3).
- Discrepancy: the review brief quotes "VRAM 1469 MiB" - no artifact shows 1469;
  artifacts say 1369-1399 -> 1394 (typo in the brief, not in the wave, N3).

### B2. Deviation D1 (nsys silent no-collection re-run) - CONFIRMED, one MAJOR wording defect
Operative chain verified sound:
- Plain boot ran uninstrumented: nsplit1_plain_boot_run.json capture block =
  start_rc 0, stop_rc 0, stop_err "", report_move_error "no report file found after
  shutdown". Exactly the silent rc=0 no-op signature.
- Fix is in code: ab_nsplit_run.py swaps measure.FT = ft_nsys_nsplit.sh at module
  load with a comment naming the failure mode.
- Re-run instrumented: nsplit1.nsys-rep 4,237,023 B + nsplit1.sqlite 11,636,736 B;
  kernel-class tables in results.json derive from the sqlite - collection is real.
- Same tree / same flags argument holds: ServerArgs blocks identical across plain
  and instrumented boot-1 (winner flags, port 18801); server_env_pins = MTILE=32
  both, knob absent both; only NSYS_SESSION/wrapper differ. KV auto-size wobble
  (500928 vs 500736 tokens) is free-VRAM-dependent auto sizing, not a flag change.
- Prefill gap plain 553.81 vs instrumented 547.24 (+1.2%) is presented as a
  "reference" but never framed as instrumentation overhead; same for decode
  15.46 vs 13.35/13.61 (N2). The +2.61% e2e verdict is computed between two
  instrumented boots (mode-consistent), so the verdict itself is unaffected.
- MAJOR M1: deviations #1 cites the discriminator "missing 'Collecting data'
  banner (present exactly once in both step-0 logs, absent there)". Verified:
  banner IS present once in step0_nsys.log:61 and dq80_nsys.log:62 and absent from
  nsplit1_plain.log - but it is ALSO absent from the instrumented nsplit1.log and
  nsplit2.log (grep -c = 0 both). Banner presence is therefore NOT a valid
  "instrumented" discriminator for this wrapper config; the md wording invites that
  reading and its own artifacts refute it. Doc-only fix: anchor D1 detection on
  report presence (rc=0 + no report file), keep the step-0 banner observation as
  historical context, and add the "silent rc=0 needs report-presence verification"
  note for future waves.

### B3. Deviation D4 (cross-MODE bitwise divergence) - CONFIRMED, correctly bounded
From the three run jsons (sha256 over choices JSON, sort_keys):
- plain (NSPLIT=1, no CUPTI) vs instrumented boot-1 (NSPLIT=1, CUPTI): 4/6 equal;
  diverged are i=2 (46382997... vs fe5fba17...) and i=4 (75875a17... vs 717dc9da...),
  both ~95-token creative generations. 2/6 cross-MODE, same knob value.
- instrumented boot-1 (NSPLIT=1) vs instrumented boot-2 (NSPLIT=2): 6/6 equal
  (results.json comparison.bitwise_diff.per_prompt, all equal:true, shas match the
  run jsons). Cross-NSPLIT, same mode.
So the divergence sits exactly on the MODE boundary, not on NSPLIT - the data says
what the md says. The md bounds the bitwise contract to same-mode ("both
CUPTI-instrumented boots") and labels the plain-boot 4/6 an out-of-scope
observation bounding cross-MODE comparisons only. Correct. Nit: the machine twin
covers only the instrumented pair; the 4/6 cross-mode comparison is md + run-json
only (recomputable; N4).

### B4. Artifact completeness + machine readability - CONFIRMED (two nits)
- ab-nsplit-results.json <-> ab-nsplit.md: every headline number matches (547.24 /
  561.50, +2.61%; chunks 14.853 / 14.476; dense 6.0947 -> 5.5481, recovery 0.5466;
  kernel-only 3.1466 -> 2.6347 = 92.8% of the 0.552 s L2-excess bound; per-half
  37.829 ms, per-pair 76.584 ms = 21.65 TF/s; MoE 4.297/4.3854; fetch 3.0577/3.1106;
  rest 1.3964/1.4251; busy 14.846/14.469 sums close exactly; decode 13.35/13.61;
  span copies 0.01919 s = 19.2 ms; T44 deltas +0.14% / -1.65%). Chunk-sum identity
  verified for both boots.
- nsplit1/2 .nsys-rep 4.24/4.26 MB and .sqlite 11.6/11.5 MB, both present,
  non-trivial, timestamps consistent with the boots.
- Env provenance: server_env_pins MTILE=32 both; FREETOKEN_GGUF_KDA_NSPLIT=2 in
  boot-2 only, absent in boot-1 and plain; NSYS_SESSION per boot. Graph-capture and
  nsys capture blocks present per boot.
- req_ns_{0..5}.json match PROMPTS exactly (spot-checked 0, 4, 5: content,
  max_tokens 96, temperature 0, stream false).
- N1: the md census sentence "1356 launches / 4.004 chunk-eq (= 322 + 34 extra
  split launches per chunk)" is inconsistent with its own class table: classes sum
  to 348.61/chunk -> ~1396 launches at 4.004 chunk-eq (boot-1's 1277 matches its
  class sum 316.26 exactly). Likely 1356-vs-1396 transcription slip; the gloss
  "322 + 34" is also loose (316.3 + 33.7 measured). Liveness numbers (34.18 -> 67.92)
  are self-consistent and unaffected. Doc-only fix.

### B5. Decode numbers - CONFIRMED (framing nit)
- 13.35 (boot-1) vs 13.61 (boot-2), radix HIT 65536 both (decode.cached_token);
  p50 13.48/13.69. Same-mode delta +1.9%, well inside the 13.5-14.7 class spread -
  "flat" is sound, and the structural argument (max-running-requests=1 -> only the
  bs=1 graph exists; bs=1 stays MMVQ, so the split cannot enter any captured graph
  in this config) makes decode knob-independent by construction. The md states this
  correctly.
- Graph re-capture IS evidenced: graph_capture_lines in both run jsons ("Start
  capturing CUDA graphs with sizes: [1]" + bs=1 capture progress) - fresh boot =
  fresh capture confirmed. The stronger wave-1 worry ("knob=2 graph bakes the
  split") is moot here for the structural reason above; the caution stays valid for
  production bs>6 graphs, as the md notes.
- N2: the same-session plain boot decoded at 15.46 (above the class ceiling), which
  is never connected to the obvious explanation - CUPTI instrumentation depresses
  decode (and prefill) ~1-2 tok/s. The noise-band sentence for 13.35 ("daytime
  noise") ignores a same-evening plain datum on the other side of the class. The
  A/B decode verdict is unaffected (same-mode comparison); add one framing line.

### B6. Tooling reuse + script quality - CONFIRMED
- ab_nsplit_stage.sh: census / trap-on-EXIT / hard timeout / SIGTERM->10 s->SIGKILL
  all present; matches the process-hygiene protocol.
- ab_nsplit_run.py: FAILPAT watchdog (measure.WATCH + backend-worker-exited
  variant), hard deadline 2400 s, nsys start-after-2 / stop-after-6 throughput
  lines (D09 window), server env pins read from /proc/<pid>/environ (validity pin),
  stale-report purge, synchronous report grab after stop (D2 mechanics), teardown,
  run json written even on failure (finally block), rc 3 on FAILPAT.
- ft_nsys_nsplit.sh: same shape as the proven ft_nsys_dq80.sh (D09 launch + graph
  trace); measure.py reused unchanged except the declared measure.FT swap.
- Nits: watchdog resolves --name via sys.argv.index hack (argparse already has it);
  nsys shutdown rc=1 recorded in all three runs, unexplained (benign; cancel path
  also covered).

### B7. Out-of-scope list - CLEAN
The wave changed no code (the 4-file kda/gguf/test changeset predates it and
touches only the in_proj path + tests). MTILE=32 is the fixed v3a anchor, not a
MoE re-tune. Fetch copies, prefill overlap and KV/pool were only measured or
observed (overlap disabled on all three partitions = pre-existing campaign state;
KV token allocation 500928 vs 500736 is auto sizing from free VRAM, both 2.98 GiB).
Nothing crosses the "Out of scope (separate tasks)" list.

## Findings

- M1 (MAJOR, doc-only, B2/D1): the "missing Collecting data banner" discriminator
  in ab-nsplit.md deviations #1 is contradicted by the wave's own instrumented
  logs (banner absent there too). Reword to the operative evidence: rc=0 start/stop
  + "no report file found after shutdown" (plain) vs report+sqlite present
  (re-runs); note the banner is not a reliable discriminator for this wrapper.
- N1 (NIT, B4): fix the NSPLIT=2 census sentence (1356 -> ~1396, or drop the
  "= 322 + 34" gloss).
- N2 (NIT, B2/B5): acknowledge instrumentation overhead explicitly (prefill
  553.81 -> 547.24, decode 15.46 -> 13.35/13.61, plain above class ceiling); the
  same-mode verdicts do not change.
- N3 (NIT, B1): brief quotes VRAM 1469 MiB; artifacts say 1369-1399 -> 1394
  (typo lives in the review brief, not the wave). nsplit1_plain.log lacks a .pid
  sidecar.
- N4 (NIT, B3/B4): results.json bitwise twin covers only the instrumented pair;
  consider adding the plain-boot shas for a complete machine record.
- N5 (NIT, B6): sys.argv watchdog lookup; unexplained benign nsys shutdown rc=1.
- No FP findings; no BLOCKER.

## Overall verdict

PASS - the A/B wave is valid and its verdicts stand: T44 gate PASS, T46 liveness
34.18 -> 67.92 launches/chunk with per-pair 76.584 ms = 21.65 TF/s, dense-term
recovery 0.5466 s/chunk above the judge-on-NET window in the good direction, e2e
+2.61% prefill, decode flat and structurally knob-independent at bs=1, bitwise
6/6 same-mode cross-boot with D4 correctly bounded to cross-MODE. Hygiene
compliant, no out-of-scope crossing, artifacts complete and machine-readable.
The single MAJOR is a one-paragraph doc reword in ab-nsplit.md deviations #1
(M1); N1-N5 are cheap polish. No re-measurement needed.
