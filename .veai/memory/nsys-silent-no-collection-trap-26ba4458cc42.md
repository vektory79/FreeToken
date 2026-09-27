---
name: "nsys-silent-no-collection-trap"
description: "nsys start/stop can silently no-op (rc=0, no report) when injection wrapper not swapped; verify with mini-probe"
type: project
lastUpdated: 2026-09-27T15:46
lastRecall: 2026-09-27T19:40
---

# nsys silent no-collection trap: rc=0 does NOT imply a report was produced

Found 2026-09-19 during the dense-q80-gemm N-split A/B wave (RTX 5090, nsys 2026.1.3; provenance: ab-nsplit wave deviations D1/D2).

- Incident: the first instrumented boot silently collected NOTHING - the driver forgot the measure.FT wrapper swap, so the nsys injection never attached; `nsys start`/`nsys stop` both no-oped with rc=0 and no error. The wave nearly reported un-instrumented numbers as the instrumented leg.
- Rule: an rc=0 from nsys start/stop proves NOTHING about collection. Before trusting a traced run: (a) verify the report file exists and is non-trivial AFTER `nsys stop` (stop writes the report synchronously into the stop-cwd - grab it post-stop, not from launch-time paths); (b) confirm expected kernel names/launch counts inside the exported sqlite (the T46 liveness pattern); (c) pin the mechanics with a cheap mini-probe before the real run when the wrapper chain is new or was edited.
- Related caveat (same wave): instrumented vs PLAIN boots diverged 2/6 prompts bitwise on ~96-token generations, while same-mode cross-boot agreed 6/6. Bitwise output comparisons must be bounded to the SAME instrumentation mode; do not compare a traced boot's outputs against a plain boot's outputs.

- Banner caveat (2026-09-19, review-b-abnsplit.md M1): the "Collecting data" banner in nsys output is wrapper-config-dependent and is NOT a reliable instrumented-vs-plain discriminator - it was absent from the instrumented nsplit1/2 logs too (0 matches) while present in step-0 logs. The only verified signal is report presence + expected kernel counts in the exported sqlite.

## 2026-09-27 additions (T2 lever-2 probe, vektory79)
- TRAP: --duration counts from LAUNCH, not from readiness: a boot that takes 68-99 s gets killed at the duration mark before decode -> empty/truncated report (attempt 1 died ~72 s in). Pattern that worked: `nsys profile --delay=60` with NO --duration, then let the harness SIGTERM the server. Verify collection with a mini-probe first (rc=0 AND non-empty report containing graph-replay kernels) - the silent-no-collection trap above still applies.
- TRAP: after nsys terminates the app it can RE-PARENT an engine worker that keeps holding VRAM (25.7 GiB observed after attempt 1). Sweep `nvidia-smi --query-compute-apps=pid` after every nsys run and kill the pid directly; probe scripts should do this automatically (the T2 probe script now sweeps compute-apps).
- Practical scale: 468 decode steps identified via ensure_experts instances (42/step); sqlite export ~240 MB from a 77.5 MB report; probe boot decoded at 17.76 tok/s (+6% day class vs the 16.86-17.36 band, structure identical).
