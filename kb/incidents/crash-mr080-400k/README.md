# ft-serve-crash-mr080-400k

User report: `ft serve` with --memory-ratio 0.80 --kv-reserve-tokens 400000
--kv-cache-dtype fp8 --max-prefill-length 8191 (winner recipe of 2026-09-20)
crashes at boot. No error text provided by user -> reproduce + root-cause.
Artifacts: error-report.md, serve.log, runner stdout.

## Status and next step for a fresh session

Status: OPEN - root cause and repro recipe are pinned in
[error-report.md](error-report.md); the recipe is config-infeasible-by-margin
on this desktop (needs >= 29.28 GiB free at boot), no fix landed yet.

Next step: reproduce on this rig with the ready runner
[crash-mr080-repro.sh](../../harness/repro/crash-mr080-repro.sh)
(usage card: [harness/repro/README.md](../../harness/repro/README.md)),
capture the server log and runner stdout, and compare the boot traceback
against the archived first reproduction ([serve.log](serve.log),
[runner.out](runner.out)).