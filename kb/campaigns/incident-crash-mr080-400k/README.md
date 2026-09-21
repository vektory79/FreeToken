# ft-serve-crash-mr080-400k

User report: `ft serve` with --memory-ratio 0.80 --kv-reserve-tokens 400000
--kv-cache-dtype fp8 --max-prefill-length 8191 (winner recipe of 2026-09-20)
crashes at boot. No error text provided by user -> reproduce + root-cause.
Artifacts: error-report.md, serve.log, runner stdout.
