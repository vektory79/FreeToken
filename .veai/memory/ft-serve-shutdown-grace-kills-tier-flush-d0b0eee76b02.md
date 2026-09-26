---
name: "ft-serve-shutdown-grace-kills-tier-flush"
description: "ft serve tier-flush on llama-swap stop FIXED: worker SIGTERM handler, commit f5e51fd"
type: project
lastUpdated: 2026-09-26T11:17
lastRecall: 2026-09-26T11:12
---

# ft serve: default 10 s SIGINT grace kills the session-tier shutdown flush (HW-confirmed 2026-09-25)

User report: accumulated cache not flushed to L2 when the app is stopped. Root cause P1-H1 CONFIRMED by arm arm_sa.sh on RTX 5090 (GLM-5.3 GGUF, user's exact flags, vektory79 @ e5dd1f2).

Process model: `ft serve` = uvicorn parent + spawned scheduler worker (owns the tier store) + tokenizer/detokenizer (launch.py:170-210). Non-shell mode does NOT detach the process group (launch.py:82) - terminal Ctrl+C hits parent AND worker. The worker's only graceful path is KeyboardInterrupt on SIGINT (launch.py:128-137); NO SIGTERM handler exists in any worker (api_server.py:155).

Kill chain: parent lifespan -> FrontendManager.shutdown -> _shutdown_backend_workers relays SIGINT, p.join(timeout=FREETOKEN_WORKER_SIGINT_GRACE_S=10) (api_server.py:66 default 10, :187), then "SIGINT grace expired; escalating to SIGTERM" (:192-194), reap join 5 s, p.kill() + "SIGTERM grace expired; sending SIGKILL" (:108-113).

Measured (arm_sa): a complete flush of 21 segments / 8.68 GiB took the FULL 10 s at ~600 MB/s - knife-edge vs the default; any larger live L1 guarantees escalation. Stop A (default grace, after ~169k-token fill, l1=3221.9 MiB): escalation line at exactly +10.0 s after "Graceful stop: reaping 2 backend worker(s)"; "session tier: flushed" and "session tier final:" ABSENT; 15/21 journal records survived (6.10 GiB vs 8.68 control); conversation B truncated at ~33% of its tokens; compaction never ran.

Loss semantics: PARTIAL, not empty-dir. `_write_blob` appends blob + fsync BEFORE the journal record (session_tier.py:538-579): SIGKILL between them = bytes durable but record absent -> silently lost on replay; mid-record kill = torn tail dropped (:976-986); crc-framed records written before SIGTERM survive. Reboot after partial loss resumes degraded, not zero (deepest surviving segment gave 99.9% cached on one conversation). SIGKILL mid-flush also skips compact() entirely.

Safe corner: stopping DURING an active streaming request is fine - uvicorn waits for the in-flight response before running the lifespan shutdown, so the 10 s escalation never fires; idle stop is the dangerous case.

Working workaround: FREETOKEN_WORKER_SIGINT_GRACE_S=180 (control run: complete flush, torn_tail=False). Candidate code fixes (NOT yet implemented): adaptive default scaled to L1 bytes, or a real SIGTERM handler in workers. Phase-1/2 arms never hit this because run_arm3.sh waits up to 180 s after SIGINT before escalating.

Why: flush time scales with live L1 (~600 MB/s NVMe), so the default 10 s is only enough for tiny sessions; the user's 400k-reserve GLM sessions exceed it.
How to apply: any manual/HW stop of a tier-enabled ft serve with nontrivial L1 - set FREETOKEN_WORKER_SIGINT_GRACE_S>=180 or expect partial L2 loss; when diagnosing "cache not flushed", first grep for the escalation pair and the absence of "session tier: flushed" / "session tier final:".

Artifacts: .tasks/session-cache-tiering/arm_sa.sh, arm_sa_console.log, arm_sa_stopA_*.json/txt, arm_sa_stopB_* (gitignored).

## FIXED (2026-09-25, uncommitted diff) + llama-swap v258 facts
Code fix (review x1 PASS, no blockers; muse gate failure proven pre-existing via stash round-trip): workers now arm a SIGTERM handler - python/freetoken/server/launch.py `_arm_sigterm_handler` (SIG_IGN both SIGTERM+SIGINT BEFORE raise KeyboardInterrupt -> enters the existing except block; one-shot defense vs the parent's relayed SIGINT landing mid-teardown), armed in `_run_scheduler` (all TP ranks) and `_run_tokenize_worker` (the detokenizer role runs in the SAME tokenize worker process - no separate entry); scheduler except-block and tokenizer/server.py KeyboardInterrupt blocks hardened to SIG_IGN SIGTERM too; parent/uvicorn/shell/daemon handlers untouched. Tests +3 in tests/server/test_lifespan_shutdown.py (unit raise_signal, subprocess group-SIGTERM, SIGTERM-then-relayed-SIGINT reentrancy; gate 8 passed). HW arm_sg (GLM user flags + env=180, setsid group-leader, kill -TERM -pgid): flush COMPLETE - "session tier: flushed 22 live segments" + final line, journal 22 records torn_tail=False, 9.38 GiB, reboot replayed 22, resume cached=95,616 @ 4.79 s; group SIGINT regression arm also flushes (exit 0). SIGTERM process exit code is 143 by design (uvicorn re-raises the stop signal AFTER "Application shutdown complete") - not a defect. Stop-during-load now exits 1 + traceback (matches Ctrl+C-during-load; optional hardening).
llama-swap v258 stop semantics (verified from tag 69cb75a source): launches cmd DIRECTLY (no shell; YAML block scalar -> shlex tokens), SysProcAttr Setpgid:true (child = group leader); stop = syscall.Kill(-pid, SIGTERM) to the WHOLE GROUP (runtime_unix.go:36-44), never SIGINT; SIGKILL to group after unloadTimeout (per-model, default 10 s; SIGTERM->SIGKILL gap); DAEMON shutdown is different: hard 30 s budget, models torn down via ctx-cancel killProcess with parentCancelGraceTimeout = 1 s -> SIGKILL group (llama-swap.go:470-527, process_command.go:78) - quitting the daemon with a loaded model gives the flush only ~1 s; unload the model via API first. cmdStop per-model knob REPLACES the signal (e.g. /bin/kill -INT -- -${PID} would SIGINT the group). checkEndpoint/sendLoadingState/responseHeader do not affect the stop path.
User action items: their llama-swap config needs NO change (env FREETOKEN_WORKER_SIGINT_GRACE_S=180 + unloadTimeout 180 are ample; flush ~10-27 s). P2 (600 s abort) stays client-side: raise the agent SDK timeout.

COMMITTED 2026-09-26: fix = f5e51fd "fix(server): run worker graceful shutdown on SIGTERM" (4 files +157/-5); kb = b5ef0e9 "docs(kb): add ft serve shutdown signals topic and supersede verdict" (new topic kb/topics/ft-serve-shutdown-signals.md + supersede-verdict section in session-cache-tiering.md + index hooks; checks table_lint 0 / links 618-0). Both on vektory79, NOT pushed. User's llama-swap config unchanged and sufficient (env 180 + unloadTimeout 180).
