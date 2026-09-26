---
name: "ft-serve-600s-abort-client-timeout"
description: "ft serve ~600s decode abort = client SDK default timeout; 'Aborting request' log + 0-token ring row discriminate"
type: project
lastUpdated: 2026-09-25T22:52
lastRecall: 2026-09-26T18:28
---

# ft serve: decode aborting after ~10 minutes is a CLIENT-side timeout, not a server bug (HW-confirmed 2026-09-25)

User report: a decode running longer than ~10 min aborts and the driving agent stops (e.g. "write the full Java+Vulkan triangle program"). CONFIRMED P2-H1: no 600 s mechanism exists anywhere in the server request path (grep for 600/10*60/deadline/ttl across server/, scheduler/, engine/, moe/, daemon/ is clean); the culprit is the CLIENT - openai-python/anthropic-python SDKs default to a 600 s request timeout.

Server-side detection signature for a streaming client disconnect (verified live with curl --max-time 120 as a stand-in):
- exactly ONE WARNING: "Aborting request for user %s" (api_server.py:458-468 via stream_with_cancellation; openai_api.py:205-208); detection < 1 s after the cut.
- The INFO line "Client disconnected for user" does NOT appear in this path: uvicorn/Starlette cancels the stream task on http.disconnect BEFORE the in-loop request.is_disconnected() check fires; the `except CancelledError -> abort_user` handler logs only the WARNING.
- /v1/requests row: duration_ms ~= the client timeout (measured 119432 for a 120 s cut), completion_tokens=0, prompt_tokens=0, stream=true, status=200, error=null.
- Server stays alive (real chat POST -> 200 within ~3 s); generation is aborted via AbortMsg -> ErrorReplyMsg("request aborted") (scheduler.py:940-946).
- Non-stream (generate_full) never polls disconnect: decode runs to completion server-side; the client just sees its own timeout, with no special server log lines at all.

Discriminator vs a server-side abort (P2-H2, refuted in this case): "CPU MoE flag-handshake watchdog fired" (moe/cpu_executor.py:927-993, 10 s doorbell), "Backend supervisor: ...", "Backend worker is gone and cannot be restarted" + dead server / connection reset. Also see ft-serve-test-and-e2e-gotchas for the backend-death hang pattern.

Fix is client-side: raise or disable the SDK timeout (and/or stream so chunks keep flowing). No server change needed.

Why: any agent harness on stock SDK defaults will kill every >10 min generation and misread it as a server fault.
How to apply: when a user reports time-correlated (~600 s) request deaths, first check the server log at the abort second for "Aborting request for user" + the /v1/requests row - their presence with a LIVE server proves the client cut the connection.
