#!/usr/bin/env bash
# nsys wrapper for the dense-q80 step-0 capture (D09 recipe, session denseq80cap).
# Called by measure.start_server as: ft_nsys_dq80.sh serve --port ... ("$@" -> ft).
# nsys 2026.1.3 `launch` rejects -o/--sample; capture-from-launch + graph trace.
exec /usr/local/bin/nsys launch --session=denseq80cap --trace=cuda \
  --cuda-graph-trace=node \
  /media/ai/src/FreeToken/.venv/bin/ft "$@"
