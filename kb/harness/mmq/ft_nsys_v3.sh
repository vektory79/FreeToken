#!/usr/bin/env bash
# v3a probe nsys wrapper (v2 ft_nsys_v2.sh recipe, session name v3aprobe).
# Called by measure.start_server as: ft_nsys_v3.sh serve --port ... ("$@" -> ft).
# nsys 2026.1.3 `launch` rejects -o; capture-from-launch + --cuda-graph-trace=node.
exec /usr/local/bin/nsys launch --session=v3aprobe --trace=cuda \
  --cuda-graph-trace=node \
  /media/ai/src/FreeToken/.venv/bin/ft "$@"
