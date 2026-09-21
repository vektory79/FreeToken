#!/usr/bin/env bash
# nsys wrapper for the N-split A/B captures (D09 recipe; nsys 2026.1.3:
# `launch` rejects -o/--sample, so capture-from-launch + graph trace).
# Session name comes from NSYS_SESSION (nsplit1cap / nsplit2cap).
# Called by measure.start_server as: ft_nsys_nsplit.sh serve --port ... ("$@" -> ft).
exec /usr/local/bin/nsys launch --session="${NSYS_SESSION:-nsplitcap}" --trace=cuda \
  --cuda-graph-trace=node \
  /media/ai/src/FreeToken/.venv/bin/ft "$@"
