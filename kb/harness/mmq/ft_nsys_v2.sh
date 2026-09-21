#!/usr/bin/env bash
# v2 probe nsys wrapper, reconstructed from step0-profile.md (the original
# ft_nsys.sh was temporary and deleted after Step 0). Called by
# measure.start_server as: ft_nsys_v2.sh serve --port ... ("$@" goes to ft).
# Session name is HARDCODED (must match v2ab_probe.py SESSION).
# Lessons from the first probe attempts (see v2-ab.md):
#   - nsys 2026.1.3 `nsys launch` rejects -o (report name defaults to
#     report<n>.nsys-rep, written by `nsys stop` into the stop cmd's cwd)
#   - mid-run `nsys start` on hot CUDA contexts did not engage device-activity
#     collection (boot-phase data only) -> capture from launch instead
#   - default graph trace mode records a whole graph as one activity -> need
#     --cuda-graph-trace=node to see moe_vec_q inside decode graphs
exec /usr/local/bin/nsys launch --session=v2probecap --trace=cuda \
  --cuda-graph-trace=node \
  /media/ai/src/FreeToken/.venv/bin/ft "$@"
