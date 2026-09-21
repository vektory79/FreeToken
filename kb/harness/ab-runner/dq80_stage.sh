#!/usr/bin/env bash
# Serial stage runner for the dense-q80 step-0 capture (process-hygiene:
# preflight census, trap-on-EXIT cleanup, FAILPAT is in step0_run.py's
# watchdog, hard timeout, SIGTERM->SIGKILL escalation, postflight census).
set -u
REPO=/media/ai/src/FreeToken
cd "$REPO" || exit 1

census() {
  echo "--- census $(date +%H:%M:%S)"
  pgrep -af '[f]t serve|[n]sys' || echo "  (no ft serve / nsys)"
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
  echo "  sem.mp-: $(ls /dev/shm 2>/dev/null | grep -c '^sem\.mp-')"
}
kill_tree() {
  pkill -TERM -f 'ft serve' 2>/dev/null
  sleep 10
  pkill -KILL -f 'ft serve' 2>/dev/null
  pkill -KILL -f 'ft_nsys_dq80' 2>/dev/null
}
cleanup() {
  echo "[stage] EXIT cleanup"
  kill_tree
  /usr/local/bin/nsys shutdown --session=denseq80cap >/dev/null 2>&1
  # stray aborted-iteration reports land in CWD (D09 trap)
  rm -f "$REPO"/report1.nsys-rep "$REPO"/report1.sqlite
  census
}
trap cleanup EXIT

census
echo "[stage] boot + fill + nsys capture (env FREETOKEN_GGUF_MOE_MTILE=32)"
timeout 2400 .venv/bin/python .tasks/dense-q80-gemm/step0_run.py
rc=$?
echo "[stage] step0_run rc=$rc"
kill_tree
sleep 3
census
exit $rc
