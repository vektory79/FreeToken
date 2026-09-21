#!/usr/bin/env bash
# Serial stage runner for the N-split A/B boots (process-hygiene protocol:
# preflight census, trap-on-EXIT cleanup, hard timeout, SIGTERM->10 s->SIGKILL,
# postflight census). FAILPAT lives in ab_nsplit_run.py's watchdog.
# Usage: NSPLIT_NAME=nsplit1|nsplit2 ./ab_nsplit_stage.sh
set -u
REPO=/media/ai/src/FreeToken
cd "$REPO" || exit 1
NAME=${NSPLIT_NAME:?NSPLIT_NAME must be set (nsplit1 | nsplit2)}
SESSION="${NAME}cap"

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
  pkill -KILL -f 'ft_nsys_nsplit' 2>/dev/null
}
cleanup() {
  echo "[stage] EXIT cleanup"
  kill_tree
  /usr/local/bin/nsys shutdown --session="$SESSION" >/dev/null 2>&1
  # stray aborted-iteration reports land in CWD (D09 trap); successful runs
  # have already moved+renamed their report inside ab_nsplit_run.py
  rm -f "$REPO"/report1.nsys-rep "$REPO"/report1.sqlite
  census
}
trap cleanup EXIT

census
echo "[stage] boot + fill + nsys capture (NSPLIT_NAME=$NAME, MTILE=32)"
timeout 2400 .venv/bin/python .tasks/dense-q80-gemm/ab_nsplit_run.py --name "$NAME"
rc=$?
echo "[stage] ab_nsplit_run rc=$rc"
kill_tree
sleep 3
census
exit $rc
