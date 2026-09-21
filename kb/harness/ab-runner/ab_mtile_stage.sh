#!/usr/bin/env bash
# Serial stage runner for the dense m-tile A/B boots (process-hygiene protocol:
# preflight census, trap-on-EXIT cleanup, hard timeout, SIGTERM->10 s->SIGKILL,
# postflight census). FAILPAT lives in ab_mtile_run.py's watchdog.
# Usage: MTILE_NAME=mtile4|mtile8|mtile16|mtile32|mtile64 ./ab_mtile_stage.sh
set -u
REPO=/media/ai/src/FreeToken
cd "$REPO" || exit 1
NAME=${MTILE_NAME:?MTILE_NAME must be set (mtile4|mtile8|mtile16|mtile32|mtile64)}
SESSION="${NAME}cap"
TASKDIR="$REPO/.tasks/dense-q80-gemm"

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
  # have already moved+renamed their report inside ab_mtile_run.py
  rm -f "$REPO"/report1.nsys-rep "$REPO"/report1.sqlite
  census
}
trap cleanup EXIT

census
echo "[stage] boot + fill + decode + prompts + nsys capture (MTILE_NAME=$NAME, MOE_MTILE=32)"
timeout 2400 .venv/bin/python "$TASKDIR/ab_mtile_run.py" --name "$NAME"
rc=$?
echo "[stage] ab_mtile_run rc=$rc"
kill_tree
sleep 3
# sqlite export for the analyzer; report presence is verified post-stop inside
# ab_mtile_run.py (moved+renamed), the banner is NOT a discriminator (M1)
if [ -f "$TASKDIR/$NAME.nsys-rep" ]; then
  echo "[stage] exporting sqlite"
  /usr/local/bin/nsys export --type sqlite --force-overwrite=true \
    -o "$TASKDIR/$NAME.sqlite" "$TASKDIR/$NAME.nsys-rep"
  echo "[stage] export rc=$?"
else
  echo "[stage] NO REPORT after run - nsys collection silently absent"
  rc=4
fi
census
exit $rc
