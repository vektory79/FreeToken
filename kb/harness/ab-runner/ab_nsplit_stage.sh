#!/usr/bin/env bash
# Serial stage runner for the dense-q80 A/B boots (process-hygiene protocol:
# preflight census, trap-on-EXIT cleanup, hard timeout, SIGTERM->10 s->SIGKILL,
# postflight census). FAILPAT lives in the run script's watchdog.
# Usage:
#   NSPLIT_NAME=nsplit1|nsplit2 ./ab_nsplit_stage.sh   (runs ab_nsplit_run.py)
#   MTILE_NAME=mtile4|mtile8|mtile16|mtile32|mtile64 ./ab_nsplit_stage.sh
#                                                      (runs ab_mtile_run.py)
set -u
REPO=/media/ai/src/FreeToken
cd "$REPO" || exit 1
TASKDIR="$REPO/.tasks/dense-q80-gemm"
NAME=${NSPLIT_NAME:-${MTILE_NAME:?set NSPLIT_NAME (nsplit1|nsplit2) or MTILE_NAME (mtile4|mtile8|mtile16|mtile32|mtile64)}}
if [ -n "${NSPLIT_NAME:-}" ]; then RUNNER=ab_nsplit_run.py; else RUNNER=ab_mtile_run.py; fi
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
  # have already moved+renamed their report inside the run script
  rm -f "$REPO"/report1.nsys-rep "$REPO"/report1.sqlite
  census
}
trap cleanup EXIT

census
echo "[stage] boot + fill + nsys capture (NAME=$NAME, runner=$RUNNER, MTILE=32)"
timeout 2400 .venv/bin/python "$TASKDIR/$RUNNER" --name "$NAME"
rc=$?
echo "[stage] $RUNNER rc=$rc"
kill_tree
sleep 3
# sqlite export for the analyzer; report presence is verified post-stop inside
# the run script (moved+renamed), the banner is NOT a discriminator (M1)
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