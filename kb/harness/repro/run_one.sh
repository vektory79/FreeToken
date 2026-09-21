#!/bin/bash
# Hygiene runner for one measure.py run: FAILPAT watchdog + hard timeout + trap cleanup.
# Usage: run_one.sh <name> <logname> "<extra args>"
set -u
DIR=/media/ai/src/FreeToken/.tasks/ft-gguf-serve-tuning
LOGDIR="$DIR/logs"
NAME="$1"; LOG="$2"; EXTRA="${3:-}"
SRVLOG="$LOGDIR/$LOG"
PY=/media/ai/src/FreeToken/.venv/bin/python
HARD=900
FAILPAT='OutOfMemoryError|AssertionError|Backend worker is gone|backend worker .* exited'

cleanup() {
  [ -n "${WATCHPID:-}" ] && kill "$WATCHPID" 2>/dev/null
  pkill -TERM -f '[f]t serve' 2>/dev/null
  for i in $(seq 1 30); do pgrep -f '[f]t serve' >/dev/null 2>&1 || break; sleep 1; done
  pkill -KILL -f '[f]t serve' 2>/dev/null
}
trap cleanup EXIT

( while sleep 3; do
    M=$(grep -E "$FAILPAT" "$SRVLOG" 2>/dev/null | tail -n 30)
    if [ -n "$M" ]; then
      echo "=== FAILPAT HIT ==="
      echo "$M"
      echo "=== Tried to allocate ==="
      grep -B3 -A6 'Tried to allocate' "$SRVLOG" 2>/dev/null | tail -n 40
      pkill -KILL -f '[f]t serve' 2>/dev/null
      pkill -TERM -f 'measure\.py full' 2>/dev/null
      exit 42
    fi
  done ) &
WATCHPID=$!

cd "$DIR" || exit 1
rm -f "$SRVLOG"
START=$(date +%s)
timeout --signal=TERM --kill-after=30s "$HARD" "$PY" measure.py full --name "$NAME" --log "$LOG" --extra "$EXTRA"
RC=$?
END=$(date +%s)
echo "=== measure.py rc=$RC wall=$((END-START))s ==="
exit $RC
