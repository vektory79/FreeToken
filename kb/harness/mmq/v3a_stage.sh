#!/usr/bin/env bash
# v3a stage runner: census -> one serialized measure.py full run -> census.
# Trap-on-EXIT cleanup + FAILPAT watchdog; server PID exported to the watchdog
# via the measure.py pidfile (ORCHESTRATION 7 pattern).
# Usage: v3a_stage.sh <name> <log> "<ENV=V>"   (empty 3rd arg = baseline boot)
set -u
NAME="$1"; LOG="$2"; ENVP="$3"
TDIR=/media/ai/src/FreeToken/.tasks/ft-gguf-serve-tuning
VDIR=/media/ai/src/FreeToken/.tasks/mmq-v3-stationary-moe
OUT="$VDIR/v3a_${NAME}.out"
LOGF="$TDIR/logs/$LOG"
PIDF="$LOGF.pid"
FAILLIST='OutOfMemoryError|AssertionError|Backend worker is gone|backend worker .* exited'

census() {
  echo "=== census $1 $(date +%H:%M:%S) ==="
  pgrep -af '[f]t serve' || echo 'no ft serve'
  nvidia-smi --query-gpu=memory.used --format=csv,noheader
  echo "sem.mp-: $(ls /dev/shm | grep -c '^sem\.mp-')"
}

MEASP=""
cleanup() {
  rc=$?
  if [ -n "${WPID:-}" ]; then kill "$WPID" 2>/dev/null; wait "$WPID" 2>/dev/null; fi
  if pgrep -f '[f]t serve' >/dev/null 2>&1; then
    echo "cleanup: SIGTERM ft serve"
    pkill -TERM -f '[f]t serve' 2>/dev/null
    for i in $(seq 30); do pgrep -f '[f]t serve' >/dev/null 2>&1 || break; sleep 1; done
    if pgrep -f '[f]t serve' >/dev/null 2>&1; then
      echo "cleanup: 30 s grace expired -> SIGKILL"
      pkill -KILL -f '[f]t serve' 2>/dev/null
      sleep 3
    fi
  fi
  census "post-$NAME"
  exit $rc
}
trap cleanup EXIT

watchdog() {
  while :; do
    sleep 5
    if [ -z "$MEASP" ] && [ -f "$PIDF" ]; then
      MEASP=$(cat "$PIDF" 2>/dev/null || true)
      [ -n "$MEASP" ] && echo "watchdog: server pid=$MEASP"
    fi
    if tail -c 20000 "$LOGF" 2>/dev/null | grep -Eq "$FAILLIST"; then
      echo "WATCHDOG FAILPAT hit in $LOGF:"
      tail -c 20000 "$LOGF" 2>/dev/null | grep -E "$FAILLIST" | head -8
      echo "Tried to allocate lines:"
      tail -c 20000 "$LOGF" 2>/dev/null | grep 'Tried to allocate' | head -3
      [ -n "$MEASP" ] && { kill -9 "$MEASP" 2>/dev/null; pkill -9 -P "$MEASP" 2>/dev/null; }
      pkill -9 -f '[f]t serve' 2>/dev/null
      return
    fi
  done
}

census "pre-$NAME"
watchdog &
WPID=$!
if [ -n "$ENVP" ]; then
  timeout 2400 python3 "$TDIR/measure.py" full --name "$NAME" --log "$LOG" \
    --extra "--max-prefill-length 8191 --memory-ratio 0.85 --max-running-requests 1" \
    --env "$ENVP" > "$OUT" 2>&1
else
  timeout 2400 python3 "$TDIR/measure.py" full --name "$NAME" --log "$LOG" \
    --extra "--max-prefill-length 8191 --memory-ratio 0.85 --max-running-requests 1" \
    > "$OUT" 2>&1
fi
RC=$?
echo "=== stage $NAME env='$ENVP' rc=$RC ==="
grep -E '"ready_s"|"why"|"error"|"steady_tok_s"|"cached_token"|"new_token"' "$OUT" | head -12
kill "$WPID" 2>/dev/null
wait "$WPID" 2>/dev/null
WPID=""
census "mid-$NAME"
exit $RC
