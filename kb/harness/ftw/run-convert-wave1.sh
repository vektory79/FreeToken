#!/usr/bin/env bash
# Wave-1 hardware validation runner (ftw-gguf-fastpath TASK, Hardware validation step 1):
# bare GGUF -> FTW conversion of GLM-5.3-Flash UD-Q3_K_XL.
# Hygiene: trap-on-EXIT cleanup, hard timeout 1800 s, FAILPAT tail watchdog (5 s poll),
# SIGTERM -> SIGKILL escalation, setsid process-group kill.
set -u
SRC=/media/ai/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL.gguf
OUT=/media/ai/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL-FTW
LOG=/tmp/ftw-convert-glm53-q3k-wave1.log
REPO=/media/ai/src/FreeToken
TIMEOUT_S=1800
FAILLPAT='AssertionError|Traceback|OutOfMemoryError|No space left'
CHILD=""

cleanup() {
  rc=$?
  trap - EXIT INT TERM
  if [ -n "$CHILD" ] && kill -0 -- -"$CHILD" 2>/dev/null; then
    echo "[runner] cleanup: SIGTERM process group -$CHILD"
    kill -TERM -- -"$CHILD" 2>/dev/null
    for _ in $(seq 1 30); do
      kill -0 -- -"$CHILD" 2>/dev/null || break
      sleep 1
    done
    if kill -0 -- -"$CHILD" 2>/dev/null; then
      echo "[runner] cleanup: SIGKILL process group -$CHILD"
      kill -KILL -- -"$CHILD" 2>/dev/null
    fi
  fi
  exit "$rc"
}
trap cleanup EXIT INT TERM

[ -f "$SRC" ] || { echo "[runner] FATAL: missing source $SRC"; exit 2; }
if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
  echo "[runner] FATAL: $OUT exists and is non-empty - refusing"; exit 3
fi
: > "$LOG"
echo "[runner] start $(date -Is)"
echo "[runner] src=$(stat -c %s "$SRC") bytes out=$OUT timeout=${TIMEOUT_S}s failpat=$FAILLPAT"

cd "$REPO"
FREETOKEN_CONVERT_PROGRESS=1 setsid uv run ft checkpoint \
  --model "$SRC" --out "$OUT" --moe-backend offload >"$LOG" 2>&1 &
CHILD=$!
START=$(date +%s)
RC=0
while kill -0 "$CHILD" 2>/dev/null; do
  if [ $(( $(date +%s) - START )) -ge "$TIMEOUT_S" ]; then
    echo "[runner] HARD TIMEOUT ${TIMEOUT_S}s -> SIGKILL group -$CHILD"
    kill -KILL -- -"$CHILD" 2>/dev/null
    RC=124
    break
  fi
  HIT=$(grep -n -E "$FAILLPAT" "$LOG" 2>/dev/null)
  if [ -n "$HIT" ]; then
    echo "[runner] FAILPAT HIT:"
    printf '%s\n' "$HIT" | head -40
    kill -KILL -- -"$CHILD" 2>/dev/null
    RC=9
    break
  fi
  sleep 5
done
wait_rc=0
wait "$CHILD" 2>/dev/null || wait_rc=$?
if [ "$RC" -eq 0 ]; then RC=$wait_rc; fi

WALL=$(( $(date +%s) - START ))
echo "[runner] child rc=$RC wall=${WALL}s"
HIT=$(grep -n -E "$FAILLPAT" "$LOG" 2>/dev/null)
if [ -n "$HIT" ]; then
  echo "[runner] FAILPAT matches in final log:"
  printf '%s\n' "$HIT" | head -40
else
  echo "[runner] FAILPAT: no matches in final log"
fi
echo "[runner] last 20 log lines:"
tail -20 "$LOG"
echo "[runner] post-census:"
pgrep -af '[f]t serve' || true
pgrep -af '[f]t checkpoint' || true
ls -la "$OUT" 2>/dev/null || echo "(no out dir)"
exit "$RC"
