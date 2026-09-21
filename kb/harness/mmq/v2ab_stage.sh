#!/usr/bin/env bash
# v2 A/B stage runner: census -> one serialized measure.py run -> census.
# Usage: v2ab_stage.sh <name> <log> <mode: prefill|full> "<extra-flags>" "<ENV=V>"
# Serial only; measure.py refuses to start if another ft serve is alive.
set -u
NAME="$1"; LOG="$2"; MODE="$3"; EXTRA="$4"; ENVP="$5"
DIR=/media/ai/src/FreeToken/.tasks/ft-gguf-serve-tuning
OUT=/media/ai/src/FreeToken/.tasks/mmq-prefill-kernel/v2ab_${NAME}.out

echo "=== census BEFORE stage ${NAME} ==="
pgrep -af '[f]t serve' || echo 'no ft serve'
nvidia-smi --query-gpu=memory.used --format=csv,noheader
ls /dev/shm | grep -c '^sem\.mp-'

timeout 1800 python3 "$DIR/measure.py" "$MODE" --name "$NAME" --log "$LOG" \
  --extra "$EXTRA" --env "$ENVP" > "$OUT" 2>&1
RC=$?
echo "=== stage ${NAME} mode=${MODE} env=${ENVP} rc=$RC ==="
tail -4 "$OUT"

echo "=== census AFTER stage ${NAME} ==="
pgrep -af '[f]t serve' || echo 'no ft serve'
nvidia-smi --query-gpu=memory.used --format=csv,noheader
ls /dev/shm | grep -c '^sem\.mp-'
exit $RC
