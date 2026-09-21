#!/usr/bin/env bash
# v0 A/B stage runner: boot + 65k prefill + decode @cached prefix, winner flags.
# Usage: v0ab_stage.sh <name> <logname>
# Serial only; measure.py refuses to start if another ft serve is alive.
set -u
NAME="$1"; LOG="$2"
DIR=/media/ai/src/FreeToken/.tasks/ft-gguf-serve-tuning
OUT=/media/ai/src/FreeToken/.tasks/mmq-prefill-kernel/v0ab_${NAME}.out

echo "=== census BEFORE stage ${NAME} ==="
pgrep -af '[f]t serve' || echo 'no ft serve'
nvidia-smi --query-gpu=memory.used --format=csv,noheader
ls /dev/shm | grep -c '^sem\.mp-'

timeout 1800 python3 "$DIR/measure.py" full --name "$NAME" --log "$LOG" \
  --extra "--max-running-requests 1 --max-prefill-length 8191 --memory-ratio 0.85" \
  > "$OUT" 2>&1
RC=$?
echo "=== measure rc=$RC ==="
tail -3 "$OUT"

echo "=== census AFTER stage ${NAME} ==="
pgrep -af '[f]t serve' || echo 'no ft serve'
nvidia-smi --query-gpu=memory.used --format=csv,noheader
ls /dev/shm | grep -c '^sem\.mp-'
exit $RC
