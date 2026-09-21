#!/usr/bin/env bash
# Reproduce user crash: winner recipe --memory-ratio 0.80 --kv-reserve-tokens 400000
# --kv-cache-dtype fp8 --max-prefill-length 8191 --moe-strategy hybrid (port 18081).
# Exact user command; runner stdout + server log both archived in the task folder.
set -u
REPO=/media/ai/src/FreeToken
TD=$REPO/.tasks/ft-serve-crash-mr080-400k
LOG=$TD/serve.log
MODEL=/media/ai/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL-FTW/
PORT=18081
FAILLIST='AssertionError|OutOfMemoryError|Backend worker is gone|backend worker .* exited|Traceback|CUDA error|ValueError'
FT_PID=""
WD_PID=""

cleanup() {
  [ -n "${WD_PID}" ] && kill "${WD_PID}" 2>/dev/null
  if [ -n "${FT_PID}" ] && kill -0 "${FT_PID}" 2>/dev/null; then
    echo "[runner] SIGTERM -> ft (${FT_PID})"
    kill -TERM "${FT_PID}" 2>/dev/null
    for i in $(seq 1 30); do kill -0 "${FT_PID}" 2>/dev/null || break; sleep 1; done
    if kill -0 "${FT_PID}" 2>/dev/null; then
      echo "[runner] uvicorn hung (backend death / grace expiry) - kill -9 tree"
      pkill -9 -P "${FT_PID}" 2>/dev/null
      kill -9 "${FT_PID}" 2>/dev/null
    fi
  fi
}
trap cleanup EXIT

watchdog() {
  while true; do
    if grep -E "${FAILLIST}" "${LOG}" >/dev/null 2>&1; then
      echo "FAILPAT-HIT:"
      grep -E "${FAILLIST}" "${LOG}" | tail -8
      exit 42
    fi
    sleep 3
  done
}

T0=$(date +%s)
echo "[runner] VRAM before boot:"
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader

"${REPO}/.venv/bin/ft" serve --port ${PORT} --host 127.0.0.1 --model-path "${MODEL}" \
  --moe-cache-auto --kv-reserve-tokens 400000 --kv-cache-dtype fp8 \
  --memory-ratio 0.80 --max-prefill-length 8191 --moe-strategy hybrid \
  --moe-cpu-threads 16 --max-running-requests 1 >"${LOG}" 2>&1 &
FT_PID=$!
echo "[runner] ft pid ${FT_PID}, log ${LOG}"
watchdog &
WD_PID=$!

READY=0
for i in $(seq 1 250); do
  NOW=$(date +%s)
  if [ $((NOW - T0)) -ge 900 ]; then
    echo "[runner] HARD-TIMEOUT 900s exceeded"
    break
  fi
  if ! kill -0 "${WD_PID}" 2>/dev/null; then
    echo "[runner] watchdog exited (FAILPAT hit)"
    break
  fi
  if ! kill -0 "${FT_PID}" 2>/dev/null; then
    echo "[runner] SERVER-DIED during boot"
    break
  fi
  code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:${PORT}/ready 2>/dev/null)
  if [ "${code}" = "200" ]; then READY=1; break; fi
  sleep 3
done

if [ "${READY}" != "1" ]; then
  echo "BOOT-VERDICT: FAIL (ready not reached)"
  echo "RC=1"
  exit 1
fi
T1=$(date +%s)
echo "BOOT-VERDICT: READY after $((T1 - T0))s"

MODEL_ID=$(curl -s http://127.0.0.1:${PORT}/v1/models | sed -n 's/.*"id":"\([^"]*\)".*/\1/p' | head -1)
echo "SERVED-MODEL-ID: ${MODEL_ID}"
RESP=$(curl -s -w '\nHTTP=%{http_code}' http://127.0.0.1:${PORT}/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL_ID}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word OK.\"}],\"max_tokens\":8}")
echo "${RESP}"
if echo "${RESP}" | grep -q 'HTTP=200' && echo "${RESP}" | grep -q '"usage"'; then
  echo "CHAT-VERDICT: PASS"
  echo "[runner] VRAM at peak:"
  nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader
  echo "RC=0"
  exit 0
else
  echo "CHAT-VERDICT: FAIL"
  echo "RC=2"
  exit 2
fi
