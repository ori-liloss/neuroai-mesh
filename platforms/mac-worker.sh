#!/bin/sh
# Neuro worker for a Mac with Apple Silicon - one command starts the local model and the worker.
#   cd ~/neuroai-mesh && git pull && sh platforms/mac-worker.sh
# Needs llama.cpp once:  brew install llama.cpp
# The model is downloaded by llama-server the first time: Qwen3.5-4B (~2.9 GB), or Qwen3.5-9B
# (~6 GB, slower but passes the tests more often) with NEURO_MODEL=9b.
# If a model already answers on 127.0.0.1:8080 it is used as is. Ctrl+C stops the worker and
# the model this script started. The Mac stays awake while it runs.
set -eu
cd "$(dirname "$0")/.." || exit 1
[ -f mesh.py ] || { echo "mesh.py was not found next to this script."; exit 1; }

# -f: llama-server answers 503 while it is still loading, and that is not "up" yet.
model_up() { curl -sf -m 2 http://127.0.0.1:8080/health >/dev/null 2>&1; }
LOG="$HOME/llama-server.log"
PID_FILE="$HOME/.neuro-llama.pid"
# Parallel slots: llama-server answers NEURO_MODEL_SLOTS requests at once (1-8, default 2),
# 8k context each, and the worker runs one generate job per slot.
MODEL_SLOTS="${NEURO_MODEL_SLOTS:-2}"
case "$MODEL_SLOTS" in 1|2|3|4|5|6|7|8) ;; *) MODEL_SLOTS=2 ;; esac
MODEL_CTX=$((MODEL_SLOTS * 8192))
case "${NEURO_MODEL:-4b}" in 9b|9B) HF_MODEL="unsloth/Qwen3.5-9B-GGUF:UD-Q4_K_XL" ;; *) HF_MODEL="unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL" ;; esac

start_model() {
  llama-server -hf "$HF_MODEL" --host 127.0.0.1 --port 8080 -np "$MODEL_SLOTS" --kv-unified -c "$MODEL_CTX" --jinja >>"$LOG" 2>&1 &
  echo $! >"$PID_FILE"
}

LLAMA_PID=""
WATCH_PID=""
cleanup() {
  if [ -n "$WATCH_PID" ]; then
    kill "$WATCH_PID" 2>/dev/null || true
  fi
  if [ -n "$LLAMA_PID" ]; then
    kill "$(cat "$PID_FILE" 2>/dev/null || echo "$LLAMA_PID")" 2>/dev/null || true
    rm -f "$PID_FILE"
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

if model_up; then
  echo "[1/2] A model is already answering on 127.0.0.1:8080. Using it."
else
  command -v llama-server >/dev/null 2>&1 || { echo "llama-server was not found. Run: brew install llama.cpp"; exit 1; }
  echo "[1/2] Starting the model on 127.0.0.1:8080 (first time downloads ~2.9 GB; log: $LOG)..."
  : >"$LOG"
  start_model
  LLAMA_PID=$(cat "$PID_FILE")
  waited=0
  until model_up; do
    if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
      echo "The model stopped. Last lines of $LOG:"
      tail -n 15 "$LOG"
      exit 1
    fi
    waited=$((waited + 2))
    [ "$waited" -le 900 ] || { echo "The model did not start in 15 minutes. See $LOG"; exit 1; }
    sleep 2
  done
  echo "      The model is ready."

  # Watchdog: if the model process dies, or stops answering for 5 minutes, start it again, so the
  # worker does not sit idle on "Connection refused".
  (
    trap - EXIT INT TERM
    silent=0
    while sleep 30; do
      if model_up; then silent=0; continue; fi
      pid=$(cat "$PID_FILE" 2>/dev/null || true)
      if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        silent=$((silent + 1))
        [ "$silent" -ge 10 ] || continue
        echo "$(date '+%H:%M:%S') The model stopped answering for 5 minutes. Restarting it..."
        kill "$pid" 2>/dev/null || true
        sleep 5
      else
        echo "$(date '+%H:%M:%S') The model stopped. Last lines of $LOG:"
        tail -n 5 "$LOG"
        echo "$(date '+%H:%M:%S') Restarting the model..."
      fi
      silent=0
      start_model
    done
  ) &
  WATCH_PID=$!
fi

echo "[2/2] Starting the Neuro worker (it stays awake while running)."
if command -v caffeinate >/dev/null 2>&1; then
  caffeinate -i bash platforms/start-mac.sh
else
  bash platforms/start-mac.sh
fi
