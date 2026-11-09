#!/bin/sh
# Neuro worker for a Mac with Apple Silicon - one command starts the local model and the worker.
#   cd ~/neuroai-mesh && git pull && sh platforms/mac-worker.sh
# Needs llama.cpp once:  brew install llama.cpp
# The model (Qwen3.5-4B, about 2.9 GB) is downloaded by llama-server the first time.
# If a model already answers on 127.0.0.1:8080 it is used as is. Ctrl+C stops the worker and
# the model this script started. The Mac stays awake while it runs.
set -eu
cd "$(dirname "$0")/.." || exit 1
[ -f mesh.py ] || { echo "mesh.py was not found next to this script."; exit 1; }

model_up() { curl -s -m 2 http://127.0.0.1:8080/health >/dev/null 2>&1; }
LOG="$HOME/llama-server.log"
# Parallel slots: llama-server answers NEURO_MODEL_SLOTS requests at once (1-8, default 2),
# 8k context each, and the worker runs one generate job per slot.
MODEL_SLOTS="${NEURO_MODEL_SLOTS:-2}"
case "$MODEL_SLOTS" in 1|2|3|4|5|6|7|8) ;; *) MODEL_SLOTS=2 ;; esac
MODEL_CTX=$((MODEL_SLOTS * 8192))
LLAMA_PID=""
cleanup() {
  if [ -n "$LLAMA_PID" ]; then
    kill "$LLAMA_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

if model_up; then
  echo "[1/2] A model is already answering on 127.0.0.1:8080. Using it."
else
  command -v llama-server >/dev/null 2>&1 || { echo "llama-server was not found. Run: brew install llama.cpp"; exit 1; }
  echo "[1/2] Starting the model on 127.0.0.1:8080 (first time downloads ~2.9 GB; log: $LOG)..."
  llama-server -hf unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL --host 127.0.0.1 --port 8080 -np "$MODEL_SLOTS" --kv-unified -c "$MODEL_CTX" --jinja >"$LOG" 2>&1 &
  LLAMA_PID=$!
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
fi

echo "[2/2] Starting the Neuro worker (it stays awake while running)."
if command -v caffeinate >/dev/null 2>&1; then
  caffeinate -i bash platforms/start-mac.sh
else
  bash platforms/start-mac.sh
fi
