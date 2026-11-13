#!/bin/sh
# Neuro worker for a Mac with Apple Silicon - one command starts the local model and the worker.
#   cd ~/neuroai-mesh && git pull && sh platforms/mac-worker.sh
# Needs llama.cpp once:  brew install llama.cpp
# The model is downloaded by llama-server the first time: Qwen3.5-4B (~2.9 GB), or Qwen3.5-9B
# (~6 GB, slower but passes the tests more often) with NEURO_MODEL=9b.
# If a model already answers on 127.0.0.1:8080 it is used as is. Ctrl+C stops the worker and
# the model this script started. The Mac stays awake while it runs.
# Neuro Router on the main PC reaches the model through platforms/lan_relay.py: it listens on this
# Mac's private LAN address (port 8080, home LAN peers only) and passes traffic to 127.0.0.1:8080,
# where the model itself stays. NEURO_MODEL_LAN=0 turns the relay off; NEURO_MODEL_HOST picks the
# address when the Mac has several.
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

# llama-server keeps old prompts in a RAM cache of up to 8 GB by default (--cache-ram). Every job
# has a new prompt, so it only grows and takes RAM the model needs. Builds older than the option
# refuse unknown flags, so it is added only when --help lists it.
cache_args() {
  if llama-server --help 2>&1 | grep -q -- '--cache-ram'; then echo "--cache-ram 0"; fi
}

start_model() {
  CACHE_ARGS="$(cache_args)"
  # shellcheck disable=SC2086  # CACHE_ARGS is empty or two words
  llama-server -hf "$HF_MODEL" --host 127.0.0.1 --port 8080 -np "$MODEL_SLOTS" --kv-unified -c "$MODEL_CTX" --jinja $CACHE_ARGS >>"$LOG" 2>&1 &
  echo $! >"$PID_FILE"
}

# This Mac's private LAN address (10/8, 172.16/12, 192.168/16), or nothing.
lan_ip() {
  for ip in "${NEURO_MODEL_HOST:-}" "$(ipconfig getifaddr en0 2>/dev/null || true)" "$(ipconfig getifaddr en1 2>/dev/null || true)"; do
    case "$ip" in
      10.*|192.168.*|172.1[6-9].*|172.2[0-9].*|172.3[01].*) echo "$ip"; return 0 ;;
    esac
  done
  return 1
}

LLAMA_PID=""
WATCH_PID=""
RELAY_PID=""
cleanup() {
  if [ -n "$WATCH_PID" ]; then
    kill "$WATCH_PID" 2>/dev/null || true
  fi
  if [ -n "$RELAY_PID" ]; then
    kill "$RELAY_PID" 2>/dev/null || true
  fi
  if [ -n "$LLAMA_PID" ]; then
    kill "$(cat "$PID_FILE" 2>/dev/null || echo "$LLAMA_PID")" 2>/dev/null || true
    rm -f "$PID_FILE"
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

# Watchdog: if the model process dies, or stops answering for 5 minutes, start it again, so the
# worker does not sit idle on "Connection refused". It runs also when an already running model was
# reused: that model belongs to an earlier window, and closing that window stops it (the Mac sat
# without a model for hours that way).
start_watchdog() {
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
}

if model_up; then
  echo "[1/2] A model is already answering on 127.0.0.1:8080. Using it."
  command -v llama-server >/dev/null 2>&1 && start_watchdog
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
  start_watchdog
fi

if [ "${NEURO_MODEL_LAN:-1}" != "0" ]; then
  if LAN_IP=$(lan_ip); then
    python3 platforms/lan_relay.py --listen "$LAN_IP" --port 8080 --target 127.0.0.1:8080 &
    RELAY_PID=$!
    echo "      Neuro Router can use this model at http://$LAN_IP:8080 (home LAN only)."
  else
    echo "      No private LAN address found: the model stays on this Mac only (set NEURO_MODEL_HOST)."
  fi
fi

echo "[2/2] Starting the Neuro worker (it stays awake while running)."
if command -v caffeinate >/dev/null 2>&1; then
  caffeinate -i bash platforms/start-mac.sh
else
  bash platforms/start-mac.sh
fi
