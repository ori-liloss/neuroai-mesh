#!/bin/sh
# Neuro worker for the Steam Deck (SteamOS) - one command does everything:
#   1. downloads llama.cpp for the Deck's GPU (Vulkan), once
#   2. downloads a small model (Qwen3.5-4B, about 2.9 GB), once
#   3. starts the model on 127.0.0.1:8080 (this computer only), unless one is already running
#   4. starts the Neuro worker and shows the pairing code
# Run it in Desktop mode, from Konsole:
#   git clone https://github.com/ori-liloss/neuroai-mesh.git ~/neuro-cluster && sh ~/neuro-cluster/platforms/steamdeck.sh
# Next time:  sh ~/neuro-cluster/platforms/steamdeck.sh
# Another main PC address:  NEURO_MESH_URL=http://192.168.1.20:8780 sh ~/neuro-cluster/platforms/steamdeck.sh
# Ctrl+C stops the worker and the model it started. Nothing is installed system-wide.
set -eu

# Keep the Deck awake while this runs.
if [ -z "${NEURO_INHIBITED:-}" ] && command -v systemd-inhibit >/dev/null 2>&1; then
  NEURO_INHIBITED=1 exec systemd-inhibit --what=sleep:idle --why="Neuro worker" sh "$0" "$@"
fi

cd "$(dirname "$0")/.." || exit 1
[ -f mesh.py ] || { echo "mesh.py was not found next to this script."; exit 1; }
for tool in python3 curl tar; do
  command -v "$tool" >/dev/null 2>&1 || { echo "$tool was not found. Install it first."; exit 1; }
done

NEURO_MESH_URL=${NEURO_MESH_URL:-http://192.168.7.14:8780}
export NEURO_MESH_URL
LLAMA_DIR="$HOME/llama"
MODEL_DIR="$HOME/models"
MODEL="$MODEL_DIR/Qwen3.5-4B-UD-Q4_K_XL.gguf"
MODEL_URL="https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-UD-Q4_K_XL.gguf"
LOG="$HOME/llama-server.log"

model_up() { curl -s -m 2 http://127.0.0.1:8080/health >/dev/null 2>&1; }

LLAMA_PID=""
cleanup() {
  if [ -n "$LLAMA_PID" ]; then
    kill "$LLAMA_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

if model_up; then
  echo "[1/4] A model is already answering on 127.0.0.1:8080. Using it."
else
  find_server() { find "$LLAMA_DIR" -name llama-server -type f 2>/dev/null | head -1; }
  SERVER=$(find_server)
  if [ -z "$SERVER" ]; then
    echo "[1/4] Downloading llama.cpp (Vulkan build)..."
    mkdir -p "$LLAMA_DIR"
    # The releases list, not releases/latest: for llama.cpp "latest" has no binaries.
    URL=$(curl -s -m 30 "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=1" \
      | grep -oE 'https://[^"]*ubuntu-vulkan-x64\.tar\.gz' | head -1)
    [ -n "$URL" ] || { echo "Could not find the llama.cpp download. Check the internet and run again."; exit 1; }
    echo "      $URL"
    curl -L -f -o "$LLAMA_DIR/pkg.tar.gz" "$URL"
    tar xzf "$LLAMA_DIR/pkg.tar.gz" -C "$LLAMA_DIR"
    rm -f "$LLAMA_DIR/pkg.tar.gz"
    SERVER=$(find_server)
  fi
  [ -n "$SERVER" ] || { echo "llama-server was not found after the download."; exit 1; }
  chmod +x "$(dirname "$SERVER")"/* 2>/dev/null || true

  if [ ! -s "$MODEL" ]; then
    echo "[2/4] Downloading the model (about 2.9 GB). It resumes if interrupted."
    mkdir -p "$MODEL_DIR"
    curl -L -f -C - -o "$MODEL.part" "$MODEL_URL"
    mv "$MODEL.part" "$MODEL"
  else
    echo "[2/4] Model already downloaded."
  fi

  echo "[3/4] Starting the model on 127.0.0.1:8080 (log: $LOG)..."
  ( cd "$(dirname "$SERVER")" && exec "$SERVER" -m "$MODEL" --host 127.0.0.1 --port 8080 -c 8192 --jinja -ngl 99 ) >"$LOG" 2>&1 &
  LLAMA_PID=$!
  waited=0
  until model_up; do
    if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
      echo "The model stopped. Last lines of $LOG:"
      tail -n 15 "$LOG"
      exit 1
    fi
    waited=$((waited + 2))
    [ "$waited" -le 240 ] || { echo "The model did not start in 4 minutes. See $LOG"; exit 1; }
    sleep 2
  done
  echo "      The model is ready."
fi

echo "[4/4] Starting the Neuro worker. Main PC: $NEURO_MESH_URL"
echo "      A pairing code (Pairing code: ### ###) appears in about 30 seconds."
echo "      Approve it on the main PC's Neuro page. Ctrl+C stops everything."
python3 -u mesh.py work
