#!/bin/sh
# Neuro worker for the Steam Deck (SteamOS) - one command does everything:
#   1. downloads llama.cpp for the Deck's GPU (Vulkan), once
#   2. downloads a model once: Qwen3.5-4B (~2.9 GB), or Qwen3.5-9B (~6 GB) with NEURO_MODEL=9b
#   3. starts the model on 127.0.0.1:8080 (this computer only), unless one is already running
#   4. starts the Neuro worker and shows the pairing code
# Run it in Desktop mode, from Konsole:
#   git clone https://github.com/ori-liloss/neuroai-mesh.git ~/neuroai-mesh && sh ~/neuroai-mesh/platforms/steamdeck.sh
# Next time:  sh ~/neuroai-mesh/platforms/steamdeck.sh
# If the main PC is not found by itself:  NEURO_MESH_URL=http://<main PC>:8780 sh ~/neuroai-mesh/platforms/steamdeck.sh
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

# No NEURO_MESH_URL: mesh.py finds the main PC on this network and remembers it in mesh_url.txt.
LLAMA_DIR="$HOME/llama"
MODEL_DIR="$HOME/models"
# NEURO_MODEL=9b uses Qwen3.5-9B (about 6 GB): slower, but its answers pass the tests more often.
case "${NEURO_MODEL:-4b}" in
  9b|9B) MODEL_NAME="Qwen3.5-9B-UD-Q4_K_XL.gguf"; MODEL_REPO="unsloth/Qwen3.5-9B-GGUF" ;;
  *) MODEL_NAME="Qwen3.5-4B-UD-Q4_K_XL.gguf"; MODEL_REPO="unsloth/Qwen3.5-4B-GGUF" ;;
esac
MODEL="$MODEL_DIR/$MODEL_NAME"
MODEL_URL="https://huggingface.co/$MODEL_REPO/resolve/main/$MODEL_NAME"
LOG="$HOME/llama-server.log"
# Parallel slots: llama-server answers NEURO_MODEL_SLOTS requests at once (1-8, default 2),
# 8k context each, and the worker runs one generate job per slot.
MODEL_SLOTS="${NEURO_MODEL_SLOTS:-2}"
case "$MODEL_SLOTS" in 1|2|3|4|5|6|7|8) ;; *) MODEL_SLOTS=2 ;; esac
MODEL_CTX=$((MODEL_SLOTS * 8192))

model_up() { curl -s -m 2 http://127.0.0.1:8080/health >/dev/null 2>&1; }
PID_FILE="$HOME/.neuro-llama.pid"

start_model() {
  ( cd "$(dirname "$SERVER")" && exec "$SERVER" -m "$MODEL" --host 127.0.0.1 --port 8080 -np "$MODEL_SLOTS" --kv-unified -c "$MODEL_CTX" --jinja -ngl 99 ) >>"$LOG" 2>&1 &
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
  echo "[1/4] A model is already answering on 127.0.0.1:8080. Using it."
else
  find_server() { find "$LLAMA_DIR" -name llama-server -type f 2>/dev/null | head -1; }
  SERVER=$(find_server)
  if [ -z "$SERVER" ]; then
    echo "[1/4] Downloading llama.cpp (Vulkan build)..."
    mkdir -p "$LLAMA_DIR"
    # The releases list, not releases/latest: for llama.cpp "latest" has no binaries.
    # Only the browser_download_url fields: the release notes also contain links.
    URL=$(curl -s -m 30 "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=1" \
      | grep -oE '"browser_download_url": *"[^"]*ubuntu-vulkan-x64\.tar\.gz"' | head -1 \
      | grep -oE 'https://[^"]*' || true)
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
    echo "[2/4] Downloading $MODEL_NAME (3-6 GB). It resumes if interrupted."
    mkdir -p "$MODEL_DIR"
    curl -L -f -C - -o "$MODEL.part" "$MODEL_URL"
    mv "$MODEL.part" "$MODEL"
  else
    echo "[2/4] Model already downloaded."
  fi

  echo "[3/4] Starting the model on 127.0.0.1:8080 (log: $LOG)..."
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
    [ "$waited" -le 240 ] || { echo "The model did not start in 4 minutes. See $LOG"; exit 1; }
    sleep 2
  done
  echo "      The model is ready."

  # Watchdog: the worker cannot generate without the model, and nothing else restarts it. If the
  # model process dies, or stops answering for 5 minutes, start it again. Hours of "Connection
  # refused" with the worker idle are what this prevents.
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

# A "Neuro Worker" icon on the Desktop (Desktop mode), so next time is one double-click.
LAUNCHER="$HOME/Desktop/neuro-worker.desktop"
if [ -d "$HOME/Desktop" ] && [ ! -e "$LAUNCHER" ]; then
  SCRIPT="$(pwd)/platforms/steamdeck.sh"
  cat > "$LAUNCHER" <<EOF
[Desktop Entry]
Type=Application
Name=Neuro Worker
Comment=Start the Neuro worker and its local model
Exec=konsole --hold -e sh "$SCRIPT"
Icon=utilities-terminal
Terminal=false
EOF
  chmod +x "$LAUNCHER"
  echo "      Added a 'Neuro Worker' icon to the Desktop. Next time, double-click it."
fi

# verify_luau jobs need the luau command line tool. If it is missing, fetch the official
# release once into ~/.local/share/neuro/luau (where mesh.py looks after NEURO_LUAU and PATH).
# Opt out with NEURO_NO_LUAU_DOWNLOAD=1. Any failure only means no verify_luau jobs here.
LUAU_DIR="$HOME/.local/share/neuro/luau"
get_luau() {
  [ -n "${NEURO_NO_LUAU_DOWNLOAD:-}" ] && [ "${NEURO_NO_LUAU_DOWNLOAD}" != "0" ] && return 0
  [ -n "${NEURO_LUAU:-}" ] && [ -f "${NEURO_LUAU}" ] && return 0
  command -v luau >/dev/null 2>&1 && return 0
  [ -s "$LUAU_DIR/luau" ] && return 0
  echo "      luau was not found. Downloading the official release for verify_luau jobs (NEURO_NO_LUAU_DOWNLOAD=1 skips)..."
  # Only assets[].browser_download_url, read as JSON: the release notes are never searched.
  url=$(curl -s -m 30 -H "User-Agent: neuro-worker" "https://api.github.com/repos/luau-lang/luau/releases?per_page=1" \
    | python3 -c '
import json, re, sys
try:
    releases = json.load(sys.stdin)
except ValueError:
    sys.exit(0)
for release in releases if isinstance(releases, list) else []:
    for asset in release.get("assets") or []:
        url = str(asset.get("browser_download_url") or "")
        if re.fullmatch(r"https://github\.com/luau-lang/luau/releases/download/[^/]+/luau-ubuntu\.zip", url):
            print(url)
            sys.exit(0)
') || return 1
  [ -n "$url" ] || { echo "      No luau-ubuntu.zip in the latest release."; return 1; }
  echo "      $url"
  tmp=$(mktemp -d) || return 1
  if curl -L -f -s -m 120 -o "$tmp/luau.zip" "$url" \
    && python3 -m zipfile -e "$tmp/luau.zip" "$tmp/x" \
    && found=$(find "$tmp/x" -name luau -type f | head -n 1) \
    && [ -n "$found" ] \
    && mkdir -p "$LUAU_DIR" \
    && mv "$found" "$LUAU_DIR/luau.part" \
    && chmod +x "$LUAU_DIR/luau.part" \
    && mv "$LUAU_DIR/luau.part" "$LUAU_DIR/luau"; then
    rm -rf "$tmp"
    echo "      luau is in $LUAU_DIR/luau"
    return 0
  fi
  rm -rf "$tmp"
  rm -f "$LUAU_DIR/luau.part"
  return 1
}
get_luau || echo "      luau download skipped. The worker runs without verify_luau jobs."

echo "[4/4] Starting the Neuro worker. Main PC: ${NEURO_MESH_URL:-found on this network}"
echo "      A pairing code (Pairing code: ### ###) appears in about 30 seconds."
echo "      Approve it on the main PC's Neuro page. Ctrl+C stops everything."
python3 -u mesh.py work
