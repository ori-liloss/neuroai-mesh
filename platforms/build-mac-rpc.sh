#!/bin/sh
# Build the Mac memory listener from the same llama.cpp commit as the Windows server.
# The listener does not start the model. start-mac.sh starts it when this binary exists.
set -eu

COMMIT=c0ab6e5c0
ROOT="$HOME/llama.cpp"

if ! command -v git >/dev/null 2>&1 || ! command -v cmake >/dev/null 2>&1; then
  echo "git and cmake are required. Install them, then run this script again."
  exit 1
fi

if [ ! -d "$ROOT/.git" ]; then
  git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git "$ROOT"
fi

cd "$ROOT"
git fetch --depth 1 origin "$COMMIT" || git fetch origin
git checkout "$COMMIT"
cmake -B build -DGGML_RPC=ON -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target ggml-rpc-server -j
echo "Built. Next: bash platforms/start-mac.sh"
echo "The model already running on Windows stays there until that server is started again."
