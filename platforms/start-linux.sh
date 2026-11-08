#!/bin/sh
# NeuroAI Mesh worker for Linux. Local network only. Does not start llama.cpp RPC.
set -eu

cd "$(dirname "$0")" || exit 1
if [ ! -f mesh.py ]; then
  cd .. || exit 1
fi

lan_ip=$(hostname -I 2>/dev/null | awk '{print $1}')

echo "Copy the neuro-cluster folder onto this Linux machine."
echo "Copy cluster.token from the Windows machine next to mesh.py."
echo "Install Python 3."
echo "Run this script from inside the folder."
echo "Paste this address into NeuroAI Mesh:"
if [ -n "$lan_ip" ]; then
  echo "http://${lan_ip}:8781"
else
  echo "http://THIS-MACHINE-IP:8781"
fi
echo "Local network only. Do not open this port to the internet."
echo "This script does not start llama.cpp RPC."

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 was not found."
  exit 1
fi

if [ ! -f cluster.token ]; then
  echo "cluster.token is missing beside mesh.py."
  exit 1
fi

python3 mesh.py work
