#!/bin/sh
# NeuroAI Mesh worker for macOS. This is not the dashboard.
set -eu

cd "$(dirname "$0")" || exit 1
if [ ! -f mesh.py ]; then
  cd .. || exit 1
fi

echo "This Mac joins the Windows dashboard. It does not open the dashboard."
echo "If mesh.py serve is running in another window, press Ctrl+C there first."
echo "Copy cluster.token from the Windows PC and put it next to mesh.py."

if [ -x /opt/homebrew/bin/python3 ]; then
  PY=/opt/homebrew/bin/python3
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  echo "Python 3 was not found. Install it with: brew install python"
  echo "Then run this script again. The command is python3, not python."
  exit 1
fi

if [ ! -f cluster.token ]; then
  echo "cluster.token is missing beside mesh.py."
  echo "On the Windows PC it is next to mesh.py. Copy that file here. Do not send it to the internet."
  exit 1
fi

lan_ip=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || true)
echo "Paste this address into the Mac bay on the Windows page:"
if [ -n "$lan_ip" ]; then
  echo "http://${lan_ip}:8781"
else
  echo "http://THIS-MAC-IP:8781"
fi
echo "Local network only."

exec "$PY" -u mesh.py work
