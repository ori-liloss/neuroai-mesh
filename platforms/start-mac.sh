#!/bin/sh
# NeuroAI Mesh worker for macOS. One command joins the Windows page.
set -eu

cd "$(dirname "$0")" || exit 1
if [ ! -f mesh.py ]; then
  cd .. || exit 1
fi

if [ -x /opt/homebrew/bin/python3 ]; then
  PY=/opt/homebrew/bin/python3
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  echo "Python 3 was not found. Install it with: brew install python"
  echo "Then run this script again. The command is python3, not python."
  exit 1
fi

if command -v git >/dev/null 2>&1 && [ -d .git ]; then
  git pull --ff-only || true
fi

echo "This Mac joins the Windows page by itself. Leave that page open."
echo "Do not run mesh.py serve here."
echo "Ctrl+C stops this worker. The command is: bash platforms/start-mac.sh"

iface=$(route -n get default 2>/dev/null | awk '/interface:/{print $2}')
lan_ip=""
if [ -n "$iface" ]; then
  lan_ip=$(ipconfig getifaddr "$iface" 2>/dev/null || true)
fi
echo "Addresses on this Mac:"
for name in en0 en1 en2 en3 en4 en5 bridge0; do
  ip=$(ipconfig getifaddr "$name" 2>/dev/null || true)
  if [ -n "$ip" ]; then
    echo "  $name  $ip"
  fi
done
if [ -n "$lan_ip" ]; then
  echo "Default route: $lan_ip"
fi

exec "$PY" -u mesh.py work
