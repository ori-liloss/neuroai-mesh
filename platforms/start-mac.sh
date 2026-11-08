#!/bin/sh
# NeuroAI Mesh worker for macOS. This is not the dashboard.
set -eu

cd "$(dirname "$0")" || exit 1
if [ ! -f mesh.py ]; then
  cd .. || exit 1
fi

echo "This Mac joins the Windows dashboard. It does not open the dashboard."
echo "If mesh.py serve is running in another window, press Ctrl+C there first."
echo "cluster.token must be the file from the Windows PC, beside mesh.py."
echo "If this Mac already created its own cluster.token, delete it and copy the Windows file."

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

hint=$("$PY" -c "import hashlib, pathlib; print(hashlib.sha256(pathlib.Path('cluster.token').read_text().strip().encode()).hexdigest()[:8])")
echo "token id ${hint}"
echo "It must match the token id on the Windows page. If it does not, replace this file and run the script again."

iface=$(route -n get default 2>/dev/null | awk '/interface:/{print $2}')
lan_ip=""
if [ -n "$iface" ]; then
  lan_ip=$(ipconfig getifaddr "$iface" 2>/dev/null || true)
fi

echo "Addresses on this Mac:"
for name in en0 en1 en2 en3 en4 en5 bridge0; do
  ip=$(ipconfig getifaddr "$name" 2>/dev/null || true)
  if [ -n "$ip" ]; then
    echo "  $name  http://${ip}:8781"
  fi
done
if [ -n "$lan_ip" ]; then
  echo "Default route: http://${lan_ip}:8781"
else
  echo "Default route: no address. The Windows page still searches the local network."
fi
echo "Local network only."

exec "$PY" -u mesh.py work
