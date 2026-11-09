#!/bin/sh
# Neuro worker for Linux. One command joins the Windows page.
set -eu

cd "$(dirname "$0")" || exit 1
if [ ! -f mesh.py ]; then
  cd .. || exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 was not found."
  exit 1
fi

echo "This computer joins the Windows page by itself. Leave that page open."
lan_ip=$(hostname -I 2>/dev/null | awk '{print $1}' || true)
if [ -n "$lan_ip" ]; then
  echo "Address: $lan_ip"
fi
echo "Local network only."

exec python3 -u mesh.py work
