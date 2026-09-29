#!/usr/bin/env bash
# Start Qlib Lab and open it in your browser. Runs on this computer only (127.0.0.1).
set -euo pipefail
export PYTHONUTF8=1
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || exec ./install.sh
PORT=8103
URL="http://127.0.0.1:$PORT"
up() { (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; }
openurl() {
  if command -v open >/dev/null 2>&1; then open "$URL"
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL" >/dev/null 2>&1
  else echo "Open $URL in your browser"; fi
}
if up; then echo "Qlib Lab is already running - opening $URL"; openurl; exit 0; fi
[ -n "${NO_BROWSER:-}" ] || ( for _ in $(seq 1 240); do if up; then openurl; exit 0; fi; sleep 0.5; done ) &
echo "Starting Qlib Lab at $URL  (Ctrl+C to stop)"
exec .venv/bin/python ui/app.py
