#!/usr/bin/env bash
# Train Link Simulator - launcher for macOS / Linux. Requires python3 (preinstalled on macOS and most Linux).
set -e
cd "$(dirname "$0")/site" 2>/dev/null || cd "$(dirname "$0")"
PORT=8777
while lsof -i :$PORT >/dev/null 2>&1; do PORT=$((PORT+1)); done
echo
echo "  Train Link Simulator"
echo "  Open: http://localhost:$PORT/"
echo "  Keep this window open while you use the simulator; Ctrl+C to stop."
echo
(sleep 1; (open "http://localhost:$PORT/" 2>/dev/null || xdg-open "http://localhost:$PORT/" 2>/dev/null || true)) &
exec python3 -m http.server "$PORT" --bind 127.0.0.1
