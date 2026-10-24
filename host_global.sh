#!/usr/bin/env bash
# Expose the local Streamlit app (default port 8501) to the internet.
# Order of preference: Cloudflare Tunnel -> ngrok -> localtunnel.
#
#   chmod +x host_global.sh && ./host_global.sh
#   PORT=8600 ./host_global.sh
#
# Windows users: run `python host_global.py` instead (same behaviour).
set -euo pipefail

cd "$(dirname "$0")"
PORT="${PORT:-8501}"
STREAMLIT_PID=""
TUNNEL_PID=""
LOG="$(mktemp)"

if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

cleanup() {
  echo; echo "Stopping..."
  [ -n "$TUNNEL_PID" ] && kill "$TUNNEL_PID" 2>/dev/null || true
  [ -n "$STREAMLIT_PID" ] && kill "$STREAMLIT_PID" 2>/dev/null || true
  rm -f "$LOG"
}
trap cleanup EXIT INT TERM

port_open() { (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; }

if ! port_open; then
  echo "[start] streamlit on port $PORT"
  python -m streamlit run app.py --server.port "$PORT" --server.headless true >/dev/null 2>&1 &
  STREAMLIT_PID=$!
fi

for _ in $(seq 1 60); do port_open && break; sleep 1; done
port_open || { echo "ERROR: app did not start on port $PORT"; exit 1; }

try_tunnel() {  # name, url-regex, command...
  local name="$1" regex="$2"; shift 2
  command -v "$1" >/dev/null 2>&1 || { echo "[skip] $name not installed"; return 1; }
  : > "$LOG"
  "$@" >"$LOG" 2>&1 &
  TUNNEL_PID=$!
  for _ in $(seq 1 45); do
    URL="$(grep -Eo "$regex" "$LOG" | head -n1 || true)"
    if [ -n "$URL" ]; then
      echo "$URL" > public_url.txt
      echo "================================================================"
      echo "  PUBLIC URL ($name): $URL"
      echo "================================================================"
      echo "  Set APP_PASSWORD in .env before sharing. Ctrl+C to stop."
      wait "$TUNNEL_PID" || true
      return 0
    fi
    kill -0 "$TUNNEL_PID" 2>/dev/null || break
    sleep 1
  done
  echo "[fail] $name produced no URL"; kill "$TUNNEL_PID" 2>/dev/null || true; TUNNEL_PID=""
  return 1
}

try_tunnel cloudflare 'https://[a-z0-9-]+\.trycloudflare\.com' cloudflared tunnel --url "http://localhost:$PORT" --no-autoupdate && exit 0
try_tunnel ngrok 'https://[a-z0-9-]+\.ngrok[-a-z]*\.(app|io|dev)' ngrok http "$PORT" --log stdout && exit 0
try_tunnel localtunnel 'https://[a-z0-9-]+\.loca\.lt' npx --yes localtunnel --port "$PORT" && exit 0

echo "ERROR: no tunnel could be started. See README.md > Global hosting."
exit 2
