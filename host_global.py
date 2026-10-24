#!/usr/bin/env python3
"""Expose the local Streamlit app to the internet through a tunnel.

Examples
--------
    python host_global.py                       # try cloudflared, then ngrok, then localtunnel
    python host_global.py --provider cloudflare
    python host_global.py --provider ngrok
    python host_global.py --provider localtunnel
    python host_global.py --no-streamlit        # app already running on the port

The public URL is printed and also written to ``public_url.txt``.
Set APP_PASSWORD in .env before sharing the link - anyone with the URL can use the app.
"""

from __future__ import annotations

import argparse
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
URL_PATTERNS = {
    "cloudflare": re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com"),
    "ngrok": re.compile(r"https://[a-z0-9-]+\.ngrok[-a-z]*\.(?:app|io|dev)"),
    "localtunnel": re.compile(r"https://[a-z0-9-]+\.loca\.lt"),
}
INSTALL_HINTS = {
    "cloudflare": "Windows: winget install --id Cloudflare.cloudflared | macOS: brew install cloudflared | "
    "Linux: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/",
    "ngrok": "Install from https://ngrok.com/download and run: ngrok config add-authtoken <token>",
    "localtunnel": "Install Node.js (https://nodejs.org) - npx downloads localtunnel automatically",
}


def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def wait_for_port(port: int, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if port_open(port):
            return True
        time.sleep(0.5)
    return False


def tunnel_command(provider: str, port: int) -> list[str] | None:
    if provider == "cloudflare":
        exe = shutil.which("cloudflared")
        return [exe, "tunnel", "--url", f"http://localhost:{port}", "--no-autoupdate"] if exe else None
    if provider == "ngrok":
        exe = shutil.which("ngrok")
        return [exe, "http", str(port), "--log", "stdout"] if exe else None
    if provider == "localtunnel":
        exe = shutil.which("npx")
        return [exe, "--yes", "localtunnel", "--port", str(port)] if exe else None
    raise ValueError(provider)


def start_tunnel(provider: str, port: int, wait_s: float) -> tuple[subprocess.Popen[str], str] | None:
    command = tunnel_command(provider, port)
    if command is None:
        print(f"[skip] {provider}: not installed.  {INSTALL_HINTS[provider]}")
        return None
    print(f"[start] {provider}: {' '.join(command)}")
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, encoding="utf-8", errors="replace"
    )
    found: dict[str, str] = {}

    def pump() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            match = URL_PATTERNS[provider].search(line)
            if match and "url" not in found and "api." not in match.group(0):
                found["url"] = match.group(0)

    threading.Thread(target=pump, daemon=True).start()
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if "url" in found:
            return process, found["url"]
        if process.poll() is not None:
            break
        time.sleep(0.3)
    print(f"[fail] {provider}: no public URL after {wait_s:.0f}s")
    process.terminate()
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", choices=["auto", "cloudflare", "ngrok", "localtunnel"], default="auto")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--no-streamlit", action="store_true", help="do not start Streamlit (already running)")
    parser.add_argument("--wait", type=float, default=45.0, help="seconds to wait for each tunnel URL")
    args = parser.parse_args()

    streamlit_proc: subprocess.Popen[bytes] | None = None
    if not args.no_streamlit and not port_open(args.port):
        print(f"[start] streamlit on port {args.port}")
        streamlit_proc = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"), "--server.port", str(args.port), "--server.headless", "true"],
            cwd=ROOT,
        )
    if not wait_for_port(args.port, 60):
        print(f"ERROR: nothing is listening on port {args.port}. Start the app first: streamlit run app.py")
        if streamlit_proc:
            streamlit_proc.terminate()
        return 1

    providers = ["cloudflare", "ngrok", "localtunnel"] if args.provider == "auto" else [args.provider]
    tunnel: tuple[subprocess.Popen[str], str] | None = None
    for provider in providers:
        tunnel = start_tunnel(provider, args.port, args.wait)
        if tunnel:
            used = provider
            break
    if tunnel is None:
        print("ERROR: could not open any tunnel. See README.md > Global hosting.")
        if streamlit_proc:
            streamlit_proc.terminate()
        return 2

    process, url = tunnel
    (ROOT / "public_url.txt").write_text(url + "\n", encoding="utf-8")
    bar = "=" * 64
    print(f"\n{bar}\n  PUBLIC URL ({used}):  {url}\n{bar}")
    if used == "localtunnel":
        print("  localtunnel asks visitors for a password: it is the host machine's public IP (https://loca.lt/mytunnelpassword)")
    print("  Press Ctrl+C to stop sharing.\n")
    try:
        while process.poll() is None:
            time.sleep(1)
        print("Tunnel process exited.")
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        process.terminate()
        if streamlit_proc:
            streamlit_proc.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
