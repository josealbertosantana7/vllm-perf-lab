#!/usr/bin/env bash
# Expose a local port (default 8000) over an outbound cloudflared tunnel.
# Use this on the GPU box (Kaggle/Colab/cloud) to reach vLLM's API + /metrics
# from your Mac. Prints a public https://<random>.trycloudflare.com URL.
#
#   ./scripts/tunnel.sh 8000
set -euo pipefail
PORT="${1:-8000}"

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "installing cloudflared..."
  wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
    -O /usr/local/bin/cloudflared
  chmod +x /usr/local/bin/cloudflared
fi

echo "tunneling http://localhost:${PORT} ..."
exec cloudflared tunnel --url "http://localhost:${PORT}"
