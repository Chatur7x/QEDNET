#!/usr/bin/env bash
set -euo pipefail

export QEDNET_WORKSPACE_ROOT="${QEDNET_WORKSPACE_ROOT:-/app}"
export PYTHONPATH="${QEDNET_WORKSPACE_ROOT}/backend${PYTHONPATH:+:${PYTHONPATH}}"
export QEDNET_PYTHON="${QEDNET_PYTHON:-/opt/qednet-venv/bin/python}"
export PATH="/opt/qednet-venv/bin:${PATH}"
export HOSTNAME="0.0.0.0"
export PORT="${PORT:-10000}"

mkdir -p "${QEDNET_WORKSPACE_ROOT}/experiments/jobs" \
  "${QEDNET_WORKSPACE_ROOT}/experiments/runs" \
  "${QEDNET_WORKSPACE_ROOT}/data/uploads"

cd "${QEDNET_WORKSPACE_ROOT}/backend"
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 &
api_pid=$!

cd "${QEDNET_WORKSPACE_ROOT}"
node server.js &
web_pid=$!

shutdown() {
  kill "$api_pid" 2>/dev/null || true
  kill "$web_pid" 2>/dev/null || true
  wait "$api_pid" 2>/dev/null || true
  wait "$web_pid" 2>/dev/null || true
}
trap shutdown EXIT INT TERM

wait -n "$api_pid" "$web_pid"
