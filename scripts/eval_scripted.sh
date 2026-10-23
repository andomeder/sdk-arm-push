#!/usr/bin/env bash
# Full vf-eval path with no model: a fake chat server plays the scripted policy as tool calls.
set -euo pipefail
cd "$(dirname "$0")/.."
PORT=${SCRIPTED_PORT:-8099}
uv run --frozen python scripts/scripted_server.py --port "$PORT" & pid=$!
trap 'kill $pid 2>/dev/null' EXIT
sleep 1
SCRIPTED_KEY=x MODEL_BASE_URL="http://127.0.0.1:$PORT/v1" MODEL_NAME=scripted \
  MODEL_API_KEY_VAR=SCRIPTED_KEY N=${N:-8} scripts/eval.sh "$@"
