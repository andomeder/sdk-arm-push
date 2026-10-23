#!/usr/bin/env bash
# Evaluate a model on sdk-arm-push. Nothing is uploaded (--no-push).
#   scripts/eval.sh                       # uses .env (local server by default)
#   MODEL_BASE_URL=... MODEL_NAME=... MODEL_API_KEY_VAR=OPENAI_API_KEY scripts/eval.sh
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/_env.sh
: "${MODEL_BASE_URL:?}" "${MODEL_NAME:?}" "${MODEL_API_KEY_VAR:?}"
[ -n "${!MODEL_API_KEY_VAR:-}" ] || { echo "\$$MODEL_API_KEY_VAR is empty (use any non-empty value for a local server)" >&2; exit 1; }
exec uv run --frozen vf-eval sdk-arm-push \
  -m "$MODEL_NAME" \
  --client.base-url "$MODEL_BASE_URL" \
  --client.api-key-var "$MODEL_API_KEY_VAR" \
  --env.agent.harness.id null \
  --env.agent.runtime.type subprocess \
  --env.agent.max-turns "${MAX_TURNS:-80}" \
  --env.taskset.seed-offset "${SEED_OFFSET:-1000}" \
  -n "${N:-8}" \
  --no-push \
  -o outputs "$@"
