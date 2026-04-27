#!/usr/bin/env bash
set -euo pipefail

API_BASE="${API_BASE:-http://localhost:4000/v1}"
MODEL="${MODEL:-llama-3-8b}"
PROMPT="${PROMPT:-Say hello from vLLM in one short sentence.}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-420}"
POLL_SECONDS="${POLL_SECONDS:-5}"
LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:-}"

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required" >&2
  exit 1
fi

api_root="${API_BASE%/v1}"
health_url="${api_root}/health"
models_url="${API_BASE%/}/models"
chat_url="${API_BASE%/}/chat/completions"

headers=(-H "Content-Type: application/json")
if [[ -n "$LITELLM_MASTER_KEY" ]]; then
  headers+=(-H "Authorization: Bearer ${LITELLM_MASTER_KEY}")
fi

deadline=$((SECONDS + MAX_WAIT_SECONDS))

echo "Waiting for API at ${health_url}"
until curl -fsS "$health_url" >/dev/null; do
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for ${health_url}" >&2
    exit 1
  fi
  sleep "$POLL_SECONDS"
done

echo "Checking model list at ${models_url}"
models_file="$(mktemp)"
models_status="$(
  curl -sS \
    --max-time 30 \
    -w "%{http_code}" \
    -o "$models_file" \
    "${headers[@]}" \
    "$models_url"
)"

if [[ "$models_status" != "200" ]]; then
  echo "Could not list models; HTTP ${models_status}" >&2
  python3 -m json.tool "$models_file" 2>/dev/null || cat "$models_file" >&2
  rm -f "$models_file"
  exit 1
fi

if ! python3 - "$models_file" "$MODEL" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)

requested = sys.argv[2]
models = [item.get("id") for item in data.get("data", []) if item.get("id")]

if requested in models:
    print(f"Model '{requested}' is available.")
    raise SystemExit(0)

print(f"Model '{requested}' is not advertised by this API.", file=sys.stderr)
if models:
    print("Available models:", file=sys.stderr)
    for model in models:
        print(f"  - {model}", file=sys.stderr)
else:
    print("No models were returned by /v1/models.", file=sys.stderr)
raise SystemExit(1)
PY
then
  rm -f "$models_file"
  exit 1
fi
rm -f "$models_file"

payload="$(
  MODEL="$MODEL" PROMPT="$PROMPT" python3 - <<'PY'
import json
import os

print(json.dumps({
    "model": os.environ["MODEL"],
    "messages": [{"role": "user", "content": os.environ["PROMPT"]}],
    "temperature": 0,
    "max_tokens": 80,
}))
PY
)"

echo "Sending chat request for model '${MODEL}'"
echo "This may take several minutes if it starts a new vLLM container."

response_file="$(mktemp)"
status="$(
  curl -sS \
    --max-time "$MAX_WAIT_SECONDS" \
    -w "%{http_code}" \
    -o "$response_file" \
    "${headers[@]}" \
    -d "$payload" \
    "$chat_url"
)"

if [[ "$status" != "200" ]]; then
  echo "Chat request failed with HTTP ${status}" >&2
  python3 -m json.tool "$response_file" 2>/dev/null || cat "$response_file" >&2
  rm -f "$response_file"
  exit 1
fi

python3 - "$response_file" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)

message = data["choices"][0]["message"]["content"]
print()
print("Model response:")
print(message)
PY

rm -f "$response_file"
