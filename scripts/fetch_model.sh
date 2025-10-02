#!/usr/bin/env bash
set -euo pipefail

mkdir -p artifacts

if [[ -z "${MODEL_URL:-}" ]]; then
  echo "ERROR: MODEL_URL env var not set. Set it to a direct download URL for model.pth" >&2
  exit 1
fi

echo "Downloading model from: $MODEL_URL"
curl -L "$MODEL_URL" -o artifacts/model.pth
echo "OK: artifacts/model.pth downloaded"
