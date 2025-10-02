#!/usr/bin/env bash
set -euo pipefail
mkdir -p artifacts
: "${MODEL_URL:?Set MODEL_URL to your model.pth URL}"
: "${EMB_URL:?Set EMB_URL to your emb_matrix.pt URL}"
echo "Downloading model…"
curl -L "$MODEL_URL" -o artifacts/model.pth
echo "Downloading embedding matrix…"
curl -L "$EMB_URL" -o artifacts/emb_matrix.pt
echo "Done."
