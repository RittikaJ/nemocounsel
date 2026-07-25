#!/usr/bin/env bash
set -euo pipefail

# Runtime LoRA serving (fastest POC): no merge required.
# Exposes OpenAI-compatible API at /v1/*
#
# Env vars you can override:
#   BASE_MODEL, ADAPTER_PATH, HOST, PORT, MAX_MODEL_LEN,
#   GPU_MEMORY_UTILIZATION, SERVED_MODEL_NAME

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_MODEL="${BASE_MODEL:-nvidia/Llama-3.1-Nemotron-Nano-8B-v1}"
ADAPTER_PATH="${ADAPTER_PATH:-$HERE/../autoresearch/nemotron/adapters/ledgar-best}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8002}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-clause_lora}"

if ! command -v vllm >/dev/null 2>&1; then
  echo "ERROR: vllm is not installed in this environment." >&2
  echo "Install example: pip install vllm" >&2
  exit 1
fi

if [ ! -d "$ADAPTER_PATH" ]; then
  echo "ERROR: adapter folder not found: $ADAPTER_PATH" >&2
  exit 1
fi

echo "Starting vLLM with runtime LoRA..."
echo "BASE_MODEL=$BASE_MODEL"
echo "ADAPTER_PATH=$ADAPTER_PATH"
echo "HOST=$HOST PORT=$PORT MODEL_NAME=$SERVED_MODEL_NAME"

exec vllm serve "$BASE_MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --enable-lora \
  --lora-modules "${SERVED_MODEL_NAME}=${ADAPTER_PATH}" \
  --served-model-name "$SERVED_MODEL_NAME"
