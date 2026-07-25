#!/usr/bin/env bash
set -euo pipefail

# Quick test for vLLM OpenAI-compatible endpoint.
#
# Env vars:
#   VLLM_URL    default http://localhost:8002
#   MODEL_NAME  default clause_lora
#   PROMPT      default legal clause example

VLLM_URL="${VLLM_URL:-http://localhost:8002}"
MODEL_NAME="${MODEL_NAME:-clause_lora}"
PROMPT="${PROMPT:-Classify this clause type: The receiving party shall keep all non-public information confidential for 3 years. Reply with one label only.}"

curl -sS "${VLLM_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL_NAME}\",
    \"messages\": [
      {\"role\": \"system\", \"content\": \"You are a legal clause classifier.\"},
      {\"role\": \"user\", \"content\": \"${PROMPT}\"}
    ],
    \"temperature\": 0,
    \"max_tokens\": 32
  }"

echo
