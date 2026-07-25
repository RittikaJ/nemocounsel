# Deployment guide

## Prerequisites

- NVIDIA GPU supported by the installed PyTorch/vLLM build
- `vllm`, `fastapi`, `uvicorn`, and the OpenAI Python client
- Access to `nvidia/Llama-3.1-Nemotron-Nano-8B-v1`
- A reproduced NemoCounsel adapter directory

The repository excludes model and adapter weights. Reproduce the accepted
adapter with `autoresearch/nemotron/train.py` or set `ADAPTER_PATH` to an
equivalent exported PEFT adapter.

## Start vLLM

```bash
export BASE_MODEL=nvidia/Llama-3.1-Nemotron-Nano-8B-v1
export ADAPTER_PATH="$PWD/autoresearch/nemotron/adapters/ledgar-best"
export PORT=8002
bash serving/serve_vllm_lora.sh
```

The served model name defaults to `clause_lora`. Override it with
`SERVED_MODEL_NAME`.

## Start the FastAPI bridge

```bash
export VLLM_BASE_URL=http://localhost:8002/v1
export MODEL_NAME=clause_lora
uvicorn serving.api:app --host 0.0.0.0 --port 8003
```

## Health check

```bash
curl http://localhost:8003/health
```

## Classify a contract

```bash
curl -X POST http://localhost:8003/classify \
  -H 'Content-Type: application/json' \
  -d '{
    "contract": "Neither party is liable for delays caused by events beyond its reasonable control.\n\nThis agreement is governed by the laws of the State of Delaware."
  }'
```

For progressive UI updates, send the same payload to
`/classify/stream`. The response is an SSE stream with `start`, `progress`,
`result`, and `done` events.

## Merge the adapter (optional)

Runtime LoRA is recommended for the hackathon because adapters can be swapped
without rewriting the base model. To create a standalone merged directory:

```bash
python serving/merge_lora.py \
  --base-model nvidia/Llama-3.1-Nemotron-Nano-8B-v1 \
  --adapter autoresearch/nemotron/adapters/ledgar-best \
  --output models/nemotron-ledgar-merged \
  --dtype bfloat16 \
  --device-map auto
```

The merged output remains ignored by Git.

## Production considerations

- Restrict CORS instead of using the prototype wildcard.
- Add authentication, request-size limits, timeouts, and structured logging.
- Replace newline-based clause splitting with a document-aware parser.
- Run a larger held-out evaluation before legal or production use.
