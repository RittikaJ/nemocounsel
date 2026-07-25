# NemoCounsel serving

The serving bundle loads the Nemotron base model with the trained LoRA adapter
in vLLM and exposes contract-oriented endpoints through FastAPI.

## Runtime LoRA

From the repository root:

```bash
export BASE_MODEL=nvidia/Llama-3.1-Nemotron-Nano-8B-v1
export ADAPTER_PATH="$PWD/autoresearch/nemotron/adapters/ledgar-best"
bash serving/serve_vllm_lora.sh
```

Optional variables are `HOST`, `PORT`, `MAX_MODEL_LEN`,
`GPU_MEMORY_UTILIZATION`, and `SERVED_MODEL_NAME`.

Start the bridge separately:

```bash
pip install fastapi uvicorn openai
uvicorn serving.api:app --host 0.0.0.0 --port 8003
```

Test vLLM directly with:

```bash
bash serving/smoke_test_chat.sh
```

See [`../docs/deployment.md`](../docs/deployment.md) for API examples and the
optional adapter-merge workflow.
