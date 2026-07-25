# NemoCounsel architecture

## Training pipeline

1. `prepare.py` loads LEDGAR, fixes the seed and validation slice, constructs
   the prompt, parses generated labels, and computes macro-F1.
2. `train.py` loads Llama 3.1 Nemotron Nano 8B in 4-bit NF4 and attaches LoRA
   adapters to the attention and MLP projection layers.
3. The autonomous agent tests one change at a time. Improved experiments are
   kept; rejected configurations are restored while their scores remain in the
   attempt history.
4. The accepted adapter and tokenizer are exported to
   `autoresearch/nemotron/adapters/ledgar-best`.

## Inference pipeline

```text
Contract text
    |
    v
FastAPI bridge (:8003)
    |-- split non-empty clause blocks
    |-- POST /v1/chat/completions
    v
vLLM (:8002)
    |-- Nemotron Nano 8B base
    |-- NemoCounsel LoRA adapter
    v
One LEDGAR provision label per clause
```

The API offers regular JSON and server-sent-event responses. Runtime LoRA keeps
the base model immutable and allows an improved adapter to be swapped by
restarting only vLLM.

## Boundaries

- Training is offline and GPU-intensive.
- vLLM owns model execution and its OpenAI-compatible API.
- FastAPI owns clause splitting, request/response shaping, and SSE streaming.
- A frontend can consume FastAPI without depending on model-server details.
