#!/usr/bin/env python3
"""
Merge a PEFT LoRA adapter into a base model and save a standalone merged model.

Example:
python vllm_poc/merge_lora.py \
  --base-model nvidia/Llama-3.1-Nemotron-Nano-8B-v1 \
  --adapter autoresearch/nemotron/adapters/ledgar-best \
  --output models/nemotron-ledgar-merged
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DTYPE_MAP = {
    "auto": "auto",
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base-model", required=True, help="Path or HF ID of base model")
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter directory")
    parser.add_argument("--output", required=True, help="Output directory for merged model")
    parser.add_argument(
        "--dtype",
        choices=tuple(DTYPE_MAP.keys()),
        default="bfloat16",
        help="Model dtype while loading/merging",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help='Transformers device_map (default: "auto"). Use "cpu" to avoid occupying GPUs.',
    )
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = DTYPE_MAP[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    peft_model = PeftModel.from_pretrained(base_model, args.adapter)
    merged_model = peft_model.merge_and_unload()

    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    print(f"Merged model saved to: {output_dir}")


if __name__ == "__main__":
    main()
