# Nemotron autoresearch

This directory contains NemoCounsel's autonomous QLoRA research loop for
`nvidia/Llama-3.1-Nemotron-Nano-8B-v1`.

## File contract

| File | Role |
|---|---|
| `prepare.py` | Locked dataset, prompt, parser, validation, metric, and result logger |
| `train.py` | Agent-editable model and training experiment |
| `program.md` | Research-agent rules and hypothesis queue |
| `run_experiment.sh` | Executes a run and records its score |
| `all_attempts.tsv` | Every completed experiment, including rejected attempts |
| `results.tsv` | Scored run history |
| `render_attempts_report.py` | Regenerates the root result visualization |

## Accepted configuration

- Base model: Llama 3.1 Nemotron Nano 8B v1
- Training examples: 60,000
- Context: 768 tokens with target-label space reserved
- LoRA: rank 32, alpha 64, attention and MLP projections
- Training: two epochs, effective batch size 16, cosine schedule
- Validation: macro-F1 0.8246, accuracy 0.9150, unparsed 0/200

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash run_experiment.sh
```

The model and dataset are downloaded on demand. Generated adapters, checkpoints,
environments, and raw logs are ignored by Git.

To regenerate the experiment table image:

```bash
python render_attempts_report.py
```
