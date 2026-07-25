# BERT GPU baseline

This directory preserves the lightweight BERT-tiny autoresearch baseline used
to validate the locked-evaluation approach before the Nemotron experiments.

The best recorded run reached macro-F1 0.7373 on a 2,000-example validation
slice. Its metric should not be compared as a single paired measurement with
the Nemotron headline result because the validation-slice sizes differ.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash run_experiment.sh
```

As in the Nemotron loop, `prepare.py` is locked and only `train.py` is
agent-editable. Both `all_attempts.tsv` and `results.tsv` are retained.
