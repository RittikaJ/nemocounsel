# NemoCounsel Medium report

Run `python3 medium_report/build_report.py` from the experiment root to rebuild
the article, derived TSV files, charts, and self-contained HTML.

The publication dataset intentionally contains only the six Nemotron experiments.
The raw experiment ledger is not rewritten.

Install the report-only dependencies with:

```bash
python3 -m pip install -r medium_report/requirements-report.txt
```

The builder reparses `last_run.log` when that local file is present. The
checked-in `data/final_training_history.tsv` makes the graphics reproducible
from a clean repository clone where the raw training log is intentionally
absent.
