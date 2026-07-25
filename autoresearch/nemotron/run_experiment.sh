#!/usr/bin/env bash
# Runs one experiment: executes the (agent-edited) train.py using a Python
# environment with torch/transformers/datasets/scikit-learn/peft/bitsandbytes
# installed, tees all output to last_run.log, and prints the parsed macro-F1.
#
# Usage:  bash run_experiment.sh
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve a Python interpreter, in order of preference:
#   1. A venv local to this folder (./.venv) -- create one with:
#        uv venv && uv pip install -r requirements.txt
#   2. Whatever `python3` is on PATH (e.g. an existing conda env with the deps)
if [ -x "$HERE/.venv/bin/python" ]; then
  PY="$HERE/.venv/bin/python"
else
  PY="$(command -v python3 || true)"
fi

if [ -z "${PY:-}" ] || [ ! -x "$PY" ]; then
  echo "ERROR: no usable Python found (checked ./.venv, PATH)." >&2
  echo "Create one with: uv venv && uv pip install -r requirements.txt" >&2
  exit 1
fi

cd "$HERE"
echo "=== running train.py at $(date) (python: $PY) ==="
"$PY" train.py 2>&1 | tee last_run.log
status=${PIPESTATUS[0]}

echo "=========================================="
if [ "$status" -ne 0 ]; then
  echo "RUN FAILED (exit $status). See last_run.log (last 80 lines):"
  tail -n 80 last_run.log
  exit "$status"
fi

# Parse the score contract line: MACRO_F1=<float>
score=$(grep -oE 'MACRO_F1=[0-9.]+' last_run.log | tail -n1 | cut -d= -f2)
if [ -z "$score" ]; then
  echo "WARNING: no MACRO_F1= line found in output." >&2
  exit 1
fi
echo "PARSED macro_f1 = $score"

# Append every attempt to the full history log. The decision starts as pending;
# the autoresearch accept/reject step updates it to kept or discarded.
accuracy=$(grep -oE 'accuracy=[0-9.]+' last_run.log | tail -n1 | cut -d= -f2)
unparsed=$(grep -oE 'unparsed=[0-9]+/[0-9]+' last_run.log | tail -n1)
description=$(grep -m1 '^DESCRIPTION = ' train.py | sed -E 's/^DESCRIPTION = "(.*)"$/\1/')
attempts_file="$HERE/all_attempts.tsv"
if [ ! -f "$attempts_file" ]; then
  echo -e "timestamp\tmacro_f1\taccuracy\tunparsed\tdecision\tdescription" > "$attempts_file"
fi
echo -e "$(date +%s)\t$score\t${accuracy:-NA}\t${unparsed:-NA}\tpending\t$description" >> "$attempts_file"
