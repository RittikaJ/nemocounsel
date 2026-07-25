#!/usr/bin/env bash
# Runs one experiment: executes the (agent-edited) train.py using a Python
# environment with torch/transformers/datasets/scikit-learn/accelerate
# installed, tees all output to last_run.log, and prints the parsed macro-F1.
#
# Usage:  bash run_experiment.sh
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve a Python interpreter, in order of preference:
#   1. A venv local to this folder (./.venv) -- create one on the GPU box with:
#        uv venv && uv pip install torch transformers datasets scikit-learn accelerate
#   2. A sibling ../.venv (matches the layout of the original Mac project)
#   3. Whatever `python3` is on PATH (e.g. an existing conda/system env with the deps)
if [ -x "$HERE/.venv/bin/python" ]; then
  PY="$HERE/.venv/bin/python"
elif [ -x "$HERE/../.venv/bin/python" ]; then
  PY="$HERE/../.venv/bin/python"
else
  PY="$(command -v python3 || true)"
fi

if [ -z "${PY:-}" ] || [ ! -x "$PY" ]; then
  echo "ERROR: no usable Python found (checked ./.venv, ../.venv, PATH)." >&2
  echo "Create one with: uv venv && uv pip install torch transformers datasets scikit-learn accelerate" >&2
  exit 1
fi

cd "$HERE"
echo "=== running train.py at $(date) (python: $PY) ==="
"$PY" train.py 2>&1 | tee last_run.log
status=${PIPESTATUS[0]}

echo "=========================================="
if [ "$status" -ne 0 ]; then
  echo "RUN FAILED (exit $status). See last_run.log (last 50 lines):"
  tail -n 50 last_run.log
  exit "$status"
fi

# Parse the score contract line: MACRO_F1=<float>
score=$(grep -oE 'MACRO_F1=[0-9.]+' last_run.log | tail -n1 | cut -d= -f2)
if [ -z "$score" ]; then
  echo "WARNING: no MACRO_F1= line found in output." >&2
  exit 1
fi
echo "PARSED macro_f1 = $score"

# Append every attempt (kept or discarded) to an untracked full history log,
# so `git reset --hard` on rejected experiments doesn't erase them from results.tsv.
accuracy=$(grep -oE 'accuracy=[0-9.]+' last_run.log | tail -n1 | cut -d= -f2)
description=$(grep -m1 '^DESCRIPTION = ' train.py | sed -E 's/^DESCRIPTION = "(.*)"$/\1/')
attempts_file="$HERE/all_attempts.tsv"
if [ ! -f "$attempts_file" ]; then
  echo -e "timestamp\tmacro_f1\taccuracy\tdescription" > "$attempts_file"
fi
echo -e "$(date +%s)\t$score\t${accuracy:-NA}\t$description" >> "$attempts_file"
