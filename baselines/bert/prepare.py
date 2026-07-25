"""
prepare.py  --  LOCKED FILE. The research agent must NOT modify this file.

This is the "fixed" half of the AutoResearch setup (Karpathy-style):
data loading, tokenizer, the evaluation metric, the time budget, and the
results log. Locking it guarantees the agent cannot game the score by
changing how the model is graded -- it can only change train.py.

The objective for this project (LEDGAR clause classification) is:

        MAXIMIZE  macro-F1  on a fixed validation slice.

(Karpathy's original minimizes val_bpb; we invert it: higher macro-F1 is better.)

GPU VARIANT: same contract as the Mac version, but TIME_BUDGET is raised
since a real NVIDIA GPU trains far faster than BERT-tiny on `mps`/cpu --
120s barely lets a GPU run finish an epoch on 40k rows. Raise further if
your card is faster / you use a bigger model.
"""

import os
import time
import subprocess

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score


# ----------------------------------------------------------------------------
# FIXED CONSTANTS  (the agent must treat these as immutable)
# ----------------------------------------------------------------------------
SEED = 42
VAL_SIZE = 2000            # the validation slice is FIXED so every experiment is graded equally
TIME_BUDGET = 300          # seconds per experiment. Karpathy uses 300 on an H100 -- matched here
                           # since we're targeting a real NVIDIA GPU. Raise further if needed.
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results.tsv")


# ----------------------------------------------------------------------------
# DEVICE
# ----------------------------------------------------------------------------
def pick_device():
    """cuda (GPU) -> mps (Apple) -> cpu, exactly like the source notebook."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = pick_device()


# ----------------------------------------------------------------------------
# DATA  (identical calls to the LEDGAR_first_finetune notebook)
# ----------------------------------------------------------------------------
def load_data(train_size):
    """
    Load LEDGAR and return (train_ds, val_ds, label_names, num_labels).

    train_size is the ONE data knob the agent may vary (via train.py). The
    validation slice is always the same VAL_SIZE rows with the same SEED, so
    scores are comparable across experiments.
    """
    ds = load_dataset("coastalcph/lex_glue", "ledgar")
    label_names = ds["train"].features["label"].names
    num_labels = len(label_names)

    train_ds = ds["train"].shuffle(seed=SEED).select(range(train_size))
    val_ds = ds["validation"].shuffle(seed=SEED).select(range(VAL_SIZE))
    return train_ds, val_ds, label_names, num_labels


def get_tokenizer(model_name):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name)


# ----------------------------------------------------------------------------
# EVALUATION  (the LOCKED objective -- do not reimplement this in train.py)
# ----------------------------------------------------------------------------
def compute_metrics(eval_pred):
    """Standard classification metrics. macro_f1 is the objective; accuracy is FYI."""
    preds = np.argmax(eval_pred.predictions, axis=1)
    gold = eval_pred.label_ids
    return {
        "accuracy": accuracy_score(gold, preds),
        "macro_f1": f1_score(gold, preds, average="macro", zero_division=0),
    }


def evaluate_macro_f1(trainer, val_tok):
    """
    Run the given (already-trained) Trainer on the fixed validation set and
    return {"macro_f1":..., "accuracy":...}. This is what train.py must report.
    """
    out = trainer.predict(val_tok)
    preds = np.argmax(out.predictions, axis=1)
    gold = out.label_ids
    return {
        "macro_f1": float(f1_score(gold, preds, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(gold, preds)),
    }


# ----------------------------------------------------------------------------
# RESULTS LOG  (append one row per experiment -- Karpathy's results.tsv)
# ----------------------------------------------------------------------------
def _git_commit_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(__file__),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "nogit"


def log_result(macro_f1, accuracy, seconds, status, description):
    """Append a tab-separated row: commit, macro_f1, accuracy, seconds, status, description."""
    header = "commit\tmacro_f1\taccuracy\tseconds\tstatus\tdescription\n"
    row = f"{_git_commit_hash()}\t{macro_f1:.4f}\t{accuracy:.4f}\t{seconds:.0f}\t{status}\t{description}\n"
    exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a") as f:
        if not exists:
            f.write(header)
        f.write(row)


# ----------------------------------------------------------------------------
# TIME BUDGET  (stop training near TIME_BUDGET so slow configs still return a score)
# ----------------------------------------------------------------------------
def make_time_budget_callback():
    """A TrainerCallback that ends training once TIME_BUDGET seconds have elapsed."""
    from transformers import TrainerCallback

    class _TimeBudget(TrainerCallback):
        def __init__(self, budget):
            self.budget = budget
            self.start = None

        def on_train_begin(self, args, state, control, **kwargs):
            self.start = time.time()

        def on_step_end(self, args, state, control, **kwargs):
            if self.start is not None and (time.time() - self.start) > self.budget:
                control.should_training_stop = True
            return control

    return _TimeBudget(TIME_BUDGET)
