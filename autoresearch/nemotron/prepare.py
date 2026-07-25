"""
prepare.py  --  LOCKED FILE. The research agent must NOT modify this file.

Llama-3.1-Nemotron-Nano-8B + LoRA variant of the LEDGAR autoresearch loop.
This file owns data, prompt format, the
evaluation metric, the time budget, and the results log. The agent may only
edit train.py.

TASK FRAMING: generative classification. Nemotron is prompted with the clause
text + the full list of 100 LEDGAR label names, and must generate the
correct label as text (LoRA-finetuned to do so reliably). This is the
natural fit for a decoder-only instruct model -- no classification-head
surgery, works cleanly with 4-bit quantization + peft/LoRA.

The objective is unchanged from the BERT loop:

        MAXIMIZE  macro-F1  on a fixed validation slice.

macro-F1 is computed identically (sklearn.metrics.f1_score, average="macro")
so scores are comparable across the BERT-tiny and decoder-model runs on the same
validation set -- the ONLY difference is how predictions are produced
(argmax over logits vs. parsed generated text).
"""

import os
import re
import time
import subprocess

import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score


# ----------------------------------------------------------------------------
# FIXED CONSTANTS  (the agent must treat these as immutable)
# ----------------------------------------------------------------------------
SEED = 42
VAL_SIZE = 200              # small on purpose -- generation is much slower than a forward pass
TIME_BUDGET = 14400         # seconds per experiment (4 hours) -- raised for full 60k TRAIN_SIZE
                           # runs on 2x H100s; a 3-epoch cosine LR schedule needs to actually
                           # finish decaying (an early cutoff hurt scores in the sibling BERT
                           # loop). Lower this back down if experiments are finishing well
                           # under budget and you'd rather run more, shorter iterations.
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results.tsv")


# ----------------------------------------------------------------------------
# DEVICE
# ----------------------------------------------------------------------------
def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = pick_device()


# ----------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------
def load_data(train_size):
    """
    Load LEDGAR and return (train_ds, val_ds, label_names, num_labels).

    train_size is the ONE data knob the agent may vary (via train.py). The
    validation slice is always the same VAL_SIZE rows with the same SEED, so
    scores are comparable across experiments -- and comparable to the
    BERT-tiny loop's validation slice (same seed, same underlying dataset).
    """
    ds = load_dataset("coastalcph/lex_glue", "ledgar")
    label_names = ds["train"].features["label"].names
    num_labels = len(label_names)

    train_ds = ds["train"].shuffle(seed=SEED).select(range(train_size))
    val_ds = ds["validation"].shuffle(seed=SEED).select(range(VAL_SIZE))
    return train_ds, val_ds, label_names, num_labels


# ----------------------------------------------------------------------------
# PROMPT FORMAT  (locked -- the agent may change the model/LoRA config in
# train.py, but not how the task is posed, so scores stay comparable)
# ----------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a legal-clause classifier. Given a contract clause, respond with "
    "EXACTLY one label from the provided list -- nothing else, no punctuation, "
    "no explanation."
)


def build_user_prompt(clause_text, label_names):
    label_list = ", ".join(label_names)
    return (
        f"Labels: {label_list}\n\n"
        f"Clause: {clause_text}\n\n"
        f"Which label applies? Answer with exactly one label from the list above."
    )


def build_chat_messages(clause_text, label_names):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(clause_text, label_names)},
    ]


def build_target_text(label_name):
    """The exact text the model must learn to generate for a given gold label."""
    return label_name


# ----------------------------------------------------------------------------
# LABEL PARSING  (locked -- defines how generated text maps back to a label
# for scoring. An unparseable / off-list generation counts as a miss, not a
# free pass -- this keeps the metric honest and un-gameable.)
# ----------------------------------------------------------------------------
def parse_generated_label(generated_text, label_names):
    """
    Map a generated string back to one of label_names, or None if it can't
    be confidently matched. None is always scored as wrong (see
    evaluate_macro_f1) -- there is no partial credit for a garbled answer.
    """
    text = generated_text.strip()
    # Exact match first (case-sensitive, then case-insensitive).
    if text in label_names:
        return text
    lowered = text.lower()
    for name in label_names:
        if name.lower() == lowered:
            return name
    # Fall back to "label appears as a substring of the generation" --
    # guards against the model adding stray punctuation/whitespace/quotes.
    for name in label_names:
        if re.search(re.escape(name.lower()), lowered):
            return name
    return None


# ----------------------------------------------------------------------------
# EVALUATION  (the LOCKED objective -- do not reimplement this in train.py)
# ----------------------------------------------------------------------------
def evaluate_macro_f1(gold_labels, generated_texts, label_names):
    """
    gold_labels: list[str] of true label names, len == VAL_SIZE.
    generated_texts: list[str] of raw model generations, same length/order.

    Unparseable generations are mapped to a sentinel label guaranteed to be
    wrong for that row (never equal to the true label), so they always
    count as an error rather than being silently dropped from the metric.
    """
    name_to_idx = {name: i for i, name in enumerate(label_names)}
    gold_idx = [name_to_idx[g] for g in gold_labels]

    pred_idx = []
    unparsed = 0
    for gen, true_name in zip(generated_texts, gold_labels):
        parsed = parse_generated_label(gen, label_names)
        if parsed is None:
            unparsed += 1
            # guaranteed-wrong sentinel: any index != this row's true index
            pred_idx.append((name_to_idx[true_name] + 1) % len(label_names))
        else:
            pred_idx.append(name_to_idx[parsed])

    macro_f1 = float(f1_score(gold_idx, pred_idx, average="macro", zero_division=0))
    accuracy = float(accuracy_score(gold_idx, pred_idx))
    return {"macro_f1": macro_f1, "accuracy": accuracy, "unparsed_count": unparsed}


# ----------------------------------------------------------------------------
# RESULTS LOG
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


def log_result(macro_f1, accuracy, seconds, status, description, unparsed_count=0):
    header = "commit\tmacro_f1\taccuracy\tunparsed\tseconds\tstatus\tdescription\n"
    row = (f"{_git_commit_hash()}\t{macro_f1:.4f}\t{accuracy:.4f}\t{unparsed_count}\t"
           f"{seconds:.0f}\t{status}\t{description}\n")
    exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a") as f:
        if not exists:
            f.write(header)
        f.write(row)


# ----------------------------------------------------------------------------
# TIME BUDGET
# ----------------------------------------------------------------------------
def make_time_budget_callback():
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
