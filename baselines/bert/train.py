"""
train.py  --  THE EDITABLE FILE. This is the ONLY file the research agent may change.

GPU VARIANT: seeded from the best config found on the Mac (`mps`) loop
(macro_f1=0.7228 there), carried over as the new starting point for a real
NVIDIA GPU run. TRAIN_SIZE is raised to the 60000 max and TIME_BUDGET (in the
sibling prepare.py) is raised to 300s, since a GPU trains this tiny model far
faster than `mps`/cpu -- re-baseline once here before continuing the loop.

Contract: the LAST thing this script prints MUST be a line of the exact form
    MACRO_F1=<float>
so the runner can parse the score. (accuracy is printed too, for humans.)
"""

import time
import random

import numpy as np
import torch
import torch.nn as nn

import prepare  # LOCKED helpers: data, tokenizer, metrics, time budget, logging


# ============================================================================
# KNOBS the agent may tune (and it may also rewrite the code below).
# ============================================================================
MODEL_NAME = "google/bert_uncased_L-2_H-128_A-2"   # BERT-tiny (~4.4M params)
TRAIN_SIZE = 60000       # how many training clauses (max 60000) -- raised for GPU headroom
MAX_LENGTH = 128         # tokens per clause
EPOCHS = 3
BATCH_SIZE = 32
LR = 1e-3

WARMUP_RATIO = 0.1
LR_SCHEDULER = "cosine"

CLASS_WEIGHT_POWER = 0.75  # weight = (1/freq)^power; 1.0=inverse-freq, 0.5=sqrt, 0.25=too soft

DESCRIPTION = "GPU re-baseline: best Mac config (macro_f1=0.7228) with TRAIN_SIZE raised to 60000"


def main():
    # Reproducibility
    random.seed(prepare.SEED)
    np.random.seed(prepare.SEED)
    torch.manual_seed(prepare.SEED)

    device = prepare.DEVICE
    print("Running on:", device)

    # ---- data (locked loader; only TRAIN_SIZE varies) ----
    train_ds, val_ds, label_names, num_labels = prepare.load_data(TRAIN_SIZE)
    print(f"labels={num_labels}  train={len(train_ds)}  val={len(val_ds)}")

    # ---- tokenize ----
    from transformers import DataCollatorWithPadding
    tokenizer = prepare.get_tokenizer(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    train_tok = train_ds.map(tokenize, batched=True).rename_column("label", "labels")
    val_tok = val_ds.map(tokenize, batched=True).rename_column("label", "labels")
    collator = DataCollatorWithPadding(tokenizer)

    # ---- model ----
    from transformers import (AutoModelForSequenceClassification,
                              TrainingArguments, Trainer)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels
    ).to(device)

    # ---- class weights (inverse frequency, from train split only) ----
    label_counts = np.bincount(train_tok["labels"], minlength=num_labels).astype(np.float64)
    label_counts[label_counts == 0] = 1.0  # avoid div-by-zero for unseen-in-sample classes
    class_weights = (1.0 / label_counts) ** CLASS_WEIGHT_POWER
    class_weights = class_weights / class_weights.mean()  # normalize so avg weight ~= 1
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = nn.CrossEntropyLoss(weight=class_weights_t)
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    # ---- trainer ----
    args = TrainingArguments(
        output_dir="ledgar-run",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=64,
        learning_rate=LR,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=100,
        report_to="none",
        fp16=(device == "cuda"),
    )
    trainer = WeightedTrainer(
        model=model, args=args,
        train_dataset=train_tok, eval_dataset=val_tok,
        compute_metrics=prepare.compute_metrics,
        data_collator=collator, processing_class=tokenizer,
        callbacks=[prepare.make_time_budget_callback()],   # respect the time budget
    )

    # ---- train + evaluate (locked metric) ----
    t0 = time.time()
    trainer.train()
    seconds = time.time() - t0

    scores = prepare.evaluate_macro_f1(trainer, val_tok)
    macro_f1, accuracy = scores["macro_f1"], scores["accuracy"]

    # log for the results.tsv history
    prepare.log_result(macro_f1, accuracy, seconds, status="ok", description=DESCRIPTION)

    print(f"accuracy={accuracy:.4f}")
    # IMPORTANT: this MUST be the last line and in this exact format (runner greps it):
    print(f"MACRO_F1={macro_f1:.4f}")


if __name__ == "__main__":
    main()
