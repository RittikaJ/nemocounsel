# AutoResearch program: maximize LEDGAR macro-F1 (NVIDIA GPU variant)

You are an autonomous ML research agent. Your job is to improve a legal-clause
classifier by editing **`train.py`** and running experiments in a loop, keeping only
changes that improve the score. This mirrors Andrej Karpathy's `autoresearch`, adapted
for LEDGAR classification -- this folder is the GPU-tuned sibling of the original
Mac (`mps`) loop.

## The one metric (higher is better)

`train.py` prints a final line:

```
MACRO_F1=<float>
```

**Maximize `macro_f1`** on the fixed validation slice. Macro-F1 averages every class
equally, so it rewards handling the *rare* clause types too -- not just the common ones.
Accuracy is printed for context but is NOT the objective (it is misleading on this
imbalanced dataset).

## Setup (do this once)

1. Create a working branch: `git checkout -b autoresearch/<short-tag>`.
2. Read `README.md`, `prepare.py`, and `train.py` so you understand the starting point.
3. Confirm `python -c "import torch; print(torch.cuda.is_available())"` prints `True`
   in whichever Python `run_experiment.sh` resolves (see its comments -- it checks
   `./.venv`, then `../.venv`, then `python3` on PATH).
4. Confirm the starting config runs: `bash run_experiment.sh` -- note the printed
   `MACRO_F1` as your current best. `train.py` starts from the best config found on
   the Mac loop (macro_f1=0.7228 there) with `TRAIN_SIZE` raised to 60000, so your
   first run is a **re-baseline on this hardware**, not a fresh start -- the score
   may differ from 0.7228 since the GPU's `TIME_BUDGET` (300s, in `prepare.py`) and
   full 60k data are new variables.

## Hard rules

- **Only edit `train.py`.** `prepare.py` is LOCKED (data, tokenizer, evaluation, time
  budget, results log). Never modify it -- it guarantees the score can't be gamed.
- **No new package installs** beyond what's already in the environment
  (transformers, datasets, scikit-learn, torch, accelerate). If you need one of these
  installed fresh (`uv pip install ...`), that's fine -- it's not a "new" package, it's
  the required baseline set.
- **Respect the time budget.** Each run is capped (see `prepare.TIME_BUDGET`, 300s
  here vs 120s on the Mac loop); keep experiments fast so you can do many.
- **Simpler is better.** All else equal, prefer the smaller/cleaner `train.py`. Reject
  changes that add lots of complexity for a marginal gain. A deletion that keeps the
  score is a win.

## The loop (repeat -- NEVER STOP until the human interrupts)

1. Form ONE hypothesis (e.g. "distilbert instead of bert-tiny now that we have GPU
   headroom", "bigger batch size now that time budget is generous", "a larger MAX_LENGTH
   captures more clause context").
2. Edit `train.py` to test exactly that.
3. Run: `bash run_experiment.sh`. Read the printed `MACRO_F1`.
4. Decide:
   - **Improved** over your best-so-far -> `git add -A && git commit -m "<desc> macro_f1=<x>"`
     and update your best.
   - **Not improved** -> `git reset --hard` to discard the change.
5. A row is appended to `results.tsv` automatically every run (commit, macro_f1,
   accuracy, seconds, status, description) -- and to `all_attempts.tsv` (untracked,
   survives `git reset --hard`), which logs EVERY attempt including discarded ones.
   Keep the description in `train.py`'s `DESCRIPTION` variable meaningful.
6. On a crash: read the last ~50 lines of `last_run.log`, try to fix it, rerun. If an
   approach keeps failing, abandon it and try something else.
7. Go to step 1.

## Ideas to explore (a starting search space -- GPU-specific opportunities first)

With real GPU headroom (vs. a Mac's `mps`/cpu), these become newly cheap:

- **Bigger model:** swap `MODEL_NAME` to `distilbert-base-uncased` or a small
  RoBERTa -- was too slow to be worth it on `mps`, may pay off here. Fast-tokenizer
  models only.
- **Longer sequences:** raise `MAX_LENGTH` (128 -> 256/384) -- legal clauses often
  get truncated at 128 tokens; a GPU affords the extra compute.
- **Bigger batches:** raise `BATCH_SIZE` (32 -> 64/128) with proportionally scaled LR.
- **Full data, more epochs:** now that 60000 rows fit inside the time budget with
  room to spare, try more epochs on the full set instead of truncating early.

Carried over from the Mac loop (still worth re-verifying since TIME_BUDGET/TRAIN_SIZE
changed):

- **Loss:** class-weighted cross-entropy (weight ∝ (1/freq)^power -- power=0.75 won
  on the Mac loop, sqrt=0.5 was close behind; power=1.0 (full inverse-freq) and
  power=0.25 both underperformed. Label smoothing did NOT help on the Mac loop.
- **Optimization:** learning rate (5e-4 -> 1e-3 kept improving on the Mac loop, 1.5e-3
  started to hurt -- may shift again with a different batch size or model). Cosine
  schedule + 10% warmup beat the default linear/no-warmup schedule. Watch for the LR
  schedule finishing its decay within however long the run actually takes -- an
  incomplete decay (time-budget cutoff mid-schedule) changes the picture, so re-tune
  EPOCHS/TRAIN_SIZE together, not in isolation.
- **Data:** larger TRAIN_SIZE helped monotonically on the Mac loop up to the point
  where it started truncating the LR schedule too early; oversampling rare classes
  is still untried.

Start with the cheapest high-leverage ideas before expensive ones (bigger model,
much longer sequences).

## Morning review (for the human)

Open `results.tsv` sorted by `macro_f1`, check out the best commit, and read `train.py`
there to see what won. Fold new intuitions back into this file's "Ideas" section.
