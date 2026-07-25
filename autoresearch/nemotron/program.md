# AutoResearch program: maximize LEDGAR macro-F1 (Nemotron Nano 8B + LoRA)

You are an autonomous ML research agent. Your job is to improve a legal-clause
classifier by editing **`train.py`** and running experiments in a loop, keeping only
changes that improve the score. This mirrors Andrej Karpathy's `autoresearch`, adapted
for LEDGAR classification with NVIDIA's Llama 3.1 Nemotron Nano 8B.

## Why this is a different kind of run than the BERT baseline

Nemotron Nano 8B is much larger than BERT-tiny and is a decoder-only
**generative** model, not an encoder classifier. Instead of a softmax over
100 classes, it is prompted with the clause + the full label list and must
**generate the correct label as text**. This means:

- Training is LoRA (4-bit quantized base via bitsandbytes + peft) -- full
  fine-tuning of 7B params is unnecessary and much slower.
- Evaluation requires **generation**, not a single forward pass -- far slower
  per validation example than the BERT loop's batched logits. This is why
  `VAL_SIZE=200` (not 2000) and why the full-data run has a multi-hour
  `TIME_BUDGET`.
- A generated string must be **parsed back into one of the 100 label names**
  (see `prepare.parse_generated_label`). An unparseable generation is scored
  as a guaranteed miss -- never dropped or given partial credit. Watch the
  `unparsed=<n>/<val_size>` line in the output; a rising unparsed count as
  you change prompt/generation settings is a real regression even if
  macro_f1 looks flat, since it means the model is drifting off-format.

## The one metric (higher is better) -- SAME contract as the BERT loops

`train.py` prints a final line:

```
MACRO_F1=<float>
```

Computed identically to the BERT loops (`sklearn.metrics.f1_score`,
`average="macro"`) on the same LEDGAR validation split (same `SEED=42`), so
scores are meaningfully comparable across BERT-tiny and Nemotron runs -- keeping
in mind VAL_SIZE differs (200 here vs 2000 there), which affects variance,
not the metric's definition.

## Setup (do this once)

1. Create a working branch: `git checkout -b autoresearch/<short-tag>`.
2. Read `README.md`, `prepare.py`, and `train.py` so you understand the starting point.
3. Confirm GPU + bitsandbytes are working:
   `python -c "import torch, bitsandbytes; print(torch.cuda.is_available())"`
4. Confirm the starting config runs: `bash run_experiment.sh` -- this downloads
   `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` on first run, so it will be slower than
   subsequent ones purely from the download/cache-warm. Note the printed
   `MACRO_F1` as your current best.

## Hard rules

- **Only edit `train.py`.** `prepare.py` is LOCKED -- data, prompt format, label
  parsing, evaluation, time budget, results log. Never modify it. In particular:
  do NOT change the prompt format or label-parsing logic to make parsing easier
  in a way that changes what counts as a correct answer -- that would be gaming
  the score, not improving the model. You MAY change how train.py constructs
  training examples (e.g. few-shot exemplars in the training prompt) as long as
  the EVAL prompt still goes through `prepare.build_chat_messages` unchanged.
- **No new package installs** beyond what's already in `requirements.txt`
  (torch, transformers, datasets, scikit-learn, accelerate, peft, bitsandbytes).
- **Respect the time budget** (`prepare.TIME_BUDGET`, 14400s by default). Generation
  during eval is NOT covered by the training-time-budget callback -- if eval itself
  is taking too long, reduce `MAX_NEW_TOKENS` or reconsider VAL_SIZE-affecting knobs
  (VAL_SIZE itself is locked in prepare.py, but batching eval generation is fair game).
- **Simpler is better.** Prefer the smaller/cleaner `train.py`. A deletion that keeps
  the score is a win. Do not add complexity (e.g. multi-stage prompting, self-consistency
  voting) for a marginal gain unless you've confirmed simpler options are exhausted.

## The loop (repeat -- NEVER STOP until the human interrupts)

1. Form ONE hypothesis (e.g. "raising LoRA rank helps", "more train examples helps",
   "a clearer system prompt reduces unparsed generations", "lower generation temperature
   settings -- currently greedy decoding -- change nothing since do_sample=False already").
2. Edit `train.py` to test exactly that.
3. Run: `bash run_experiment.sh`. Read the printed `MACRO_F1` AND `unparsed=<n>/<val_size>`.
4. Decide:
   - **Improved** over your best-so-far -> `git add -A && git commit -m "<desc> macro_f1=<x>"`
     and update your best.
   - **Not improved** -> `git restore -- train.py` to discard only the tested
     implementation. Never use `git reset --hard` in the monorepo.
5. A row is appended to `results.tsv` automatically every run (commit, macro_f1,
   accuracy, unparsed, seconds, status, description) and to `all_attempts.tsv`,
   logging EVERY attempt including discarded ones. Update its pending decision
   to `kept` or `discarded`, preserve both tables, and keep `DESCRIPTION` in
   train.py meaningful.
6. On a crash (commonly OOM given the 7B model): read the last ~80 lines of
   `last_run.log`. Common fixes: lower BATCH_SIZE, raise GRAD_ACCUM_STEPS to
   compensate, lower MAX_LENGTH, or check `nvidia-smi` for other processes
   already holding GPU memory (there was a stray Jupyter process on this
   cluster consuming 53-71GB across both GPUs -- check before assuming your
   own config is the problem).
7. Go to step 1.

## Ideas to explore (a starting search space)

- **LoRA config:** raise `LORA_R` (16 -> 32/64) and `LORA_ALPHA` proportionally;
  add more `LORA_TARGET_MODULES` (currently just attention projections --
  try adding `gate_proj`, `up_proj`, `down_proj` for the MLP layers too).
- **Data:** test data volume deliberately. The accepted configuration uses the
  full 60,000-example training slice; smaller controlled subsets can still be
  useful when isolating a hypothesis.
- **Prompting:** few-shot exemplars in the training (not eval) prompt; clause
  truncation strategy for very long clauses (currently a flat `MAX_LENGTH=512`
  token cutoff -- clauses may lose the informative part if truncated poorly).
- **Optimization:** learning rate, warmup ratio, more epochs. QLoRA papers
  typically use higher LR (1e-4 to 3e-4) than full fine-tuning -- current
  default is `2e-4`.
- **Reduce unparsed generations:** if `unparsed` is nonzero, that's actual signal
  worth chasing before tuning anything else -- a model that answers off-format
  is losing points for reasons unrelated to classification quality. Try lowering
  `MAX_NEW_TOKENS` further (label names are short) or reinforcing the "exactly
  one label, nothing else" instruction during training, not just at eval time.

Start with the cheapest, fastest-to-verify ideas (LoRA target modules, learning
rate) before the expensive ones (much larger TRAIN_SIZE, which multiplies both
training and (worse) per-example generation time at eval).

## Morning review (for the human)

Open `results.tsv` sorted by `macro_f1`, inspect the best accepted state, and
read `train.py` to see what won. Compare against `../../baselines/bert` to
measure whether the larger model justifies its additional compute. Fold new
intuition back into this file's "Ideas" section.
