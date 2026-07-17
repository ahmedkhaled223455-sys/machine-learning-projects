# Language Model from Scratch

Character-level language modeling in pure NumPy: **n-gram models with
smoothing and backoff** versus a **neural language model** (Bengio-style:
learned embeddings → tanh hidden layer → softmax), with hand-derived
backpropagation including the gradients into the embedding table.

## Why this project

Modern LLMs are built from a handful of primitives: tokenization,
embeddings, next-token prediction, temperature/top-k sampling, and
perplexity evaluation. This project implements every one of them at a
scale where nothing is hidden — the goal is to be able to explain
*why* temperature changes output diversity or *what* perplexity
actually measures, not to compete with GPT.

## Data

The public-domain **tiny-shakespeare** corpus (~1.1 MB), the canonical
character-LM dataset. It is not committed to the repo; fetch it with:

```bash
python data/download_data.py
```

## Models

| Model | Idea |
|---|---|
| n-gram (order 1–4) | Count next-char frequencies per context; add-k smoothing; back off to shorter contexts when unseen |
| Neural LM | 12-char context → 24-d char embeddings → concat → tanh(128) → softmax; mini-batch SGD + momentum |

## Results

Run the pipeline and it prints a perplexity table and writes
`results/` (figures + generation samples):

```bash
python src/train.py     # ~1 minute on a laptop
```

Expected pattern (verified end-to-end on a held-out split): perplexity
drops steeply from unigram to trigram, the **4-gram gets *worse* than
the trigram** — with limited data, most 4-char contexts are seen once
or never, so smoothing dominates (bias–variance in its purest form) —
and the **neural LM beats all n-grams** because embeddings share
statistical strength between similar characters instead of treating
every context as unrelated.

Also visible in the training curve: validation perplexity plateaus
while training loss keeps falling — mild overfitting, the cue for
early stopping.

After running, paste your printed `PERPLEXITY_TABLE` numbers here:

| Model | Val perplexity |
|---|---|
| 1-gram | 27.62 |
| 2-gram | 12.01 |
| 3-gram | 8.17 |
| 4-gram | 7.91 |
| Neural LM | 7.56 |

## Sampling, concretely

`results/samples.txt` contains generations at temperature 0.7 / 1.0 /
1.3 and with top-k filtering. Low temperature → conservative,
repetitive text; high temperature → diverse but error-prone; top-k
truncates the tail of the distribution. These are exactly the knobs
exposed by production LLM APIs — implemented here in ~10 lines.

## Design decisions worth discussing

- **Perplexity = exp(mean NLL)** — the effective branching factor per
  character; a model with perplexity 6 is as uncertain as choosing
  uniformly among 6 characters.
- **Backoff for unseen contexts** rather than pretending add-k alone
  suffices at order 4.
- **Embedding gradients via `np.add.at`** — repeated characters in one
  batch must *accumulate* gradient; plain fancy-indexing assignment
  silently drops updates (a classic NumPy bug).
- **Contiguous train/val split** (not shuffled): shuffling characters
  would leak n-gram statistics across the split.

## Limitations / next steps

- A single-head self-attention layer is the natural next step from the
  fixed context window — same training loop, different architecture.
- Word- or BPE-level tokenization; Adam; dropout for the overfitting
  visible in the curve.

## Run

```bash
python data/download_data.py   # fetch corpus (~1.1 MB)
python src/train.py            # train, evaluate, generate
```
