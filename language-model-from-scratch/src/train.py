"""Train and compare character-level language models on the corpus.

- N-gram models of order 1-4 (add-k smoothing + backoff)
- Neural LM (embeddings -> tanh hidden -> softmax), trained with
  hand-derived backpropagation

Evaluation: held-out perplexity (the standard LM metric — the effective
branching factor per character). Generation: samples at several
temperatures, with and without top-k, saved to results/samples.txt.

Run:  python data/download_data.py && python src/train.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lm import NeuralLM, NGramLM, Vocab

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus.txt"
RESULTS = ROOT / "results"
SEED = 0
MAX_CHARS = 200_000        # cap for quick runs; full corpus also works


def main():
    RESULTS.mkdir(exist_ok=True)
    if not CORPUS.exists():
        raise SystemExit("No corpus found. Run: python data/download_data.py")
    text = CORPUS.read_text(encoding="utf-8")[:MAX_CHARS]
    vocab = Vocab(text)
    ids = vocab.encode(text)
    split = int(0.9 * len(ids))
    train_ids, val_ids = ids[:split], ids[split:]
    print(f"corpus {len(ids):,} chars | vocab {vocab.size} | "
          f"train {len(train_ids):,} / val {len(val_ids):,}")

    # ---- n-gram family ----------------------------------------------------
    ppl = {}
    ngrams = {}
    for order in [1, 2, 3, 4]:
        model = NGramLM(order, vocab.size, k=0.5).fit(train_ids)
        ngrams[order] = model
        ppl[f"ngram_{order}"] = round(model.perplexity(val_ids), 2)
        print(f"{order}-gram val perplexity: {ppl[f'ngram_{order}']}")

    # ---- neural LM --------------------------------------------------------
    print("training neural LM ...")
    nlm = NeuralLM(vocab.size, context=12, emb_dim=24, hidden=128, seed=SEED)
    nlm.fit(train_ids, epochs=10, batch_size=128, lr=0.1, momentum=0.9,
            seed=SEED, val_ids=val_ids)
    ppl["neural_lm"] = round(nlm.perplexity(val_ids), 2)
    print(f"neural LM val perplexity: {ppl['neural_lm']}")

    # ---- figures ----------------------------------------------------------
    names = list(ppl.keys())
    plt.figure(figsize=(7, 4))
    bars = plt.bar(names, [ppl[n] for n in names],
                   color=["#8896AB"] * 4 + ["#B1624E"])
    plt.bar_label(bars, fmt="%.1f")
    plt.ylabel("validation perplexity (lower is better)")
    plt.title("Character-level language models")
    plt.tight_layout()
    plt.savefig(RESULTS / "perplexity_comparison.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(1, len(nlm.history["train_loss"]) + 1),
             nlm.history["train_loss"], "o-", label="train loss (nats/char)")
    plt.plot(np.arange(1, len(nlm.history["val_ppl"]) + 1),
             np.log(nlm.history["val_ppl"]), "s-",
             label="log val perplexity")
    plt.xlabel("epoch")
    plt.legend()
    plt.title("Neural LM training")
    plt.tight_layout()
    plt.savefig(RESULTS / "training_curve.png", dpi=150)
    plt.close()

    # ---- generation -------------------------------------------------------
    best_ngram = ngrams[4]
    lines = []
    for temp in [0.7, 1.0, 1.3]:
        lines.append(f"=== neural LM | temperature {temp} ===")
        lines.append(nlm.generate(vocab, "The ", 300, temperature=temp,
                                  seed=SEED))
        lines.append("")
    lines.append("=== neural LM | temperature 0.8, top_k 12 ===")
    lines.append(nlm.generate(vocab, "The ", 300, temperature=0.8, top_k=12,
                              seed=SEED))
    lines.append("")
    lines.append("=== 4-gram | temperature 1.0 ===")
    lines.append(best_ngram.generate(vocab, "The ", 300, seed=SEED))
    (RESULTS / "samples.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote generation samples -> {RESULTS / 'samples.txt'}")

    with open(RESULTS / "metrics.json", "w") as f:
        json.dump({"val_perplexity": ppl,
                   "corpus_chars": len(ids),
                   "vocab_size": vocab.size,
                   "neural_config": {"context": 12, "emb_dim": 24,
                                     "hidden": 128, "epochs": 10}},
                  f, indent=2)
    print("PERPLEXITY_TABLE " + json.dumps(ppl))


if __name__ == "__main__":
    main()
