"""Generate data, train the transformer and its ablations, benchmark against
classical baselines, and write results/metrics.json, results/results.md and
results/examples.png.

    python src/train_eval.py            # full run (CI)
    python src/train_eval.py --quick    # small smoke run
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from traj_baselines import RidgeForecaster, constant_acceleration, constant_velocity
from traj_dataset import build_dataset, dense_traffic_config
from traj_metrics import summarize, top1
from traj_model import TrainConfig, TrajectoryTransformer, predict, train_model

RESULTS = Path(__file__).resolve().parents[1] / "results"


def evaluate(pred, scores, data) -> dict:
    """Metrics on all samples and on the lane-change subset; K-mode and top-1."""
    out = {}
    lc = data["lc"]
    for name, sel in (("all", np.ones(len(lc), bool)), ("lane_change", lc)):
        if pred.ndim == 4:
            res = {f"{k}@K": v for k, v in summarize(pred[sel], data["fut"][sel]).items()}
            res.update({f"{k}@1": v for k, v in summarize(top1(pred, scores)[sel], data["fut"][sel]).items()})
        else:
            res = {f"{k}@1": v for k, v in summarize(pred[sel], data["fut"][sel]).items()}
        out[name] = res
    return out


def plot_examples(data, pred, scores, path, n=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = np.where(data["lc"])[0][:n]
    fig, axes = plt.subplots(1, len(idx), figsize=(4 * len(idx), 3.2))
    for ax, i in zip(np.atleast_1d(axes), idx):
        h, m = data["hist"][i], data["mask"][i]
        for a in np.where(m)[0][1:]:
            ax.plot(h[a, :, 0], h[a, :, 1], color="0.7", lw=1)
        ax.plot(h[0, :, 0], h[0, :, 1], "k-", lw=2, label="history")
        ax.plot(data["fut"][i, :, 0], data["fut"][i, :, 1], "g-", lw=2, label="truth")
        p = np.exp(scores[i] - scores[i].max())
        p /= p.sum()
        for k in range(pred.shape[1]):
            ax.plot(pred[i, k, :, 0], pred[i, k, :, 1], "--", lw=1 + 2 * p[k], color="tab:blue")
        ax.set_xlim(-70, 110)
        ax.set_ylim(-9, 9)
        ax.set_xlabel("x (m)")
    np.atleast_1d(axes)[0].set_ylabel("y (m)")
    np.atleast_1d(axes)[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(quick: bool = False):
    RESULTS.mkdir(exist_ok=True)
    n_train, n_test, n_shift, epochs = (40, 15, 10, 2) if quick else (500, 100, 80, 20)

    t0 = time.time()
    train = build_dataset(n_train, seed=1)
    test = build_dataset(n_test, seed=2)
    shift = build_dataset(n_shift, seed=3, cfg=dense_traffic_config())
    print(f"data: train {len(train['fut'])}, test {len(test['fut'])}, dense-traffic {len(shift['fut'])} "
          f"samples ({time.time() - t0:.0f}s); lane-change share {test['lc'].mean():.2f}")

    results = {"n_train": int(len(train["fut"])), "n_test": int(len(test["fut"])),
               "n_shift": int(len(shift["fut"])), "models": {}}

    ridge = RidgeForecaster(1.0).fit(train["hist"], train["fut"])
    for name, fn in (("constant_velocity", constant_velocity),
                     ("constant_acceleration", constant_acceleration),
                     ("ridge", ridge.predict)):
        results["models"][name] = {"test": evaluate(fn(test["hist"]), None, test),
                                   "dense_traffic": evaluate(fn(shift["hist"]), None, shift)}

    variants = {
        "transformer_K6": dict(n_modes=6, use_social=True),
        "transformer_K6_no_social": dict(n_modes=6, use_social=False),
        "transformer_K1": dict(n_modes=1, use_social=True),
    }
    best_pred = None
    for name, kw in variants.items():
        print(f"training {name}")
        torch.manual_seed(0)
        model = TrajectoryTransformer(**kw)
        t1 = time.time()
        hist = train_model(model, train, TrainConfig(epochs=epochs))
        p_test, s_test = predict(model, test)
        p_shift, s_shift = predict(model, shift)
        results["models"][name] = {"test": evaluate(p_test, s_test, test),
                                   "dense_traffic": evaluate(p_shift, s_shift, shift),
                                   "train_loss": hist, "train_seconds": time.time() - t1,
                                   "params": sum(p.numel() for p in model.parameters())}
        if name == "transformer_K6":
            best_pred = (p_test, s_test)

    (RESULTS / "metrics.json").write_text(json.dumps(results, indent=2))
    plot_examples(test, *best_pred, RESULTS / "examples.png")
    md = to_markdown(results)
    (RESULTS / "results.md").write_text(md)
    print(md)
    return results


def to_markdown(results: dict) -> str:
    lines = [f"Train {results['n_train']} / test {results['n_test']} / dense-traffic "
             f"{results['n_shift']} samples. Metrics in metres; MR = miss rate (final error > 2 m).", ""]
    for split in ("test", "dense_traffic"):
        for subset in ("all", "lane_change"):
            lines += [f"### {split} / {subset}", "",
                      "| model | minADE@6 | minFDE@6 | MR@6 | ADE@1 | FDE@1 | MR@1 |",
                      "|---|---|---|---|---|---|---|"]
            for name, r in results["models"].items():
                m = r[split][subset]
                k = [f"{m[c]:.3f}" if c in m else "--" for c in ("minADE@K", "minFDE@K", "MR@K")]
                one = [f"{m[c]:.3f}" for c in ("minADE@1", "minFDE@1", "MR@1")]
                lines.append(f"| {name} | " + " | ".join(k + one) + " |")
            lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
