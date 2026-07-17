"""Train the from-scratch MLP on two nonlinear 2-D benchmarks
(two moons, two spirals), verify backprop with gradient checking, and
compare against a linear baseline to show why the hidden layers matter.

Run:  python src/train.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nn import MLP, gradient_check

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEED = 0


# -- synthetic benchmarks -----------------------------------------------------
def make_moons(n: int, noise: float, rng) -> tuple[np.ndarray, np.ndarray]:
    n2 = n // 2
    t = rng.uniform(0, np.pi, n2)
    upper = np.column_stack([np.cos(t), np.sin(t)])
    lower = np.column_stack([1 - np.cos(t), 0.5 - np.sin(t)])
    X = np.vstack([upper, lower]) + rng.normal(0, noise, (2 * n2, 2))
    y = np.r_[np.zeros(n2), np.ones(n2)]
    return X, y


def make_spirals(n: int, noise: float, rng) -> tuple[np.ndarray, np.ndarray]:
    n2 = n // 2
    t = np.sqrt(rng.uniform(0.05, 1.0, n2)) * 3.0 * np.pi
    r = t / (3.0 * np.pi) * 2.0
    arm1 = np.column_stack([r * np.cos(t), r * np.sin(t)])
    arm2 = np.column_stack([r * np.cos(t + np.pi), r * np.sin(t + np.pi)])
    X = np.vstack([arm1, arm2]) + rng.normal(0, noise, (2 * n2, 2))
    y = np.r_[np.zeros(n2), np.ones(n2)]
    return X, y


# -- linear baseline ----------------------------------------------------------
class LogisticBaseline:
    def fit(self, X, y, lr=0.3, epochs=1500):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(epochs):
            p = 1 / (1 + np.exp(-np.clip(X @ self.w + self.b, -30, 30)))
            err = p - y
            self.w -= lr * (X.T @ err / n)
            self.b -= lr * err.mean()
        return self

    def predict(self, X):
        return ((X @ self.w + self.b) >= 0).astype(int)


def decision_boundary_plot(ax, model, X, y, title):
    pad = 0.4
    xs = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 250)
    ys = np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 250)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    proba = model.predict_proba(grid).reshape(gx.shape)
    ax.contourf(gx, gy, proba, levels=20, cmap="RdBu_r", alpha=0.6)
    ax.contour(gx, gy, proba, levels=[0.5], colors="k", linewidths=1)
    ax.scatter(*X[y == 0].T, s=8, c="#1f4e79", label="class 0")
    ax.scatter(*X[y == 1].T, s=8, c="#a63603", label="class 1")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def run_benchmark(name, X, y, hidden, lr, epochs, rng):
    n = len(y)
    perm = rng.permutation(n)
    n_test = int(0.25 * n)
    te, tr = perm[:n_test], perm[n_test:]

    baseline = LogisticBaseline().fit(X[tr], y[tr])
    base_acc = float((baseline.predict(X[te]) == y[te]).mean())

    model = MLP([2, *hidden, 1], seed=SEED, l2=1e-5)
    grad_err = gradient_check(model, X[tr][:40], y[tr][:40])
    model.fit(X[tr], y[tr], lr=lr, epochs=epochs, batch_size=64, seed=SEED)
    test_acc = float((model.predict(X[te]) == y[te]).mean())
    train_acc = float((model.predict(X[tr]) == y[tr]).mean())

    print(f"{name:8s} | linear baseline {base_acc:.3f} | "
          f"MLP train {train_acc:.3f} test {test_acc:.3f} | "
          f"gradient check {grad_err:.2e}")
    return {"model": model, "baseline_acc": base_acc, "test_acc": test_acc,
            "train_acc": train_acc, "grad_check_max_rel_err": grad_err,
            "split": (tr, te)}


def main():
    RESULTS.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    X_m, y_m = make_moons(1000, noise=0.12, rng=rng)
    X_s, y_s = make_spirals(1200, noise=0.06, rng=rng)

    res_m = run_benchmark("moons", X_m, y_m, hidden=[32, 32],
                          lr=0.10, epochs=800, rng=rng)
    res_s = run_benchmark("spirals", X_s, y_s, hidden=[64, 64],
                          lr=0.05, epochs=2500, rng=rng)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    decision_boundary_plot(
        axes[0], res_m["model"], X_m, y_m,
        f"two moons — MLP[32,32] test acc {res_m['test_acc']:.3f}")
    decision_boundary_plot(
        axes[1], res_s["model"], X_s, y_s,
        f"two spirals — MLP[64,64] test acc {res_s['test_acc']:.3f}")
    plt.tight_layout()
    plt.savefig(RESULTS / "decision_boundaries.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6.5, 4))
    plt.semilogy(res_m["model"].loss_history, label="moons MLP[32,32]")
    plt.semilogy(res_s["model"].loss_history, label="spirals MLP[64,64]")
    plt.xlabel("epoch")
    plt.ylabel("training loss (log scale)")
    plt.title("Training curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "loss_curves.png", dpi=150)
    plt.close()

    out = {}
    for name, r in [("moons", res_m), ("spirals", res_s)]:
        out[name] = {k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in r.items() if k not in ("model", "split")}
        out[name]["grad_check_max_rel_err"] = f"{r['grad_check_max_rel_err']:.2e}"
    with open(RESULTS / "metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved metrics and figures to {RESULTS}")


if __name__ == "__main__":
    main()
