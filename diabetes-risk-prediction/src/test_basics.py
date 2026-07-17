"""Sanity tests for the from-scratch models and metrics.

Run:  python src/test_basics.py
"""

import numpy as np

from evaluation import auc, classification_metrics, roc_curve, stratified_kfold
from models import DecisionTree, LogisticRegressionGD, RandomForest


def test_metrics_hand_computed():
    y_true = [1, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 0, 1, 1, 0]
    m = classification_metrics(y_true, y_pred)
    assert m["confusion"] == {"tp": 2, "fp": 1, "fn": 1, "tn": 2}
    assert abs(m["accuracy"] - 4 / 6) < 1e-12
    assert abs(m["precision"] - 2 / 3) < 1e-12
    assert abs(m["recall"] - 2 / 3) < 1e-12


def test_auc_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    assert abs(auc(y, np.array([0.1, 0.2, 0.8, 0.9])) - 1.0) < 1e-12
    assert abs(auc(y, np.array([0.9, 0.8, 0.2, 0.1])) - 0.0) < 1e-12
    trapz = getattr(np, "trapezoid", None) or np.trapz
    fpr, tpr = roc_curve(y, np.array([0.5, 0.5, 0.5, 0.5]))
    assert abs(trapz(tpr, fpr) - 0.5) < 1e-12  # ties -> diagonal


def test_logreg_separable():
    rng = np.random.default_rng(0)
    X0 = rng.normal(-2, 0.5, size=(100, 2))
    X1 = rng.normal(+2, 0.5, size=(100, 2))
    X = np.vstack([X0, X1])
    y = np.r_[np.zeros(100), np.ones(100)]
    model = LogisticRegressionGD(lr=0.5, epochs=500).fit(X, y)
    assert (model.predict(X) == y).mean() > 0.99


def test_tree_learns_xor():
    # XOR has zero first-split Gini gain, so a greedy tree needs extra
    # depth to recover -- a classic property worth remembering.
    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, size=(400, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)   # not linearly separable
    tree = DecisionTree(max_depth=8, min_samples_leaf=5).fit(X, y)
    assert (tree.predict(X) == y).mean() > 0.90
    forest = RandomForest(n_trees=30, max_depth=8, seed=3).fit(X, y)
    assert (forest.predict(X) == y).mean() > 0.90


def test_stratified_kfold_preserves_ratio():
    y = np.r_[np.ones(30), np.zeros(70)]
    for tr, va in stratified_kfold(y, k=5, seed=0):
        assert 0.2 <= y[va].mean() <= 0.4          # close to global 30%
        assert len(np.intersect1d(tr, va)) == 0


if __name__ == "__main__":
    for fn in [test_metrics_hand_computed, test_auc_perfect_and_random,
               test_logreg_separable, test_tree_learns_xor,
               test_stratified_kfold_preserves_ratio]:
        fn()
        print(f"PASS {fn.__name__}")
    print("All tests passed.")
