"""Bearing fault diagnosis study.

Generates vibration records from many simulated machines (different
bearings, resonances, gear meshes, speeds, fault severities), then

1. ablates the signal-processing chain with a physics rule (threshold tuned
   on training machines only),
2. compares classifiers on generic statistical features vs. physics
   (envelope fault-frequency) features,
all evaluated on machines never seen in training.

    python src/run_study.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from bearing import CLASSES, FS, random_machine, simulate
from diagnose import (METHODS, SoftmaxClassifier, fault_evidence, fault_indicators,
                      rule_diagnosis, spectrum_for, statistical_features)
from dsp import best_band, envelope_spectrum, tsa_residual, bandpass

RESULTS = Path(__file__).resolve().parents[1] / "results"


def make_dataset(n_machines, records_per_class, rng):
    rows = []
    for mi in range(n_machines):
        m = random_machine(rng)
        for ci, fault in enumerate(CLASSES):
            for _ in range(records_per_class):
                shaft = rng.uniform(15, 40)
                sev = rng.uniform(0.15, 1.2)
                x = simulate(m, fault, shaft, sev, rng)
                rows.append({"machine": mi, "label": ci, "shaft": shaft, "severity": sev, "x": x,
                             "orders": m.bearing.defect_orders(), "ftf": m.bearing.ftf})
    return rows


def indicators(rows, method):
    return np.stack([fault_indicators(r["x"], r["shaft"], r["orders"], method) for r in rows])


def evidence(rows):
    ev = [fault_evidence(r["x"], r["shaft"], r["orders"], r["ftf"], "full") for r in rows]
    return np.stack([e[0] for e in ev]), np.stack([e[1] for e in ev])


def tune_threshold(ind, y):
    scores = np.log(ind).mean(-1).max(-1)
    best_t, best_acc = 0.0, -1.0
    for t in np.quantile(scores, np.linspace(0.01, 0.99, 199)):
        acc = np.mean([rule_diagnosis(i, t) == yy for i, yy in zip(ind, y)])
        if acc > best_acc:
            best_t, best_acc = float(t), acc
    return best_t


def metrics(pred, y):
    cm = np.zeros((4, 4), int)
    for p, t in zip(pred, y):
        cm[t, p] += 1
    recall = cm.diagonal() / cm.sum(1).clip(min=1)
    # missed fault = a faulty record called healthy; false alarm = healthy called faulty
    missed = cm[1:, 0].sum() / max(cm[1:].sum(), 1)
    false_alarm = cm[0, 1:].sum() / max(cm[0].sum(), 1)
    return {"accuracy": float(np.mean(pred == y)), "recall": dict(zip(CLASSES, map(float, recall))),
            "missed_fault_rate": float(missed), "false_alarm_rate": float(false_alarm),
            "confusion": cm.tolist()}


def phys_features(ind):
    return np.log(ind).reshape(len(ind), -1)


def plot_example(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = next(r for r in rows if CLASSES[r["label"]] == "outer" and r["severity"] < 0.5)
    x, shaft, bpfo = r["x"], r["shaft"], r["orders"]["outer"] * r["shaft"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    f, A = spectrum_for(x, shaft, "raw_spectrum")
    ax[0].semilogy(f, A, lw=0.6)
    ax[0].set_xlim(0, FS / 2)
    ax[0].set_title("Raw spectrum: shaft harmonics, gear mesh, noise")
    ax[0].set_xlabel("Hz")
    res = tsa_residual(x, FS, shaft)
    max_fault = max(r["orders"].values()) * shaft
    lo, hi, sk = best_band(res, FS, min_bandwidth=3.0 * max_fault)
    ax[1].axvspan(lo, hi, color="orange", alpha=0.3, label=f"SK band {lo:.0f}-{hi:.0f} Hz")
    from dsp import spectral_kurtosis
    fs_, s = spectral_kurtosis(res, FS, 64)
    ax[1].plot(fs_, s)
    ax[1].set_title("Spectral kurtosis of TSA residual (64-sample window)")
    ax[1].set_xlabel("Hz")
    ax[1].legend(fontsize=7)
    f, E = envelope_spectrum(bandpass(res, FS, lo, hi), FS)
    ax[2].plot(f, E, lw=0.8)
    for h in (1, 2, 3):
        ax[2].axvline(h * bpfo, color="r", ls="--", lw=0.8)
    ax[2].set_xlim(0, 4 * bpfo)
    ax[2].set_title("Envelope spectrum: peaks at BPFO harmonics (red)")
    ax[2].set_xlabel("Hz")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(quick=False):
    RESULTS.mkdir(exist_ok=True)
    rng = np.random.default_rng(11)
    n_tr, n_te, per = (6, 4, 3) if quick else (30, 15, 8)
    t0 = time.perf_counter()
    train, test = make_dataset(n_tr, per, rng), make_dataset(n_te, per, rng)
    y_tr = np.array([r["label"] for r in train])
    y_te = np.array([r["label"] for r in test])
    out = {"train_records": len(train), "test_records": len(test),
           "train_machines": n_tr, "test_machines": n_te}

    ablation = {}
    for method in METHODS:
        i_tr, i_te = indicators(train, method), indicators(test, method)
        thr = tune_threshold(i_tr, y_tr)
        pred = np.array([rule_diagnosis(i, thr) for i in i_te])
        ablation[method] = metrics(pred, y_te)
    # final step: add modulation-sideband evidence to separate coinciding fault frequencies
    (i_tr, m_tr), (i_te, m_te) = evidence(train), evidence(test)
    thr = tune_threshold(i_tr, y_tr)
    # tie margin and sideband-evidence threshold tuned on TRAINING machines only
    best = (-1.0, 0.25, 0.4)
    for tm in (0.1, 0.15, 0.25, 0.4):
        for se in (0.2, 0.3, 0.4, 0.6, 0.8):
            acc = np.mean([rule_diagnosis(i, thr, m, tm, se) == yy for i, m, yy in zip(i_tr, m_tr, y_tr)])
            if acc > best[0]:
                best = (acc, tm, se)
    _, tie_margin, sb_ev = best
    out["sideband_rule_params"] = {"tie_margin": tie_margin, "sideband_evidence": sb_ev}
    pred_sb = np.array([rule_diagnosis(i, thr, m, tie_margin, sb_ev) for i, m in zip(i_te, m_te)])
    ablation["full_plus_sidebands"] = metrics(pred_sb, y_te)
    out["rule_ablation"] = ablation

    stat_tr = np.stack([statistical_features(r["x"]) for r in train])
    stat_te = np.stack([statistical_features(r["x"]) for r in test])
    p_tr = np.hstack([phys_features(i_tr), np.log(m_tr[:, 1:])])
    p_te = np.hstack([phys_features(i_te), np.log(m_te[:, 1:])])
    clf = {}
    for name, (Xtr, Xte) in {"statistical_features": (stat_tr, stat_te),
                             "physics_envelope_features": (p_tr, p_te),
                             "both": (np.hstack([stat_tr, p_tr]), np.hstack([stat_te, p_te]))}.items():
        model = SoftmaxClassifier().fit(Xtr, y_tr, 4)
        clf[name] = metrics(model.predict(Xte), y_te)
    out["classifiers"] = clf

    # accuracy vs fault severity for the full physics chain
    sev = np.array([r["severity"] for r in test])
    pred_full = pred_sb
    bins = [0.15, 0.4, 0.7, 1.2]
    out["recall_by_severity_full_rule"] = {
        f"{lo:.2f}-{hi:.2f}": float(np.mean(pred_full[(y_te > 0) & (sev >= lo) & (sev < hi)]
                                            == y_te[(y_te > 0) & (sev >= lo) & (sev < hi)]))
        for lo, hi in zip(bins[:-1], bins[1:])}
    out["seconds"] = time.perf_counter() - t0
    plot_example(test, RESULTS / "diagnosis_example.png")
    (RESULTS / "metrics.json").write_text(json.dumps(out, indent=2))
    md = to_markdown(out)
    (RESULTS / "results.md").write_text(md)
    print(md)
    return out


def to_markdown(o):
    L = [f"Train: {o['train_records']} records from {o['train_machines']} machines; "
         f"test: {o['test_records']} records from {o['test_machines']} **unseen** machines.", "",
         "## Signal-processing ablation (physics rule, threshold tuned on training machines)", "",
         "| chain | accuracy | missed faults | false alarms | outer | inner | ball |", "|---|---|---|---|---|---|---|"]
    for k, m in o["rule_ablation"].items():
        r = m["recall"]
        L.append(f"| {k} | {m['accuracy']:.3f} | {m['missed_fault_rate']:.3f} | {m['false_alarm_rate']:.3f} | "
                 f"{r['outer']:.2f} | {r['inner']:.2f} | {r['ball']:.2f} |")
    L += ["", "## Classifiers on unseen machines", "",
          "| features | accuracy | missed faults | false alarms |", "|---|---|---|---|"]
    for k, m in o["classifiers"].items():
        L.append(f"| {k} | {m['accuracy']:.3f} | {m['missed_fault_rate']:.3f} | {m['false_alarm_rate']:.3f} |")
    L += ["", "## Fault recall vs severity (full chain + sidebands)", "",
          "| severity | recall |", "|---|---|"]
    L += [f"| {k} | {v:.2f} |" for k, v in o["recall_by_severity_full_rule"].items()]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
