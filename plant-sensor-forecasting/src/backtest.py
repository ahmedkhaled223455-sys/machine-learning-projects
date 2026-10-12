"""End-to-end backtest: classical baselines vs PyTorch pure / hybrid
quantile networks (single and deep ensemble), across four scenarios:

* in_dist        -- same operating regime as training
* regime_shift   -- the plant runs at levels above anything seen in training
* faults_naive   -- level sensor with dropouts, spikes and stuck runs;
                    model inputs only forward-filled
* faults_cleaned -- same faults, inputs repaired by the robust input layer

Then: a calibration monitor run on clean data and on a slowly drifting
sensor, and a champion/challenger promotion decision.

Run:  python src/backtest.py            (neural models need torch;
                                          without it, baselines only)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import fc_baselines as bl
from fc_metrics import (apply_shift, combine_ensemble, fit_quantile_shift,
                        from_residuals, mae_by_horizon, score)
from fc_windows import make_windows
from plant_sim import SHIFT_TARGETS, simulate
from retrain_policy import CalibrationMonitor, ModelRegistry, promotion_decision
from sensor_faults import inject_faults

try:
    import torch
    import quantile_models as qm
    HAVE_TORCH = True
except ImportError:  # baselines still run; CI installs torch
    HAVE_TORCH = False

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SCENARIOS = ["in_dist", "regime_shift", "faults_naive", "faults_cleaned"]
GATED = ["in_dist", "faults_cleaned"]   # scenarios a deployable model must pass


def build_data(n_train=30000, n_eval=8000):
    train = simulate(n_train, seed=1)
    val = simulate(n_eval, seed=2)
    test = simulate(n_eval, seed=3)
    shift = simulate(n_eval, SHIFT_TARGETS, seed=4)
    faulty, fault_log = inject_faults(test.level_meas, seed=5)
    drifting, _ = inject_faults(test.level_meas, seed=6, dropout_frac=0.0,
                                n_spikes=0, n_stuck=0, drift_total=0.4)
    w = {
        "train": make_windows(train),
        "val": make_windows(val),
        "in_dist": make_windows(test),
        "regime_shift": make_windows(shift),
        "faults_naive": make_windows(test, level_input=faulty),
        "faults_cleaned": make_windows(test, level_input=faulty, clean=True),
        "drift": make_windows(test, level_input=drifting, clean=True),
    }
    fault_counts = {k: len(v) for k, v in fault_log.items()}
    return w, fault_counts


def evaluate_point(fn, w) -> tuple[dict, dict]:
    resid = w["val"].target - fn(w["val"])
    out, preds = {}, {}
    for s in SCENARIOS + ["drift"]:
        q = from_residuals(fn(w[s]), resid)
        preds[s] = q
        if s in SCENARIOS:
            out[s] = score(w[s].target, q)
    return out, preds


def evaluate_quantile(qfn, w) -> tuple[dict, dict]:
    """Quantile model, recalibrated on validation data (split-conformal
    style per-horizon, per-quantile shift)."""
    shifts = fit_quantile_shift(w["val"].target, qfn(w["val"]))
    out, preds = {}, {}
    for s in SCENARIOS + ["drift"]:
        q = apply_shift(qfn(w[s]), shifts)
        preds[s] = q
        if s in SCENARIOS:
            out[s] = score(w[s].target, q)
    return out, preds


def main(members: int = 5):
    t0 = time.time()
    RESULTS.mkdir(exist_ok=True)
    w, fault_counts = build_data()
    print(f"windows: train={len(w['train'])} val={len(w['val'])} "
          f"test={len(w['in_dist'])}  faults injected: {fault_counts}")

    metrics, preds = {}, {}
    arx = bl.ARX().fit(w["train"])
    for name, fn in [("persistence", bl.persistence), ("holt", bl.holt),
                     ("arx", arx.predict), ("physics", bl.physics)]:
        metrics[name], preds[name] = evaluate_point(fn, w)

    if HAVE_TORCH:
        torch.set_num_threads(2)
        ens = {}
        for kind in ("pure", "hybrid"):
            t1 = time.time()
            ens[kind] = qm.train_ensemble(w["train"], w["val"],
                                          hybrid=(kind == "hybrid"),
                                          members=members)
            epochs = [len(m.history) for m in ens[kind]]
            print(f"trained {kind} ensemble ({members} members, epochs {epochs}) "
                  f"in {time.time() - t1:.0f}s")
            single = ens[kind][0]
            metrics[f"{kind}_net"], preds[f"{kind}_net"] = evaluate_quantile(
                single.predict, w)
            models = ens[kind]
            metrics[f"{kind}_ensemble"], preds[f"{kind}_ensemble"] = evaluate_quantile(
                lambda x, models=models: combine_ensemble(qm.predict_ensemble(models, x)), w)
        ens["hybrid"][0].save(str(RESULTS / "hybrid_member0.pt"))

    # ---- report ---------------------------------------------------------
    lines = ["| model | scenario | MAE | pinball | 90% coverage | 90% width | infeasible |",
             "|---|---|---|---|---|---|---|"]
    for s in SCENARIOS:
        for name, m in metrics.items():
            r = m[s]
            lines.append(f"| {name} | {s} | {r['mae']:.4f} | {r['pinball']:.4f} | "
                         f"{r['coverage90']:.3f} | {r['width90']:.3f} | {r['infeasible']:.3f} |")
    table = "\n".join(lines)
    print(table)

    best = "hybrid_ensemble" if HAVE_TORCH else "arx"
    horizon = {name: mae_by_horizon(w["in_dist"].target, p["in_dist"])
               for name, p in preds.items()}

    # ---- monitoring: clean data should not trigger; drift should --------
    mon = CalibrationMonitor()
    monitor = {
        "clean": mon.run(w["in_dist"].target, preds[best]["in_dist"]),
        "drift": mon.run(w["drift"].target, preds[best]["drift"]),
    }
    for k, v in monitor.items():
        print(f"monitor[{k}]: trigger_at={v['trigger_at']} "
              f"min rolling coverage={v['min_coverage']}")

    # ---- champion / challenger -----------------------------------------
    reg_path = RESULTS / "registry.json"
    if reg_path.exists():
        reg_path.unlink()
    registry = ModelRegistry(reg_path)
    gated = lambda name: {s: metrics[name][s] for s in GATED}  # noqa: E731
    registry.register("arx-v1", gated("arx"))
    registry.promote("arx-v1")
    if best == "arx":
        decision = {"promote": False, "reasons": ["no challenger (torch not installed)"]}
    else:
        decision = promotion_decision(gated(best), registry.live_metrics())
        registry.register(f"{best}-v1", gated(best))
        if decision["promote"]:
            registry.promote(f"{best}-v1")
    print(f"promotion of {best} over arx: {decision}")

    out = {"metrics": metrics, "mae_by_horizon": horizon, "monitor": monitor,
           "promotion": decision, "live_model": registry.state["live"],
           "fault_counts": fault_counts, "torch": HAVE_TORCH,
           "runtime_s": round(time.time() - t0, 1)}
    (RESULTS / "metrics.json").write_text(json.dumps(out, indent=2))
    (RESULTS / "backtest.md").write_text("# Backtest results\n\n" + table + "\n")
    _figure(w, preds, best)
    print(f"done in {time.time() - t0:.0f}s -> {RESULTS}")
    return out


def _figure(w, preds, best):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n in ("arx", "pure_ensemble", best) if n in preds]
    names = list(dict.fromkeys(names))
    h = 15  # 16-minute-ahead forecast
    sl = slice(1500, 2100)
    fig, axes = plt.subplots(len(names), 1, figsize=(9, 2.6 * len(names)), sharex=True)
    axes = np.atleast_1d(axes)
    y = w["regime_shift"].target[sl, h]
    x = np.arange(len(y))
    for ax, n in zip(axes, names):
        q = preds[n]["regime_shift"][sl, h]
        ax.fill_between(x, q[:, 0], q[:, 4], alpha=0.25, color="#31708E", label="90% interval")
        ax.plot(x, q[:, 2], color="#31708E", lw=1, label="median")
        ax.plot(x, y, color="#B1624E", lw=1, label="true level")
        ax.set_title(f"{n}: 16-min-ahead forecast, regime outside training range")
        ax.set_ylabel("level [m]")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("minute")
    fig.tight_layout()
    fig.savefig(RESULTS / "regime_shift_forecasts.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
