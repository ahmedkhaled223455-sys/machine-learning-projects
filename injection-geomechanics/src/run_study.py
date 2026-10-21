"""End-to-end study: verify the solvers, quantify the poroelastic coupling,
build surrogates and a reduced-order model for the maximum safe injection
rate, and run a Monte Carlo fault-reactivation risk assessment.

    python src/run_study.py           # full study (~1-2 min on a laptop)
    python src/run_study.py --quick   # small smoke run
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import consolidation as tz
from fault_stability import (DP_CAP, allowable_dp, critical_dp, fault_normal, fracture_dp,
                             in_situ, stress_path_coefficient)
from radial_flow import R_WELL, YEAR, Grid, diffusivity, interp_at, simulate, theis_dp
from rom import PODRBF
from surrogate import RidgeSurrogate, physics_estimate

RESULTS = Path(__file__).resolve().parents[1] / "results"
HORIZON = 10 * YEAR
DAY = 86400.0
FRAC_SAFETY = 0.9


# ------------------------------------------------------------------ sampling
def sample(N, rng, logk=(-15.0, -13.5)):
    p = {
        "log_k": rng.uniform(*logk, N),                 # outer permeability, log10 m2 (~1-30 mD)
        "log_kratio": rng.uniform(-1, 1, N),            # near-well k / outer k (damage or stimulation)
        "r_inner": rng.uniform(5, 100, N),              # m
        "h": rng.uniform(20, 150, N),                   # m
        "phi": rng.uniform(0.08, 0.25, N),
        "log_Re": rng.uniform(np.log10(3000), np.log10(30000), N),  # closed boundary radius
        "r_fault": rng.uniform(300, 2500, N),           # m
        "depth": rng.uniform(1500, 3500, N),
        "shmin": rng.uniform(0.55, 0.75, N),
        "shmax": rng.uniform(0.75, 0.95, N),
        "strike": rng.uniform(0, 90, N),
        "dip": rng.uniform(40, 80, N),
        "mu_f": rng.uniform(0.5, 0.8, N),
        "nu": rng.uniform(0.2, 0.3, N),
    }
    return p


def geomechanics(p):
    S, p0 = in_situ(p["depth"], p["shmin"], p["shmax"])
    n = fault_normal(p["strike"], p["dip"])
    A = stress_path_coefficient(p["nu"])
    slip = critical_dp(S, p0, n, p["mu_f"], A)
    frac = FRAC_SAFETY * fracture_dp(S, p0, A)
    return slip, frac


def truth(p):
    """Full simulation: unit-rate pressure at the fault and the well -> safe rate."""
    k = 10 ** p["log_k"]
    g, dp = simulate(HORIZON, p["h"], p["phi"], k, k * 10 ** p["log_kratio"], p["r_inner"],
                     10 ** p["log_Re"])
    at_fault = interp_at(g, dp, p["r_fault"])
    at_well = dp[:, 0]
    slip, frac = geomechanics(p)
    q = np.minimum(slip / at_fault, frac / at_well)
    return {"q": q, "slip": slip, "frac": frac, "grid": g, "dp": dp,
            "fault_governs": slip / at_fault <= frac / at_well}


# ------------------------------------------------------------------ features
def raw_features(p, slip, frac):
    return np.column_stack([p["log_k"], p["log_kratio"], p["r_inner"], p["h"], p["phi"],
                            p["log_Re"], np.log10(p["r_fault"]), np.log10(slip), np.log10(frac)])


def physics_q(p, slip, frac):
    k = 10 ** p["log_k"]
    return physics_estimate(HORIZON, p["r_fault"], slip, frac, k, k * 10 ** p["log_kratio"],
                            p["r_inner"], p["h"], p["phi"], 10 ** p["log_Re"])


def dimensionless_features(p, slip, frac):
    k = 10 ** p["log_k"]
    Re = 10 ** p["log_Re"]
    tD = diffusivity(k, p["phi"]) * HORIZON / Re ** 2       # how strongly the boundary is felt
    return np.column_stack([np.log10(tD), np.log10(p["r_fault"] / Re), p["log_kratio"],
                            np.log10(p["r_inner"]), np.log10(slip / frac)])


def score(pred_log, true_log):
    err = pred_log - true_log
    return {"rmse_log10": float(np.sqrt(np.mean(err ** 2))),
            "median_abs_pct_error": float(np.median(np.abs(10 ** err - 1)) * 100),
            "unsafe_overprediction_pct": float(np.mean(err > np.log10(1.1)) * 100)}


# ------------------------------------------------------------------ study
def verification(quick):
    out = {}
    errs = []
    for nz in (10, 20, 40, 80):
        z, u = tz.solve_numeric(0.2, nz, nz * 4)
        errs.append(float(np.abs(u - tz.analytic_pressure(z, 0.2)).max()))
    out["terzaghi_max_error_by_grid"] = errs
    out["terzaghi_observed_order"] = [float(v) for v in np.log2(np.array(errs[:-1]) / np.array(errs[1:]))]
    k = np.array([1e-14, 1e-13, 1e-15])
    h = np.array([50.0, 100.0, 150.0])
    phi = np.array([0.2, 0.15, 0.1])
    radii = (200.0, 1000.0, 5000.0)
    rows = []
    for nc, ns in ((60, 75), (120, 150), (240, 300)) if quick else ((60, 75), (120, 150), (240, 300), (480, 600)):
        g, dp = simulate(HORIZON, h, phi, k, n_cells=nc, n_steps=ns)
        e = [abs(float(interp_at(g, dp, r)[i] / theis_dp(r, HORIZON, 1, k[i], h[i], phi[i]) - 1))
             for i in range(3) for r in radii]
        rows.append({"cells": nc, "steps": ns, "max_rel_error_vs_theis": max(e)})
    out["theis_convergence"] = rows
    return out


def coupling_study(rng, N):
    p = sample(N, rng)
    S, p0 = in_situ(p["depth"], p["shmin"], p["shmax"])
    n = fault_normal(p["strike"], p["dip"])
    A = stress_path_coefficient(p["nu"])
    unc = critical_dp(S, p0, n, p["mu_f"], 0.0 * A)
    cpl = critical_dp(S, p0, n, p["mu_f"], A)
    allow, by_fault = allowable_dp(S, p0, n, p["mu_f"], A, FRAC_SAFETY)
    ok = (unc > 0) & (unc < DP_CAP) & (cpl < DP_CAP)
    return {"n_faults": N,
            "already_critical_pct": float(np.mean(cpl == 0) * 100),
            "median_critical_dp_MPa_uncoupled": float(np.median(unc[ok]) / 1e6),
            "median_critical_dp_MPa_coupled": float(np.median(cpl[ok]) / 1e6),
            "median_ratio_coupled_over_uncoupled": float(np.median(cpl[ok] / unc[ok])),
            "fault_governs_pct": float(np.mean(by_fault) * 100)}, (unc, cpl, ok)


def rom_study(train, test, t_tr, t_te, n_modes):
    r_common = np.logspace(np.log10(R_WELL * 1.2), np.log10(2500), 60)

    def fields(p, res):
        k = 10 ** p["log_k"]
        norm = k * p["h"]
        f = np.stack([interp_at(res["grid"], res["dp"], np.full(len(k), r)) for r in r_common], 1)
        return np.log10(f * norm[:, None])

    def pars(p):
        # the normalised field depends on the dimensionless groups, not on k, h, phi separately
        Re = 10 ** p["log_Re"]
        tD = diffusivity(10 ** p["log_k"], p["phi"]) * HORIZON / Re ** 2
        return np.column_stack([np.log10(tD), p["log_Re"], p["log_kratio"], np.log10(p["r_inner"])])

    F_tr, F_te = fields(train, t_tr), fields(test, t_te)
    # length scale and ridge chosen on a separate validation set (not this test set)
    rom = PODRBF(n_modes=n_modes, length_scale=1.5, ridge=1e-4).fit(pars(train), F_tr)
    t0 = time.perf_counter()
    pred = rom.predict(pars(test))
    dt_rom = (time.perf_counter() - t0) / len(F_te)
    proj = rom.project(F_te)
    rel = np.abs(10 ** (pred - F_te) - 1)
    rel_proj = np.abs(10 ** (proj - F_te) - 1)
    return {"n_modes": n_modes, "energy_captured": float(rom.energy[n_modes - 1]),
            "projection_median_rel_error_pct": float(np.median(rel_proj) * 100),
            "prediction_median_rel_error_pct": float(np.median(rel) * 100),
            "prediction_p95_rel_error_pct": float(np.percentile(rel, 95) * 100),
            "seconds_per_profile_rom": dt_rom}


def monte_carlo(rng, N, hybrid):
    site = {"log_k": rng.normal(-14.0, 0.3, N), "log_kratio": np.zeros(N), "r_inner": np.full(N, 20.0),
            "h": np.full(N, 80.0), "phi": np.full(N, 0.15), "log_Re": np.full(N, np.log10(8000.0)),
            "r_fault": np.full(N, 1500.0), "depth": np.full(N, 2500.0),
            "shmin": rng.normal(0.66, 0.02, N), "shmax": np.full(N, 0.85),
            "strike": np.full(N, 20.0), "dip": np.full(N, 60.0),
            "mu_f": rng.uniform(0.55, 0.7, N), "nu": np.full(N, 0.25)}
    t0 = time.perf_counter()
    tr = truth(site)
    t_sim = time.perf_counter() - t0
    ok = tr["slip"] > 0
    q_true = tr["q"] * DAY
    t0 = time.perf_counter()
    q_hyb = np.zeros(N)
    q_hyb[ok] = 10 ** hybrid({k: v[ok] for k, v in site.items()}, tr["slip"][ok], tr["frac"][ok])  # m3/day
    t_sur = time.perf_counter() - t0
    rates = np.logspace(2, 4.5, 60)
    p_true = [float(np.mean(~ok | (q_true < q))) for q in rates]
    p_hyb = [float(np.mean(~ok | (q_hyb < q))) for q in rates]

    def p05(qs):
        return float(np.percentile(np.where(ok, qs, 0.0), 5))

    return {"n_realisations": N, "already_critical_pct": float(np.mean(~ok) * 100),
            "rate_at_5pct_risk_m3_per_day_simulator": p05(q_true),
            "rate_at_5pct_risk_m3_per_day_hybrid": p05(q_hyb),
            "seconds_simulator": t_sim, "seconds_surrogate": t_sur,
            "speedup": t_sim / max(t_sur, 1e-9)}, (rates, p_true, p_hyb)


def plots(curves, coupling, parity, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    unc, cpl, ok = coupling
    ax[0].scatter(unc[ok] / 1e6, cpl[ok] / 1e6, s=4, alpha=0.4)
    m = max(unc[ok].max(), cpl[ok].max()) / 1e6
    ax[0].plot([0, m], [0, m], "k--", lw=1)
    ax[0].set_xlabel("critical dp, pressure only (MPa)")
    ax[0].set_ylabel("critical dp, with poroelastic stress path (MPa)")
    ax[0].set_title("Coupling changes fault stability")
    for name, (yt, yp) in parity.items():
        ax[1].scatter(yt, yp, s=4, alpha=0.4, label=name)
    lo, hi = ax[1].get_xlim()
    ax[1].plot([lo, hi], [lo, hi], "k--", lw=1)
    ax[1].set_xlabel("log10 safe rate, simulator (m3/day)")
    ax[1].set_ylabel("log10 safe rate, surrogate")
    ax[1].set_title("Out-of-range permeability test")
    ax[1].legend(fontsize=7)
    rates, pt, ph = curves
    ax[2].semilogx(rates, pt, "k-", label="full simulator")
    ax[2].semilogx(rates, ph, "--", label="hybrid surrogate")
    ax[2].axhline(0.05, color="r", lw=0.8)
    ax[2].set_xlabel("injection rate (m3/day)")
    ax[2].set_ylabel("P(fault reactivation or fracturing in 10 yr)")
    ax[2].set_title("Monte Carlo risk curve")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(quick=False):
    RESULTS.mkdir(exist_ok=True)
    rng = np.random.default_rng(7)
    n_tr, n_te, n_mc, n_cpl = (300, 100, 200, 500) if quick else (2000, 600, 2000, 4000)
    out = {"verification": verification(quick)}
    out["coupling"], coupling = coupling_study(rng, n_cpl)

    t0 = time.perf_counter()
    train, test, ood = sample(n_tr, rng), sample(n_te, rng), sample(n_te, rng, logk=(-13.5, -12.8))
    T_tr, T_te, T_ood = truth(train), truth(test), truth(ood)
    out["simulation_seconds"] = time.perf_counter() - t0

    def keep(p, T):
        """Drop faults that are already critically stressed (no safe rate exists)."""
        m = T["slip"] > 0
        T2 = {k: v[m] for k, v in T.items() if k != "grid"}
        T2["grid"] = Grid(T["grid"].r_faces[m], T["grid"].r_cent[m])  # keep rows aligned with dp
        return {k: v[m] for k, v in p.items()}, T2

    (train, T_tr), (test, T_te), (ood, T_ood) = keep(train, T_tr), keep(test, T_te), keep(ood, T_ood)
    y = {s: np.log10(T["q"] * DAY) for s, T in (("train", T_tr), ("test", T_te), ("ood", T_ood))}

    blackbox = RidgeSurrogate(1e-4).fit(raw_features(train, T_tr["slip"], T_tr["frac"]), y["train"])
    phys_tr = np.log10(physics_q(train, T_tr["slip"], T_tr["frac"]) * DAY)
    corr = RidgeSurrogate(1e-4).fit(dimensionless_features(train, T_tr["slip"], T_tr["frac"]), y["train"] - phys_tr)

    def hybrid(p, slip, frac):
        return np.log10(physics_q(p, slip, frac) * DAY) + corr.predict(dimensionless_features(p, slip, frac))

    sur = {}
    parity = {}
    for split, p, T in (("test_in_range", test, T_te), ("test_out_of_range_k", ood, T_ood)):
        yt = y["test" if split.startswith("test_in") else "ood"]
        preds = {"black_box_ridge": blackbox.predict(raw_features(p, T["slip"], T["frac"])),
                 "physics_only": np.log10(physics_q(p, T["slip"], T["frac"]) * DAY),
                 "physics_hybrid": hybrid(p, T["slip"], T["frac"])}
        sur[split] = {k: score(v, yt) for k, v in preds.items()}
        if split == "test_out_of_range_k":
            parity = {k: (yt, v) for k, v in preds.items()}
    out["surrogates"] = sur
    out["samples"] = {"train": int(len(y["train"])), "test": int(len(y["test"])), "ood": int(len(y["ood"]))}
    out["rom"] = rom_study(train, test, T_tr, T_te, n_modes=8)
    out["monte_carlo"], curves = monte_carlo(rng, n_mc, hybrid)
    plots(curves, coupling, parity, RESULTS / "summary.png")
    (RESULTS / "metrics.json").write_text(json.dumps(out, indent=2))
    md = to_markdown(out)
    (RESULTS / "results.md").write_text(md)
    print(md)
    return out


def to_markdown(o):
    v, c, r, mc = o["verification"], o["coupling"], o["rom"], o["monte_carlo"]
    L = ["## Verification", "",
         f"- Terzaghi consolidation: max error {v['terzaghi_max_error_by_grid'][-1]:.1e} (of u0) on the finest grid; "
         f"observed order {', '.join(f'{x:.2f}' for x in v['terzaghi_observed_order'])}",
         "- Radial injection solver vs Theis (max relative error over 3 reservoirs x 3 radii, 10 years):", "",
         "| cells | steps | max rel. error |", "|---|---|---|"]
    L += [f"| {row['cells']} | {row['steps']} | {row['max_rel_error_vs_theis'] * 100:.3f}% |" for row in v["theis_convergence"]]
    L += ["", "## Poroelastic coupling", "",
          f"- {c['n_faults']} random faults; {c['already_critical_pct']:.1f}% already critically stressed",
          f"- median critical pressure increase: {c['median_critical_dp_MPa_uncoupled']:.1f} MPa (pressure only) vs "
          f"{c['median_critical_dp_MPa_coupled']:.1f} MPa with the stress path "
          f"(median ratio {c['median_ratio_coupled_over_uncoupled']:.2f})",
          f"- fault slip (not hydraulic fracturing) is the binding limit for {c['fault_governs_pct']:.0f}% of faults", "",
          "## Safe-rate surrogates (log10 m3/day)", "",
          f"Samples: train {o['samples']['train']}, test {o['samples']['test']}, out-of-range {o['samples']['ood']}; "
          f"simulation time {o['simulation_seconds']:.0f} s.", ""]
    for split, rows in o["surrogates"].items():
        L += [f"### {split}", "", "| model | RMSE (log10) | median abs. error | unsafe over-prediction (>10%) |", "|---|---|---|---|"]
        L += [f"| {k} | {m['rmse_log10']:.3f} | {m['median_abs_pct_error']:.1f}% | {m['unsafe_overprediction_pct']:.1f}% |"
              for k, m in rows.items()]
        L.append("")
    L += ["## Reduced-order model (POD + RBF)", "",
          f"- {r['n_modes']} POD modes capture {r['energy_captured'] * 100:.4f}% of snapshot energy",
          f"- pressure-profile error: {r['projection_median_rel_error_pct']:.2f}% (truncation only), "
          f"{r['prediction_median_rel_error_pct']:.2f}% median / {r['prediction_p95_rel_error_pct']:.2f}% p95 (predicted)",
          f"- {r['seconds_per_profile_rom'] * 1e6:.0f} microseconds per profile", "",
          "## Monte Carlo fault-reactivation risk", "",
          f"- {mc['n_realisations']} realisations of uncertain permeability, friction, and minimum stress; "
          f"{mc['already_critical_pct']:.1f}% already critical",
          f"- rate at 5% risk: {mc['rate_at_5pct_risk_m3_per_day_simulator']:.0f} m3/day (simulator) vs "
          f"{mc['rate_at_5pct_risk_m3_per_day_hybrid']:.0f} m3/day (hybrid surrogate)",
          f"- simulator {mc['seconds_simulator']:.1f} s vs surrogate {mc['seconds_surrogate'] * 1e3:.1f} ms "
          f"({mc['speedup']:.0f}x faster)"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
