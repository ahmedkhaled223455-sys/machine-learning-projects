"""Verification tests: bearing kinematics against published reference values,
DSP primitives against analytic signals, and end-to-end diagnosis checks."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bearing import (FS, SKF_6205, Bearing, Machine, random_machine,  # noqa: E402
                     simulate)
from diagnose import (SoftmaxClassifier, fault_evidence, fault_indicators,  # noqa: E402
                      rule_diagnosis, statistical_features)
from dsp import (analytic_signal, bandpass, best_band, envelope,  # noqa: E402
                 envelope_spectrum, spectral_kurtosis, time_synchronous_average,
                 tsa_residual)


# ---------------- kinematics ----------------

def test_skf6205_defect_frequencies_match_reference():
    # Widely published values for the SKF 6205 (e.g. CWRU bearing data center)
    assert np.isclose(SKF_6205.bpfo, 3.5848, atol=1e-3)
    assert np.isclose(SKF_6205.bpfi, 5.4152, atol=1e-3)
    assert np.isclose(2 * SKF_6205.bsf, 4.7135, atol=1e-3)
    assert np.isclose(SKF_6205.ftf, 0.3983, atol=1e-3)


def test_kinematic_identities():
    b = Bearing(n_balls=11, d=9.0, D=50.0, angle_deg=12.0)
    assert np.isclose(b.bpfo + b.bpfi, b.n_balls)        # BPFO + BPFI = n
    assert np.isclose(b.bpfo, b.n_balls * b.ftf)         # BPFO = n * FTF


# ---------------- DSP primitives ----------------

def test_hilbert_envelope_recovers_am_modulation():
    t = np.arange(FS) / FS
    mod = 1 + 0.5 * np.cos(2 * np.pi * 7 * t)
    x = mod * np.cos(2 * np.pi * 1000 * t)
    assert np.abs(envelope(x)[200:-200] - mod[200:-200]).max() < 1e-2


def test_analytic_signal_real_part_is_input():
    x = np.random.default_rng(0).normal(size=1001)       # odd length branch
    assert np.allclose(analytic_signal(x).real, x)
    y = np.random.default_rng(1).normal(size=1000)       # even length branch
    assert np.allclose(analytic_signal(y).real, y)


def test_bandpass_keeps_inside_removes_outside():
    t = np.arange(FS) / FS
    x = np.sin(2 * np.pi * 500 * t) + np.sin(2 * np.pi * 3000 * t)
    y = bandpass(x, FS, 2500, 3500)
    assert np.allclose(y, np.sin(2 * np.pi * 3000 * t), atol=1e-9)


def test_envelope_spectrum_finds_impact_rate():
    rng = np.random.default_rng(0)
    m = Machine(SKF_6205, f_res=3500, zeta=0.03, gear_teeth=25, harm_amp=0.0, lf_noise=0.0)
    x = simulate(m, "outer", 25.0, 1.0, rng, noise=0.05)
    f, E = envelope_spectrum(bandpass(x, FS, 3000, 4000), FS)
    lo = f > 20
    assert abs(f[lo][np.argmax(E[lo])] - SKF_6205.bpfo * 25.0) < 2.0


def test_spectral_kurtosis_gaussian_vs_impulsive():
    rng = np.random.default_rng(0)
    _, sk_noise = spectral_kurtosis(rng.normal(size=FS), FS, 64)
    assert abs(np.median(sk_noise[2:-2])) < 0.15           # ~0 for stationary Gaussian noise
    m = Machine(SKF_6205, f_res=3500, zeta=0.03, gear_teeth=25, harm_amp=0.0, lf_noise=0.0)
    x = simulate(m, "inner", 25.0, 2.0, rng, noise=0.1)
    f, sk = spectral_kurtosis(x, FS, 64)
    near = np.abs(f - 3500) < 300
    assert sk[near].max() > 1.0                             # impulsive band stands out


def test_best_band_finds_resonance_and_respects_min_bandwidth():
    rng = np.random.default_rng(1)
    m = Machine(SKF_6205, f_res=4000, zeta=0.03, gear_teeth=25, harm_amp=1.0, lf_noise=1.0)
    x = simulate(m, "inner", 25.0, 1.0, rng)
    lo, hi, _ = best_band(x, FS, min_bandwidth=600)
    assert lo < 4000 < hi
    assert hi - lo >= 600


def test_band_is_wide_enough_for_fast_outer_faults():
    """Regression test: at high shaft speed the most kurtotic window can be
    narrower than the impact rate, which erases the fault from the envelope."""
    rng = np.random.default_rng(12)
    widths_ok, detected = 0, 0
    for _ in range(8):
        m = random_machine(rng)
        shaft = rng.uniform(35, 40)
        x = simulate(m, "outer", shaft, 1.0, rng)
        max_fault = max(m.bearing.defect_orders().values()) * shaft
        lo, hi, _ = best_band(tsa_residual(x, FS, shaft), FS, min_bandwidth=3 * max_fault)
        widths_ok += (hi - lo) >= 3 * max_fault
        ind, mod = fault_evidence(x, shaft, m.bearing.defect_orders(), m.bearing.ftf, "full")
        detected += rule_diagnosis(ind, 1.0, mod) == 1
    assert widths_ok == 8
    assert detected >= 7


def test_tsa_preserves_synchronous_and_suppresses_noise():
    rng = np.random.default_rng(0)
    t = np.arange(FS) / FS
    shaft = 25.0
    sync = np.sin(2 * np.pi * shaft * t) + 0.5 * np.sin(2 * np.pi * 3 * shaft * t)
    noise = rng.normal(size=FS)
    tsa, n_revs = time_synchronous_average(sync + noise, FS, shaft)
    clean, _ = time_synchronous_average(sync, FS, shaft)
    resid_std = np.std(tsa - clean)
    assert n_revs == 25
    assert resid_std < 1.5 / np.sqrt(n_revs)               # noise shrinks ~ 1/sqrt(N)
    assert np.std(tsa_residual(sync, FS, shaft)) < 0.02    # synchronous part removed


# ---------------- diagnosis ----------------

@pytest.mark.parametrize("fault", ["outer", "inner", "ball"])
def test_full_chain_identifies_clear_faults(fault):
    rng = np.random.default_rng({"outer": 3, "inner": 4, "ball": 5}[fault])
    hits = 0
    for _ in range(5):
        m = random_machine(rng)
        shaft = rng.uniform(15, 40)
        x = simulate(m, fault, shaft, 1.0, rng)
        ind, mod = fault_evidence(x, shaft, m.bearing.defect_orders(), m.bearing.ftf, "full")
        hits += rule_diagnosis(ind, 1.0, mod) == {"outer": 1, "inner": 2, "ball": 3}[fault]
    assert hits >= 4


def _coincident_bearing():
    """A geometry where 2*BSF falls within 1% of BPFO (frequency alone cannot tell them apart)."""
    for n in range(8, 14):
        for ratio in np.linspace(0.15, 0.25, 2001):
            b = Bearing(n, ratio * 50.0, 50.0)
            if abs(2 * b.bsf - b.bpfo) / b.bpfo < 0.005:
                return b
    raise AssertionError("no coincident geometry found")


@pytest.mark.parametrize("fault", ["outer", "ball"])
def test_sidebands_separate_coinciding_fault_frequencies(fault):
    b = _coincident_bearing()
    rng = np.random.default_rng(7)
    m = Machine(b, f_res=3500, zeta=0.03, gear_teeth=25, harm_amp=1.0, lf_noise=1.0)
    correct = 0
    for _ in range(5):
        shaft = rng.uniform(18, 35)
        x = simulate(m, fault, shaft, 1.0, rng)
        ind, mod = fault_evidence(x, shaft, b.defect_orders(), b.ftf, "full")
        correct += rule_diagnosis(ind, 1.0, mod) == {"outer": 1, "ball": 3}[fault]
    assert correct >= 4


def test_healthy_machines_do_not_alarm():
    rng = np.random.default_rng(6)
    alarms = 0
    for _ in range(8):
        m = random_machine(rng)
        shaft = rng.uniform(15, 40)
        ind = fault_indicators(simulate(m, "healthy", shaft, 1.0, rng), shaft,
                               m.bearing.defect_orders(), "full")
        alarms += rule_diagnosis(ind, threshold=1.0) != 0
    assert alarms <= 1


def test_statistical_features_shape_and_kurtosis():
    rng = np.random.default_rng(0)
    feats = statistical_features(rng.normal(size=FS))
    assert feats.shape == (14,)
    assert abs(feats[3]) < 0.2                               # excess kurtosis of Gaussian ~ 0


def test_softmax_classifier_separates_blobs():
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0], [4, 0], [0, 4], [4, 4]])
    X = np.vstack([c + rng.normal(size=(50, 2)) * 0.5 for c in centers])
    y = np.repeat(np.arange(4), 50)
    clf = SoftmaxClassifier().fit(X, y, 4)
    assert np.mean(clf.predict(X) == y) > 0.97
    assert np.allclose(clf.predict_proba(X).sum(1), 1.0)
