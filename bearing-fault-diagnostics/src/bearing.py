"""Rolling-element bearing kinematics and a physics-based vibration simulator.

Characteristic defect frequencies (per unit shaft speed, outer race fixed):

    FTF  = 1/2 (1 - d/D cos a)                     cage
    BPFO = n/2 (1 - d/D cos a)                     outer-race defect
    BPFI = n/2 (1 + d/D cos a)                     inner-race defect
    BSF  = D/(2d) (1 - (d/D cos a)^2)              ball spin; a ball defect
                                                    strikes both races -> 2*BSF

The simulator builds an accelerometer signal the way the physics produces
it: each time a rolling element meets a defect it excites a lightly damped
structural resonance. Inner-race impacts are amplitude-modulated at shaft
speed (the defect rotates through the load zone); ball impacts at the cage
frequency. Rolling elements slip, so impact spacing jitters by ~1%. On top
come strong shaft harmonics, a gear-mesh tone with sidebands, and
low-frequency broadband noise -- the things that hide bearing faults in a
raw spectrum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FS = 12_000          # Hz, sampling rate
DURATION = 1.0       # s
CLASSES = ("healthy", "outer", "inner", "ball")


@dataclass
class Bearing:
    n_balls: int
    d: float          # ball diameter (mm)
    D: float          # pitch diameter (mm)
    angle_deg: float = 0.0

    @property
    def _r(self):
        return self.d / self.D * np.cos(np.radians(self.angle_deg))

    @property
    def ftf(self):
        return 0.5 * (1 - self._r)

    @property
    def bpfo(self):
        return self.n_balls / 2 * (1 - self._r)

    @property
    def bpfi(self):
        return self.n_balls / 2 * (1 + self._r)

    @property
    def bsf(self):
        return self.D / (2 * self.d) * (1 - self._r ** 2)

    def defect_orders(self):
        """Impact rate per shaft revolution for each fault class."""
        return {"outer": self.bpfo, "inner": self.bpfi, "ball": 2 * self.bsf}


SKF_6205 = Bearing(n_balls=9, d=7.94, D=39.04)   # common reference bearing


@dataclass
class Machine:
    bearing: Bearing
    f_res: float        # structural resonance excited by impacts (Hz)
    zeta: float         # damping ratio
    gear_teeth: int     # gear mesh order (interference)
    harm_amp: float     # shaft-harmonic amplitude
    lf_noise: float     # low-frequency masking noise level


def random_machine(rng: np.random.Generator) -> Machine:
    n = int(rng.integers(8, 14))
    D = rng.uniform(35, 70)
    d = D * rng.uniform(0.15, 0.25)
    return Machine(Bearing(n, d, D, rng.uniform(0, 15)), f_res=rng.uniform(2500, 4800),
                   zeta=rng.uniform(0.02, 0.06), gear_teeth=int(rng.integers(18, 40)),
                   harm_amp=rng.uniform(1.0, 3.0), lf_noise=rng.uniform(0.5, 1.5))


def _impulse_response(f_res, zeta, fs=FS, length_s=0.02):
    t = np.arange(int(length_s * fs)) / fs
    wd = 2 * np.pi * f_res * np.sqrt(1 - zeta ** 2)
    return np.exp(-2 * np.pi * f_res * zeta * t) * np.sin(wd * t)


def _lowpass_noise(rng, n, cutoff, fs=FS):
    spec = np.fft.rfft(rng.normal(size=n))
    f = np.fft.rfftfreq(n, 1 / fs)
    spec[f > cutoff] = 0
    x = np.fft.irfft(spec, n)
    return x / x.std()


def simulate(machine: Machine, fault: str, shaft_hz: float, severity: float,
             rng: np.random.Generator, fs=FS, duration=DURATION, noise=0.3):
    """One vibration record (numpy array) for the given fault class."""
    n = int(fs * duration)
    t = np.arange(n) / fs
    x = machine.harm_amp * (np.sin(2 * np.pi * shaft_hz * t + rng.uniform(0, 6.3))
                            + 0.5 * np.sin(2 * np.pi * 2 * shaft_hz * t + rng.uniform(0, 6.3)))
    fm = machine.gear_teeth * shaft_hz                       # gear mesh with shaft sidebands
    x += 0.8 * (1 + 0.4 * np.cos(2 * np.pi * shaft_hz * t)) * np.sin(2 * np.pi * fm * t)
    x += machine.lf_noise * _lowpass_noise(rng, n, 1500.0, fs)
    x += noise * rng.normal(size=n)
    if fault != "healthy":
        rate = machine.bearing.defect_orders()[fault] * shaft_hz
        period = 1.0 / rate
        times, tk = [], rng.uniform(0, period)
        while tk < duration:
            times.append(tk)
            tk += period * (1 + 0.01 * rng.normal())             # rolling-element slip
        times = np.array(times)
        if fault == "inner":
            amp = 1 + 0.8 * np.cos(2 * np.pi * shaft_hz * times)          # load-zone modulation
        elif fault == "ball":
            amp = 1 + 0.6 * np.cos(2 * np.pi * machine.bearing.ftf * shaft_hz * times)
        else:
            amp = np.ones_like(times)
        train = np.zeros(n)
        np.add.at(train, np.minimum((times * fs).astype(int), n - 1), severity * np.maximum(amp, 0))
        x += np.convolve(train, _impulse_response(machine.f_res, machine.zeta, fs))[:n]
    return x
