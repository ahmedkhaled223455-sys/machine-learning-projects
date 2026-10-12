"""PyTorch multi-horizon quantile forecasters: pure vs physics-hybrid.

Two model families share one network and one training loop:

* pure    -- forecasts the level as (last observed level) + learned change.
             Everything about the dynamics must be learned from data.
* hybrid  -- forecasts the level as (physics rollout) + learned correction.
             The nominal mass balance supplies the structure; the network
             only learns what physics gets wrong (valve-coefficient error,
             the unmeasured disturbance it can infer from recent history).

Outputs are five quantiles per horizon step. Quantile crossing is
impossible by construction: the network predicts a median plus positive
(softplus) gaps that are cumulatively added above and below it.

Loss = pinball loss + a soft penalty on physically infeasible levels
(below an empty tank or above its rim).

A deep ensemble (independently seeded members) supplies epistemic
uncertainty; ``combine_ensemble`` folds member disagreement into the
predictive interval, which is what keeps coverage honest when the plant
moves into a regime none of the members saw in training.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from fc_metrics import QUANTILES
from fc_windows import H, Windows
from plant_sim import L_MAX

Q = len(QUANTILES)


def build_inputs(w: Windows, hybrid: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return (features, base). The network predicts target - base."""
    last = w.past_level[:, -1:]
    parts = [
        w.past_level - last,                 # recent shape of the level
        w.past_fin,                          # recent measured inflow
        w.fut_fin, w.fut_u,                  # known controls
        last, np.sqrt(np.maximum(last, 0)),  # operating point
    ]
    if hybrid:
        parts.append(w.phys - last)          # what physics expects
        base = w.phys
    else:
        base = np.repeat(last, H, axis=1)
    return np.hstack(parts).astype(np.float32), base.astype(np.float32)


class QuantileMLP(nn.Module):
    def __init__(self, n_in: int, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, H * Q),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.body(x).view(-1, H, Q)
        center = raw[..., 2]
        # offset so untrained gaps start small (~0.05 m) instead of ~0.7 m
        gaps = nn.functional.softplus(raw - 3.0) + 1e-4
        q1 = center - gaps[..., 1]
        q0 = q1 - gaps[..., 0]
        q3 = center + gaps[..., 3]
        q4 = q3 + gaps[..., 4]
        return torch.stack([q0, q1, center, q3, q4], dim=-1)


def pinball_loss(y: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    taus = torch.as_tensor(QUANTILES, dtype=q.dtype, device=q.device)
    diff = y.unsqueeze(-1) - q
    return torch.maximum(taus * diff, (taus - 1.0) * diff).mean()


def feasibility_penalty(level_q: torch.Tensor) -> torch.Tensor:
    """Hinge penalty on quantiles outside the tank: [0, L_MAX]."""
    return (torch.relu(-level_q) + torch.relu(level_q - L_MAX)).mean()


@dataclass
class TrainConfig:
    hidden: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch: int = 512
    max_epochs: int = 40
    patience: int = 5
    physics_weight: float = 1.0
    seed: int = 0


@dataclass
class QuantileForecaster:
    hybrid: bool
    config: TrainConfig = field(default_factory=TrainConfig)
    net: QuantileMLP | None = None
    mu: np.ndarray | None = None
    sd: np.ndarray | None = None
    history: list = field(default_factory=list)

    def _x(self, w: Windows) -> tuple[torch.Tensor, np.ndarray]:
        x, base = build_inputs(w, self.hybrid)
        x = (x - self.mu) / self.sd
        return torch.from_numpy(x.astype(np.float32)), base

    def fit(self, train: Windows, val: Windows) -> "QuantileForecaster":
        cfg = self.config
        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        x_raw, _ = build_inputs(train, self.hybrid)
        self.mu = x_raw.mean(axis=0)
        self.sd = x_raw.std(axis=0) + 1e-6
        xt, bt = self._x(train)
        yt = torch.from_numpy((train.target - bt).astype(np.float32))
        bt_t = torch.from_numpy(bt)
        xv, bv = self._x(val)
        yv = torch.from_numpy((val.target - bv).astype(np.float32))

        self.net = QuantileMLP(xt.shape[1], cfg.hidden)
        opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr,
                               weight_decay=cfg.weight_decay)
        best, best_state, bad = float("inf"), None, 0
        n = len(xt)
        for epoch in range(cfg.max_epochs):
            self.net.train()
            order = rng.permutation(n)
            for s in range(0, n, cfg.batch):
                idx = torch.from_numpy(order[s:s + cfg.batch])
                q_rel = self.net(xt[idx])
                loss = pinball_loss(yt[idx], q_rel)
                if cfg.physics_weight:
                    level_q = q_rel + bt_t[idx].unsqueeze(-1)
                    loss = loss + cfg.physics_weight * feasibility_penalty(level_q)
                opt.zero_grad()
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                val_loss = float(pinball_loss(yv, self.net(xv)))
            self.history.append(val_loss)
            if val_loss < best - 1e-6:
                best, bad = val_loss, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
        self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    def predict(self, w: Windows) -> np.ndarray:
        x, base = self._x(w)
        with torch.no_grad():
            q_rel = self.net(x).numpy()
        return base[..., None] + q_rel

    def save(self, path: str) -> None:
        torch.save({"hybrid": self.hybrid, "state": self.net.state_dict(),
                    "mu": self.mu, "sd": self.sd,
                    "hidden": self.config.hidden}, path)


def train_ensemble(train: Windows, val: Windows, hybrid: bool,
                   members: int = 5, base_seed: int = 0,
                   **cfg) -> list[QuantileForecaster]:
    return [QuantileForecaster(hybrid, TrainConfig(seed=base_seed + m, **cfg)).fit(train, val)
            for m in range(members)]


def predict_ensemble(models: list[QuantileForecaster], w: Windows) -> np.ndarray:
    """Stack member forecasts: (M, N, H, Q)."""
    return np.stack([m.predict(w) for m in models])
