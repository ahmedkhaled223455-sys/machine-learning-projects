"""Transformer motion forecaster (PyTorch).

Architecture (a compact version of the factorized designs used in AV
motion forecasting, e.g. Scene Transformer / Wayformer-style encoders):

  1. Temporal encoder   -- self-attention over each agent's 2 s history
  2. Social encoder     -- self-attention across agents (target + neighbours),
                           with a key-padding mask for empty neighbour slots.
                           No positional encoding across agents, so the model
                           is invariant to neighbour ordering by construction.
  3. Mode decoder       -- K learned mode queries self-attend, then
                           cross-attend to the scene tokens; each query emits
                           one future trajectory and a confidence score.
  4. Physics anchor     -- trajectories are predicted as residuals on a
                           constant-velocity rollout, so the network only has
                           to learn the deviation from simple kinematics.

Training uses the winner-takes-all objective standard for multimodal
forecasting: regress only the mode closest to the truth, and train the
scores to classify which mode that was.

Multi-head attention is written from primitives (not nn.MultiheadAttention)
and is checked against an independent NumPy reference in the tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from traj_dataset import F_STEPS, H_STEPS
from traj_sim import DT

POS_SCALE = 10.0  # metres -> network units


class MultiHeadAttention(nn.Module):
    def __init__(self, d: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d % n_heads == 0
        self.h, self.dh = n_heads, d // n_heads
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
        self.last_weights = None

    def forward(self, xq, xkv, key_mask=None):
        """xq (B, Lq, d), xkv (B, Lk, d), key_mask (B, Lk) bool, True = keep."""
        B, Lq, d = xq.shape
        Lk = xkv.shape[1]
        q = self.q(xq).view(B, Lq, self.h, self.dh).transpose(1, 2)
        k = self.k(xkv).view(B, Lk, self.h, self.dh).transpose(1, 2)
        v = self.v(xkv).view(B, Lk, self.h, self.dh).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dh)
        if key_mask is not None:
            scores = scores.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        w = torch.softmax(scores, dim=-1)
        self.last_weights = w.detach()
        out = (self.drop(w) @ v).transpose(1, 2).reshape(B, Lq, d)
        return self.o(out)


class Block(nn.Module):
    """Pre-norm transformer block; self-attention, or cross-attention if kv is given."""

    def __init__(self, d: int, n_heads: int, ff_mult: int = 2, dropout: float = 0.0,
                 cross: bool = False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln_kv = nn.LayerNorm(d) if cross else None
        self.attn = MultiHeadAttention(d, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(ff_mult * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, kv=None, key_mask=None):
        h = self.ln1(x)
        src = self.ln_kv(kv) if kv is not None else h
        x = x + self.drop(self.attn(h, src, key_mask))
        return x + self.drop(self.ff(self.ln2(x)))


def cv_anchor(hist: torch.Tensor, t_future: torch.Tensor, fit_steps: int = 5) -> torch.Tensor:
    """Constant-velocity rollout of the target (same rule as the CV baseline)."""
    xy = hist[:, 0, :, :2]
    v = (xy[:, -1] - xy[:, -1 - fit_steps]) / (fit_steps * DT)
    return xy[:, -1][:, None] + v[:, None] * t_future[None, :, None]


class TrajectoryTransformer(nn.Module):
    def __init__(self, d: int = 64, n_heads: int = 4, n_temporal: int = 2, n_social: int = 2,
                 n_decoder: int = 2, n_modes: int = 6, use_social: bool = True,
                 dropout: float = 0.1, in_feats: int = 4):
        super().__init__()
        self.n_modes = n_modes
        self.use_social = use_social
        self.embed = nn.Linear(in_feats, d)
        self.time_pos = nn.Parameter(torch.randn(H_STEPS, d) * 0.02)
        self.agent_type = nn.Embedding(2, d)  # 0 = target, 1 = neighbour
        self.temporal = nn.ModuleList([Block(d, n_heads, dropout=dropout) for _ in range(n_temporal)])
        self.social = nn.ModuleList([Block(d, n_heads, dropout=dropout) for _ in range(n_social)])
        self.mode_queries = nn.Parameter(torch.randn(n_modes, d) * 0.5)
        self.dec_self = nn.ModuleList([Block(d, n_heads, dropout=dropout) for _ in range(n_decoder)])
        self.dec_cross = nn.ModuleList([Block(d, n_heads, dropout=dropout, cross=True)
                                        for _ in range(n_decoder)])
        self.ln_out = nn.LayerNorm(d)
        self.traj_head = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, F_STEPS * 2))
        self.score_head = nn.Linear(d, 1)
        self.register_buffer("t_future", DT * torch.arange(1, F_STEPS + 1, dtype=torch.float32))

    def forward(self, hist: torch.Tensor, mask: torch.Tensor):
        """hist (B, A, T, 4) metres & m/s in the target frame; mask (B, A) bool.
        Returns trajectories (B, K, F, 2) in metres and mode logits (B, K)."""
        B, A, T, _ = hist.shape
        if not self.use_social:
            mask = mask.clone()
            mask[:, 1:] = False
        x = self.embed(hist / POS_SCALE) + self.time_pos
        types = torch.ones(A, dtype=torch.long, device=hist.device)
        types[0] = 0
        x = x + self.agent_type(types)[None, :, None, :]
        d = x.shape[-1]
        x = x.reshape(B * A, T, d)
        for blk in self.temporal:
            x = blk(x)
        tokens = x[:, -1].reshape(B, A, d)
        for blk in self.social:
            tokens = blk(tokens, key_mask=mask)
        q = self.mode_queries.unsqueeze(0).expand(B, -1, -1)
        for sa, ca in zip(self.dec_self, self.dec_cross):
            q = sa(q)
            q = ca(q, kv=tokens, key_mask=mask)
        q = self.ln_out(q)
        resid = self.traj_head(q).view(B, self.n_modes, F_STEPS, 2) * POS_SCALE
        traj = cv_anchor(hist, self.t_future)[:, None] + resid
        return traj, self.score_head(q).squeeze(-1)


def wta_loss(traj, scores, fut, cls_weight: float = 0.5):
    """Winner-takes-all: regress the closest mode, classify which one it was."""
    err = (traj - fut[:, None]).norm(dim=-1).mean(-1)  # (B, K)
    best = err.argmin(-1)
    sel = traj[torch.arange(len(traj), device=traj.device), best]
    reg = F.smooth_l1_loss(sel, fut)
    cls = F.cross_entropy(scores, best) if traj.shape[1] > 1 else torch.zeros((), device=traj.device)
    return reg + cls_weight * cls, reg.detach(), cls.detach()


@dataclass
class TrainConfig:
    epochs: int = 20
    batch: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0


def _tensors(data: dict):
    return (torch.as_tensor(data["hist"], dtype=torch.float32),
            torch.as_tensor(data["mask"], dtype=torch.bool),
            torch.as_tensor(data["fut"], dtype=torch.float32))


def train_model(model: nn.Module, data: dict, cfg: TrainConfig, log=print) -> list:
    g = torch.Generator().manual_seed(cfg.seed)
    hist, mask, fut = _tensors(data)
    n = len(hist)
    steps = math.ceil(n / cfg.batch)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.epochs * steps)
    history = []
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for s in range(steps):
            idx = perm[s * cfg.batch:(s + 1) * cfg.batch]
            traj, scores = model(hist[idx], mask[idx])
            loss, _, _ = wta_loss(traj, scores, fut[idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss) * len(idx)
        history.append(tot / n)
        log(f"  epoch {epoch + 1:2d}/{cfg.epochs}  loss {history[-1]:.4f}")
    return history


@torch.no_grad()
def predict(model: nn.Module, data: dict, batch: int = 1024):
    model.eval()
    hist, mask, _ = _tensors(data)
    trajs, scores = [], []
    for s in range(0, len(hist), batch):
        t, c = model(hist[s:s + batch], mask[s:s + batch])
        trajs.append(t.numpy())
        scores.append(c.numpy())
    return np.concatenate(trajs), np.concatenate(scores)
