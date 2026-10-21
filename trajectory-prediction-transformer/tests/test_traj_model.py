"""PyTorch model tests. Skipped automatically where torch is not installed
(the repo-wide lint job); the project's own CI job installs torch."""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attention_ref import multihead_attention  # noqa: E402
from traj_baselines import constant_velocity  # noqa: E402
from traj_dataset import F_STEPS, build_dataset  # noqa: E402
from traj_model import (MultiHeadAttention, TrajectoryTransformer,  # noqa: E402
                        wta_loss)


def _np(p):
    return p.detach().double().numpy()


def test_attention_matches_numpy_reference():
    torch.manual_seed(0)
    B, Lq, Lk, d, h = 3, 4, 6, 16, 4
    mha = MultiHeadAttention(d, h).double()
    xq, xkv = torch.randn(B, Lq, d, dtype=torch.float64), torch.randn(B, Lk, d, dtype=torch.float64)
    mask = torch.rand(B, Lk) > 0.4
    mask[:, 0] = True
    out = mha(xq, xkv, mask)
    ref, w_ref = multihead_attention(
        _np(xq), _np(xkv), _np(mha.q.weight), _np(mha.k.weight), _np(mha.v.weight), _np(mha.o.weight),
        _np(mha.q.bias), _np(mha.k.bias), _np(mha.v.bias), _np(mha.o.bias), n_heads=h, key_mask=mask.numpy())
    assert np.allclose(_np(out), ref, atol=1e-10)
    assert np.allclose(mha.last_weights.numpy(), w_ref, atol=1e-10)


def test_attention_matches_torch_sdpa():
    torch.manual_seed(1)
    B, L, d, h = 2, 5, 16, 2
    mha = MultiHeadAttention(d, h)
    x = torch.randn(B, L, d)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 0, 1, 1, 1]], dtype=torch.bool)
    ours = mha(x, x, mask)

    def heads(t):
        return t.view(B, L, h, d // h).transpose(1, 2)

    ref = torch.nn.functional.scaled_dot_product_attention(
        heads(mha.q(x)), heads(mha.k(x)), heads(mha.v(x)), attn_mask=mask[:, None, None, :])
    ref = mha.o(ref.transpose(1, 2).reshape(B, L, d))
    assert torch.allclose(ours, ref, atol=1e-5)


@pytest.fixture(scope="module")
def batch():
    d = build_dataset(4, seed=0)
    return (torch.as_tensor(d["hist"][:32], dtype=torch.float32),
            torch.as_tensor(d["mask"][:32]),
            torch.as_tensor(d["fut"][:32], dtype=torch.float32),
            d)


def _model(**kw):
    torch.manual_seed(0)
    return TrajectoryTransformer(d=32, n_heads=4, **kw).eval()


def test_output_shapes(batch):
    hist, mask, _, _ = batch
    traj, scores = _model(n_modes=6)(hist, mask)
    assert traj.shape == (len(hist), 6, F_STEPS, 2)
    assert scores.shape == (len(hist), 6)
    assert torch.isfinite(traj).all() and torch.isfinite(scores).all()


def test_padded_neighbour_slots_are_ignored(batch):
    hist, mask, _, _ = batch
    model = _model()
    with torch.no_grad():
        a, sa = model(hist, mask)
        noisy = hist.clone()
        noisy[~mask] = torch.randn_like(noisy[~mask]) * 100.0
        b, sb = model(noisy, mask)
    assert (~mask).any()
    assert torch.allclose(a, b, atol=1e-4) and torch.allclose(sa, sb, atol=1e-4)


def test_invariant_to_neighbour_order(batch):
    hist, mask, _, _ = batch
    model = _model()
    perm = torch.tensor([0, 3, 1, 8, 2, 7, 4, 6, 5])  # keep the target in slot 0
    with torch.no_grad():
        a, sa = model(hist, mask)
        b, sb = model(hist[:, perm], mask[:, perm])
    assert torch.allclose(a, b, atol=1e-4) and torch.allclose(sa, sb, atol=1e-4)


def test_no_social_ablation_ignores_neighbours(batch):
    hist, mask, _, _ = batch
    model = _model(use_social=False)
    with torch.no_grad():
        a, _ = model(hist, mask)
        moved = hist.clone()
        moved[:, 1:] += 5.0
        b, _ = model(moved, mask)
    assert torch.allclose(a, b, atol=1e-4)


def test_social_model_uses_neighbours(batch):
    hist, mask, _, _ = batch
    model = _model(use_social=True)
    with torch.no_grad():
        a, _ = model(hist, mask)
        moved = hist.clone()
        moved[:, 1:] += 5.0
        b, _ = model(moved, mask)
    assert not torch.allclose(a, b, atol=1e-4)


def test_zero_residual_reduces_to_constant_velocity(batch):
    hist, mask, _, d = batch
    model = _model(n_modes=3)
    last = model.traj_head[-1]
    torch.nn.init.zeros_(last.weight)
    torch.nn.init.zeros_(last.bias)
    with torch.no_grad():
        traj, _ = model(hist, mask)
    cv = constant_velocity(d["hist"][:32])
    for k in range(3):
        assert np.allclose(traj[:, k].numpy(), cv, atol=1e-3)


def test_wta_loss_selects_closest_mode():
    fut = torch.randn(4, F_STEPS, 2)
    traj = torch.stack([fut + 5.0, fut, fut - 5.0], 1)
    scores = torch.zeros(4, 3)
    loss, reg, cls = wta_loss(traj, scores, fut)
    assert float(reg) == 0.0
    assert np.isclose(float(cls), np.log(3), atol=1e-6)
    scores[:, 1] = 20.0
    _, _, cls2 = wta_loss(traj, scores, fut)
    assert float(cls2) < 1e-6


def test_overfits_a_small_batch(batch):
    hist, mask, fut, _ = batch
    torch.manual_seed(0)
    model = TrajectoryTransformer(d=32, n_heads=4, dropout=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(150):
        traj, scores = model(hist, mask)
        loss, reg, _ = wta_loss(traj, scores, fut)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(reg))
    assert losses[-1] < 0.3 * losses[0]
