# -*- coding: utf-8 -*-
"""Correctness checks for every part of the pipeline. Run: pytest -q"""
import torch

from reverie import env, losses, models, planning, probe


def test_env_shapes_and_bounds():
    state, obs = env.env_reset(8)
    assert state.shape == (8, 2) and obs.shape == (8, 1, 28, 28)
    assert state.min() >= 0.0 and state.max() <= 1.0
    # Clamping holds the agent inside the room.
    edge = torch.ones(4, 2)
    moved = env.apply_action(edge, torch.tensor([0, 1, 2, 3]))
    assert moved.min() >= 0.0 and moved.max() <= 1.0


def test_render_moves_with_state():
    up = env.render_observation(torch.tensor([[0.5, 0.9]]))
    down = env.render_observation(torch.tensor([[0.5, 0.1]]))
    # "up" blob mass sits nearer the top rows than "down".
    top_mass_up = up[0, 0, :14].sum()
    top_mass_down = down[0, 0, :14].sum()
    assert top_mass_up > top_mass_down


def test_dataset_keys():
    d = env.build_transition_dataset(32)
    for k in ("obs", "action", "next_obs", "state", "next_state"):
        assert k in d
    assert d["action"].shape == (32,)


def test_ema_update_moves_toward_online():
    online = models.init_encoder_params(16)
    target = models.init_target_encoder(online)
    with torch.no_grad():
        for p in online.parameters():
            p.add_(1.0)
    before = next(target.parameters()).clone()
    models.ema_update(target, online, momentum=0.9)
    after = next(target.parameters())
    assert not torch.allclose(before, after)
    assert torch.all(after >= before)  # moved toward the larger online weights


def test_predictor_residual_shape():
    pred = models.init_predictor_params(16, action_dim=8)
    z = torch.randn(5, 16)
    a = torch.randint(0, 4, (5,))
    out = pred(z, a)
    assert out.shape == z.shape


def test_vicreg_detects_collapse():
    collapsed = torch.zeros(64, 16)
    healthy = torch.randn(64, 16)
    assert losses.variance_loss(collapsed) > losses.variance_loss(healthy)
    assert losses.collapse_metric(collapsed) < losses.collapse_metric(healthy)


def test_covariance_zero_for_identity_cov():
    torch.manual_seed(0)
    z = torch.randn(10000, 8)  # independent dims -> near-zero off-diagonal
    assert losses.covariance_loss(z) < 0.05


def test_rollout_length():
    pred = models.init_predictor_params(16, action_dim=8)
    z0 = torch.randn(3, 16)
    acts = torch.randint(0, 4, (3, 7))
    traj = probe.rollout_latent_dynamics(pred, z0, acts)
    assert traj.shape == (3, 8, 16)


def test_planner_prefers_lower_cost():
    seqs = torch.randint(0, 4, (10, 5))
    costs = torch.arange(10, dtype=torch.float32)
    best, c = planning.select_best_plan(seqs, costs)
    assert c == 0.0 and torch.equal(best, seqs[0])
