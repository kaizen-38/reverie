# -*- coding: utf-8 -*-
"""Part 5: multi-step latent rollouts and a linear probe.

Two questions about the learned representation. First, do predicted latents stay
faithful to reality over a multi-step rollout, or do they drift? Second, is the
true (x, y) state linearly decodable from a frozen embedding? A high probe R^2
is the evidence that the encoder learned geometry, not just a collapse-avoiding
hash.
"""
from __future__ import annotations

import torch

from .env import render_observation
from .models import Encoder, Predictor, encode_batch


@torch.no_grad()
def rollout_latent_dynamics(predictor: Predictor, z0: torch.Tensor,
                            actions: torch.Tensor) -> torch.Tensor:
    """Roll the predictor forward over an action sequence. (step 025)

    `actions` is (B, H). Returns (B, H + 1, D) including the start embedding.
    """
    z = z0
    traj = [z]
    for t in range(actions.shape[1]):
        z = predictor(z, actions[:, t])
        traj.append(z)
    return torch.stack(traj, dim=1)


@torch.no_grad()
def multi_step_prediction_error(encoder: Encoder, predictor: Predictor,
                                states, actions, *, step_size=0.1, size=28):
    """Per-horizon MSE between rolled latents and encoded true obs. (step 026)"""
    from .env import apply_action
    dev = states.device
    z0 = encoder(render_observation(states, size=size).to(dev))
    pred_traj = rollout_latent_dynamics(predictor, z0, actions)

    s = states
    true = [encoder(render_observation(s, size=size).to(dev))]
    for t in range(actions.shape[1]):
        s = apply_action(s, actions[:, t], step_size=step_size)
        true.append(encoder(render_observation(s, size=size).to(dev)))
    true_traj = torch.stack(true, dim=1)

    err = (pred_traj - true_traj).pow(2).mean(dim=(0, 2))  # (H + 1,)
    return err


def init_linear_probe(embed_dim: int, out_dim: int = 2) -> torch.nn.Linear:
    """A single linear layer: embedding -> state. (step 027)"""
    return torch.nn.Linear(embed_dim, out_dim)


def train_linear_probe(encoder: Encoder, obs, states, *, embed_dim=32,
                       epochs=300, lr=1e-2, device="cpu"):
    """Fit a linear probe on frozen embeddings to predict true state. (step 028)"""
    dev = torch.device(device)
    with torch.no_grad():
        z = encode_batch(encoder, obs.to(dev)).detach()
    y = states.to(dev)
    probe = init_linear_probe(embed_dim, y.shape[1]).to(dev)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(probe(z), y)
        loss.backward()
        opt.step()
    return probe


@torch.no_grad()
def probe_state_recovery(probe, encoder: Encoder, obs, states, *, device="cpu"):
    """Report probe MSE and R^2 of recovered state. (step 029)"""
    dev = torch.device(device)
    z = encode_batch(encoder, obs.to(dev))
    y = states.to(dev)
    pred = probe(z)
    mse = torch.nn.functional.mse_loss(pred, y)
    ss_res = (y - pred).pow(2).sum()
    ss_tot = (y - y.mean(dim=0, keepdim=True)).pow(2).sum()
    r2 = 1.0 - ss_res / ss_tot
    return {"probe_mse": float(mse), "probe_r2": float(r2)}
