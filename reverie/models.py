# -*- coding: utf-8 -*-
"""Parts 2 and 3: the encoder, its slow EMA target, and the predictor.

JEPA keeps two encoders. The online encoder is trained by gradient descent; the
target encoder is an exponential moving average of it and provides the
stop-gradient prediction targets. The predictor rolls dynamics forward purely in
embedding space, conditioned on the action, so there is no decoder and no pixel
reconstruction anywhere in the loop.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .env import N_ACTIONS


class Encoder(nn.Module):
    """Pixel observation -> L2-normalized latent embedding. (steps 008, 009)"""

    def __init__(self, in_ch: int = 1, embed_dim: int = 32, width: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, stride=2, padding=1), nn.GroupNorm(8, width), nn.SiLU(),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1), nn.GroupNorm(8, width * 2), nn.SiLU(),
            nn.Conv2d(width * 2, width * 2, 3, stride=2, padding=1), nn.GroupNorm(8, width * 2), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(width * 2, embed_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def init_encoder_params(embed_dim: int = 32, **kw) -> Encoder:
    """Construct a fresh online encoder. (step 008)"""
    return Encoder(embed_dim=embed_dim, **kw)


def encoder_forward(encoder: Encoder, obs: torch.Tensor) -> torch.Tensor:
    """Encode a batch of observations to embeddings. (step 009)"""
    return encoder(obs)


@torch.no_grad()
def init_target_encoder(encoder: Encoder) -> Encoder:
    """Clone the online encoder into a frozen EMA target. (step 010)"""
    target = copy.deepcopy(encoder)
    for p in target.parameters():
        p.requires_grad_(False)
    return target


@torch.no_grad()
def ema_update(target: Encoder, online: Encoder, momentum: float = 0.99) -> None:
    """In-place target <- momentum * target + (1 - momentum) * online. (step 011)"""
    for pt, po in zip(target.parameters(), online.parameters()):
        pt.mul_(momentum).add_(po.detach(), alpha=1.0 - momentum)
    for bt, bo in zip(target.buffers(), online.buffers()):
        bt.copy_(bo)


def encode_batch(encoder: Encoder, obs: torch.Tensor, batch_size: int = 512,
                 *, no_grad: bool = True) -> torch.Tensor:
    """Encode a large tensor of observations in chunks. (step 012)"""
    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        return torch.cat([encoder(obs[i:i + batch_size])
                          for i in range(0, obs.shape[0], batch_size)], dim=0)


class Predictor(nn.Module):
    """Action-conditioned latent dynamics model. (steps 013-016)

    Predicts the *change* in embedding and adds it back as a residual, which
    keeps multi-step latent rollouts stable instead of drifting.
    """

    def __init__(self, embed_dim: int = 32, action_dim: int = 16, hidden: int = 128,
                 n_actions: int = N_ACTIONS):
        super().__init__()
        self.embed_dim = embed_dim
        self.action_embed = nn.Embedding(n_actions, action_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim + action_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, embed_dim),
        )

    def embed_action(self, action: torch.Tensor) -> torch.Tensor:  # (step 014)
        return self.action_embed(action)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:  # (step 015)
        a = self.embed_action(action)
        return z + self.net(torch.cat([z, a], dim=-1))


def init_predictor_params(embed_dim: int = 32, **kw) -> Predictor:
    """Construct a fresh predictor. (step 013)"""
    return Predictor(embed_dim=embed_dim, **kw)


def predict_next_embedding(predictor: Predictor, z: torch.Tensor,
                           action: torch.Tensor) -> torch.Tensor:
    """Predict the next embedding given current embedding and action. (step 016)"""
    return predictor(z, action)
