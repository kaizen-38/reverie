# -*- coding: utf-8 -*-
"""Part 1: a small, self-contained 2D room environment.

The agent is a point in the unit square. Four discrete actions nudge it by a
fixed step and the walls clamp it back inside. Observations are rendered pixel
images; the JEPA model never sees the underlying (x, y) state, it has to learn
one. All functions are deliberately pure and tensor-first so a whole dataset of
transitions can be collected in a few vectorized calls.
"""
from __future__ import annotations

import torch

# Action ids. Order matters: embed_action and the planner assume this layout.
ACTIONS = torch.tensor(
    [[0.0, 1.0],    # 0: up
     [0.0, -1.0],   # 1: down
     [-1.0, 0.0],   # 2: left
     [1.0, 0.0]],   # 3: right
    dtype=torch.float32,
)
N_ACTIONS = ACTIONS.shape[0]


def init_env_state(batch: int, *, generator: torch.Generator | None = None,
                   device="cpu") -> torch.Tensor:
    """Sample `batch` agent positions uniformly in the unit room. (step 001)"""
    return torch.rand(batch, 2, generator=generator, device=device)


def apply_action(state: torch.Tensor, action: torch.Tensor,
                 step_size: float = 0.1) -> torch.Tensor:
    """Move each agent by one discrete action and clamp to the room. (step 002)"""
    delta = ACTIONS.to(state)[action] * step_size
    return (state + delta).clamp_(0.0, 1.0)


def render_observation(state: torch.Tensor, size: int = 28,
                       sigma: float = 0.08) -> torch.Tensor:
    """Render a (B, 1, size, size) image: a Gaussian blob at the agent. (step 003)

    A soft blob (rather than a single hot pixel) gives the encoder a smooth,
    sub-pixel signal, which is what lets a linear probe later recover the
    continuous state so cleanly.
    """
    if state.dim() == 1:
        state = state[None]
    b = state.shape[0]
    coords = torch.linspace(0.0, 1.0, size, device=state.device)
    gy = coords.view(1, size, 1)
    gx = coords.view(1, 1, size)
    cx = state[:, 0].view(b, 1, 1)
    cy = state[:, 1].view(b, 1, 1)
    # Flip y so "up" (increasing y) moves the blob toward the top of the image.
    d2 = (gx - cx) ** 2 + (gy - (1.0 - cy)) ** 2
    img = torch.exp(-d2 / (2.0 * sigma * sigma))
    return img.unsqueeze(1)


def env_reset(batch: int, *, size: int = 28, generator: torch.Generator | None = None,
              device="cpu"):
    """Reset: return fresh state and its observation. (step 004)"""
    state = init_env_state(batch, generator=generator, device=device)
    return state, render_observation(state, size=size)


def env_step(state: torch.Tensor, action: torch.Tensor, *, size: int = 28,
             step_size: float = 0.1):
    """Advance one step: return (next_state, next_obs). (step 005)"""
    next_state = apply_action(state, action, step_size=step_size)
    return next_state, render_observation(next_state, size=size)


def collect_random_transitions(n: int, *, size: int = 28, step_size: float = 0.1,
                               seed: int = 0, device="cpu"):
    """Roll a uniform-random policy and collect `n` transitions. (step 006)

    Returns obs, action, next_obs, state, next_state. States are kept only so a
    probe can be scored against ground truth; the model is never given them.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    state = init_env_state(n, generator=g, device=device)
    action = torch.randint(0, N_ACTIONS, (n,), generator=g, device=device)
    next_state = apply_action(state, action, step_size=step_size)
    obs = render_observation(state, size=size)
    next_obs = render_observation(next_state, size=size)
    return obs, action, next_obs, state, next_state


def build_transition_dataset(n: int, **kw):
    """Package transitions into a dict of tensors ready for training. (step 007)"""
    obs, action, next_obs, state, next_state = collect_random_transitions(n, **kw)
    return {
        "obs": obs,
        "action": action,
        "next_obs": next_obs,
        "state": state,
        "next_state": next_state,
    }
