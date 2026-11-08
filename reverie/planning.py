# -*- coding: utf-8 -*-
"""Part 6: random-shooting MPC that plans entirely in latent space.

The agent never plans over pixels. It encodes a goal image once, samples random
action sequences, rolls each one through the predictor, and keeps the first
action of whichever sequence lands closest to the goal embedding. Replanning
every step makes it model-predictive control (MPC). If the learned latent space
and dynamics are good, this closes the loop on a goal with no learned policy.
"""
from __future__ import annotations

import torch

from .env import N_ACTIONS, apply_action, render_observation
from .models import Encoder, Predictor
from .probe import rollout_latent_dynamics


@torch.no_grad()
def encode_goal(encoder: Encoder, goal_state: torch.Tensor, *, size=28) -> torch.Tensor:
    """Encode a goal observation into the latent target. (step 030)"""
    return encoder(render_observation(goal_state, size=size).to(goal_state.device))


def latent_cost(z: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
    """Squared latent distance to the goal. (step 031)"""
    return (z - z_goal).pow(2).sum(dim=-1)


@torch.no_grad()
def sample_action_sequences(n: int, horizon: int, *, n_actions=N_ACTIONS,
                            generator=None, device="cpu") -> torch.Tensor:
    """Sample `n` random action sequences of length `horizon`. (step 032)"""
    return torch.randint(0, n_actions, (n, horizon), generator=generator, device=device)


@torch.no_grad()
def score_action_sequences(predictor: Predictor, z0: torch.Tensor,
                           sequences: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
    """Roll every candidate sequence and score its terminal latent cost. (step 033)"""
    n = sequences.shape[0]
    z_start = z0.expand(n, -1)
    traj = rollout_latent_dynamics(predictor, z_start, sequences)
    return latent_cost(traj[:, -1], z_goal)


@torch.no_grad()
def select_best_plan(sequences: torch.Tensor, costs: torch.Tensor):
    """Pick the lowest-cost sequence. (step 034)"""
    best = int(torch.argmin(costs))
    return sequences[best], costs[best]


@torch.no_grad()
def mpc_step(encoder: Encoder, predictor: Predictor, state: torch.Tensor,
             z_goal: torch.Tensor, *, n_samples=512, horizon=10, size=28,
             generator=None):
    """Plan from the current state and return the first action to take. (step 035)"""
    dev = state.device
    z0 = encoder(render_observation(state, size=size).to(dev))
    seqs = sample_action_sequences(n_samples, horizon, generator=generator, device=dev)
    costs = score_action_sequences(predictor, z0, seqs, z_goal)
    best, _ = select_best_plan(seqs, costs)
    return int(best[0])


@torch.no_grad()
def run_mpc_episode(encoder: Encoder, predictor: Predictor, start_state, goal_state,
                    *, max_steps=50, step_size=0.1, size=28, success_radius=0.07,
                    n_samples=512, horizon=10, generator=None):
    """Drive the real environment toward a goal with latent MPC. (step 036)"""
    dev = start_state.device
    z_goal = encode_goal(encoder, goal_state, size=size)
    state = start_state.clone()
    traj = [state.clone()]
    for _ in range(max_steps):
        a = mpc_step(encoder, predictor, state, z_goal, n_samples=n_samples,
                     horizon=horizon, size=size, generator=generator)
        state = apply_action(state, torch.tensor([a], device=dev), step_size=step_size)
        traj.append(state.clone())
        if torch.linalg.norm(state - goal_state) < success_radius:
            return {"success": True, "steps": len(traj) - 1, "traj": traj,
                    "final_dist": float(torch.linalg.norm(state - goal_state))}
    return {"success": False, "steps": max_steps, "traj": traj,
            "final_dist": float(torch.linalg.norm(state - goal_state))}


@torch.no_grad()
def evaluate_planner(encoder: Encoder, predictor: Predictor, *, n_episodes=50,
                     seed=123, device="cpu", **episode_kw):
    """Success rate and mean final distance over random goals. (step 037)"""
    dev = torch.device(device)
    g = torch.Generator(device=dev).manual_seed(seed)
    succ, dists, steps = 0, [], []
    for _ in range(n_episodes):
        start = torch.rand(1, 2, generator=g, device=dev)
        goal = torch.rand(1, 2, generator=g, device=dev)
        out = run_mpc_episode(encoder, predictor, start, goal, generator=g, **episode_kw)
        succ += int(out["success"])
        dists.append(out["final_dist"])
        steps.append(out["steps"])
    return {
        "success_rate": succ / n_episodes,
        "mean_final_dist": float(torch.tensor(dists).mean()),
        "mean_steps": float(torch.tensor(steps, dtype=torch.float32).mean()),
        "n_episodes": n_episodes,
    }
