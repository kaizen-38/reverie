# -*- coding: utf-8 -*-
"""Part 4 (training loop): one JEPA step and the full training routine."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .losses import collapse_metric, jepa_loss
from .models import (Encoder, Predictor, ema_update, init_encoder_params,
                     init_predictor_params, init_target_encoder)


@dataclass
class JEPAConfig:
    embed_dim: int = 32
    action_dim: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.99
    sim_coeff: float = 25.0
    var_coeff: float = 25.0
    cov_coeff: float = 1.0
    batch_size: int = 256
    steps: int = 1500
    device: str = "cpu"
    log_every: int = 100
    history: list = field(default_factory=list)


def jepa_training_step(online: Encoder, target: Encoder, predictor: Predictor,
                       opt, batch, cfg: JEPAConfig):
    """One optimization step on a single transition batch. (step 023)"""
    obs, action, next_obs = batch["obs"], batch["action"], batch["next_obs"]

    z = online(obs)                                   # online embedding of s_t
    with torch.no_grad():
        target_z = target(next_obs)                   # stop-grad target of s_{t+1}
    pred_z = predictor(z, action)                     # rolled forward in latent space

    loss, stats = jepa_loss(pred_z, target_z, z, sim_coeff=cfg.sim_coeff,
                            var_coeff=cfg.var_coeff, cov_coeff=cfg.cov_coeff)

    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    ema_update(target, online, cfg.momentum)          # slow target tracks online

    stats["loss"] = loss.detach()
    stats["collapse"] = collapse_metric(z.detach())
    return stats


def train_jepa(dataset, cfg: JEPAConfig | None = None):
    """Train encoder + predictor on a transition dataset. (step 024)"""
    cfg = cfg or JEPAConfig()
    dev = torch.device(cfg.device)
    data = {k: v.to(dev) for k, v in dataset.items()}
    n = data["obs"].shape[0]

    online = init_encoder_params(cfg.embed_dim).to(dev)
    target = init_target_encoder(online)
    predictor = init_predictor_params(cfg.embed_dim, action_dim=cfg.action_dim).to(dev)

    opt = torch.optim.AdamW(list(online.parameters()) + list(predictor.parameters()),
                            lr=cfg.lr, weight_decay=cfg.weight_decay)

    g = torch.Generator(device=dev).manual_seed(0)
    for step in range(cfg.steps):
        idx = torch.randint(0, n, (cfg.batch_size,), generator=g, device=dev)
        batch = {"obs": data["obs"][idx], "action": data["action"][idx],
                 "next_obs": data["next_obs"][idx]}
        stats = jepa_training_step(online, target, predictor, opt, batch, cfg)
        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            row = {"step": step, **{k: float(v) for k, v in stats.items()}}
            cfg.history.append(row)
            print(f"step {step:5d}  loss {row['loss']:.4f}  pred {row['pred']:.4f}  "
                  f"cov {row['cov']:.4f}  collapse(std) {row['collapse']:.3f}")

    return {"online": online, "target": target, "predictor": predictor, "cfg": cfg}
