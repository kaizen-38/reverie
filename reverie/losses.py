# -*- coding: utf-8 -*-
"""Part 4: the VICReg losses that make JEPA training collapse-resistant.

Without a decoder or contrastive negatives, nothing stops the encoder from
mapping every observation to the same vector and driving the prediction loss to
zero. VICReg prevents that with two regularizers on the batch of embeddings: a
variance hinge that keeps every dimension above a target standard deviation, and
a covariance penalty that pushes the off-diagonal covariance toward zero so the
dimensions carry independent information.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def prediction_loss(pred_z: torch.Tensor, target_z: torch.Tensor) -> torch.Tensor:
    """MSE between predicted and (stop-gradient) target embeddings. (step 017)"""
    return F.mse_loss(pred_z, target_z)


def variance_loss(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """Hinge that keeps each embedding dimension's std at or above gamma. (step 018)"""
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    return F.relu(gamma - std).mean()


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """Sum of squared off-diagonal covariances, normalized by dim. (step 019)"""
    b, d = z.shape
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / (b - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return off_diag.pow(2).sum() / d


def vicreg_regularizer(z: torch.Tensor, var_coeff: float = 25.0,
                       cov_coeff: float = 1.0, gamma: float = 1.0):
    """Combine variance + covariance terms for one batch of embeddings. (step 020)"""
    v = variance_loss(z, gamma=gamma)
    c = covariance_loss(z)
    return var_coeff * v + cov_coeff * c, v, c


def jepa_loss(pred_z, target_z, online_z, *, sim_coeff=25.0, var_coeff=25.0,
              cov_coeff=1.0, gamma=1.0):
    """Full JEPA objective: prediction + VICReg on both embedding streams. (step 021)

    The regularizer is applied to the online embeddings and the predictions so
    that neither the encoder nor the predictor can collapse its output.
    """
    pred = prediction_loss(pred_z, target_z)
    reg_z, v_z, c_z = vicreg_regularizer(online_z, var_coeff, cov_coeff, gamma)
    reg_p, v_p, c_p = vicreg_regularizer(pred_z, var_coeff, cov_coeff, gamma)
    total = sim_coeff * pred + 0.5 * (reg_z + reg_p)
    stats = {
        "pred": pred.detach(),
        "var": (0.5 * (v_z + v_p)).detach(),
        "cov": (0.5 * (c_z + c_p)).detach(),
    }
    return total, stats


@torch.no_grad()
def collapse_metric(z: torch.Tensor) -> torch.Tensor:
    """Mean per-dimension std of embeddings; near 0 means collapse. (step 022)"""
    return torch.sqrt(z.var(dim=0, unbiased=False) + 1e-8).mean()
