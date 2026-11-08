# -*- coding: utf-8 -*-
"""Load the fused VICReg covariance kernel, with a pure-PyTorch fallback.

The native op is compiled on first use via torch.utils.cpp_extension.load. If a
toolchain is missing, `fused_covariance_loss` transparently falls back to the
reference implementation, so nothing in the training loop ever breaks.
"""
from __future__ import annotations

import pathlib

import torch

_EXT = None
_SRC = pathlib.Path(__file__).with_name("vicreg_cov.cpp")


def _load():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        _EXT = load(name="vicreg_cov", sources=[str(_SRC)], verbose=False)
    return _EXT


def reference_covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """The unfused PyTorch path this kernel replaces."""
    b, d = z.shape
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / (b - 1)
    off = cov - torch.diag(torch.diag(cov))
    return off.pow(2).sum() / d


class _FusedCovLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z):
        loss, zc, off = _load().forward(z)
        ctx.save_for_backward(zc, off)
        return loss

    @staticmethod
    def backward(ctx, grad_out):
        zc, off = ctx.saved_tensors
        return _load().backward(grad_out, zc, off)


def fused_covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """Fused covariance loss; falls back to reference if the build fails."""
    try:
        return _FusedCovLoss.apply(z)
    except Exception:
        return reference_covariance_loss(z)
