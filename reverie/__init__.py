# -*- coding: utf-8 -*-
"""Reverie: an action-conditioned JEPA world model that plans by imagining
latent rollouts. Pure PyTorch, no decoder, no pixels in the loop, no policy."""
from . import env, losses, models, planning, probe, train  # noqa: F401
from .experiment import jepa_world_model_experiment  # noqa: F401
from .train import JEPAConfig, train_jepa  # noqa: F401

__version__ = "0.1.0"
