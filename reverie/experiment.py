# -*- coding: utf-8 -*-
"""Step 038: the end-to-end JEPA world-model experiment.

Collect transitions, train the JEPA, probe the representation for recovered
state, measure multi-step latent drift, and run the MPC planner. One call,
every metric the project is judged on.
"""
from __future__ import annotations

import torch

from .env import build_transition_dataset
from .planning import evaluate_planner
from .probe import multi_step_prediction_error, probe_state_recovery, train_linear_probe
from .train import JEPAConfig, train_jepa


def jepa_world_model_experiment(*, n_transitions=20000, size=28, device="cpu",
                                cfg: JEPAConfig | None = None, n_eval_episodes=50,
                                probe_n=4000, verbose=True):
    """Run the full pipeline and return a dict of metrics. (step 038)"""
    cfg = cfg or JEPAConfig(device=device)
    cfg.device = device

    data = build_transition_dataset(n_transitions, size=size, device=device)
    if verbose:
        print(f"collected {n_transitions} transitions; training JEPA on {device}")

    trained = train_jepa(data, cfg)
    online, predictor = trained["online"], trained["predictor"]

    probe = train_linear_probe(online, data["obs"][:probe_n], data["state"][:probe_n],
                               embed_dim=cfg.embed_dim, device=device)
    probe_metrics = probe_state_recovery(probe, online, data["obs"][:probe_n],
                                         data["state"][:probe_n], device=device)

    g = torch.Generator(device=torch.device(device)).manual_seed(7)
    eval_states = torch.rand(256, 2, generator=g, device=device)
    eval_actions = torch.randint(0, 4, (256, 10), generator=g, device=device)
    drift = multi_step_prediction_error(online, predictor, eval_states, eval_actions,
                                        size=size)

    plan_metrics = evaluate_planner(online, predictor, n_episodes=n_eval_episodes,
                                    device=device, size=size)

    metrics = {
        **probe_metrics,
        "multi_step_error": [float(x) for x in drift],
        "latent_drift_h10": float(drift[-1]),
        **plan_metrics,
        "final_collapse_std": cfg.history[-1]["collapse"] if cfg.history else None,
    }
    if verbose:
        print("\n=== Reverie results ===")
        print(f"probe R^2 (state recovery): {metrics['probe_r2']:.3f}")
        print(f"probe MSE:                  {metrics['probe_mse']:.5f}")
        print(f"latent drift @ H=10:        {metrics['latent_drift_h10']:.4f}")
        print(f"MPC success rate:           {metrics['success_rate']*100:.1f}%  "
              f"({plan_metrics['n_episodes']} goals)")
        print(f"MPC mean final distance:    {metrics['mean_final_dist']:.4f}")
        print(f"collapse std (healthy>0):   {metrics['final_collapse_std']:.3f}")
    return {"metrics": metrics, "trained": trained, "probe": probe}
