# -*- coding: utf-8 -*-
"""Run the full Reverie experiment and write metrics + a plot.

    python scripts/run_experiment.py --device mps --steps 1500
"""
import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from reverie.experiment import jepa_world_model_experiment
from reverie.train import JEPAConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--transitions", type=int, default=20000)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--out", default="assets/metrics.json")
    args = ap.parse_args()

    torch.manual_seed(0)
    cfg = JEPAConfig(device=args.device, steps=args.steps)
    t0 = time.time()
    result = jepa_world_model_experiment(
        n_transitions=args.transitions, device=args.device, cfg=cfg,
        n_eval_episodes=args.episodes)
    result["metrics"]["wall_seconds"] = round(time.time() - t0, 1)
    result["metrics"]["device"] = args.device
    result["metrics"]["train_steps"] = args.steps

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result["metrics"], indent=2))
    print(f"\nwrote {out}  ({result['metrics']['wall_seconds']}s on {args.device})")

    try:
        _plot(result, out.parent / "training.png")
    except Exception as e:  # plotting is optional
        print(f"(skipped plot: {e})")


def _plot(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hist = result["trained"]["cfg"].history
    steps = [h["step"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].plot(steps, [h["pred"] for h in hist], label="prediction")
    ax[0].plot(steps, [h["collapse"] for h in hist], label="embedding std")
    ax[0].set_xlabel("step"); ax[0].set_title("training"); ax[0].legend()
    err = result["metrics"]["multi_step_error"]
    ax[1].plot(range(len(err)), err, marker="o")
    ax[1].set_xlabel("rollout horizon"); ax[1].set_title("latent drift")
    fig.tight_layout(); fig.savefig(path, dpi=120)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
