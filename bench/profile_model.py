# -*- coding: utf-8 -*-
"""Profile a JEPA training step with torch.profiler and print the hot ops.

    python bench/profile_model.py --device cpu

This is the kind of trace you read before deciding what to fuse or move to a
custom kernel: it attributes wall time to individual aten ops on the device.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
from torch.profiler import ProfilerActivity, profile

from reverie.models import (init_encoder_params, init_predictor_params,
                            init_target_encoder)
from reverie.train import JEPAConfig, jepa_training_step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    dev = torch.device(args.device)
    online = init_encoder_params(32).to(dev)
    target = init_target_encoder(online)
    predictor = init_predictor_params(32).to(dev)
    opt = torch.optim.AdamW(list(online.parameters()) + list(predictor.parameters()), lr=1e-3)
    cfg = JEPAConfig(device=args.device)
    batch = {"obs": torch.randn(256, 1, 28, 28, device=dev),
             "action": torch.randint(0, 4, (256,), device=dev),
             "next_obs": torch.randn(256, 1, 28, 28, device=dev)}

    activities = [ProfilerActivity.CPU]
    if args.device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    for _ in range(5):  # warmup
        jepa_training_step(online, target, predictor, opt, batch, cfg)

    with profile(activities=activities, record_shapes=True) as prof:
        for _ in range(args.steps):
            jepa_training_step(online, target, predictor, opt, batch, cfg)

    sort_key = "self_cuda_time_total" if args.device == "cuda" else "self_cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=15))


if __name__ == "__main__":
    main()
