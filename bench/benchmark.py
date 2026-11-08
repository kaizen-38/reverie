# -*- coding: utf-8 -*-
"""Performance study: fused kernel vs reference, and torch.compile vs eager.

    python bench/benchmark.py --device cpu

Writes bench/results.json. All numbers are median wall time over repeated runs
after warmup; correctness is asserted against the reference before timing.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from kernels.fused import fused_covariance_loss, reference_covariance_loss
from reverie.train import JEPAConfig, jepa_training_step
from reverie.models import (init_encoder_params, init_predictor_params,
                            init_target_encoder)


def _time(fn, iters, warmup=10):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3  # ms


def bench_kernel(device, iters=200):
    rows = []
    for b, d in [(256, 32), (512, 64), (1024, 128), (2048, 256)]:
        z = torch.randn(b, d, device=device, requires_grad=True)

        def ref():
            g, = torch.autograd.grad(reference_covariance_loss(z), z)
            return g

        def fused():
            g, = torch.autograd.grad(fused_covariance_loss(z), z)
            return g

        assert torch.allclose(reference_covariance_loss(z), fused_covariance_loss(z), atol=1e-5)
        t_ref = _time(ref, iters)
        t_fused = _time(fused, iters)
        rows.append({"batch": b, "dim": d, "reference_ms": round(t_ref, 4),
                     "fused_ms": round(t_fused, 4), "speedup": round(t_ref / t_fused, 2)})
        print(f"  cov loss  B={b:5d} D={d:4d}  ref {t_ref:7.3f}ms  "
              f"fused {t_fused:7.3f}ms  {t_ref / t_fused:.2f}x")
    return rows


def bench_compile(device, iters=50):
    dev = torch.device(device)
    online = init_encoder_params(32).to(dev)
    target = init_target_encoder(online)
    predictor = init_predictor_params(32).to(dev)
    opt = torch.optim.AdamW(list(online.parameters()) + list(predictor.parameters()), lr=1e-3)
    cfg = JEPAConfig(device=device)
    batch = {"obs": torch.randn(256, 1, 28, 28, device=dev),
             "action": torch.randint(0, 4, (256,), device=dev),
             "next_obs": torch.randn(256, 1, 28, 28, device=dev)}

    def step():
        jepa_training_step(online, target, predictor, opt, batch, cfg)

    t_eager = _time(step, iters, warmup=5)

    compiled = torch.compile(online)
    row = {"eager_step_ms": round(t_eager, 3)}
    try:
        online.forward_orig = online.forward
        online_c = torch.compile(online, mode="reduce-overhead")

        def step_c():
            z = online_c(batch["obs"])
            with torch.no_grad():
                tz = target(batch["next_obs"])
            pz = predictor(z, batch["action"])
            from reverie.losses import jepa_loss
            loss, _ = jepa_loss(pz, tz, z)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        t_comp = _time(step_c, iters, warmup=8)
        row["compiled_step_ms"] = round(t_comp, 3)
        row["compile_speedup"] = round(t_eager / t_comp, 2)
        print(f"  training step  eager {t_eager:.3f}ms  compiled {t_comp:.3f}ms  "
              f"{t_eager / t_comp:.2f}x")
    except Exception as e:
        row["compiled_step_ms"] = None
        row["compile_note"] = str(e)[:160]
        print(f"  torch.compile unavailable here: {str(e)[:80]}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="bench/results.json")
    args = ap.parse_args()

    print(f"benchmarking on {args.device}")
    print("fused VICReg covariance kernel vs reference:")
    kernel = bench_kernel(args.device)
    print("torch.compile vs eager training step:")
    compile_row = bench_compile(args.device)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"device": args.device, "kernel": kernel,
                               "compile": compile_row}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
