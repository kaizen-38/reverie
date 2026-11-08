# Reverie

An action-conditioned **JEPA world model** built from scratch in pure PyTorch. The
agent learns to represent a 2D room and to *imagine* how that room changes under
its actions, entirely in embedding space, then plans by rolling those imagined
latents toward a goal. There is **no decoder, no pixel reconstruction, and no
learned policy** anywhere in the loop.

The modeling follows the [Deep-ML "JEPA World Model from Scratch"](https://www.deep-ml.com/projects/jepa-world-model-from-scratch)
project (38 steps, 6 parts). On top of the model, Reverie adds a small
performance-engineering layer — a fused C++ loss kernel, a profiler pass, and a
`torch.compile` benchmark — because a model is only useful once it runs well on
the hardware you have.

## What a JEPA world model is

Instead of predicting pixels, JEPA predicts the *embedding* of the next
observation. Two encoders are kept: an online encoder trained by gradient
descent, and a slow exponential-moving-average (EMA) target encoder that
produces stop-gradient prediction targets. A predictor rolls the latent forward
conditioned on the action. Because there are no contrastive negatives and no
decoder, nothing stops the encoder from mapping everything to a single point —
so **VICReg** variance and covariance terms keep the embedding dimensions spread
out and decorrelated. The result is a representation you can plan in.

```
obs_t ──▶ online encoder ──▶ z_t ──┐
                                   ├─▶ predictor(z_t, a_t) ──▶ ẑ_{t+1}  ──▶ MSE ──▶ target z_{t+1}
action a_t ────────────────────────┘                                              ▲ (stop-grad)
obs_{t+1} ──▶ EMA target encoder ──────────────────────────────────────────────────┘
             + VICReg(variance, covariance) on z to prevent collapse
```

## Results

Trained on 20,000 random transitions for 4,000 steps on CPU (~5 min, seed 0):

| Metric | Value | What it means |
| --- | --- | --- |
| Linear-probe R² (state recovery) | **0.97** | True (x, y) agent state is linearly decodable from a frozen embedding — the encoder learned geometry, not a collapse-avoiding hash. |
| Embedding std (collapse monitor) | **1.02** | Healthy; VICReg held variance up (collapse would drive this toward 0). |
| Multi-step latent drift @ H=10 | **0.10** | Imagined 10-step rollouts stay close to the encoded ground truth instead of diverging. |
| MPC goal-reaching success | **74%** (50 goals) | Random-shooting MPC in latent space, no learned policy, mean final distance 0.093. |

Reproduce:

```bash
pip install -r requirements.txt
python scripts/run_experiment.py --device cpu --steps 4000   # writes assets/metrics.json + training.png
pytest -q                                                    # 9 correctness tests
```

## The performance layer

| Component | Result |
| --- | --- |
| Fused C++ covariance-loss kernel (`kernels/vicreg_cov.cpp`, via `torch.utils.cpp_extension`) | **1.25–1.54× faster** than the reference PyTorch path; forward value and analytic backward both verified against autograd with `torch.autograd.gradcheck`. |
| `torch.compile` vs eager training step | ~1× on this small CPU model (overhead-bound); the speedup shows up on GPU and larger models — the benchmark reports whatever it measures, honestly. |
| `torch.profiler` pass (`bench/profile_model.py`) | Attributes wall time per `aten` op, which is how you decide what to fuse in the first place. |

```bash
python bench/benchmark.py --device cpu        # writes bench/results.json
python bench/profile_model.py --device cpu    # prints the hot-op table
```

The covariance loss is a natural fusion target: the reference path materializes a
centered matrix, a full D×D covariance, and an off-diagonal copy. The kernel
fuses that into one pass and reuses the centered buffer in the backward. It is a
drop-in replacement (`kernels/fused.fused_covariance_loss`) with a transparent
pure-PyTorch fallback, so the repo still runs if no compiler is present.

## Layout

```
reverie/
  env.py          # Part 1  2D room: state, actions, pixel render, transition dataset
  models.py       # Parts 2-3  encoder, EMA target, action-conditioned predictor
  losses.py       # Part 4  VICReg variance/covariance, JEPA objective, collapse metric
  train.py        # Part 4  training step + loop
  probe.py        # Part 5  latent rollouts, multi-step error, linear probe
  planning.py     # Part 6  latent-space random-shooting MPC
  experiment.py   # Step 38  end-to-end experiment
kernels/          # fused C++ covariance kernel + loader/fallback
bench/            # kernel vs reference, torch.compile, profiler
tests/            # correctness checks for every part
scripts/          # run the full experiment
```

## Credits

Project spec from [Deep-ML](https://www.deep-ml.com/projects/jepa-world-model-from-scratch).
JEPA is due to LeCun et al.; VICReg to Bardes, Ponce, and LeCun. Implementation,
C++ kernel, and benchmarks are my own.

MIT licensed.
