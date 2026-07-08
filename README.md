# Ontological Inversion: V3 Computational Experiment

Code and data for the neural network experiment reported in "Ontological
Inversion: A Computational Framework for Permanent Generative Prior
Replacement via Contrast-Driven Prediction Error Under the Free Energy
Principle."

## Overview

This repository tests whether a trained neural network's default,
unconstrained generative output can be shifted from one environment to
another, and whether that shift is stable once achieved. A VAE+GRU system
is trained first on a baseline image domain, then on a mixed stream where
a swept rehearsal ratio `r` controls how much of the baseline domain is
retained during transition. Nothing about convergence, onset speed, or
trajectory shape is guaranteed by the model's construction. Behavior is
emergent from gradient descent and could have failed to show any of the
patterns reported below.

## Key Findings

**Decoupling of learning and acceptance.** Across the full sweep,
representational separability (`w_proxy`, a linear probe's accuracy at
distinguishing the two domains in latent space) stays within a narrow
band of 0.9686 to 0.9983 regardless of rehearsal ratio, including the
condition where the model never sees the target domain during transition.
Default generative behavior (`dmn_proxy`, what the model's free-running,
unconditioned output resembles) ranges across nearly the full 0 to 1
interval over the same sweep, from 0.0110 to 0.9989. A model can represent
a domain almost perfectly without its default output reflecting that
representation at all.

**Cognitive relapse.** At rehearsal ratios of 0.4 and 0.5, `dmn_proxy`
does not rise monotonically to a plateau. It rises rapidly early in
training, in most seeds approaching full capture of the target domain,
and then partially reverts toward the baseline domain while training
continues under unchanged conditions. Mean decline from peak to final
epoch is 0.241 at r=0.4 and 0.212 at r=0.5, against 0.001 at r=0.0 and
0.032 at r=0.2. This pattern is not present at r=0.0 or r=1.0.

## Architecture

- **V (vision) model**: convolutional VAE. Encoder: `Conv2d(1→32)` →
  `Conv2d(32→64)` → FC to `(mu, logvar)`, latent dimension 32. Decoder
  mirrors this back to 28×28. 341,825 parameters.
- **M (memory) model**: single-layer GRU, hidden size 256, predicts the
  next latent vector from the current one via teacher forcing on
  pseudo-sequences of length 8 built from the VAE's encoder means.
  Supports a free-running mode where it feeds its own output back in as
  the next input, with no real image input. 230,944 parameters.
- **Combined**: 572,769 trainable parameters, kept small enough to run
  the full sweep on a single consumer-grade GPU.

## Datasets

- **Environment A (baseline)**: MNIST
- **Environment B (target)**: FashionMNIST

Both use the standard 60,000/10,000 train/test split, with the training
partition further divided into 50,000 for training and the last 10,000
for validation. Chosen for tractability, not because they resemble a
realistic pair of competing environments. See the Limitations section of
the paper.

## Experimental Design

**Stage 1 (baseline)**: V+M trained jointly on Environment A only, until
validation loss plateaus (early stopping, patience 5 epochs, max 50
epochs). One run per seed; the checkpoint is reused across all six
Stage 2 conditions for that seed.

**Stage 2 (rehearsal)**: Training continues from the Stage 1 checkpoint
on a mixed stream: fraction `r` of each batch from Environment A,
`(1 - r)` from Environment B. Fixed at 15 epochs (0 through 14), no early
stopping, fresh optimizer state.

**Sweep**: r ∈ {0.0, 0.20, 0.30, 0.40, 0.50, 1.0} × seeds {1..15}
= 15 Stage 1 baselines + 90 Stage 2 runs = 1,350 per-epoch records.

r=0.0 is a negative control in one direction (immediate, unmixed switch
to B). r=1.0 is a negative control in the other (no exposure to B at
all during Stage 2).

## Metrics

Recorded at every Stage 2 epoch:

| Proxy | What it measures | How |
|---|---|---|
| `F_proxy` | Generative fit to the current training mixture | VAE-only ELBO on the r-matched validation stream |
| `w_proxy` | Representational separability of A and B | 5-fold CV logistic regression accuracy on the VAE encoder's latent means, full A/B test sets |
| `dmn_proxy` | Default, unconstrained generative behavior | Same probe applied to the GRU's free-running rollout output (10 rollouts × 50 steps per epoch) |

## Repository Structure

```
config_v3.py           All hyperparameters, r values, seed list
model_v3.py             VAE + GRU classes, loss functions, free-running rollout
probe_decoder_v3.py     Linear probe training (w_proxy) and dmn_proxy computation
run_experiment_v3.py    Main runner: Stage 1 + Stage 2 sweep, resumable
analyze_results_v3.py   Load raw .npz output, produce all_runs_v3.csv, summary_by_r_v3.csv, epoch_to_threshold_v3.csv
make_figures_v3.py      Read CSVs, produce the 6-panel results figure
requirements.txt        Python dependencies
SETUP_COLAB.md          Colab-specific setup guide
README.md               This file
```

## Quick Start

```bash
pip install -r requirements.txt

# Fast local sanity check (~2 min, CPU, no GPU needed)
python run_experiment_v3.py --debug
python analyze_results_v3.py --debug
python make_figures_v3.py --debug
```

## Running the Full Sweep

```bash
python run_experiment_v3.py      # 15 Stage 1 + 90 Stage 2 runs
python analyze_results_v3.py     # produces all_runs_v3.csv, summary_by_r_v3.csv, epoch_to_threshold_v3.csv
python make_figures_v3.py        # produces v3_results.png
```

On a T4-class GPU, the full sweep takes roughly 2 to 5 hours. See
`SETUP_COLAB.md` for Colab-specific setup, including Drive persistence
and resume instructions.

## Resume After Interruption

`run_experiment_v3.py` checks for existing checkpoints and `.npz` output
files before starting each run and skips anything already completed.
Re-running the script after an interruption continues from where it left
off rather than restarting from scratch.

## Output Structure

```
checkpoints/
  stage1_seed1.pt ... stage1_seed15.pt

results_v3/
  raw/
    seed1_r0.0.npz ... seed15_r1.0.npz   (90 files)
  all_runs_v3.csv
  summary_by_r_v3.csv
  epoch_to_threshold_v3.csv
  figures/
    v3_results.png
```

## Known Limitations

See the paper's Limitations section for the complete account. In brief:
the 15-epoch Stage 2 window is fixed and untested beyond that point, so
whether the relapse pattern at r=0.4/0.5 stabilizes, continues, or
reverses again is unknown; MNIST and FashionMNIST are a highly separable
toy domain pair, not a realistic model of competing environments; and the
rehearsal ratio `r` is a crude batch-composition parameter, not a
considered model of how contrast operates.

## References

- Ha, D. & Schmidhuber, J. (2018). "World Models." arXiv:1803.10122.
- Kingma, D.P. & Welling, M. (2013). "Auto-Encoding Variational Bayes." arXiv:1312.6114.
- Alain, G. & Bengio, Y. (2016). "Understanding Intermediate Layers Using Linear Classifier Probes." arXiv:1610.01644.
- Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T., & Wayne, G. (2019). "Experience Replay for Continual Learning." NeurIPS 2019.
- Raichle, M.E., et al. (2001). "A default mode of brain function." PNAS 98(2), 676-682.