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

## Core Theory

This experiment is built on a specific claim from the free energy
principle: a predictive system does not passively receive its
environment, it constructs a generative model of what causes its sensory
input, and continuously updates that model to minimize the gap between
prediction and evidence. What the system experiences at any moment is
that model's current best hypothesis, not a direct readout of the world.
This raises a question the paper treats as literal rather than
metaphorical: if a system's environment is a model it has learned, can a
different, engineered environment be trained into that same system so
thoroughly that it displaces the original as the system's default
hypothesis, the state this paper calls ontological inversion.

The VAE's evidence lower bound (ELBO) is not a loose analogy for
variational free energy, it is the same mathematical object up to sign:
both are bounds on surprisal obtained by trading reconstruction accuracy
against the complexity of an approximate posterior. Training the VAE by
minimizing its ELBO is, in this narrow computational sense, minimizing
free energy directly. This is what licenses treating the V+M system as a
structural instantiation of the theory rather than an unrelated
architecture that happens to share vocabulary with it.

MNIST and FashionMNIST stand in for two competing environments: a
baseline (Environment A) the system is originally trained on, and a
target (Environment B) the system is later pushed toward under varying
amounts of retained baseline exposure. They were chosen for tractability
and fast iteration, not because they resemble any realistic pair of
environments a generative system might be asked to choose between. A
linear probe separates them at over 96 percent accuracy in every tested
condition, including the one where the model never sees Environment B at
all. This makes the domain pair well suited to testing whether
ontological inversion can be induced and whether it holds, and poorly
suited to claiming anything about how large or how similar two competing
environments would need to be for a comparable effect to appear
elsewhere. See Known Limitations below.

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

## Model Architecture

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
for validation.

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

## Code Architecture and Execution Flow

This section explains how the code actually executes, not just what
files exist, so the design choices behind the reported results are
visible without reading the paper's Methods section.

### Why V and M share an optimizer but not a gradient path

`model_v3.py` defines two networks, `VAE` and `GRUPredictor`, trained by
a single Adam optimizer with a combined loss
(`vae_loss + GRU_LOSS_WEIGHT * gru_loss`). They share an optimizer step
for convenience and matched training progress, not because their
objectives are coupled. Inside `train_one_epoch()` in
`run_experiment_v3.py`, the VAE's encoder output (`mu`) is detached from
the computational graph before being reshaped into GRU input sequences:

```python
mu_detached = mu[:n_usable].detach()
mu_seq = mu_detached.view(-1, cfg.SEQ_LENGTH, cfg.LATENT_DIM)
```

This detach is the single most consequential line in the training loop.
Without it, the GRU's next-step prediction loss would backpropagate into
the VAE encoder, and the encoder would be incentivized to organize its
latent space to make next-step prediction easier, not to reconstruct
images faithfully. Detaching keeps the two objectives on separate
gradient paths: the VAE learns a representation driven purely by
reconstruction and KL, and the GRU learns to predict forward through
whatever representation the VAE happens to produce. This separation is
what makes `w_proxy` (a property of the VAE's representation alone) and
`dmn_proxy` (a property of the GRU's rollout through that representation)
measure genuinely different things rather than two views of the same
optimization pressure.

### Why the probe is retrained every epoch, twice, for different purposes

`probe_decoder_v3.py` fits a fresh `LogisticRegression` at every Stage 2
epoch, twice, for two purposes that are easy to conflate:

1. `train_domain_probe()` runs 5-fold stratified cross-validation and
   reports the mean held-out accuracy as `w_proxy`. This is the only
   place `w_proxy` comes from.
2. The same function then fits a second `LogisticRegression` on the full
   dataset, no held-out fold, and returns it as `fitted_probe`. This
   second fit is never used to report an accuracy number. It exists only
   so `compute_dmn_proxy()` has a classifier to score the GRU's generated
   frames against.

This split matters because reusing the cross-validated model to score
generated frames, or reporting the full-fit model's training accuracy as
`w_proxy`, would both be forms of the same error: using a number computed
one way to answer a question it was not computed to answer. The CV
accuracy answers "how separable are A and B in this latent space." The
full-fit model is a tool for a different downstream measurement, not a
second, better estimate of the same thing.

### How dmn_proxy is actually produced

`compute_dmn_proxy()` runs `gru.free_running_rollout()` ten times per
epoch, each seeded independently (`seed = repeat_idx * 1000 + 42`), for
50 steps each, with no real image input after the first step. At every
step, the GRU's own output becomes its next input; the resulting latent
is decoded to an image by the VAE decoder, then re-encoded by the VAE
encoder to recover the `mu` vector the probe was trained on, then
classified. The 500 resulting P(B) values (10 rollouts times 50 steps)
are averaged into a single `dmn_proxy` number for that epoch. Every value
in the F_proxy / w_proxy / dmn_proxy columns of `all_runs_v3.csv` is
produced by this loop, once per (seed, r, epoch) combination.

### How the threshold-crossing analysis handles censoring

`analyze_results_v3.py` finds, for each (seed, r) run, the first epoch
where `dmn_proxy >= 0.90`. Runs that never cross are kept as rows with
`epoch_to_dmn_0.9 = NaN` and `never_crossed = True`, not dropped.
`epoch_to_threshold_v3.csv` is this per-run table; the mean and standard
deviation reported anywhere in the paper for "epoch to crossing" are
computed only over rows where a crossing happened, with the fraction that
never crossed reported as a separate column, never folded into the mean.

### How the cognitive relapse statistic is computed, and a gap to close

The peak-versus-final `dmn_proxy` analysis behind the cognitive relapse
finding, the overshoot then partial regression seen at r=0.4 and r=0.5,
is **not currently a function in `analyze_results_v3.py`**. It was
computed directly against `all_runs_v3.csv` for the paper, not generated
by any script in this repository as shipped. To make it reproducible from
the documented pipeline, add this to `analyze_results_v3.py` and call it
from `main()`:

```python
def compute_relapse_stats(df):
    """
    Per-seed peak dmn_proxy, the epoch it occurs at, the final-epoch
    (epoch 14) value, and the decline between them. This is the
    calculation behind the cognitive relapse finding.
    """
    rows = []
    for (seed, r_val), group in df.groupby(['seed', 'r']):
        g = group.sort_values('epoch')
        vals = g['dmn_proxy'].values
        epochs = g['epoch'].values
        peak_idx = int(np.argmax(vals))
        rows.append({
            'seed': seed,
            'r': r_val,
            'peak_dmn': float(vals[peak_idx]),
            'peak_epoch': int(epochs[peak_idx]),
            'final_dmn': float(vals[-1]),
            'decline': float(vals[peak_idx] - vals[-1]),
        })
    return pd.DataFrame(rows)
```

The logic: per seed, find the maximum `dmn_proxy` reached anywhere in the
15-epoch run and the epoch it occurred at, compare it against the value
at epoch 14, and call the difference the decline. Averaging `decline`
across the 15 seeds for a given r reproduces the numbers in the paper's
overshoot-and-regression table. This depends on `all_runs_v3.csv`
existing, which itself depends on the fix described next.

### The all_runs_v3.csv dependency

As shipped, `analyze_results_v3.py`'s `main()` only writes
`summary_by_r_v3.csv` and `epoch_to_threshold_v3.csv`. It loads the full
per-epoch dataframe (`load_all_runs()`) but never saves it. Every
per-epoch analysis in this repository, including the relapse statistic
above, depends on that dataframe being saved as `all_runs_v3.csv`. Add
this inside `main()`, immediately after `df = load_all_runs(raw_dir)`:

```python
all_runs_path = os.path.join(out_dir, "all_runs_v3.csv")
df.to_csv(all_runs_path, index=False)
print(f"Saved: {all_runs_path}")
```

Without this line, a clean clone of this repository can reproduce the
summary and threshold-crossing tables in the paper, but not the
trajectory-level relapse finding, since the data it depends on is never
written to disk.

### File Reference

```
config_v3.py             All hyperparameters, r values, seed list
model_v3.py               VAE + GRU classes, loss functions, free-running rollout
probe_decoder_v3.py       Linear probe training (w_proxy) and dmn_proxy computation
run_experiment_v3.py      Main runner: Stage 1 + Stage 2 sweep, resumable
analyze_results_v3.py     Load raw .npz output, produce the three CSVs described above
make_figures_v3.py        Read CSVs, produce the 6-panel results figure
requirements.txt          Python dependencies
SETUP_COLAB.md            Colab-specific setup guide
README.md                 This file
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