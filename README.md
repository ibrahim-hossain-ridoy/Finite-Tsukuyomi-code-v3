# Continual Learning Dominance-Shift Experiment (v3)

## Overview

This codebase tests whether a trained neural network's internal "sense of
which reality is dominant" can be shifted from one environment to another,
and whether the speed of that shift depends on how much of the old
environment is retained during transition ("rehearsal ratio," r).

This is v3 of a larger project. v1/v2 used hand-designed ODEs to model this
process, which meant convergence was guaranteed by the functional form of
the equations rather than discovered. v3 replaces the ODEs with an actual
trained VAE+GRU network, so behavior is emergent from gradient descent and
can genuinely fail to show the predicted pattern. It did not fail. The
result below is real, measured, and includes an honest account of where it
is ambiguous.

## What This Measures

Three proxy metrics, computed at every training epoch of Stage 2:

| Proxy | What it stands for | How it's measured |
|---|---|---|
| `F_proxy` | Generative model quality | Validation ELBO (reconstruction + KL) |
| `w_proxy` | Domain separability in latent space | Linear probe accuracy on encoder's latent vectors, classifying real held-out images as domain A or B |
| `dmn_proxy` | Which domain the network "defaults to" when left to imagine freely | Linear probe accuracy applied to the GRU's free-running (unconditioned) rollout output |

`dmn_proxy` is the key metric. It answers: when the network is not being
fed real input and is just generating on its own, does its internal
"imagination" look like the old environment or the new one? This is
modeled after the free-running "dream" mode in Ha & Schmidhuber's World
Models, and the probe methodology follows Alain & Bengio (2016).

## Architecture

- **V (vision) model**: convolutional VAE. Encoder: Conv2d(1→32)→Conv2d(32→64)→FC
  to (mu, logvar), latent dim = 32. Decoder mirrors this back to 28×28.
- **M (memory) model**: single-layer GRU, hidden size 256, predicts the next
  latent vector from the current one. Supports a free-running mode where it
  feeds its own output back in as the next input, with no real image input.
- Combined parameter count kept under 2M so the whole sweep runs on a
  Colab free-tier T4 GPU.

## Datasets

- **Environment A (baseline)**: MNIST
- **Environment B (target)**: FashionMNIST

Chosen for simplicity and speed, not because they are a realistic model of
anything. See Limitations.

## Experimental Design

**Stage 1 (baseline)**: Train V+M jointly on Environment A only, until
validation loss plateaus (early stopping, patience = 5 epochs). One run per
seed, checkpoint reused across all Stage 2 conditions for that seed.

**Stage 2 (rehearsal)**: Continue training the same weights on a mixed
stream: fraction `r` of each batch from Environment A, `(1 - r)` from
Environment B. Fixed at 15 epochs (epoch 0 through 14), no early stopping.

**Sweep**: r ∈ {0.0, 0.20, 0.30, 0.40, 0.50, 1.0} × seeds {1..15}
= 15 Stage 1 baselines + 90 Stage 2 runs.

r=0.0 is a negative control (immediate, unmixed switch to B). r=1.0 is a
second negative control (no exposure to B at all — model should never
shift).

## Results Summary

Full sweep complete: 90/90 Stage 2 runs, 15 seeds per condition.

The main question was whether the network's `dmn_proxy` crosses 0.9
(strongly favors Environment B) within the 15-epoch window, and if so, how
fast, as a function of r.

| r | Mean epoch to cross 0.9 | Std | Crossed / Total |
|---|---|---|---|
| 0.0 | 0.00 | 0.00 | 15/15 |
| 0.2 | 0.20 | 0.41 | 15/15 |
| 0.3 | 0.73 | 0.59 | 15/15 |
| 0.4 | 2.13 | 1.06 | 15/15 |
| 0.5 | 3.57 | 2.07 | 7/15 |
| 1.0 | undefined | undefined | 0/15 |

**Finding 1 — dose-response is real and monotonic (r = 0.0–0.5).** Higher
retention of the old environment slows the shift toward the new one, in a
network where nothing about this relationship was hand-coded. This is the
core positive result.

**Finding 2 — r = 1.0 is a clean, complete block.** Zero of 15 seeds ever
crossed the threshold. `dmn_proxy` stayed near 0.0000–0.01 throughout. With
no exposure to Environment B, no shift occurs. This is the expected null
result and it held perfectly.

**Finding 3 — r = 0.5 is not bimodal, it's just slow.** 8 of 15 seeds never
crossed 0.9 within 15 epochs. Checking their actual final-epoch `dmn_proxy`
values (0.36, 0.55, 0.57, 0.58, 0.63, 0.64, 0.66, 0.89 — mean ≈ 0.61) shows
these runs were still progressing toward the threshold, not stuck near
zero the way r=1.0 runs were. This means the r=0.5 mean-epoch-to-cross
number in the table above is a **censored estimate**: it only reflects
the 7 seeds that finished in time, and understates how long the true
shift takes at this retention level.

The natural next inference — that these 8 seeds would have crossed given
more epochs — is plausible given the trajectory shape, but it is **not
tested**. No run in this sweep went past epoch 14. This is stated as an
open question, not a confirmed result.

## Analysis Methodology

`analyze_results_v3.py` performs censoring-aware analysis:

- For each `(seed, r)` run, finds the first epoch at which `dmn_proxy >= 0.9`.
- If no such epoch exists within the 15-epoch window, the run is marked
  `never_crossed = True` rather than silently dropped or imputed.
- Summary statistics (`mean_epoch`, `std_epoch`) are computed only over
  seeds that crossed, and are reported alongside `n_crossed / n_total` so
  the reader can see how much of each mean is based on a partial sample.
- `fraction_never_crossed` is reported per condition as its own column,
  not folded into the mean.

This matters because naively averaging `epoch_to_dmn_0.9` and dropping
`None` values (which is what a first-pass implementation did) silently
biases the r=0.5 mean toward the fastest-shifting seeds and hides the
condition's true variability.

Output: `results_v3/epoch_to_threshold_v3.csv` (one row per run) and the
printed summary table above.

## Figures

`make_figures_v3.py` produces a two-panel figure:

- **Panel A**: mean epoch-to-cross vs. r, error bars = std, annotated with
  `n_crossed/n_total` at each point. r=1.0 is deliberately **not** plotted
  as a data point on this axis (there is no defined mean when 0 seeds
  cross) — it is called out in a separate text annotation instead, to
  avoid the visual impression of a downward trend that isn't there.
- **Panel B**: fraction of seeds that never crossed the threshold, by r.
  This is the cleaner of the two panels for showing the r=0.5 and r=1.0
  effects at a glance.

Saved to `results_v3/figures/`.

## File Map

| File | Purpose |
|------|---------|
| `config_v3.py` | All hyperparameters, r values, seed list (no magic numbers elsewhere) |
| `model_v3.py` | VAE + GRU classes, loss functions, free-running rollout |
| `probe_decoder_v3.py` | Linear probe training (`w_proxy`) and `dmn_proxy` computation |
| `run_experiment_v3.py` | Main runner: Stage 1 + Stage 2 sweep, resumable |
| `analyze_results_v3.py` | Load `.npz` → censoring-aware CSVs + summary stats |
| `make_figures_v3.py` | Read CSVs → final 2-panel figure |
| `tests/test_pipeline.py` | pytest sanity checks |
| `requirements.txt` | Python dependencies |
| `SETUP_COLAB.md` | Colab-specific setup guide |

## Quick Start

```bash
pip install -r requirements.txt

# Fast local sanity check (~2 min, CPU, no GPU needed)
python run_experiment_v3.py --debug
python analyze_results_v3.py --debug
python make_figures_v3.py --debug
```

## Running the Full Sweep (Colab GPU — see SETUP_COLAB.md)

```bash
python run_experiment_v3.py      # 15 Stage 1 + 90 Stage 2 runs
python analyze_results_v3.py     # produces epoch_to_threshold_v3.csv, summary_by_r_v3.csv
python make_figures_v3.py        # produces the 2-panel figure
```

## Resume After Interruption

The runner checks for existing checkpoints and `.npz` output files before
starting each run and skips anything already completed. Re-running
`python run_experiment_v3.py` after a Colab disconnect will continue from
where it left off, not restart from scratch.

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
    final_figure_v3.png
```

## Running Tests

```bash
pytest tests/test_pipeline.py -v
```

## Known Limitations

- **n=15 seeds per condition.** Stronger than the original n=5 pilot, but
  still modest, especially at r=0.5 where variance is highest (std=2.07)
  and censoring is heaviest (53% never crossed).
- **The 15-epoch window is fixed and untested beyond that point.** The
  claim that r=0.5's censored seeds "would eventually cross" is an
  inference from trajectory shape, not a demonstrated result. A follow-up
  run extending to 25–30 epochs for r=0.4 and r=0.5 specifically would
  resolve this directly.
- **MNIST vs FashionMNIST are highly separable domains.** A linear probe
  can distinguish them with very high accuracy almost immediately, which
  likely makes the `dmn_proxy` and `w_proxy` metrics rise faster and
  cleaner than they would for more visually similar or naturalistic
  domains. The dose-response pattern is real within this setup; whether
  it generalizes to harder-to-distinguish environments is untested.
- **No architecture or hyperparameter sensitivity analysis.** Latent
  dimension, GRU hidden size, learning rate, and batch size were fixed
  before the sweep and not varied. Whether the dose-response relationship
  holds under different architectural choices is unknown.
- **`dmn_proxy` is a proxy, not a validated measure of anything like
  "default mode" in a biological sense.** The free-running rollout method
  is a structural analogy to DMN-related literature (see References), not
  a claim of biological correspondence.
- **Single dataset pair.** Only one baseline/target domain pair was
  tested. Whether retention ratio has the same effect across other domain
  pairs (e.g., more visually similar ones) is untested.

## References

- Ha & Schmidhuber (2018), "World Models," NeurIPS
- Rolnick, Ahuja, Schwarz, Lillicrap, Wayne (2019), "Experience Replay for
  Continual Learning," NeurIPS
- Alain & Bengio (2016), "Understanding Intermediate Layers Using Linear
  Classifier Probes"
- Kingma & Welling (2013), "Auto-Encoding Variational Bayes"
- Andrews-Hanna, Smallwood, Spreng (2014), "The default network and
  self-generated thought," Annals of the NY Academy of Sciences (motivates
  the free-running-generation framing of `dmn_proxy`, not a validated
  correspondence)