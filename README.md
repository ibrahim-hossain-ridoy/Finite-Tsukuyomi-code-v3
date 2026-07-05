# Continual Learning Dominance-Shift Experiment (v3)

## Overview

This codebase implements a computational neuroscience-adjacent ML experiment
studying how **rehearsal ratio** (the fraction of old-environment data mixed
into new-environment training) affects three proxy metrics in a VAE+GRU
"world model":

- **F_proxy**: generative quality (validation ELBO)
- **w_proxy**: domain separability in latent space (linear probe accuracy)
- **dmn_proxy**: default-mode imagination (where the GRU's free-running
  rollout lands when left to run freely)

The v3 pipeline replaces v2's hand-designed ODEs with actual trainable neural
networks, so convergence behavior is emergent from gradient descent and can
genuinely fail — making positive results scientifically meaningful.

## File Map

| File | Purpose |
|------|---------|
| `config_v3.py` | All hyperparameters (no magic numbers elsewhere) |
| `model_v3.py` | VAE + GRU classes, loss functions, free-running rollout |
| `probe_decoder_v3.py` | Linear probe training (w_proxy) and dmn_proxy computation |
| `run_experiment_v3.py` | Main runner: Stage 1 + Stage 2 sweep |
| `analyze_results_v3.py` | Load .npz → CSVs + Pearson correlations |
| `make_figures_v3.py` | Read CSVs → 4-panel figure |
| `tests/test_pipeline.py` | pytest sanity checks |
| `requirements.txt` | Python dependencies |
| `SETUP_COLAB.md` | Colab-specific setup guide |

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the debug self-test (~2 min on CPU)

```bash
python run_experiment_v3.py --debug
```

This runs the full pipeline (Stage 1 + one Stage 2 run) on 500 training
images per domain, 1 epoch, 1 seed, 1 r value (0.40).

### 3. Analyze debug output

```bash
python analyze_results_v3.py --debug
python make_figures_v3.py --debug
```

### 4. Run the full sweep (on Colab GPU — see SETUP_COLAB.md)

```bash
python run_experiment_v3.py
```

This runs 5 Stage 1 baselines and 30 Stage 2 runs (6 r values × 5 seeds).

### 5. Analyze full results

```bash
python analyze_results_v3.py
python make_figures_v3.py
```

## Running a Single (r, seed) Combination

For quick testing of a specific condition, you can modify `config_v3.py`
temporarily:

```python
SEEDS = [3]
R_VALUES = [0.30]
```

Or use the `--debug` flag and adjust `DEBUG_SEEDS` and `DEBUG_R_VALUES`
in the config.

The runner will skip any already-completed runs, so you can add conditions
incrementally.

## Resume After Interruption

The runner automatically detects completed work:
- If a Stage 1 checkpoint exists for a seed, it's skipped.
- If a `.npz` results file exists for a (seed, r) pair, it's skipped.

Simply re-run `python run_experiment_v3.py` to resume where you left off.

## Output Structure

```
checkpoints/
  stage1_seed1.pt, ..., stage1_seed5.pt

results_v3/
  raw/
    seed1_r0.0.npz, seed1_r0.2.npz, ..., seed5_r1.0.npz
  all_runs_v3.csv        (every metric at every epoch for every run)
  summary_by_r_v3.csv    (final-epoch mean ± std by r)
  figures/
    v3_results.png       (4-panel figure)
```

## Running Tests

```bash
pytest tests/test_pipeline.py -v
```

## Experimental Design

**Stage 1** (baseline): Train VAE+GRU on MNIST only (early stopping,
patience=5). One run per seed, reused across all r conditions.

**Stage 2** (rehearsal): Continue from Stage 1 checkpoint on mixed batches
where fraction r comes from MNIST (rehearsal) and (1-r) from FashionMNIST.
Fixed 15 epochs, no early stopping.

Sweep: r ∈ {0.0, 0.20, 0.30, 0.40, 0.50, 1.0} × seeds {1, 2, 3, 4, 5}
= 30 independent Stage 2 runs.

### References

- Ha & Schmidhuber (2018), "World Models"
- Rolnick et al. (2019), "Experience Replay for Continual Learning," NeurIPS
- Alain & Bengio (2016), "Understanding Intermediate Layers Using Linear
  Classifier Probes"
- Kingma & Welling (2013), "Auto-Encoding Variational Bayes"


## Known Limitations (Pilot Run)

This is a pilot run with n=5 seeds per condition. Results should be treated
as preliminary, not confirmatory.

- At r=0.5, 2 of 5 seeds never crossed the dmn >= 0.9 threshold within the
  14-epoch training window. The reported mean epoch-to-threshold at this
  condition is computed only over the seeds that crossed, which likely
  understates the true retention effect (censored data, not missing at
  random).
- A larger seed count (15+ per condition, especially at r=0.4 and r=0.5)
  is needed before drawing any statistical conclusion about the
  dose-response relationship between retention ratio and dominance-shift
  speed.
