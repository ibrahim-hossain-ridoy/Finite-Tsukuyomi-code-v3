"""
Configuration for the v3 continual learning dominance-shift experiment.

Scientific rationale
--------------------
The v3 pipeline replaces v2's hand-designed ODEs with a trainable VAE+GRU
system (a "world model" in the sense of Ha & Schmidhuber, 2018). All
hyperparameters are centralized here so that (1) experiments are fully
reproducible given this single file, and (2) there are no magic numbers
scattered across the codebase that could silently change behavior.

The sweep parameters (R_VALUES, SEEDS) define the experimental design:
  - R_VALUES controls the rehearsal ratio — the fraction of old-environment
    data (Environment A / MNIST) mixed into new-environment training
    (Environment B / FashionMNIST), directly testing experience-replay
    methodology (Rolnick et al., "Experience Replay for Continual Learning,"
    NeurIPS 2019).
  - SEEDS provides independent replicates for statistical power.

Stage 1 trains on Environment A only. Stage 2 continues from the Stage 1
checkpoint on a mixed A+B stream at each rehearsal ratio r.
"""

import os

# ═══════════════════════════════════════════════
# Model Architecture
# ═══════════════════════════════════════════════
LATENT_DIM = 32           # VAE latent dimensionality
GRU_HIDDEN = 256          # GRU hidden state size
GRU_LAYERS = 1            # Number of GRU layers

# ═══════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════
BATCH_SIZE = 128          # Images per training batch
SEQ_LENGTH = 8            # Pseudo-sequence length for GRU next-step prediction
LR = 1e-3                 # Adam learning rate
BETA = 1.0                # Weight on KL divergence term in VAE loss
GRU_LOSS_WEIGHT = 1.0     # Relative weight of GRU loss vs VAE loss

# Stage 1 — baseline training on Environment A only
STAGE1_MAX_EPOCHS = 50
STAGE1_PATIENCE = 5       # Early stopping patience (epochs without improvement)

# Stage 2 — rehearsal training on mixed A+B stream
STAGE2_EPOCHS = 15        # Fixed epoch count, no early stopping

# ═══════════════════════════════════════════════
# Experimental Sweep
# ═══════════════════════════════════════════════
R_VALUES = [0.0, 0.20, 0.30, 0.40, 0.50, 1.0]
SEEDS = [1, 2, 3, 4, 5]

# ═══════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════
ROLLOUT_STEPS = 50        # Steps per free-running GRU rollout
ROLLOUT_REPEATS = 10      # Independent rollouts averaged per dmn_proxy measurement
PROBE_CV_FOLDS = 5        # Stratified k-fold CV for the linear domain probe
PROBE_MAX_ITER = 1000     # LogisticRegression solver max iterations

# ═══════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════
DATA_DIR = "./data"
CHECKPOINT_DIR = "./checkpoints"
RESULTS_DIR = "./results_v3"
RAW_RESULTS_DIR = os.path.join(RESULTS_DIR, "raw")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# ═══════════════════════════════════════════════
# Debug Overrides (activated by --debug flag)
# ═══════════════════════════════════════════════
# These reduce data, epochs, and repeats so the full pipeline completes
# in under 2 minutes on CPU for rapid integration testing.
DEBUG_TRAIN_SUBSET = 500       # Training images per domain
DEBUG_EPOCHS = 1               # Used for both Stage 1 and Stage 2
DEBUG_SEEDS = [1]
DEBUG_R_VALUES = [0.40]
DEBUG_ROLLOUT_REPEATS = 2
DEBUG_STAGE1_PATIENCE = 1

# ═══════════════════════════════════════════════
# Reproducibility
# ═══════════════════════════════════════════════
# torch.backends.cudnn.deterministic and benchmark settings are applied
# at runtime in run_experiment_v3.py, not here, because they are runtime
# state rather than configuration values.
