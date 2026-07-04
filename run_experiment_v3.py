"""
Main experiment runner for the v3 continual learning dominance-shift experiment.

Scientific rationale
--------------------
This script executes the two-stage training protocol:

  Stage 1 — Baseline: Train VAE+GRU on Environment A (MNIST) until
  convergence (early stopping). This establishes the model's baseline
  representations before exposure to a new domain.

  Stage 2 — Rehearsal: Continue training the same model on a mixed
  data stream where a fraction r of each batch comes from Environment A
  (rehearsal) and (1-r) from Environment B (FashionMNIST). This directly
  tests experience-replay methodology (Rolnick et al., NeurIPS 2019).

The sweep covers 6 rehearsal ratios × 5 seeds = 30 independent Stage 2
runs. Stage 1 is run once per seed and reused across all r conditions.

Resume logic: if a .npz results file or Stage 1 checkpoint already exists,
that run is skipped. This makes the script safe to re-run after a Colab
session timeout without recomputing finished work.

Usage:
    python run_experiment_v3.py           # Full sweep (run on Colab GPU)
    python run_experiment_v3.py --debug   # Quick CPU test (~2 min)
"""

import argparse
import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset
import torchvision
import torchvision.transforms as transforms

import config_v3 as cfg
from model_v3 import VAE, GRUPredictor, vae_loss_fn, gru_loss_fn
from probe_decoder_v3 import train_domain_probe, compute_dmn_proxy

# ═══════════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Reproducibility
# ═══════════════════════════════════════════════════════════════════════

def set_all_seeds(seed):
    """
    Set all random seeds for full reproducibility.

    Setting torch.backends.cudnn.deterministic = True and benchmark = False
    trades some GPU speed for exact reproducibility across runs. This is
    important for a controlled experiment where we need to isolate the
    effect of the rehearsal ratio r from stochastic variation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════

def get_datasets(debug=False):
    """
    Download and prepare MNIST (Env A) and FashionMNIST (Env B).

    Both are normalized to [0, 1] with no other preprocessing, keeping
    the two domains on comparable footing for the probes.

    Returns train/val/test splits for each domain. Validation is carved
    from the training set (last 10K of the 60K).
    """
    transform = transforms.ToTensor()  # Scales to [0, 1]

    mnist_full = torchvision.datasets.MNIST(
        cfg.DATA_DIR, train=True, download=True, transform=transform
    )
    mnist_test = torchvision.datasets.MNIST(
        cfg.DATA_DIR, train=False, download=True, transform=transform
    )
    fmnist_full = torchvision.datasets.FashionMNIST(
        cfg.DATA_DIR, train=True, download=True, transform=transform
    )
    fmnist_test = torchvision.datasets.FashionMNIST(
        cfg.DATA_DIR, train=False, download=True, transform=transform
    )

    if debug:
        n = cfg.DEBUG_TRAIN_SUBSET
        # In debug mode, use small subsets (no val/train split distinction
        # needed for this tiny amount — just use first n for train, next
        # min(n, remaining) for val)
        mnist_train = Subset(mnist_full, range(n))
        mnist_val = Subset(mnist_full, range(n, min(2 * n, len(mnist_full))))
        fmnist_train = Subset(fmnist_full, range(n))
        fmnist_val = Subset(fmnist_full, range(n, min(2 * n, len(fmnist_full))))
        # Also subset test sets for speed
        mnist_test = Subset(mnist_test, range(min(n, len(mnist_test))))
        fmnist_test = Subset(fmnist_test, range(min(n, len(fmnist_test))))
    else:
        # Split 60K training into 50K train + 10K val
        mnist_train = Subset(mnist_full, range(50000))
        mnist_val = Subset(mnist_full, range(50000, 60000))
        fmnist_train = Subset(fmnist_full, range(50000))
        fmnist_val = Subset(fmnist_full, range(50000, 60000))

    return {
        'mnist_train': mnist_train,
        'mnist_val': mnist_val,
        'mnist_test': mnist_test,
        'fmnist_train': fmnist_train,
        'fmnist_val': fmnist_val,
        'fmnist_test': fmnist_test,
    }


class MixedBatchLoader:
    """
    Yields batches composed of r fraction from dataset A and (1-r) from
    dataset B, combined into a single shuffled batch.

    This matches real experience-replay methodology (Rolnick et al., 2019),
    where the replay buffer provides a fixed fraction of each training batch
    using actual stored data rather than synthetic/generated samples.

    For r=0.0: all from B (pure new environment, no rehearsal).
    For r=1.0: all from A (no new environment, continued baseline training).
    """

    def __init__(self, dataset_a, dataset_b, r, batch_size, shuffle=True):
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b
        self.r = r
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.n_a = max(0, round(r * batch_size))
        self.n_b = batch_size - self.n_a

        # Create separate loaders for each domain
        # Use drop_last=True to keep batch sizes consistent
        if self.n_a > 0:
            self.loader_a = DataLoader(
                dataset_a, batch_size=self.n_a, shuffle=shuffle, drop_last=True
            )
        else:
            self.loader_a = None

        if self.n_b > 0:
            self.loader_b = DataLoader(
                dataset_b, batch_size=self.n_b, shuffle=shuffle, drop_last=True
            )
        else:
            self.loader_b = None

    def __iter__(self):
        iter_a = iter(self.loader_a) if self.loader_a else None
        iter_b = iter(self.loader_b) if self.loader_b else None

        # Iterate until the larger dataset is exhausted
        exhausted = False
        while not exhausted:
            images_parts = []
            labels_parts = []

            if iter_a is not None:
                try:
                    img_a, lbl_a = next(iter_a)
                    images_parts.append(img_a)
                    labels_parts.append(lbl_a)
                except StopIteration:
                    exhausted = True
                    continue

            if iter_b is not None:
                try:
                    img_b, lbl_b = next(iter_b)
                    images_parts.append(img_b)
                    labels_parts.append(lbl_b)
                except StopIteration:
                    exhausted = True
                    continue

            if not images_parts:
                break

            images = torch.cat(images_parts, dim=0)
            labels = torch.cat(labels_parts, dim=0)

            # Shuffle the combined batch so A and B are interleaved
            perm = torch.randperm(images.size(0))
            yield images[perm], labels[perm]

    def __len__(self):
        # Approximate: limited by the smaller of the two datasets' iteration counts
        lens = []
        if self.loader_a:
            lens.append(len(self.loader_a))
        if self.loader_b:
            lens.append(len(self.loader_b))
        return min(lens) if lens else 0


# ═══════════════════════════════════════════════════════════════════════
# Training Helpers
# ═══════════════════════════════════════════════════════════════════════

def train_one_epoch(vae, gru, optimizer, data_loader, device):
    """
    Train VAE and GRU jointly for one epoch.

    The VAE is trained on reconstruction + KL, the GRU on next-step
    prediction in latent space. Pseudo-sequences are created by reshaping
    each batch's mu vectors (detached from the VAE graph so GRU loss
    doesn't alter encoder gradients) into chunks of SEQ_LENGTH.
    """
    vae.train()
    gru.train()

    total_loss_sum = 0.0
    vae_loss_sum = 0.0
    gru_loss_sum = 0.0
    n_batches = 0

    for images, _ in data_loader:
        images = images.to(device)
        batch_size_actual = images.size(0)

        # ── VAE forward ──
        recon, mu, logvar = vae(images)
        loss_vae, recon_l, kl_l = vae_loss_fn(recon, images, mu, logvar)

        # ── GRU forward (pseudo-sequences from mu) ──
        # Detach mu so GRU loss does not backprop through the VAE encoder.
        # This keeps the VAE's representation learning independent from
        # the GRU's prediction objective.
        n_usable = (batch_size_actual // cfg.SEQ_LENGTH) * cfg.SEQ_LENGTH
        if n_usable >= cfg.SEQ_LENGTH:
            mu_detached = mu[:n_usable].detach()
            mu_seq = mu_detached.view(-1, cfg.SEQ_LENGTH, cfg.LATENT_DIM)
            gru_input = mu_seq[:, :-1, :]    # (n_seq, seq_len-1, latent_dim)
            gru_target = mu_seq[:, 1:, :]    # (n_seq, seq_len-1, latent_dim)
            gru_pred, _ = gru(gru_input)
            loss_gru = gru_loss_fn(gru_pred, gru_target)
        else:
            # Batch too small for even one sequence (rare, debug mode only)
            loss_gru = torch.tensor(0.0, device=device)

        # ── Joint optimization ──
        total_loss = loss_vae + cfg.GRU_LOSS_WEIGHT * loss_gru
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        total_loss_sum += total_loss.item()
        vae_loss_sum += loss_vae.item()
        gru_loss_sum += loss_gru.item()
        n_batches += 1

    return {
        'total': total_loss_sum / max(n_batches, 1),
        'vae': vae_loss_sum / max(n_batches, 1),
        'gru': gru_loss_sum / max(n_batches, 1),
    }


@torch.no_grad()
def evaluate_val_loss(vae, gru, data_loader, device):
    """
    Compute validation loss (VAE + GRU) on a held-out set.
    Returns the combined loss (used for early stopping in Stage 1)
    and the VAE-only ELBO (used as F_proxy in Stage 2).
    """
    vae.eval()
    gru.eval()

    total_loss_sum = 0.0
    vae_loss_sum = 0.0
    n_batches = 0

    for images, _ in data_loader:
        images = images.to(device)
        batch_size_actual = images.size(0)

        recon, mu, logvar = vae(images)
        loss_vae, _, _ = vae_loss_fn(recon, images, mu, logvar)

        n_usable = (batch_size_actual // cfg.SEQ_LENGTH) * cfg.SEQ_LENGTH
        if n_usable >= cfg.SEQ_LENGTH:
            mu_seq = mu[:n_usable].view(-1, cfg.SEQ_LENGTH, cfg.LATENT_DIM)
            gru_input = mu_seq[:, :-1, :]
            gru_target = mu_seq[:, 1:, :]
            gru_pred, _ = gru(gru_input)
            loss_gru = gru_loss_fn(gru_pred, gru_target)
        else:
            loss_gru = torch.tensor(0.0)

        combined = loss_vae + cfg.GRU_LOSS_WEIGHT * loss_gru
        total_loss_sum += combined.item()
        vae_loss_sum += loss_vae.item()
        n_batches += 1

    avg_total = total_loss_sum / max(n_batches, 1)
    avg_vae = vae_loss_sum / max(n_batches, 1)
    return avg_total, avg_vae


# ═══════════════════════════════════════════════════════════════════════
# Checkpoint I/O
# ═══════════════════════════════════════════════════════════════════════

def save_checkpoint(vae, gru, epoch, val_loss, path):
    """Save model weights and training metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'vae_state_dict': vae.state_dict(),
        'gru_state_dict': gru.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss,
    }, path)


def load_checkpoint(path, device):
    """Load model weights from a checkpoint, returning fresh model instances."""
    vae = VAE().to(device)
    gru = GRUPredictor().to(device)
    ckpt = torch.load(path, map_location=device, weights_only=True)
    vae.load_state_dict(ckpt['vae_state_dict'])
    gru.load_state_dict(ckpt['gru_state_dict'])
    return vae, gru


# ═══════════════════════════════════════════════════════════════════════
# Stage 1 — Baseline Training
# ═══════════════════════════════════════════════════════════════════════

def run_stage1(seed, datasets, device, debug=False):
    """
    Train VAE+GRU on Environment A (MNIST) until validation loss plateaus.

    Uses early stopping with patience from config. The checkpoint is saved
    at the best validation loss and reused across all r conditions in
    Stage 2 for this seed.
    """
    ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, f"stage1_seed{seed}.pt")

    if os.path.exists(ckpt_path):
        logger.info(f"Skipping Stage 1 for seed={seed}, checkpoint found: {ckpt_path}")
        return ckpt_path

    logger.info(f"=== Stage 1: seed={seed} ===")
    set_all_seeds(seed)

    max_epochs = cfg.DEBUG_EPOCHS if debug else cfg.STAGE1_MAX_EPOCHS
    patience = cfg.DEBUG_STAGE1_PATIENCE if debug else cfg.STAGE1_PATIENCE

    vae = VAE().to(device)
    gru = GRUPredictor().to(device)
    optimizer = torch.optim.Adam(
        list(vae.parameters()) + list(gru.parameters()), lr=cfg.LR
    )

    train_loader = DataLoader(
        datasets['mnist_train'], batch_size=cfg.BATCH_SIZE,
        shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        datasets['mnist_val'], batch_size=cfg.BATCH_SIZE, shuffle=False,
    )

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(max_epochs):
        t0 = time.time()
        train_metrics = train_one_epoch(vae, gru, optimizer, train_loader, device)
        val_combined, val_vae = evaluate_val_loss(vae, gru, val_loader, device)
        elapsed = time.time() - t0

        logger.info(
            f"  Stage1 seed={seed} epoch={epoch}/{max_epochs-1} "
            f"train_loss={train_metrics['total']:.4f} "
            f"val_loss={val_combined:.4f} val_elbo={val_vae:.4f} "
            f"({elapsed:.1f}s)"
        )

        if val_combined < best_val_loss:
            best_val_loss = val_combined
            patience_counter = 0
            save_checkpoint(vae, gru, epoch, best_val_loss, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(
                    f"  Early stopping at epoch {epoch} "
                    f"(no improvement for {patience} epochs)"
                )
                break

    logger.info(f"  Stage 1 complete. Best val loss: {best_val_loss:.4f}")
    return ckpt_path


# ═══════════════════════════════════════════════════════════════════════
# Stage 2 — Rehearsal Training
# ═══════════════════════════════════════════════════════════════════════

def run_stage2(seed, r, stage1_ckpt, datasets, device, debug=False):
    """
    Continue training from Stage 1 checkpoint on a mixed A+B data stream.

    Each batch contains r fraction from Environment A and (1-r) from
    Environment B. Three metrics are logged at every epoch:
      F_proxy  — validation ELBO on a matched-proportion held-out set.
      w_proxy  — linear probe accuracy distinguishing A from B in latent space.
      dmn_proxy — P(env=B) from GRU free-running rollout decoded images.

    The optimizer is fresh (not loaded from Stage 1) to avoid momentum
    artifacts from the single-domain phase bleeding into the mixed phase.
    """
    results_path = os.path.join(cfg.RAW_RESULTS_DIR, f"seed{seed}_r{r}.npz")
    if os.path.exists(results_path):
        logger.info(f"Skipping seed={seed}, r={r}: results already exist at {results_path}")
        return

    logger.info(f"=== Stage 2: seed={seed}, r={r} ===")
    set_all_seeds(seed)

    n_epochs = cfg.DEBUG_EPOCHS if debug else cfg.STAGE2_EPOCHS
    n_rollout_repeats = cfg.DEBUG_ROLLOUT_REPEATS if debug else cfg.ROLLOUT_REPEATS

    # Load Stage 1 weights (fresh optimizer, no momentum carryover)
    vae, gru = load_checkpoint(stage1_ckpt, device)
    optimizer = torch.optim.Adam(
        list(vae.parameters()) + list(gru.parameters()), lr=cfg.LR
    )

    # Mixed training data
    train_loader = MixedBatchLoader(
        datasets['mnist_train'], datasets['fmnist_train'],
        r=r, batch_size=cfg.BATCH_SIZE, shuffle=True,
    )

    # Mixed validation data (same proportion as training)
    val_loader = MixedBatchLoader(
        datasets['mnist_val'], datasets['fmnist_val'],
        r=r, batch_size=cfg.BATCH_SIZE, shuffle=False,
    )

    # Test loaders for probe (always full test sets, not mixed)
    test_loader_a = DataLoader(
        datasets['mnist_test'], batch_size=256, shuffle=False,
    )
    test_loader_b = DataLoader(
        datasets['fmnist_test'], batch_size=256, shuffle=False,
    )

    metrics = {'epoch': [], 'F_proxy': [], 'w_proxy': [], 'dmn_proxy': []}

    for epoch in range(n_epochs):
        t0 = time.time()

        # ── Train ──
        train_metrics = train_one_epoch(vae, gru, optimizer, train_loader, device)

        # ── Metric 1: F_proxy (validation ELBO) ──
        _, f_proxy = evaluate_val_loss(vae, gru, val_loader, device)

        # ── Metric 2: w_proxy (linear probe accuracy) ──
        w_proxy, fitted_probe = train_domain_probe(
            vae, test_loader_a, test_loader_b, device
        )

        # ── Metric 3: dmn_proxy (free-running rollout classification) ──
        dmn_proxy = compute_dmn_proxy(
            gru, vae, fitted_probe, device,
            rollout_steps=cfg.ROLLOUT_STEPS,
            n_repeats=n_rollout_repeats,
        )

        elapsed = time.time() - t0

        metrics['epoch'].append(epoch)
        metrics['F_proxy'].append(f_proxy)
        metrics['w_proxy'].append(w_proxy)
        metrics['dmn_proxy'].append(dmn_proxy)

        logger.info(
            f"  Stage2 seed={seed} r={r} epoch={epoch}/{n_epochs-1} "
            f"F={f_proxy:.4f} w={w_proxy:.4f} dmn={dmn_proxy:.4f} "
            f"train_loss={train_metrics['total']:.4f} ({elapsed:.1f}s)"
        )

    # Save results
    os.makedirs(cfg.RAW_RESULTS_DIR, exist_ok=True)
    np.savez(
        results_path,
        epoch=np.array(metrics['epoch']),
        F_proxy=np.array(metrics['F_proxy']),
        w_proxy=np.array(metrics['w_proxy']),
        dmn_proxy=np.array(metrics['dmn_proxy']),
    )
    logger.info(f"  Saved results to {results_path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Run the v3 continual learning dominance-shift experiment."
    )
    parser.add_argument(
        '--debug', action='store_true',
        help=(
            "Run in debug mode: tiny data subset (500 images/domain), "
            "1 epoch, 1 seed, 1 r value. Completes in ~2 min on CPU."
        ),
    )
    args = parser.parse_args()

    debug = args.debug
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    if debug:
        logger.info("=== DEBUG MODE: reduced data/epochs/seeds ===")

    seeds = cfg.DEBUG_SEEDS if debug else cfg.SEEDS
    r_values = cfg.DEBUG_R_VALUES if debug else cfg.R_VALUES

    total_stage2_runs = len(seeds) * len(r_values)
    logger.info(
        f"Sweep: {len(seeds)} seeds × {len(r_values)} r values = "
        f"{total_stage2_runs} Stage 2 runs"
    )

    # Create output directories
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.RAW_RESULTS_DIR, exist_ok=True)
    os.makedirs(cfg.FIGURES_DIR, exist_ok=True)

    # Download datasets once
    logger.info("Loading datasets...")
    datasets = get_datasets(debug=debug)
    logger.info("Datasets ready.")

    sweep_start = time.time()
    completed = 0

    for seed in seeds:
        # ── Stage 1 (one per seed, reused across all r) ──
        stage1_ckpt = run_stage1(seed, datasets, device, debug=debug)

        # ── Stage 2 (one per seed × r combination) ──
        for r in r_values:
            run_stage2(seed, r, stage1_ckpt, datasets, device, debug=debug)
            completed += 1

            # ETA estimate
            elapsed = time.time() - sweep_start
            per_run = elapsed / completed
            remaining = (total_stage2_runs - completed) * per_run
            logger.info(
                f"Progress: {completed}/{total_stage2_runs} runs done. "
                f"ETA: {remaining / 60:.1f} min remaining."
            )

    total_elapsed = time.time() - sweep_start
    logger.info(f"All runs complete in {total_elapsed / 60:.1f} minutes.")


if __name__ == '__main__':
    main()
