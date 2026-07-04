"""
Linear probe and dmn_proxy computation for the v3 experiment.

Scientific rationale
--------------------
This module implements the "linear classifier probe" methodology from
Alain & Bengio (2016), "Understanding Intermediate Layers Using Linear
Classifier Probes." A fresh logistic regression is trained on the VAE
encoder's latent representations (mu, not sampled z, to remove stochastic
noise) to classify which environment (A or B) each image came from.

The probe's cross-validated accuracy (w_proxy) measures how linearly
separable the two domains are in latent space — a proxy for whether the
model has developed distinct internal representations for each environment
versus blending them.

dmn_proxy measures the model's default-mode behavior: when the GRU runs
freely (consuming its own predictions), what domain does it "imagine"?
This is computed by decoding free-running rollout images and classifying
them with the same probe. A dmn_proxy near 0.5 means the model's
imagination is balanced between domains; near 1.0 means it defaults to
Environment B.
"""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

import config_v3 as cfg


def _encode_dataset(vae, data_loader, device):
    """
    Encode all images in a DataLoader through the VAE encoder, returning
    the mu vectors (deterministic, not sampled) as a numpy array.

    Using mu rather than sampled z follows Alain & Bengio (2016): the
    probe should evaluate the learned representation, not a noisy sample.
    """
    vae.eval()
    all_mu = []
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            mu, _ = vae.encode(images)
            all_mu.append(mu.cpu().numpy())
    return np.concatenate(all_mu, axis=0)


def train_domain_probe(vae, test_loader_a, test_loader_b, device,
                       cv_folds=None, max_iter=None):
    """
    Train a linear classifier probe to distinguish Environment A from
    Environment B in the VAE's latent space.

    Uses scikit-learn's LogisticRegression with stratified k-fold CV,
    following Alain & Bengio (2016). The probe is trained on held-out
    test images from both domains — it never sees training data.

    Args:
        vae:            Trained VAE model (used in eval mode).
        test_loader_a:  DataLoader for Environment A test images.
        test_loader_b:  DataLoader for Environment B test images.
        device:         'cpu' or 'cuda'.
        cv_folds:       Number of stratified CV folds (default from config).
        max_iter:       Solver max iterations (default from config).

    Returns:
        accuracy:       Mean cross-validated accuracy (float in [0, 1]).
        fitted_probe:   LogisticRegression fitted on the full dataset,
                        ready for use in dmn_proxy computation.
    """
    if cv_folds is None:
        cv_folds = cfg.PROBE_CV_FOLDS
    if max_iter is None:
        max_iter = cfg.PROBE_MAX_ITER

    # Encode both domains
    mu_a = _encode_dataset(vae, test_loader_a, device)
    mu_b = _encode_dataset(vae, test_loader_b, device)

    # Labels: 0 = Environment A, 1 = Environment B
    X = np.concatenate([mu_a, mu_b], axis=0)
    y = np.concatenate([
        np.zeros(len(mu_a), dtype=int),
        np.ones(len(mu_b), dtype=int),
    ])

    # Cross-validated accuracy (the w_proxy metric)
    probe_cv = LogisticRegression(max_iter=max_iter, solver='lbfgs')
    scores = cross_val_score(
        probe_cv, X, y, cv=cv_folds, scoring='accuracy'
    )
    accuracy = float(np.mean(scores))

    # Fit on full data for reuse in dmn_proxy (not double-dipping: the
    # CV score above is already computed; this full fit is only used
    # to classify generated rollout images, not to report accuracy).
    fitted_probe = LogisticRegression(max_iter=max_iter, solver='lbfgs')
    fitted_probe.fit(X, y)

    return accuracy, fitted_probe


def compute_dmn_proxy(gru, vae, fitted_probe, device,
                      rollout_steps=None, n_repeats=None):
    """
    Compute dmn_proxy: the GRU's default-mode network proxy.

    Runs multiple free-running rollouts, decodes each generated latent
    to image space, classifies each image using the fitted domain probe,
    and returns the mean P(Environment B) across all frames and repeats.

    Multiple rollouts (n_repeats) are averaged because a single rollout
    is a noisy sample of the model's unconditioned generative behavior.

    Args:
        gru:            Trained GRUPredictor model.
        vae:            Trained VAE model (for decoding).
        fitted_probe:   LogisticRegression fitted by train_domain_probe.
        device:         'cpu' or 'cuda'.
        rollout_steps:  Steps per rollout (default from config).
        n_repeats:      Number of independent rollouts (default from config).

    Returns:
        dmn_proxy:      Float in [0, 1], mean P(env=B) across all generated
                        frames. Near 0 = model imagines mostly A; near 1 =
                        mostly B; near 0.5 = balanced.
    """
    if rollout_steps is None:
        rollout_steps = cfg.ROLLOUT_STEPS
    if n_repeats is None:
        n_repeats = cfg.ROLLOUT_REPEATS

    all_probs = []

    for repeat_idx in range(n_repeats):
        # Each rollout uses a different seed for the starting latent
        decoded_images = gru.free_running_rollout(
            steps=rollout_steps,
            vae_decoder=vae,
            seed=repeat_idx * 1000 + 42,   # deterministic but varied
            device=device,
        )

        # Flatten decoded images for the probe: (steps, 28*28)
        flat_images = []
        for img in decoded_images:
            flat_images.append(img.view(1, -1).numpy())
        flat_images = np.concatenate(flat_images, axis=0)  # (steps, 784)

        # Encode through VAE to get latent representations for probe
        # (the probe was trained on latent mu vectors, not raw pixels)
        with torch.no_grad():
            img_tensor = torch.cat(decoded_images, dim=0).to(device)
            mu, _ = vae.encode(img_tensor)
            mu_np = mu.cpu().numpy()

        # P(env=B) for each generated frame
        probs = fitted_probe.predict_proba(mu_np)[:, 1]  # column 1 = P(B)
        all_probs.append(probs)

    # Average across all frames and all repeats
    dmn_proxy = float(np.mean(np.concatenate(all_probs)))
    return dmn_proxy
