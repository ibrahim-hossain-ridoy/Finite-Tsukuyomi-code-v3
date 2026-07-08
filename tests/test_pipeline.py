"""
Sanity checks for the v3 continual learning experiment pipeline.

These tests verify basic structural correctness: shapes, value ranges,
and API contracts. They run on CPU with no data download required for
most tests, completing in under 30 seconds.

Run:  pytest tests/test_pipeline.py -v
"""

import sys
import os

import numpy as np
import pytest
import torch

# Add parent directory to path so imports work when running from tests/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config_v3 as cfg
from model_v3 import VAE, GRUPredictor, vae_loss_fn, gru_loss_fn


class TestVAE:
    """Tests for the VAE model."""

    def setup_method(self):
        self.vae = VAE()
        self.batch = torch.randn(4, 1, 28, 28).clamp(0, 1)

    def test_forward_output_shape(self):
        """VAE forward pass should produce output matching input shape."""
        recon, mu, logvar = self.vae(self.batch)
        assert recon.shape == self.batch.shape, (
            f"Expected {self.batch.shape}, got {recon.shape}"
        )

    def test_forward_output_range(self):
        """VAE output should be in [0, 1] due to sigmoid activation."""
        recon, _, _ = self.vae(self.batch)
        assert recon.min() >= 0.0, f"Min value {recon.min()} < 0"
        assert recon.max() <= 1.0, f"Max value {recon.max()} > 1"

    def test_encode_shapes(self):
        """Encoder should produce mu and logvar of shape (B, latent_dim)."""
        mu, logvar = self.vae.encode(self.batch)
        expected = (4, cfg.LATENT_DIM)
        assert mu.shape == expected, f"mu shape {mu.shape} != {expected}"
        assert logvar.shape == expected, f"logvar shape {logvar.shape} != {expected}"

    def test_decode_shape(self):
        """Decoder should map latent vectors back to image shape."""
        z = torch.randn(4, cfg.LATENT_DIM)
        recon = self.vae.decode(z)
        assert recon.shape == (4, 1, 28, 28)


class TestGRU:
    """Tests for the GRU next-step predictor."""

    def setup_method(self):
        self.gru = GRUPredictor()

    def test_forward_output_shape(self):
        """GRU forward should match input sequence shape."""
        z_seq = torch.randn(2, 7, cfg.LATENT_DIM)
        z_pred, hidden = self.gru(z_seq)
        assert z_pred.shape == (2, 7, cfg.LATENT_DIM), (
            f"Expected (2, 7, {cfg.LATENT_DIM}), got {z_pred.shape}"
        )

    def test_hidden_state_shape(self):
        """GRU hidden state should have correct shape."""
        z_seq = torch.randn(2, 7, cfg.LATENT_DIM)
        _, hidden = self.gru(z_seq)
        assert hidden.shape == (cfg.GRU_LAYERS, 2, cfg.GRU_HIDDEN)

    def test_free_running_rollout(self):
        """Free-running rollout should produce the expected number of frames."""
        vae = VAE()
        steps = 10
        images = self.gru.free_running_rollout(
            steps=steps, vae_decoder=vae, seed=42, device='cpu'
        )
        assert len(images) == steps, f"Expected {steps} frames, got {len(images)}"
        for img in images:
            assert img.shape == (1, 1, 28, 28), f"Frame shape {img.shape} unexpected"


class TestLossFunctions:
    """Tests for loss computation."""

    def test_vae_loss_finite(self):
        """VAE loss should produce finite values."""
        vae = VAE()
        x = torch.randn(4, 1, 28, 28).clamp(0, 1)
        recon, mu, logvar = vae(x)
        total, recon_l, kl_l = vae_loss_fn(recon, x, mu, logvar)
        assert torch.isfinite(total), f"Total loss not finite: {total}"
        assert torch.isfinite(recon_l), f"Recon loss not finite: {recon_l}"
        assert torch.isfinite(kl_l), f"KL loss not finite: {kl_l}"

    def test_gru_loss_finite(self):
        """GRU loss should produce finite values."""
        pred = torch.randn(2, 5, cfg.LATENT_DIM)
        target = torch.randn(2, 5, cfg.LATENT_DIM)
        loss = gru_loss_fn(pred, target)
        assert torch.isfinite(loss), f"GRU loss not finite: {loss}"


class TestConfig:
    """Tests for configuration values."""

    def test_config_loads(self):
        """Config should load without errors and have expected attributes."""
        required_attrs = [
            'LATENT_DIM', 'GRU_HIDDEN', 'GRU_LAYERS',
            'BATCH_SIZE', 'SEQ_LENGTH', 'LR', 'BETA',
            'STAGE1_MAX_EPOCHS', 'STAGE1_PATIENCE', 'STAGE2_EPOCHS',
            'R_VALUES', 'SEEDS',
            'ROLLOUT_STEPS', 'ROLLOUT_REPEATS',
            'DATA_DIR', 'CHECKPOINT_DIR', 'RESULTS_DIR',
        ]
        for attr in required_attrs:
            assert hasattr(cfg, attr), f"Config missing attribute: {attr}"

    def test_r_values(self):
        """R_VALUES should be the expected sweep values."""
        expected = [0.0, 0.20, 0.30, 0.40, 0.50, 1.0]
        assert cfg.R_VALUES == expected

    def test_seeds(self):
        """SEEDS should be the full 15-seed sweep."""
        assert cfg.SEEDS == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


class TestParameterCount:
    """Verify total parameter count stays under budget."""

    def test_under_2m(self):
        """V+M combined should have fewer than 2 million parameters."""
        vae = VAE()
        gru = GRUPredictor()
        total = sum(p.numel() for p in vae.parameters())
        total += sum(p.numel() for p in gru.parameters())
        assert total < 2_000_000, (
            f"Total params {total:,} exceeds 2M budget"
        )
        # Also verify it's in the expected ballpark (~573K)
        assert total > 500_000, (
            f"Total params {total:,} unexpectedly low — check architecture"
        )


class TestProbeInterface:
    """Test that probe functions have the expected interface."""

    def test_probe_returns_valid_accuracy(self):
        """
        Quick smoke test: fit a probe on synthetic latent vectors.
        Should return accuracy in [0, 1].
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        # Synthetic well-separated data (should give high accuracy)
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(50, cfg.LATENT_DIM) + 2,
            np.random.randn(50, cfg.LATENT_DIM) - 2,
        ])
        y = np.array([0] * 50 + [1] * 50)

        probe = LogisticRegression(max_iter=cfg.PROBE_MAX_ITER, solver='lbfgs')
        scores = cross_val_score(probe, X, y, cv=3, scoring='accuracy')
        accuracy = float(np.mean(scores))

        assert 0.0 <= accuracy <= 1.0, f"Accuracy {accuracy} out of range"
