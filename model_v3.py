"""
Model definitions for the v3 continual learning dominance-shift experiment.

Scientific rationale
--------------------
The V model (VAE) learns a compressed latent representation of visual input,
while the M model (GRU) learns the statistical structure of the latent space
via next-step prediction. Together they form a "world model" (Ha &
Schmidhuber, "World Models," 2018) capable of generating imagined experience
through free-running rollout — the GRU consuming its own predictions with
no real sensory input.

The key scientific advance over v2 is that convergence behavior (whether the
three proxy metrics stabilize at meaningful values) is emergent from gradient
descent rather than guaranteed by the functional form of hand-designed ODEs.
The model can genuinely fail to converge, making positive convergence results
scientifically meaningful.

Architecture summary
--------------------
  V (VAE):  Conv encoder → latent (dim 32) → Conv decoder.  ~342K params.
  M (GRU):  GRU(32→256) + FC head(256→32).                  ~231K params.
  Total:    ~573K params, well under the 2M ceiling for Colab free T4.

Reconstruction loss
-------------------
Binary cross-entropy (BCE) is used rather than MSE because MNIST and
FashionMNIST pixel values are concentrated near 0 and 1, making the
Bernoulli likelihood a better probabilistic fit than a Gaussian decoder.
BCE also tends to produce sharper reconstructions on this type of data.
See Kingma & Welling (2013), "Auto-Encoding Variational Bayes," for the
standard treatment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config_v3 as cfg


# ═══════════════════════════════════════════════════════════════════════
# V Model — Variational Autoencoder
# ═══════════════════════════════════════════════════════════════════════

class VAE(nn.Module):
    """
    Convolutional VAE for 28×28 grayscale images.

    Encoder: Conv(1→32) → Conv(32→64) → Flatten → FC to (mu, logvar).
    Decoder: FC → Reshape → ConvTranspose(64→32) → ConvTranspose(32→1).
    Latent dimension is controlled by config_v3.LATENT_DIM (default 32).
    """

    def __init__(self, latent_dim=None):
        super().__init__()
        self.latent_dim = latent_dim or cfg.LATENT_DIM

        # ── Encoder ──
        self.enc_conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        # After two stride-2 convolutions on 28×28: 28→14→7, so 64×7×7 = 3136
        self.enc_fc_mu = nn.Linear(64 * 7 * 7, self.latent_dim)
        self.enc_fc_logvar = nn.Linear(64 * 7 * 7, self.latent_dim)

        # ── Decoder ──
        self.dec_fc = nn.Linear(self.latent_dim, 64 * 7 * 7)
        self.dec_conv1 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.dec_conv2 = nn.ConvTranspose2d(
            32, 1, kernel_size=3, stride=2, padding=1, output_padding=1
        )

    def encode(self, x):
        """Encode images to latent parameters (mu, logvar)."""
        h = F.relu(self.enc_conv1(x))     # (B, 32, 14, 14)
        h = F.relu(self.enc_conv2(h))     # (B, 64, 7, 7)
        h = h.view(h.size(0), -1)         # (B, 3136)
        return self.enc_fc_mu(h), self.enc_fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + std * eps."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        """Decode latent vectors back to image space."""
        h = F.relu(self.dec_fc(z))                 # (B, 3136)
        h = h.view(h.size(0), 64, 7, 7)           # (B, 64, 7, 7)
        h = F.relu(self.dec_conv1(h))              # (B, 32, 14, 14)
        return torch.sigmoid(self.dec_conv2(h))    # (B, 1, 28, 28)

    def forward(self, x):
        """Full forward pass: encode → reparameterize → decode."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# ═══════════════════════════════════════════════════════════════════════
# M Model — GRU Next-Step Predictor
# ═══════════════════════════════════════════════════════════════════════

class GRUPredictor(nn.Module):
    """
    GRU-based next-step predictor in latent space.

    Given a sequence of latent vectors z_1, ..., z_{T-1}, predicts
    z_2, ..., z_T (teacher-forced during training). During free-running
    rollout, the model consumes its own predictions autoregressively.

    The GRU learns the statistical distribution of the latent space rather
    than real temporal dynamics (MNIST has no temporal structure). In
    free-running mode, the generated latents reflect the training
    distribution — which is exactly what dmn_proxy measures.
    """

    def __init__(self, input_dim=None, hidden_dim=None, num_layers=None):
        super().__init__()
        self.input_dim = input_dim or cfg.LATENT_DIM
        self.hidden_dim = hidden_dim or cfg.GRU_HIDDEN
        self.num_layers = num_layers or cfg.GRU_LAYERS

        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.output_head = nn.Linear(self.hidden_dim, self.input_dim)

    def forward(self, z_seq, hidden=None):
        """
        Forward pass over a sequence of latent vectors.

        Args:
            z_seq:  (batch, seq_len, input_dim) — sequence of latent vectors.
            hidden: Optional initial hidden state.

        Returns:
            z_pred: (batch, seq_len, input_dim) — predicted next latents.
            hidden: Updated hidden state.
        """
        gru_out, hidden = self.gru(z_seq, hidden)   # (batch, seq_len, hidden_dim)
        z_pred = self.output_head(gru_out)           # (batch, seq_len, input_dim)
        return z_pred, hidden

    def free_running_rollout(self, steps, vae_decoder, seed=None,
                             start_latent=None, device='cpu'):
        """
        Generate a sequence of decoded images by running the GRU
        autoregressively — each step's output becomes the next step's
        input, with no real image data after step 0.

        This is the core method for measuring dmn_proxy: the generated
        images reveal which domain the model's "imagination" defaults to
        when left to run freely.

        Args:
            steps:         Number of rollout steps (default 50).
            vae_decoder:   VAE.decode method (or the VAE module itself).
            seed:          Random seed for the starting latent (if start_latent
                           is None). Each rollout should use a different seed
                           for averaging purposes.
            start_latent:  Optional (latent_dim,) tensor. If provided, used
                           as the initial input instead of a random vector.
            device:        'cpu' or 'cuda'.

        Returns:
            decoded_images: List of `steps` tensors, each (1, 1, 28, 28).
        """
        self.eval()
        decode_fn = vae_decoder.decode if hasattr(vae_decoder, 'decode') else vae_decoder

        with torch.no_grad():
            if start_latent is not None:
                z = start_latent.unsqueeze(0).to(device)   # (1, latent_dim)
            else:
                if seed is not None:
                    torch.manual_seed(seed)
                z = torch.randn(1, self.input_dim, device=device)

            hidden = torch.zeros(
                self.num_layers, 1, self.hidden_dim, device=device
            )

            decoded_images = []
            for _ in range(steps):
                z_input = z.unsqueeze(1)                   # (1, 1, input_dim)
                gru_out, hidden = self.gru(z_input, hidden)
                z = self.output_head(gru_out.squeeze(1))   # (1, input_dim)
                img = decode_fn(z)                         # (1, 1, 28, 28)
                decoded_images.append(img.cpu())

        return decoded_images


# ═══════════════════════════════════════════════════════════════════════
# Loss Functions
# ═══════════════════════════════════════════════════════════════════════

def vae_loss_fn(recon, target, mu, logvar, beta=None):
    """
    VAE loss = reconstruction + beta * KL divergence.

    Reconstruction uses binary cross-entropy (BCE), summed over pixels and
    averaged over the batch. BCE is chosen because MNIST/FashionMNIST pixels
    are near-binary (concentrated at 0 and 1), making the Bernoulli
    likelihood a better probabilistic fit than a Gaussian (MSE). This is
    the standard choice for VAEs on this data (Kingma & Welling, 2013).

    KL divergence is computed in closed form for the Gaussian posterior
    q(z|x) = N(mu, diag(exp(logvar))) against the standard normal prior
    p(z) = N(0, I). Summed over latent dimensions, averaged over batch.

    Returns:
        total_loss, recon_loss, kl_loss (all scalar tensors).
    """
    if beta is None:
        beta = cfg.BETA

    batch_size = target.size(0)

    # Reconstruction: sum over pixels (C×H×W), average over batch
    recon_loss = F.binary_cross_entropy(
        recon, target, reduction='sum'
    ) / batch_size

    # KL: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    ) / batch_size

    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss


def gru_loss_fn(z_pred, z_target):
    """
    GRU next-step prediction loss: MSE between predicted and actual
    next-step latent vectors.
    """
    return F.mse_loss(z_pred, z_target)
