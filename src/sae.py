"""
Sparse Autoencoder (SAE) — BatchTopK architecture.

Implemented in PyTorch for efficient GPU training.
Activation inputs come from EpiBERT (collected as numpy, converted to torch).

Architecture:
  - Encoder: Linear(d → expansion*d) + TopK sparsity (per batch)
  - Decoder: Linear(expansion*d → d, no bias) + unit-norm columns
  - Tied init: decoder = encoder.T, renormalized each step
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import get_logger, cfg, sae_path

log = get_logger("sae")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# BatchTopK SAE
# ─────────────────────────────────────────────────────────────────────────────

class BatchTopKSAE(nn.Module):
    """
    Sparse Autoencoder with BatchTopK sparsity constraint.

    BatchTopK: activates the top-k features across the entire batch (not per sample).
    This encourages global feature usage and reduces dead features.

    Parameters
    ----------
    d_input   : int  input dimension (EpiBERT hidden dim, typically 1024)
    expansion : int  latent_dim = expansion × d_input
    k         : int  number of active features per sample (TopK target)
    """

    def __init__(self, d_input, expansion=8, k=64):
        super().__init__()
        self.d_input   = d_input
        self.d_latent  = d_input * expansion
        self.k         = k
        self.expansion = expansion

        # Encoder: bias + weight
        self.encoder_bias = nn.Parameter(torch.zeros(self.d_latent))
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(d_input, self.d_latent)
            )
        )

        # Decoder (no bias; columns are unit-norm)
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.d_latent, d_input)
            )
        )
        self._normalize_decoder()

        # Tied init: decoder = encoder.T
        with torch.no_grad():
            self.W_dec.data = self.W_enc.data.T.clone()
            self._normalize_decoder()

        # Dead feature tracking: running count of steps since last activation
        self.register_buffer('steps_since_active',
                             torch.zeros(self.d_latent, dtype=torch.long))

    @torch.no_grad()
    def _normalize_decoder(self):
        """Unit-normalize decoder columns."""
        norms = self.W_dec.data.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.W_dec.data = self.W_dec.data / norms

    def encode(self, x):
        """
        Encode x with BatchTopK sparsity.
        x: (batch, d_input)
        Returns z: (batch, d_latent) sparse activations
        """
        # Pre-activation
        pre_act = x @ self.W_enc + self.encoder_bias  # (B, d_latent)
        pre_act = F.relu(pre_act)                       # Relu before TopK

        # BatchTopK: select top-k activations across the whole batch
        B = x.shape[0]
        total_k = self.k * B   # total active across batch

        # Flatten to find global top-k
        flat = pre_act.reshape(-1)  # (B * d_latent,)
        if total_k >= flat.shape[0]:
            # All features active — shouldn't happen but guard anyway
            return pre_act

        threshold = torch.topk(flat, total_k, sorted=False).values.min()
        mask = (pre_act >= threshold).float()
        z = pre_act * mask
        return z

    def decode(self, z):
        """
        Decode sparse latents to reconstruction.
        z: (batch, d_latent)
        Returns x_hat: (batch, d_input)
        """
        return z @ self.W_dec

    def forward(self, x):
        """
        Full forward pass.
        Returns (x_hat, z, pre_act) for loss computation.
        """
        pre_act = x @ self.W_enc + self.encoder_bias
        pre_act = F.relu(pre_act)

        B = x.shape[0]
        total_k = self.k * B
        flat = pre_act.reshape(-1)
        if total_k < flat.shape[0]:
            threshold = torch.topk(flat, total_k, sorted=False).values.min()
            mask = (pre_act >= threshold).float()
            z = pre_act * mask
        else:
            z = pre_act

        x_hat = self.decode(z)
        return x_hat, z, pre_act

    @torch.no_grad()
    def update_dead_features(self, z):
        """Track which features were active in this batch."""
        active = (z > 0).any(dim=0)  # (d_latent,) bool
        self.steps_since_active[active] = 0
        self.steps_since_active[~active] += 1

    def get_dead_features(self, threshold_steps):
        """Return mask of dead features (not activated in threshold_steps)."""
        return self.steps_since_active > threshold_steps

    @torch.no_grad()
    def resample_dead_features(self, x_batch, dead_mask, noise_scale=0.2):
        """
        Resample dead encoder/decoder columns using high-loss examples.
        Standard SAE resampling trick.
        """
        n_dead = dead_mask.sum().item()
        if n_dead == 0:
            return 0

        # Find high-loss examples for resampling
        x_hat, z, _ = self.forward(x_batch)
        losses = (x_batch - x_hat).pow(2).sum(dim=1)
        probs = losses / losses.sum()
        sampled_idx = torch.multinomial(probs, n_dead, replacement=True)
        new_directions = x_batch[sampled_idx]  # (n_dead, d_input)

        # Add noise
        new_directions = new_directions + noise_scale * torch.randn_like(new_directions)

        # Normalize
        new_directions = F.normalize(new_directions, dim=1)

        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        self.W_enc.data[:, dead_indices] = new_directions.T
        self.W_dec.data[dead_indices, :]  = new_directions
        self.encoder_bias.data[dead_indices] = 0.0
        self.steps_since_active[dead_indices] = 0

        return n_dead

    def compute_metrics(self, x, x_hat, z):
        """
        Compute QC metrics for this batch.
        Returns dict.
        """
        # Normalized MSE
        norm_mse = ((x - x_hat).pow(2).sum(dim=1) /
                    x.pow(2).sum(dim=1).clamp(min=1e-8)).mean()

        # L0: mean number of active features per sample
        l0 = (z > 0).float().sum(dim=1).mean()

        return {
            'norm_mse': norm_mse.item(),
            'l0':       l0.item(),
            'l2_loss':  F.mse_loss(x_hat, x).item(),
        }

    def save(self, path):
        """Save SAE checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'state_dict': self.state_dict(),
            'd_input': self.d_input,
            'expansion': self.expansion,
            'k': self.k,
            'd_latent': self.d_latent,
        }, path)
        log.info(f"Saved SAE → {path}")

    @classmethod
    def load(cls, path, device=None):
        """Load SAE from checkpoint."""
        if device is None:
            device = DEVICE
        ckpt = torch.load(path, map_location=device, weights_only=False)
        sae = cls(
            d_input=ckpt['d_input'],
            expansion=ckpt['expansion'],
            k=ckpt['k'],
        )
        sae.load_state_dict(ckpt['state_dict'])
        sae = sae.to(device)
        log.info(f"Loaded SAE from {path} (d_input={ckpt['d_input']}, "
                 f"expansion={ckpt['expansion']}, k={ckpt['k']})")
        return sae


# ─────────────────────────────────────────────────────────────────────────────
# SAE loss
# ─────────────────────────────────────────────────────────────────────────────

def sae_loss(x, x_hat, z, l1_coeff=0.0):
    """
    SAE reconstruction loss.
    L = MSE(x, x_hat) + l1_coeff * mean_L1(z)
    (L1 term not used with BatchTopK; included for compatibility.)
    """
    mse = F.mse_loss(x_hat, x)
    l1  = z.abs().mean() if l1_coeff > 0 else torch.tensor(0.0)
    return mse + l1_coeff * l1, mse, l1


# ─────────────────────────────────────────────────────────────────────────────
# Activation dataset for SAE training
# ─────────────────────────────────────────────────────────────────────────────

class ActivationDataset(torch.utils.data.Dataset):
    """
    Dataset of mean-pooled EpiBERT activations for SAE training.
    Loads from .pt files on disk.
    """

    def __init__(self, tensors):
        """
        tensors: list of np.ndarray or torch.Tensor, each (n_i, d)
        All are concatenated along axis 0.
        """
        if isinstance(tensors[0], np.ndarray):
            all_data = np.concatenate(tensors, axis=0)
            self.data = torch.from_numpy(all_data.astype(np.float32))
        else:
            self.data = torch.cat(tensors, dim=0).float()

        log.info(f"ActivationDataset: {self.data.shape} samples loaded")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @classmethod
    def from_paths(cls, paths):
        """Load from list of .pt file paths."""
        from utils import load_activations
        tensors = [load_activations(p).float() for p in paths if os.path.isfile(p)]
        if not tensors:
            raise ValueError(f"No valid .pt files found in {paths}")
        return cls(tensors)
