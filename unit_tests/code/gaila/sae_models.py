"""Sparse Autoencoder (SAE) model definitions.

Each SAE class is a self-contained nn.Module that implements:
  - encode(z) -> activations (sparse)
  - decode(activations) -> z_hat (reconstructed latent)
  - forward(z) -> (z_hat, activations, loss_dict)

The loss_dict contains the SAE's own regularization terms (e.g. L1 sparsity).
This allows swapping architectures to also swap regularization strategies.

Supported architectures:
  - StandardSAE : ReLU encoder, tied/untied decoder, L1 sparsity loss.
  - GatedSAE    : Gated activation (Rajamanoharan et al., 2024), L1 loss.
  - TopKSAE     : Hard top-k activation, no L1 needed.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseSAE(nn.Module):
    """Interface contract every SAE implementation must satisfy."""

    #: Number of input / output features (latent dimension of the backbone).
    latent_dim: int
    #: Number of SAE feature neurons (dictionary size).
    num_features: int

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """Return the sparse activation vector A for latent z."""
        raise NotImplementedError

    def decode(self, a: torch.Tensor) -> torch.Tensor:
        """Return the reconstructed latent z_hat from activation a."""
        raise NotImplementedError

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Return (z_hat, activations, loss_dict).

        loss_dict contains named scalar tensors (e.g. 'l1', 'aux').
        Callers combine these with their own task loss.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Standard SAE  (ReLU encoder, L1 sparsity)
# ---------------------------------------------------------------------------

class StandardSAE(BaseSAE):
    """Vanilla SAE with ReLU sparse activations and L1 regularization.

    Architecture:
        encode : z -> ReLU(W_enc @ (z - b_dec) + b_enc)
        decode : a -> W_dec @ a + b_dec

    The decoder weight matrix is NOT tied to the encoder but is
    unit-norm column-normalised after every gradient step (via
    `normalize_decoder()`).

    Args:
        latent_dim   : Dimension of the backbone latent space.
        num_features : Number of SAE neurons (dictionary size).
        l1_coeff     : Coefficient λ for the L1 sparsity penalty.
    """

    def __init__(
        self,
        latent_dim: int,
        num_features: int,
        l1_coeff: float = 1e-3,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_features = num_features
        self.l1_coeff = l1_coeff

        # Pre-encoder bias (learned; lives in latent space).
        self.b_dec = nn.Parameter(torch.zeros(latent_dim))

        # Encoder
        self.W_enc = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(latent_dim, num_features))
        )
        self.b_enc = nn.Parameter(torch.zeros(num_features))

        # Decoder (untied)
        self.W_dec = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(num_features, latent_dim))
        )

        self.normalize_decoder()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def normalize_decoder(self):
        """Normalize decoder columns to unit norm."""
        norms = self.W_dec.data.norm(dim=1, keepdim=True).clamp(min=1.0)
        self.W_dec.data = self.W_dec.data / norms

    # ------------------------------------------------------------------
    def encode(self, z: torch.Tensor) -> torch.Tensor:
        z_cent = z - self.b_dec
        pre_act = z_cent @ self.W_enc + self.b_enc  # (..., num_features)
        return F.relu(pre_act)

    def decode(self, a: torch.Tensor) -> torch.Tensor:
        return a @ self.W_dec + self.b_dec  # (..., latent_dim)

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        a = self.encode(z)
        z_hat = self.decode(a)
        l1 = self.l1_coeff * a.abs().sum(dim=-1).mean()
        return z_hat, a, {"l1": l1}


# ---------------------------------------------------------------------------
# Gated SAE  (Rajamanoharan et al., 2024)
# ---------------------------------------------------------------------------

class GatedSAE(BaseSAE):
    """Gated SAE with separate magnitude and gate paths.

    The gate path learns *whether* a feature fires; the magnitude path
    learns *how much*.  This improves feature separation with a similar
    L1 budget.

    Architecture:
        gate  : pi_gate = W_gate @ (z - b_dec) + b_gate
        mag   : f_mag   = ReLU(W_mag  @ (z - b_dec) + b_mag)
        act   : a       = f_mag * (pi_gate > 0).float()
        decode: z_hat   = a @ W_dec + b_dec

    Loss extras: L1 on ReLU(pi_gate) for auxiliary gradient flow.
    """

    def __init__(
        self,
        latent_dim: int,
        num_features: int,
        l1_coeff: float = 1e-3,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_features = num_features
        self.l1_coeff = l1_coeff

        self.b_dec = nn.Parameter(torch.zeros(latent_dim))

        self.W_gate = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(latent_dim, num_features))
        )
        self.b_gate = nn.Parameter(torch.zeros(num_features))

        self.W_mag = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(latent_dim, num_features))
        )
        self.b_mag = nn.Parameter(torch.zeros(num_features))

        self.W_dec = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(num_features, latent_dim))
        )

        self.normalize_decoder()

    @torch.no_grad()
    def normalize_decoder(self):
        norms = self.W_dec.data.norm(dim=1, keepdim=True).clamp(min=1.0)
        self.W_dec.data = self.W_dec.data / norms

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        z_cent = z - self.b_dec
        gate_logits = z_cent @ self.W_gate + self.b_gate
        gate = (gate_logits > 0).float()
        mag = F.relu(z_cent @ self.W_mag + self.b_mag)
        return mag * gate

    def decode(self, a: torch.Tensor) -> torch.Tensor:
        return a @ self.W_dec + self.b_dec

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        z_cent = z - self.b_dec
        gate_logits = z_cent @ self.W_gate + self.b_gate
        gate = (gate_logits > 0).float()
        mag = F.relu(z_cent @ self.W_mag + self.b_mag)
        a = mag * gate
        z_hat = self.decode(a)
        # Auxiliary L1 on soft gate for gradient flow (STE-style).
        l1_aux = self.l1_coeff * F.relu(gate_logits).sum(dim=-1).mean()
        l1 = self.l1_coeff * a.abs().sum(dim=-1).mean()
        return z_hat, a, {"l1": l1, "l1_aux": l1_aux}


# ---------------------------------------------------------------------------
# TopK SAE  (hard sparsity, no L1 needed)
# ---------------------------------------------------------------------------

class TopKSAE(BaseSAE):
    """SAE that keeps exactly the top-k activations; no L1 required.

    Args:
        k : Number of features to keep active per sample.
    """

    def __init__(
        self,
        latent_dim: int,
        num_features: int,
        k: int = 32,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_features = num_features
        self.k = k

        self.b_dec = nn.Parameter(torch.zeros(latent_dim))

        self.W_enc = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(latent_dim, num_features))
        )
        self.b_enc = nn.Parameter(torch.zeros(num_features))

        self.W_dec = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(num_features, latent_dim))
        )

        self.normalize_decoder()

    @torch.no_grad()
    def normalize_decoder(self):
        norms = self.W_dec.data.norm(dim=1, keepdim=True).clamp(min=1.0)
        self.W_dec.data = self.W_dec.data / norms

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        z_cent = z - self.b_dec
        pre_act = z_cent @ self.W_enc + self.b_enc  # (..., num_features)
        # Keep top-k; zero the rest.
        topk_vals, topk_idx = pre_act.topk(self.k, dim=-1)
        a = torch.zeros_like(pre_act)
        a.scatter_(-1, topk_idx, F.relu(topk_vals))
        return a

    def decode(self, a: torch.Tensor) -> torch.Tensor:
        return a @ self.W_dec + self.b_dec

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        a = self.encode(z)
        z_hat = self.decode(a)
        # No explicit L1; sparsity is controlled by k.
        return z_hat, a, {}


# ---------------------------------------------------------------------------
# Registry helper
# ---------------------------------------------------------------------------

SAE_REGISTRY: Dict[str, type] = {
    "standard": StandardSAE,
    "gated": GatedSAE,
    "topk": TopKSAE,
}


def build_sae(
    arch: str,
    latent_dim: int,
    num_features: int,
    **kwargs,
) -> BaseSAE:
    """Instantiate an SAE by name.

    Args:
        arch        : Architecture name (one of ``SAE_REGISTRY`` keys).
        latent_dim  : Latent dimension of the backbone.
        num_features: SAE dictionary size.
        **kwargs    : Architecture-specific kwargs (e.g. l1_coeff, k).
    """
    if arch not in SAE_REGISTRY:
        raise ValueError(
            f"Unknown SAE architecture '{arch}'. "
            f"Choose from: {list(SAE_REGISTRY)}"
        )
    return SAE_REGISTRY[arch](latent_dim=latent_dim, num_features=num_features, **kwargs)
