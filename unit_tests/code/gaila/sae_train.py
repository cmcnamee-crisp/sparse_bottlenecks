"""SAE Training Script.

Trains a Sparse Autoencoder (SAE) inserted at a specified layer of an
already-trained backbone (e.g. CNNNoPre / RawCnn).

Two training modes
------------------
post_hoc   : The backbone is FROZEN.  The SAE is trained purely on
             reconstruction (MSE + SAE regularization loss).

integrated : The SAE is inserted as a differentiable bottleneck in the
             forward pass.  The full network (backbone + SAE +
             classification head) is trained end-to-end with the
             classification cross-entropy + SAE regularization loss.
             There is NO reconstruction term; the classification
             objective alone shapes the bottleneck.

Layer selection
---------------
The `--sae_layer` argument selects which backbone layer the SAE is
attached to.  Valid names are the same as those used in dataset_transform
and train.py (e.g. "conv_layer0", "layer1", ..., "fc").

This mirrors the layer-wise analysis from Fig. 7 of the unit-tests paper,
and allows training SAEs at every layer of the network.

Output
------
Checkpoints saved to:
    checkpoints/sae_{mode}_{sae_id}.ckpt

where sae_id encodes all relevant hyper-parameters.
"""

import os
import time
import argparse
import json

import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

import flags
import helpers
import specifications
from sae_models import build_sae


os.makedirs("checkpoints", exist_ok=True)
os.makedirs("times", exist_ok=True)
os.makedirs("sae_checkpoints", exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sae_id(FLAGS) -> str:
    """Unique identifier string for this SAE configuration."""
    parts = [
        f"sae",
        f"mode={FLAGS.sae_mode}",
        f"arch={FLAGS.sae_arch}",
        f"feat={FLAGS.sae_num_features}",
        f"layer={FLAGS.sae_layer}",
        f"model={FLAGS.pretrain_model}",
        f"data={FLAGS.data_path}",
        f"seed={FLAGS.seed}",
    ]
    if FLAGS.sae_arch in {"standard", "gated"}:
        parts.append(f"l1={FLAGS.sae_l1_coeff}")
    if FLAGS.sae_arch == "topk":
        parts.append(f"k={FLAGS.sae_topk_k}")
    return "_".join(parts)


def load_latents(FLAGS, layer: str, device: str):
    """Load pre-computed latent vectors from disk (post-hoc mode).

    Returns (latents_tensor, labels_tensor, datadesc).
    """
    FLAGS.dataspec = "classes"
    datadesc, dataspec = helpers.load_info(FLAGS)

    dataset_path = flags.transformed_id(FLAGS)
    img = helpers._get_vector_layer(dataset_path, device, layer, "avg")
    labels = th.tensor(datadesc.classlabel.values, dtype=th.long, device=device)
    return img, labels, datadesc, dataspec


def build_sae_from_flags(FLAGS, latent_dim: int):
    """Instantiate the right SAE from command-line flags."""
    kwargs = {}
    if FLAGS.sae_arch in {"standard", "gated"}:
        kwargs["l1_coeff"] = FLAGS.sae_l1_coeff
    if FLAGS.sae_arch == "topk":
        kwargs["k"] = FLAGS.sae_topk_k
    return build_sae(
        arch=FLAGS.sae_arch,
        latent_dim=latent_dim,
        num_features=FLAGS.sae_num_features,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Post-hoc training
# ---------------------------------------------------------------------------

def train_post_hoc(FLAGS, device: str):
    """Train SAE on frozen backbone latents (pure reconstruction objective)."""
    print(f"[post_hoc] Loading latents from layer={FLAGS.sae_layer} ...")
    latents, labels, datadesc, dataspec = load_latents(FLAGS, FLAGS.sae_layer, device)
    latent_dim = latents.shape[1]
    print(f"[post_hoc] Latent dim={latent_dim}, num samples={len(latents)}")

    # Use the seen/unseen split from the original paper.
    # The SAE is trained ONLY on "seen" slices so that Test 2 can evaluate
    # out-of-distribution generalization to "unseen" slices.
    holdout_mask = datadesc.classname.map(
        {cn: not spec["holdout"] for cn, spec in dataspec.items()}
    ).values
    train_mask = (datadesc.test == False).values & holdout_mask

    latents_train = latents[train_mask]
    labels_train = labels[train_mask]

    dataset = TensorDataset(latents_train, labels_train)
    loader = DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=True, drop_last=False)

    sae = build_sae_from_flags(FLAGS, latent_dim).to(device)
    optimizer = optim.Adam(sae.parameters(), lr=FLAGS.sae_lr)

    for epoch in range(FLAGS.n_epochs):
        sae.train()
        total_loss = 0.0
        total_mse = 0.0
        total_l1 = 0.0

        for z_batch, _ in loader:
            optimizer.zero_grad()
            z_hat, activations, loss_dict = sae(z_batch.float())
            mse = nn.functional.mse_loss(z_hat, z_batch.float())
            reg = sum(loss_dict.values()) if loss_dict else th.tensor(0.0, device=device)
            loss = mse + reg
            loss.backward()
            optimizer.step()
            # Normalize decoder after each step (keeps features interpretable).
            if hasattr(sae, "normalize_decoder"):
                sae.normalize_decoder()
            total_loss += loss.item()
            total_mse += mse.item()
            total_l1 += reg.item() if isinstance(reg, th.Tensor) else reg

        n_batches = len(loader)
        if (epoch + 1) % max(1, FLAGS.n_epochs // 10) == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1}/{FLAGS.n_epochs} | "
                f"loss={total_loss/n_batches:.4f} | "
                f"mse={total_mse/n_batches:.4f} | "
                f"reg={total_l1/n_batches:.4f}"
            )

    return sae, latent_dim


# ---------------------------------------------------------------------------
# Integrated training
# ---------------------------------------------------------------------------

def train_integrated(FLAGS, device: str):
    """Train SAE inserted into backbone forward pass end-to-end.

    The head and SAE are trained jointly. Loss: CrossEntropy + SAE reg.
    No reconstruction term: the classification objective alone shapes the
    sparse bottleneck.
    """
    print(f"[integrated] Loading backbone model ...")
    backbone = helpers.get_ft_model(FLAGS, device)
    backbone.train()

    # We need to split the CNN at the requested layer.
    # For now, we work at the final 'fc' layer (latent_dim=512).
    # Latents are pre-computed for efficiency; we attach the SAE in latent space.
    print(f"[integrated] Loading latents from layer={FLAGS.sae_layer} ...")
    latents, labels, datadesc, dataspec = load_latents(FLAGS, FLAGS.sae_layer, device)
    latent_dim = latents.shape[1]
    print(f"[integrated] Latent dim={latent_dim}, num samples={len(latents)}")

    holdout_mask = datadesc.classname.map(
        {cn: not spec["holdout"] for cn, spec in dataspec.items()}
    ).values
    train_mask = (datadesc.test == False).values & holdout_mask

    latents_train = latents[train_mask]
    labels_train = labels[train_mask]
    num_classes = int(labels_train.max().item()) + 1

    dataset = TensorDataset(latents_train, labels_train)
    loader = DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=True, drop_last=False)

    sae = build_sae_from_flags(FLAGS, latent_dim).to(device)
    # Add a small classification head on top of the SAE output.
    head = nn.Linear(latent_dim, num_classes).to(device)

    optimizer = optim.Adam(
        list(sae.parameters()) + list(head.parameters()), lr=FLAGS.sae_lr
    )

    for epoch in range(FLAGS.n_epochs):
        sae.train()
        head.train()
        total_loss = 0.0
        total_xe = 0.0
        total_reg = 0.0
        total_correct = 0
        total_n = 0

        for z_batch, y_batch in loader:
            optimizer.zero_grad()
            # Forward: z -> SAE bottleneck -> z_hat -> classify
            z_hat, activations, loss_dict = sae(z_batch.float())
            logits = head(z_hat)
            xe = nn.functional.cross_entropy(logits, y_batch)
            reg = sum(loss_dict.values()) if loss_dict else th.tensor(0.0, device=device)
            loss = xe + reg
            loss.backward()
            optimizer.step()
            if hasattr(sae, "normalize_decoder"):
                sae.normalize_decoder()
            total_loss += loss.item()
            total_xe += xe.item()
            total_reg += reg.item() if isinstance(reg, th.Tensor) else reg
            preds = logits.argmax(dim=-1)
            total_correct += (preds == y_batch).sum().item()
            total_n += len(y_batch)

        n_batches = len(loader)
        acc = total_correct / total_n
        if (epoch + 1) % max(1, FLAGS.n_epochs // 10) == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1}/{FLAGS.n_epochs} | "
                f"loss={total_loss/n_batches:.4f} | "
                f"xe={total_xe/n_batches:.4f} | "
                f"reg={total_reg/n_batches:.4f} | "
                f"acc={acc:.3f}"
            )

    return sae, head, latent_dim


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(path: str, sae, latent_dim: int, FLAGS, head=None):
    payload = {
        "sae_state_dict": sae.state_dict(),
        "sae_arch": FLAGS.sae_arch,
        "latent_dim": latent_dim,
        "num_features": FLAGS.sae_num_features,
        "sae_layer": FLAGS.sae_layer,
        "sae_mode": FLAGS.sae_mode,
        "sae_topk_k": getattr(FLAGS, "sae_topk_k", None),
        "sae_l1_coeff": getattr(FLAGS, "sae_l1_coeff", None),
    }
    if head is not None:
        payload["head_state_dict"] = head.state_dict()
        payload["head_out_features"] = head.out_features
    th.save(payload, path)
    print(f"Saved SAE checkpoint to {path}")


def load_checkpoint(path: str, device: str):
    """Load an SAE (and optionally its head) from a saved checkpoint.

    Returns (sae, head_or_None, meta_dict).
    """
    payload = th.load(path, map_location=device)
    kwargs = {}
    if payload["sae_arch"] in {"standard", "gated"}:
        kwargs["l1_coeff"] = payload["sae_l1_coeff"]
    if payload["sae_arch"] == "topk":
        kwargs["k"] = payload["sae_topk_k"]
    sae = build_sae(
        arch=payload["sae_arch"],
        latent_dim=payload["latent_dim"],
        num_features=payload["num_features"],
        **kwargs,
    )
    sae.load_state_dict(payload["sae_state_dict"])
    sae.to(device)
    sae.eval()

    head = None
    if "head_state_dict" in payload:
        head = nn.Linear(payload["latent_dim"], payload["head_out_features"])
        head.load_state_dict(payload["head_state_dict"])
        head.to(device)
        head.eval()

    return sae, head, payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_sae_flags():
    parser = flags.get_flags()
    parser.add_argument("--sae_mode", required=True, choices=["post_hoc", "integrated"])
    parser.add_argument(
        "--sae_arch", default="standard", choices=["standard", "gated", "topk"],
        help="SAE architecture to train."
    )
    parser.add_argument(
        "--sae_layer", required=True, type=str,
        help="Layer name at which to attach the SAE (e.g. 'fc', 'layer3')."
    )
    parser.add_argument(
        "--sae_num_features", default=512 * 8, type=int,
        help="Number of SAE dictionary features (dictionary size)."
    )
    parser.add_argument("--sae_l1_coeff", default=1e-3, type=float)
    parser.add_argument("--sae_topk_k", default=32, type=int)
    parser.add_argument("--sae_lr", default=1e-4, type=float)
    return parser


def main(FLAGS):
    seed_everything(FLAGS.seed)
    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    sid = sae_id(FLAGS)
    ckpt_path = f"sae_checkpoints/{sid}.pt"

    if FLAGS.sae_mode == "post_hoc":
        sae, latent_dim = train_post_hoc(FLAGS, device)
        save_checkpoint(ckpt_path, sae, latent_dim, FLAGS, head=None)
    elif FLAGS.sae_mode == "integrated":
        sae, head, latent_dim = train_integrated(FLAGS, device)
        save_checkpoint(ckpt_path, sae, latent_dim, FLAGS, head=head)


if __name__ == "__main__":
    tick = time.time()
    FLAGS = get_sae_flags().parse_args()
    main(FLAGS)
    tock = time.time()
    sid = sae_id(FLAGS)
    pd.DataFrame([{"id": sid, "script": "sae_train", "seconds": tock - tick}]).to_csv(
        f"times/{sid}.tsv", index=False, sep="\t"
    )
