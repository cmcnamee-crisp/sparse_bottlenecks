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

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

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
    # Latents are saved as (N, C*H*W) flattened vectors by _format_layer.
    img = helpers._get_vector_layer(dataset_path, device, layer)
    
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
        if HAS_WANDB:
            wandb.log({
                "epoch": epoch + 1,
                "loss": total_loss / n_batches,
                "mse": total_mse / n_batches,
                "reg": total_l1 / n_batches,
            })

    return sae, latent_dim


# ---------------------------------------------------------------------------
# Integrated training
# ---------------------------------------------------------------------------

def train_integrated(FLAGS, device: str):
    """Train SAE inserted into backbone forward pass end-to-end.

    The backbone and SAE are trained jointly. Loss: CrossEntropy + SAE reg.
    No reconstruction term: the classification objective alone shapes the
    sparse bottleneck.
    """
    print(f"[integrated] Initializing new backbone model ...")
    from models import cnn
    
    FLAGS.dataspec = "classes"
    datadesc, dataspec = helpers.load_info(FLAGS)
    num_classes = len(datadesc.classlabel.unique())
    
    backbone = cnn.CNN(num_classes=num_classes).to(device)
    backbone.train()

    # Cache the preprocessed image dataset so we only load PNGs once per seed.
    cache_dir = "data/cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = f"{cache_dir}/integrated_{FLAGS.data_path}_{FLAGS.seed}.pt"

    if os.path.exists(cache_path):
        print(f"[integrated] Loading cached dataset from {cache_path}")
        cached = th.load(cache_path)
        train_images = cached["train_images"]
        train_labels = cached["train_labels"]
    else:
        print(f"[integrated] Loading image dataset (will cache for reuse) ...")
        datamodule = helpers.get_image_datamodule(
            FLAGS.data_path, datadesc, dataspec, FLAGS.batch_size, device,
        )
        datamodule.setup(stage="fit")
        # Collect all training data into tensors for caching.
        all_imgs, all_labels = [], []
        for batch in datamodule.train_dataloader():
            imgs, labels = batch
            all_imgs.append(imgs)
            all_labels.append(labels)
        train_images = th.cat(all_imgs)
        train_labels = th.cat(all_labels)
        th.save({"train_images": train_images, "train_labels": train_labels}, cache_path)
        print(f"[integrated] Cached dataset to {cache_path}")

    dataset = th.utils.data.TensorDataset(train_images, train_labels)
    loader = th.utils.data.DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=True)
    
    # We need to compute the latent dimension dynamically.
    dummy_input = th.zeros(1, 3, 224, 224).to(device)
    with th.no_grad():
        dummy_z = backbone.forward_to_layer(dummy_input, FLAGS.sae_layer)
        latent_dim = dummy_z.flatten(1).shape[1]

    print(f"[integrated] Instantiating SAE at layer={FLAGS.sae_layer} with latent dim={latent_dim}")

    sae = build_sae_from_flags(FLAGS, latent_dim).to(device)

    optimizer = optim.Adam(
        list(sae.parameters()) + list(backbone.parameters()), lr=FLAGS.sae_lr
    )

    for epoch in range(FLAGS.n_epochs):
        sae.train()
        backbone.train()
        total_loss = 0.0
        total_xe = 0.0
        total_reg = 0.0
        total_correct = 0
        total_n = 0

        for batch in loader:
            x_batch, y_batch = batch
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            
            # Forward path (all on GPU, gradients flow end-to-end):
            # 1. Image -> Backbone up to sae_layer
            z = backbone.forward_to_layer(x_batch, FLAGS.sae_layer)
            # Flatten for SAE (conv layers are 4D; fc is already 2D).
            z_flat = z.flatten(1)
            
            # 2. Latent -> SAE Bottleneck -> Reconstructed Latent (z_hat)
            z_hat, activations, loss_dict = sae(z_flat.float())
            
            # 3. Reconstructed Latent -> Rest of Backbone -> Logits
            logits = backbone.forward_from_layer(z_hat, FLAGS.sae_layer)
            
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
        if HAS_WANDB:
            wandb.log({
                "epoch": epoch + 1,
                "loss": total_loss / n_batches,
                "xe": total_xe / n_batches,
                "reg": total_reg / n_batches,
                "train_acc": acc,
            })

    return sae, backbone, latent_dim


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(path: str, sae, latent_dim: int, FLAGS, backbone_model=None):
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
    if backbone_model is not None:
        payload["backbone_state_dict"] = backbone_model.state_dict()
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

    backbone_model = None
    if "backbone_state_dict" in payload:
        from models import cnn
        # Infer num_classes from the saved classifier weight shape.
        num_classes = payload["backbone_state_dict"]["ff_head.weight"].shape[0]
        backbone_model = cnn.CNN(num_classes=num_classes)
        backbone_model.load_state_dict(payload["backbone_state_dict"])
        backbone_model.to(device)
        backbone_model.eval()

    return sae, backbone_model, payload


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

    # Initialize wandb for cluster monitoring.
    if HAS_WANDB:
        wandb.init(
            project="sae-unit-tests",
            name=sid,
            config=vars(FLAGS),
            reinit=True,
        )

    if FLAGS.sae_mode == "post_hoc":
        sae, latent_dim = train_post_hoc(FLAGS, device)
        save_checkpoint(ckpt_path, sae, latent_dim, FLAGS, backbone_model=None)
    elif FLAGS.sae_mode == "integrated":
        sae, backbone, latent_dim = train_integrated(FLAGS, device)
        save_checkpoint(ckpt_path, sae, latent_dim, FLAGS, backbone_model=backbone)

    if HAS_WANDB:
        wandb.finish()


if __name__ == "__main__":
    tick = time.time()
    FLAGS = get_sae_flags().parse_args()
    main(FLAGS)
    tock = time.time()
    sid = sae_id(FLAGS)
    pd.DataFrame([{"id": sid, "script": "sae_train", "seconds": tock - tick}]).to_csv(
        f"times/{sid}.tsv", index=False, sep="\t"
    )
