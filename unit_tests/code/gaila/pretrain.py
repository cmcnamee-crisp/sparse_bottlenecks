"""Trains a probe/model at every layer.

Reports accuracies across the different classes. This esp. useful for the probing models
where some classes are entirely heldout.
"""
import torch as th

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger

import pandas as pd
import os
import time

import helpers

from models import cnn, resnet

import evaluate
import flags

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("times", exist_ok=True)
os.makedirs("results", exist_ok=True)


def main(FLAGS):
    seed_everything(FLAGS.seed)
    gpus = 1 if th.cuda.is_available() else 0
    device = "cuda" if th.cuda.is_available() else "cpu"
    if helpers.invalid_flags(FLAGS):
        exit()
    assert FLAGS.pretrain_model in {"RawCnn", "RawRes"}
    datadesc, dataspec = helpers.load_info(FLAGS)
    num_classes = len(datadesc.classlabel.unique())

    datamodule = helpers.get_image_datamodule(
        FLAGS.data_path, datadesc, dataspec, FLAGS.batch_size, device,
    )
    if FLAGS.pretrain_model == "RawCnn":
        model = cnn.CNN(num_classes=num_classes)
    elif FLAGS.pretrain_model == "RawRes":
        model = resnet.ResNetWrapper(num_classes=num_classes)
    else:
        assert False
    pretrain_id = flags.pretrain_id(FLAGS)
    os.makedirs(f"results/{FLAGS.jobname}", exist_ok=True)
    
    # Non-invasive WandB logging hook
    wandb_logger = WandbLogger(project="sparse-bottlenecks", name=pretrain_id, log_model=False)
    
    trainer = pl.Trainer(
        accelerator="gpu" if gpus > 0 else "cpu",
        devices=gpus if gpus > 0 else "auto",
        max_epochs=FLAGS.n_epochs,
        min_epochs=FLAGS.n_epochs,
        deterministic=True,  # reproducibility
        logger=wandb_logger, # Injects wandb into lightning loop
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.save_checkpoint(f"checkpoints/{pretrain_id}.ckpt")

    pd.DataFrame(
        evaluate.get_predictions(
            FLAGS, dataspec, datamodule, trainer, model, num_classes, dataspec,
        )
    ).to_csv(
        f"results/{FLAGS.jobname}/{pretrain_id}.tsv", sep="\t", index=False,
    )


if __name__ == "__main__":
    tick = time.time()
    FLAGS = flags.get_flags().parse_args()
    main(flags.get_flags().parse_args())
    tock = time.time()
    train_id = flags.train_id(FLAGS)
    pd.DataFrame([{"id": train_id, "script": "ft", "seconds": tock - tick}]).to_csv(
        f"times/{train_id}.tsv", index=False, sep="\t"
    )
