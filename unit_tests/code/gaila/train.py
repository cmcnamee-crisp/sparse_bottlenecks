"""Trains a probe/model at every layer.

This is used both for training as well as t2.

Reports accuracies across the different classes. This esp. useful for the probing models
where some classes are entirely heldout.
"""
import torch as th

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

import pandas as pd
import os
import time
import tqdm

import helpers

from models import linear

import evaluate
import flags

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("times", exist_ok=True)


def main(FLAGS):
    seed_everything(FLAGS.seed)
    gpus = 1 if th.cuda.is_available() else 0
    device = "cuda" if th.cuda.is_available() else "cpu"

    out = []
    for dataspec in tqdm.tqdm(
        ["shape", "layout", "stroke", "classes"], desc="Dataspecs"
    ):
        FLAGS.dataspec = dataspec
        if helpers.invalid_flags(FLAGS):
            continue
        datadesc, dataspec = helpers.load_info(FLAGS)
        num_classes = len(datadesc.classlabel.unique())

        for layer in tqdm.tqdm(flags.get_layers(FLAGS), desc="Layers"):
            datamodule, num_features = helpers.get_datamodule_layer(
                flags.transformed_id(FLAGS),
                datadesc,
                dataspec,
                FLAGS.batch_size,
                device,
                layer=layer,
                flatten="avg",
            )
            model = linear.Linear(num_features=num_features, num_output=num_classes)
            train_id = flags.train_layerwise_id(FLAGS, layer)
            trainer = pl.Trainer(
                accelerator="gpu" if gpus > 0 else "cpu",
                devices=gpus if gpus > 0 else "auto",
                max_epochs=FLAGS.n_epochs,
                min_epochs=FLAGS.n_epochs,
                deterministic=True,  # reproducibility
            )
            trainer.fit(model, datamodule=datamodule)
            trainer.save_checkpoint(f"checkpoints/{train_id}.ckpt")
            out.extend(
                evaluate.get_predictions(
                    FLAGS,
                    dataspec,
                    datamodule,
                    trainer,
                    model,
                    num_classes,
                    dataspec,
                    {"layer": layer, "num_classes": num_classes,},
                )
            )
    pd.DataFrame(out).to_csv(
        f"results/{FLAGS.jobname}/{train_id}.tsv", sep="\t", index=False,
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
