"""T1 a probe/model at every layer.

Reports accuracies across the different classes. This esp. useful for the probing models
where some classes are entirely heldout.
"""
import pytorch_lightning as pl
import torch as th
from pytorch_lightning import seed_everything

import pandas as pd
import os
import time

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
    data_path_train = FLAGS.data_path
    if "color" in data_path_train:
        data_path_t1 = "t1colors"
    else:
        data_path_t1 = "t1"
    out = []
    FLAGS.data_path = data_path_t1
    # Initialize the missing dataspec flag before load_info uses it
    if not hasattr(FLAGS, "dataspec") or FLAGS.dataspec is None:
        FLAGS.dataspec = "classes"
    dataset_path = flags.transformed_id(FLAGS, data_path_train)
    if helpers.invalid_flags(FLAGS):
        exit()
    datadesc, dataspec = helpers.load_info(FLAGS)
    num_classes = len(datadesc.classlabel.unique())
    layer = flags.get_layers(FLAGS)[-1]
    datamodule, _ = helpers.get_datamodule_layer(
        dataset_path,
        datadesc,
        dataspec,
        FLAGS.batch_size,
        device,
        layer=layer,
        flatten="avg",
    )
    FLAGS.data_path = data_path_train
    train_id = flags.train_layerwise_id(FLAGS, layer)
    trainer = pl.Trainer(
        accelerator="gpu" if gpus > 0 else "cpu",
        devices=gpus if gpus > 0 else "auto",
        max_epochs=FLAGS.n_epochs,
        min_epochs=FLAGS.n_epochs,
        deterministic=True,  # reproducibility
    )
    model = linear.Linear.load_from_checkpoint(
        checkpoint_path=f"checkpoints/{train_id}.ckpt"
    )
    out.extend(
        evaluate.get_predictions(
            FLAGS,
            dataspec,
            datamodule,
            trainer,
            model,
            num_classes,
            dataspec,
            info={"layer": layer},
        )
    )
    t1_id = flags.t1_id(FLAGS)
    pd.DataFrame(out).to_csv(
        f"results/{FLAGS.jobname}/{t1_id}.tsv",
        sep="\t",
        index=False,
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
