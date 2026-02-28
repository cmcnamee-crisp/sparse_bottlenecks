"""Runs t3 (is_modular) and t4 (is_causal).
"""
import time


import pytorch_lightning as pl
import pandas as pd
import torch as th

import evaluate
import flags
import helpers
from interventions import inlp

from models import linear


def main(FLAGS):
    device = "cuda" if th.cuda.is_available() else "cpu"
    pl.seed_everything(FLAGS.seed)
    gpus = 1 if th.cuda.is_available() else 0

    # TODO: Intervene at various layers, measure impact on downstream.
    layer = flags.get_layers(FLAGS)[-1]
    out = []

    dataspec_downstreams = ["shape", "layout", "stroke", "classes"]
    dataspec_intervenes = ["shape", "layout", "stroke", "classes"]

    if "color" in FLAGS.data_path:
        # dataspec_downstreams += ["color"]
        dataspec_intervenes += ["color"]
    trainer = pl.Trainer(
        accelerator="gpu" if gpus > 0 else "cpu",
        devices=gpus if gpus > 0 else "auto",
        max_epochs=FLAGS.n_epochs,
        min_epochs=FLAGS.n_epochs,
        deterministic=True,  # reproducibility
    )
    for dataspec_downstream in dataspec_downstreams:
        FLAGS.dataspec = dataspec_downstream
        if helpers.causal_invalid_flags(FLAGS):
            continue
        train_holdout_many = FLAGS.holdout_many
        if dataspec_downstream == "classes":
            FLAGS.holdout_many = False

        datadesc, dataspec = helpers.load_info(FLAGS)
        train_id = flags.train_layerwise_id(FLAGS, layer)
        num_classes = len(datadesc.classlabel.unique())
        model = linear.Linear.load_from_checkpoint(
            checkpoint_path=f"checkpoints/{train_id}.ckpt"
        )
        FLAGS.holdout_many = train_holdout_many
        datamodule, _ = helpers.get_datamodule_layer(
            # The data is independent of the dataspec (downstream or intervene.)
            flags.transformed_id(FLAGS),
            # but the labels are dependent of the dataspec (downstream or intervene.)
            datadesc,
            dataspec,
            FLAGS.batch_size,
            device,
            layer=layer,
            flatten="avg",
        )
        for dataspec_intervene in dataspec_intervenes:
            FLAGS.dataspec = dataspec_intervene
            if helpers.invalid_flags(FLAGS):
                continue
            _, dataspec_intervene_spec = helpers.load_info(FLAGS)
            train_id = flags.train_layerwise_id(FLAGS, layer)
            W = linear.Linear.load_from_checkpoint(
                checkpoint_path=f"checkpoints/{train_id}.ckpt",
            ).linear.weight
            model.load_projection(inlp.get_projection(W))
            FLAGS.dataspec = dataspec_downstream
            out.extend(
                evaluate.get_predictions(
                    FLAGS,
                    dataspec,
                    datamodule,
                    trainer,
                    model,
                    num_classes,
                    dataspec_intervene_spec,
                    {
                        "layer": layer,
                        "dataspec_intervene": dataspec_intervene,
                        "dataspec_downstream": dataspec_downstream,
                        "num_classes": num_classes,
                    },
                )
            )
    output_id = flags.t_causal_id(FLAGS)
    pd.DataFrame(out).to_csv(
        f"results/{FLAGS.jobname}/{output_id}.tsv", sep="\t", index=False,
    )


if __name__ == "__main__":
    tick = time.time()
    FLAGS = flags.get_flags().parse_args()
    main(FLAGS)
    tock = time.time()
    id = flags.t3_id(FLAGS)
    pd.DataFrame([{"id": id, "script": "t3", "seconds": tock - tick}]).to_csv(
        f"times/{id}.tsv", index=False, sep="\t"
    )
